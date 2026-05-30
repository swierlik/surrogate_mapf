"""Online LMAPF evaluation via WPPL's py_driver in Docker (wppl_sim image).

Each candidate is a flat parameter vector theta for a CNN guidance policy.
The policy is called every `update_interval` timesteps to regenerate edge weights
from observed traffic patterns (online GGO, Zang et al. AAAI 2025).

Architecture: 3-layer CNN (nc=10 → 32 → 32 → 5), 4,271 trainable parameters.
  Input  (10, H, W): edge_usage[4] + wait_usage[1] + edge_weights[4] + wait_costs[1]
  Output  (5, H, W): new_edge_weights[4] + new_wait_costs[1]

Interface mirrors evaluate.py:
    evaluate_online_batch(thetas, ...) -> (mean_throughputs, all_throughputs)
"""

import json
import os
import subprocess
import tempfile
import textwrap
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
ONLINE_GGO_ROOT = PROJECT_ROOT.parent / "OnlineGGO"

MAP_REL_PATH   = "CMAES/maps/competition/human/pibt_warehouse_33x36_w_mode.json"
CONFIG_REL_PATH = "CMAES/WPPL/configs/pibt_default_no_rot.json"

DOCKER_IMAGE   = "wppl_sim"
N_VALID_VERTICES = 948
N_VALID_EDGES    = 3126
H, W             = 33, 36
N_PARAMS         = 4271   # nc=10, n_hid_chan=32, kernel_size=3


def get_n_params():
    return N_PARAMS


def evaluate_online_batch(
    thetas,
    num_agents=400,
    simulation_steps=1000,
    update_interval=20,
    n_evals=2,
    base_seed=0,
    n_workers=4,
    chunk_size=20,
):
    """Evaluate a batch of policy parameter vectors via online LMAPF simulation.

    Large batches are split into chunks of `chunk_size` candidates, each run in
    a separate Docker call, to keep peak memory bounded.

    Args:
        thetas: np.ndarray of shape (batch_size, N_PARAMS).
        num_agents: number of agents.
        simulation_steps: total timesteps per simulation (default 1000).
        update_interval: timesteps between policy calls (default 20 → 50 calls/sim).
        n_evals: stochastic evaluation repeats per solution.
        base_seed: starting random seed.
        n_workers: parallel chains inside each Docker call.
        chunk_size: candidates per Docker call (limits peak memory).

    Returns:
        mean_throughputs: np.ndarray (batch_size,)
        all_throughputs:  np.ndarray (batch_size, n_evals)
    """
    batch_size = len(thetas)
    all_results = [None] * batch_size

    # Process in chunks to bound peak memory inside Docker
    for chunk_start in range(0, batch_size, chunk_size):
        chunk_end = min(chunk_start + chunk_size, batch_size)
        chunk_thetas = thetas[chunk_start:chunk_end]
        chunk_seed = base_seed + chunk_start * n_evals

        chunk_mean, chunk_all = _run_docker_batch(
            chunk_thetas,
            num_agents=num_agents,
            simulation_steps=simulation_steps,
            update_interval=update_interval,
            n_evals=n_evals,
            base_seed=chunk_seed,
            n_workers=n_workers,
        )
        for i, (m, a) in enumerate(zip(chunk_mean, chunk_all)):
            all_results[chunk_start + i] = (m, a)

    mean_throughputs = np.array([r[0] for r in all_results])
    all_throughputs  = np.array([r[1] for r in all_results])
    return mean_throughputs, all_throughputs


def _run_docker_batch(
    thetas,
    num_agents,
    simulation_steps,
    update_interval,
    n_evals,
    base_seed,
    n_workers,
):
    """Run a single Docker call for a sub-batch of thetas."""
    # Use Docker-internal paths (ONLINE_GGO_ROOT is mounted at /onlineggo)
    map_path    = f"/onlineggo/{MAP_REL_PATH}"
    config_path = f"/onlineggo/{CONFIG_REL_PATH}"

    batch_data = [theta.tolist() for theta in thetas]

    script = _build_online_eval_script(
        batch_data=batch_data,
        map_path=map_path,
        config_path=config_path,
        num_agents=num_agents,
        simulation_steps=simulation_steps,
        update_interval=update_interval,
        n_evals=n_evals,
        base_seed=base_seed,
        n_workers=n_workers,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(PROJECT_ROOT), delete=False, encoding="utf-8"
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{PROJECT_ROOT}:/workspace",
                "-v", f"{str(ONLINE_GGO_ROOT)}:/onlineggo",
                "-w", "/MAPF/codes",
                DOCKER_IMAGE,
                "python3", f"/workspace/{os.path.basename(script_path)}",
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Online Docker eval failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        output_lines = result.stdout.strip().split("\n")
        result_data  = json.loads(output_lines[-1])
        all_tp  = np.array(result_data["all_throughputs"])  # (chunk, n_evals)
        mean_tp = all_tp.mean(axis=1)
        return mean_tp, all_tp

    finally:
        os.unlink(script_path)


def _build_online_eval_script(
    batch_data, map_path, config_path,
    num_agents, simulation_steps, update_interval,
    n_evals, base_seed, n_workers,
):
    """Generate a self-contained Python script that runs inside the wppl_sim container."""

    return textwrap.dedent(f"""\
import sys, json, numpy as np, torch, torch.nn as nn
from multiprocessing import Pool

sys.path.insert(0, '/MAPF/codes/build')
import py_driver

# ── constants ──────────────────────────────────────────────────────────────
MAP_PATH    = {json.dumps(map_path)}
CONFIG_PATH = {json.dumps(config_path)}
H, W        = {H}, {W}
N_V         = {N_VALID_VERTICES}   # valid vertices (wait costs)
N_E         = {N_VALID_EDGES}      # valid edges    (edge weights)
NUM_AGENTS  = {num_agents}
SIM_STEPS   = {simulation_steps}
UPDATE_INT  = {update_interval}
N_SEGS      = SIM_STEPS // UPDATE_INT
N_EVALS     = {n_evals}
BASE_SEED   = {base_seed}
N_WORKERS   = {n_workers}

batch_data  = {json.dumps(batch_data)}

# ── map helpers ─────────────────────────────────────────────────────────────
def _build_map_mask():
    with open(MAP_PATH) as f:
        layout = json.load(f)["layout"]
    OBJ = ".@ews"
    WALL = OBJ.index("@")
    grid = [[OBJ.index(c) for c in row] for row in layout]
    return np.array(grid) == WALL   # True = obstacle

WALL_MASK = _build_map_mask()  # (H, W) bool

def uncompress_vertices(compressed):
    "Expand (N_V,) → (H, W)."
    out = np.zeros(H * W, dtype=np.float32)
    j = 0
    for i in range(H * W):
        y, x = divmod(i, W)
        if not WALL_MASK[y, x]:
            out[i] = compressed[j]
            j += 1
    return out.reshape(H, W)

def uncompress_edges(compressed):
    "Expand (N_E,) → (H, W, 4)  [right, up, left, down]."
    out = np.zeros((H * W * 4,), dtype=np.float32)
    j = 0
    for i in range(H * W):
        y, x = divmod(i, W)
        if WALL_MASK[y, x]:
            continue
        for dy, dx, d in [( 0, 1, 0), (-1, 0, 1), ( 0,-1, 2), ( 1, 0, 3)]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < H and 0 <= nx < W and not WALL_MASK[ny, nx]:
                out[i * 4 + d] = compressed[j]
                j += 1
    return out.reshape(H, W, 4)

def compress_edges(full_hww4):
    "Compress (H, W, 4) → (N_E,)."
    out = []
    for y in range(H):
        for x in range(W):
            if WALL_MASK[y, x]:
                continue
            for dy, dx, d in [( 0, 1, 0), (-1, 0, 1), ( 0,-1, 2), ( 1, 0, 3)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and not WALL_MASK[ny, nx]:
                    out.append(float(full_hww4[y, x, d]))
    return out

def compress_vertices(full_hw):
    "Compress (H, W) → (N_V,)."
    return [float(full_hw[y, x])
            for y in range(H) for x in range(W) if not WALL_MASK[y, x]]

def normalize(arr, lo=0.1, hi=100.0):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-6:
        return np.clip(arr, lo, hi)
    return lo + (arr - mn) * (hi - lo) / (mx - mn)

# ── CNN policy model ─────────────────────────────────────────────────────────
class OnlineCNNPolicy(nn.Module):
    "3-layer CNN: nc=10 → 32 → 32 → 5  (4271 trainable params)."
    def __init__(self):
        super().__init__()
        nc, nh, k, p = 10, 32, 3, 1
        self.net = nn.Sequential(
            nn.Conv2d(nc, nh, k, 1, p), nn.ReLU(inplace=True), nn.BatchNorm2d(nh),
            nn.Conv2d(nh, nh,  1, 1, 0), nn.ReLU(inplace=True), nn.BatchNorm2d(nh),
            nn.Conv2d(nh,  5,  1, 1, 0), nn.ReLU(inplace=True), nn.BatchNorm2d(5),
        )

    def forward(self, x):
        return self.net(x)   # (1, 5, H, W)

def policy_from_theta(theta_list):
    model = OnlineCNNPolicy()
    params = torch.tensor(theta_list, dtype=torch.float32)
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(params[offset:offset + n].view_as(p))
            offset += n
    model.eval()
    return model

# ── single-chain evaluation ──────────────────────────────────────────────────
def run_chain(task):
    sol_idx, eval_idx, theta_list, seed = task

    with open(CONFIG_PATH) as f:
        config_str = json.dumps(json.load(f))

    policy = policy_from_theta(theta_list)

    # Initial weights: uniform
    edge_weights = [1.0] * N_E
    wait_costs   = [1.0] * N_V

    last_pos   = None
    last_tasks = None
    total_finished = 0

    for seg in range(N_SEGS):
        kwargs = dict(
            map_json_path=MAP_PATH,
            simulation_steps=UPDATE_INT,
            gen_random=True,
            num_tasks=100000,
            num_agents=NUM_AGENTS,
            weights=json.dumps(edge_weights),
            wait_costs=json.dumps(wait_costs),
            plan_time_limit=1,
            seed=seed + seg,
            preprocess_time_limit=60,
            file_storage_path=f'large_files/chain_{{sol_idx}}_{{eval_idx}}/',
            task_assignment_strategy='online_generate',
            num_tasks_reveal=1,
            left_w_weight=1.0,
            right_w_weight=1.0,
            config=config_str,
        )
        if last_pos is not None:
            kwargs['init_agent']     = True
            kwargs['init_agent_pos'] = str(last_pos)
        if last_tasks is not None:
            kwargs['init_task']     = True
            kwargs['init_task_ids'] = str(last_tasks)

        result = json.loads(py_driver.run(**kwargs))
        last_pos   = result['final_pos']
        last_tasks = result['final_tasks']
        total_finished += result['num_task_finished']

        # ── build 10-channel observation ──────────────────────────────────
        # Channels: edge_usage[4] wait_usage[1] edge_weights[4] wait_costs[1]
        edge_usage_flat  = np.array(result['edge_usage_matrix'], dtype=np.float32)
        wait_usage_flat  = np.array(result['vertex_wait_matrix'], dtype=np.float32)

        edge_usage_hw4 = edge_usage_flat.reshape(H, W, 4)   # (H,W,4)
        wait_usage_hw  = wait_usage_flat.reshape(H, W)       # (H,W)

        ew_hw4 = uncompress_edges(np.array(edge_weights, dtype=np.float32))
        wc_hw  = uncompress_vertices(np.array(wait_costs, dtype=np.float32))

        # Normalize each to [0,1] for stable CNN input
        def _norm01(a):
            mn, mx = a.min(), a.max()
            return (a - mn) / (mx - mn + 1e-8)

        obs = np.stack([
            _norm01(edge_usage_hw4[:,:,0]),   # right usage
            _norm01(edge_usage_hw4[:,:,1]),   # up usage
            _norm01(edge_usage_hw4[:,:,2]),   # left usage
            _norm01(edge_usage_hw4[:,:,3]),   # down usage
            _norm01(wait_usage_hw),           # wait usage
            _norm01(ew_hw4[:,:,0]),           # right weight
            _norm01(ew_hw4[:,:,1]),           # up weight
            _norm01(ew_hw4[:,:,2]),           # left weight
            _norm01(ew_hw4[:,:,3]),           # down weight
            _norm01(wc_hw),                   # wait cost
        ], axis=0).astype(np.float32)         # (10, H, W)

        obs_t = torch.from_numpy(obs).unsqueeze(0)  # (1, 10, H, W)
        with torch.no_grad():
            out = policy(obs_t).squeeze(0).numpy()  # (5, H, W)

        # out[0:4] → edge weights, out[4] → wait costs (post-ReLU, ≥0)
        new_ew_hw4 = np.stack([out[0], out[1], out[2], out[3]], axis=2)  # (H,W,4)
        new_wc_hw  = out[4]                                                # (H,W)

        # Normalize and compress back to valid-edge format
        new_ew_hw4 = normalize(new_ew_hw4 + 0.1)
        new_wc_hw  = normalize(new_wc_hw  + 0.1)
        edge_weights = compress_edges(new_ew_hw4)
        wait_costs   = compress_vertices(new_wc_hw)

    throughput = total_finished / SIM_STEPS
    return (sol_idx, eval_idx, throughput)


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    tasks = []
    for sol_idx, theta_list in enumerate(batch_data):
        for eval_idx in range(N_EVALS):
            seed = BASE_SEED + sol_idx * N_EVALS + eval_idx
            tasks.append((sol_idx, eval_idx, theta_list, seed))

    print(f"Running {{len(tasks)}} online chains with {{N_WORKERS}} workers...", file=sys.stderr)

    if N_WORKERS > 1:
        with Pool(processes=N_WORKERS) as pool:
            results = pool.map(run_chain, tasks)
    else:
        results = [run_chain(t) for t in tasks]

    n_solutions = len(batch_data)
    all_throughputs = [[0.0] * N_EVALS for _ in range(n_solutions)]
    for sol_idx, eval_idx, tp in results:
        all_throughputs[sol_idx][eval_idx] = tp
        mean_tp = sum(all_throughputs[sol_idx]) / N_EVALS
        print(f"Solution {{sol_idx+1}}/{{n_solutions}} eval {{eval_idx}}: tp={{tp:.4f}}", file=sys.stderr)

    print(json.dumps({{"all_throughputs": all_throughputs}}))
""")
