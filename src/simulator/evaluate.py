"""Wrapper around the C++ LMAPF simulator (py_driver) running in Docker.

Provides:
    evaluate_batch(solutions, ...) -> (mean_throughputs, all_throughputs)

The solution vector has shape (n_wait + n_edge,) = (948 + 3126,) = (4074,)
for the warehouse-33x36 map.  Layout: [wait_costs..., edge_weights...].

Internally, we call `py_driver.run()` inside the Docker container `ggo_sim`.
"""

import json
import subprocess
import tempfile
import os
import numpy as np
from pathlib import Path

# Project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
GGO_ROOT = PROJECT_ROOT / "ggo_public"

# Map metadata
MAP_REL_PATH = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
CONFIG_REL_PATH = "WPPL/configs/pibt_default_no_rot.json"
COMP_OBJ_TYPES = ".@ews"
OBSTACLE_IDX = COMP_OBJ_TYPES.index("@")

# Docker image name
DOCKER_IMAGE = "ggo_sim"


def _layout_to_np(layout):
    """Convert list-of-strings map layout to integer numpy array."""
    return np.array(
        [[COMP_OBJ_TYPES.index(ch) for ch in line] for line in layout],
        dtype=int,
    )


def _count_valid_vertices(map_np):
    return int(np.sum(map_np != OBSTACLE_IDX))


def _count_valid_edges(map_np):
    """Count bi-directed valid edges."""
    h, w = map_np.shape
    n = 0
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        for x in range(h):
            for y in range(w):
                if map_np[x, y] == OBSTACLE_IDX:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and map_np[nx, ny] != OBSTACLE_IDX:
                    n += 1
    return n


class MapInfo:
    """Lazily loaded map metadata (cached after first load)."""

    _instance = None

    def __init__(self):
        map_path = GGO_ROOT / MAP_REL_PATH
        with open(map_path, "r") as f:
            raw = json.load(f)
        self.name = raw["name"]
        self.n_row = raw["n_row"]
        self.n_col = raw["n_col"]
        self.layout = raw["layout"]
        self.map_np = _layout_to_np(self.layout)
        self.n_valid_vertices = _count_valid_vertices(self.map_np)
        self.n_valid_edges = _count_valid_edges(self.map_np)
        self.sol_size = self.n_valid_vertices + self.n_valid_edges

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


def get_sol_size():
    """Return the dimensionality of the solution vector (4074 for warehouse-33x36)."""
    return MapInfo.get().sol_size


def normalize_solution(solution, lb=0.1, ub=100.0):
    """Min-max normalize a single solution vector, separately for wait and edge parts.

    Matches ggo_public's bound_handle="normalization":
        - wait_costs[:n_v] normalized independently
        - edge_weights[n_v:] normalized independently
    Both mapped to [lb, ub].
    """
    info = MapInfo.get()
    n_v = info.n_valid_vertices
    sol = solution.copy()

    for start, end in [(0, n_v), (n_v, len(sol))]:
        part = sol[start:end]
        mn, mx = part.min(), part.max()
        if mx - mn < 1e-3:
            sol[start:end] = np.clip(part, lb, ub)
        else:
            sol[start:end] = lb + (part - mn) * (ub - lb) / (mx - mn)

    return sol


def evaluate_batch(
    solutions,
    num_agents=400,
    simulation_steps=1000,
    n_evals=5,
    base_seed=42,
    normalize=True,
    n_workers=8,
):
    """Evaluate a batch of solutions via Docker with parallel workers.

    Args:
        solutions: np.ndarray of shape (batch_size, sol_size).
        num_agents: Number of agents.
        simulation_steps: Timesteps per simulation.
        n_evals: Number of stochastic simulation runs per solution.
        base_seed: Starting random seed.
        normalize: Whether to apply min-max normalization.
        n_workers: Number of parallel worker processes inside Docker.

    Returns:
        mean_throughputs: np.ndarray of shape (batch_size,).
        all_throughputs: np.ndarray of shape (batch_size, n_evals).
    """
    info = MapInfo.get()
    n_v = info.n_valid_vertices

    if normalize:
        sols = np.array([normalize_solution(s) for s in solutions])
    else:
        sols = solutions.copy()

    batch_data = []
    for sol in sols:
        batch_data.append({
            "wait_costs": sol[:n_v].tolist(),
            "edge_weights": sol[n_v:].tolist(),
        })

    script = _build_batch_eval_script(
        batch_data=batch_data,
        num_agents=num_agents,
        simulation_steps=simulation_steps,
        n_evals=n_evals,
        base_seed=base_seed,
        n_workers=n_workers,
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(PROJECT_ROOT), delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "-v", f"{PROJECT_ROOT}:/workspace",
                "-w", "/usr/project",
                DOCKER_IMAGE,
                "python3", f"/workspace/{os.path.basename(script_path)}",
            ],
            capture_output=True,
            text=True,
            timeout=7200,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Docker simulation failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

        output_lines = result.stdout.strip().split("\n")
        result_data = json.loads(output_lines[-1])
        all_tp = np.array(result_data["all_throughputs"])  # (batch, n_evals)
        mean_tp = all_tp.mean(axis=1)                      # (batch,)
        return mean_tp, all_tp

    finally:
        os.unlink(script_path)


def _build_batch_eval_script(batch_data, num_agents, simulation_steps,
                             n_evals, base_seed, n_workers=8):
    """Generate a Python script that evaluates a batch in parallel."""
    return f'''import json, sys
from multiprocessing import Pool

MAP_PATH = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"
CONFIG_PATH = "WPPL/configs/pibt_default_no_rot.json"

batch_data = {json.dumps(batch_data)}
N_EVALS = {n_evals}
BASE_SEED = {base_seed}
NUM_AGENTS = {num_agents}
SIM_STEPS = {simulation_steps}
N_WORKERS = {n_workers}


def run_single_sim(task):
    """Run one (solution_idx, eval_idx) simulation. Each worker imports py_driver."""
    import py_driver
    sol_idx, eval_idx, wait_costs, edge_weights, config_str = task

    kwargs = {{
        "scenario": "COMPETITION",
        "map_json_path": MAP_PATH,
        "simulation_steps": SIM_STEPS,
        "gen_random": True,
        "num_tasks": 100000,
        "num_agents": NUM_AGENTS,
        "weights": json.dumps(edge_weights),
        "wait_costs": json.dumps(wait_costs),
        "plan_time_limit": 1,
        "seed": BASE_SEED + eval_idx,
        "preprocess_time_limit": 1800,
        "file_storage_path": "large_files",
        "task_assignment_strategy": "roundrobin",
        "num_tasks_reveal": 1,
        "config": config_str,
    }}
    result = json.loads(py_driver.run(**kwargs))
    return (sol_idx, eval_idx, result["throughput"])


if __name__ == "__main__":
    with open(CONFIG_PATH) as f:
        config_str = json.dumps(json.load(f))

    # Build flat list of all (sol_idx, eval_idx) tasks
    tasks = []
    for sol_idx, sol_data in enumerate(batch_data):
        for eval_idx in range(N_EVALS):
            tasks.append((
                sol_idx, eval_idx,
                sol_data["wait_costs"], sol_data["edge_weights"],
                config_str,
            ))

    print(f"Running {{len(tasks)}} simulations with {{N_WORKERS}} workers...", file=sys.stderr)

    with Pool(processes=N_WORKERS) as pool:
        results = pool.map(run_single_sim, tasks)

    # Reassemble into (batch_size, n_evals) structure
    n_solutions = len(batch_data)
    all_throughputs = [[0.0] * N_EVALS for _ in range(n_solutions)]
    for sol_idx, eval_idx, tp in results:
        all_throughputs[sol_idx][eval_idx] = tp

    for idx in range(n_solutions):
        mean_tp = sum(all_throughputs[idx]) / N_EVALS
        print(f"Solution {{idx+1}}/{{n_solutions}}: throughput={{mean_tp:.4f}}", file=sys.stderr)

    print(json.dumps({{"all_throughputs": all_throughputs}}))
'''
