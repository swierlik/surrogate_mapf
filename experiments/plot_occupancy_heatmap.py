"""
2-panel agent occupancy heatmap: Vanilla CMA-ES vs Surrogate V3.
Runs one simulation per solution in Docker and plots tile_usage side by side.

Usage:
    python experiments/plot_occupancy_heatmap.py
"""

import sys, os, json, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.simulator.evaluate import (
    normalize_solution, MapInfo, MAP_REL_PATH,
    DOCKER_IMAGE, PROJECT_ROOT,
)

OUT_DIR      = "Thesis_Final"
NUM_AGENTS   = 400
SIM_STEPS    = 1000
SEED         = 42

import pandas as pd

def get_gen0_best():
    """Return the highest-throughput solution from generation 0."""
    log  = pd.read_csv("results/baseline/cmaes_log.csv")
    sols = np.load("results/baseline/cmaes_solutions.npy")
    gen0 = log[log["generation"] == 0]
    best_idx = gen0["mean_throughput"].idxmax()
    return sols[best_idx]

# PANELS: (label, solution_array_or_path)
# resolved at runtime so gen-0 array is extracted from the big solutions file
PANEL_DEFS = [
    ("Generation 0 (random init)", "gen0"),
    ("Optimized solution",          "results/surrogate_v3/best_solution.npy"),
]


def run_sim_tile_usage(solution_or_path):
    info = MapInfo.get()
    if isinstance(solution_or_path, str):
        sol = normalize_solution(np.load(solution_or_path))
    else:
        sol = normalize_solution(solution_or_path)
    n_v  = info.n_valid_vertices

    wait_costs   = sol[:n_v].tolist()
    edge_weights = sol[n_v:].tolist()

    script = (
        "import json, sys\n"
        "import py_driver\n"
        f"MAP_PATH    = '{MAP_REL_PATH}'\n"
        "CONFIG_PATH = 'WPPL/configs/pibt_default_no_rot.json'\n"
        "with open(CONFIG_PATH) as f:\n"
        "    config_str = json.dumps(json.load(f))\n"
        f"WAIT   = {json.dumps(wait_costs)}\n"
        f"EDGES  = {json.dumps(edge_weights)}\n"
        f"result = json.loads(py_driver.run(\n"
        f"    scenario='COMPETITION',\n"
        f"    map_json_path=MAP_PATH,\n"
        f"    simulation_steps={SIM_STEPS},\n"
        f"    gen_random=True,\n"
        f"    num_tasks=100000,\n"
        f"    num_agents={NUM_AGENTS},\n"
        f"    weights=json.dumps(EDGES),\n"
        f"    wait_costs=json.dumps(WAIT),\n"
        f"    plan_time_limit=1,\n"
        f"    seed={SEED},\n"
        f"    preprocess_time_limit=1800,\n"
        f"    file_storage_path='large_files',\n"
        f"    task_assignment_strategy='roundrobin',\n"
        f"    num_tasks_reveal=1,\n"
        f"    config=config_str,\n"
        f"))\n"
        'print(json.dumps({"tile_usage": result["tile_usage"], '
        '"throughput": result["throughput"]}))\n'
    )

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(PROJECT_ROOT), delete=False
    ) as f:
        f.write(script)
        script_path = f.name

    try:
        print(f"  Running Docker simulation ...")
        proc = subprocess.run(
            ["docker", "run", "--rm",
             "-v", f"{PROJECT_ROOT}:/workspace",
             "-w", "/usr/project",
             DOCKER_IMAGE,
             "python3", f"/workspace/{os.path.basename(script_path)}"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Docker failed:\n{proc.stderr}")
        data       = json.loads(proc.stdout.strip().split("\n")[-1])
        tile_usage = np.array(data["tile_usage"]).reshape(info.n_row, info.n_col)
        throughput = data["throughput"]
        print(f"    throughput = {throughput:.4f}")
        return tile_usage, throughput
    finally:
        os.unlink(script_path)


if __name__ == "__main__":
    info = MapInfo.get()
    obstacle_mask = (info.map_np == 1)   # True where wall

    results = []
    for label, src in PANEL_DEFS:
        sol = get_gen0_best() if src == "gen0" else src
        tile_usage, tp = run_sim_tile_usage(sol)
        tile_usage = tile_usage.astype(float)
        tile_usage[obstacle_mask] = np.nan
        results.append((label, tile_usage, tp))

    # shared colour scale: 99th percentile of valid cells across both panels
    all_valid = np.concatenate([r[1][~np.isnan(r[1])] for r in results])
    vmax = np.percentile(all_valid, 99)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5),
                             gridspec_kw={"wspace": 0.06})

    for ax, (label, tile_usage, tp) in zip(axes, results):
        im = ax.imshow(tile_usage, cmap=cmap, vmin=0, vmax=vmax,
                       aspect="auto", interpolation="nearest")
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_xlabel(f"throughput = {tp:.3f}", fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.025, pad=0.03)
    cbar.set_label("Agent visits per cell", fontsize=9)
    cbar.ax.tick_params(labelsize=9)

    fig.suptitle("Agent occupancy: Warehouse-33-36", fontsize=11, y=1.01)
    plt.tight_layout()

    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_occupancy_heatmap.{ext}"),
                    dpi=300, bbox_inches="tight")
    print(f"Saved to {OUT_DIR}/fig_occupancy_heatmap.{{pdf,png}}")
    plt.show()
