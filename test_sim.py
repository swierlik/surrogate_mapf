"""Minimal test: run one PIBT simulation on warehouse-33x36 with 400 agents.

Run inside Docker:
    docker run --rm -v %cd%:/workspace ggo_sim python3 /workspace/test_sim.py
"""
import json
import time
import numpy as np

import py_driver

# ---------- helpers (standalone, no env_search dependency) ----------

COMP_OBJ_TYPES = ".@ews"  # competition map object types
OBSTACLE_IDX = COMP_OBJ_TYPES.index("@")  # = 1


def layout_to_np(layout):
    """Convert list-of-strings map layout to integer numpy array."""
    rows = []
    for line in layout:
        rows.append([COMP_OBJ_TYPES.index(ch) for ch in line])
    return np.array(rows, dtype=int)


def count_valid_vertices(map_np):
    return int(np.sum(map_np != OBSTACLE_IDX))


def count_valid_edges(map_np):
    """Count bi-directed valid edges (both (u,v) and (v,u))."""
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


# ---------- load map ----------

MAP_PATH = "maps/competition/human/pibt_warehouse_33x36_w_mode.json"

with open(MAP_PATH, "r") as f:
    raw_map = json.load(f)

map_np = layout_to_np(raw_map["layout"])
n_v = count_valid_vertices(map_np)
n_e = count_valid_edges(map_np)
print(f"Map: {raw_map['name']}  ({raw_map['n_row']}x{raw_map['n_col']})")
print(f"Valid vertices: {n_v},  valid edges (bi-dir): {n_e}")

# ---------- load PIBT config ----------

CONFIG_PATH = "WPPL/configs/pibt_default_no_rot.json"
with open(CONFIG_PATH, "r") as f:
    config = json.load(f)
config_str = json.dumps(config)

# ---------- uniform (unweighted) edge weights and wait costs ----------

edge_weights = [1.0] * n_e
wait_costs = [1.0] * n_v

NUM_AGENTS = 400
SIM_STEPS = 1000

kwargs = {
    "scenario": "COMPETITION",
    "map_json_path": MAP_PATH,
    "simulation_steps": SIM_STEPS,
    "gen_random": True,
    "num_tasks": 100000,
    "num_agents": NUM_AGENTS,
    "weights": json.dumps(edge_weights),
    "wait_costs": json.dumps(wait_costs),
    "plan_time_limit": 1,
    "seed": 42,
    "preprocess_time_limit": 1800,
    "file_storage_path": "large_files",
    "task_assignment_strategy": "roundrobin",
    "num_tasks_reveal": 1,
    "config": config_str,
}

print(f"\nRunning PIBT simulation ({SIM_STEPS} steps, {NUM_AGENTS} agents)...")
t0 = time.time()
result_str = py_driver.run(**kwargs)
elapsed = time.time() - t0

result = json.loads(result_str)

throughput = result["throughput"]
print(f"\n=== Results ===")
print(f"Throughput: {throughput:.4f}")
print(f"Wall time:  {elapsed:.1f}s")
print(f"\n(Baseline unweighted ~5-6; CMA-ES optimized target ~7.64)")
