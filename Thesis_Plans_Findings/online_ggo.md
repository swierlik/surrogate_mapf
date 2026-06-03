# Extending Surrogate-Assisted CMA-ES to Online Guidance Graph Optimization

## Agent Task Summary

We have a working surrogate-assisted CMA-ES framework for **offline** Guidance Graph Optimization (GGO) in Lifelong Multi-Agent Path Finding (LMAPF). The task is to extend this framework to the **online** GGO case from Zang et al. (AAAI 2025). The core surrogate infrastructure (pre-screening, evolution control, ensemble uncertainty, retraining) stays identical. Only the evaluation function and surrogate input representation change.

---

## Context: What Already Exists

### Our Current Offline Pipeline

We have a working implementation based on Zhang et al. (IJCAI 2024):

- **Repository used:** https://github.com/lunjohnzhang/ggo_public
- **What CMA-ES optimizes:** Raw edge weight vector ω ∈ ℝ^4,074 (for warehouse-33-36 map)
- **Evaluation function:** `evaluate_offline(omega) → throughput`
  - Takes ω, forms a static guidance graph, runs 5 LMAPF simulations of 1,000 timesteps, returns average throughput
- **Surrogate input:** ω reshaped as (33, 36, 5) tensor (4 movement directions + 1 wait cost per cell)
- **Surrogate models implemented:** XGBoost (baseline), lightweight CNN, hybrid (CNN features + XGBoost), ensemble of 5 CNNs for uncertainty-aware pre-screening
- **Surrogate-assisted loop:** warmup → pre-screen top 20% → evolution control every 3rd gen → retrain on accumulated data

### What We're Adding

Online GGO from Zang et al. (AAAI 2025):

- **Repository:** https://github.com/zanghz21/OnlineGGO
- **Paper:** https://arxiv.org/abs/2411.16506
- **What CMA-ES optimizes:** Parameters θ of a small policy network that dynamically generates guidance graphs from real-time traffic
- **Key difference:** The guidance graph is NOT static. During each 1,000-timestep simulation, the policy is called every m=20 timesteps to regenerate the guidance graph based on observed traffic patterns. This means ~50 policy invocations per evaluation.

---

## Paper Summary: Online GGO (Zang et al., AAAI 2025)

### Core Idea

Instead of optimizing a fixed set of edge weights (offline), optimize a **guidance policy** π_θ that takes real-time traffic observations as input and outputs updated edge weights. The policy adapts the guidance graph on-the-fly during simulation, responding to changing congestion patterns.

### Formal Definition

**Guidance Policy:** A function π_θ : O → ℝ^{|E_g|} that computes updated edge weights ω' given observation o collected from the LMAPF simulation. The policy is parameterized by θ ∈ Θ.

### Two Pipelines (Section 3.1 of the paper)

**Pipeline 1: Direct Planning (on+PIBT)**

- Run PIBT for m=20 timesteps
- Collect edge usage (how often each edge was used) and current agent goals
- Feed edge usage + goals into policy π_θ to generate new guidance graph weights
- Recompute heuristic tables (this is why it's 4× slower)
- Repeat for ⌊N/m⌋ = 50 iterations per 1,000-timestep simulation
- **Policy network:** CNN with **3,119 parameters**

**Pipeline 2: Guide-Path Planning (on+GPIBT)**

- Each agent plans a guide path when assigned a new goal
- The policy generates guidance graph weights based on guide-path edge usage
- Guidance graph updates every time an agent replans
- Agents follow guide paths, resolving collisions with PIBT rules
- **Policy network:** Windowed quadratic network with **560 parameters**

### Policy Network Architectures (Appendices A.2 and A.3)

**For PIBT (CNN, 3,119 params):**

- Input: traffic observation tensor (edge usage + current goals)
- 3 convolutional layers, operates on the h×w grid
- Output: updated edge weights as h×w×5 tensor
- Note: this is structurally similar to (but NOT the same as) the PIU update model from Zhang 2024 which has 4,231 params

**For GPIBT (Windowed Quadratic, 560 params):**

- Smaller network, specialized for guide-path updates
- Lower dimensionality makes it even easier for our surrogate

### CMA-ES Setup (Section 3.2)

The optimization is **identical in structure** to the offline case:

1. Sample batch of b parameter vectors θ_1, ..., θ_b from Gaussian
2. Evaluate each by running LMAPF simulation with policy π\_{θ_k}
3. Compute average throughput over N_e simulation runs
4. Rank, select top candidates, update Gaussian
5. Repeat for I iterations

The only difference is step 2: instead of forming a static guidance graph from ω, we instantiate a policy network from θ and run an online simulation where the policy is called repeatedly.

### Key Experimental Parameters (from paper)

| Parameter                  | Value                                                       |
| -------------------------- | ----------------------------------------------------------- |
| Maps                       | sortation-33-57, warehouse-33-57, empty-32-32, random-32-32 |
| Simulation length N        | 1,000 timesteps                                             |
| Guidance update interval m | 20 timesteps (for on+PIBT)                                  |
| PIBT policy params         | 3,119 (CNN)                                                 |
| GPIBT policy params        | 560 (quadratic network)                                     |

### Results Summary

- on+PIBT outperforms off+PIBT by up to 30.75% throughput improvement
- on+GPIBT outperforms hm+GPIBT (human-designed) by up to 31.59%
- on+PIBT is ~4× slower per timestep than off+PIBT (heuristic recomputation)
- on+GPIBT is ~7× slower than off+GPIBT (guidance graph updates per agent)
- All runtimes still < 0.028 seconds per timestep (acceptable for real-world use)

### Why Surrogate Assistance Matters Even More Here

The paper's conclusion explicitly states: _"as our approach uses CMA-ES, which requires many evaluations in LMAPF simulators, future work could explore surrogate-assisted optimization to improve sample efficiency."_

Each online evaluation is 4× more expensive than offline. So a 2× reduction in evaluation count from our surrogate → ~8× wall-clock speedup. This is the primary motivation for this extension.

---

## Implementation Plan

### Step 1: Set Up OnlineGGO Repository

```bash
git clone https://github.com/zanghz21/OnlineGGO.git
cd OnlineGGO
```

Follow their build instructions. The repo contains:

- C++ LMAPF simulator with online guidance support (PIBT + GPIBT)
- Python CMA-ES optimization wrapper
- Policy network definitions (CNN for PIBT, quadratic for GPIBT)
- Map files and configuration

**Verify it works:** Run their provided example to confirm you can:

1. Instantiate a policy network from a parameter vector θ
2. Run an online LMAPF simulation that calls the policy every 20 timesteps
3. Get a throughput value back

### Step 2: Write the Online Evaluation Wrapper

Create a function that matches the interface of our existing offline wrapper:

```python
def evaluate_online(theta: np.ndarray, config: dict) -> float:
    """
    Evaluate a policy parameter vector by running online LMAPF simulation.

    Args:
        theta: Parameter vector for the policy network.
               Shape: (3119,) for PIBT CNN or (560,) for GPIBT quadratic.
        config: Dictionary containing:
            - map_path: path to the map file
            - num_agents: number of agents (e.g., 400 for warehouse)
            - num_simulations: N_e, number of sims to average over
            - simulation_length: N, timesteps per simulation (1000)
            - update_interval: m, how often to call the policy (20)
            - algorithm: 'PIBT' or 'GPIBT'

    Returns:
        Average throughput over num_simulations runs.
    """
    # 1. Construct policy network from theta
    # 2. Run N_e online LMAPF simulations with this policy
    # 3. Return average throughput
    pass
```

This wrapper needs to interface with the OnlineGGO C++ simulator. Look at how their existing CMA-ES optimization script calls the simulator — it likely has a Python-C++ bridge (pybind11 or subprocess calls). Mirror that interface.

**Key files to inspect in OnlineGGO repo:**

- The main optimization script (likely in a `scripts/` or `python/` directory)
- The policy network definition (PyTorch module)
- The simulator interface (how Python sends θ to C++ and gets throughput back)

### Step 3: Adapt Surrogate Input Representation

This is the critical design decision. We have three options, ordered by implementation ease:

#### Option A: XGBoost on Raw θ (Easiest, do this first)

```python
# Surrogate input: raw parameter vector
# For PIBT: theta.shape = (3119,)
# For GPIBT: theta.shape = (560,)

# Train XGBoost exactly as before, just with different input dim
surrogate = XGBRegressor()
surrogate.fit(theta_dataset, throughput_dataset)
predicted = surrogate.predict(theta_candidates)
```

This is the fastest to implement. The dimensionality (3,119) is actually LOWER than our offline case (4,074), so XGBoost should perform at least as well. Start here for a quick feasibility check.

#### Option B: MLP on Raw θ (Simple neural alternative)

```python
class MLPSurrogate(nn.Module):
    def __init__(self, input_dim=3119):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x)
```

Replace the CNN surrogate since θ is not spatially structured — it's neural network weights, not a grid tensor. An MLP is the natural architecture for unstructured vectors.

#### Option C: Functional Output Approach (Most principled, do if time allows)

Instead of feeding raw θ to the surrogate, convert θ into something spatially meaningful:

```python
def theta_to_functional_features(theta, canonical_observations):
    """
    Convert policy parameters to functional features by running the policy
    on a set of canonical traffic observations.

    Args:
        theta: policy parameter vector (3119,)
        canonical_observations: list of 3-5 representative traffic
            observation tensors collected from early simulations

    Returns:
        Stacked guidance graph tensors, shape (num_obs, 33, 36, 5)
        or flattened to a feature vector
    """
    policy = construct_policy_network(theta)
    features = []
    for obs in canonical_observations:
        guidance_weights = policy(obs)  # shape (33, 36, 5)
        features.append(guidance_weights)
    return np.stack(features)  # shape (num_obs, 33, 36, 5)
```

This converts θ into guidance graph tensors, which lets us reuse our existing CNN surrogate. The canonical observations should be collected during the warmup phase — pick 3-5 diverse traffic snapshots from different timesteps/simulations.

**Recommended approach:** Implement Option A first (XGBoost, 1 hour), then Option B (MLP, 2 hours), then Option C if results are promising and time allows.

### Step 4: Plug Into Existing Surrogate-Assisted CMA-ES Loop

The loop structure is IDENTICAL to our offline pipeline. The only changes are:

```python
# BEFORE (offline):
# cma_es.search_dim = 4074
# evaluate = evaluate_offline
# surrogate_input = omega.reshape(33, 36, 5)

# AFTER (online):
# cma_es.search_dim = 3119  (or 560 for GPIBT)
# evaluate = evaluate_online
# surrogate_input = theta  (raw vector for Option A/B)
#                   OR theta_to_functional_features(theta) for Option C
```

Everything else stays the same:

- Warmup: 5-8 generations of vanilla CMA-ES
- Pre-screening: top 20% by surrogate prediction
- Evolution control: every 3rd generation, full population evaluation
- Uncertainty ensemble: 5 models, select high-predicted OR high-disagreement
- Retraining: from scratch on accumulated dataset every 2-3 generations

### Step 5: Run Experiments

#### Experiment 1: Online Vanilla CMA-ES Baseline

- Run CMA-ES on the online policy optimization WITHOUT surrogate
- Use the same setup as the paper: on+PIBT, warehouse map
- Log: throughput per generation, total simulations, wall-clock time
- This is our baseline to beat

#### Experiment 2: Surrogate-Assisted Online CMA-ES

- Run with each surrogate variant (XGBoost, MLP, hybrid if applicable)
- Same CMA-ES hyperparameters, same random seeds
- Compare: simulations to reach 95% of baseline throughput
- Report wall-clock time savings (should be dramatic given 4× eval cost)

#### Experiment 3: Cross-Comparison

- Compare speedup ratios: offline surrogate speedup vs online surrogate speedup
- The online case should show larger wall-clock savings due to more expensive evals
- This demonstrates the framework generalizes and becomes MORE valuable as evaluation cost increases

### Step 6: Collect Canonical Observations (for Option C only)

If implementing the functional output approach:

```python
def collect_canonical_observations(n_obs=5):
    """
    Run a few online simulations with random/uniform policy parameters
    and collect diverse traffic observation tensors.

    These become fixed reference inputs for converting θ → functional features.
    """
    observations = []
    # Run simulation with uniform weights (no guidance)
    obs_uniform = run_simulation_and_collect_traffic(theta=None)
    observations.append(obs_uniform)

    # Run simulations with a few random θ vectors
    for _ in range(n_obs - 1):
        theta_random = np.random.randn(3119) * 0.1
        obs = run_simulation_and_collect_traffic(theta=theta_random)
        observations.append(obs)

    return observations
```

Store these once at the start — they're reused for all surrogate predictions.

---

## File Changes Summary

### New Files to Create

```
src/
├── simulator/
│   └── evaluate_online.py      # Online evaluation wrapper
├── surrogate/
│   └── mlp_model.py            # MLP surrogate for θ vectors
├── experiments/
│   ├── 05_online_baseline.py   # Vanilla CMA-ES for online GGO
│   └── 06_online_surrogate.py  # Surrogate-assisted online GGO
└── utils/
    └── functional_features.py  # (Optional) θ → guidance tensor conversion
```

### Files to Modify

```
src/optimizer/surrogate_cmaes.py
  - Parameterize search_dim (4074 vs 3119 vs 560)
  - Parameterize evaluate function (offline vs online)
  - Parameterize surrogate input transformation

configs/
  - Add online_pibt.yaml and online_gpibt.yaml configs
```

### Files That Stay Identical

```
src/optimizer/surrogate_cmaes.py  (core loop logic)
src/surrogate/xgboost_model.py   (works on any flat vector)
src/surrogate/training.py        (retrain logic is model-agnostic)
src/evolution_control.py          (same mechanism)
src/utils/metrics.py              (Spearman ρ, MSE)
```

---

## Expected Timeline

| Task                                        | Time Estimate                       |
| ------------------------------------------- | ----------------------------------- |
| Clone OnlineGGO, build, verify it runs      | 2-4 hours                           |
| Write evaluate_online wrapper               | 2-4 hours                           |
| Implement XGBoost surrogate on raw θ        | 1 hour                              |
| Implement MLP surrogate                     | 2 hours                             |
| Run online vanilla CMA-ES baseline          | Compute time (let it run overnight) |
| Run surrogate-assisted experiments          | Compute time (several hours each)   |
| Collect results, compare with offline       | 2-3 hours                           |
| **(Optional)** Functional features approach | 4-6 hours                           |
| **Total implementation time**               | **~2 days**                         |
| **Total including compute**                 | **~3-4 days**                       |

---

## Key Gotchas and Warnings

1. **The OnlineGGO repo may depend on a specific version of the LMAPF simulator.** Check if it has its own C++ simulator or if it references ggo_public. There may be build dependency conflicts. Resolve this FIRST before writing any Python code.

2. **The policy network architecture matters for reproducibility.** Make sure you're using their EXACT CNN architecture (3 conv layers, specific kernel sizes, specific activation functions). The parameter count must match 3,119 for PIBT. If your instantiated network has a different param count, something is wrong.

3. **θ is initialized differently than ω.** In the offline case, ω starts from a uniform distribution and is bounded. In the online case, θ starts from a standard normal Gaussian (mean=0, std=1) and is unbounded. Make sure your CMA-ES initialization matches the paper's setup.

4. **Online simulations are ~4× slower.** Budget your compute time accordingly. If offline baseline took X hours, online baseline will take ~4X hours. Plan to run overnight.

5. **Neural network weight spaces have symmetries.** Two different θ vectors can produce the identical policy (e.g., permuting hidden units). This makes the surrogate's job harder than the offline case. If XGBoost on raw θ shows poor rank correlation (<0.3), this is likely the cause, and the functional output approach (Option C) becomes necessary.

6. **Start with on+PIBT (3,119 params), not on+GPIBT (560 params).** PIBT is simpler (no guide paths, no LNS), has a more direct comparison to our offline case (also PIBT-based), and the paper has more complete results for it.

7. **Use the same map as our offline experiments (warehouse-33-36 if available, or warehouse-33-57 from the online paper).** Check which maps are shared between the two repos. Using the same map enables direct comparison of offline vs online surrogate speedup.

---

## Success Criteria

The extension is successful if:

1. ✅ Online vanilla CMA-ES baseline reproduces results roughly consistent with the paper
2. ✅ At least one surrogate variant (XGBoost or MLP) achieves Spearman ρ > 0.3 on online policy parameters
3. ✅ Surrogate-assisted online CMA-ES reaches 95% of baseline throughput with fewer total simulations
4. ✅ Wall-clock speedup is demonstrated (ideally > 2×, given the 4× per-eval cost)

The extension is a bonus success if:

- 🌟 Functional features approach (Option C) outperforms raw θ approaches
- 🌟 Online surrogate speedup ratio exceeds offline surrogate speedup ratio
- 🌟 Results hold for both on+PIBT and on+GPIBT
