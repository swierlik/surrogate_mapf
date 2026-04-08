# Accelerating Lifelong MAPF Optimization Via Surrogate-Assisted Evolution Strategies

## Thesis Overview

### Problem Statement

Guidance Graph Optimization (GGO) for Lifelong Multi-Agent Path Finding (LMAPF) relies on CMA-ES to search for optimal edge weights that maximize agent throughput. The current state-of-the-art (Zhang et al., IJCAI 2024; Zang et al., AAAI 2025) requires **50,000 full LMAPF simulations** to converge — each consisting of 1,000 timesteps with hundreds of agents. A single optimization run takes 1.2 to 55 hours depending on the map and algorithm. This computational cost is the primary bottleneck preventing real-time adaptation of guidance graphs in dynamic warehouse environments.

### Core Hypothesis

A surrogate model trained to predict the throughput of candidate guidance graphs can **pre-screen** CMA-ES candidates, filtering out low-quality solutions before they consume expensive simulation time. By exploiting the spatial structure of the guidance graph (represented as an h×w×5 tensor on a grid map), a convolutional surrogate can achieve sufficient **rank accuracy** to reduce the total number of required simulations by 2–3× while maintaining comparable final throughput.

### Contribution

This thesis implements and evaluates a surrogate-assisted CMA-ES framework for offline Guidance Graph Optimization, demonstrating that:

1. Surrogate models can learn the mapping from edge weights to throughput with sufficient rank correlation to enable effective pre-screening.
2. Convolutional architectures that exploit the spatial grid structure of guidance graphs outperform feature-agnostic baselines (XGBoost on flattened vectors).
3. The surrogate-assisted approach achieves near-equivalent throughput to vanilla CMA-ES while requiring significantly fewer real simulator evaluations.

---

## Background and Related Work

### Lifelong MAPF

Lifelong Multi-Agent Path Finding continuously assigns new goals to agents upon task completion. The objective is to maximize **throughput**: the average number of goals reached per timestep. PIBT (Priority Inheritance with Backtracking) is the state-of-the-art rule-based algorithm — extremely fast but with no inherent quality guarantees.

### Guidance Graphs

A guidance graph $G_g(V_g, E_g, \omega)$ overlays the warehouse grid with directed edge weights that alter movement and wait costs. Agents plan cost-minimal paths on this weighted graph instead of shortest paths, which implicitly creates "highways" and reduces congestion. For a 4-neighbor grid of dimension h×w, the edge weight vector $\omega \in \mathbb{R}^{|E_g|}$ encodes 5 values per non-obstacle cell: movement costs for up/down/left/right and a wait cost.

### Current Optimization Approach (Zhang et al., IJCAI 2024)

CMA-ES directly optimizes $\omega$ by:
1. Sampling a batch of b=100 candidate weight vectors from a multivariate Gaussian.
2. Evaluating each candidate by running $N_e$=5 LMAPF simulations of 1,000 timesteps.
3. Ranking candidates by throughput and updating the Gaussian toward high-throughput regions.
4. Repeating for I=100 iterations → **50,000 total simulations**.

Key properties:
- Only **relative** magnitudes of edge weights matter (scaling invariance).
- Min-max normalization is used for bounds handling.
- The weight tensor is represented as h×w×5 (4 movement directions + wait).

### Online Extension (Zang et al., AAAI 2025)

Instead of optimizing static edge weights, CMA-ES optimizes the parameters θ of a small CNN policy (3,119 params for PIBT, 560 for GPIBT) that dynamically generates guidance graphs from real-time traffic observations every m=20 timesteps. This achieves up to 30.75% throughput improvement over offline guidance but is ~4× more expensive per evaluation. Both papers explicitly identify computational cost as a key limitation and suggest surrogate-assisted optimization as future work.

---

## Methodology

### Primary Benchmark

| Parameter | Value |
|-----------|-------|
| **Map** | warehouse-33-36 |
| **MAPF Algorithm** | PIBT |
| **Number of Agents** | 400 |
| **Edge Weight Dimensions** | 4,074 (948 wait + 3,126 move) |
| **Tensor Representation** | 33 × 36 × 5 |
| **Simulation Length** | 1,000 timesteps |
| **Evaluations per Candidate** | 5 simulations (averaged) |
| **Baseline Total Simulations** | 50,000 |
| **Baseline Throughput Target** | 7.64 ± 0.01 (CMA-ES from Zhang 2024) |

### Surrogate Input/Output

```
Input:  ω represented as tensor of shape (33, 36, 5)
        - Channels 0-3: movement costs (left, right, up, down)
        - Channel 4: wait cost
        - Obstacle cells masked to zero
        - Min-max normalized (scaling invariance)

Output: Scalar predicted throughput τ ∈ ℝ
```

**Critical note:** The surrogate does NOT need to predict exact throughput. It needs to correctly **rank** candidates — specifically, it must reliably place the true top-20% of a generation within its predicted top-20%. The evaluation metric is **Spearman rank correlation** (ρ), not MSE.

### Surrogate Models (in order of implementation)

#### Model 1: XGBoost Baseline
- **Input:** Flattened ω vector (4,074 features)
- **Purpose:** Feasibility check — does *any* surrogate work?
- **Expected rank correlation:** ρ ≈ 0.3–0.5
- **Training time:** Milliseconds
- **Implementation:** scikit-learn or xgboost library, default hyperparameters

#### Model 2: Lightweight CNN
- **Input:** (33, 36, 5) tensor + binary obstacle mask channel → (33, 36, 6)
- **Architecture:**
  ```
  Conv2d(6, 16, kernel=3, padding=1) → BatchNorm → ReLU → Dropout(0.3)
  Conv2d(16, 32, kernel=3, padding=1) → BatchNorm → ReLU → Dropout(0.3)
  Conv2d(32, 64, kernel=3, padding=1) → BatchNorm → ReLU
  GlobalAveragePool → FC(64, 32) → ReLU → Dropout(0.5) → FC(32, 1)
  ```
- **Total parameters:** ~15,000–20,000
- **Training:** Adam optimizer, lr=1e-3, weight decay=1e-4, early stopping on validation loss
- **Expected rank correlation:** ρ ≈ 0.5–0.7
- **Key regularization:** Dropout, weight decay, and data augmentation (vertical flip of warehouse map preserves throughput symmetry)

#### Model 3: Hybrid (CNN Feature Extractor + XGBoost)
- **Approach:** Use trained CNN's penultimate layer (64-dim global-average-pooled features) as input to XGBoost
- **Rationale:** Combines CNN's spatial awareness with XGBoost's sample efficiency and robustness
- **Expected rank correlation:** ρ ≈ 0.5–0.7 (potentially best of both)

### Surrogate-Assisted CMA-ES Loop

```
Phase 1: WARMUP (Generations 1–8)
├── Run vanilla CMA-ES (no surrogate)
├── 8 generations × 100 candidates × 5 sims = 4,000 simulations
├── Collect 800 labeled (ω, throughput) pairs
└── Train all three surrogate models on this dataset

Phase 2: SURROGATE-ASSISTED (Generations 9–100)
├── For each generation:
│   ├── CMA-ES proposes 100 candidates
│   ├── Surrogate predicts throughput for all 100
│   ├── Select top-20 (top 20%) for real evaluation
│   ├── Run 20 × 5 = 100 real simulations
│   ├── Feed real throughput values back to CMA-ES
│   └── Add 20 new (ω, throughput) pairs to training set
├── Every 3rd generation: EVOLUTION CONTROL
│   ├── Evaluate ALL 100 candidates on real simulator
│   ├── Compare surrogate rankings vs real rankings (log Spearman ρ)
│   └── Retrain surrogate from scratch on full accumulated dataset
└── Non-control generations: retrain surrogate every 2–3 generations

Phase 3: RETURN best solution found across all real evaluations
```

### Evolution Control (Critical Safety Mechanism)

Without periodic full-population evaluations, CMA-ES will exploit inaccuracies in the surrogate and converge to solutions that look good to the model but perform poorly in reality. Evolution control prevents this by:

1. **Grounding the optimizer** — real evaluations every 3rd generation keep CMA-ES honest.
2. **Refreshing training data** — full generations provide diverse samples from the current search region.
3. **Providing diagnostics** — Spearman ρ between surrogate and real rankings at each control generation tracks whether the surrogate is drifting.

If Spearman ρ drops below 0.3 at any control generation, fall back to vanilla CMA-ES for 2–3 generations before re-engaging the surrogate.

### Expected Simulation Budget

| Component | Generations | Candidates Evaluated | Simulations |
|-----------|------------|---------------------|-------------|
| Warmup | 8 | 800 | 4,000 |
| Surrogate-assisted (non-control) | ~62 | 62 × 20 = 1,240 | 6,200 |
| Evolution control (every 3rd) | ~30 | 30 × 100 = 3,000 | 15,000 |
| **Total** | **100** | **~5,040** | **~25,200** |

**Target speedup: ~2× reduction** in simulations (25,200 vs 50,000) while reaching ≥95% of baseline final throughput. A more aggressive configuration (top-10%, control every 5th gen) could achieve ~3× but with higher risk.

---

## Evaluation Plan

### Primary Metrics

1. **Simulation efficiency:** Total real simulations required to reach X% of baseline's final throughput (for X = 90, 95, 99).
2. **Wall-clock time:** End-to-end optimization time including surrogate training overhead.
3. **Final throughput:** Best throughput achieved at the end of the full optimization run.
4. **Surrogate accuracy:** Spearman rank correlation ρ tracked across generations (reported at each evolution control point).

### Experiments

#### Experiment 1: Surrogate Feasibility (offline, static)
- Take all (ω, throughput) pairs from a full vanilla CMA-ES run.
- Train each surrogate model with 5-fold cross-validation.
- Report Spearman ρ and MSE at training set sizes of 200, 500, 1000, 2000, 5000.
- **Purpose:** Determines whether the surrogate approach is viable before building the full loop.

#### Experiment 2: Surrogate-Assisted vs Vanilla CMA-ES
- Run vanilla CMA-ES for 100 generations (baseline).
- Run surrogate-assisted CMA-ES for 100 generations (three surrogate variants).
- Same map, same agent count, same CMA-ES hyperparameters, same random seeds.
- Compare convergence curves: throughput vs number of real simulations.
- Run each configuration 3–5 times with different seeds for statistical validity.
- **Purpose:** Core result of the thesis.

#### Experiment 3: Ablation Study
- Vary pre-screening aggressiveness: top 50%, 20%, 10%.
- Vary evolution control frequency: every 2nd, 3rd, 5th generation.
- Vary warmup length: 3, 5, 8, 10 generations.
- **Purpose:** Understand sensitivity to hyperparameters.

#### Experiment 4: Generalization (stretch goal)
- Replicate Experiment 2 on a second map (random-32-32-20, setup 1 from Zhang 2024).
- **Purpose:** Show the approach is not map-specific.

---

## Implementation Plan

### Codebase and Dependencies

- **Simulator:** Use the C++ LMAPF implementation from Zhang et al. 2024's repository ([ggo_public](https://github.com/lunjohnzhang/ggo_public)).
- **CMA-ES:** [pyribs](https://pyribs.org/) (same library used by Zhang et al. 2024) or [pycma](https://github.com/CMA-ES/pycma).
- **Surrogate models:** PyTorch (CNN), XGBoost/LightGBM (tree baseline), scikit-learn (utilities).
- **Experiment tracking:** Weights & Biases or simple CSV logging.
- **Language:** Python wrapper around C++ simulator.

### Repository Structure

```
thesis-surrogate-ggo/
├── THESIS_PLAN.md              # This document
├── README.md                   # Setup and run instructions
├── configs/                    # Experiment configuration files
│   ├── baseline_cmaes.yaml
│   ├── surrogate_xgboost.yaml
│   ├── surrogate_cnn.yaml
│   └── surrogate_hybrid.yaml
├── src/
│   ├── simulator/              # Wrapper around C++ LMAPF simulator
│   │   ├── evaluate.py         # ω → throughput evaluation function
│   │   └── maps/               # Map files (warehouse-33-36, etc.)
│   ├── optimizer/              # CMA-ES integration
│   │   ├── vanilla_cmaes.py    # Baseline optimizer
│   │   └── surrogate_cmaes.py  # Surrogate-assisted optimizer
│   ├── surrogate/              # Surrogate models
│   │   ├── xgboost_model.py    # XGBoost baseline
│   │   ├── cnn_model.py        # Lightweight CNN
│   │   ├── hybrid_model.py     # CNN features + XGBoost
│   │   └── training.py         # Training and retraining logic
│   ├── evolution_control.py    # Evolution control logic
│   └── utils/
│       ├── data.py             # Dataset management (ω, throughput pairs)
│       ├── metrics.py          # Spearman ρ, MSE, ranking utilities
│       └── visualization.py    # Plotting convergence curves, etc.
├── experiments/
│   ├── 01_feasibility.py       # Experiment 1: static surrogate eval
│   ├── 02_comparison.py        # Experiment 2: main comparison
│   ├── 03_ablation.py          # Experiment 3: hyperparameter sweep
│   └── 04_generalization.py    # Experiment 4: second map
├── notebooks/                  # Analysis and visualization notebooks
└── results/                    # Saved results, plots, logs
```

---

## Step-by-Step Action Plan

### Phase 0: Infrastructure (Week 1–2)
- [ ] Clone and build the ggo_public repository
- [ ] Verify the C++ simulator compiles and runs on your machine
- [ ] Write a Python wrapper: `evaluate(omega_tensor) → throughput`
- [ ] Reproduce Zhang 2024 Setup 2 baseline results (PIBT + CMA-ES, warehouse-33-36, 400 agents)
- [ ] Confirm you can match their reported throughput of 7.64 ± 0.01
- [ ] Set up experiment logging

### Phase 1: Data Collection and Feasibility (Week 3–4)
- [ ] Run a full vanilla CMA-ES optimization (100 generations)
- [ ] Save ALL (ω, throughput) pairs — this is your retrospective dataset
- [ ] **Experiment 1:** Train XGBoost on subsets of this data, report Spearman ρ
- [ ] **Experiment 1:** Train small CNN on subsets, report Spearman ρ
- [ ] **Experiment 1:** Train hybrid, report Spearman ρ
- [ ] **DECISION GATE:** If best Spearman ρ > 0.4 at 500 samples → proceed. If < 0.3 → revisit feature engineering or architecture before continuing.

### Phase 2: Surrogate-Assisted Loop (Week 5–7)
- [ ] Implement the surrogate-assisted CMA-ES loop (Phase 2 from Methodology)
- [ ] Implement evolution control mechanism
- [ ] Implement surrogate retraining logic (retrain from scratch on accumulated data)
- [ ] **Experiment 2:** Run surrogate-assisted CMA-ES with XGBoost surrogate
- [ ] **Experiment 2:** Run surrogate-assisted CMA-ES with CNN surrogate
- [ ] **Experiment 2:** Run surrogate-assisted CMA-ES with hybrid surrogate
- [ ] Run each 3–5 times with different seeds
- [ ] Compare convergence curves against vanilla baseline

### Phase 3: Analysis and Ablation (Week 8–9)
- [ ] **Experiment 3:** Ablation on pre-screening ratio (50%, 20%, 10%)
- [ ] **Experiment 3:** Ablation on evolution control frequency
- [ ] **Experiment 3:** Ablation on warmup length
- [ ] Analyze surrogate drift: plot Spearman ρ over generations
- [ ] Visualize best guidance graphs from surrogate-assisted vs vanilla CMA-ES

### Phase 4: Writing and Stretch Goals (Week 10–12)
- [ ] Write thesis (Introduction, Background, Method, Experiments, Conclusion)
- [ ] **(Stretch)** Experiment 4: replicate on random-32-32-20
- [ ] **(Stretch)** Single demonstration on online policy optimization (Zang 2025)
- [ ] Final revisions and submission

---

## Key References

1. **Zhang, Y. et al. (2024).** "Guidance Graph Optimization for Lifelong Multi-Agent Path Finding." *IJCAI 2024.* — Defines GGO, introduces CMA-ES and PIU methods. **Primary baseline.**
2. **Zang, H. et al. (2025).** "Online Guidance Graph Optimization for Lifelong Multi-Agent Path Finding." *AAAI 2025.* — Extends to online dynamic guidance policies. Explicitly calls for surrogate-assisted optimization in the conclusion.
3. **Hansen, N. (2016).** "The CMA Evolution Strategy: A Tutorial." — CMA-ES reference.
4. **Jin, Y. (2011).** "Surrogate-Assisted Evolutionary Computation: Recent Advances and Future Challenges." *Swarm and Evolutionary Computation.* — Survey of surrogate-assisted optimization methods.
5. **Zhang, Y. et al. (2022).** "Deep Surrogate Assisted MAP-Elites for Automated Hearthstone Deckbuilding." *GECCO 2022.* — Cited by both GGO papers as a relevant surrogate-assisted approach.

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Surrogate rank correlation too low (<0.3) | Medium | High | Feature engineering, PCA on CMA-ES population, simpler map first |
| CMA-ES exploits surrogate inaccuracies | High | High | Evolution control every 3rd generation, fallback to vanilla |
| C++ simulator build/integration issues | Medium | Medium | Start infrastructure phase early, contact paper authors if needed |
| 2× speedup not statistically significant | Low | Medium | Run enough seeds (5+), use appropriate statistical tests |
| Insufficient time for all experiments | Medium | Low | Prioritize Experiments 1 and 2; Experiments 3 and 4 are stretch goals |
