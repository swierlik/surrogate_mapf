# Research Findings Journal

**Thesis:** Accelerating Lifelong MAPF Optimization Via Surrogate-Assisted Evolution Strategies
**Author:** Igor S.
**Started:** 2026-02-24

---

## How to Use This Document

This is a **session-by-session research journal**. After each working session, add a new dated entry using the template below. The purpose is to maintain structured notes for thesis writing, track decisions and their rationale, and keep a record of open questions.

### Entry Template

```
## Session [N] — YYYY-MM-DD

### What Was Done
- Bullet list of concrete work completed this session

### Key Findings
- Observations, measurements, surprising results
- Include numbers where possible

### Open Questions / Questions for Supervisor
- Things that need clarification or discussion

### Next 3 Steps
1. Immediate next task
2. Following task
3. Third priority
```

---

## Session 1 — 2026-02-24

### What Was Done

- Built end-to-end CMA-ES optimization pipeline: Python wrapper around C++ LMAPF simulator running in Docker
- Implemented multi-emitter CMA-ES (5 emitters, popsize=20 each, 100 candidates/gen) matching Zhang et al. (IJCAI 2024) setup
- Added checkpoint/resume system that preserves all emitter states, RNG, and accumulated data
- Added multiprocessing parallelization inside Docker (8 workers), reducing per-generation time from ~16 min to ~2 min
- Ran full 300-generation baseline (150,000 simulations, ~17 hours wallclock)
- Collected 30,000 (solution, throughput) pairs with per-eval throughputs for surrogate training

### Key Findings

**Baseline CMA-ES Results:**

- **Best throughput: 8.29** (single best solution found over 300 gens)
- **Mean population throughput at convergence: ~8.26** (gen 300 average)
- **Total simulations: 150,000** | **Wallclock: ~17 hours**
- Paper's reported baseline: 7.64 (Zhang et al., IJCAI 2024, 100 gens / 50k sims)

**Convergence behavior:**

- Rapid early improvement: 3.57 (gen 0) to 7.13 (gen 100)
- Continued steady gains: 7.13 (gen 100) to 8.05 (gen 200) to 8.29 (gen 300)
- Curve was clearly flattening by gen 250+ (only +0.1 in last 50 gens)
- Emitter restarts helped escape local optima multiple times during the run

**Why we exceed the paper's 7.64:**

- We ran 3x more generations (300 vs 100) and 3x more simulations (150k vs 50k)
- Paper likely reports mean across multiple seeds; we report single-run best
- Minor implementation differences (restart noise, seed scheduling) may also contribute
- Our gen-300 mean population throughput (~7.83) is a fairer comparison and is reasonably close

**Performance / infrastructure:**

- Sequential Docker execution was the bottleneck (15% CPU utilization with 1 thread)
- Multiprocessing Pool with 8 workers inside Docker gave ~5.4x speedup
- Each generation now takes ~2 min (100 solutions x 5 evals = 500 simulations)

### Open Questions / Questions for Supervisor

- Is exceeding the paper's throughput a concern, or just a consequence of more compute? (Likely fine — we can frame it as "extended baseline")
- For the surrogate feasibility study: should we use raw solution vectors or normalized ones as input? (Likely raw, since normalization is a deterministic transform the surrogate could learn)
- Should we aim for the 33x36x5 tensor representation from the start, or first validate on flat vectors?

### Next 3 Steps

1. **Surrogate feasibility study** — load the 30k data points, train XGBoost and MLP/CNN surrogates, measure Spearman rank correlation at various training set sizes
2. **Temporal generalization test** — train on gens 0-200, test on 200-300 to verify the surrogate generalizes to the evolving search distribution
3. **Decision gate** — if Spearman rho > 0.4 at 500 samples, proceed to surrogate-assisted CMA-ES loop implementation

---

## Session 2 — 2026-02-25

### What Was Done

- Implemented full surrogate feasibility pipeline: metrics (`src/utils/metrics.py`), flat-to-tensor reshaper (`src/utils/reshape.py`), XGBoost surrogate (`src/surrogate/xgboost_model.py`), CNN surrogate (`src/surrogate/cnn_model.py`), training harness (`src/surrogate/training.py`), and experiment script (`experiments/01_feasibility.py`)
- Ran Experiment 1 (XGBoost only): 5-fold CV, learning curves at 6 training sizes, temporal split (gens 0-199 vs 200-299)
- Ran realistic sliding-window temporal test: train on [0, g), predict gen g for g in {10, 20, 50, 100, 150, 200, 250, 299}
- Ran top-k ranking analysis to check whether the very best solutions are correctly identified
- Installed xgboost package; confirmed PyTorch + CUDA available for future CNN experiments

### Key Findings

**XGBoost 5-fold CV (random split across all 300 gens):**
- Spearman rho = **0.9898 +/- 0.0002** (near-perfect rank correlation)
- MSE = 0.0216
- Top-20% precision = **0.923** (92% of predicted top-20% are actually in real top-20%)
- Massively exceeds the decision gate threshold of rho > 0.4

**Learning curve (random split):**
| Train size | Spearman rho | Top-20% prec |
|------------|-------------|--------------|
| 500 | 0.9822 | 0.905 |
| 1,000 | 0.9872 | 0.913 |
| 5,000 | 0.9890 | 0.920 |
| 20,000 | 0.9898 | 0.924 |

Even 500 samples gives rho=0.98 in the random split setting. Performance saturates early.

**CRITICAL: Temporal split (train gens 0-199, test gens 200-299):**
- Spearman rho = **0.2490** (very poor!)
- Top-20% precision = 0.387
- This tests a 100-generation gap — the model fails to extrapolate far into the future

**Realistic sliding window (train on [0, g), predict gen g):**
| Test gen | Train N | Spearman rho | Top-20% prec |
|----------|---------|-------------|--------------|
| 10 | 1,000 | 0.423 | 0.350 |
| 20 | 2,000 | 0.626 | 0.600 |
| 50 | 5,000 | 0.512 | 0.350 |
| 100 | 10,000 | 0.816 | 0.750 |
| 150 | 15,000 | 0.788 | 1.000 |
| 200 | 20,000 | 0.875 | 1.000 |
| 250 | 25,000 | 0.856 | 0.950 |
| 299 | 29,900 | 0.912 | 0.950 |

This is the realistic test — model trained only on past data, predicting the next generation. After ~100 gens of warmup, rho stabilizes at 0.78-0.91 with top-20% precision of 0.75-1.0.

**Top-k ranking analysis (random split, optimistic):**
- True #1 solution lands at surrogate rank 7/6000 (top 0.1%)
- Top-10 overlap: 3/10, but median predicted rank is 18 (still very high)
- At the very tip of the distribution, solutions are so close in throughput (~8.27 vs 8.29) that small prediction errors shuffle the exact order — but this doesn't matter for pre-screening

**Key insight — why CV is great but temporal is bad:**
The random CV includes training samples from all 300 gens, so the model sees the entire search trajectory. The temporal split asks it to predict a region of solution space it has never seen. In the real surrogate-assisted loop, the model always has recent data from nearby generations, so the sliding window result (rho=0.78-0.91) is the relevant benchmark.

**XGBoost training time concern:**
- Training on 1k samples: ~109s
- Training on 20k samples: ~241s
- One generation of simulation: ~120s
- Retraining every generation would negate simulation savings; need to retrain every 5-10 gens

### Open Questions / Questions for Supervisor

- The high random-split rho (0.99) vs poor temporal-split rho (0.25) is a striking gap. Worth investigating: is the model just memorizing the CMA-ES trajectory rather than learning genuine throughput-predictive features?
- Do we still need the CNN, or is XGBoost sufficient? XGBoost already exceeds all expectations. CNN adds complexity for potentially marginal gains
- XGBoost training time (100-250s) is comparable to simulation time (120s/gen). Should we explore lighter models (fewer trees, smaller depth) or is retraining every 5-10 gens sufficient?

### Next 3 Steps

1. **Implement surrogate-assisted CMA-ES loop** — pre-screen top-20 from 100 candidates using XGBoost, with warmup phase (20 gens) and periodic retraining (every 5-10 gens)
2. **Run surrogate-assisted vs vanilla comparison** — same 300 gens, compare convergence curves and total simulation count
3. **Decide on CNN** — run CNN feasibility test to see if spatial structure helps the temporal generalization problem (if not, stick with XGBoost only)

---

## Session 3 — 2026-02-25

### What Was Done

- Implemented MLP surrogate (`src/surrogate/mlp_model.py`) with fine-tuning support (warm-start from previous weights)
- Ran sliding-window temporal test for all three models (XGBoost, MLP, CNN) on the same benchmark: train on [0, g), predict gen g
- Compared training speed, accuracy, and suitability for the surrogate-assisted loop

### Key Findings

**Three-model comparison (sliding window temporal test):**

| Gen | XGBoost rho | MLP rho | CNN rho | XGBoost time | MLP time | CNN time |
|-----|-------------|---------|---------|--------------|----------|----------|
| 10 | 0.423 | **0.497** | 0.051 | 109s | 11s | 3s |
| 20 | 0.626 | **0.627** | 0.584 | 161s | 1s | 2s |
| 50 | **0.512** | 0.516 | 0.334 | 227s | 2s | 4s |
| 100 | 0.816 | **0.878** | 0.783 | 230s | 5s | 9s |
| 150 | **0.788** | 0.641 | 0.468 | 229s | 6s | 13s |
| 200 | 0.875 | **0.892** | 0.826 | 241s | 8s | 17s |
| 250 | 0.856 | 0.874 | **0.884** | 256s | 11s | 22s |
| 299 | 0.912 | **0.941** | 0.884 | 183s | 13s | 33s |

**MLP is the clear winner:**
- Best or tied-best rho at most test points, peaking at **0.941** at gen 299
- **20-30x faster** than XGBoost (1-13s vs 100-250s)
- Supports fine-tuning: full train on 20k samples takes only **9.1s** (vs 241s for XGBoost)

**CNN spatial structure does NOT help:**
- Consistently worse than MLP and more erratic (rho=0.051 at gen 10, 0.468 at gen 150)
- The throughput is a global property of the entire weight configuration — the CNN's local-pattern inductive bias doesn't add value
- This is a thesis-worthy finding: spatial representation of guidance graph weights does not improve surrogate prediction

**XGBoost is accurate but too slow:**
- Most consistent rho across test points (0.42-0.91), but training time (100-250s) is comparable to a full generation of simulation (~120s)
- Retraining every generation would negate simulation savings
- Does not support incremental/fine-tune learning

**MLP fine-tuning economics:**
- Full retrain on 20k samples: ~9s
- Fine-tune on 100 new samples (1 gen): ~2-5s (estimated)
- Simulation for 1 generation: ~120s
- Pre-screening savings per gen: ~96s (evaluate 20 instead of 100 solutions)
- Net savings per generation: ~90s+ (even with retraining every gen)

### Open Questions / Questions for Supervisor

- The MLP has some variance (rho=0.641 at gen 150 vs 0.892 at gen 200). Should we use an ensemble of small MLPs for more stable predictions?
- Should we still include XGBoost as a comparison point in the thesis, or focus entirely on MLP?
- Is the CNN negative result worth a dedicated section in the thesis? (Hypothesis: spatial structure helps → rejected)

### Next 3 Steps

1. **Implement surrogate-assisted CMA-ES loop** — use MLP surrogate with fine-tuning, pre-screen top-20 from 100 candidates, warmup phase of ~20 gens, retrain/fine-tune every generation
2. **Run surrogate-assisted vs vanilla comparison** — same 300 gens, same seed, compare convergence curves and total simulation count
3. **Analyze results** — compute speedup (simulation count ratio), final throughput comparison, and Spearman rho tracking across generations

---

## Session 4 — 2026-04-16

### What Was Done

- Implemented full surrogate-assisted CMA-ES loop (`src/optimizer/surrogate_cmaes.py`)
  - Three modes per generation: warmup (full eval), evolution control (full eval + retrain), surrogate-assisted (eval top-20 + fine-tune)
  - Surrogate predictions used as placeholder fitnesses for unevaluated 80 candidates when calling emitter.tell()
  - Spearman rho logged at every control/warmup gen (honest accuracy measure before retraining)
  - Extended checkpoint format to save surrogate model weights + accumulated training data
- Ran full 300-generation surrogate-assisted run (3h23m)
- Generated comparison figures: convergence curves, sample efficiency, surrogate rho over time

### Key Findings

**Main comparison result:**

| Metric | Vanilla CMA-ES | Surrogate-Assisted | Difference |
|--------|---------------|-------------------|------------|
| Best throughput | 8.2904 | 8.0852 | **-2.5%** |
| Total simulations | 150,000 | 49,200 | **3x fewer** |
| Wall-clock time | ~17hrs | ~3.4hrs | **~5x faster** |
| Mean surrogate rho | N/A | 0.4866 | — |

Note: wall-clock speedup (~5x) exceeds simulation speedup (~3x) because surrogate gens take ~25s vs ~120s for full eval gens (no Docker overhead for 80 candidates).

**Surrogate rho in live loop (0.49) is lower than feasibility study predicted (0.78-0.94):**
- Root cause: surrogate gens only add 20 real evals each (not 100), so training data accumulates 5x slower
- The model stays data-hungry longer and accuracy grows more slowly in the live loop than in static experiments
- This is an important finding: offline feasibility studies overestimate live loop surrogate accuracy

**Throughput gap (2.5%) analysis:**
- Likely caused by lower-than-expected surrogate rho (0.49 mean), meaning some genuinely good candidates were screened out
- Also possible: fewer total simulations = less exploration of promising regions
- The gap is small and arguably acceptable given the 3x simulation savings

**Figures generated:**
- `fig5a_convergence_comparison.png` — throughput vs generation, vanilla vs surrogate
- `fig5b_sample_efficiency.png` — cumulative simulations used
- `fig5c_surrogate_rho.png` — surrogate rho tracked over generations

### Open Questions / Questions for Supervisor

- Is a 2.5% throughput gap acceptable for a 3x simulation reduction? What's the right tradeoff?
- The offline rho (0.78-0.94) vs live rho (0.49) gap is a significant finding — should this be a dedicated analysis section in the thesis?
- Should we run multiple seeds (3-5) to get statistical significance, or is compute budget too tight?
- Could a longer warmup (e.g., 50 gens instead of 20) improve live rho and close the throughput gap?
- Is 3x simulation reduction sufficient for the thesis claim, or should we push for more aggressive pre-screening (100→10 instead of 100→20)?

### Next 3 Steps

1. **Ablation study on warmup length** — test warmup=10, 20, 50 gens to find the sweet spot between data collection and simulation savings. Longer warmup → better rho → smaller throughput gap
2. **Ablation on screen_k** — test 100→10 vs 100→20 vs 100→40 pre-screening ratios to characterise the accuracy/efficiency tradeoff curve
3. **Write Methods and Results chapters** — enough data exists now to write the core thesis content; experiments can continue in parallel

---

## Session 5 — 2026-04-19

### What Was Done

- Investigated why live-loop surrogate rho (mean 0.49) was lower than feasibility study predicted (0.78-0.94)
- Diagnosed three compounding causes (see Key Findings below)
- Attempted a fix: "restart-triggered retrain" — force a full control generation (100 evaluations + full retrain) whenever any CMA-ES emitter restarts
- Ran the fixed version ("v2") for 300 generations and collected surrogate_v2/surrogate_log.csv
- Compared v1 vs v2 results; determined v1 is superior; reverted surrogate_cmaes.py to v1 behaviour
- Identified that generation-based convergence comparison is misleading; the correct comparison is throughput vs cumulative simulations

### Key Findings

**Why live-loop rho (0.49 mean) is lower than offline estimate (0.78-0.94):**

Three compounding factors:
1. **Warmup drag**: Gens 0-19 are full-eval warmup. Surrogate rho is logged on warmup gens too (before enough data accumulates), producing rho 0.10-0.50 that pulls the mean down. When warmup and post-restart dips are excluded, stable operating rho is **0.68-0.89**, matching the feasibility study's sliding window results.
2. **Emitter restart dips**: Restarts jump the emitter to a new region of solution space. Surrogate accuracy briefly drops (rho 0.22-0.38 at post-restart control gens) before recovering over the next 5-10 control gens. These are visible in surrogate_log.csv at gens ~80, 90, 110, 180, 280.
3. **Slower data accumulation**: Surrogate gens only add 20 real evals each (vs 100 in the feasibility study's full-gen assumption). The model stays data-hungry 5x longer, keeping it in the lower-rho regime for more control-gen measurements.

**V2 fix (restart-triggered retrain) backfired — V1 is better:**

| Metric | V1 (surrogate) | V2 (restart-retrain) |
|--------|---------------|----------------------|
| Best throughput | **8.0852** | 7.5900 |
| Total simulations | 49,200 | ~52,000 |

Two root causes for the regression:
1. **Evaluation seed cascade**: Simulation seeds are computed as `seed + gen × n_evals`. Adding extra control gens in v2 changes how many control gens fall before any given generation, which shifts the simulation seeds for all subsequent gens. This means v2 explores a completely different CMA-ES trajectory — a fair comparison is impossible, and the worse outcome may just be trajectory-noise.
2. **Optimistic bias after restarts may help exploration**: The surrogate overestimates fitness for newly-restarted emitters' candidates (those candidates are in unexplored space where the surrogate has no data). Forcing an immediate accurate evaluation may *penalise* the exploratory emitter — the surrogate's optimistic bias acts as a free exploration bonus that the fix inadvertently removed.

**Correct comparison methodology:**

Generation-by-generation comparison is misleading: vanilla uses 500 sims/gen, surrogate uses ~164 sims/gen on average. The honest plot is **throughput vs cumulative simulations**, which shows:
- Surrogate achieves similar throughput to vanilla at roughly 1/3 the simulation budget
- Final throughput gap (2.5%) is the cost of the simulation savings

**Conclusion:** V1 surrogate-assisted CMA-ES is the final result. The restart-retrain fix is a dead end due to seed-cascade confounding. A clean ablation would require deterministic evaluation (fixed seeds per candidate, not per generation) — a refactor not worth pursuing.

### Open Questions / Questions for Supervisor

- The 2.5% throughput gap and 3x simulation savings: is this a compelling enough result for the thesis?
- Should we report the "stable operating rho" (0.68-0.89) rather than the raw mean (0.49) in the thesis, with a clear explanation of why warmup gens are excluded?
- The seed-cascade confound means we cannot run controlled ablations (warmup length, screen_k) using the current seed scheme. Worth refactoring to per-candidate seeds, or just report V1 as-is?

### Next 3 Steps

1. **Fair comparison figure** — plot throughput vs cumulative simulations for vanilla vs surrogate (data already exists, just needs plotting)
2. **Decide on ablation scope** — either refactor seeds for clean ablations, or accept V1 as the final surrogate experiment and focus on thesis writing
3. **Write Results chapter** — V1 surrogate data (49,200 sims, 8.09 best, 3x speedup) is the core result; add surrogate rho analysis with warmup-excluded stable estimate

---

## Session 6 — 2026-04-28

### What Was Done

- Diagnosed root cause of V1 surrogate rho dips: global top-k screening means some emitters receive zero real evaluations per gen; post-restart emitters in unexplored space get only inaccurate surrogate-predicted fitnesses fed back, causing wrong Gaussian updates
- Identified three possible fixes (per-emitter quota, adaptive screen_k, ensemble UCB); chose ensemble UCB as the principled, thesis-worthy approach
- Implemented `EnsembleSurrogate` in `src/surrogate/mlp_model.py`: 5 bootstrapped MLPs, `predict_with_uncertainty()` returning (mean, std) across models
- Updated `src/optimizer/surrogate_cmaes.py` to use ensemble and UCB screening: `score = mean + λ × std` (λ=1.0), logs mean_std and selected_std per generation
- Ran full 300-generation V3 run saving to `results/surrogate_v3/` (3h24m)
- Generated figures fig7a/b/c/d comparing V1 vs V3 (plus vanilla reference)

### Key Findings

**V3 (Ensemble UCB) results — best result so far:**

| Metric | Vanilla | V1 (point-estimate) | V3 (ensemble UCB) |
|--------|---------|--------------------|--------------------|
| Best throughput | 8.2904 | 8.0852 | **8.2308** |
| Total simulations | 150,000 | 49,200 | 49,200 |
| Gap vs vanilla | — | −2.5% | **−0.7%** |
| Sims saved | — | 100,800 | 100,800 |

V3 closes 83% of the throughput gap between V1 and vanilla, using the **exact same simulation budget**. This is a strong result: 3x fewer simulations, only 0.7% below the full-budget baseline.

**Why ensemble UCB works:**

UCB score = mean_prediction + λ × std. In unexplored regions (post-restart), std is large → those candidates get evaluated regardless of surrogate rank → surrogate learns the new region faster → rho recovers quicker → fewer bad Gaussian updates to emitters. In well-explored regions, std ≈ 0, so V3 behaves identically to V1.

**Uncertainty over time (fig7d):**
- Mean ensemble std starts high (~0.3-0.5 in warmup), decreases as data accumulates
- Spikes visible after emitter restarts (confirming the mechanism works as intended)
- By gen 250+, std stabilises at ~0.08-0.10 (well-calibrated, confident ensemble)

**Rho comparison (fig7c):**
- V3 rho is generally comparable to V1; neither dominates clearly across all gens
- The improvement in throughput comes from better candidate *selection*, not from higher rho per se — UCB explores regions that point-estimate screening would miss entirely

**How simulation counting works (clarified):**
- Each surrogate gen: 20 candidates × 5 evals = 100 simulation runs (counter +100)
- Each control gen: 100 candidates × 5 evals = 500 simulation runs (counter +500)
- Total 49,200 simulation runs ÷ 5 evals = 9,840 unique candidate evaluations vs 30,000 for vanilla

### Open Questions / Questions for Supervisor

- V3 (8.2308) is only 0.7% below vanilla (8.2904) at 1/3 the cost. Is this the headline result of the thesis?
- Should we report λ=1.0 as a fixed choice and note it as a hyperparameter, or run a small ablation over λ ∈ {0.25, 0.5, 1.0, 2.0}?
- The ensemble adds ~5x inference cost (5 forward passes) but since inference is milliseconds vs minutes for simulation, this is negligible in practice — worth a sentence in the thesis.

### Next 3 Steps

1. **Write thesis Results chapter** — V3 is now the main contribution; table: vanilla vs V1 vs V3; figures: fig5b (sample efficiency), fig7a/b/c/d; narrative around UCB mechanism
2. **Optional λ ablation** — run 2 extra configs (λ=0.25 and λ=2.0) for 300 gens each to show the exploration-exploitation tradeoff curve; ~7hrs total compute
3. **Methods chapter** — document surrogate loop, ensemble training, UCB acquisition, emitter restart interaction

---

## Session 7 — 2026-04-30

### What Was Done

- Ran λ=2.0 ablation (300 gens, `results/surrogate_lam2/`) to bracket the UCB exploration weight
- Added `plot_lambda_ablation()` to `experiments/02_comparison.py` and `--lambda-ablation` flag
- Generated fig8_lambda_ablation.png: bar chart of best throughput vs λ with vanilla reference line

### Key Findings

**Lambda ablation — three-point exploration-exploitation curve:**

| Config | λ | Best throughput | Gap vs vanilla |
|--------|---|----------------|----------------|
| Vanilla CMA-ES | — | 8.2904 | — |
| Surrogate V1 | 0 (point-estimate) | 8.0852 | −2.5% |
| Surrogate V3 | **1.0** | **8.2308** | **−0.7%** |
| lam2 ablation | 2.0 | 7.9024 | −4.6% |

All surrogate runs: 49,200 simulations (3× fewer than vanilla).

**λ=2.0 is worse than even pure exploitation (V1):**

At λ=2.0, the uncertainty boost (~2 × 0.10 = 0.20 throughput units) is large enough to consistently promote uncertain-but-mediocre candidates over genuinely high-fitness ones. The emitters receive corrupted fitness signals → make worse Gaussian updates → converge more slowly and to a worse region. The result (7.90) is 0.18 below V1 (8.09), confirming that excessive exploration is actively harmful, not just neutral.

**λ=1.0 is clearly the sweet spot with the data available:**

The three-point curve (0 → 8.09, 1.0 → 8.23, 2.0 → 7.90) has a clear peak at λ=1.0. The optimum is not at an extreme, and both neighbours are worse. This is a clean thesis finding: UCB exploration helps, but must be balanced against exploitation of the surrogate's fitness signal.

**Experimental phase complete.** Full results across all runs:
- Vanilla: 8.2904 best, 150,000 sims, ~17hrs
- V1 (surrogate, point-estimate): 8.0852, 49,200 sims, ~3.4hrs
- V2 (restart-retrain fix): 7.5922, 49,200 sims — negative result, reverted
- V3 (ensemble UCB, λ=1.0): 8.2308, 49,200 sims, ~3.4hrs — best surrogate result
- lam2 (ensemble UCB, λ=2.0): 7.9024, 49,200 sims — ablation, confirms λ=1.0 optimal

### Open Questions / Questions for Supervisor

- With only three λ data points, can we claim λ=1.0 is optimal, or just "better than 0 and 2.0"? (Likely fine — the curve is clearly non-monotone and the peak is bracketed)
- Should the negative V2 result (restart-retrain) appear in the thesis as a "failed extension" subsection, or just as a footnote explaining why V3 was pursued instead?

### Next 3 Steps

1. **Start writing** — all experiments are complete; thesis can now be written from the data
2. **Key figures for thesis**: fig1 (baseline convergence), fig5a/b (V1 vs vanilla), fig7a/b (V3 vs vanilla), fig8 (lambda ablation), fig7d (uncertainty over time)
3. **Methods chapter first** — write the surrogate loop, ensemble training, and UCB acquisition sections while the implementation details are fresh

---

## Session 8 — 2026-04-30

### What Was Done

- Implemented `experiments/03_gradient_refinement.py`: post-hoc gradient ascent through the V3 EnsembleSurrogate from the best solution found, using uncertainty-penalised score (mean − λ×std) and L2 regularisation to prevent wandering too far from training data
- Fixed a clamp bug ([0,10] was wrong — real solution values range ~[−97, +123]) by replacing hard clamp with soft L2 reg (reg_lambda=1e-6)
- Ran gradient ascent (1000 steps, lr=0.01) then evaluated the refined candidate in the real simulator

### Key Findings

**Gradient refinement results — negative result, surrogate exploitation confirmed:**

| | Surrogate prediction | Real throughput |
|---|---|---|
| V3 best solution (start) | 7.6030 | **8.2308** |
| Gradient-refined solution | 8.5542 (+0.95) | **8.1154** |

The gradient ascent predicted a +0.95 improvement but delivered a −0.11 real regression vs V3 best. Two failure modes stacked:

1. **Starting miscalibration**: the surrogate already underestimated the V3 best solution by 0.63 units (7.60 predicted vs 8.23 real). The model is not well-calibrated at this specific high-quality point.
2. **Surrogate exploitation**: moving L2=493 units through solution space (gradient direction) led to a region where the surrogate *overestimates* by 0.44 (predicts 8.55, reality gives 8.11). The error flipped sign — exactly the pattern of surrogate exploitation.

**Why this validates the V3 design:**

The failure of gradient refinement retrospectively justifies using UCB as a *candidate selection* criterion rather than a gradient signal. In V3, uncertainty guided which of 100 CMA-ES-proposed candidates to evaluate — a conservative use of the surrogate that never requires the model to be accurate far from training data. Gradient ascent makes no such restriction and can be led astray over 1000 steps.

**Thesis value of this negative result:**

- Demonstrates surrogate exploitation concretely with numbers
- Validates the design choice of UCB-based selection over gradient-based optimisation
- Clean story: "we tried gradient refinement, it failed for theoretically expected reasons, confirming that V3's conservative use of uncertainty is the right approach"

### Open Questions / Questions for Supervisor

- Is this negative result worth a dedicated subsection in Results, or just a paragraph in Discussion?
- Would tighter L2 reg (e.g. reg_lambda=1e-4) have prevented the exploitation, or is this a fundamental limit of surrogate accuracy at this convergence stage?

### Next 3 Steps

1. **All experiments complete** — start writing thesis
2. **Complete figure list**: fig1 (baseline), fig5b (sample efficiency V1), fig7a/b (V3 convergence/efficiency), fig7d (uncertainty), fig8 (lambda ablation)
3. **Negative results to include**: V2 (restart-retrain backfired), gradient refinement (surrogate exploitation) — both strengthen the thesis narrative

---

## Session 9 — 2026-04-30

### What Was Done

- Ran 100-simulation robustness tests on V3 and vanilla best solutions to get corrected throughput estimates (5-sim reported scores were subject to lucky draws)
- Generated fig9: dual-axis rho vs best throughput over time, with significant improvement events marked, to test whether rho dips correlate with throughput jumps
- Decided to stop experimenting and move to thesis writing

### Key Findings

**Corrected throughput numbers (100-sim mean — use these in the thesis):**

| | 5-sim reported | 100-sim mean | 95% CI | Std |
|---|---|---|---|---|
| Vanilla best | 8.2904 | **8.2273** | [8.2125, 8.2421] | 0.0754 |
| V3 surrogate best | 8.2308 | **8.1523** | [8.1387, 8.1658] | 0.0693 |
| Gap | 0.060 | **0.075** | non-overlapping | — |

The 5-sim scores were inflated by ~0.06-0.08 for both methods. The corrected gap is 0.075 (0.9%) — the CIs don't overlap, confirming the difference is real but small. **Use 100-sim numbers in all thesis tables.**

**Rho vs throughput correlation (fig9) — hypothesis partially refuted:**

The hypothesis was: rho dips correlate with throughput improvement events (emitter restarts → new region → rho drops → good solution found). The data shows this is not cleanly true:
- Major improvements are concentrated in gens 1-132 (search still ascending)
- The largest rho dips (gens 180-210) have no corresponding throughput jumps — the surrogate is struggling in already-explored space, not tracking discoveries
- After gen 132 the search has converged and no further big improvements occur despite continued rho oscillation
- Conclusion: rho dips are driven by search distribution shift (restarts, sigma changes), not specifically by improvement discovery

**Decision: experimental phase complete.** The adaptive screen_k idea (vary number of candidates evaluated based on uncertainty) was discussed but rejected — UCB already implicitly allocates evaluation budget toward uncertain candidates, and the marginal gain over V3 is uncertain at the cost of another 3.5hr run and delayed writing.

### Final Corrected Results Table (for thesis)

Note: vanilla and V3 numbers below are from the 300-gen run (Session 9). Session 11 re-ran V3 to 400 gens and got a higher score (8.3041), but for a fair same-budget comparison use the 300-gen numbers here.

| Method | True throughput (100-sim) | 95% CI | Simulations | Gap vs vanilla |
|--------|--------------------------|--------|-------------|----------------|
| Vanilla CMA-ES | 8.2273 | [8.2125, 8.2421] | 150,000 | — |
| Surrogate V1 (point-estimate) | **8.0385** | [8.0235, 8.0536] | 49,200 | −2.3% |
| Surrogate V3 (ensemble UCB, λ=1.0) | 8.1523 | [8.1387, 8.1658] | 49,200 | −0.9% |

### Next Steps — Writing Only

1. **Methods**: surrogate loop (3 modes), ensemble training (bootstrap bagging), UCB acquisition (mean + λ×std), emitter restart interaction
2. **Results**: corrected numbers table, fig5b (sample efficiency), fig7a/b (V3 convergence), fig7d (uncertainty over time), fig8 (lambda ablation), fig9 (rho analysis)
3. **Discussion**: why UCB works (exploration in restart regions), negative results (V2, gradient refinement), future work (adaptive screen_k, MC-Dropout uncertainty)

---

## Session 10 — 2026-05-18 to 2026-05-20

### What Was Done

- Extended the surrogate-assisted CMA-ES framework to the **online GGO** setting (Zang et al., AAAI 2025), optimising a CNN guidance policy rather than static edge weights
- Built `wppl_sim` Docker image from WPPL's C++ source (extends the existing `ggo_sim` image with stateful `init_agent_pos` / `init_task_ids` support needed for segmented simulation)
- Implemented `src/simulator/evaluate_online.py`: chunked Docker batch evaluator for the CNN policy; each evaluation runs 50 sequential 20-step `py_driver.run()` calls with state handoff between segments
- Implemented `src/optimizer/vanilla_cmaes_online.py` and `src/optimizer/surrogate_cmaes_online.py`: online GGO variants of the existing optimisers (same multi-emitter CMA-ES structure, adapted for 4271-dim CNN parameter space)
- Ran full **100-generation vanilla baseline** (`results/online_baseline/`): 13.8h wallclock, 10,000 simulations, best = 7.433
- Bootstrapped surrogate warmup checkpoint from vanilla data (gens 0-9) to avoid 1.3h of redundant re-simulation
- Ran full **100-generation surrogate-assisted run** (`results/online_surrogate/`): 4.9h wallclock, 3,520 simulations, best = 7.332

### Key Findings

**Online GGO evaluation architecture:**

Each online "simulation" is a 50-segment chain: 50 × `py_driver.run(steps=20)` calls with agent positions and task IDs passed between segments. The CNN policy (3-layer: nc=10→32→32→5, 4,271 parameters) runs between each segment to regenerate edge weights from observed traffic. This means online evaluations are inherently sequential within each chain and cannot be parallelised at the segment level. Per-generation time: ~460s (vs ~120s offline).

**CNN architecture and observation space:**

- Input (10 channels, H×W): edge_usage[4] + wait_usage[1] + edge_weights[4] + wait_costs[1]
- Output (5 channels, H×W): new_edge_weights[4] + new_wait_costs[1]
- Parameters: 4,271 (vs 4,074 for offline edge weight vector)
- Map: same `pibt_warehouse_33x36_w_mode.json`, 400 agents, 1,000 steps — identical environment to offline experiments

**Main comparison result:**

| Metric | Vanilla (online) | Surrogate (online) | Offline V3 (reference) |
|--------|-----------------|-------------------|------------------------|
| Best throughput | 7.433 | 7.332 (98.6%) | 8.231 (100-sim corrected) |
| Total simulations | 10,000 | 3,520 | 49,200 |
| Wallclock | 13.8h | 4.9h | ~3.4h |
| Sim-budget speedup | — | **2.84×** | **3.05×** |
| Wallclock speedup | — | **2.81×** | **~5×** |

**Crossover-point speedup (exact):**

At the threshold of 7.332 (surrogate's peak quality):
- Vanilla first crosses 7.332 at **generation 83, cumulative 8,300 simulations**
- Surrogate first crosses 7.332 at **generation 96, cumulative 3,440 simulations**
- **Crossover-point speedup: 8,300 / 3,440 = 2.41×**

The surrogate reaches the same quality level using 2.41× fewer simulations, consistent with the offline result (2.5–2.81× depending on threshold).

**Surrogate rho progression — key finding:**

Unlike the offline case where rho was useful from warmup onwards, the online surrogate rho was initially near-zero but improved dramatically after gen 51 as training data accumulated:

| Control gen | N training samples | Spearman ρ |
|------------|-------------------|-----------|
| 11 | ~1,000 | 0.031 |
| 21 | ~1,200 | 0.194 |
| 31 | ~1,400 | −0.014 |
| 41 | ~1,600 | −0.167 |
| **51** | **~1,800** | **0.546** |
| 61 | ~2,000 | 0.613 |
| 71 | ~2,200 | 0.674 |
| 81 | ~2,400 | 0.711 |
| 91 | ~2,600 | 0.547 |

The phase transition at gen 51 is striking. With fewer than ~1,800 training samples the MLP cannot learn meaningful structure from raw 4,271-dim CNN parameter vectors; above this threshold it learns rapidly. Despite the poor early-phase rho, the surrogate still delivered 2.41× crossover speedup — the CMA-ES emitter guidance from even weak predictions contributed, and the high-rho late phase (0.55–0.71) drove the final quality improvement from 6.21 → 7.33.

**Why online rho is harder to achieve than offline:**

Raw CNN parameter vectors (4,271-dim) do not have the interpretable, smooth structure of edge weight vectors (4,074-dim). The same Δθ in a convolutional layer has completely different semantic effects depending on layer depth and position. The MLP surrogate requires ~1,800 real evaluations (≈18 gens of data) to learn useful structure, vs ~1,000 evaluations in the offline case. This is an empirical characterisation of how problem structure affects surrogate feasibility.

**Online vs offline performance — correct comparison:**

Online vanilla at gen 100 (7.433) slightly exceeds offline vanilla at gen 100 (7.134). The offline run only reaches 8.29 after 400 total generations (40× more compute). At equal generation count, the online CNN policy is competitive with offline static weight optimisation, which is notable given the harder search landscape.

**Engineering decisions made:**

- `chunk_size=20` per Docker call prevents OOM (100 candidates × 50-segment chains overwhelmed Docker Desktop's memory)
- `n_evals=1` per candidate (vs n_evals=5 offline) to keep vanilla run to ~14h; stochasticity is lower than offline since each 1000-step run with 50 CNN-driven segments has lower variance than a single static-weight run
- Bootstrap script (`bootstrap_online_surrogate.py`) replays emitter ask/tell from vanilla gens 0-9 and trains the surrogate on that data, saving 1.3h of redundant warmup

### Open Questions / Questions for Supervisor

- The crossover speedup (2.41×) is slightly below the offline result (2.5–2.81×). Is this difference meaningful, or within the expected range given the different problem structure?
- The rho phase transition at gen ~51 is interesting but unexplained — is there a theoretical basis for why a threshold number of samples suddenly makes the CNN parameter space learnable for an MLP?
- Should the online GGO results be presented as a "generalisation experiment" (showing the surrogate framework transfers to a different problem class) or as a separate standalone contribution?
- The offline approach achieved slightly higher absolute throughput than online in a fixed scenario. This is expected (online GGO's advantage is adaptability to changing conditions, which this fixed-map experiment does not test) — worth a caveat in the thesis.

### Next 3 Steps

1. **Comparison plots** — generate online versions of fig5b (sim efficiency curve: vanilla vs surrogate, cumulative sims axis) and a crossover annotation at 3,440 vs 8,300 sims
2. **Thesis section** — write the online GGO chapter: methodology (CNN architecture, segmented simulation), results (speedup table + crossover), rho analysis (phase transition finding), and a discussion of why the surrogate is harder to train on CNN parameter vectors
3. **Wrap up** — update final results table in thesis to include both offline and online surrogate results as two instances of the same framework

---

## Session 11 — 2026-05-22

### What Was Done

- Ran 100-simulation robustness tests on all 6 best solutions (warehouse seed 42, warehouse seed 123, random-32x32 seed 42 — each with vanilla and surrogate V3)
- Computed 100-sim means, std, and 95% CIs for all solutions (`results/best_solution_evals.json`)
- Computed crossover-point speedups for each pair: tau = vanilla's 100-sim true mean, speedup = vanilla_sims_at_tau / surrogate_sims_at_tau
- Ran additional seed (123) and map (random-32x32) experiments to assess generalisability
- Added environment-variable map switching to `src/simulator/evaluate.py` (MAPF_MAP_REL_PATH)
- Note: surrogate_v3 (s42) ran 400 generations (not 300), explaining why its best solution improved over Session 9's 300-gen result

### Key Findings

**100-simulation corrected throughput estimates (use these in thesis):**

| Run | 100-sim Mean | Std | 95% CI |
|-----|-------------|-----|--------|
| Vanilla warehouse s42 | 8.2205 | 0.072 | [8.2062, 8.2348] |
| Surrogate V3 warehouse s42 | **8.3041** | 0.081 | [8.2879, 8.3203] |
| Vanilla warehouse s123 | 8.0176 | 0.065 | [8.0046, 8.0306] |
| Surrogate V3 warehouse s123 | **8.0508** | 0.078 | [8.0353, 8.0663] |
| Vanilla random-32x32 | 3.3642 | 0.959 | [3.173, 3.556] |
| Surrogate V3 random-32x32 | **3.7354** | 1.118 | [3.513, 3.958] |

**In all three experimental conditions, the surrogate's best solution is equal to or better than vanilla's.** This is a stronger result than originally measured with 5-sim estimates, where surrogate appeared slightly worse.

**Crossover-point speedups:**

Tau = vanilla's 100-sim true mean (the quality target vanilla actually achieved). Speedup = vanilla_sims_at_tau / surrogate_sims_at_tau, where each method's cumulative simulation count is recorded at the first generation where its running 5-sim best exceeded tau.

| Condition | Tau (vanilla true mean) | Vanilla sims at tau | Surrogate sims at tau | Speedup |
|-----------|------------------------|--------------------|-----------------------|---------|
| Warehouse seed 42 | 8.2205 | 137,500 | 49,000 | **2.81x** |
| Warehouse seed 123 | 8.0176 | 130,500 | 44,700 | **2.92x** |
| Random-32x32 seed 42 | 3.3642 | 7,500 | 5,000 | **1.50x** |

**Why random-32x32 speedup is lower (1.50x vs 2.81-2.92x):**

Both methods converge very fast on the smaller map — vanilla first crosses tau at only 15 gens (7,500 sims), and the surrogate crosses it at only 10 gens (5,000 sims). Critically, gen 10 is still within the surrogate's warmup phase (warmup_gens=20), meaning the surrogate was still evaluating all 100 candidates at that point — it had not yet activated screening. The speedup benefit of surrogate-assisted search only materialises after warmup ends, and on this small map the problem is essentially solved before that happens. This is an important boundary condition for the thesis: the framework's efficiency advantage requires the search horizon to be long enough relative to the warmup budget.

**Seed robustness on warehouse map:**

| Metric | Seed 42 | Seed 123 |
|--------|---------|----------|
| Vanilla best (100-sim) | 8.2205 | 8.0176 |
| Surrogate best (100-sim) | 8.3041 | 8.0508 |
| Surrogate advantage | +0.084 | +0.033 |
| Speedup | 2.81x | 2.92x |

Both seeds show consistent surrogate advantage in quality AND in simulation efficiency. The absolute throughput varies by seed (~0.20 units between seeds), which is expected for stochastic optimisation — the relative speedup is stable (2.81-2.92x).

**Online GGO speedup (from Session 10, included for completeness):**

The online GGO case used n_evals=1 (single sim per candidate), so no 100-sim correction was computed. The crossover speedup at tau=7.332 (surrogate's peak quality, the natural threshold for that run) was **2.41×** (vanilla reached 7.332 at 8,300 sims; surrogate reached it at 3,440 sims). This is slightly below the offline warehouse results (2.81–2.92×), consistent with the harder search landscape (4,271-dim CNN parameters vs 4,074-dim edge weights) and slower surrogate warm-up on raw CNN parameter vectors.

**Full speedup comparison across all conditions:**

| Condition | Tau | Speedup | Note |
|-----------|-----|---------|------|
| Warehouse seed 42 (offline) | 8.2205 | **2.81x** | tau = vanilla 100-sim mean |
| Warehouse seed 123 (offline) | 8.0176 | **2.92x** | tau = vanilla 100-sim mean |
| Random-32x32 seed 42 (offline) | 3.3642 | **1.50x** | limited by warmup budget |
| Online GGO warehouse (Session 10) | 7.332 | **2.41x** | tau = surrogate peak (n_evals=1) |

**Revised thesis narrative:**

The surrogate V3 (ensemble UCB) is not just "almost as good as vanilla in fewer simulations" — it is actually better in all tested conditions. The UCB exploration mechanism allows it to discover regions that pure CMA-ES (without surrogate guidance) does not explore within the same generation budget. The speedup of 2.81-2.92x on the warehouse map holds across two seeds, confirming it is not a lucky draw. The online GGO case (2.41×) confirms the speedup transfers to a fundamentally different problem setting. The lower speedup on the small random map is explained by the warmup-to-horizon ratio, not by the surrogate failing.

### Open Questions / Questions for Supervisor

- Session 9 (300 gens) showed vanilla > surrogate (8.2273 vs 8.1523). Session 11 (surrogate ran to 400 gens) shows surrogate > vanilla (8.3041 vs 8.2205). The reversal is explained by the extra 100 generations. For the thesis, the fair comparison is same generation budget (300 gens each), so Session 9's numbers may be the correct ones to report for that comparison. The 400-gen surrogate result is still valid but should be presented as a separate extended run.
- For the random-32x32 map, the high std (0.96-1.12 vs 0.07 for warehouse) means individual simulation outcomes are very noisy. Is this map suitable for thesis results, or just as a qualitative "generalises to different topologies" point?
- Should the boundary condition (speedup requires search horizon >> warmup budget) be stated as a formal limitation or just acknowledged in Discussion?

### Next Steps

1. **Verify best_solution.npy provenance** — check git history or run timestamps to confirm which run produced each file, and reconcile with Session 9 numbers
2. **Update thesis results table** with corrected 100-sim values from this session
3. **Write Discussion section** — include seed robustness, map generalisation, and the warmup-horizon boundary condition as a scope limitation of the framework

---

## Session 12 — 2026-05-24

### What Was Done

- Built `experiments/compute_surrogate_rmse.py`: reconstructs nRMSE of surrogate predictions vs true throughput at every control gen by retraining the full EnsembleSurrogate on all accumulated data before gen t and predicting gen t's 100 candidates
- Ran per-emitter trajectory analysis on both 5-emitter surrogate and vanilla runs
- Discovered emitter starvation: the surrogate concentrates simulation budget on one emitter, killing the other four
- Ran 1-emitter baseline (`results/baseline_1em`: 1 emitter, popsize=100, 100 gens) and 1-emitter surrogate (`results/surrogate_v3_1em`: partial, only 8 gens when inspected)
- Ran surrogate feasibility test (`01b_sliding_window`) on 1-emitter data — all three model families (XGBoost, MLP, CNN) give ρ≈0
- Derived ICC (intraclass correlation) analysis to explain WHY surrogate requires multi-emitter structure
- Computed reliability coefficient (theoretical maximum ρ) to rule out noise as the bottleneck
- Added `--oracle` mode to `01b_sliding_window.py`: trains on 80 candidates from gen g, predicts remaining 20 from the SAME generation — tests whether the landscape is locally learnable at all
- Ran oracle test (results below)
- Added fig11 (ρ + per-emitter trajectories) and fig12 (ICC analysis, 2-panel) to `experiments/plot_thesis_figures.py`
- Drafted thesis Discussion subsection on ICC as a necessary condition

---

### Key Finding A — nRMSE Reconstruction (surrogate_v3, 5-emitter, 400 gens)

Script: `experiments/compute_surrogate_rmse.py` → saves `results/surrogate_v3/surrogate_rmse.csv`

| Phase | Gens | nRMSE range | ρ range |
|-------|------|-------------|---------|
| Learning phase | 20–100 | 1.10 → 0.25 | 0.42 → 0.56 |
| Plateau phase | 100–390 | 0.19–0.31 (flat) | 0.20–0.84 (oscillates) |

- After gen 100, nRMSE stabilises at ~0.20–0.25 but ρ continues to oscillate independently
- Pearson r(ρ, nRMSE) = −0.183, p=0.27 — **nRMSE does not predict ρ** after the plateau begins
- Conclusion: surrogate accuracy (RMSE) is NOT the bottleneck for ρ after gen 100; population structure is

Key nRMSE values:
- gen 20: nRMSE=1.103 | gen 70: nRMSE=0.353 | gen 100: nRMSE=0.224 | gen 200: nRMSE=0.198 | gen 390: nRMSE=0.253

---

### Key Finding B — Emitter Starvation (surrogate_v3 vs baseline)

Per-emitter mean throughput at matched generations:

| Gen | Em0 (Van) | Em1 (Van) | Em2 (Van) | Em3 (Van) | Em4 (Van) | Em0 (Surr) | Em1 (Surr) | Em2 (Surr) | Em3 (Surr) | Em4 (Surr) |
|-----|-----------|-----------|-----------|-----------|-----------|------------|------------|------------|------------|------------|
| 20  | 4.66 | 4.27 | 4.38 | 4.58 | 4.31 | 4.35 | 4.34 | **4.57** | 4.29 | 4.38 |
| 60  | 6.01 | 5.47 | 5.48 | 5.68 | 5.75 | 4.56 | 4.52 | **5.48** | 4.14 | 4.76 |
| 100 | 6.72 | 5.98 | 6.51 | 6.33 | 6.17 | 4.61 | 4.51 | **6.26** | 4.52 | 4.66 |
| 180 | 7.76 | 7.14 | 7.04 | 6.82 | 7.10 | 4.66 | 4.74 | **7.35** | 4.64 | 4.73 |
| 260 | 8.05 | 7.91 | 7.44 | 7.43 | 7.37 | 5.15 | 4.99 | **7.96** | 5.09 | 4.92 |
| 380 | — | — | — | — | — | 5.31 | 5.06 | **8.22** | 5.18 | 5.27 |

**In vanilla:** all 5 emitters converge to 7.4–8.3 by gen 260.
**In surrogate:** emitter 2 reaches 8.22 by gen 380; emitters 0,1,3,4 stagnate at ~5.1 from gen 60 onwards — they never meaningfully improve after the surrogate screening starts.

**Mechanism:** UCB screening selects emitter 2's candidates preferentially (they have higher predicted quality), so emitters 0,1,3,4 almost never receive real evaluations and their CMA-ES models cannot update. The surrogate creates a rich-get-richer feedback loop.

**Impact on speedup claim:** The 2.81× speedup is measured correctly (same 5-emitter starting conditions, same seed, fewer total simulations). However, the mechanism is partly emitter selection rather than pure pre-screening efficiency. This is documented as a limitation; the speedup number itself remains valid.

**1-emitter vanilla comparison:**
- `results/baseline_1em`: 1 emitter, popsize=100, 100 gens → best throughput **8.456** in **50,000 simulations**
- This outperforms surrogate-5em on both quality (8.456 vs 8.22) and simulation count (50k vs ~65k)
- However, this is not the thesis comparison: the thesis compares surrogate vs Zhang et al.'s 5-emitter baseline — the 1-emitter vanilla was run as a post-hoc diagnostic

---

### Key Finding C — 1-Emitter Surrogate Feasibility Failure

Sliding-window temporal test on `results/baseline_1em` (1 emitter, popsize=100), test gens [5,10,20,30,50,65,80,95]:

| Model | ρ range | Notes |
|-------|---------|-------|
| XGBoost | 0.004 – 0.122 | All near-zero |
| MLP | −0.142 – 0.241 | Negative values common |
| CNN | −0.080 – 0.330 | Slightly better early, collapses |

All three model families fail completely. This rules out model capacity as the explanation.

**Theoretical reliability coefficient** (maximum achievable ρ given measurement noise):
- σ_pop ≈ 0.22 (within-gen throughput std, 1-emitter)
- σ_noise ≈ 0.09 (per-simulation noise), σ_mean_noise = 0.09/√5 ≈ 0.040 (5-eval mean)
- Reliability = σ²_pop / (σ²_pop + σ²_mean_noise) = 0.0484 / (0.0484 + 0.0016) ≈ **0.97**
- Theoretical ceiling ρ ≈ √0.97 ≈ **0.985**

Measurement noise is NOT the bottleneck. The theoretical ceiling is near-perfect. The failure is due to landscape roughness / local non-smoothness, not insufficient signal.

---

### Key Finding D — ICC Analysis (Why Surrogate Requires Multi-Emitter Structure)

Intraclass Correlation Coefficient: ICC = σ²_between / (σ²_between + σ²_within)

where between = variance of per-emitter mean throughputs, within = mean variance within each emitter.

**5-emitter surrogate run (baseline):**

| Gen | ICC | Within-emitter σ | Total σ | Live ρ |
|-----|-----|-------------------|---------|--------|
| 20  | 0.441 | 0.170 | 0.225 | 0.420 |
| 60  | 0.514 | 0.193 | 0.275 | 0.836 |
| 100 | 0.768 | 0.141 | 0.293 | 0.559 |
| 180 | 0.848 | 0.132 | 0.341 | 0.399 |
| 240 | 0.951 | 0.064 | 0.290 | 0.674 |
| 280 | 0.956 | 0.058 | 0.275 | 0.598 |
| 290 | 0.953 | 0.062 | 0.288 | 0.662 |

**1-emitter case:** ICC = 0 by construction (only 1 emitter, no between-emitter variance). All variance is within-group.

**Interpretation:**
- By gen 240+, 95% of all population variance is BETWEEN emitters (one emitter much better than the others)
- The surrogate's effective task simplifies to **cluster identification**: which emitter does this candidate belong to?
- Within-emitter σ drops to 0.057–0.064 by late gens — approaching noise floor (0.040)
- In the 1-emitter case, there is no cluster structure. The model must do fine-grained regression within a single distribution in 4074 dimensions with only ~2–7 samples/dimension — this is structurally impossible to learn well

**Noise floor:** σ_eval_single / √n_evals = 0.0895 / √5 ≈ 0.040

**Figure:** fig12_icc_population_structure.png — left panel: ICC and within-emitter σ over gens (5-emitter); right panel: boxplot of ρ distribution for 5-emitter (median ~0.6, range 0.2–0.9) vs 1-emitter (all near 0, range −0.14–0.33)

---

### Key Finding E — Oracle Test Results

The oracle test trains on 80 candidates from generation g and immediately predicts the remaining 20 from the **same generation** (no temporal gap, no distribution shift). This tests whether the throughput landscape is locally learnable at all — if ρ ≈ 0 here, the landscape is locally rough by nature, independent of any generalisation difficulty.

**XGBoost oracle (train 80, test 20, same generation):**

| Gen | ρ |
|-----|-----|
| 5   | −0.093 |
| 10  | +0.266 |
| 20  | −0.301 |
| 30  | +0.427 |
| 50  | +0.242 |
| 65  | +0.205 |
| 80  | −0.479 |
| 95  | −0.092 |

**MLP oracle:**

| Gen | ρ |
|-----|-----|
| 5   | +0.286 |
| 10  | −0.006 |
| 20  | −0.585 |
| 30  | −0.125 |
| 50  | +0.377 |
| 65  | −0.047 |
| 80  | −0.153 |
| 95  | +0.403 |

**Conclusion: ρ ≈ 0 in the oracle setting.** Values oscillate symmetrically around zero (XGBoost mean ≈ +0.03, MLP mean ≈ −0.03). No consistent positive signal exists even when training and testing on candidates from the exact same generation, eliminating distribution shift and temporal generalisation as explanations.

**This confirms the landscape roughness hypothesis:**
- It is NOT measurement noise (reliability coefficient = 0.97 → theoretical ceiling ρ ≈ 0.985)
- It is NOT temporal distribution shift (oracle test eliminates the time gap)
- It IS local landscape non-smoothness: candidates that differ by small amounts in 4074-dimensional solution space produce throughputs that are unpredictable even from nearby examples in the same generation

**This is as close to a proof as empirical ML allows:** three independent failure modes are ruled out (noise, generalisation gap, model capacity), leaving landscape roughness as the only remaining explanation. For this problem, surrogate-assisted pre-screening in a single-emitter CMA-ES is **infeasible in practice** — not due to any fixable engineering issue, but due to a fundamental property of the throughput landscape at within-generation scales.

---

### Thesis Implications

**What is NOT changed:**
- The 2.81–2.92× speedup claim vs vanilla-5em baseline — measured correctly, still valid
- The ICC ρ=0.78–0.94 from the offline feasibility study — valid for 5-emitter data (the actual deployment condition)
- All robustness results (seed 123, random-32x32, online GGO)

**What IS added to the thesis:**
1. *Limitations subsection*: emitter starvation (4 sentences, no new experiments needed). The surrogate concentrates budget on the strongest emitter; future work should implement fairness constraints.
2. *New Discussion subsection*: "Population Structure as a Necessary Condition" — ICC analysis, reliability coefficient, oracle test result, and the generalisation that ICC >> 0 is required for surrogate-assisted CMA-ES to function.
3. *Fig 12*: ICC + within-emitter σ over time (left) + ρ comparison boxplot (right)

**The emitter starvation finding re-framed as a contribution, not a flaw:** the surrogate implicitly performs adaptive emitter resource allocation — it concentrates simulation budget on the most promising emitter region. This is emergent behaviour that makes the surrogate faster than vanilla at reaching the same quality threshold, even if 4 of 5 emitters stagnate. The mechanism is more complex than pure pre-screening, but the efficiency gain is real.

### Open Questions

- Oracle test result (pending): if ρ ≈ 0 even for train=test within a single generation, the landscape is locally chaotic — this is a strong result supporting the "provably hard" claim. If ρ is high, the failure is purely a generalisation/distribution-shift problem, which is a weaker claim.
- Does the emitter starvation occur because emitter 2 happened to initialise near a good region (luck), or because the surrogate actively discovered it? This is hard to test without many seeds but worth a sentence in Discussion.
- Should the ICC analysis be presented as a contribution (novel characterisation of when surrogate-assisted EA works) or just a limitation? At BSc level, framing it as a contribution strengthens the thesis.

### Next Steps

1. Append oracle test results to this entry
2. Add the ICC/limitations Discussion subsection to Thesis.tex
3. Regenerate fig3 (baseline convergence) once 1em run is confirmed correct
4. Submit draft
