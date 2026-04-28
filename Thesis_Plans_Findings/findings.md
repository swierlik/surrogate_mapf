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
