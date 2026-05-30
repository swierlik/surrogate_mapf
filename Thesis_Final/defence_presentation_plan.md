# Defence Presentation Plan
**Accelerating Lifelong MAPF Optimization via Uncertainty-Based Surrogate-Assisted Evolution Strategies**
10 minutes | Audience: data scientists, no MAPF background | Format: figure-heavy

---

## Overall Arc (one sentence)

> "Evaluating one solution costs a full multi-agent simulation — we replaced most of those simulations with a cheap ML model, achieved a 2.81x speedup, and then figured out exactly *why* it worked (and when it would not)."

---

## Slide-by-Slide Breakdown

---

### Slide 1 — Hook / Motivation (45 s)

**Title:** "Your optimizer is spending 9.5 hours in a warehouse simulator"

**Visual:** A single striking number in large text:
> **150,000 simulations. 9.5 hours. Per run.**

Or a photo of a robot warehouse (Amazon/Ocado), then that number below it.

**What you say:**
- LMAPF = robots in a warehouse, constantly getting new delivery tasks. Think Amazon fulfillment.
- The state-of-the-art optimizer (GGO, Zhang et al. 2024) uses CMA-ES to learn traffic rules for the warehouse.
- One problem: to score a single candidate solution, you must run a full multi-agent simulation. 5 seeds × 1000 steps each.
- A standard run needs 150k such evaluations. That is 9.5 hours on 8 cores.
- **Punch line:** Can we replace most simulations with a cheap ML model?

---

### Slide 2 — Problem Setup (60 s)

**Title:** "What are we optimizing?"

**Visual:** Figure from thesis showing the warehouse map with edge/flow weights overlaid (or a schematic of the warehouse graph). If you do not have a clean figure, a simple diagram:

```
Warehouse graph G = (V, E)
Each edge e  →  weight wₑ ∈ ℝ
Full vector w ∈ ℝ^4074
Objective: maximise throughput τ(w) [tasks/timestep]
```

**What you say:**
- The warehouse is a directed graph. Every edge gets a scalar flow weight.
- Robots use those weights as soft directional preferences when planning paths.
- The optimizer (multi-emitter CMA-ES, 5 emitters × 20 candidates) searches this 4074-dimensional space.
- Throughput τ = mean tasks completed per timestep. One τ evaluation = run the whole simulation.
- This is the expensive black-box function we want to avoid calling.

---

### Slide 3 — Baseline Performance (45 s)

**Figure:** **Fig 3** — Baseline convergence curve (5 emitters, 300 generations).

**What you say:**
- Here is what the vanilla CMA-ES does: converges smoothly to ~8.2 tasks/step over 300 generations.
- Each generation = 100 real simulations (5 emitters × 20 candidates).
- 300 generations = 30,000 simulation calls per run. Repeated across 5 seeds = 150k total.
- This is our budget baseline. Everything we do must be compared against this.

---

### Slide 4 — The Surrogate Idea (60 s)

**Title:** "Replace 80% of simulations with an MLP"

**Visual:** A clean pipeline diagram (draw it yourself or describe it):

```
Generation t
│
├─ CMA-ES proposes 100 candidates (5 emitters × 20)
│
├─ [SURROGATE] Score all 100 cheaply:
│     score = μ̂(x) + λ·σ̂(x)   ← UCB acquisition
│
├─ Keep top 20 candidates  ← send to real simulator
│
└─ Train surrogate on new (x, τ) pairs
```

**What you say:**
- The surrogate is an ensemble of 5 bootstrapped MLPs, trained on past (solution, throughput) pairs.
- Each generation: propose 100 candidates, score all 100 with the surrogate (microseconds), send only the top 20 to the simulator.
- Screening ratio: 20 real evals instead of 100. That is the 5x reduction target.
- The UCB score = predicted mean + λ × predicted uncertainty. This prevents the optimizer from cheating the surrogate — if a region is unexplored, uncertainty is high and those candidates still get evaluated.
- λ = 1.0 is the final choice (ablation shows both λ=0 and λ=2 are worse).

---

### Slide 5 — Main Result (90 s)

**Figure:** `fig4_sample_efficiency.png`

**What you say:**
- X-axis: cumulative simulation calls. Y-axis: best throughput found so far.
- Vanilla CMA-ES (blue) needs ~137k simulations to first cross the τ=8.23 threshold.
- Surrogate V3 (orange, ensemble + UCB) crosses the same threshold at ~49k simulations.
- That is a **2.81x speedup** — same final quality, 65% fewer real simulations.
- The crossover metric is conservative: it measures the first time each method reaches the same quality level, giving vanilla the benefit of the doubt.
- Wall-clock time drops from 9.5 hours to 3.5 hours (2.71x) because simulation dominates the budget.

---

### Slide 6 — Online Extension (45 s)

**Figure:** **Fig 7** — Online GGO convergence.

**Title:** "Same pipeline, different problem"

**What you say:**
- Zang et al. 2025 extend GGO to an online setting where a CNN policy regenerates edge weights every 20 steps from live traffic. Search space grows to 4,271 dimensions.
- We applied the same surrogate pipeline without modification.
- Result: 10,000 → 3,520 simulations, **2.84x reduction**, 2.41x crossover speedup.
- Shows the approach is not specific to static edge weights — it generalises as long as the population has diversity.

---

### Slide 7 — Why Does It Work? The ICC Finding (90 s)

**Figure:** `fig11_rho_emitter_structure.png`

**Title:** "We got 2.81x — then we asked why"

**What you say:**
- We could have stopped at the speedup number. Instead we asked: *why does this work at all — and under what conditions would it completely fail?*
- That question turned out to be more interesting than the speedup itself.
- The figure has two panels. Start with the top one.
- **Top panel** — per-emitter mean throughput over generations: emitter 2 (red, marked ★) diverges to ~8.2 by generation 150 while emitters 0, 1, 3, 4 all plateau near 5.1. The 5 emitters are exploring very different regions of the search space.
- **Bottom panel** — ICC (intraclass correlation coefficient). ICC measures how much of the population's total variance comes from between-emitter differences vs within-emitter spread.
  - ICC ≈ 1 → emitters are far apart. The surrogate just needs to identify which emitter a candidate belongs to — a coarse, learnable task.
  - ICC ≈ 0 → all emitters cluster together. Any surrogate ranking is noise.
- Orange = ICC in the vanilla run, growing gradually from ~0.4 to ~0.95 as emitters naturally diverge. Red dashed = surrogate run, jumps to ~1.0 by generation 40 because UCB concentrates budget on emitter 2.
- Within-emitter σ (blue) falls to the measurement noise floor — candidates inside one emitter's distribution become almost indistinguishable.

---

### Slide 8 — When It Fails: Diagnostic Tests + Table III (90 s)

**Visual:** Table III (`tab:rho_comparison`) from the thesis — all architectures, sliding-window median ρ.

**Title:** "What happens without emitter divergence?"

**What you say:**
- To stress-test the claim, we ran the surrogate on a single-emitter baseline. ICC = 0 by construction.
- Three diagnostic tests rule out every other explanation:
  - **Capacity:** XGBoost gets ρ = 0.999 when trained and tested on the same generation. The model can memorise. Capacity is not the problem.
  - **Distribution shift:** Oracle mode — train on 80% of the same generation, test on the other 20% — both XGBoost and MLP collapse to ρ ≈ 0. Not the problem.
  - **Noise:** Theory gives a ceiling of ρ ≤ 0.99 given seed variance. Not the problem.
- The only remaining explanation is **landscape roughness**: within one emitter's distribution, candidates are so similar that knowing 80 of them tells you nothing about the other 20.
- We then tested four architectures — MLP, LargeCNN, CellTransformer, GNN — to check whether a richer model could overcome this. Table III: best result is LargeCNN at median ρ = 0.17, far below the ρ > 0.8 generally required for reliable screening. Architecture does not help.
- **Closing point:** the surrogate is not a feasible approach in the single-emitter case. ICC is a necessary condition. The 2.81x speedup exists because multi-emitter CMA-ES happened to create exactly the population structure a surrogate needs.

---

### Slide 9 — Conclusions (45 s)

**Title:** "What we learned"

**Visual:** Four bullet points (no wall of text):

1. **Surrogate pre-screening works: 2.81x fewer simulations, same quality.**
   - Ensemble MLPs + UCB acquisition is the right combination.

2. **ICC is the necessary precondition.**
   - Multi-emitter diversity is what makes ranking meaningful. Check ICC before deploying any surrogate in this setting.

3. **The speedup is primarily emitter selection, not surrogate ranking.**
   - A single-emitter baseline (no surrogate) reaches the same threshold at 41,500 simulations: 3.30x, beating V3's 2.81x. The surrogate's net contribution is implicit budget concentration on the best emitter, confirmed by the ICC analysis.

4. **The open problem is not the surrogate — it is the fitness landscape.**
   - All four architectures fail within a single emitter because the landscape is rough, not because the models are too weak. The productive directions are alternative genotype encodings that expose smoother structure, and feature engineering that encodes warehouse graph topology into the surrogate input. LargeCNN is the starting point if either of those unlocks reliable within-emitter ranking.

**Closing sentence:**
> "The practical take-away: for this problem, a single focused emitter beats the surrogate. The surrogate research direction is closed; the landscape encoding problem is open."

---

## Timing Summary

| Slide | Topic | Time |
|---|---|---|
| 1 | Hook / scale of the problem | 0:45 |
| 2 | Problem setup (what is being optimized) | 1:00 |
| 3 | Baseline performance (Fig 3) | 0:45 |
| 4 | Surrogate pipeline (UCB diagram) | 1:00 |
| 5 | Main result (fig4_sample_efficiency) | 1:30 |
| 6 | Online extension (Fig 7) | 0:45 |
| 7 | Why it works — ICC (fig11_rho_emitter_structure) | 1:30 |
| 8 | When it fails — diagnostic tests + Table III | 1:30 |
| 9 | Conclusions | 0:45 |
| **Total** | | **~9:00** |

Buffer: ~60 seconds for slide transitions and breathing.

---

## Likely Defence Questions and Suggested Answers

**Q: Why not use Gaussian Processes instead of an MLP ensemble?**
> GPs scale as O(n³) in training data. After 200+ generations we have thousands of (w, τ) pairs in 4074 dimensions — GPs become computationally prohibitive. The bootstrapped MLP ensemble gives calibrated uncertainty at a fraction of the cost, and its rank correlation (0.78–0.94) is strong enough for pre-screening.

**Q: How do you know the speedup is real and not just because you are picking the best emitter?**
> It is not. A single-emitter baseline with population size 100 and no surrogate reaches the same threshold at 41,500 simulations, a 3.30x reduction, which beats V3's 2.81x. The surrogate's net effect is implicit emitter concentration via UCB starvation, not ranking quality within an emitter. The ICC analysis explains why: once emitters diverge, coarse region identification is enough, and a simple focused search replicates that without the ML overhead.

**Q: What happens if ICC is low?**
> Then per-emitter ρ collapses toward zero and the surrogate cannot rank candidates reliably. The screening step would actively hurt performance by discarding good candidates at random. The ICC figure gives you a generation-by-generation warning sign.

**Q: Would this work in domains other than LMAPF?**
> The pipeline is general: multi-emitter CMA-ES + ensemble MLP + UCB. The only domain-specific assumption is that fitness evaluations are expensive and the population has multi-modal structure (high ICC). Any black-box optimization problem with those two properties is a candidate.

**Q: Why λ=1.0 specifically?**
> Ablation study (Section VI.B). λ=0 (pure exploitation) degrades because the surrogate is over-trusted as distributions shift. λ=2.0 over-explores and wastes budget on uncertain but poor candidates. λ=1.0 balances the two; it also has a natural interpretation as a one-standard-deviation UCB bound.

**Q: Is λ=0 vs λ=1 really a meaningful difference if emitter starvation means the dominant emitter gets most of the budget anyway?**
> Partially fair. With extreme starvation, most of the top-20 slots go to emitter 2 regardless of λ, so cross-emitter selection is not the main mechanism. The real risk of λ=0 is that the 80 candidates that are not simulated still receive surrogate-predicted fitness values that feed into the CMA-ES distribution update. As emitter 2's distribution shifts into new regions, the surrogate has no data there and overestimates fitness — those inflated placeholder values bias the update. UCB (λ=1) counters this by giving high-uncertainty frontier candidates a score bonus, ensuring some get real simulation and the placeholder predictions stay honest. The +1.4% quality gain (8.04 vs 8.15, 100-simulation corrected means) is modest but real and consistent with this mechanism.

**Q: Is ICC really a necessary condition, or just an empirical finding with the models you tried?**
> It is strong empirical evidence, not a mathematical proof. The three diagnostic tests rule out distribution shift and measurement noise as explanations, leaving landscape roughness as the cause of within-emitter ρ ≈ 0. We subsequently extended the test to four architectures: MLP, LargeCNN (a five-layer spatial CNN), CellTransformer (per-cell attention over all 948 warehouse cells), and a GAT-based graph neural network operating directly on the warehouse topology. All four were evaluated in both oracle mode (train on 80% of the same generation, test on the remaining 20%) and sliding-window mode (train on all past data, test on the next generation). In oracle mode every architecture produced ρ ≈ 0. In sliding-window mode the best result was LargeCNN at median ρ = 0.17, well below the ρ > 0.8 generally required for reliable candidate screening. The finding therefore holds across all model classes tested, including a domain-aware GNN. LargeCNN is the most practical architecture for follow-up work given its training time; CellTransformer and the GNN require substantially more compute per generation.

---

## Presentation Tips

- **Do not define LMAPF in formal terms** — say "robots in a warehouse with continuous delivery tasks." The audience is data scientists, not roboticists.
- **Anchor every abstract claim to a number**: "9.5 hours," "2.81x," "49k vs 137k." Numbers beat words.
- **Slide 7 (ICC) is the hardest** — practice the one-paragraph ICC explanation until it is conversational. The ICC analogy: "imagine 5 groups of students taking the same test — ICC measures how much group membership predicts your score."
- **Figures 1 and 11 are your two key visuals.** If pressed for time, cut slides 8–9 before cutting these.
- **Avoid the word 'obviously'** — nothing is obvious to this audience.
- Bring a printed copy of Table II in case slides fail.
