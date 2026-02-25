# Methodologies

**How we interpret system data — and what we've learned about which approaches yield precise results.**

This document captures the methodological principles underlying Prime's interpretation layer. These aren't theoretical preferences — they emerged from empirical validation on real datasets (C-MAPSS FD001–FD004), where we could directly measure which approaches improved prediction accuracy and which degraded it.

---

## The Core Claim

Systems lose coherence before they fail. The relationships between signals change structure before any individual signal crosses a threshold. Detecting this geometric precursor — rather than waiting for threshold violations — is the foundation of everything Prime does.

This is not a metaphor. On FD001 (100 turbofan engines, 24 sensors), geometric features combined with sensor data achieved Test RMSE 11.72 and NASA Score 188, beating the published AGCNN deep learning benchmark (RMSE 12.4, NASA 226) by 5.5% and 17% respectively. The geometry doesn't replace the sensor data — it reveals structure the sensors alone cannot express.

| Approach | Test RMSE | NASA Score | What It Tells Us |
|---|---|---|---|
| Sensor features only | 14.5 | 355 | Raw signals miss relational dynamics |
| Geometry features only | 75.6 | 138,972 | Geometry alone overfits catastrophically |
| Combined (LGB Asym α=1.6) | 11.72 | 188 | Geometry + sensors = precise prediction |
| Published AGCNN benchmark | 12.4 | 226 | Black-box deep learning, no interpretability |

The lesson: geometry is a force multiplier for sensor data, not a replacement. And the specific geometry feature that generalized — RT centroid distance (feature importance rank #2 at 697) — measures how far the system's current state sits from its reference trajectory. Every other windowed geometry feature overfit in isolation. Knowing which geometric measures transfer and which don't is itself a methodological finding.

---

## Methodologies That Increase Precision

### 1. Separate Computation from Interpretation

This is the single most consequential architectural decision in the framework, and it has direct methodological implications for anyone analyzing system data.

**The principle:** Manifold computes. Prime interprets. The compute layer never classifies, labels, or judges. It produces eigenvalues, derivatives, entropies, and covariance structures. Prime then decides what those numbers mean in context.

**Why it matters for precision:** The moment you let interpretation leak into computation, you bake assumptions into your math. We discovered this concretely when typology labels (`trajectory_type`, `stability_class`, `is_chaotic`) were creeping into Manifold. Those labels carry domain assumptions — "chaotic" means something different for a turbine bearing versus a chemical reaction. Removing them from the compute layer meant the same eigendecomposition serves both use cases without hidden bias.

**The practical rule:** If you can't point to a specific equation justifying a number in your output, it's interpretation masquerading as computation. Move it upstream.

### 2. Enforce Minimum Data Requirements

Most analysis tools will happily compute a Lyapunov exponent from 50 data points and hand you a number that looks precise but is mathematically meaningless. Prime refuses to do this.

Every engine in the framework declares its minimum data requirement:

| Engine | Minimum Observations | Rationale |
|---|---|---|
| Lyapunov | 10,000 | Attractor reconstruction (Wolf et al., 1985) |
| Hurst | 256 | R/S rescaling statistics (Mandelbrot & Wallis, 1969) |
| Recurrence | 500 | Sufficient recurrence density |
| Entropy | 100 | Symbol sequence statistics |
| GARCH | 250 | Volatility clustering detection |
| Attractor | 1,000 | Strange attractor reconstruction |

**The methodology:** Before computing anything, check whether you have enough data for the result to be reliable. Telling a user "insufficient data for this analysis" is more precise than giving them a confident wrong answer. This is a hard requirement, not a suggestion — `00_config.sql` encodes these as filterable thresholds.

### 3. Require Multi-Pillar Agreement

No single metric should drive a conclusion. Prime uses a convergent evidence model where confidence scales with the number of independent analytical pillars that agree:

| Pillars Agreeing | Confidence | Recommendation |
|---|---|---|
| 1 of 4 | 25% — Low | Flag for monitoring |
| 2 of 4 | 50% — Moderate | Investigate |
| 3 of 4 | 75% — High | Act with confidence |
| 4 of 4 | 95% — Very High | Strong conclusion |

The pillars are structurally different analyses — geometry (eigenvalue trajectories), dynamics (derivatives and collapse detection), thermodynamics (entropy evolution), and signal-level behavior (canary identification). Each can be individually fooled by noise or edge cases. Requiring convergence across methods that fail in different ways dramatically reduces false positives.

**Empirical validation:** The canary signal analysis uses exactly this approach — three independent methods (velocity acceleration, collapse correlation, single-signal RUL prediction) each identify which sensor drives failure. The consensus ranking across all three is far more reliable than any single method.

### 4. Track Derivatives, Not Just Values

A system at coherence ratio 0.6 that's been stable for months is fundamentally different from a system at 0.6 that was at 0.9 last week. Absolute values are snapshots. Derivatives are trajectories.

Prime computes the full derivative chain for key metrics:

- **Velocity** — rate of change (first derivative)
- **Acceleration** — rate of change of rate of change (second derivative)
- **Jerk** — third derivative (signals approaching discontinuities)
- **Curvature** — how sharply the trajectory is bending

The coherence velocity analysis classifies system state not by where coherence *is* but by how fast it's *changing*:

| |Δ coherence| | Status |
|---|---|
| ≥ 0.10 | ALARM — Rapid transition |
| ≥ 0.05 | WARNING — Fast change |
| ≥ 0.02 | WATCH — Drifting |
| < 0.02 | STABLE |

This methodology — privileging trajectory over position — is why the framework can detect dimensional collapse early. The eigenvalue spectrum starts rotating before it collapses, and acceleration (second derivative) catches the rotation before velocity (first derivative) confirms it.

### 5. Classify Before You Compute

Not every analysis is appropriate for every signal. A monotonic trend and a chaotic oscillation need fundamentally different mathematical treatment. Computing everything on everything wastes resources and, worse, produces misleading results (a Lyapunov exponent on a linear trend is nonsense, not insight).

Prime's typology chain — `signal_vector → gaussian_fingerprint → gaussian_similarity` — exists to match signals to appropriate engines before computation begins. The manifest system then tells Manifold exactly which engines to run on which signals, preventing wasted computation and garbage-in results.

**The methodology:** Characterize your data's structure first. Then apply the mathematics that structure warrants. This is the opposite of the "compute 500 features and let the ML model sort it out" approach, which works until it doesn't (see: geometry-only model overfitting to RMSE 75.6).

### 6. Geometry-Neutral Labels

Prime's state classification uses geometry-neutral language deliberately:

| State | Meaning |
|---|---|
| `BASELINE_STABLE` | Eigenstructure matches reference, stable |
| `TRANSITIONING` | Eigenstructure rotating or dimensions collapsing |
| `SHIFTED_STABLE` | Stable eigenstructure at a different operating point |
| `INDETERMINATE` | Cannot reliably classify |

There is no `HEALTHY` or `FAILED` label. A bearing losing coherence is failure. A plastic film losing coherence during curing is the experiment succeeding. Same eigenvalue trajectory, opposite interpretation. The geometry engine doesn't know the difference, and it shouldn't — that's the user's domain knowledge to supply.

**The methodology:** Measure geometry. Report geometry. Let the domain expert attach meaning. This is how the same framework serves turbofan degradation, chemical process monitoring, and research signal discovery without modification.

---

## Methodologies That Reduce Precision

These are anti-patterns we've validated empirically — approaches that seem reasonable but degrade results.

### 1. Normalizing Without Understanding What You Destroy

Z-score normalization makes total variance constant across windows. That sounds like responsible preprocessing — until you realize that *changes in total variance* are exactly what eigenvalue trajectory analysis needs to detect dimensional collapse.

**The finding:** Normalizing metrics like coefficient of variation and range ratio via z-score can eliminate the very dynamics the geometry engine is designed to track. Eigenvalue spikes that look like bugs may be real physics — a system suddenly coupling or decoupling its modes produces exactly this signature.

**The methodology:** Before normalizing anything, ask: "What information does this transformation destroy?" Use distribution-aware normalization — check kurtosis, skewness, and outlier fraction to select the appropriate method (z-score, robust, MAD) rather than defaulting to z-score for everything. The `recommend_normalization` primitive in pmtvs does exactly this.

### 2. Post-Hoc Feature Importance Instead of Structural Causality

SHAP values telling you "sensor 7 is important" after training a model is correlation, not causation. It tells you what the model learned to use, not what's actually driving system degradation.

**The alternative:** Prime's canary signal analysis identifies structural causality through three independent methods:

1. **Velocity acceleration** — which signal's rate of change increases most in late life?
2. **Collapse correlation** — which signal's features track eigenvalue trajectory changes?
3. **Single-signal RUL prediction** — which signal alone predicts lifecycle length?

When all three converge on the same sensor, you have structural evidence that the signal is driving the degradation pattern, not just correlated with it. This is the difference between "the model thinks this matters" and "this is physically causing the system to lose coherence."

### 3. Computing Everything Because You Can

Running 281 signal analysis functions on every signal in a dataset produces a wall of numbers, most of which are noise. The geometry-only model on FD001 — 40 features, all geometric — achieved Test RMSE 75.6. That's not just bad; it's *catastrophically* overfit (CV was 2.3, meaning the geometry features memorized the training set perfectly and generalized not at all).

**The methodology:** Feature selection is not optional. The combined model used 214 features and achieved Test RMSE 11.72 with a negative CV-test gap (test was *better* than cross-validation). The difference: typology-driven feature selection ensured only features appropriate to each signal's structure were included. More features ≠ more precision. The right features = more precision.

### 4. Trusting Speed Over Correctness

During development, we discovered that "fast" computation runs were returning incorrect results due to broken mathematical implementations, while slower runs using validated primitives produced correct output. A fast wrong answer delivered with confidence is worse than no answer at all.

**The methodology:** Every calculation must trace back to a specific function with a specific mathematical justification. If you can't audit the path from raw data to output number, you don't have a result — you have a hope. This is why pmtvs exposes 281 individually testable primitives rather than one monolithic "analyze" function: each primitive can be validated in isolation against known-good test cases.

### 5. Assuming One Failure Mode

FD004 (the hardest C-MAPSS subset) contains two distinct fault modes. Geometry features that dramatically improved FD001–FD003 performance actually *hurt* FD004 because the eigenvalue signatures of the two fault modes overlap in ways that confuse a model trained on mixed data.

| Dataset | Fault Modes | Regimes | Geometry Impact |
|---|---|---|---|
| FD001 | 1 | 1 | Helps significantly |
| FD002 | 1 | 6 | Helps |
| FD003 | 2 | 1 | Helps |
| FD004 | 2 | 6 | Hurts (mode heterogeneity) |

**The methodology:** Before combining feature types, understand the structure of your failure population. Covariance deconvolution — separating observed covariance into within-regime and between-regime components (`C_clean = C_observed - C_between_regime`) — is one approach to handling this. But the deeper lesson is: geometry is not universally helpful. It's helpful when the system's failure modes have distinct geometric signatures. When they don't, forcing geometry into the model adds noise, not signal.

---

## The Meta-Methodology: Transparent Geometric Interpretation

Every methodology above serves a single principle: **the user should be able to trace any result back to the math that produced it, understand what assumptions were made, and know when the framework doesn't have enough information to give a reliable answer.**

This is the opposite of black-box AI, where a model ingests data and emits a prediction with no explanation of how it got there. Both approaches can achieve similar accuracy on benchmarks. The difference emerges when something goes wrong — when the prediction is wrong, when the data distribution shifts, when the user needs to defend their analysis to a thesis committee or a plant manager.

The "2am grad student" test: if a student can upload a parquet file, get results, and defend every number in front of their committee by pointing to specific mathematical operations on specific data, the methodology is working. If they get numbers they can't trace or explain, it's not — no matter how accurate those numbers happen to be on the test set.

---

## Summary of Methodological Principles

| Principle | Increases Precision | Decreases Precision |
|---|---|---|
| **Architecture** | Separate compute from interpretation | Let classification leak into math |
| **Data sufficiency** | Enforce minimum observation counts | Compute from insufficient data |
| **Evidence** | Require multi-pillar convergence | Trust single-metric conclusions |
| **Dynamics** | Track velocity, acceleration, curvature | Monitor absolute values only |
| **Feature selection** | Classify signals, then compute | Compute everything, hope ML sorts it out |
| **Labels** | Geometry-neutral, domain-agnostic | Bake domain assumptions into engine output |
| **Normalization** | Distribution-aware, preserves dynamics | Default z-score without checking what's lost |
| **Causality** | Structural causality via independent methods | Post-hoc feature importance |
| **Correctness** | Auditable primitives, traceable results | Fast but unvalidated computation |
| **Failure modes** | Characterize population structure first | Assume homogeneous failure population |

---

*These methodologies are living documentation. As the framework encounters new domains and new failure modes, the principles that hold will be reinforced and the ones that don't will be revised. The commitment is to transparency: when we change our mind, we document why.*
