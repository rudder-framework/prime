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

**The principle:** Manifold engines compute. Prime interprets. The compute layer never classifies, labels, or judges. It produces eigenvalues, derivatives, entropies, and covariance structures. Prime then decides what those numbers mean in context.

**Why it matters for precision:** The moment you let interpretation leak into computation, you bake assumptions into your math. We discovered this concretely when typology labels (`trajectory_type`, `stability_class`, `is_chaotic`) were creeping into the Manifold layer. Those labels carry domain assumptions — "chaotic" means something different for a turbine bearing versus a chemical reaction. Removing them from the compute layer meant the same eigendecomposition serves both use cases without hidden bias.

**The practical rule:** If you can't point to a specific equation justifying a number in your output, it's interpretation masquerading as computation. Move it upstream.

**The canonical architecture:** Manifold engines are dumb muscle — they output numbers. SQL/DuckDB is the brain — it handles all layers, joins, windows, aggregations, and classifications. Orchestrators are pure orchestrators with no inline compute. All computation lives in engines and modules; orchestrators only call, sequence, and pass data.

### 2. Classify Before You Compute (Signal Typology)

Not every analysis is appropriate for every signal. A monotonic trend and a chaotic oscillation need fundamentally different mathematical treatment. Computing everything on everything wastes resources and, worse, produces misleading results (a Lyapunov exponent on a linear trend is nonsense, not insight).

Prime's typology chain — `signal_vector → gaussian_fingerprint → gaussian_similarity` — exists to match signals to appropriate engines before computation begins. This chain is the core of Signal Typology and is a hard requirement, not optional. The manifest system then tells Manifold exactly which engines to run on which signals, preventing wasted computation and garbage-in results.

**The methodology:** Characterize your data's structure first. Then apply the mathematics that structure warrants. This is the opposite of the "compute 500 features and let the ML model sort it out" approach, which works until it doesn't (see: geometry-only model overfitting to RMSE 75.6).

### 3. Cryptanalytic Key Extraction (Known-Plaintext Analysis)

This methodology treats signal classification as a decryption problem and represents a novel contribution to the intersection of cryptanalysis and prognostics. No existing literature applies known-plaintext attack methodology to sensor data for degradation analysis or failure mode discovery.

**The Turing parallel.** Turing didn't reverse-engineer the Enigma rotors. He exploited cribs — known plaintext fragments like weather reports and standard military headers. Prime does the same: it exploits known outcomes (engines that failed early vs. engines that ran normal) to extract the key that distinguishes them, without requiring a full model of the degradation physics.

**The framework:**

- **Ciphertext** = raw sensor streams from all engines in the population
- **Plaintext** = known outcome labels (early-fail vs. normal)
- **The key** = the sensor-time combination where the two populations diverge

**Key extraction techniques, mapped from cryptanalysis:**

- **Differential analysis** (Biham & Shamir): Compute group mean differences across all sensors and lifecycle bins. The divergence pattern IS the key. Where the difference is zero, that sensor tells you nothing. Where it diverges, the failure signature lives.
- **Frequency analysis** (classical cryptanalysis): KL divergence and mutual information identify which sensor-time combinations carry key-bearing information.
- **XOR-analog subtraction**: Direct residual between mean fail and mean normal profiles. Subtracting plaintext from ciphertext yields the key.

**Multi-key hypothesis — confirmed empirically.** A single key is unlikely to unlock all doors. Multiple failure modes require multiple keys with partial coverage. Each key is a different wedge into the cipher. The number of keys and their coverage patterns reveal domain structure.

**Residual key extraction.** After Key 1 classifies the population, the engines Key 1 misses become Key 2's training data. This process continues until the residual population is too small or too homogeneous for further extraction.

**Validation results across C-MAPSS:**

| Dataset | Fault Modes | Ops | Keys Found | Key 2 AUC | First Useful | Interpretation |
|---|---|---|---|---|---|---|
| FD001 | 1 (HPC) | 1 | 1 | — | Never | Single mode, weak key — no encryption |
| FD002 | 1 (HPC) | 6 | 2 | 0.941 | 75% lifecycle | Second key is ops-confounded, not real fault mode |
| FD003 | 2 (HPC + Fan) | 1 | 2 | 1.000 | 5% lifecycle | Perfect separation from first observation window |
| FD004 | 2 (HPC + Fan) | 6 | 2 | 0.986 | 35% lifecycle | Near-perfect, delayed by regime noise |

**What the 2×2 matrix proves:**

- **Fault mode count determines key count.** Datasets with two fault modes produce two keys. Datasets with one fault mode produce one (or a confounded second).
- **Operating condition complexity determines detection timing.** FD003 (2 modes, 1 op) detects at 5%. FD004 (2 modes, 6 ops) detects at 35%. Regime noise delays but does not prevent key extraction.
- **The "first useful" timestamp discriminates real keys from confounded ones.** Real fault-mode keys activate early (5%, 35%). Regime-confounded keys activate late (75%). The framework doesn't just find keys — it tells you which ones are real.
- **The framework discovers domain structure without being told it exists.** Nobody told the system there were two fault modes in FD004. It found them by extracting keys from the residual population.

**Key extraction operates on raw sensor data.** Raw signals are the ciphertext. Every single cycle gives 24 clean sensor readings — no windowing, no aggregation, no statistical stability question required. Cycle 15 of engine 47, sensor 7 reads 553.2 — that's a fact, not an estimate. Keys extracted from geometry would require long windows to achieve accuracy, making them reliable only late in the lifecycle when they're no longer needed. The cryptanalytic framework operates at the source.

**Geometry validates downstream.** Once enough cycles accumulate for a stable fingerprint, the geometry confirms what the raw keys detected early. Geometry is verification, not the source of key extraction.

**Where cryptanalytic keys fit in the ML pipeline:** The keys solve a classification problem (which fault mode is this engine?). RUL regression requires both fault-mode identity AND degradation trajectory. Current keys know the former; they don't directly encode the latter. The actionable extension is trajectory-based key extraction: instead of labeling engines by final outcome (binary: short-lived vs. long-lived), compare RUL-correlation at each lifecycle bin across all sensors. The key becomes the sensor-time combination where eventual RUL first becomes predictable — degradation-rate divergence rather than fault-mode identity.

**Connection to active research:** At CRYPTO 2019, Gohr demonstrated that neural networks could build differential distinguishers surpassing traditional cryptanalysis methods on the NSA block cipher SPECK32/64. A key open problem in that work is interpretability — the neural distinguisher is a black box. Prime's cryptanalytic framework addresses this directly: the keys are interpretable, traceable to specific sensors and lifecycle windows, and their count reveals structural properties of the domain. This represents a complementary approach where interpretability is the primary advantage over black-box neural distinguishers.

**Publication novelty:** No existing literature applies known-plaintext attack methodology to sensor data for failure mode discovery. No existing framework uses residual key extraction to automatically determine the number of distinct failure mechanisms in a domain. No existing approach uses the detection-timing of extracted keys to discriminate between structural and confounded population differences.

### 4. Enforce Minimum Data Requirements

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

### 5. Require Multi-Pillar Agreement

No single metric should drive a conclusion. Prime uses a convergent evidence model where confidence scales with the number of independent analytical pillars that agree:

| Pillars Agreeing | Confidence | Recommendation |
|---|---|---|
| 1 of 4 | 25% — Low | Flag for monitoring |
| 2 of 4 | 50% — Moderate | Investigate |
| 3 of 4 | 75% — High | Act with confidence |
| 4 of 4 | 95% — Very High | Strong conclusion |

The pillars are structurally different analyses — geometry (eigenvalue trajectories), dynamics (derivatives and collapse detection), thermodynamics (entropy evolution), and signal-level behavior (canary identification). Each can be individually fooled by noise or edge cases. Requiring convergence across methods that fail in different ways dramatically reduces false positives.

**Empirical validation:** The canary signal analysis uses exactly this approach — three independent methods (velocity acceleration, collapse correlation, single-signal RUL prediction) each identify which sensor drives failure. The consensus ranking across all three is far more reliable than any single method.

### 6. Track Derivatives, Not Just Values

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

### 7. Geometry-Neutral Labels

Prime's state classification uses geometry-neutral language deliberately:

| State | Meaning |
|---|---|
| `BASELINE_STABLE` | Eigenstructure matches reference, stable |
| `TRANSITIONING` | Eigenstructure rotating or dimensions collapsing |
| `SHIFTED_STABLE` | Stable eigenstructure at a different operating point |
| `INDETERMINATE` | Cannot reliably classify |

There is no `HEALTHY` or `FAILED` label. A bearing losing coherence is failure. A plastic film losing coherence during curing is the experiment succeeding. Same eigenvalue trajectory, opposite interpretation. The geometry engine doesn't know the difference, and it shouldn't — that's the user's domain knowledge to supply.

**The methodology:** Measure geometry. Report geometry. Let the domain expert attach meaning. This is how the same framework serves turbofan degradation, chemical process monitoring, and research signal discovery without modification.

**Naming convention:** In Prime's vocabulary, "cohort" refers to user-defined groups (engines, patients, regions). "Cluster" refers to groups discovered by Prime's geometry. Users define cohorts. Prime discovers clusters. This distinction is enforced throughout the codebase and documentation.

### 8. The Four-Layer Architecture

Prime's canonical architecture flows through four layers, each building on the previous:

1. **Signal Typology** — What IS this signal? Raw characterization, fingerprinting, and now cryptanalytic key extraction. Operates on raw data, no windowing required.
2. **Behavioral Geometry** — How does this signal relate to others? Eigenvalue decomposition, manifold structure, reference trajectory geometry. Requires sufficient data windows.
3. **Dynamical Systems** — How is the system evolving? Lyapunov exponents, attractor reconstruction, stability analysis. Requires substantial observation history.
4. **Causal Mechanics** — What's causing what? Transfer entropy, Granger causality, causal network reconstruction. Requires the most data and validated lower layers.

Each layer gates the next. You cannot meaningfully compute geometry without typology. You cannot meaningfully assess dynamics without geometry. Skipping layers produces numbers without meaning.

**The insight no existing framework captures:** Dynamical systems researchers start at Layer 2 or 3. ML researchers skip straight to feature extraction. Nobody else begins at "what is this signal" and builds the entire pipeline from that determination. This foundational step — which the relevant experts are trained to skip — is the actual source of Prime's precision.

---

## Methodologies That Reduce Precision

These are anti-patterns we've validated empirically — approaches that seem reasonable but degrade results.

### 1. Normalizing Without Understanding What You Destroy

Z-score normalization makes total variance constant across windows. That sounds like responsible preprocessing — until you realize that *changes in total variance* are exactly what eigenvalue trajectory analysis needs to detect dimensional collapse.

**The finding:** Normalizing metrics like coefficient of variation and range ratio via z-score can eliminate the very dynamics the geometry engine is designed to track. Eigenvalue spikes that look like bugs may be real physics — a system suddenly coupling or decoupling its modes produces exactly this signature.

**The methodology:** Before normalizing anything, ask: "What information does this transformation destroy?" Use distribution-aware normalization — check kurtosis, skewness, and outlier fraction to select the appropriate method (z-score, robust, MAD) rather than defaulting to z-score for everything. The `recommend_normalization` primitive does exactly this.

### 2. Post-Hoc Feature Importance Instead of Structural Causality

SHAP values telling you "sensor 7 is important" after training a model is correlation, not causation. It tells you what the model learned to use, not what's actually driving system degradation.

**The alternative:** Prime's canary signal analysis identifies structural causality through three independent methods:

1. **Velocity acceleration** — which signal's rate of change increases most in late life?
2. **Collapse correlation** — which signal's features track eigenvalue trajectory changes?
3. **Single-signal RUL prediction** — which signal alone predicts lifecycle length?

When all three converge on the same sensor, you have structural evidence that the signal is driving the degradation pattern, not just correlated with it. This is the difference between "the model thinks this matters" and "this is physically causing the system to lose coherence."

**The cryptanalytic parallel:** Key extraction provides a fourth independent method. The sensors that carry key information (high KL divergence, high mutual information between sensor value and outcome label) should converge with the canary analysis. When they do, you have four-pillar confirmation. When they don't, you've discovered that the failure mechanism involves interactions the single-signal methods can't capture.

### 3. Computing Everything Because You Can

Running all available signal analysis functions on every signal in a dataset produces a wall of numbers, most of which are noise. The geometry-only model on FD001 — 40 features, all geometric — achieved Test RMSE 75.6. That's not just bad; it's *catastrophically* overfit (CV was 2.3, meaning the geometry features memorized the training set perfectly and generalized not at all).

**The methodology:** Feature selection is not optional. The combined model used 214 features and achieved Test RMSE 11.72 with a negative CV-test gap (test was *better* than cross-validation). The difference: typology-driven feature selection ensured only features appropriate to each signal's structure were included. More features ≠ more precision. The right features = more precision.

### 4. Trusting Speed Over Correctness

During development, we discovered that "fast" computation runs were returning incorrect results due to broken mathematical implementations, while slower runs using validated primitives produced correct output. A fast wrong answer delivered with confidence is worse than no answer at all.

**The methodology:** Every calculation must trace back to a specific function with a specific mathematical justification. If you can't audit the path from raw data to output number, you don't have a result — you have a hope. This is why the framework exposes individually testable primitives rather than one monolithic "analyze" function: each primitive can be validated in isolation against known-good test cases.

### 5. Assuming One Failure Mode

FD004 (the hardest C-MAPSS subset) contains two distinct fault modes. Geometry features that dramatically improved FD001–FD003 performance actually *hurt* FD004 because the eigenvalue signatures of the two fault modes overlap in ways that confuse a model trained on mixed data.

| Dataset | Fault Modes | Regimes | Geometry Impact |
|---|---|---|---|
| FD001 | 1 | 1 | Helps significantly |
| FD002 | 1 | 6 | Helps |
| FD003 | 2 | 1 | Helps |
| FD004 | 2 | 6 | Hurts (mode heterogeneity) |

**The methodology:** Before combining feature types, understand the structure of your failure population. Covariance deconvolution — separating observed covariance into within-regime and between-regime components (`C_clean = C_observed - C_between_regime`) — is one approach to handling this. But the deeper lesson is: geometry is not universally helpful. It's helpful when the system's failure modes have distinct geometric signatures. When they don't, forcing geometry into the model adds noise, not signal.

**The cryptanalytic solution:** Key extraction automatically discovers the number of fault modes and identifies which engines belong to which mode. Once engines are assigned to mode-specific cohorts via keyring features, mode-specific geometry and RUL models can be trained within each cohort. This transforms the mixed-population problem into multiple clean single-population problems.

### 6. Extracting Keys from Geometry Instead of Raw Data

Geometry features (Gaussian fingerprints, eigenvalue trajectories, centroid distances) require windowed data to achieve statistical stability. A fingerprint computed from 20–30 cycles is mathematically shaky. This means geometric keys would only be reliable late in the lifecycle — exactly when they're no longer needed.

**The methodology:** Extract cryptanalytic keys from raw sensor data only. Every cycle gives 24 clean sensor readings with no windowing, no aggregation, no statistical stability question. Raw signals are the ciphertext. Geometry validates the keys downstream once enough data accumulates for stable computation. The decryption framework operates at the source, not on intermediate computations.

---

## The Meta-Methodology: Transparent Geometric Interpretation

Every methodology above serves a single principle: **the user should be able to trace any result back to the math that produced it, understand what assumptions were made, and know when the framework doesn't have enough information to give a reliable answer.**

This is the opposite of black-box AI, where a model ingests data and emits a prediction with no explanation of how it got there. Both approaches can achieve similar accuracy on benchmarks. The difference emerges when something goes wrong — when the prediction is wrong, when the data distribution shifts, when the user needs to defend their analysis to a thesis committee or a plant manager.

The "2am grad student" test: if a student can upload a parquet file, get results, and defend every number in front of their committee by pointing to specific mathematical operations on specific data, the methodology is working. If they get numbers they can't trace or explain, it's not — no matter how accurate those numbers happen to be on the test set.

---

## Summary of Methodological Principles

| Principle | Increases Precision | Decreases Precision |
|---|---|---|
| **Architecture** | Manifold computes, Prime interprets | Let classification leak into math |
| **Data sufficiency** | Enforce minimum observation counts | Compute from insufficient data |
| **Evidence** | Require multi-pillar convergence | Trust single-metric conclusions |
| **Dynamics** | Track velocity, acceleration, curvature | Monitor absolute values only |
| **Typology** | Classify signals, then compute | Compute everything, hope ML sorts it out |
| **Labels** | Geometry-neutral, domain-agnostic | Bake domain assumptions into engine output |
| **Normalization** | Distribution-aware, preserves dynamics | Default z-score without checking what's lost |
| **Causality** | Structural causality via independent methods | Post-hoc feature importance |
| **Correctness** | Auditable primitives, traceable results | Fast but unvalidated computation |
| **Failure modes** | Discover via cryptanalytic key extraction | Assume homogeneous failure population |
| **Key source** | Raw sensor data (immediate, no windowing) | Geometry data (requires long windows, too late) |
| **Domain agnosticism** | Same apparatus, different keys per domain | Hardcode domain assumptions into pipeline |

---

## Current Benchmark Results

Best validated results across C-MAPSS datasets:

| Dataset | Best RMSE | Best NASA | Configuration |
|---|---|---|---|
| FD001 | 11.72 | 188 | Combined sensor + geometry, LGB Asym α=1.6 |
| FD004 | 14.29 | 1,005 | G_train_NASA, normalized + feature-selected |

FD004 target: NASA < 1,000. Current gap: 5 points. Active work: trajectory-based cryptanalytic key extraction targeting degradation-rate divergence rather than fault-mode identity.

---

*These methodologies are living documentation. As the framework encounters new domains and new failure modes, the principles that hold will be reinforced and the ones that don't will be revised. The commitment is to transparency: when we change our mind, we document why.*
