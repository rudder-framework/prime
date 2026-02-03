# The Ørthon Alphabet
## Complete Signal Typology & Engine Complement

> "A signal that cannot be fully described cannot be correctly measured."

---

## The Revelation

Typology is not a preprocessing step. **Typology IS the language.**

Every signal gets a complete identity card — a "word" made of letters from
10 dimensions. Each letter determines which engines run, which don't, what
window sizes to use, and what the numbers mean downstream.

The pipeline becomes:

```
TYPOLOGY (alphabet)     → defines what we can say about the system
SIGNAL ENGINES (words)  → measures exactly what typology prescribed
SIGNAL VECTORS          → organizes measurements into feature space
EIGENDECOMPOSITION      → captures energy and shape of that space
GEOMETRY DYNAMICS       → tracks how shape changes over time
SQL                     → asks questions of everything above
```

Typology sets the complexity ceiling. A simple system gets simple engines.
A chaotic multi-regime system gets the full arsenal. Nothing is overcomputed.
Nothing is undercomputed.

---

## The 10 Dimensions

Every signal is described by exactly 10 dimensions.
Together they form the complete identity card.

### Dimension 1: CONTINUITY
**Question:** "What kind of values does this signal take?"

| Value        | Meaning                          | Example                  |
|--------------|----------------------------------|--------------------------|
| CONTINUOUS   | Real-valued analog measurements  | Temperature, pressure    |
| DISCRETE     | Integer states or on/off         | Valve position, alarms   |
| EVENT        | Sparse occurrences in time       | Fault triggers, shutdowns|
| CONSTANT     | No variation (within tolerance)  | Dead sensor, setpoint    |

**Gate rule:** CONSTANT signals stop here. No engines run. They go on the
board as constants (still useful for context) but consume zero compute.
DISCRETE and EVENT signals get specialized treatment — no derivatives,
no interpolation, no spectral analysis.

**How measured:** Range check (ptp < tolerance), unique value count,
zero-crossing rate. This is Level 0 territory.

---

### Dimension 2: STATIONARITY
**Question:** "Does the statistical character change over time?"

| Value                  | Meaning                                | Action              |
|------------------------|----------------------------------------|----------------------|
| STATIONARY             | Constant mean and variance             | Global stats valid   |
| TREND_STATIONARY       | Deterministic trend, stationary around it | Detrend first     |
| DIFFERENCE_STATIONARY  | Stochastic trend / heteroscedastic     | Difference or roll   |
| NON_STATIONARY         | Fundamental character changes          | Rolling windows only |

**How measured:** ADF + KPSS joint test (Level 1 — already built and validated).
Variance ratio confirms. ACF decay characterizes memory timescale.

**Engine consequence:**
- STATIONARY → global engines valid (kurtosis, skewness, entropy)
- NON_STATIONARY → rolling engines ONLY (rolling_kurtosis, rolling_entropy, etc.)
- TREND_STATIONARY → detrend, then treat as stationary
- DIFFERENCE_STATIONARY → first-difference, then re-test

---

### Dimension 3: TEMPORAL PATTERN
**Question:** "What is the dominant temporal behavior?"

| Value          | Meaning                                      |
|----------------|----------------------------------------------|
| PERIODIC       | Repeating with fixed period (or near-fixed)  |
| QUASI_PERIODIC | Repeating but period varies                  |
| TRENDING       | Monotonic drift in one direction             |
| MEAN_REVERTING | Oscillates around an equilibrium             |
| CHAOTIC        | Deterministic but sensitive to initial cond.  |
| RANDOM         | No predictable temporal structure            |
| CONSTANT       | (inherited from Dimension 1)                 |

**How measured:** This is the heart of Level 2 (Masters classification).

Decision tree inputs:
- Dominant frequency detection (FFT peak SNR)
- Spectral flatness (flat = broadband, peaked = periodic)
- Autocorrelation decay rate (fast = random, slow = persistent)
- Permutation entropy (low = deterministic, high = random)
- Turning point ratio (low = trending, normal = random)
- Number of zero crossings of detrended signal
- Lyapunov exponent (positive = chaotic, negative = stable)

Decision tree logic:
```
IF spectral_peak_snr > threshold AND period_stable:
    → PERIODIC
ELIF spectral_peak_snr > threshold AND period_varies:
    → QUASI_PERIODIC
ELIF turning_points < expected AND monotonic:
    → TRENDING
ELIF hurst < 0.45 AND mean_reverting_test:
    → MEAN_REVERTING
ELIF lyapunov > 0 AND determinism > threshold:
    → CHAOTIC
ELSE:
    → RANDOM
```

**Engine consequence:**
- PERIODIC → harmonics_ratio, band_ratios, spectral_entropy, THD, phase
- QUASI_PERIODIC → band_ratios, spectral_entropy, instantaneous frequency
- TRENDING → hurst, rate_of_change_ratio, trend_r2
- MEAN_REVERTING → hurst, half_life, equilibrium estimation
- CHAOTIC → lyapunov, attractor_dim, recurrence, sample_entropy
- RANDOM → entropy, permutation_entropy

---

### Dimension 4: MEMORY
**Question:** "How long does the past influence the future?"

| Value           | Hurst Range  | Meaning                        |
|-----------------|--------------|--------------------------------|
| LONG_MEMORY     | > 0.65       | Strong persistence, trends last|
| SHORT_MEMORY    | 0.45 – 0.65  | Weak correlation, moderate     |
| ANTI_PERSISTENT | < 0.45       | Mean-reverting, overshoots     |

**How measured:** Hurst exponent (R/S method primary, DFA for validation).
ACF half-life provides the timescale in lags.

**Engine consequence:**
- LONG_MEMORY → hurst, rate_of_change_ratio, rolling windows need to be LARGE
- SHORT_MEMORY → standard window sizes
- ANTI_PERSISTENT → hurst, half-life estimation, smaller windows capture dynamics

**Window size rule:** `window_size = max(64, 4 × acf_half_life)`
Memory determines the minimum window for any rolling engine to be meaningful.

---

### Dimension 5: COMPLEXITY
**Question:** "How many effective degrees of freedom does this signal have?"

| Value  | Permutation Entropy | Meaning                           |
|--------|--------------------|------------------------------------|
| LOW    | < 0.3              | Highly regular, few patterns       |
| MEDIUM | 0.3 – 0.7          | Structured but variable            |
| HIGH   | > 0.7              | Nearly random, many patterns       |

**How measured:** Permutation entropy (order 3-5), sample entropy, SVD entropy.
These are complementary views of the same question.

**Engine consequence:**
- LOW complexity → minimal engine set (core + periodicity engines)
- MEDIUM → standard engine set
- HIGH → full entropy suite, chaos indicators, recurrence analysis

**Geometry consequence:** This is the preview of effective_dim. A signal with
low complexity will contribute few dimensions to the state geometry. High
complexity signals need more eigenvalues to capture their behavior.

---

### Dimension 6: DISTRIBUTION SHAPE
**Question:** "What does the amplitude distribution look like?"

| Value        | Kurtosis | Skewness | Meaning                   |
|--------------|----------|----------|---------------------------|
| GAUSSIAN     | 2.5–4.0  | |s|<0.5  | Normal-like tails          |
| HEAVY_TAILED | > 4.0    | any      | Extreme events present     |
| LIGHT_TAILED | < 2.5    | any      | Bounded, no extremes       |
| SKEWED_RIGHT | any      | > 0.5    | Asymmetric toward high     |
| SKEWED_LEFT  | any      | < -0.5   | Asymmetric toward low      |

**How measured:** Kurtosis (excess), skewness, Jarque-Bera test.
For non-stationary signals, use rolling distribution statistics.

**Engine consequence:**
- HEAVY_TAILED → crest_factor critical, peak_ratio, extreme value analysis
- GAUSSIAN → standard statistical engines sufficient
- SKEWED → directional analysis matters (degradation often skews right)

**Physics meaning:** Heavy tails in a degradation context = intermittent faults.
The system is mostly okay but occasionally spikes. That's information.

---

### Dimension 7: AMPLITUDE CHARACTER
**Question:** "What does the signal 'feel' like in the time domain?"

| Value     | Meaning                                    |
|-----------|--------------------------------------------|
| SMOOTH    | Low high-frequency content, gradual changes|
| NOISY     | Significant broadband noise floor          |
| IMPULSIVE | Sharp transient spikes above background    |
| MIXED     | Smooth baseline with intermittent impulses  |

**How measured:** Spectral centroid position, crest factor, kurtosis,
zero-crossing rate relative to signal bandwidth.

```
IF crest_factor > 6 AND kurtosis > 6:
    → IMPULSIVE
ELIF spectral_centroid < 0.2 × nyquist AND crest_factor < 4:
    → SMOOTH
ELIF spectral_centroid > 0.4 × nyquist:
    → NOISY
ELSE:
    → MIXED
```

**Engine consequence:**
- SMOOTH → rolling engines, trend detection, slow dynamics
- NOISY → entropy, sample_entropy (noise characterization)
- IMPULSIVE → crest_factor, peak_ratio, envelope analysis
- MIXED → full suite needed (both slow dynamics and impulse detection)

---

### Dimension 8: SPECTRAL CHARACTER
**Question:** "Where does the energy live in frequency?"

| Value      | Meaning                                     |
|------------|---------------------------------------------|
| NARROWBAND | Energy concentrated in few frequencies       |
| BROADBAND  | Energy spread across entire spectrum         |
| HARMONIC   | Fundamental + integer multiples              |
| ONE_OVER_F | Power law spectrum (pink/red noise)          |

**How measured:** Spectral flatness (Wiener entropy), harmonic-to-noise ratio,
spectral slope estimation (log-log fit of PSD).

**Engine consequence:**
- NARROWBAND → dominant frequency, bandwidth, Q-factor
- BROADBAND → spectral_entropy, band_ratios
- HARMONIC → harmonics_ratio, THD, individual harmonic tracking
- ONE_OVER_F → hurst (related to spectral slope), DFA

**Cross-reference:** PERIODIC signals are almost always NARROWBAND or HARMONIC.
RANDOM signals are almost always BROADBAND. CHAOTIC signals often show ONE_OVER_F.
These cross-checks validate the classification.

---

### Dimension 9: VOLATILITY
**Question:** "Is the variance itself changing, and does it cluster?"

| Value                | Meaning                                  |
|----------------------|------------------------------------------|
| HOMOSCEDASTIC        | Constant variance throughout             |
| HETEROSCEDASTIC      | Variance changes over time               |
| VOLATILITY_CLUSTERING| Variance clusters (calm/stormy periods)  |

**How measured:** Variance ratio (Dimension 2 already computes this),
ARCH/GARCH test on residuals, rolling variance standard deviation.

**Engine consequence:**
- HOMOSCEDASTIC → standard analysis, global variance meaningful
- HETEROSCEDASTIC → rolling_kurtosis, rolling_entropy, rolling_crest_factor
- VOLATILITY_CLUSTERING → GARCH modeling, regime detection priority

**Physics meaning:** Volatility clustering in degradation = the system
alternates between stable and unstable regimes. This is pre-failure behavior.

---

### Dimension 10: DETERMINISM
**Question:** "How much of the signal is predictable vs. random?"

| Value          | Meaning                                    |
|----------------|--------------------------------------------|
| DETERMINISTIC  | Nearly all structure is predictable        |
| STOCHASTIC     | Noise-dominated, little predictable struct.|
| MIXED          | Deterministic core + stochastic noise      |

**How measured:** Recurrence quantification analysis (determinism metric),
surrogate data testing (compare signal to phase-shuffled version),
predictability from permutation entropy.

**Engine consequence:**
- DETERMINISTIC → attractor reconstruction, Lyapunov, recurrence plots
- STOCHASTIC → entropy measures, statistical engines only
- MIXED → separate deterministic and stochastic components, analyze both

**Why this matters for geometry:** Deterministic signals trace trajectories
in state space. Stochastic signals fill clouds. The eigenvalue decomposition
captures fundamentally different physics depending on this dimension.

---

## The Complete Identity Card

A fully described signal looks like this:

```yaml
signal_id: "sensor_temperature_01"
typology:
  continuity: CONTINUOUS
  stationarity: NON_STATIONARY
  temporal_pattern: TRENDING
  memory: LONG_MEMORY
  complexity: LOW
  distribution: SKEWED_RIGHT
  amplitude: SMOOTH
  spectral: ONE_OVER_F
  volatility: HETEROSCEDASTIC
  determinism: MIXED

# AUTOMATICALLY DERIVED from the 10 dimensions:
engines:
  - kurtosis             # core (always)
  - skewness             # core (always)
  - crest_factor         # core (always)
  - rolling_kurtosis     # NON_STATIONARY
  - rolling_entropy      # NON_STATIONARY
  - rolling_crest_factor # NON_STATIONARY
  - hurst                # LONG_MEMORY + TRENDING
  - rate_of_change_ratio # TRENDING
  - sample_entropy       # MIXED determinism

window_size: 256          # 4 × acf_half_life (64)
derivative_depth: 2       # TRENDING needs velocity + acceleration
eigenvalue_budget: 3      # LOW complexity → few dimensions needed
```

A different signal gets a completely different card:

```yaml
signal_id: "vibration_bearing_x"
typology:
  continuity: CONTINUOUS
  stationarity: STATIONARY
  temporal_pattern: PERIODIC
  memory: SHORT_MEMORY
  complexity: MEDIUM
  distribution: HEAVY_TAILED
  amplitude: IMPULSIVE
  spectral: HARMONIC
  volatility: HOMOSCEDASTIC
  determinism: DETERMINISTIC

engines:
  - kurtosis
  - skewness
  - crest_factor
  - harmonics_ratio      # PERIODIC + HARMONIC
  - band_ratios          # PERIODIC
  - spectral_entropy     # PERIODIC
  - thd                  # HARMONIC
  - peak_ratio           # IMPULSIVE + HEAVY_TAILED

window_size: 128          # SHORT_MEMORY → standard
derivative_depth: 1       # STATIONARY → only velocity needed
eigenvalue_budget: 5      # MEDIUM complexity
```

Same system. Same pipeline. Completely different analysis.
**Typology made the decision. Engines obey.**

---

## What Each Layer Produces

```
Layer 1: TYPOLOGY
  Input:  observations.parquet (signal_id, I, value)
  Output: typology.parquet (one row per signal, all 10 dimensions)
          manifest.yaml (engine list, window sizes, parameters per signal)

Layer 2: SIGNAL ENGINES
  Input:  observations.parquet + manifest.yaml
  Output: signal_vector.parquet (signal_id, I, engine_1, engine_2, ...)
          One column per engine, one row per window

Layer 3: EIGENDECOMPOSITION
  Input:  signal_vector.parquet
  Output: state_vector.parquet (centroid — WHERE the system is)
          state_geometry.parquet (eigenvalues — SHAPE of the system)

Layer 4: GEOMETRY DYNAMICS
  Input:  state_geometry.parquet
  Output: geometry_dynamics.parquet (velocity, acceleration, jerk of shape)
          effective_dim trajectory, coherence tracking

Layer 5: SQL
  Input:  everything above
  Output: cohorts, z-scores, alerts, rankings, cross-comparisons
          No new computation. Just questions.
```

---

## Build Order

| Step | What                          | Validates Against           |
|------|-------------------------------|-----------------------------|
| 1    | Level 0: Data Validation      | Corrupt/missing data        |
| 2    | Level 1: Stationarity         | ✅ 5/5 benchmarks pass      |
| 3    | Dimension 1: Continuity       | Known digital/event signals |
| 4    | Level 2: Temporal Pattern     | Sine, trend, Lorenz, noise  |
| 5    | Dimension 4: Memory           | Known Hurst values          |
| 6    | Dimension 5: Complexity       | Permutation entropy benchmarks|
| 7    | Dimensions 6-7: Distribution + Amplitude | Statistical tests |
| 8    | Dimension 8: Spectral         | Known spectral signatures   |
| 9    | Dimension 9: Volatility       | GARCH benchmark data        |
| 10   | Dimension 10: Determinism     | Lorenz (det.) vs noise (stoch.)|
| 11   | Integration test: all 10      | Full identity cards for all benchmark signals |
| 12   | Manifest generator            | Engine lists match expectations |

Each step must pass validation before the next is built.

---

## The Principle

> The same eigenvalue decomposition means completely different things
> depending on the system type. Typology defines the landscape.
> Without the landscape, the math is meaningless.
>
> — The paper we wrote yesterday

This alphabet makes that concrete. Every signal gets a word.
Every word maps to engines. Every engine feeds vectors.
Vectors become eigenvalues. Eigenvalues become geometry.
Geometry becomes physics.

**Typology is the foundation. Everything else is consequence.**

---

*geometry leads — ørthon*
