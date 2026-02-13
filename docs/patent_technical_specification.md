# ORTHON/MANIFOLD Patent Technical Specification

## Domain-Agnostic Dynamical Systems Analysis Platform

**Prepared for Patent Attorney Review**
**Date: February 2026**

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Canonical Data Format: observations.parquet](#2-canonical-data-format-observationsparquet)
3. [Signal Typology and Classification Methods](#3-signal-typology-and-classification-methods)
4. [Manifest Generation and Windowing Methodology](#4-manifest-generation-and-windowing-methodology)
5. [Signal Vector Equations](#5-signal-vector-equations)
6. [Geometry Equations](#6-geometry-equations)
7. [Dynamical Systems Equations](#7-dynamical-systems-equations)
8. [Information Flow Equations](#8-information-flow-equations)
9. [SQL Classification Layer](#9-sql-classification-layer)
10. [ORTHON Interpretation Methods](#10-orthon-interpretation-methods)
11. [Machine Learning Feature Engineering and Regression](#11-machine-learning-feature-engineering-and-regression)
12. [Additional Computational Primitives](#12-additional-computational-primitives)
13. [System Boundaries: ORTHON vs. Manifold](#13-system-boundaries-orthon-vs-manifold)
14. [Annotated File Trees](#14-annotated-file-trees)

---

## 1. System Architecture Overview

The platform consists of two independent repositories that form a complete dynamical systems analysis pipeline:

**ORTHON** (Brain): Orchestration, signal classification, interpretation, and decision-making. ORTHON decides *what* to compute and *how* to interpret results.

**Manifold/Engines** (Muscle): Pure computation with no classification or decision logic. Engines computes *numbers* based on instructions from ORTHON.

### Data Flow

```
Raw Data (CSV/Parquet)
    |
    v
ORTHON: Stage 01 - Validate observations
    |
    v
ORTHON: Stage 02 - Compute 27 raw typology measures per signal
    |
    v
ORTHON: Stage 03 - Classify signals across 10 dimensions
    |
    v
ORTHON: Stage 04 - Generate manifest.yaml (engine selection + windowing)
    |
    v
[observations.parquet + typology.parquet + manifest.yaml]
    |
    v
Manifold/Engines: Signal Vector computation (per-signal features)
    |
    v
Manifold/Engines: State Vector computation (system centroids)
    |
    v
Manifold/Engines: Geometry computation (eigendecomposition, pairwise)
    |
    v
Manifold/Engines: Dynamics computation (FTLE, Lyapunov, RQA)
    |
    v
Manifold/Engines: Information Flow (transfer entropy, Granger causality)
    |
    v
[14 output parquet files]
    |
    v
ORTHON: SQL Classification (150+ views in DuckDB)
    |
    v
ORTHON: Stage 06 - Interpret (physics, dynamics)
    |
    v
ORTHON: Stage 07 - Predict (RUL, health, anomaly)
    |
    v
ORTHON: Stage 08 - Alert (early warning, fingerprinting)
```

---

## 2. Canonical Data Format: observations.parquet

### Schema

| Column    | Type    | Required | Description |
|-----------|---------|----------|-------------|
| cohort    | String  | Optional | Groups related signals (e.g., engine_1, pump_A) |
| signal_id | String  | Required | Which measurement (e.g., temp, pressure) |
| I         | UInt32  | Required | Sequential observation index within signal |
| value     | Float64 | Required | The measurement value |

### Critical Constraints

The `I` column is the canonical observation index. It MUST be sequential (0, 1, 2, ...) per signal_id. It is NOT a timestamp.

```
CORRECT:
signal_id | I | value
----------|---|------
temp      | 0 | 45.2
temp      | 1 | 45.4
temp      | 2 | 45.6
pressure  | 0 | 101.3
pressure  | 1 | 101.5

INCORRECT (timestamps in I):
signal_id | I          | value
----------|------------|------
temp      | 1596760568 | 45.2
temp      | 1596760569 | 45.4
```

### Auto-Repair Capabilities

The validation system (`orthon/ingest/validate_observations.py`) automatically detects and repairs:

| Issue | Detection Method | Auto-Fix |
|-------|-----------------|----------|
| I contains timestamps | `I.max() > n_rows * 10` | Sort by I, regenerate as 0,1,2... |
| Duplicate (signal_id, I) | Group count > 1 | Sort and regenerate I |
| Missing I column | Column not present | Create from row order |
| Column named 'timestamp' | Alias detection (20+ aliases) | Rename to 'I' |
| Column named 'y' | Alias detection | Rename to 'value' |
| Null signal_id | Null count > 0 | Remove rows |
| Non-numeric values | Cast failure | Cast to Float64 |
| Wide-format data | Multiple value columns | Melt to long format |

---

## 3. Signal Typology and Classification Methods

### 3.1 Raw Typology Computation (27 Measures)

**File**: `orthon/ingest/typology_raw.py`

For each signal, ORTHON computes 27 statistical measures:

| # | Measure | Equation | Purpose |
|---|---------|----------|---------|
| 1 | dominant_frequency | argmax(PSD(signal)) | FFT peak frequency |
| 2 | spectral_peak_snr | 10 * log10(peak_power / noise_floor) | Peak signal-to-noise ratio (dB) |
| 3 | spectral_flatness | exp(mean(log(PSD))) / mean(PSD) | Spectrum shape (0=peaked, 1=flat) |
| 4 | spectral_slope | linear_fit(log(f), log(PSD))[0] | Power law exponent |
| 5 | acf_half_life | min{k : ACF(k) < 0.5} | Autocorrelation decay time |
| 6 | turning_point_ratio | n_extrema / (n_samples - 2) | Fraction of local extrema |
| 7 | hurst | slope(log(R/S) vs log(k)) | Long-range dependence (R/S method) |
| 8 | perm_entropy | -sum(p_k * log2(p_k)) / log2(m!) | Ordinal pattern entropy, normalized |
| 9 | sample_entropy | -log(A/B) | Template matching complexity |
| 10 | lyapunov_proxy | slope(log(divergence) vs time) | Sensitivity to initial conditions |
| 11 | variance_ratio | var(first_half) / var(second_half) | Heteroscedasticity detector |
| 12 | n_samples | count(signal) | Signal length |
| 13 | signal_std | std(signal) | Standard deviation |
| 14 | unique_ratio | n_unique / n_samples | Value diversity |
| 15 | sparsity | count(value == 0) / n_samples | Zero fraction |
| 16 | kurtosis | E[(x-mu)^4] / sigma^4 - 3 | Tail heaviness |
| 17 | crest_factor | max(|x|) / RMS(x) | Peak-to-RMS ratio |
| 18 | zero_run_ratio | longest_zero_run / n_samples | Longest silent period |
| 19 | is_integer | all(x == floor(x)) | Discrete value detection |
| 20 | determinism_score | DET from RQA on subsample | Recurrence determinism |
| 21 | harmonic_noise_ratio | sum(harmonic_power) / sum(noise_power) | Harmonic content strength |
| 22 | adf_pvalue | Augmented Dickey-Fuller p-value | Unit root test |
| 23 | kpss_pvalue | KPSS test p-value | Stationarity test |
| 24 | segment_means | mean(signal) per segment | Segment-level trends |
| 25 | monotonic_count | count of monotonic segments | Trend persistence |
| 26 | acf_values | ACF at lags 1,5,10 | Short-range correlation |
| 27 | variance_growth | slope(log(var(blocks)) vs log(block_size)) | Scale-dependent variance |

### 3.2 Two-Stage Classification Pipeline

Classification proceeds in two stages: first detecting discrete/sparse signals, then classifying remaining continuous signals.

#### Stage 1: Discrete/Sparse Detection (PR5)

**File**: `orthon/typology/discrete_sparse.py`
**Config**: `orthon/config/discrete_sparse_config.py`

This stage runs FIRST and catches non-continuous signals before the continuous classifier sees them.

| Type | Detection Conditions | Spectral Label |
|------|---------------------|----------------|
| **CONSTANT** | signal_std approximately 0 OR unique_ratio < 0.001 | NONE |
| **BINARY** | Exactly 2 unique values | SWITCHING |
| **DISCRETE** | is_integer AND unique_ratio < 5% | QUANTIZED |
| **IMPULSIVE** | kurtosis > 20 AND crest_factor > 10 | BROADBAND |
| **EVENT** | sparsity > 80% AND kurtosis > 10 | SPARSE |
| **STEP** | n_changepoints >= 2 AND unique_ratio < 10% | DC_DOMINANT |
| **INTERMITTENT** | zero_run_ratio > 30% AND sparsity in [30%, 80%] | BURSTY |

#### Stage 2: Continuous Classification (PR4)

**File**: `orthon/typology/level2_corrections.py`
**Config**: `orthon/config/typology_config.py`

If a signal is not captured by Stage 1, the continuous decision tree applies:

**Step 0 - CONSTANT Backup**: If classify_constant_from_row(row) returns True, classify as CONSTANT.

**Step 1 - First-Bin Artifact Detection**:
```
IF dominant_frequency approximately equals 1/FFT_size AND spectral_slope < -0.3:
    Mark as artifact (not truly periodic)
```

**Step 2 - Bounded Deterministic Override** (prevents misclassifying smooth chaos as TRENDING):
```
IF hurst > 0.95 AND perm_entropy < 0.5 AND variance_ratio < 3.0:
    Set bounded_deterministic = True (skip trending checks)
```

**Step 3 - Segment Trend Detection** (catches oscillating trends like battery degradation):
```
IF segment_means are monotonic AND total_change >= 20% AND hurst > 0.60:
    Return TRENDING
```

**Step 4 - TRENDING** (High Hurst):
```
IF hurst >= 0.99:
    Return TRENDING

IF hurst > 0.85 AND (acf_half_life is None OR acf_half_life/n_samples > 0.10)
   AND sample_entropy < 0.15:
    Return TRENDING
```

**Step 5 - DRIFTING** (Non-stationary persistent, noisy trends):
```
IF 0.85 <= hurst < 0.99 AND perm_entropy >= 0.90
   AND (variance_ratio is None OR variance_ratio >= 0.2):
    Return DRIFTING
```

**Step 6 - Integrated Process Detection** (unit root override):
```
IF adf_pvalue > 0.10 AND kpss_pvalue < 0.05 AND acf_half_life is None:
    Return DRIFTING  (all three stationarity tests indicate non-stationarity)
```

**Step 7 - PERIODIC** (with spectral override):
```
Six gates must pass:
  1. Not a first-bin artifact
  2. spectral_flatness <= 0.7
  3. spectral_peak_snr >= 6.0 dB
  4. acf_half_life is not None
  5. hurst <= 0.95

Spectral override: IF spectral_peak_snr > 30 AND spectral_flatness < 0.1:
    Return PERIODIC (skip turning point check)

  6. turning_point_ratio <= 0.95
```

**Step 8 - RANDOM**:
```
IF spectral_flatness > 0.9 AND perm_entropy > 0.99:
    Return RANDOM
```

**Step 9 - CHAOTIC**:
```
IF n_samples >= 500 AND lyapunov_proxy > 0.5 AND perm_entropy > 0.95:
    Return CHAOTIC

Clean chaos variant:
IF lyapunov_proxy > 0.15 AND perm_entropy < 0.6 AND sample_entropy < 0.3:
    Return CHAOTIC
```

**Step 10 - QUASI_PERIODIC**:
```
IF turning_point_ratio < 0.7:
    Return QUASI_PERIODIC
```

**Step 11 - Default**: Return STATIONARY

### 3.3 Output: typology.parquet (10 Classification Dimensions)

| Dimension | Possible Values |
|-----------|----------------|
| temporal_pattern | PERIODIC, TRENDING, DRIFTING, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY, CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT, STEP, INTERMITTENT |
| spectral | HARMONIC, NARROWBAND, BROADBAND, RED_NOISE, BLUE_NOISE, NONE, SWITCHING, QUANTIZED, SPARSE, BURSTY, DC_DOMINANT |
| stationarity | STATIONARY, NON_STATIONARY |
| memory | SHORT_MEMORY, LONG_MEMORY |
| complexity | LOW, MEDIUM, HIGH |
| continuity | CONTINUOUS, DISCRETE |
| determinism | DETERMINISTIC, STOCHASTIC |
| distribution | GAUSSIAN, LIGHT_TAILED, HEAVY_TAILED |
| amplitude | SMOOTH, BURSTY, MIXED |
| volatility | HOMOSCEDASTIC, VOLATILITY_CLUSTERING |

---

## 4. Manifest Generation and Windowing Methodology

### 4.1 Characteristic Time Computation

**File**: `orthon/manifest/characteristic_time.py`

The system computes a data-driven "characteristic time" for each signal, representing how fast the signal changes. This determines the optimal window size.

```
Priority 1: ACF half-life (memory-based)
    IF acf_half_life exists AND acf_half_life > 0:
        characteristic_time = acf_half_life

Priority 2: Dominant frequency period
    IF dominant_frequency > 0:
        period = 1.0 / dominant_frequency
        IF period < n_samples:
            characteristic_time = period

Priority 3: Hurst-based (persistence)
    IF hurst > 0.7:
        characteristic_time = n_samples * (hurst - 0.5) * 0.2

Priority 4: Turning point ratio (oscillation rate)
    IF 0.1 < turning_point_ratio < 0.9:
        characteristic_time = 2.0 / turning_point_ratio

Fallback: max(n_samples * 0.01, 64)
```

### 4.2 Window and Stride Computation

```
window = int(characteristic_time * 2.5)
window = clamp(window, 64, 2048)
window = min(window, n_samples)

ratio = characteristic_time / n_samples
IF ratio < 0.01:       stride_fraction = 0.25 (75% overlap, fast dynamics)
ELIF ratio > 0.10:     stride_fraction = 0.75 (25% overlap, slow dynamics)
ELSE:                   stride_fraction = 0.50 (50% overlap, medium)

stride = max(8, int(window * stride_fraction))
```

### 4.3 System Window (Multi-Scale Alignment)

**File**: `orthon/manifest/system_window.py`

A common window for state_vector/geometry alignment across all signals:

```
system_tau = max(all_characteristic_times)
system_window = int(system_tau * 2.5)
system_window = clamp(system_window, 64, 4096)
```

### 4.4 Multi-Scale Representation

Each signal is classified as either spectral or trajectory representation:

```
ratio = characteristic_time / system_window
IF ratio < 0.3:    representation = 'spectral'   (fast signals: frequency content)
ELSE:               representation = 'trajectory'  (slow signals: path/trend)
```

### 4.5 Engine Selection (Typology-Guided)

**File**: `orthon/manifest/generator.py`

**Philosophy**: "If it's a maybe, run it." Only CONSTANT signals skip all engines.

**Base Engines** (always included, except CONSTANT):
- Distribution: crest_factor, kurtosis, skewness
- Spectral: spectral
- Complexity: sample_entropy, perm_entropy
- Memory: hurst, acf_decay

**Type-Specific Engine Additions**:

| Temporal Pattern | Additional Engines |
|-----------------|--------------------|
| TRENDING | hurst, rate_of_change, trend_r2, detrend_std, cusum, variance_growth |
| DRIFTING | hurst, rate_of_change, trend_r2, detrend_std, cusum, variance_growth |
| PERIODIC | harmonics, thd, frequency_bands, fundamental_freq, phase_coherence, snr |
| CHAOTIC | lyapunov, correlation_dimension, recurrence_rate, determinism, embedding_dim |
| RANDOM | spectral_entropy, band_power, frequency_bands |
| QUASI_PERIODIC | frequency_bands, harmonics, rate_of_change |
| STATIONARY | frequency_bands, spectral_entropy, acf_decay, variance_ratio, adf_stat |
| CONSTANT | **All engines removed** |
| BINARY | transition_count, duty_cycle, mean_time_between, switching_frequency |
| DISCRETE | level_histogram, transition_matrix, dwell_times, level_count, entropy |
| IMPULSIVE | peak_detection, inter_arrival, peak_amplitude_dist, envelope, rise_time |
| EVENT | event_rate, inter_event_time, event_amplitude |
| STEP | changepoint_detection, level_means, regime_duration |
| INTERMITTENT | burst_detection, activity_ratio, silence_distribution |

### 4.6 Per-Engine Minimum Window Requirements

Some engines require more samples than a signal's characteristic window provides:

| Engine | Minimum Window | Reason |
|--------|---------------|--------|
| spectral, harmonics, fundamental_freq, thd | 64 | FFT requires sufficient samples |
| frequency_bands, spectral_entropy, band_power | 64 | FFT-based |
| sample_entropy | 64 | Embedding dimension requirements |
| hurst | 128 | R/S analysis needs longer series |
| crest_factor, kurtosis, skewness, acf_decay | 32 | Work fine with small windows |

When a signal's window < engine minimum, the manifest specifies `engine_window_overrides` so Engines uses an expanded window only for that engine.

### 4.7 Manifest v2.5 Structure

```yaml
version: '2.5'
job_id: orthon-20260202-143052
created_at: '2026-02-02T14:30:52'
generator: orthon.manifest_generator v2.5

paths:
  observations: observations.parquet
  typology: typology.parquet
  output_dir: output/

system:
  window: 128
  stride: 64
  note: Common window for state_vector/geometry alignment

engine_windows:
  spectral: 64
  harmonics: 64
  hurst: 128

summary:
  total_signals: 24
  active_signals: 22
  constant_signals: 2
  signal_engines: [spectral, hurst, kurtosis, ...]
  n_signal_engines: 15

params:
  default_window: 128
  default_stride: 64
  min_samples: 64

cohorts:
  engine_1:
    sensor_02:
      engines: [kurtosis, hurst, spectral, ...]
      window_size: 32
      window_method: period
      window_confidence: high
      stride: 16
      derivative_depth: 1
      eigenvalue_budget: 5
      engine_window_overrides:
        spectral: 64
        hurst: 128
      typology:
        temporal_pattern: PERIODIC
        spectral: NARROWBAND

skip_signals:
  - engine_1/constant_signal

pair_engines: [granger, transfer_entropy]
symmetric_pair_engines: [cointegration, correlation, mutual_info]
```

---

## 5. Signal Vector Equations

**Files**: `engines/vector/engines/shape.py`, `complexity.py`, `spectral.py`, `harmonic.py`

The signal vector stage computes scale-invariant features per signal per window. Each window of raw observations is transformed into a feature vector.

### 5.1 Shape Features (Always Computed)

**Kurtosis** (4th standardized moment):
$$\text{Kurt} = \frac{E[(y - \mu)^4]}{\sigma^4} - 3$$

**Skewness** (3rd standardized moment):
$$\text{Skew} = \frac{E[(y - \mu)^3]}{\sigma^3}$$

**Crest Factor** (peak-to-RMS ratio):
$$CF = \frac{\max(|y|)}{\sqrt{\frac{1}{N}\sum y_i^2}}$$

### 5.2 Complexity Features (Always Computed)

**Permutation Entropy** (ordinal pattern complexity):
1. Create embedded vectors: [x(i), x(i+tau), ..., x(i+(m-1)*tau)]
2. Compute ordinal pattern: pi = argsort(argsort(vector))
3. Count unique patterns: p_k = count(pattern_k) / n_patterns
4. Shannon entropy: H = -sum(p_k * log2(p_k))
5. Normalize: H_norm = H / log2(m!)

**Sample Entropy** (template matching):
$$\text{SampEn}(m, r) = -\ln\frac{A}{B}$$
Where B = count of template matches of length m within tolerance r, A = count for length m+1, r = 0.2 * std(signal).

**Hurst Exponent** (Rescaled Range method):
1. For each segment length k: compute R/S = (max(cumsum) - min(cumsum)) / std
2. Fit log(R/S) vs log(k): slope = H
- H < 0.5: anti-persistent (mean-reverting)
- H = 0.5: random walk
- H > 0.5: persistent (trending)

**ACF Decay**: First lag where |ACF(k)| < threshold (default 1/e).

### 5.3 Spectral Features (Gated: all non-CONSTANT signals)

**Power Spectral Density** (Welch method with Hanning window):
$$P(f) = \frac{1}{K} \sum_{k=1}^{K} |X_k(f)|^2$$

**Spectral Slope** (power law fit):
$$\log P(f) \approx \alpha \cdot \log f + \text{const}$$

**Dominant Frequency**:
$$f_{dom} = \arg\max_f P(f)$$

**Spectral Entropy** (distribution flatness):
$$H_s = -\sum \hat{P}(f) \log_2 \hat{P}(f), \quad \hat{P}(f) = P(f) / \sum P$$

**Spectral Centroid** (center of mass):
$$f_c = \frac{\sum f \cdot P(f)}{\sum P(f)}$$

**Spectral Bandwidth** (spread around centroid):
$$BW = \sqrt{\frac{\sum (f - f_c)^2 \cdot P(f)}{\sum P(f)}}$$

### 5.4 Harmonic Features (Gated: PERIODIC, QUASI_PERIODIC)

**Fundamental Frequency**: Lowest significant peak in PSD.

**Total Harmonic Distortion**:
$$THD = \frac{\sqrt{\sum_{n=2}^{N} P(n \cdot f_0)}}{P(f_0)}$$

**Harmonic-to-Noise Ratio**:
$$HNR = \frac{\sum P(\text{harmonics})}{\sum P(\text{non-harmonics})}$$

### 5.5 Additional Signal Engines (42 Total)

Beyond core features, the system includes 42 specialized signal engines including:

- **Lyapunov** (CHAOTIC signals): Rosenstein's algorithm for maximum Lyapunov exponent
- **RQA** (CHAOTIC signals): Recurrence Quantification Analysis (determinism, laminarity, trapping time)
- **Attractor** (CHAOTIC signals): Correlation dimension via Grassberger-Procaccia
- **DMD** (CHAOTIC, RANDOM, TRENDING): Dynamic Mode Decomposition for linear dynamics
- **GARCH(1,1)** (VOLATILE signals): Time-varying volatility model
- **HMM** (STEP, DISCRETE): Hidden Markov Model for regime detection
- **Hilbert Stability** (PERIODIC): Instantaneous frequency stability analysis
- **Wavelet Stability** (all types): Time-frequency energy decomposition
- **Physics Stack** (multi-signal): Four-layer physics interpretation
- **LOF** (all types): Local Outlier Factor anomaly detection
- **Cycle Counting** (PERIODIC): Rainflow counting for fatigue analysis
- **Transition Matrix** (DISCRETE, BINARY): Markov transition probabilities

---

## 6. Geometry Equations

### 6.1 State Vector (Centroid Computation)

**File**: `engines/manifold/state/centroid.py`

For each time index I, the centroid (state vector) is the mean of all signal feature vectors:

$$\vec{c}(I) = \frac{1}{N} \sum_{i=1}^{N} \vec{s}_i(I)$$

Where N = number of signals, s_i = feature vector of signal i at index I.

The centroid represents WHERE the system is in feature space.

### 6.2 State Geometry (Eigendecomposition)

**File**: `engines/manifold/state/eigendecomp.py`, `engines/manifold/state_geometry.py`

The geometry represents HOW the system is distributed around its centroid.

**Step 1: Center signals around centroid**:
$$\mathbf{X}_{\text{centered}} = \mathbf{X}_{\text{signals}} - \vec{c}$$

**Step 2: Z-score normalize** (so all features contribute equally):
$$\mathbf{X}_{\text{norm}} = \frac{\mathbf{X}_{\text{centered}} - \mu}{\sigma}$$

**Step 3: Singular Value Decomposition**:
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$

Where:
- U (N x k): left singular vectors (signal loadings)
- Sigma (k x k): singular values
- V^T (k x D): right singular vectors (principal components)

**Step 4: Eigenvalues from SVD**:
$$\lambda_i = \frac{\sigma_i^2}{N - 1}$$

**Step 5: Derived metrics**:

**Effective Dimension** (participation ratio):
$$d_{\text{eff}} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

Interpretation: d_eff = 1 means all variance in one dimension (collapsed); d_eff = D means uniformly distributed (maximally complex).

**Explained Ratio**:
$$\text{explained}_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**Eigenvalue Entropy**:
$$S = -\sum_i p_i \log(p_i), \quad p_i = \frac{\lambda_i}{\sum_j \lambda_j}$$

**Condition Number**:
$$\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$$

**Eigenvalue Ratios**: ratio_2_1 = lambda_2 / lambda_1, ratio_3_1 = lambda_3 / lambda_1

**Eigenvector Continuity Enforcement**: Across sequential windows, eigenvectors may arbitrarily flip sign. The system enforces continuity:
```
FOR each window after the first:
    IF dot(ev_current, ev_previous) < 0:
        ev_current = -ev_current  (flip sign)
        flip_count += 1
```
In TEP data, 71% of windows required sign flip correction.

### 6.3 Signal Geometry (Signal-to-State Relationships)

**File**: `engines/manifold/signal_geometry.py`

Per-signal relationship to the system state:

**Distance to Centroid**:
$$d_i = \|\vec{s}_i - \vec{c}\|_2 = \sqrt{\sum_j (s_{i,j} - c_j)^2}$$

**Coherence to PC1** (alignment with dominant mode):
$$\text{coherence}_i = \frac{\langle \vec{s}'_i, \vec{PC}_1 \rangle}{\|\vec{s}'_i\| \cdot \|\vec{PC}_1\|}$$

Where s'_i = s_i - c (centered signal). Range: [-1, 1].

**Contribution** (projection onto centroid direction):
$$\text{contribution}_i = \frac{\langle \vec{s}_i, \vec{c} \rangle}{\|\vec{c}\|}$$

**Residual** (component orthogonal to centroid):
$$\vec{p}_i = \frac{\langle \vec{s}_i, \vec{c} \rangle}{\|\vec{c}\|^2} \cdot \vec{c}, \quad \text{residual}_i = \|\vec{s}_i - \vec{p}_i\|$$

### 6.4 Signal Pairwise Geometry

**File**: `engines/manifold/signal_pairwise.py`

For each pair of signals (i, j):

**Euclidean Distance**: d(s_a, s_b) = ||s_a - s_b||_2

**Cosine Similarity**: cos(theta) = <s_a, s_b> / (||s_a|| * ||s_b||)

**Pearson Correlation**: r = Cov(s_a, s_b) / (std(s_a) * std(s_b))

**Eigenvector Co-Loading Gating** (novel innovation):
```
pc1_a = PC1_loadings[signal_a]
pc1_b = PC1_loadings[signal_b]
coloading = |pc1_a * pc1_b|
needs_granger = coloading > threshold (default 0.1)
```
This gates expensive pairwise operations (Granger causality) using cheap eigenvector information.

### 6.5 Geometry at Cohort Scale

The same eigendecomposition is applied at cohort level, where signals are cohort-level aggregates rather than individual sensors. This provides two-scale analysis:
- Scale 1 (Signal): Individual signal features per window
- Scale 2 (Cohort): Aggregated cohort features per window

Both use identical SVD mathematics at different aggregation levels.

---

## 7. Dynamical Systems Equations

### 7.1 Geometry Dynamics (Differential Geometry of State Evolution)

**File**: `engines/manifold/geometry_dynamics.py`

Derivatives of the state trajectory track how the system evolves:

**Velocity** (first derivative):
$$v_d(I) = \frac{x_d(I+1) - x_d(I-1)}{2}$$

**State Speed** (velocity magnitude):
$$s(I) = \sqrt{\sum_d v_d(I)^2}$$

**Acceleration** (second derivative):
$$a_d(I) = x_d(I+1) - 2x_d(I) + x_d(I-1)$$

**Jerk** (third derivative):
$$j_d(I) = \frac{a_d(I+1) - a_d(I-1)}{2}$$

**Curvature** (rate of trajectory bending):
$$\kappa(I) = \frac{\|\vec{a}_\perp\|}{\|\vec{v}\|^2}$$
Where a_perp is the acceleration component perpendicular to velocity.

**Effective Dimension Velocity** (collapse detection):
$$\dot{d}_{\text{eff}}(I) = \frac{d_{\text{eff}}(I+1) - d_{\text{eff}}(I-1)}{2}$$
- d_eff_velocity < -0.1: COLLAPSING (losing degrees of freedom)
- d_eff_velocity > 0.1: EXPANDING
- |d_eff_velocity| < 0.01: STABLE

### 7.2 Finite-Time Lyapunov Exponents (FTLE)

**File**: `engines/manifold/dynamics/ftle.py`

**Rosenstein's Algorithm**:
1. Embed signal into m-dimensional phase space using time delay tau:
   x_i = [y(i), y(i+tau), ..., y(i+(m-1)*tau)]
2. Find nearest neighbor for each point (excluding temporal neighbors: |i-j| >= min_separation)
3. Track divergence: d_k(i) = ||x_{i+k} - x_{nn(i)+k}||
4. Average: <log(d_k)> = (1/N) * sum_i(log(d_k(i)))
5. Linear fit: <log(d_k)> = lambda * k + const
6. FTLE = slope lambda

**Embedding Parameters**:
- Optimal delay tau: via Average Mutual Information (AMI) - first local minimum
- Optimal dimension m: via Cao's method - saturation of E1(d) ratio

**Cao's Method**:
For each dimension d:
1. Embed in d and d+1 dimensions
2. For each point i, find nearest neighbor j in d-space
3. Compute a(i,d) = ||x_i(d+1) - x_nn(i)(d+1)|| / ||x_i(d) - x_nn(i)(d)||
4. E(d) = mean(a(i,d))
5. E1(d) = E(d+1) / E(d)
6. Choose d where E1 first approaches 1

**Interpretation**:
- FTLE > 0: Exponential divergence (chaotic region)
- FTLE ~ 0: Neutral stability
- FTLE < 0: Exponential convergence (stable)
- Ridges of FTLE field identify Lagrangian Coherent Structures (dynamical barriers)

**Rolling FTLE**: Time-varying FTLE field computed over sliding windows to show stability evolution.

### 7.3 Lyapunov Exponents

**File**: `engines/manifold/dynamics/lyapunov.py`

Full Lyapunov exponent computation using both Rosenstein (max exponent) and Kantz methods.

**Lyapunov Spectrum** (QR decomposition method):
1. Initialize orthonormal basis Q = I
2. For each trajectory point: estimate local Jacobian J from nearest neighbors
3. Evolve: W = J * Q
4. QR decomposition: W = Q_new * R
5. Accumulate log(|R_ii|) for each exponent
6. Average and sort descending

**Kaplan-Yorke Dimension** (from Lyapunov spectrum):
$$D_{KY} = j + \frac{\sum_{i=1}^{j} \lambda_i}{|\lambda_{j+1}|}$$
Where j = largest index such that sum of first j exponents >= 0.

### 7.4 Recurrence Quantification Analysis (RQA)

**File**: `engines/manifold/dynamics/attractor.py`

**Recurrence Matrix**:
$$R_{i,j} = \begin{cases} 1 & \text{if } \|\vec{X}_i - \vec{X}_j\| < \epsilon \\ 0 & \text{otherwise} \end{cases}$$

Where epsilon = 10th percentile of pairwise distances.

**Recurrence Rate**: RR = sum(R) / N^2

**Determinism** (diagonal line structure): DET = sum(points on diagonal lines >= 2) / sum(R)

**Laminarity** (vertical line structure): LAM = sum(points on vertical lines >= 2) / sum(R)

**Trapping Time**: Mean length of vertical lines (average time trapped in state).

**RQA Entropy**: Shannon entropy of diagonal line length distribution.

**Correlation Dimension** (Grassberger-Procaccia):
$$C(r) = \frac{1}{N(N-1)} \sum_{i \neq j} H(r - \|\vec{X}_i - \vec{X}_j\|)$$
$$D_c = \lim_{r \to 0} \frac{\log C(r)}{\log r}$$

### 7.5 Critical Slowing Down Detection

**File**: `engines/manifold/dynamics/critical_slowing_down.py`

Near bifurcation, systems show increased autocorrelation and variance.

**CSD Composite Score**:
$$\text{CSD} = \frac{0.3 \cdot AC(\rho_1) + 0.25 \cdot ACT(\text{trend}) + 0.25 \cdot VT(\text{trend}) + 0.2 \cdot VR(\text{ratio})}{W}$$

Where:
- AC(rho_1) = clamp(rho_1, 0, 1): current autocorrelation level
- ACT(s) = clamp(10 * s, 0, 1): autocorrelation trend
- VT(s) = clamp(100 * s, 0, 1): variance trend
- VR(r) = clamp((r - 1) / 2, 0, 1): variance ratio

**Detection**: CSD_detected = (CSD_score > 0.6) AND (rho_1 > 0.7) AND (variance_trend > 0)

### 7.6 Trajectory Sensitivity Analysis

**File**: `engines/manifold/dynamics/trajectory_sensitivity.py`

Computes which variables are most sensitive at each state:
1. For each point, estimate local Jacobian via least-squares fit of neighbor evolution
2. Sensitivity per variable = column norm of Jacobian
3. Rank variables by sensitivity at each moment
4. Track sensitivity entropy (distributed vs. concentrated importance)

### 7.7 Saddle Point Detection

**File**: `engines/manifold/dynamics/saddle_detection.py`

Identifies unstable equilibria (basin boundaries):
1. Estimate local Jacobian J from nearest neighbors
2. Compute eigenvalues of J
3. Saddle = mixed eigenvalues (some positive real parts, some negative)
4. Saddle score = fraction of mixed-sign eigenvalues
5. Basin stability = 1 - rolling_mean(saddle_score)
6. Separatrix distance = distance to nearest identified saddle point

### 7.8 Ridge Proximity and Urgency

**File**: `engines/dynamics/engines/ridge.py`

Combined FTLE gradient and velocity for approach-to-barrier assessment:

**FTLE Gradient**: d(FTLE)/dI (temporal gradient)

**Urgency**: speed * sign(gradient) * |gradient|

**Time-to-Ridge**: (ridge_threshold - current_FTLE) / urgency

### 7.9 Thermodynamic Mapping

**File**: `engines/decompose/engines/thermodynamics.py`

Maps eigenvalue dynamics onto statistical mechanics analogues:

**Entropy Proxy**: S(I) = effective_dim(I)

**Energy Proxy**: E(I) = total_variance(I)

**Temperature**: T(I) = dS/dI = d(effective_dim)/dI

**Free Energy**: F(I) = E(I) - T(I) * S(I)

**Heat Capacity**: C(I) = dE/dT

| Metric | Physical Meaning |
|--------|-----------------|
| S (eff_dim) | Degrees of freedom (entropy proxy) |
| E (total_var) | Total system variability (energy proxy) |
| T (dS/dI) | Rate of dimensional change (temperature) |
| F (E - TS) | Constrained energy (free energy) |
| C (dE/dT) | Energy sensitivity to temperature |

---

## 8. Information Flow Equations

### 8.1 Transfer Entropy

**File**: `engines/pairwise/engines/information.py`, `engines/primitives/information/transfer.py`

Measures directional information flow from source X to target Y:

$$TE(X \to Y) = H(Y_f | Y_p) - H(Y_f | Y_p, X_p)$$

Where:
- Y_f = target's future value
- Y_p = target's past values
- X_p = source's past values

**Algorithm**:
1. Discretize both signals into n_bins bins
2. Build joint state vectors for target and source histories
3. Compute conditional entropies via joint histograms
4. TE = H(Y_f, Y_p) - H(Y_p) - H(Y_f, Y_p, X_p) + H(Y_p, X_p)

**Normalized TE**: TE / H(Y_future) (fraction of uncertainty reduced)

**Net TE**: TE(A->B) - TE(B->A) (directional asymmetry)

### 8.2 Granger Causality

**File**: `engines/primitives/pairwise/causality.py`

Tests if past values of X help predict Y beyond Y's own history:

**Restricted model**: Y_t = alpha + sum(beta_i * Y_{t-i}) + epsilon_R

**Unrestricted model**: Y_t = alpha + sum(beta_i * Y_{t-i}) + sum(gamma_j * X_{t-j}) + epsilon_U

**F-statistic**: F = ((SSR_R - SSR_U) / p) / (SSR_U / (n - 2p - 1))

**Optimal lag selection via BIC**: BIC = n * ln(SSR/n) + k * ln(n)

### 8.3 Convergent Cross-Mapping (CCM)

**File**: `engines/primitives/pairwise/causality.py`

Nonlinear causality test based on Takens' embedding theorem:
1. Embed both signals in delay space
2. For each point in M_B, find k nearest neighbors
3. Predict A using weighted average of neighbors' A values
4. Correlation rho_AB = corr(predicted_A, actual_A)

If A causes B: rho_AB > rho_BA (B's attractor contains information about A).

### 8.4 Mutual Information

$$I(X;Y) = H(X) + H(Y) - H(X,Y)$$

Captures ANY statistical dependence (linear or nonlinear). Normalized: I_norm = I / min(H(X), H(Y)).

### 8.5 Copula Analysis

**File**: `engines/pairwise/engines/copula.py`

Models joint tail dependence beyond linear correlation:

**Gaussian Copula**: Normal dependence structure, theta = Pearson correlation

**Clayton Copula**: Lower tail dependence
$$C(u,v) = (u^{-\theta} + v^{-\theta} - 1)^{-1/\theta}$$
$$\lambda_L = 2^{-1/\theta}$$

**Gumbel Copula**: Upper tail dependence
$$C(u,v) = \exp(-((-\ln u)^\theta + (-\ln v)^\theta)^{1/\theta})$$
$$\lambda_U = 2 - 2^{1/\theta}$$

**Frank Copula**: Symmetric tail behavior
$$C(u,v) = -\frac{1}{\theta}\ln\left(1 + \frac{(e^{-\theta u} - 1)(e^{-\theta v} - 1)}{e^{-\theta} - 1}\right)$$

### 8.6 Cointegration Analysis

**File**: `engines/pairwise/engines/cointegration.py`

Tests if two signals share a common stochastic trend:
1. OLS regression: Y = beta * X + epsilon (hedge ratio)
2. ADF test on residuals: if p < 0.05, signals are cointegrated
3. Half-life: -log(2) / log(|AR1_coefficient|) (mean reversion speed)

### 8.7 Partial Information Decomposition

**File**: `engines/primitives/information/decomposition.py`

Decomposes multivariate information into:
- **Redundancy**: min(I(X1;Y), I(X2;Y)) - available from either source
- **Unique Information**: max(0, I(Xi;Y) - Redundancy) - only one source provides
- **Synergy**: I(X1,X2;Y) - Red - Uniq1 - Uniq2 - only available jointly

---

## 9. SQL Classification Layer

**Directory**: `orthon/sql/` (52+ SQL files, 150+ views)

All classification logic resides in ORTHON, NOT in Engines. The SQL layer runs on DuckDB and transforms raw numerical outputs into interpretable classifications.

### 9.1 Master Classification Views

**File**: `orthon/sql/layers/classification.sql`

#### v_trajectory_type (Lyapunov-Based Chaos Classification)

| Lyapunov (lambda) | Classification | Meaning |
|-------------------|---------------|---------|
| lambda > 0.1 | CHAOTIC | Exponential trajectory divergence |
| lambda > 0.01 | QUASI_PERIODIC | Edge of chaos |
| lambda > -0.01 | OSCILLATING | Limit cycle behavior |
| lambda > -0.1 | CONVERGING | Damped oscillation |
| lambda < -0.1 | STABLE | Fixed point attractor |

#### v_stability_class (Numeric Stability Scoring)

stability_score = LEAST(1.0, GREATEST(-1.0, lyapunov * 10)). Range: [-1, 1] where negative = stable.

#### v_geometry_windowed (Collapse Detection)

| effective_dim_velocity | Status | Lifecycle Stage |
|----------------------|--------|-----------------|
| < -0.1 | COLLAPSING | early_warning (0-20%), mid_life (20-50%), late_stage (50-80%), imminent (>80%) |
| > 0.1 | EXPANDING | - |
| abs < 0.01 | STABLE | - |
| else | DRIFTING | - |

#### v_anomaly_severity (Z-Score Based)

| |z-score| | Severity |
|-----------|----------|
| > 5 | CRITICAL |
| > 4 | SEVERE |
| > 3 | MODERATE |
| > 2 | MILD |
| else | NORMAL |

#### v_coupling_strength

| |correlation| | Classification |
|---------------|----------------|
| > 0.9 | STRONGLY_COUPLED |
| > 0.7 | MODERATELY_COUPLED |
| > 0.4 | WEAKLY_COUPLED |
| else | UNCOUPLED |

#### v_system_health (Unified Health Score)

Combines all classifications into single score [0, 1]:

| Condition | Health Score |
|-----------|-------------|
| geometry_status = 'collapsing' | 0.2 |
| stability_class = 'unstable' | 0.3 |
| trajectory_type = 'chaotic' | 0.4 |
| geometry_status = 'drifting' | 0.6 |
| stability_class = 'quasi_periodic' | 0.65 |
| trajectory_type = 'oscillating' | 0.8 |
| stability_class IN ('stable', 'strongly_stable') | 1.0 |

Risk levels: CRITICAL, HIGH, ELEVATED, MODERATE, LOW.

### 9.2 Dynamics Stability Layer

**File**: `orthon/sql/layers/30_dynamics_stability.sql`

Window-level RQA classification:
- DET > 0.5 AND LAM < 0.2: PERIODIC
- DET > 0.3 AND LAM > 0.3: INTERMITTENT
- DET < 0.1 AND LAM < 0.1: STOCHASTIC
- DET < 0.1 AND LAM > 0.2: TRAPPED_STOCHASTIC

Entity stability score = 0.7 * lyapunov_sigmoid + 0.3 * mean_determinism.

### 9.3 Regime Transitions

**File**: `orthon/sql/layers/31_regime_transitions.sql`

Stability proxies: coherence_velocity = d(coherence)/dI, state_acceleration = d^2(effective_dim)/dI^2.

Regime states: DESTABILIZING (losing coherence, state growing), STABILIZING, STABLE, TRANSIENT.

### 9.4 Basin Stability

**File**: `orthon/sql/layers/32_basin_stability.sql`

Sigmoid-normalized scoring with fleet-relative comparison:
basin_score = 0.30*coherence + 0.30*velocity + 0.20*coherence_stability + 0.20*velocity_stability.

### 9.5 Topology Health

**File**: `orthon/sql/layers/40_topology_health.sql`

Betti number interpretation:
- beta_0 > 1: FRAGMENTED (critical)
- beta_1 = 0: COLLAPSED (warning)
- beta_1 = 1: HEALTHY_CYCLE
- beta_1 = 2: QUASI_PERIODIC
- beta_1 > 2: COMPLEX

### 9.6 Information Health

**File**: `orthon/sql/layers/50_information_health.sql`

Hierarchy score = (n_nodes - n_feedback_loops) / n_nodes:
- > 0.8: HIERARCHICAL (healthy, clear DAG)
- > 0.5: MIXED
- > 0.2: COUPLED
- else: CIRCULAR (critical, cascade risk)

### 9.7 Calculus Foundation

**File**: `orthon/sql/layers/01_calculus.sql`

SQL window functions compute derivatives directly:
- v_dy: velocity via central differences
- v_d2y: acceleration
- v_d3y: jerk
- v_curvature: |d2y| / (1 + dy^2)^(3/2)
- v_arc_length: cumulative path length

### 9.8 Causality Analysis

**File**: `orthon/sql/layers/04_causality.sql`

SQL-based Granger proxy, transfer entropy proxy via binned approximation, causal role identification (SOURCE, SINK, CONDUIT, DRIVER, FOLLOWER, ISOLATED).

### 9.9 ML Feature Export

**File**: `orthon/sql/ml/11_ml_features.sql`

60+ features per entity for ML consumption: signal-level, entity-level, pairwise, dynamics, causality, and clustering metrics.

---

## 10. ORTHON Interpretation Methods

### 10.1 Physics Interpreter (Symplectic Structure Loss)

**File**: `orthon/services/physics_interpreter.py`

Four-layer degradation detection framework:

**L4: Thermodynamics** - Is energy conserved?
- Conservation check: CV(energy) < 0.1 AND |energy_trend| < 0.01 * |energy_mean|
- Dissipation rate: max(0, -dE/dI)

**L3: Mechanics** - Where is energy flowing?
- Flow asymmetry via Gini coefficient of signal energies
- Energy sources/sinks per signal

**L2: Coherence** - Through what couplings?
- Spectral coherence: lambda_1 / sum(lambda)
- Decoupling detection: coherence_trend < -0.001 OR coherence < 0.8 * baseline

**L1: State** - Resulting phase space position?
- State distance: ||phi(current) - phi(baseline)||
- State velocity: d(state_distance)/dI

**The Orthon Signal** (degradation signature):
```
ORTHON_signal = energy_dissipating AND (decoupling OR fragmenting) AND state_diverging
```
When all three conditions occur simultaneously: CRITICAL severity.

### 10.2 Dynamics Interpreter

**File**: `orthon/services/dynamics_interpreter.py`

**Basin Stability Score** (0-1):
```
score = 0.30 * coherence_score + 0.30 * velocity_score
      + 0.20 * coherence_stability + 0.20 * velocity_stability

Where:
  coherence_score = 1 / (1 + exp(-10 * (mean_coherence - 0.5)))
  velocity_score  = 1 / (1 + exp(10 * (mean_velocity - 0.3)))
```

Classification: DEEP_BASIN (>0.7), MODERATE_BASIN (0.5-0.7), SHALLOW_BASIN (0.3-0.5), UNSTABLE (<0.3).

**Birth Certificate** (early-life prognosis from first 20% of data):
```
score = 0.4 * coupling + 0.4 * stability + 0.2 * consistency
```
Grades: EXCELLENT (>0.65), GOOD (0.5-0.65), FAIR (0.35-0.5), POOR (<0.35).

### 10.3 Tipping Classification

**File**: `orthon/engines/tipping_engine.py`

| Tipping Type | Causality Direction | Early Warning? |
|-------------|-------------------|----------------|
| B-Tipping (Bifurcation) | Geometry -> Mass | YES (CSD detectable) |
| R-Tipping (Rate-induced) | Mass -> Geometry | NO |
| Resonance (Bidirectional) | Both directions | YES (coupling increase) |

### 10.4 Diagnostic Report (9 Levels)

**File**: `orthon/engines/diagnostic_report.py`

| Level | Assessment | Source |
|-------|-----------|--------|
| 0 | Typology classification | Signal type |
| 1 | Stationarity | ADF/KPSS tests |
| 2 | Signal classification | 10 dimensions |
| 3 | Geometry (eigenstructure) | Effective dim, alignment |
| 4 | Mass (energy) | Total variance trends |
| 5 | Structure (geometry x mass) | Compression, absorption |
| 6 | Stability | Lyapunov, CSD |
| 7 | Tipping | B/R/Resonance |
| 8 | Spin Glass (disorder) | Parisi overlap, magnetization |

Health score = 0.3 * eff_dim_ratio + 0.25 * stability + 0.25 * absorption + 0.2 * pattern.

---

## 11. Machine Learning Feature Engineering and Regression

### 11.1 LASSO Feature Selection

**File**: `orthon/ml/lasso.py`

L1-regularized regression drives unimportant features to exactly zero:

**Objective**: min (1/2n) ||y - X*beta||^2 + alpha * ||beta||_1

**Coordinate Descent Solver**:
```
For each feature j:
    beta_j = soft_threshold(X_j^T(y - X*beta + X_j*beta_j) / ||X_j||^2, alpha / ||X_j||^2)

soft_threshold(z, lambda) = sign(z) * max(|z| - lambda, 0)
```

**Alpha Selection via Cross-Validation**:
- alpha_max = max(|X^T * y|) / n_samples (smallest alpha where all coefficients = 0)
- alpha_min = alpha_max * 0.001
- Grid: logspace(log10(alpha_max), log10(alpha_min), n_alphas)
- Select alpha with minimum cross-validation MSE

### 11.2 RUL Prediction

**File**: `orthon/prediction/rul.py`

**Models**: Random Forest (100 trees, max_depth=10) or Gradient Boosting (100 estimators, learning_rate=0.1)

**Default Features** (14):
- Physics: effective_dim, state_velocity, entropy, free_energy
- Primitives: rms, kurtosis, skewness, crest_factor, hurst, sample_entropy
- Dynamics: lyapunov_exponent, correlation_dim, recurrence_rate, determinism

**NASA Scoring Function**:
```
s_i = exp(-d_i / 13) - 1  if d_i < 0 (early prediction, less penalized)
s_i = exp(d_i / 10) - 1   if d_i >= 0 (late prediction, heavily penalized)
```

### 11.3 Health Scoring

**File**: `orthon/prediction/health.py`

Weighted combination:
```
health = 0.40 * statistical_health + 0.35 * dynamic_health + 0.25 * structural_health
```

- Statistical: 100 * (1 / (1 + exp(mean_zscore_deviation - 2)))
- Dynamic: Lyapunov-based: 100 * (1 / (1 + exp(lambda * 10)))
- Structural: Persistence entropy from topology

### 11.4 Anomaly Detection

**File**: `orthon/prediction/anomaly.py`

Four methods:
1. **Z-Score**: max per-feature z-score > threshold (default 3)
2. **Isolation Forest**: Tree-based isolation (contamination=0.1)
3. **Local Outlier Factor**: Density-based (n_neighbors=20)
4. **Combined**: Majority voting ensemble

### 11.5 Early Failure Fingerprinting

**File**: `orthon/early_warning/failure_fingerprint_detector.py`

**Core Discovery**: Engines that fail atypically show inverted derivative patterns in the first 10% of life.

**Smoking Gun Rule** (from C-MAPSS analysis):
```
sensor_11_early_d1 < -0.004 AND sensor_14_early_d1 > 0.02
=> 67% precision, 100% recall for early failure
```

**Algorithm**:
1. Compute per-signal first-derivative statistics in first 10% of life
2. Compute cross-signal d1 correlations
3. Compare to population baseline via z-scores
4. Apply rule-based risk stratification

### 11.6 ML-Based Failure Prediction

**File**: `orthon/early_warning/ml_predictor.py`

Three models:
1. **Early Failure Classifier**: GradientBoosting (P(early_failure))
2. **Lifecycle Regressor**: GradientBoosting (predicted total cycles)
3. **Atypical Detection**: Mahalanobis distance D^2 = (x - mu)^T * Sigma^-1 * (x - mu)

**Risk Score**: 0.4 * early_prob + 0.3 * atypical_score + 0.3 * lifecycle_risk

---

## 12. Additional Computational Primitives

### 12.1 Individual Signal Primitives (24 files)

**Directory**: `engines/primitives/individual/`

| File | Equations |
|------|-----------|
| calculus.py | Derivatives (central differences), integrals (trapezoidal), curvature |
| statistics.py | Mean, variance, skewness, kurtosis, crest factor, zero crossings |
| correlation.py | Autocorrelation, partial autocorrelation (Yule-Walker), ACF decay |
| entropy.py | Permutation entropy, sample entropy, approximate entropy, Lempel-Ziv complexity |
| spectral.py | FFT, PSD (Welch), dominant frequency, spectral centroid, bandwidth, entropy |
| hilbert.py | Analytic signal, envelope, instantaneous frequency, instantaneous phase |
| fractal.py | Hurst exponent (R/S), DFA (detrended fluctuation analysis) |
| stationarity.py | ADF test, KPSS test, trend detection, changepoint detection |
| similarity.py | Cosine similarity, Euclidean distance, DTW, correlation, cross-correlation |
| geometry.py | Covariance matrix, eigendecomposition, effective dimension, condition number |
| information.py | Transfer entropy, conditional entropy, Granger causality, phase coupling |
| memory.py | Rescaled range, long-range correlation, variance growth |
| normalization.py | Z-score, robust (IQR), MAD, min-max, quantile normalization |
| dynamics.py | Phase space reconstruction, velocity, acceleration |
| complexity.py | Multiscale entropy, entropy rate |
| temporal.py | Turning points, trend fit, rate of change |
| derivatives.py | Higher-order derivatives |

### 12.2 Embedding Primitives

**File**: `engines/primitives/embedding/delay.py`

- Time delay embedding (Takens' theorem)
- Optimal delay via AMI (Average Mutual Information)
- Optimal dimension via Cao's method and FNN (False Nearest Neighbors)
- Multivariate embedding

### 12.3 Topology Primitives

**Files**: `engines/primitives/topology/persistence.py`, `distance.py`

- Persistent homology via Vietoris-Rips filtration
- Betti numbers (H0: connected components, H1: loops, H2: voids)
- Persistence entropy: H = -sum(p_i * ln(p_i)) where p_i = persistence_i / sum(persistence)
- Wasserstein distance between persistence diagrams
- Bottleneck distance (max-cost matching)

### 12.4 Network Primitives

**Directory**: `engines/primitives/network/`

| File | Algorithms |
|------|-----------|
| centrality.py | Degree, betweenness (Brandes), eigenvector (power iteration), closeness |
| community.py | Modularity optimization, Louvain, spectral clustering, label propagation |
| paths.py | Floyd-Warshall, Dijkstra, diameter, eccentricity, radius |
| structure.py | Density, clustering coefficient, connected components, assortativity |

### 12.5 Matrix Primitives

**Directory**: `engines/primitives/matrix/`

| File | Content |
|------|---------|
| covariance.py | Covariance and correlation matrices |
| decomposition.py | Eigendecomposition, SVD, PCA loadings, factor scores |
| dmd.py | Dynamic Mode Decomposition (Schmid 2010) |
| graph.py | Distance matrix, adjacency matrix, Laplacian, recurrence matrix |
| information.py | MI matrix, TE matrix, Granger causality matrix |

### 12.6 Fingerprinting

**Files**: `engines/fingerprint/engines/gaussian.py`, `similarity.py`

**Gaussian Fingerprinting**: Summarize entity behavior as multivariate Gaussian (mean vector + covariance matrix).

**Bhattacharyya Similarity**:
$$D_B = \frac{1}{8}(\mu_1 - \mu_2)^T \Sigma^{-1} (\mu_1 - \mu_2) + \frac{1}{2}\ln\frac{|\Sigma|}{|\Sigma_1|^{1/2} |\Sigma_2|^{1/2}}$$

Where Sigma = (Sigma_1 + Sigma_2) / 2. Captures both mean and variance differences.

### 12.7 Normalization Recommendation

**File**: `engines/manifold/normalization.py`

Automatic method selection:
- outlier_fraction > 5%: MAD normalization
- kurtosis > 10: MAD normalization
- kurtosis > 3: Robust (IQR) normalization
- skewness > 1: Robust normalization
- Otherwise: Z-score normalization

---

## 13. System Boundaries: ORTHON vs. Manifold

### Clear Responsibility Division

| Responsibility | ORTHON (Brain) | Manifold/Engines (Muscle) |
|---------------|---------------|--------------------------|
| Signal classification | YES - 10 dimensions, 27 measures | NO |
| Engine selection | YES - manifest.yaml generation | NO - reads manifest |
| Window sizing | YES - characteristic time computation | NO - uses manifest windows |
| Feature computation | NO | YES - 42+ signal engines |
| Eigendecomposition | NO | YES - SVD, effective_dim |
| Lyapunov/FTLE | NO | YES - Rosenstein's algorithm |
| RQA | NO | YES - recurrence matrices |
| Classification of results | YES - SQL views | NO - returns raw numbers |
| Health scoring | YES - unified score | NO |
| RUL prediction | YES - ML models | NO |
| Anomaly detection | YES - ensemble methods | NO |
| Physics interpretation | YES - symplectic loss | NO |
| Visualization | YES - explorer, flow-viz | NO |
| Data validation | YES - schema, I column | NO |
| Typology creation | YES | NO (Engines reads it) |

### Data Flow Contract

1. ORTHON creates: observations.parquet, typology.parquet, manifest.yaml
2. Engines reads: observations.parquet, typology.parquet, manifest.yaml
3. Engines produces: 14 output parquet files
4. ORTHON classifies: SQL views on Engines outputs

### Architectural Invariants

- Engines contains NO classification logic, NO thresholds, NO domain-specific rules
- ORTHON contains NO numerical computation (except typology's 27 measures)
- All output paths are FIXED (never change)
- observations.parquet always goes to $ENGINES_DATA_DIR (no subdirectories)
- cohort is a grouping key, NOT a compute key
- Scale-invariant features only (no absolute values like rms, mean, std)
- Lyapunov for chaos detection, NOT coefficient of variation
- Insufficient data returns NaN, never skips

---

## 14. Annotated File Trees

### 14.1 ORTHON Repository

```
~/orthon/
├── CLAUDE.md                                    # AI instructions and architecture documentation
├── README.md                                    # Project overview
├── pyproject.toml                               # Package definition (PyPI-ready)
├── Dockerfile                                   # Explorer container definition
├── fly.toml                                     # Fly.io deployment config
│
├── orthon/
│   ├── __init__.py                              # Package initialization
│   ├── cli.py                                   # Main CLI entry point
│   │
│   ├── entry_points/                            # Pipeline stage orchestrators
│   │   ├── __init__.py                          # Stage imports
│   │   ├── stage_01_validate.py                 # Validate observations (remove constants, duplicates)
│   │   ├── stage_02_typology.py                 # Compute 27 raw typology measures per signal
│   │   ├── stage_03_classify.py                 # Apply two-stage classification (discrete/sparse + continuous)
│   │   ├── stage_04_manifest.py                 # Generate manifest.yaml with engine selection + windowing
│   │   ├── stage_05_diagnostic.py               # Run full diagnostic assessment (9 levels)
│   │   ├── stage_06_interpret.py                # Interpret Engines outputs (physics + dynamics)
│   │   ├── stage_07_predict.py                  # Predict RUL, health, anomalies
│   │   ├── stage_08_alert.py                    # Early warning / failure fingerprints
│   │   ├── stage_09_explore.py                  # Manifold visualization
│   │   ├── stage_10_inspect.py                  # File inspection / capability detection
│   │   ├── stage_11_fetch.py                    # Read, profile, validate raw data
│   │   ├── stage_12_stream.py                   # Real-time streaming analysis
│   │   ├── stage_13_train.py                    # Train ML models on Engines features
│   │   └── csv_to_atlas.py                      # One-command CSV-to-Atlas pipeline
│   │
│   ├── config/                                  # Configuration (all thresholds, no magic numbers)
│   │   ├── typology_config.py                   # PR4 continuous classification thresholds
│   │   ├── discrete_sparse_config.py            # PR5 discrete/sparse detection thresholds
│   │   ├── stability_config.py                  # Stability assessment configuration
│   │   ├── domains.py                           # Physics domain definitions (7 domains)
│   │   ├── recommender.py                       # Engine recommendation logic
│   │   └── engine_rules.yaml                    # Engine selection rules
│   │
│   ├── typology/                                # Signal classification logic
│   │   ├── level2_corrections.py                # Config-driven continuous classification (PR4)
│   │   ├── discrete_sparse.py                   # Discrete/sparse detection (PR5)
│   │   ├── constant_detection.py                # CV-based constant signal detection
│   │   ├── control_detection.py                 # Control signal detection
│   │   ├── classification_stability.py          # Classification stability assessment
│   │   ├── window_factor.py                     # Window factor computation per signal type
│   │   └── tests/                               # Classification tests
│   │
│   ├── manifest/                                # Manifest generation system
│   │   ├── generator.py                         # v2.5 manifest: engine gating, per-engine windows
│   │   ├── characteristic_time.py               # Data-driven characteristic time computation
│   │   ├── system_window.py                     # Multi-scale system window computation
│   │   ├── domain_clock.py                      # Domain-specific timing
│   │   ├── window_recommender.py                # Window recommendation logic
│   │   └── tests/                               # Manifest tests
│   │
│   ├── ingest/                                  # Data ingestion and validation
│   │   ├── typology_raw.py                      # Computes 27 raw measures per signal
│   │   ├── validate_observations.py             # Validates & repairs observations.parquet
│   │   ├── data_reader.py                       # Read CSV/parquet/TSV with profiling
│   │   ├── validation.py                        # SignalValidator, ValidationConfig
│   │   ├── schema_enforcer.py                   # Schema validation and enforcement
│   │   ├── normalize.py                         # Data normalization
│   │   ├── transform.py                         # Data transformation utilities
│   │   ├── paths.py                             # FIXED output path definitions
│   │   ├── streaming.py                         # Streaming ingestion
│   │   └── upload.py                            # Upload handling
│   │
│   ├── services/                                # Interpreters and orchestration
│   │   ├── physics_interpreter.py               # Symplectic structure loss detection (L4-L1)
│   │   ├── dynamics_interpreter.py              # Lyapunov, basin stability, regime, birth certificate
│   │   ├── state_analyzer.py                    # State velocity/acceleration anomaly detection
│   │   ├── fingerprint_service.py               # Healthy/deviation/failure fingerprint matching
│   │   ├── tuning_service.py                    # AI-guided threshold optimization
│   │   ├── concierge.py                         # Natural language interface
│   │   └── job_manager.py                       # Job lifecycle management
│   │
│   ├── engines/                                 # ORTHON diagnostic engines (not Manifold)
│   │   ├── diagnostic_report.py                 # Full 9-level diagnostic pipeline
│   │   ├── typology_engine.py                   # Level 0: Typology assessment
│   │   ├── stationarity_engine.py               # Level 1: Stationarity assessment
│   │   ├── classification_engine.py             # Level 2: Classification assessment
│   │   ├── stability_engine.py                  # Level 6: Lyapunov + CSD detection
│   │   ├── tipping_engine.py                    # Level 7: B/R/Resonance tipping classification
│   │   ├── spin_glass.py                        # Level 8: Spin glass disorder model
│   │   ├── mass_engine.py                       # Level 4: Mass (energy) assessment
│   │   ├── structure_engine.py                  # Level 5: Structure assessment
│   │   ├── signal_geometry.py                   # Signal-level geometry
│   │   └── trajectory_monitor.py                # Trajectory monitoring
│   │
│   ├── prediction/                              # Predictive models
│   │   ├── rul.py                               # RUL prediction (Random Forest, Gradient Boosting)
│   │   ├── health.py                            # Health scoring (0-100, 3-component weighted)
│   │   ├── anomaly.py                           # Multi-method anomaly detection (z-score, IF, LOF)
│   │   ├── base.py                              # Base predictor class
│   │   └── cli.py                               # Prediction CLI
│   │
│   ├── early_warning/                           # Early failure detection
│   │   ├── failure_fingerprint_detector.py      # Early-life d1 pattern matching ("smoking gun")
│   │   └── ml_predictor.py                      # GradientBoosting + Mahalanobis atypical detection
│   │
│   ├── ml/                                      # ML training pipeline
│   │   ├── lasso.py                             # LASSO feature selection (coordinate descent)
│   │   └── entry_points/                        # ML entry points
│   │       ├── train.py                         # Model training
│   │       ├── predict.py                       # Model prediction
│   │       ├── features.py                      # Feature extraction
│   │       ├── ablation.py                      # Feature ablation study
│   │       ├── benchmark.py                     # Performance benchmarking
│   │       └── baseline.py                      # Baseline model comparison
│   │
│   ├── sql/                                     # SQL classification layer (52+ files)
│   │   ├── layers/                              # Core classification logic (27 files)
│   │   │   ├── classification.sql               # Master: v_trajectory_type, v_system_health, etc.
│   │   │   ├── 00_load.sql                      # Data loading
│   │   │   ├── 00_observations.sql              # Observation views
│   │   │   ├── 00_config.sql                    # Configuration
│   │   │   ├── 00_configuration_audit.sql       # Audit views
│   │   │   ├── 00_index_detection.sql           # Index detection
│   │   │   ├── 01_calculus.sql                  # Derivatives, curvature, arc length
│   │   │   ├── 01_typology.sql                  # Typology classification via SQL
│   │   │   ├── 02_geometry.sql                  # Correlation, lagged correlation
│   │   │   ├── 02_statistics.sql                # Statistical summaries
│   │   │   ├── 03_dynamics.sql                  # Regime detection, attractor identification
│   │   │   ├── 03_signal_class.sql              # Signal class (ANALOG, DIGITAL, EVENT)
│   │   │   ├── 04_causality.sql                 # Granger proxy, transfer entropy proxy
│   │   │   ├── 08_entropy.sql                   # Entropy calculations
│   │   │   ├── 30_dynamics_stability.sql        # Lyapunov + RQA classification
│   │   │   ├── 31_regime_transitions.sql        # Temporal degradation detection
│   │   │   ├── 32_basin_stability.sql           # Perturbation tolerance scoring
│   │   │   ├── 40_topology_health.sql           # Betti number interpretation
│   │   │   ├── 50_information_health.sql        # Causal network hierarchy scoring
│   │   │   ├── atlas_analytics.sql              # Atlas analytics views
│   │   │   ├── atlas_breaks.sql                 # Break classification
│   │   │   ├── atlas_ftle.sql                   # FTLE classification
│   │   │   ├── atlas_ridge_proximity.sql        # Ridge proximity views
│   │   │   ├── atlas_topology.sql               # Topology atlas views
│   │   │   ├── atlas_velocity_field.sql         # Velocity field views
│   │   │   ├── break_classification.sql         # Break type classification
│   │   │   ├── constants_units.sql              # Physical constants
│   │   │   └── typology_v2.sql                  # Typology v2 views
│   │   │
│   │   ├── views/                               # Dashboard views (6 files)
│   │   │   ├── 01_classification_units.sql      # Classification per unit
│   │   │   ├── 02_work_orders.sql               # Work order generation
│   │   │   ├── 04_visualization.sql             # Visualization data
│   │   │   ├── 05_summaries.sql                 # Summary statistics
│   │   │   └── 06_general_views.sql             # General purpose views
│   │   │
│   │   ├── reports/                             # Deep analysis reports (20 files)
│   │   │   ├── 00_run_all.sql                   # Execute all reports
│   │   │   ├── 01_baseline_geometry.sql         # Baseline geometry analysis
│   │   │   ├── 02_stable_baseline.sql           # Stable baseline discovery
│   │   │   ├── 03_drift_detection.sql           # Drift detection report
│   │   │   ├── 04_signal_ranking.sql            # Signal importance ranking
│   │   │   ├── 05_periodicity.sql               # Periodicity analysis
│   │   │   ├── 06_regime_detection.sql          # Regime detection report
│   │   │   ├── 07_correlation_changes.sql       # Temporal correlation changes
│   │   │   ├── 08_lead_lag.sql                  # Lead-lag relationship analysis
│   │   │   ├── 09_causality_influence.sql       # Causal influence mapping
│   │   │   ├── 10_process_health.sql            # Process health assessment
│   │   │   ├── 33_birth_certificate.sql         # Entity birth certificate (early-life prognosis)
│   │   │   ├── 60_ground_truth.sql              # Ground truth comparison
│   │   │   ├── 61_lead_time_analysis.sql        # Lead time before failure
│   │   │   ├── 62_fault_signatures.sql          # Fault signature extraction
│   │   │   └── 63_threshold_optimization.sql    # Threshold optimization
│   │   │
│   │   ├── stages/                              # Reporting stages (6 files)
│   │   │   ├── 01_typology.sql                  # Typology stage reporting
│   │   │   ├── 02_signal_vector.sql             # Signal vector reporting
│   │   │   ├── 03_state_vector.sql              # State vector reporting
│   │   │   ├── 04_geometry.sql                  # Geometry reporting
│   │   │   ├── 05_dynamics.sql                  # Dynamics reporting
│   │   │   └── 06_physics.sql                   # Physics reporting
│   │   │
│   │   └── ml/                                  # ML feature extraction (2 files)
│   │       ├── 11_ml_features.sql               # 60+ ML features per entity
│   │       └── 26_ml_feature_export.sql         # Feature export for ML pipelines
│   │
│   ├── explorer/                                # Browser-based visualization
│   │   ├── server.py                            # HTTP server (0.0.0.0 for containers)
│   │   ├── cli.py                               # Explorer CLI
│   │   ├── loader.py                            # DuckDB data loader
│   │   ├── renderer.py                          # 2D/3D rendering engine
│   │   └── static/                              # Frontend assets
│   │       ├── index.html                       # SQL query interface (DuckDB-WASM)
│   │       ├── explorer.html                    # Pipeline data browser
│   │       ├── flow_viz.html                    # Flow visualization (3-act structure)
│   │       ├── atlas.html                       # Dynamical atlas viewer
│   │       ├── wizard.html                      # Setup wizard
│   │       ├── flow-viz.js                      # Flow visualization logic
│   │       └── orthon-queries.js                # Pre-built SQL queries
│   │
│   ├── cohorts/                                 # Cohort discovery and baseline
│   │   ├── detection.py                         # Auto signal classification (CONSTANT/SYSTEM/COMPONENT/ORPHAN)
│   │   ├── baseline.py                          # 5 baseline modes (first_n_percent, stable_discovery, etc.)
│   │   └── discovery.py                         # Cohort structure discovery
│   │
│   ├── streaming/                               # Real-time analysis
│   │   ├── analyzers.py                         # Progressive eigenanalysis
│   │   ├── data_sources.py                      # Data source connectors
│   │   ├── websocket_server.py                  # WebSocket server for live data
│   │   └── cli.py                               # Streaming CLI
│   │
│   ├── inspection/                              # Data inspection
│   │   ├── file_inspector.py                    # File format detection and profiling
│   │   ├── capability_detector.py               # Analysis capability detection
│   │   └── results_validator.py                 # Results validation
│   │
│   ├── core/                                    # API and pipeline orchestration
│   │   ├── pipeline.py                          # Main pipeline orchestrator
│   │   ├── api.py                               # REST API
│   │   ├── server.py                            # API server
│   │   └── validation.py                        # Input validation
│   │
│   ├── shared/                                  # Shared constants and configuration
│   │   ├── physics_constants.py                 # Physical constants and units
│   │   ├── engine_registry.py                   # Engine registry
│   │   └── window_config.py                     # Window configuration
│   │
│   ├── state/                                   # State management
│   │   └── classification.py                    # State classification tracking
│   │
│   └── utils/                                   # Utilities
│       └── index_detection.py                   # Index column detection
│
├── fetchers/                                    # Data acquisition (14 domain fetchers)
│   ├── cmapss_fetcher.py                        # C-MAPSS turbofan degradation
│   ├── tep_fetcher.py                           # Tennessee Eastman Process
│   ├── cwru_bearing_fetcher.py                  # CWRU bearing fault data
│   ├── hydraulic_fetcher.py                     # Hydraulic condition monitoring
│   ├── femto_fetcher.py                         # FEMTO bearing run-to-failure
│   ├── nasa_bearing_fetcher.py                  # NASA bearing degradation
│   └── ... (8 more fetchers)                    # Various domain data sources
│
├── scripts/                                     # Utility scripts
│   ├── process_all_domains.py                   # Batch domain processing pipeline
│   ├── regenerate_manifests.py                  # Regenerate all manifests to v2.5
│   ├── analyze.py                               # Analysis utilities
│   └── test_pipeline.py                         # Pipeline testing
│
├── data/                                        # Demo TEP data
│   ├── manifest.yaml                            # Demo manifest
│   └── (parquet files)                          # Demo observations
│
└── tests/                                       # Tests
    ├── test_typology_benchmarks.py              # Typology classification benchmarks
    ├── test_state_classification.py             # State classification tests
    └── test_window_recommender.py               # Window recommendation tests
```

### 14.2 Manifold/Engines Repository

```
~/manifold/engines/
├── __main__.py                                  # CLI entry point
├── cli.py                                       # Command-line interface
│
├── engines/
│   ├── vector/                                  # Signal Vector computation
│   │   ├── engines/
│   │   │   ├── shape.py                         # Kurtosis, skewness, crest_factor
│   │   │   ├── complexity.py                    # Permutation entropy, sample entropy, hurst, ACF
│   │   │   ├── spectral.py                      # Spectral slope, dominant_freq, entropy, centroid
│   │   │   └── harmonic.py                      # Fundamental frequency, harmonics, THD
│   │   └── run.py                               # Signal vector orchestration
│   │
│   ├── manifold/                                # State/Geometry computation
│   │   ├── state/
│   │   │   ├── centroid.py                      # Mean of signal features (state vector)
│   │   │   └── eigendecomp.py                   # SVD eigendecomposition + continuity enforcement
│   │   ├── state_geometry.py                    # Per-engine eigenvalues, effective_dim per index
│   │   ├── signal_geometry.py                   # Distance, coherence, contribution, residual per signal
│   │   ├── signal_pairwise.py                   # Pairwise: distance, correlation, cosine, co-loading
│   │   ├── geometry_dynamics.py                 # Velocity, acceleration, jerk, curvature of state
│   │   ├── normalization.py                     # Z-score, robust, MAD, minmax with recommendation
│   │   ├── rolling.py                           # Generic rolling window engine
│   │   ├── registry.py                          # Engine discovery, lazy loading, config management
│   │   ├── base.py                              # Base engine class with self-configuration
│   │   ├── signal/                              # 42 individual signal engines
│   │   │   ├── spectral.py                      # FFT-based spectral features
│   │   │   ├── harmonics.py                     # Harmonic analysis (fundamental, THD, HNR)
│   │   │   ├── entropy.py                       # Shannon, Renyi entropy
│   │   │   ├── complexity.py                    # Sample entropy, permutation entropy
│   │   │   ├── hurst.py                         # Hurst exponent (R/S method)
│   │   │   ├── lyapunov.py                      # Lyapunov exponent (Rosenstein)
│   │   │   ├── rqa.py                           # Recurrence Quantification Analysis
│   │   │   ├── attractor.py                     # Correlation dimension, attractor type
│   │   │   ├── basin.py                         # Basin stability metrics
│   │   │   ├── trend.py                         # Trend strength, R-squared
│   │   │   ├── variance_ratio.py                # Variance ratio test
│   │   │   ├── variance_growth.py               # Scale-dependent variance growth
│   │   │   ├── rate_of_change.py                # First-derivative statistics
│   │   │   ├── statistics.py                    # Kurtosis, skewness, crest factor
│   │   │   ├── memory.py                        # ACF, hurst, DFA
│   │   │   ├── adf.py                           # Augmented Dickey-Fuller test
│   │   │   ├── snr.py                           # Signal-to-noise ratio
│   │   │   ├── frequency_bands.py               # Band-pass energy decomposition
│   │   │   ├── fundamental_freq.py              # Fundamental frequency detection
│   │   │   ├── phase_coherence.py               # Phase-locking value
│   │   │   ├── thd.py                           # Total Harmonic Distortion
│   │   │   ├── dmd.py                           # Dynamic Mode Decomposition
│   │   │   ├── garch.py                         # GARCH(1,1) volatility model
│   │   │   ├── hmm.py                           # Hidden Markov Model
│   │   │   ├── hilbert_stability.py             # Instantaneous frequency stability
│   │   │   ├── wavelet_stability.py             # Time-frequency energy decomposition
│   │   │   ├── physics_stack.py                 # Four-layer physics (state/coherence/energy/thermo)
│   │   │   ├── envelope.py                      # Hilbert envelope
│   │   │   ├── peak.py                          # Peak magnitude
│   │   │   ├── pulsation_index.py               # Flow variability metric
│   │   │   ├── cycle_counting.py                # Rainflow counting for fatigue
│   │   │   ├── dwell_times.py                   # State holding duration
│   │   │   ├── level_count.py                   # Occupied state-space analysis
│   │   │   ├── level_histogram.py               # State distribution shape
│   │   │   ├── transition_matrix.py             # Markov transition probabilities
│   │   │   ├── time_constant.py                 # Exponential response fitting
│   │   │   ├── lof.py                           # Local Outlier Factor
│   │   │   ├── rms.py                           # Root-mean-square
│   │   │   └── (additional engines)             # Additional specialized engines
│   │   │
│   │   ├── dynamics/                            # Dynamics computation
│   │   │   ├── ftle.py                          # Finite-Time Lyapunov Exponents (Rosenstein + Cao)
│   │   │   ├── lyapunov.py                      # Lyapunov exponent (Rosenstein/Kantz)
│   │   │   ├── attractor.py                     # Correlation dimension, RQA
│   │   │   ├── critical_slowing_down.py         # CSD composite score for bifurcation warning
│   │   │   ├── trajectory_sensitivity.py        # Local Jacobian, variable sensitivity ranking
│   │   │   ├── saddle_detection.py              # Jacobian eigenvalue analysis, basin boundaries
│   │   │   └── formal_definitions.py            # Enums: AttractorType, StabilityType, FailureMode
│   │   │
│   │   └── geometry/
│   │       └── config.py                        # Feature groups for geometry decomposition
│   │
│   ├── dynamics/                                # Dynamics engines
│   │   ├── run.py                               # Dynamics orchestration
│   │   └── engines/
│   │       ├── velocity.py                      # Speed, acceleration, curvature
│   │       ├── ftle.py                          # FTLE with forward/backward direction
│   │       ├── ftle_rolling.py                  # Time-varying FTLE field
│   │       ├── ridge.py                         # FTLE gradient, urgency, time-to-ridge
│   │       └── breaks.py                        # CUSUM break detection
│   │
│   ├── decompose/                               # Eigendecomposition engines
│   │   ├── run.py                               # Decomposition orchestration
│   │   └── engines/
│   │       ├── eigen.py                         # SVD eigendecomposition
│   │       ├── effective_dim.py                 # Participation ratio, spectral entropy
│   │       ├── condition.py                     # Condition number, spectral gaps
│   │       └── thermodynamics.py                # Temperature, free energy, heat capacity
│   │
│   ├── pairwise/                                # Pairwise signal engines
│   │   └── engines/
│   │       ├── cointegration.py                 # ADF on residuals, hedge ratio, half-life
│   │       ├── copula.py                        # Gaussian, Clayton, Gumbel, Frank copulas
│   │       ├── correlation.py                   # Pearson, Spearman, partial correlation
│   │       ├── distance.py                      # Euclidean, Manhattan, DTW, cosine
│   │       ├── information.py                   # Granger causality, transfer entropy, KL/JS divergence
│   │       └── topology.py                      # Persistence diagram comparison
│   │
│   ├── fingerprint/                             # Entity fingerprinting
│   │   └── engines/
│   │       ├── gaussian.py                      # Multivariate Gaussian fingerprint
│   │       └── similarity.py                    # Bhattacharyya similarity
│   │
│   ├── primitives/                              # Pure mathematical building blocks
│   │   ├── individual/                          # Single-signal primitives (17 files)
│   │   │   ├── calculus.py                      # Derivatives, integrals, curvature
│   │   │   ├── statistics.py                    # Moments, peak-to-peak, zero crossings
│   │   │   ├── correlation.py                   # Autocorrelation, PACF
│   │   │   ├── entropy.py                       # Permutation, sample, approximate, multiscale
│   │   │   ├── spectral.py                      # FFT, PSD, centroid, bandwidth
│   │   │   ├── hilbert.py                       # Analytic signal, envelope, inst. frequency
│   │   │   ├── fractal.py                       # Hurst (R/S), DFA
│   │   │   ├── stationarity.py                  # ADF, KPSS, changepoints
│   │   │   ├── similarity.py                    # Cosine, Euclidean, DTW, correlation
│   │   │   ├── geometry.py                      # Covariance, eigendecomp, effective_dim
│   │   │   ├── information.py                   # Transfer entropy, Granger, phase coupling
│   │   │   ├── memory.py                        # Rescaled range, long-range correlation
│   │   │   ├── normalization.py                 # Z-score, robust, MAD, min-max
│   │   │   ├── dynamics.py                      # Phase space, velocity, acceleration
│   │   │   ├── complexity.py                    # Multiscale entropy, entropy rate
│   │   │   ├── temporal.py                      # Turning points, trend fit
│   │   │   └── derivatives.py                   # Higher-order derivatives
│   │   │
│   │   ├── dynamical/                           # Dynamical systems primitives (6 files)
│   │   │   ├── ftle.py                          # FTLE (Cauchy-Green tensor, LCS detection)
│   │   │   ├── lyapunov.py                      # Lyapunov exponent/spectrum, Kaplan-Yorke dim
│   │   │   ├── rqa.py                           # Recurrence matrix, DET, LAM, trapping time
│   │   │   ├── dimension.py                     # Correlation dimension, information dimension
│   │   │   ├── saddle.py                        # Jacobian estimation, saddle scoring
│   │   │   └── sensitivity.py                   # Variable sensitivity, influence matrix
│   │   │
│   │   ├── embedding/                           # Phase space embedding
│   │   │   └── delay.py                         # Cao's method, AMI, FNN, time delay embedding
│   │   │
│   │   ├── information/                         # Information theory (5 files)
│   │   │   ├── entropy.py                       # Shannon, Renyi, Tsallis entropy
│   │   │   ├── mutual.py                        # MI, conditional MI, total correlation
│   │   │   ├── divergence.py                    # KL, JS, Hellinger, total variation
│   │   │   ├── transfer.py                      # Transfer entropy (directional)
│   │   │   └── decomposition.py                 # Partial Information Decomposition (PID)
│   │   │
│   │   ├── pairwise/                            # Pairwise primitives (6 files)
│   │   │   ├── causality.py                     # Granger causality, CCM
│   │   │   ├── correlation.py                   # Pearson, Spearman, cross-correlation, partial
│   │   │   ├── distance.py                      # DTW, Euclidean, cosine, Manhattan
│   │   │   ├── information.py                   # MI and TE between pairs
│   │   │   ├── regression.py                    # Linear regression, ratio, product
│   │   │   └── spectral.py                      # Coherence, phase, wavelet coherence
│   │   │
│   │   ├── matrix/                              # Matrix operations (5 files)
│   │   │   ├── covariance.py                    # Covariance and correlation matrices
│   │   │   ├── decomposition.py                 # Eigendecomp, SVD, PCA, factor scores
│   │   │   ├── dmd.py                           # Dynamic Mode Decomposition
│   │   │   ├── graph.py                         # Distance, adjacency, Laplacian matrices
│   │   │   └── information.py                   # MI, TE, Granger matrices
│   │   │
│   │   ├── topology/                            # Topological data analysis (2 files)
│   │   │   ├── persistence.py                   # Persistence diagrams, Betti numbers, entropy
│   │   │   └── distance.py                      # Wasserstein, bottleneck distance
│   │   │
│   │   ├── network/                             # Network analysis (4 files)
│   │   │   ├── centrality.py                    # Degree, betweenness, eigenvector, closeness
│   │   │   ├── community.py                     # Modularity, Louvain, spectral clustering
│   │   │   ├── paths.py                         # Shortest paths, diameter, eccentricity
│   │   │   └── structure.py                     # Density, clustering coefficient, components
│   │   │
│   │   ├── tests/                               # Statistical test primitives (6 files)
│   │   │   ├── bootstrap.py                     # Bootstrap CI (percentile, BCa, block)
│   │   │   ├── hypothesis.py                    # t-test, F-test, chi-squared, Mann-Whitney
│   │   │   ├── nonparametric.py                 # Mann-Kendall trend test
│   │   │   ├── normalization.py                 # Scaling and normalization
│   │   │   ├── null_models.py                   # Surrogate data, Marchenko-Pastur
│   │   │   └── stationarity_tests.py            # ADF, KPSS, Phillips-Perron
│   │   │
│   │   └── config.py                            # Primitive function configuration
│   │
│   ├── pipeline/                                # Pipeline orchestration
│   │   ├── manifest.py                          # Manifest reading and parsing
│   │   ├── signal_pipeline.py                   # Signal-level pipeline orchestration
│   │   └── cohort_pipeline.py                   # Cohort-level pipeline orchestration
│   │
│   ├── stream/                                  # Streaming infrastructure
│   │   ├── buffer.py                            # Memory-bounded signal buffer
│   │   ├── parser.py                            # Format detection (parquet/CSV)
│   │   ├── protocol.py                          # Work order encoding/decoding
│   │   └── writer.py                            # Incremental parquet writing
│   │
│   ├── validation/                              # Input validation
│   │   ├── input_validation.py                  # Schema validation, CONSTANT signal filtering
│   │   └── prerequisites.py                     # Stage dependency checking
│   │
│   ├── io/                                      # File I/O
│   │   ├── read.py                              # Parquet reading with validation
│   │   └── write.py                             # Safe parquet writing with empty markers
│   │
│   ├── config/                                  # Configuration
│   │   └── metric_requirements.py               # 7-tier minimum sample requirements
│   │
│   └── utils/                                   # Utilities
│       ├── windowing.py                         # Window generation and sizing
│       └── validation.py                        # Input guards and NaN handling
```

---

## Summary of Key Innovations

1. **27-Measure Signal Typology**: Comprehensive statistical characterization across 10 classification dimensions, enabling domain-agnostic signal classification without human intervention.

2. **Two-Stage Classification Pipeline**: Discrete/sparse detection first (CONSTANT, BINARY, DISCRETE, IMPULSIVE, EVENT, STEP, INTERMITTENT), then continuous classification (TRENDING, DRIFTING, PERIODIC, RANDOM, CHAOTIC, QUASI_PERIODIC, STATIONARY).

3. **Characteristic Time Framework**: Data-driven window computation via ACF half-life, dominant frequency period, Hurst exponent, and turning point ratio.

4. **Multi-Scale Representation**: Fast signals use spectral representation, slow signals use trajectory representation, determined by characteristic time ratio to system window.

5. **Typology-Guided Engine Selection**: Inclusive philosophy ("if it's a maybe, run it") with per-engine window overrides when signal window < engine minimum.

6. **Scale-Invariant Feature Engineering**: All features are measurement-unit-independent ratios and normalized quantities.

7. **SVD-Based State Geometry**: Eigendecomposition with eigenvector continuity enforcement (71% correction rate in TEP data).

8. **Eigenvector Co-Loading Gating**: Novel use of PC1 loadings to gate expensive pairwise computations.

9. **Finite-Time Lyapunov Exponents**: Rosenstein's algorithm with Cao's parameter-free embedding dimension detection.

10. **Thermodynamic Mapping**: Projection of eigenvalue dynamics onto statistical mechanics analogues (entropy, energy, temperature, free energy, heat capacity).

11. **Critical Slowing Down Detection**: Composite CSD score for bifurcation early warning.

12. **Unified Health Scoring**: Multi-indicator system health assessment combining Lyapunov, geometry, topology, and information flow.

13. **The Orthon Signal**: Novel degradation signature = dissipating + decoupling + diverging (symplectic structure loss).

14. **Birth Certificate System**: Early-life prognosis from first 20% of data using sigmoid-normalized basin stability.

15. **Early Failure Fingerprinting**: First-derivative pattern matching in early life identifies atypical failure modes before visible degradation.

16. **SQL-Based Dynamical Classification**: 150+ DuckDB views implementing threshold-driven classification of Engines' numerical outputs.

17. **Persistent Homology Health Assessment**: Betti number interpretation for loop collapse and fragmentation detection.

18. **Causal Network Hierarchy Scoring**: Information flow topology assessment for cascade risk identification.

19. **LASSO Feature Selection**: Automated L1-regularized feature elimination via coordinate descent with cross-validation.

20. **Gaussian Fingerprinting with Bhattacharyya Similarity**: Probabilistic entity comparison capturing both mean and variance differences.

---

*End of Patent Technical Specification*
