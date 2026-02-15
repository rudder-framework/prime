# Z-Score Audit: Prime

## Summary
- **Total instances found:** 30
- **Files affected:** 24
- **SQL layers affected:** 7
- **SQL reports affected:** 8
- **SQL views affected:** 2
- **SQL ML files affected:** 2
- **Python files affected:** 9
- **JS files affected:** 2
- **Config files affected:** 1

## Cascade Map

The z-score contamination flows through three main pipelines:

```
PIPELINE 1: Baseline Deviation (the big one)
  23_baseline_deviation.sql  →  computes 10 z-scores + z_total
    ├─ 24_incident_summary.sql  →  consumes z-scores for propagation chains
    ├─ 60_ground_truth.sql      →  consumes z_total for percentile ranking
    ├─ 61_lead_time_analysis.sql →  computes its own z-scores for lead time
    ├─ 62_fault_signatures.sql  →  consumes lead time (sigma-based)
    ├─ 63_threshold_optimization.sql →  optimizes z_total thresholds
    └─ tuning_service.py        →  aggregates detection at 2σ

PIPELINE 2: Signal-Level Anomaly
  02_statistics.sql (v_zscore)  →  global z-score per observation
  classification.sql (v_anomaly_ranked)  →  z-score of spectral_entropy
  12_ranked_derived.sql  →  same z-score for canary detection
  06_general_views.sql (v_anomalies)  →  3σ outlier detection
  06_general_views.sql (v_signal_health)  →  3σ/4σ outlier counting → health_score
  06_general_views.sql (v_entity_comparison)  →  cross-entity z-scores

PIPELINE 3: Fleet-Relative Normalization
  32_basin_stability.sql  →  sigmoid(z-score) for stability scoring
  33_birth_certificate.sql  →  sigmoid(z-score) for birth grading
  concierge.py  →  geometry z-scores for anomaly ranking
  fingerprint_service.py  →  z-scores against healthy baseline
```

---

## Findings

---

### 02_statistics.sql:59-71 — CRITICAL

**Code:**
```sql
CREATE OR REPLACE VIEW v_zscore AS
SELECT
    b.signal_id,
    b.I,
    b.y,
    (b.y - s.y_mean) / NULLIF(s.y_std, 0) AS z_score,
    CASE
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 3 THEN 'extreme'
        WHEN ABS((b.y - s.y_mean) / NULLIF(s.y_std, 0)) > 2 THEN 'outlier'
        ELSE 'normal'
    END AS z_category
FROM v_base b
JOIN v_stats_global s USING (signal_id);
```

**What it does:** Computes a global z-score for every observation point against the signal's overall mean/std. Classifies each point as extreme (>3σ), outlier (>2σ), or normal.

**Impact:** This is the foundational z-score view. Every observation gets a z_score and z_category. The global mean/std treats the entire signal lifetime as stationary — a dying engine's late-life readings are compared to the same mean as its healthy readings, suppressing the very drift we need to detect.

**Data it operates on:** Raw observation values (y) per signal.

**Used downstream by:** Any query that joins v_zscore. Also feeds the anomaly detection pipeline and explorer queries.

---

### classification.sql:201-257 — CRITICAL

**Code:**
```sql
CREATE OR REPLACE VIEW v_anomaly_ranked AS
WITH signal_stats AS (
    SELECT signal_id, cohort,
        AVG(spectral_entropy) AS mean_val,
        CASE WHEN COUNT(*) > 1 AND MAX(spectral_entropy) > MIN(spectral_entropy)
            THEN STDDEV_SAMP(spectral_entropy) ELSE NULL END AS std_val
    FROM signal_vector ...
),
signal_z AS (
    SELECT
        sv.signal_id, sv.cohort, sv.I,
        sv.spectral_entropy AS value,
        (sv.spectral_entropy - ss.mean_val) / NULLIF(ss.std_val, 0) AS z_score
    FROM signal_vector sv
    JOIN signal_stats ss USING (signal_id, cohort)
)
SELECT
    z_score, ABS(z_score) AS z_magnitude,
    RANK() OVER (PARTITION BY cohort, I ORDER BY ABS(z_score) DESC) AS deviation_rank,
    PERCENT_RANK() OVER (PARTITION BY cohort, signal_id ORDER BY ABS(z_score)) AS signal_percentile,
    RANK() OVER (PARTITION BY I ORDER BY ABS(z_score) DESC) AS fleet_deviation_rank
FROM signal_z;
```

**What it does:** Computes z-scores of spectral_entropy per signal per entity. Ranks signals by z-score magnitude within each timestep and across the fleet.

**Impact:** Drives anomaly ranking — which signals are "most anomalous" at each point in time. The z-score uses the full-signal mean/std, which means a gradually degrading signal never looks anomalous because its mean keeps shifting. Ranking by z-score magnitude may produce different orderings than ranking by raw deviation.

**Data it operates on:** spectral_entropy from signal_vector.

**Used downstream by:** Anomaly ranking views, explorer anomaly panels.

---

### 12_ranked_derived.sql:25-71 — CRITICAL

**Code:**
```sql
signal_stats AS (
    SELECT signal_id, cohort,
        AVG(spectral_entropy) AS mean_val,
        STDDEV_SAMP(spectral_entropy) AS std_val ...
),
signal_z AS (
    SELECT (sv.spectral_entropy - ss.mean_val) / NULLIF(ss.std_val, 0) AS z_score ...
),
signal_extremes AS (
    SELECT cohort, signal_id, I, ABS(z_score) AS z_magnitude, ...
    WHERE signal_percentile > 0.95  -- Top 5% extremes
)
```

**What it does:** Duplicate of v_anomaly_ranked logic. Computes z-scores of spectral_entropy to identify "extreme" signal deviations, then selects the first timestep where a signal enters its top-5% z-score range.

**Impact:** Identifies canary signals — which signals deviate first. If z-scores are suppressed by including degraded data in the mean/std, the canary detection is delayed or missed entirely.

**Data it operates on:** spectral_entropy from signal_vector.

**Used downstream by:** Canary sequence analysis, early warning ordering.

---

### 13_canary_sequence.sql:47-94 — CRITICAL

**Code:**
```sql
baseline_slopes AS (
    SELECT cohort, signal_id,
        AVG(trajectory_delta) AS baseline_slope,
        STDDEV(trajectory_delta) AS baseline_slope_std
    FROM slopes s
    WHERE s.I < life.min_I + (life.max_I - life.min_I) * 0.2
),
trajectory_departures AS (
    SELECT
        CASE WHEN b.baseline_slope_std > 0
        THEN ABS(s.trajectory_delta - b.baseline_slope) / b.baseline_slope_std
        ELSE ABS(s.trajectory_delta - b.baseline_slope)
        END AS slope_departure
)
...
WHERE slope_departure > 3.0  -- 3σ threshold
```

**What it does:** Computes baseline slope mean/std from first 20% of life. Normalizes subsequent slope changes by baseline std. Flags departure when slope change exceeds 3× baseline std.

**Impact:** This is better than global z-scores (uses baseline-only window) but still assumes the baseline slope distribution is Gaussian and stationary. The 3.0 threshold is a fixed sigma multiple.

**Data it operates on:** trajectory_delta (rolling slope of signal values).

**Used downstream by:** v_canary_sequence — canary rank ordering, early warning system.

---

### 01_typology.sql:152-176 — MODERATE

**Code:**
```sql
rolling_vol AS (
    SELECT signal_id, I,
        STDDEV(y) OVER (ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS rolling_std
),
vol_stats AS (
    SELECT signal_id, AVG(rolling_std) AS mean_std, STDDEV(rolling_std) AS std_of_std
)
SELECT
    (r.rolling_std - v.mean_std) / NULLIF(v.std_of_std, 0) AS volatility_zscore,
    CASE
        WHEN (...) > 2 THEN 'burst'
        WHEN (...) < -1 THEN 'calm'
        ELSE 'normal'
    END AS volatility_state
```

**What it does:** Computes z-score of rolling volatility (25-point window std) against the signal's global mean/std of rolling volatility. Classifies volatility state as burst (>2σ), calm (<-1σ), or normal.

**Impact:** Burst detection. Uses global mean/std of rolling_std, which again treats the signal as stationary. A signal that progressively increases volatility will shift the mean, making late-life bursts look less extreme.

**Data it operates on:** Rolling standard deviation of raw y values.

**Used downstream by:** Typology classification — burst/calm labeling.

---

### 04_causality.sql:206-218 — MODERATE

**Code:**
```sql
WITH change_events AS (
    SELECT signal_id, I,
        ABS(dy) / NULLIF(STDDEV(dy) OVER (PARTITION BY signal_id), 0) AS change_magnitude
    FROM v_dy
),
significant_changes AS (
    SELECT signal_id, I FROM change_events
    WHERE change_magnitude > 2
)
```

**What it does:** Computes z-score of |dy| against the global std(dy) per signal. Flags "significant changes" where |dy| exceeds 2σ. These events become anchor points for causal timing analysis.

**Impact:** Gates which velocity changes are considered "significant" for causal analysis. Using global std(dy) means that in signals with large late-life velocity changes, the threshold is inflated, potentially masking early meaningful changes.

**Data it operates on:** First derivative (dy) per signal.

**Used downstream by:** v_causal_timing — causal response lag estimation.

---

### 03_dynamics.sql:24-36 — LOW

**Code:**
```sql
ABS(d.dy - LAG(d.dy) OVER w) / NULLIF(ds.dy_median_abs, 0) +
ABS(d.d2y - LAG(d.d2y) OVER w) / NULLIF(ds.d2y_median_abs, 0) AS change_score,
CASE
    WHEN ABS(...) > 3 * ds.dy_median_abs
      OR ABS(...) > 3 * ds.d2y_median_abs
    THEN TRUE ELSE FALSE
END AS is_regime_change
```

**What it does:** Normalizes derivative changes by **median** (not mean/std). Uses 3× median as threshold for regime change detection.

**Impact:** This is a **robust** alternative — median-based, not mean/std-based. The 3× multiplier is still somewhat arbitrary but less susceptible to outlier contamination than z-scores. Included for completeness.

**Data it operates on:** dy, d2y derivatives.

**Used downstream by:** Regime change detection view.

---

### 05_manifold_derived.sql:295-314 — MODERATE

**Code:**
```sql
CREATE OR REPLACE VIEW v_zscore_ref AS
SELECT
    'signal_vector' AS source,
    'spectral_entropy' AS column_name,
    AVG(spectral_entropy) AS mean,
    STDDEV(spectral_entropy) AS std,
    MIN(spectral_entropy) AS min,
    MAX(spectral_entropy) AS max
FROM signal_vector
```

**What it does:** Computes and stores global mean/std for signal_vector columns. This is a normalization reference table — the z-score parameters themselves.

**Impact:** Provides the denominators for z-score normalization used elsewhere. If these stats are computed over the full dataset (healthy + degraded), they dilute the baseline and reduce sensitivity.

**Data it operates on:** All signal_vector feature columns.

**Used downstream by:** Any code that normalizes features using this reference table.

---

### 32_basin_stability.sql:92-103 — CRITICAL

**Code:**
```sql
-- Normalized scores (z-score, then sigmoid to 0-1)
1.0 / (1.0 + EXP(-(b.mean_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0))) as coherence_score,
1.0 / (1.0 + EXP((b.mean_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0))) as velocity_score,
1.0 / (1.0 + EXP((b.coherence_volatility - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0))) as coherence_stability_score,
1.0 / (1.0 + EXP((b.velocity_volatility - f.fleet_vel_vol) / NULLIF(f.fleet_vel_vol_std, 0))) as velocity_stability_score
```

**What it does:** Computes z-scores for 4 metrics (coherence, velocity, coherence_vol, velocity_vol) relative to fleet mean/std, then maps through sigmoid to [0,1]. These become the basin stability score (weighted combination).

**Impact:** Basin stability score directly. The z-score inside the sigmoid means a fleet that includes many degraded entities will have inflated fleet_std, compressing the sigmoid and making all entities look "average." The 4 z-scores feed into a weighted composite that becomes the primary stability metric.

**Data it operates on:** Entity-level coherence, velocity, and their volatilities.

**Used downstream by:** Basin stability assessment, entity health comparison.

---

### 06_general_views.sql:91-98, 127-144 — CRITICAL

**Code:**
```sql
outlier_counts AS (
    SELECT signal_id,
        COUNT(*) FILTER (WHERE ABS(y - mean_val) > 3 * std_val) AS n_outliers_3sigma,
        COUNT(*) FILTER (WHERE ABS(y - mean_val) > 4 * std_val) AS n_outliers_4sigma
    FROM observations o
    JOIN signal_stats s USING (signal_id)
)
...
-- Overall health score (0-100)
GREATEST(0, 100
    - (n_nulls::FLOAT / NULLIF(n_points, 0) * 100)
    - (n_outliers_3sigma::FLOAT / NULLIF(n_points, 0) * 50)  -- Penalize outliers
    - (CASE WHEN n_gaps > 10 THEN 20 ELSE n_gaps * 2 END)
) AS health_score
```

**What it does:** Counts observations beyond 3σ and 4σ from the signal's global mean. Penalizes the health_score by up to 50 points based on outlier rate.

**Impact:** Signal health scoring. A degrading signal that spends its last 30% of life far from its early mean will accumulate "outliers" — but because mean/std include the degraded period, the threshold is inflated and real outliers are missed. Conversely, if degradation is gradual, no point ever exceeds 3σ and health_score stays 100 despite obvious drift.

**Data it operates on:** Raw observations (y).

**Used downstream by:** v_signal_health — data quality assessment, health scoring.

---

### 06_general_views.sql:352-365 — CRITICAL

**Code:**
```sql
CREATE OR REPLACE VIEW v_anomalies AS
SELECT
    'outlier' AS anomaly_type,
    signal_id, cohort, I AS index_at, y AS value,
    'Value ' || ROUND(y, 2) || ' is ' || ROUND(ABS(y - mean) / NULLIF(std, 0), 1) || ' std from mean' AS description,
    ABS(y - mean) / NULLIF(std, 0) AS severity
FROM observations o
JOIN (SELECT signal_id, AVG(y) AS mean, STDDEV(y) AS std FROM observations GROUP BY signal_id) s
    USING (signal_id)
WHERE ABS(y - mean) > 3 * std
```

**What it does:** Detects outlier observations using global 3σ threshold. Computes severity as number of std from mean.

**Impact:** Anomaly feed for explorer alerts panel. Same global-mean problem: gradual degradation never triggers because the mean shifts with the data. This is a user-facing anomaly list — if it misses real anomalies, operators won't see alerts.

**Data it operates on:** Raw observations (y).

**Used downstream by:** Explorer anomaly feed, alert panels.

---

### 06_general_views.sql:426-441 — MODERATE

**Code:**
```sql
(e.mean_val - g.global_mean) / NULLIF(g.entity_variance, 0) AS entity_zscore,
CASE
    WHEN ABS(e.mean_val - g.global_mean) > 2 * g.entity_variance THEN 'unusual'
    WHEN ABS(e.mean_val - g.global_mean) > 1 * g.entity_variance THEN 'different'
    ELSE 'typical'
END AS entity_status
```

**What it does:** Computes cross-entity z-scores: how far each entity's mean deviates from the global fleet mean, normalized by cross-entity variance. Classifies as unusual (>2σ), different (>1σ), or typical.

**Impact:** Entity comparison ranking. If the fleet includes mostly degraded entities, a healthy entity could appear "unusual." The 1σ/2σ thresholds assume Gaussian distribution of entity means.

**Data it operates on:** Entity-level signal means.

**Used downstream by:** v_entity_comparison — fleet outlier identification.

---

### 06_general_views.sql:51 — LOW

**Code:**
```sql
(AVG(y) - MEDIAN(y)) / NULLIF(STDDEV(y), 0) AS skewness_proxy,
```

**What it does:** Approximate skewness using (mean - median) / std.

**Impact:** Signal profiling. Uses std for normalization but is computing a shape statistic, not making a detection decision. Low impact because it's descriptive.

**Data it operates on:** Raw observations (y).

**Used downstream by:** v_signal_profile — statistical fingerprinting.

---

### 01_baseline_geometry.sql:164-193 — CRITICAL

**Code:**
```sql
CASE WHEN b.baseline_correlation_std > 0
THEN (c.correlation_mean - b.baseline_correlation) / b.baseline_correlation_std
ELSE 0 END AS correlation_zscore,

CASE WHEN b.baseline_coherence_std > 0
THEN (c.coherence_mean - b.baseline_coherence) / b.baseline_coherence_std
ELSE 0 END AS coherence_zscore,

CASE WHEN bp.baseline_entropy_std > 0
THEN (cp.total_entropy - bp.baseline_entropy) / bp.baseline_entropy_std
ELSE 0 END AS entropy_zscore,

-- Overall anomaly score (sum of absolute z-scores)
ABS(correlation_zscore) + ABS(coherence_zscore) + ABS(entropy_zscore) AS anomaly_score
```

**What it does:** Computes z-scores for correlation, coherence, and entropy relative to baseline-period mean/std. Sums absolute z-scores into an anomaly_score.

**Impact:** Drives traffic-light status classification (lines 238-243): anomaly_score >6 = CRITICAL, >4 = WARNING, >2 = WATCH, else NORMAL. These are user-facing status labels. The sum-of-z-scores assumes equal weighting and independence between the three metrics. Baseline comes from stable windows (better than global), but the threshold is in sigma units.

**Data it operates on:** Correlation, coherence, entropy geometry metrics.

**Used downstream by:** Current-vs-baseline comparison, status classification (CRITICAL/WARNING/WATCH/NORMAL).

---

### 23_baseline_deviation.sql:27-230 — CRITICAL

**Code:**
```sql
-- Config
2.0 AS z_threshold_warning,    -- |z| > 2 = warning
3.0 AS z_threshold_critical,   -- |z| > 3 = critical

-- 10 z-score computations:
(p.energy_proxy - b.energy_proxy_mean) / NULLIF(b.energy_proxy_std, 0) AS z_energy_proxy,
(p.energy_velocity - b.energy_velocity_mean) / NULLIF(b.energy_velocity_std, 0) AS z_energy_velocity,
(p.dissipation_rate - b.dissipation_rate_mean) / NULLIF(b.dissipation_rate_std, 0) AS z_dissipation_rate,
(p.coherence - b.coherence_mean) / NULLIF(b.coherence_std, 0) AS z_coherence,
(p.coherence_velocity - b.coherence_velocity_mean) / NULLIF(b.coherence_velocity_std, 0) AS z_coherence_velocity,
(p.effective_dim - b.effective_dim_mean) / NULLIF(b.effective_dim_std, 0) AS z_effective_dim,
(p.eigenvalue_entropy - b.eigenvalue_entropy_mean) / NULLIF(b.eigenvalue_entropy_std, 0) AS z_eigenvalue_entropy,
(p.state_distance - b.state_distance_mean) / NULLIF(b.state_distance_std, 0) AS z_state_distance,
(p.state_velocity - b.state_velocity_mean) / NULLIF(b.state_velocity_std, 0) AS z_state_velocity,
(p.state_acceleration - b.state_acceleration_mean) / NULLIF(b.state_acceleration_std, 0) AS z_state_acceleration,

-- Flags
ABS(d.z_energy_proxy) > c.z_threshold_warning AS flag_energy_proxy, ... (10 flags)
ABS(d.z_energy_proxy) > c.z_threshold_critical AS flag_critical_energy, ... (10 critical flags)

-- Max z
GREATEST(ABS(z_energy_proxy), ABS(z_coherence), ABS(z_state_distance), ABS(z_effective_dim)) AS max_z_score

-- Severity
CASE WHEN n_critical_metrics > 0 THEN 'critical'
     WHEN n_deviating_metrics >= 2 THEN 'warning'
     WHEN n_deviating_metrics > 0 THEN 'elevated' ELSE 'normal' END AS severity
```

**What it does:** The main deviation detection system. Computes z-scores for 10 physics metrics against baseline-period mean/std (first 10% of data). Flags warnings at |z| > 2, critical at |z| > 3. Counts deviating metrics and assigns severity.

**Impact:** This is the PRIMARY anomaly detection pipeline. It determines which entities are "degrading" and how severely. The 10% baseline window is better than global, but:
- Assumes baseline variability is Gaussian
- Fixed 2σ/3σ thresholds may be too lenient for some metrics, too strict for others
- Process health (drift_z, which was derived from this family) scored 0/249 failing engines as drifting — 100% miss rate

**Data it operates on:** 10 physics metrics: energy_proxy, energy_velocity, dissipation_rate, coherence, coherence_velocity, effective_dim, eigenvalue_entropy, state_distance, state_velocity, state_acceleration.

**Used downstream by:** 24_incident_summary.sql, 60_ground_truth.sql, 63_threshold_optimization.sql, tuning_service.py, baseline_deviation parquet output, z_total column.

---

### 24_incident_summary.sql:38-393 — CRITICAL

**Code:**
```sql
max_z_score AS first_z_score,         -- line 38
max_z_score AS peak_z_score,          -- line 204
ORDER BY max_z_score DESC;            -- line 209

-- Propagation sequence (which metric deviated first)
'energy_proxy' AS metric, z_energy_proxy AS z_score, ABS(z_energy_proxy) > 2 AS deviated
... (4 metrics)

-- Display
'  Z-score:    ' || ROUND(first_z_score::DECIMAL, 2)
'  Z-score:    ' || ROUND(peak_z_score::DECIMAL, 2)

ROUND(peak_z_score::DECIMAL, 1) AS peak_z,
ORDER BY peak_z_score DESC
```

**What it does:** Consumes z-scores from 23_baseline_deviation.sql. Uses max_z_score to rank incidents by severity. Maps propagation order by which metric's z-score exceeded 2 first.

**Impact:** Incident prioritization and narrative generation. If z-scores undercount severity (as they do when baseline std is inflated), incidents get ranked incorrectly and propagation chains are mislabeled.

**Data it operates on:** z-scores from baseline_deviation table.

**Used downstream by:** User-facing incident reports, fleet summary.

---

### 60_ground_truth.sql:80-141 — MODERATE

**Code:**
```sql
z_total,
ABS(z_total) AS z_magnitude,
PERCENT_RANK() OVER (PARTITION BY cohort ORDER BY ABS(z_total)) AS z_percentile,
...
MIN(I) FILTER (WHERE z_percentile > 0.90) AS first_p90_I,
MIN(I) FILTER (WHERE z_percentile > 0.95) AS first_p95_I,
MIN(I) FILTER (WHERE z_percentile > 0.99) AS first_p99_I,
MAX(ABS(z_total)) AS max_z_total
```

**What it does:** Consumes z_total from baseline_deviation. Converts absolute z-scores to percentile ranks within each entity. Identifies first time entity crosses p90/p95/p99 thresholds.

**Impact:** Ground truth validation — comparing detection timing against known faults. Uses percentile ranks (better than fixed sigma thresholds) but the underlying metric is still z_total, which carries the z-score contamination forward. If z_total is suppressed, percentile-based detection is also degraded.

**Data it operates on:** z_total from baseline_deviation table.

**Used downstream by:** 63_threshold_optimization.sql, lead time computation vs fault ground truth.

---

### 61_lead_time_analysis.sql:102-124 — CRITICAL

**Code:**
```sql
MIN(a.I_relative) FILTER (WHERE
    b.baseline_std > 0 AND
    ABS(a.metric_value - b.baseline_mean) / b.baseline_std > 2.0 AND
    a.I_relative < 0
) AS first_2sigma_relative_I,

... > 2.5 ... AS first_2_5sigma_relative_I,
... > 3.0 ... AS first_3sigma_relative_I,

MAX(ABS((a.metric_value - b.baseline_mean) / NULLIF(b.baseline_std, 0))) AS max_z_observed
```

**What it does:** Computes z-scores inline and finds the first time each metric exceeds 2σ, 2.5σ, and 3σ thresholds, relative to fault start time. Determines detection lead time.

**Impact:** This directly measures detection performance. If the z-score formulation misses real deviations, lead times appear worse (or faults appear undetectable). Classification of EARLY_DETECTION vs NOT_DETECTED (lines 156-160) depends on whether the z-score ever crosses 2σ before fault onset.

**Data it operates on:** Physics metrics vs baseline mean/std.

**Used downstream by:** 62_fault_signatures.sql, tuning_service.py detection metrics.

---

### 63_threshold_optimization.sql:22-68 — MODERATE

**Code:**
```sql
PERCENTILE_CONT(p / 100.0) WITHIN GROUP (ORDER BY ABS(z_total)) AS z_threshold
FROM baseline_deviation CROSS JOIN generate_series(5, 95, 5) AS t(p)
...
MIN(am.I) FILTER (WHERE ABS(am.z_total) > tc.z_threshold ...) AS first_detection_I
```

**What it does:** Tests every 5th percentile of observed z_total as a potential threshold. Evaluates detection performance (true positives, false negatives, lead time) at each level.

**Impact:** Threshold optimization. This is a self-referential analysis: it optimizes the z-score threshold using z-score values. If the underlying z-scores are miscalibrated, the "optimal" threshold will also be miscalibrated. Still uses z_total as the detection metric.

**Data it operates on:** z_total from baseline_deviation.

**Used downstream by:** Threshold recommendation, optimal_z output.

---

### 08_lead_lag.sql:116-152 — CRITICAL

**Code:**
```sql
baseline_stats AS (
    SELECT o.cohort, o.signal_id,
        AVG(o.value) AS baseline_mean,
        STDDEV_POP(o.value) AS baseline_std
    FROM observations o ...
),
deviations AS (
    SELECT (o.value - b.baseline_mean) / NULLIF(b.baseline_std, 0) AS z_score ...
)
SELECT
    MIN(CASE WHEN ABS(z_score) > 2.0 THEN I END) AS first_exceed_2sigma,
    MIN(CASE WHEN ABS(z_score) > 3.0 THEN I END) AS first_exceed_3sigma,
    CASE
        WHEN RANK() OVER (ORDER BY first_exceed_2sigma NULLS LAST) <= 3 THEN 'FIRST_RESPONDER'
        WHEN RANK() OVER (ORDER BY first_exceed_2sigma NULLS LAST) <= 10 THEN 'EARLY_RESPONDER'
        WHEN first_exceed_2sigma IS NOT NULL THEN 'LATE_RESPONDER'
        ELSE 'NON_RESPONDER'
    END AS response_class
```

**What it does:** Computes z-scores per signal after baseline period. Finds first time each signal exceeds 2σ and 3σ. Ranks signals by response order to determine FIRST_RESPONDER, EARLY_RESPONDER, LATE_RESPONDER, NON_RESPONDER.

**Impact:** Lead-lag analysis. Determines which signals respond first to degradation. If z-scores are insensitive, signals that actually respond early may be classified as NON_RESPONDER. The response_class labels directly affect operational interpretation of signal importance.

**Data it operates on:** Raw observation values per signal after baseline period.

**Used downstream by:** Signal response ordering, propagation analysis.

---

### 06_regime_detection.sql:50-82 — MODERATE

**Code:**
```sql
ABS(sig_mean - LAG(sig_mean) OVER w) / NULLIF(sig_std, 0) AS mean_change_magnitude,
sig_std / NULLIF(LAG(sig_std) OVER w, 0) AS std_ratio
...
SUM(CASE WHEN mean_change_magnitude > 1.5 THEN 1 ELSE 0 END) AS n_mean_shifts
...
WHEN 100.0 * n_mean_shifts / n_signals > 25 THEN 'REGIME_CHANGE'
WHEN 100.0 * n_mean_shifts / n_signals > 5 THEN 'PARTIAL_SHIFT'
```

**What it does:** Computes mean change between adjacent windows normalized by the window's std. Uses 1.5× threshold to identify mean shifts. Classifies regime changes based on percentage of signals shifting.

**Impact:** Regime boundary detection. The normalization by window std (not global std) is better than global z-scores, but the 1.5× threshold still assumes a particular relationship between mean change and std.

**Data it operates on:** Windowed signal statistics.

**Used downstream by:** Regime detection report, boundary classification.

---

### 11_validation_thresholds.sql:260-298 — LOW

**Code:**
```sql
WHEN COUNT(DISTINCT cohort) >= 30 THEN 'VALID - Full statistical analysis'
WHEN COUNT(DISTINCT cohort) >= 10 THEN 'MARGINAL - Use percentiles, z-scores with caution'
WHEN COUNT(DISTINCT cohort) >= 5 THEN 'LIMITED - Robust stats only (median, IQR), NO z-scores'
...
END AS z_score_validity
```

**What it does:** Validation guidance — warns that z-scores are only valid with N >= 30 entities. Recommends against z-scores with small fleets.

**Impact:** Advisory only. Does not compute z-scores. Documents the limitation but doesn't enforce it.

**Data it operates on:** Fleet size metadata.

**Used downstream by:** Validation report output.

---

### 33_birth_certificate.sql:112-145 — CRITICAL

**Code:**
```sql
ROUND(1.0 / (1.0 + EXP(-(e.early_coherence - f.fleet_coh) / NULLIF(f.fleet_coh_std, 0))), 3)
    as early_coupling_score,
ROUND(1.0 / (1.0 + EXP((e.early_velocity - f.fleet_vel) / NULLIF(f.fleet_vel_std, 0))), 3)
    as early_stability_score,
ROUND(1.0 / (1.0 + EXP((e.early_coherence_std - f.fleet_coh_vol) / NULLIF(f.fleet_coh_vol_std, 0))), 3)
    as early_consistency_score,
-- Weighted combination → birth_certificate_score → birth_grade (EXCELLENT/GOOD/FAIR/POOR)
```

**What it does:** Computes sigmoid-transformed z-scores for 3 early-life metrics. The z-score `(entity - fleet) / fleet_std` is inside the sigmoid exponent. Weights: coupling 40%, stability 40%, consistency 20%.

**Impact:** Birth certificate grading. Assigns EXCELLENT/GOOD/FAIR/POOR to entities based on early-life metrics normalized by fleet statistics. If fleet includes degraded entities, fleet_std is inflated and birth grades become less discriminating.

**Data it operates on:** Early-life coherence, velocity, coherence_std.

**Used downstream by:** Birth certificate report, entity initial quality assessment.

---

### queries.js:782-794 — MODERATE

**Code:**
```javascript
sql: `
WITH stats AS (
    SELECT cohort, AVG(speed) AS mean_speed, STDDEV(speed) AS std_speed
    FROM velocity_field GROUP BY cohort
)
SELECT
    ROUND((v.speed - s.mean_speed) / NULLIF(s.std_speed, 0), 2) AS z_score,
FROM velocity_field v JOIN stats s ON v.cohort = s.cohort
WHERE v.speed > s.mean_speed + 2 * s.std_speed
ORDER BY v.speed DESC LIMIT 50`
```

**What it does:** Explorer query: finds velocity field timesteps with speed > 2σ from mean. Computes z-score for display.

**Impact:** User-facing velocity anomaly detection. Global mean/std over all timesteps makes this sensitive to the same stationarity issues.

**Data it operates on:** velocity_field speed values.

**Used downstream by:** Explorer atlas velocity panel.

---

### queries-v2.js:1013-1019 — MODERATE

**Code:**
```javascript
ROUND((e.mean - f.mean) / NULLIF(f.std, 0.001), 2) AS z_from_fleet
FROM baseline e JOIN baseline f ON ... AND f.entity_id = 'FLEET'
WHERE e.entity_id != 'FLEET'
ORDER BY ABS(z_from_fleet) DESC
```

**What it does:** Computes per-entity z-score relative to fleet baseline. Ranks entities by deviation magnitude.

**Impact:** Entity comparison in explorer. Ranks which entities deviate most from fleet normal.

**Data it operates on:** Entity baseline metrics vs fleet baseline.

**Used downstream by:** Explorer baseline deviation panel.

---

### queries-v2.js:1029-1075 — MODERATE

**Code:**
```javascript
// anomaly_current:
ROUND(z_score, 2) AS z_score, anomaly_severity
FROM anomaly WHERE is_anomaly = TRUE ORDER BY ABS(z_score) DESC

// anomaly_by_entity:
ROUND(AVG(ABS(z_score)), 2) AS avg_abs_z, ROUND(MAX(ABS(z_score)), 2) AS max_abs_z

// anomaly_by_metric:
ROUND(AVG(z_score), 2) AS avg_z, ROUND(MAX(ABS(z_score)), 2) AS max_abs_z
```

**What it does:** Three explorer queries that consume z_score from the anomaly table. Display anomalies ranked by z-score, aggregate by entity, aggregate by metric.

**Impact:** User-facing anomaly panels. All ranking/display driven by z-score values computed upstream.

**Data it operates on:** anomaly table (z_score, is_anomaly, anomaly_severity columns).

**Used downstream by:** Explorer anomaly views (current, by entity, by metric, timeline).

---

### state/classification.py:55,110 — CRITICAL

**Code:**
```python
@dataclass
class StateThresholds:
    op_shift_sigma: float = 2.0  # Operating point deviation threshold in baseline σ

# ...
op_shifted = metrics.mean_op_deviation > thresholds.op_shift_sigma
```

**What it does:** Classifies operating point as "shifted" if mean deviation exceeds 2σ (baseline standard deviation units). The `mean_op_deviation` metric is already expressed in sigma units.

**Impact:** State classification — BASELINE_STABLE vs SHIFTED_STABLE. This is a core classification decision. The 2σ threshold assumes Gaussian baseline deviation.

**Data it operates on:** mean_op_deviation from state metrics (already in sigma units).

**Used downstream by:** State classification engine, operational status determination.

---

### services/fingerprint_service.py:413-519,743 — CRITICAL

**Code:**
```python
z_score = (value - metric_range.mean) / metric_range.std
...
if abs(z_score) < 2.0: status = "normal"
elif abs(z_score) < 3.0: status = "warning"
else: status = "critical"
...
z_score_trigger=2.0  # default trigger threshold
```

**What it does:** Compares current metrics against stored healthy fingerprints using z-scores. Classifies as normal/warning/critical at 2σ/3σ. Supports custom triggers like "z > 2.5".

**Impact:** Production monitoring. This is the real-time comparison system — fingerprints define what "healthy" looks like, and z-scores determine when something is wrong. If the healthy fingerprint's std is too wide (noisy baseline), real degradation doesn't trigger.

**Data it operates on:** Current metrics vs healthy fingerprint (stored mean/std).

**Used downstream by:** API endpoint /api/fingerprints/compare, health monitoring, trigger system.

---

### services/concierge.py:667-686 — MODERATE

**Code:**
```python
# SQL inside Python:
ROUND((e.coh - s.mean_coh) / s.std_coh, 2) as coh_zscore,
ROUND((e.state - s.mean_state) / s.std_state, 2) as state_zscore,
ROUND((e.dim - s.mean_dim) / s.std_dim, 2) as dim_zscore

# Python threshold:
if abs(d['coh_zscore']) > 2:
    anomalies.append(f"coherence ({d['coh_zscore']}σ)")
```

**What it does:** Computes z-scores for coherence, state_distance, and effective_dim relative to fleet mean/std. Flags entities with |z| > 2 as anomalous.

**Impact:** Concierge service anomaly detection. Reports anomalies to users in sigma notation. Fleet-relative z-scores are affected by fleet composition.

**Data it operates on:** Entity geometry metrics vs fleet statistics.

**Used downstream by:** Concierge natural language responses, fleet health overview.

---

### cohorts/baseline.py:350-354 — MODERATE

**Code:**
```python
z_score = (current_value - baseline_value) / baseline_std
...
z_score = 0.0 if current_value == baseline_value else float("inf")
deviations[metric] = z_score
```

**What it does:** Computes z-score of current geometry metrics relative to baseline mean/std.

**Impact:** Baseline deviation quantification. Returns dict of metric → z-score for programmatic use. The z-score values are consumed by other Python services.

**Data it operates on:** Current geometry values vs baseline geometry (mean/std).

**Used downstream by:** Pipeline deviation assessment, state classification inputs.

---

### early_warning/failure_fingerprint_detector.py:237-241 — CRITICAL

**Code:**
```python
zscore = (fingerprint.signal_d1_means[signal] - self._population_d1_means[signal]) / self._population_d1_stds[signal]
fingerprint.d1_population_zscore[signal] = float(zscore)
...
if abs(zscore) > 2.0:
    fingerprint.d1_anomalous_signals.add(signal)
```

**What it does:** Computes z-score of each signal's derivative mean against population statistics. Flags signals as anomalous at |z| > 2.0.

**Impact:** Early failure detection. Identifies engines with atypical early-life derivative patterns. If population statistics include failed engines, the std is inflated and early anomalies are missed.

**Data it operates on:** Signal first-derivative means vs population mean/std.

**Used downstream by:** Failure fingerprint matching, early warning system.

---

### manifest/domain_clock.py:233-237 — LOW

**Code:**
```python
values = values - np.mean(values)
std = np.std(values)
if std > 0:
    values = values / std
```

**What it does:** Standard z-score normalization (center and scale) before spectral analysis (FFT/Welch).

**Impact:** Low — this is signal preprocessing for frequency estimation, not a detection decision. Normalizing before FFT is standard practice and doesn't affect the frequency content, only the amplitude scale.

**Data it operates on:** Signal values before spectral analysis.

**Used downstream by:** Domain clock frequency estimation.

---

### ml/entry_points/train.py:128-130, 300-301 — MODERATE

**Code:**
```python
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

**What it does:** sklearn StandardScaler: (X - mean) / std applied to ML features before training gradient boosting models.

**Impact:** ML model training. StandardScaler is appropriate for gradient boosting (which is tree-based and technically doesn't need scaling), but the scaler is fit on training data which may mix healthy and degraded engines. Used in 3 ML entry points: train.py, evaluate_test.py, cycle_features.py.

**Data it operates on:** ML feature matrices.

**Used downstream by:** Gradient boosting regressors for RUL prediction.

---

### early_warning/ml_predictor.py:165 — MODERATE

**Code:**
```python
self.scaler = StandardScaler()
```

**What it does:** StandardScaler for early warning ML predictor features.

**Impact:** Early warning ML pipeline. Same concern as above — scaler fit on mixed population.

**Data it operates on:** Early warning feature vectors.

**Used downstream by:** GradientBoostingClassifier, GradientBoostingRegressor.

---

### core/api.py:2903-2921 — LOW

**Code:**
```python
# API response format:
"coherence": {"z_score": -3.2, "status": "critical"}
...
metric: {"z_score": z, "status": status}
for metric, (z, status) in comparison.items()
```

**What it does:** API endpoint formats z-scores from fingerprint_service for client consumption.

**Impact:** Display only — passes through z-scores computed by fingerprint_service. The API format itself is neutral; the issue is in the upstream z-score computation.

**Data it operates on:** Z-scores from fingerprint_service.compare_to_healthy().

**Used downstream by:** API clients, explorer UI.

---

### entry_points/stage_07_predict.py:27,79-80 — MODERATE

**Code:**
```python
method: str = "zscore",  # default anomaly detection method
choices=['zscore', 'isolation_forest', 'lof', 'combined']
```

**What it does:** Anomaly detection entry point defaults to "zscore" method. Also supports isolation_forest, LOF, and combined.

**Impact:** Default anomaly detection method for the prediction pipeline is z-score-based. Users who don't override the default get z-score anomaly detection.

**Data it operates on:** Manifold outputs (signal_vector, state_vector).

**Used downstream by:** AnomalyDetector prediction pipeline.

---

### sql/ml/11_ml_features.sql:350-372 — MODERATE

**Code:**
```sql
CREATE OR REPLACE VIEW v_ml_feature_stats AS
SELECT 'hurst' AS feature, AVG(hurst) AS mean, STDDEV(hurst) AS std, ...
UNION ALL
SELECT 'entropy', AVG(entropy), STDDEV(entropy), ...
-- ... (for each ML feature)
```

**What it does:** Pre-computes mean/std for each ML feature column. Provides normalization constants for downstream ML pipelines.

**Impact:** ML feature normalization reference. If these stats are computed over the full dataset (healthy + degraded), StandardScaler-equivalent normalization in the ML pipeline inherits the same bias.

**Data it operates on:** All ML feature columns.

**Used downstream by:** ML training, inference feature scaling.

---

### sql/ml/26_ml_feature_export.sql:359-390 — MODERATE

**Code:**
```sql
CREATE OR REPLACE TABLE ml_feature_stats AS
SELECT 'current_coherence' AS feature,
    MIN(current_coherence) AS actual_min, MAX(current_coherence) AS actual_max,
    AVG(current_coherence) AS mean, STDDEV(current_coherence) AS std
FROM ml_features_current
```

**What it does:** Computes and stores feature statistics (min, max, mean, std) for ML production features.

**Impact:** Production ML normalization. Same concern as 11_ml_features.sql — stats computed over full dataset.

**Data it operates on:** Production ML feature columns.

**Used downstream by:** ML inference pipeline feature scaling.

---

### shared/engine_registry.py:1653,1664-1667 — LOW

**Code:**
```python
# Documentation:
z = (x - μ_baseline) / σ_baseline

# Output spec:
columns=['baseline_mean', 'baseline_std', 'z_score', 'percentile_rank', 'is_anomaly', 'anomaly_severity'],
params={'z_threshold': 2.0, 'critical_threshold': 3.0},
```

**What it does:** Engine registry metadata — documents the z-score formula and specifies output column schema with default thresholds.

**Impact:** Documentation and interface contract. Encodes z_threshold=2.0 and critical_threshold=3.0 as defaults.

**Data it operates on:** Metadata only.

**Used downstream by:** Engine interface specifications.

---

## Categorization

### Z-scores used for classification/gating (CRITICAL)
These actively make wrong decisions:

| Instance | File | Decision Made |
|----------|------|--------------|
| 1 | 23_baseline_deviation.sql:155-257 | severity = critical/warning/elevated/normal |
| 2 | 01_baseline_geometry.sql:164-243 | status = CRITICAL/WARNING/WATCH/NORMAL |
| 3 | 08_lead_lag.sql:122-152 | response_class = FIRST/EARLY/LATE/NON_RESPONDER |
| 4 | 61_lead_time_analysis.sql:102-160 | outcome = EARLY_DETECTION/NOT_DETECTED |
| 5 | state/classification.py:55,110 | BASELINE_STABLE vs SHIFTED_STABLE |
| 6 | fingerprint_service.py:413-422 | status = normal/warning/critical |
| 7 | failure_fingerprint_detector.py:237-241 | anomalous signal flagging |
| 8 | 06_general_views.sql:352-365 | outlier anomaly detection (v_anomalies) |
| 9 | 06_general_views.sql:434-441 | entity_status = unusual/different/typical |
| 10 | 33_birth_certificate.sql:112-145 | birth_grade = EXCELLENT/GOOD/FAIR/POOR |

### Z-scores used for ranking/sorting (MODERATE)
Ranking by z-score vs ranking by raw value may differ:

| Instance | File | What's Ranked |
|----------|------|--------------|
| 1 | classification.sql:201-257 | Signal deviation ranking (v_anomaly_ranked) |
| 2 | 12_ranked_derived.sql:25-71 | Canary signal first-mover ordering |
| 3 | 24_incident_summary.sql:38-393 | Incident severity ranking by peak_z |
| 4 | concierge.py:667-686 | Entity anomaly ranking |
| 5 | queries-v2.js:1013-1019 | Entity fleet deviation ranking |

### Z-scores used for normalization before further computation (CRITICAL)
These mask real physics:

| Instance | File | What's Normalized |
|----------|------|------------------|
| 1 | 32_basin_stability.sql:92-103 | 4 metrics → sigmoid → basin_stability_score |
| 2 | 33_birth_certificate.sql:112-128 | 3 metrics → sigmoid → birth_certificate_score |
| 3 | 06_general_views.sql:91-144 | 3σ/4σ outlier rate → health_score |
| 4 | 01_baseline_geometry.sql:186-193 | 3 z-scores → anomaly_score sum |
| 5 | 05_manifold_derived.sql:295-314 | z-score reference table (mean/std for normalization) |
| 6 | ml/11_ml_features.sql:350-372 | ML feature normalization stats |
| 7 | ml/26_ml_feature_export.sql:359-390 | Production ML normalization stats |

### Z-scores in config thresholds (MODERATE)
These define alert boundaries in sigma units:

| Instance | File | Threshold Value |
|----------|------|----------------|
| 1 | 23_baseline_deviation.sql:34-35 | z_threshold_warning=2.0, z_threshold_critical=3.0 |
| 2 | state/classification.py:55 | op_shift_sigma=2.0 |
| 3 | fingerprint_service.py:743 | z_score_trigger=2.0 |
| 4 | engine_registry.py:1667 | z_threshold=2.0, critical_threshold=3.0 |
| 5 | entry_points/stage_07_predict.py:27 | default method="zscore" |
| 6 | 13_canary_sequence.sql:94 | slope_departure > 3.0 (3σ) |
| 7 | 04_causality.sql:218 | change_magnitude > 2 (2σ) |
| 8 | 06_regime_detection.sql:62 | mean_change_magnitude > 1.5 |

### Z-scores in display/reporting only (LOW)
Shown to user but don't drive decisions:

| Instance | File | What's Displayed |
|----------|------|-----------------|
| 1 | core/api.py:2903-2921 | API z_score response format |
| 2 | queries-v2.js:1029-1075 | Explorer anomaly panels (consuming upstream z-scores) |
| 3 | queries.js:790 | Atlas velocity z-score display |
| 4 | 11_validation_thresholds.sql:260-298 | z_score_validity guidance text |
| 5 | 06_general_views.sql:51 | skewness_proxy normalization |
| 6 | manifest/domain_clock.py:233-237 | FFT preprocessing (standard practice) |
| 7 | shared/engine_registry.py:1653 | Documentation only |
