-- ============================================================================
-- ORTHON SQL: baseline_geometry.sql
-- ============================================================================
-- Average geometry of most stable windows = baseline
--
-- PURPOSE:
--   Once stable windows are identified (via stable_baseline.sql), this query
--   computes the "baseline geometry" by averaging metrics across those windows.
--
-- WORKFLOW:
--   1. Run stable_baseline.sql first (creates stable_baseline view)
--   2. Run this script to compute baseline geometry
--   3. Compare current geometry to baseline for anomaly detection
--
-- Usage:
--   duckdb < stable_baseline.sql
--   duckdb < baseline_geometry.sql
-- ============================================================================

-- Ensure stable_baseline exists (run stable_baseline.sql first)
-- Load geometry and physics if not already loaded

CREATE OR REPLACE VIEW geometry AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/geometry.parquet');

CREATE OR REPLACE VIEW physics AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/physics.parquet');

CREATE OR REPLACE VIEW dynamics AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/dynamics.parquet');

-- ============================================================================
-- Recompute stable baseline if not exists
-- ============================================================================

CREATE OR REPLACE VIEW stability_scored AS
SELECT
    cohort,
    I as window_idx,
    (-1 * COALESCE(lyapunov_max, 0) + COALESCE(determinism, 0)) AS stability_score
FROM dynamics
WHERE lyapunov_max IS NOT NULL OR determinism IS NOT NULL;

CREATE OR REPLACE VIEW stable_baseline AS
SELECT
    cohort,
    window_idx,
    stability_score,
    ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY stability_score DESC) AS stability_rank
FROM stability_scored;

-- ============================================================================
-- STEP 1: Get stable windows (top 100 per entity)
-- ============================================================================

CREATE OR REPLACE VIEW stable_windows AS
SELECT cohort, window_idx
FROM stable_baseline
WHERE stability_rank <= 100;

-- ============================================================================
-- STEP 2: Compute baseline geometry from stable windows
-- ============================================================================

CREATE OR REPLACE VIEW v_baseline_geometry AS
SELECT
    g.cohort,

    -- Average geometry across stable windows
    AVG(g.correlation_mean) AS baseline_correlation,
    AVG(g.coherence_mean) AS baseline_coherence,
    AVG(g.mutual_info_mean) AS baseline_mutual_info,

    -- Stability of the baseline (how consistent were stable windows?)
    STDDEV(g.correlation_mean) AS baseline_correlation_std,
    STDDEV(g.coherence_mean) AS baseline_coherence_std,
    STDDEV(g.mutual_info_mean) AS baseline_mutual_info_std,

    -- Count of windows used
    COUNT(*) AS n_stable_windows,

    -- Average stability score of baseline windows
    AVG(sb.stability_score) AS avg_baseline_stability

FROM geometry g
INNER JOIN stable_windows sw
    ON g.cohort = sw.cohort
    AND g.I = sw.window_idx
INNER JOIN stable_baseline sb
    ON g.cohort = sb.cohort
    AND g.I = sb.window_idx
GROUP BY g.cohort;

-- ============================================================================
-- STEP 3: Compute baseline physics from stable windows
-- ============================================================================

CREATE OR REPLACE VIEW v_baseline_physics AS
SELECT
    p.cohort,

    -- Average physics across stable windows
    AVG(p.total_entropy) AS baseline_entropy,
    AVG(p.total_energy) AS baseline_energy,
    AVG(p.free_energy) AS baseline_free_energy,
    AVG(p.effective_dimension) AS baseline_eff_dim,
    AVG(p.coherence_index) AS baseline_coherence_idx,

    -- Stability
    STDDEV(p.total_entropy) AS baseline_entropy_std,
    STDDEV(p.total_energy) AS baseline_energy_std,

    COUNT(*) AS n_stable_windows

FROM physics p
INNER JOIN stable_windows sw
    ON p.cohort = sw.cohort
    AND p.I = sw.window_idx
GROUP BY p.cohort;

-- ============================================================================
-- STEP 4: Compare current state to baseline
-- ============================================================================

CREATE OR REPLACE VIEW v_current_vs_baseline AS
WITH current_state AS (
    -- Get most recent window per entity
    SELECT
        cohort,
        I as window_idx,
        correlation_mean,
        coherence_mean,
        mutual_info_mean
    FROM geometry
    WHERE (cohort, I) IN (
        SELECT cohort, MAX(I) FROM geometry GROUP BY cohort
    )
),
current_physics AS (
    SELECT
        cohort,
        I as window_idx,
        total_entropy,
        total_energy,
        free_energy,
        effective_dimension
    FROM physics
    WHERE (cohort, I) IN (
        SELECT cohort, MAX(I) FROM physics GROUP BY cohort
    )
)
SELECT
    c.cohort,
    c.window_idx AS current_window,

    -- Current values
    c.correlation_mean AS current_correlation,
    c.coherence_mean AS current_coherence,

    -- Baseline values
    b.baseline_correlation,
    b.baseline_coherence,

    -- Deviation from baseline (z-score style)
    CASE
        WHEN b.baseline_correlation_std > 0
        THEN (c.correlation_mean - b.baseline_correlation) / b.baseline_correlation_std
        ELSE 0
    END AS correlation_zscore,

    CASE
        WHEN b.baseline_coherence_std > 0
        THEN (c.coherence_mean - b.baseline_coherence) / b.baseline_coherence_std
        ELSE 0
    END AS coherence_zscore,

    -- Physics deviation
    cp.total_entropy AS current_entropy,
    bp.baseline_entropy,
    CASE
        WHEN bp.baseline_entropy_std > 0
        THEN (cp.total_entropy - bp.baseline_entropy) / bp.baseline_entropy_std
        ELSE 0
    END AS entropy_zscore,

    -- Overall anomaly score (sum of absolute z-scores)
    ABS(CASE WHEN b.baseline_correlation_std > 0
        THEN (c.correlation_mean - b.baseline_correlation) / b.baseline_correlation_std ELSE 0 END)
    + ABS(CASE WHEN b.baseline_coherence_std > 0
        THEN (c.coherence_mean - b.baseline_coherence) / b.baseline_coherence_std ELSE 0 END)
    + ABS(CASE WHEN bp.baseline_entropy_std > 0
        THEN (cp.total_entropy - bp.baseline_entropy) / bp.baseline_entropy_std ELSE 0 END)
    AS anomaly_score

FROM current_state c
LEFT JOIN v_baseline_geometry b ON c.cohort = b.cohort
LEFT JOIN current_physics cp ON c.cohort = cp.cohort
LEFT JOIN v_baseline_physics bp ON c.cohort = bp.cohort
ORDER BY anomaly_score DESC;

-- ============================================================================
-- OUTPUT
-- ============================================================================

.print ''
.print '=== BASELINE GEOMETRY (from stable windows) ==='
SELECT
    cohort,
    ROUND(baseline_correlation, 4) AS baseline_corr,
    ROUND(baseline_coherence, 4) AS baseline_coh,
    ROUND(baseline_correlation_std, 4) AS corr_std,
    ROUND(baseline_coherence_std, 4) AS coh_std,
    n_stable_windows
FROM v_baseline_geometry
ORDER BY cohort;

.print ''
.print '=== BASELINE PHYSICS (from stable windows) ==='
SELECT
    cohort,
    ROUND(baseline_entropy, 4) AS baseline_H,
    ROUND(baseline_energy, 4) AS baseline_E,
    ROUND(baseline_free_energy, 4) AS baseline_F,
    ROUND(baseline_eff_dim, 2) AS baseline_dim,
    n_stable_windows
FROM v_baseline_physics
ORDER BY cohort;

.print ''
.print '=== CURRENT STATE VS BASELINE ==='
SELECT
    cohort,
    current_window,
    ROUND(correlation_zscore, 2) AS corr_z,
    ROUND(coherence_zscore, 2) AS coh_z,
    ROUND(entropy_zscore, 2) AS entropy_z,
    ROUND(anomaly_score, 2) AS anomaly_score,
    CASE
        WHEN anomaly_score > 6 THEN 'CRITICAL'
        WHEN anomaly_score > 4 THEN 'WARNING'
        WHEN anomaly_score > 2 THEN 'WATCH'
        ELSE 'NORMAL'
    END AS status
FROM v_current_vs_baseline
ORDER BY anomaly_score DESC;
