-- ============================================================================
-- baseline_geometry.sql
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
--   3. Compare current geometry to baseline for deviation detection
--
-- Usage:
--   duckdb < stable_baseline.sql
--   duckdb < baseline_geometry.sql
-- ============================================================================

-- Ensure stable_baseline exists (run stable_baseline.sql first)
-- Load geometry and physics if not already loaded

CREATE OR REPLACE VIEW geometry AS
SELECT * FROM read_parquet('/Users/jasonrudder/manifold/data/geometry.parquet');

CREATE OR REPLACE VIEW physics AS
SELECT * FROM read_parquet('/Users/jasonrudder/manifold/data/physics.parquet');

CREATE OR REPLACE VIEW dynamics AS
SELECT * FROM read_parquet('/Users/jasonrudder/manifold/data/dynamics.parquet');

-- ============================================================================
-- Recompute stable baseline if not exists
-- ============================================================================

CREATE OR REPLACE VIEW stability_scored AS
SELECT
    cohort,
    signal_0_center as window_idx,
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

    -- Baseline range (percentile bounds instead of std)
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY g.correlation_mean) AS baseline_correlation_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY g.correlation_mean) AS baseline_correlation_p95,
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY g.coherence_mean) AS baseline_coherence_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY g.coherence_mean) AS baseline_coherence_p95,

    -- Count of windows used
    COUNT(*) AS n_stable_windows,

    -- Average stability score of baseline windows
    AVG(sb.stability_score) AS avg_baseline_stability

FROM geometry g
INNER JOIN stable_windows sw
    ON g.cohort = sw.cohort
    AND g.signal_0_center = sw.window_idx
INNER JOIN stable_baseline sb
    ON g.cohort = sb.cohort
    AND g.signal_0_center = sb.window_idx
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

    -- Baseline range (percentile bounds)
    PERCENTILE_CONT(0.05) WITHIN GROUP (ORDER BY p.total_entropy) AS baseline_entropy_p05,
    PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY p.total_entropy) AS baseline_entropy_p95,

    COUNT(*) AS n_stable_windows

FROM physics p
INNER JOIN stable_windows sw
    ON p.cohort = sw.cohort
    AND p.signal_0_center = sw.window_idx
GROUP BY p.cohort;

-- ============================================================================
-- STEP 4: Compare current state to baseline
-- ============================================================================

CREATE OR REPLACE VIEW v_current_vs_baseline AS
WITH current_state AS (
    -- Get most recent window per entity
    SELECT
        cohort,
        signal_0_center as window_idx,
        correlation_mean,
        coherence_mean,
        mutual_info_mean
    FROM geometry
    WHERE (cohort, signal_0_center) IN (
        SELECT cohort, MAX(signal_0_center) FROM geometry GROUP BY cohort
    )
),
current_physics AS (
    SELECT
        cohort,
        signal_0_center as window_idx,
        total_entropy,
        total_energy,
        free_energy,
        effective_dimension
    FROM physics
    WHERE (cohort, signal_0_center) IN (
        SELECT cohort, MAX(signal_0_center) FROM physics GROUP BY cohort
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

    -- Deviation from baseline (range exceedance)
    CASE
        WHEN (b.baseline_correlation_p95 - b.baseline_correlation_p05) > 0
        THEN GREATEST(
            CASE WHEN c.correlation_mean > b.baseline_correlation_p95
                 THEN (c.correlation_mean - b.baseline_correlation_p95) / (b.baseline_correlation_p95 - b.baseline_correlation_p05)
                 WHEN c.correlation_mean < b.baseline_correlation_p05
                 THEN (b.baseline_correlation_p05 - c.correlation_mean) / (b.baseline_correlation_p95 - b.baseline_correlation_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END AS correlation_exceedance,

    CASE
        WHEN (b.baseline_coherence_p95 - b.baseline_coherence_p05) > 0
        THEN GREATEST(
            CASE WHEN c.coherence_mean > b.baseline_coherence_p95
                 THEN (c.coherence_mean - b.baseline_coherence_p95) / (b.baseline_coherence_p95 - b.baseline_coherence_p05)
                 WHEN c.coherence_mean < b.baseline_coherence_p05
                 THEN (b.baseline_coherence_p05 - c.coherence_mean) / (b.baseline_coherence_p95 - b.baseline_coherence_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END AS coherence_exceedance,

    -- Physics deviation
    cp.total_entropy AS current_entropy,
    bp.baseline_entropy,
    CASE
        WHEN (bp.baseline_entropy_p95 - bp.baseline_entropy_p05) > 0
        THEN GREATEST(
            CASE WHEN cp.total_entropy > bp.baseline_entropy_p95
                 THEN (cp.total_entropy - bp.baseline_entropy_p95) / (bp.baseline_entropy_p95 - bp.baseline_entropy_p05)
                 WHEN cp.total_entropy < bp.baseline_entropy_p05
                 THEN (bp.baseline_entropy_p05 - cp.total_entropy) / (bp.baseline_entropy_p95 - bp.baseline_entropy_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END AS entropy_exceedance,

    -- Overall deviation score (sum of exceedances)
    CASE
        WHEN (b.baseline_correlation_p95 - b.baseline_correlation_p05) > 0
        THEN GREATEST(
            CASE WHEN c.correlation_mean > b.baseline_correlation_p95
                 THEN (c.correlation_mean - b.baseline_correlation_p95) / (b.baseline_correlation_p95 - b.baseline_correlation_p05)
                 WHEN c.correlation_mean < b.baseline_correlation_p05
                 THEN (b.baseline_correlation_p05 - c.correlation_mean) / (b.baseline_correlation_p95 - b.baseline_correlation_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END
    + CASE
        WHEN (b.baseline_coherence_p95 - b.baseline_coherence_p05) > 0
        THEN GREATEST(
            CASE WHEN c.coherence_mean > b.baseline_coherence_p95
                 THEN (c.coherence_mean - b.baseline_coherence_p95) / (b.baseline_coherence_p95 - b.baseline_coherence_p05)
                 WHEN c.coherence_mean < b.baseline_coherence_p05
                 THEN (b.baseline_coherence_p05 - c.coherence_mean) / (b.baseline_coherence_p95 - b.baseline_coherence_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END
    + CASE
        WHEN (bp.baseline_entropy_p95 - bp.baseline_entropy_p05) > 0
        THEN GREATEST(
            CASE WHEN cp.total_entropy > bp.baseline_entropy_p95
                 THEN (cp.total_entropy - bp.baseline_entropy_p95) / (bp.baseline_entropy_p95 - bp.baseline_entropy_p05)
                 WHEN cp.total_entropy < bp.baseline_entropy_p05
                 THEN (bp.baseline_entropy_p05 - cp.total_entropy) / (bp.baseline_entropy_p95 - bp.baseline_entropy_p05)
                 ELSE 0 END, 0)
        ELSE 0
    END
    AS deviation_score

FROM current_state c
LEFT JOIN v_baseline_geometry b ON c.cohort = b.cohort
LEFT JOIN current_physics cp ON c.cohort = cp.cohort
LEFT JOIN v_baseline_physics bp ON c.cohort = bp.cohort
ORDER BY deviation_score DESC;

-- ============================================================================
-- OUTPUT
-- ============================================================================

.print ''
.print '=== BASELINE GEOMETRY (from stable windows) ==='
SELECT
    cohort,
    ROUND(baseline_correlation, 4) AS baseline_corr,
    ROUND(baseline_coherence, 4) AS baseline_coh,
    ROUND(baseline_correlation_p05, 4) AS corr_p05,
    ROUND(baseline_correlation_p95, 4) AS corr_p95,
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
    ROUND(correlation_exceedance, 2) AS corr_exc,
    ROUND(coherence_exceedance, 2) AS coh_exc,
    ROUND(entropy_exceedance, 2) AS entropy_exc,
    ROUND(deviation_score, 2) AS deviation_score,
    CASE
        WHEN deviation_score > 3 THEN 'CRITICAL'
        WHEN deviation_score > 1.5 THEN 'WARNING'
        WHEN deviation_score > 0.5 THEN 'WATCH'
        ELSE 'WITHIN_BASELINE'
    END AS status
FROM v_current_vs_baseline
ORDER BY deviation_score DESC;
