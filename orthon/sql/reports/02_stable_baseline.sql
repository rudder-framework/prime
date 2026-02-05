-- ============================================================================
-- ORTHON SQL: stable_baseline.sql
-- ============================================================================
-- Find the N most stable windows per entity
--
-- PURPOSE:
--   Instead of assuming first N% of data is "healthy baseline", this query
--   discovers the most geometrically stable periods in the data.
--
-- USE CASES:
--   - Financial markets (no "healthy" start - find calm periods)
--   - Bioreactors (optimal state unknown)
--   - Any system where baseline isn't obvious
--
-- STABILITY DEFINITION:
--   Low chaos (negative Lyapunov) + High determinism = Stable
--
-- Usage:
--   duckdb < stable_baseline.sql
-- ============================================================================

-- Load dynamics if not already loaded
CREATE OR REPLACE VIEW dynamics AS
SELECT * FROM read_parquet('/Users/jasonrudder/prism/data/dynamics.parquet');

-- ============================================================================
-- STEP 1: Score each window's stability
-- ============================================================================

CREATE OR REPLACE VIEW stability_scored AS
SELECT
    cohort,
    I as window_idx,

    -- Stability metrics from dynamics
    lyapunov_max,
    determinism,
    recurrence_rate,
    entropy_rate,

    -- Simple stability score: low Lyapunov + high determinism
    -- Lyapunov < 0 = stable, so we negate it
    -- Determinism near 1 = predictable
    (-1 * COALESCE(lyapunov_max, 0) + COALESCE(determinism, 0)) AS stability_score,

    -- Alternative: include recurrence (high = more stable patterns)
    (-1 * COALESCE(lyapunov_max, 0)
     + COALESCE(determinism, 0)
     + 0.5 * COALESCE(recurrence_rate, 0)) AS stability_score_extended

FROM dynamics
WHERE lyapunov_max IS NOT NULL
   OR determinism IS NOT NULL;

-- ============================================================================
-- STEP 2: Rank windows by stability within each entity
-- ============================================================================

CREATE OR REPLACE VIEW stable_baseline AS
SELECT
    cohort,
    window_idx,
    stability_score,
    stability_score_extended,
    lyapunov_max,
    determinism,
    recurrence_rate,
    ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY stability_score DESC) AS stability_rank
FROM stability_scored;

-- ============================================================================
-- STEP 3: Extract top N most stable windows (configurable)
-- ============================================================================

CREATE OR REPLACE VIEW top_stable_windows AS
SELECT *
FROM stable_baseline
WHERE stability_rank <= 100  -- Top 100 most stable windows per entity
ORDER BY cohort, stability_rank;

-- ============================================================================
-- SUMMARY: Stability distribution per entity
-- ============================================================================

CREATE OR REPLACE VIEW v_stability_summary AS
SELECT
    cohort,
    COUNT(*) AS total_windows,

    -- How many "stable" windows (positive score)?
    SUM(CASE WHEN stability_score > 0 THEN 1 ELSE 0 END) AS n_stable_windows,

    -- Stability score distribution
    AVG(stability_score) AS mean_stability,
    STDDEV(stability_score) AS std_stability,
    MIN(stability_score) AS min_stability,
    MAX(stability_score) AS max_stability,

    -- Top stable window info
    MAX(stability_score) AS best_stability_score,

    -- Percentage of time in stable state
    ROUND(100.0 * SUM(CASE WHEN stability_score > 0 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_stable

FROM stability_scored
GROUP BY cohort
ORDER BY mean_stability DESC;

-- ============================================================================
-- OUTPUT
-- ============================================================================

.print ''
.print '=== STABILITY SUMMARY BY ENTITY ==='
SELECT * FROM v_stability_summary;

.print ''
.print '=== TOP 10 MOST STABLE WINDOWS (ALL ENTITIES) ==='
SELECT
    cohort,
    window_idx,
    ROUND(stability_score, 4) AS stability_score,
    ROUND(lyapunov_max, 4) AS lyapunov,
    ROUND(determinism, 4) AS determinism,
    stability_rank
FROM stable_baseline
WHERE stability_rank <= 10
ORDER BY stability_score DESC
LIMIT 50;
