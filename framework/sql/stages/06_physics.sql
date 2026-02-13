-- ============================================================================
-- Rudder SQL: Physics Stage Reports
-- ============================================================================
-- Thermodynamic interpretation: entropy, energy, free energy
--
-- Input: physics.parquet (from Engines)
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Physics Schema
-- ----------------------------------------------------------------------------
DESCRIBE SELECT * FROM physics LIMIT 1;

-- ----------------------------------------------------------------------------
-- 2. Physics Summary by Cohort
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    COUNT(*) as n_windows,
    ROUND(AVG(total_entropy), 4) as avg_entropy,
    ROUND(AVG(total_energy), 4) as avg_energy,
    ROUND(AVG(free_energy), 4) as avg_free_energy,
    ROUND(AVG(effective_dimension), 2) as avg_eff_dim
FROM physics
GROUP BY cohort
ORDER BY cohort;

-- ----------------------------------------------------------------------------
-- 3. Entropy Trajectory
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    I,
    total_entropy,
    LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I) as prev_entropy,
    total_entropy - LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I) as entropy_change
FROM physics
ORDER BY cohort, I;

-- ----------------------------------------------------------------------------
-- 4. Energy Analysis
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    I,
    total_energy,
    kinetic_energy,
    potential_energy,
    total_energy - LAG(total_energy) OVER (PARTITION BY cohort ORDER BY I) as energy_change
FROM physics
WHERE total_energy IS NOT NULL
ORDER BY cohort, I;

-- ----------------------------------------------------------------------------
-- 5. Free Energy (F = E - TS)
-- ----------------------------------------------------------------------------
-- Low free energy = stable state
SELECT
    cohort,
    I,
    free_energy,
    total_entropy,
    total_energy,
    CASE
        WHEN free_energy < (SELECT PERCENTILE_CONT(0.1) WITHIN GROUP (ORDER BY free_energy) FROM physics) THEN 'VERY_STABLE'
        WHEN free_energy < (SELECT PERCENTILE_CONT(0.3) WITHIN GROUP (ORDER BY free_energy) FROM physics) THEN 'STABLE'
        WHEN free_energy > (SELECT PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY free_energy) FROM physics) THEN 'UNSTABLE'
        ELSE 'NORMAL'
    END as stability_state
FROM physics
ORDER BY free_energy ASC
LIMIT 100;

-- ----------------------------------------------------------------------------
-- 6. Entropy-Energy Relationship
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    CORR(total_entropy, total_energy) as entropy_energy_corr,
    COUNT(*) as n_windows
FROM physics
GROUP BY cohort
HAVING COUNT(*) > 10
ORDER BY entropy_energy_corr DESC;

-- ----------------------------------------------------------------------------
-- 7. Recent Physics (last 20 windows)
-- ----------------------------------------------------------------------------
SELECT *
FROM (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY cohort ORDER BY I DESC) as rn
    FROM physics
) ranked
WHERE rn <= 20
ORDER BY cohort, I DESC;

-- ----------------------------------------------------------------------------
-- 8. Phase Transitions (Entropy Spikes)
-- ----------------------------------------------------------------------------
SELECT
    cohort,
    I,
    total_entropy,
    LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I) as prev_entropy,
    total_entropy - LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I) as entropy_jump
FROM physics
WHERE ABS(total_entropy - LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I)) >
      2 * (SELECT STDDEV(total_entropy) FROM physics)
ORDER BY ABS(total_entropy - LAG(total_entropy) OVER (PARTITION BY cohort ORDER BY I)) DESC
LIMIT 50;
