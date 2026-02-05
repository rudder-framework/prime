-- ============================================================================
-- ORTHON SQL: 13_l4_thermodynamics.sql
-- ============================================================================
-- L4: THERMODYNAMICS - Is energy conserved?
--
-- First question in physics stack. If energy is not conserved,
-- something is wrong - investigate L3/L2/L1 for mechanism.
--
-- Columns used: energy_proxy, energy_velocity, dissipation_rate, entropy_production
-- ============================================================================

-- Energy trend per time point
CREATE OR REPLACE VIEW v_l4_energy_trend AS
SELECT
    entity_id,
    I,
    energy_proxy,
    energy_velocity,
    dissipation_rate,
    entropy_production,

    -- Classify energy trend
    CASE
        WHEN energy_velocity < -0.001 THEN 'dissipating'
        WHEN energy_velocity > 0.001 THEN 'accumulating'
        ELSE 'stable'
    END AS energy_trend,

    -- Classify entropy trend
    CASE
        WHEN entropy_production > 0.001 THEN 'increasing'
        WHEN entropy_production < -0.001 THEN 'decreasing'
        ELSE 'stable'
    END AS entropy_trend

FROM physics
WHERE energy_proxy IS NOT NULL;


-- Rolling conservation check
CREATE OR REPLACE VIEW v_l4_conservation AS
SELECT
    entity_id,
    I,
    energy_proxy,

    -- Rolling statistics
    AVG(energy_proxy) OVER w AS energy_mean,
    STDDEV(energy_proxy) OVER w AS energy_std,
    AVG(energy_velocity) OVER w AS energy_velocity_mean,

    -- Coefficient of variation
    STDDEV(energy_proxy) OVER w / NULLIF(AVG(energy_proxy) OVER w, 0) AS energy_cv,

    -- Conservation check
    CASE
        WHEN STDDEV(energy_proxy) OVER w / NULLIF(ABS(AVG(energy_proxy) OVER w), 0) < 0.1
         AND ABS(AVG(energy_velocity) OVER w) < 0.01 * ABS(AVG(energy_proxy) OVER w)
        THEN TRUE
        ELSE FALSE
    END AS energy_conserved

FROM physics
WHERE energy_proxy IS NOT NULL
WINDOW w AS (PARTITION BY entity_id ORDER BY I ROWS BETWEEN 10 PRECEDING AND CURRENT ROW);


-- Entity-level thermodynamics summary
CREATE OR REPLACE VIEW v_l4_entity_summary AS
SELECT
    entity_id,

    -- Overall energy stats
    AVG(energy_proxy) AS energy_mean,
    STDDEV(energy_proxy) AS energy_std,
    STDDEV(energy_proxy) / NULLIF(ABS(AVG(energy_proxy)), 0) AS energy_cv,

    -- Trend
    REGR_SLOPE(energy_proxy, I) AS energy_trend_slope,
    CASE
        WHEN REGR_SLOPE(energy_proxy, I) < -0.001 * AVG(energy_proxy) THEN 'dissipating'
        WHEN REGR_SLOPE(energy_proxy, I) > 0.001 * AVG(energy_proxy) THEN 'accumulating'
        ELSE 'stable'
    END AS overall_energy_trend,

    -- Dissipation
    AVG(dissipation_rate) AS mean_dissipation_rate,
    MAX(dissipation_rate) AS max_dissipation_rate,

    -- Entropy
    AVG(entropy_production) AS mean_entropy_production,
    CASE
        WHEN AVG(entropy_production) > 0.001 THEN 'increasing'
        WHEN AVG(entropy_production) < -0.001 THEN 'decreasing'
        ELSE 'stable'
    END AS overall_entropy_trend,

    -- Conservation
    CASE
        WHEN STDDEV(energy_proxy) / NULLIF(ABS(AVG(energy_proxy)), 0) < 0.1
         AND ABS(REGR_SLOPE(energy_proxy, I)) < 0.01 * ABS(AVG(energy_proxy))
        THEN TRUE
        ELSE FALSE
    END AS energy_conserved

FROM physics
WHERE energy_proxy IS NOT NULL
GROUP BY entity_id;


-- Dissipation events (rapid energy loss)
CREATE OR REPLACE VIEW v_l4_dissipation_events AS
WITH stats AS (
    SELECT
        entity_id,
        AVG(dissipation_rate) AS mean_diss,
        STDDEV(dissipation_rate) AS std_diss
    FROM physics
    WHERE dissipation_rate IS NOT NULL
    GROUP BY entity_id
)
SELECT
    p.entity_id,
    p.I,
    p.dissipation_rate,
    s.mean_diss,
    s.std_diss,
    (p.dissipation_rate - s.mean_diss) / NULLIF(s.std_diss, 0) AS dissipation_zscore,
    CASE
        WHEN p.dissipation_rate > s.mean_diss + 2 * s.std_diss THEN 'high_dissipation'
        ELSE 'normal'
    END AS event_type
FROM physics p
JOIN stats s USING (entity_id)
WHERE p.dissipation_rate > s.mean_diss + 2 * s.std_diss;


-- Verify
SELECT COUNT(*) AS thermodynamics_rows FROM v_l4_energy_trend;
