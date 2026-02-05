-- ============================================================================
-- ORTHON SQL: 22_vector_energy.sql
-- ============================================================================
-- VECTOR ENERGY: Per-Signal Energy Contribution
--
-- For each signal in each window, compute:
--   - Kinetic Energy: How fast is it moving? (accumulated squared returns)
--   - Potential Energy: How far from equilibrium? (displacement from centroid)
--   - Total Energy: Full energy contribution to the system
--
-- What You Learn:
--   - Which signal is driving the system
--   - Which is absorbing energy vs releasing it
--   - Energy migrating between vectors across windows
--
-- No new engine. Squared returns and deviation from centroid.
-- Already computed. SQL asks the question. Milliseconds.
-- ============================================================================

-- ============================================================================
-- REQUIRES: observations_enriched.parquet (per-signal data)
-- If not available, uses fallback calculations from physics.parquet
-- ============================================================================

-- Try to load observations_enriched
CREATE OR REPLACE TABLE observations_enriched AS
SELECT * FROM read_parquet('{prism_output}/observations_enriched.parquet');


-- ============================================================================
-- CENTROID PER WINDOW
-- ============================================================================
-- The equilibrium point: mean of all signals at each timestep.

CREATE OR REPLACE VIEW v_window_centroid AS
SELECT
    entity_id,
    I,
    AVG(y) AS centroid,
    STDDEV(y) AS dispersion,
    COUNT(DISTINCT signal_id) AS n_signals
FROM observations_enriched
GROUP BY entity_id, I;


-- ============================================================================
-- SIGNAL RETURNS (VELOCITY)
-- ============================================================================
-- Change in signal value between consecutive points.

CREATE OR REPLACE VIEW v_signal_returns AS
SELECT
    o.entity_id,
    o.signal_id,
    o.I,
    o.y,
    o.y - LAG(o.y) OVER w AS dy,
    POWER(o.y - LAG(o.y) OVER w, 2) AS return_squared,
    c.centroid,
    c.dispersion
FROM observations_enriched o
JOIN v_window_centroid c ON o.entity_id = c.entity_id AND o.I = c.I
WINDOW w AS (PARTITION BY o.entity_id, o.signal_id ORDER BY o.I);


-- ============================================================================
-- VECTOR ENERGY PER SIGNAL
-- ============================================================================
-- Kinetic + Potential = Total energy contribution per signal.

CREATE OR REPLACE VIEW v_vector_energy AS
SELECT
    entity_id,
    signal_id,
    I,
    y,
    centroid,

    -- Kinetic Energy: accumulated squared returns (how fast is it moving)
    -- Using rolling sum for the window
    SUM(return_squared) OVER (
        PARTITION BY entity_id, signal_id
        ORDER BY I
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) AS kinetic_energy,

    -- Instantaneous kinetic (just current return)
    return_squared AS kinetic_instant,

    -- Potential Energy: displacement from centroid (how far from equilibrium)
    POWER(y - centroid, 2) AS potential_energy,

    -- Total Energy: kinetic + potential
    SUM(return_squared) OVER (
        PARTITION BY entity_id, signal_id
        ORDER BY I
        ROWS BETWEEN 10 PRECEDING AND CURRENT ROW
    ) + POWER(y - centroid, 2) AS total_energy,

    -- Simple energy proxy (instantaneous)
    return_squared + POWER(y - centroid, 2) AS instant_energy

FROM v_signal_returns
WHERE return_squared IS NOT NULL;


-- ============================================================================
-- NORMALIZED ENERGY (ENERGY SHARE)
-- ============================================================================
-- What fraction of system energy does each signal carry?

CREATE OR REPLACE VIEW v_energy_normalized AS
SELECT
    entity_id,
    signal_id,
    I,
    kinetic_energy,
    potential_energy,
    total_energy,
    instant_energy,

    -- Energy fraction: this signal's share of total system energy
    total_energy / NULLIF(SUM(total_energy) OVER (PARTITION BY entity_id, I), 0) AS energy_fraction,

    -- Instantaneous fraction
    instant_energy / NULLIF(SUM(instant_energy) OVER (PARTITION BY entity_id, I), 0) AS instant_fraction,

    -- Kinetic fraction (of this signal's energy)
    kinetic_energy / NULLIF(total_energy, 0) AS kinetic_ratio,

    -- Potential fraction (of this signal's energy)
    potential_energy / NULLIF(total_energy, 0) AS potential_ratio,

    -- Energy rank within this window
    RANK() OVER (PARTITION BY entity_id, I ORDER BY total_energy DESC) AS energy_rank

FROM v_vector_energy;


-- ============================================================================
-- ENERGY DYNAMICS
-- ============================================================================
-- Track energy flow: is the signal gaining or losing energy?

CREATE OR REPLACE VIEW v_energy_dynamics AS
SELECT
    entity_id,
    signal_id,
    I,
    total_energy,
    energy_fraction,
    energy_rank,
    kinetic_ratio,
    potential_ratio,

    -- Energy change
    total_energy - LAG(total_energy) OVER w AS energy_delta,

    -- Fraction change
    energy_fraction - LAG(energy_fraction) OVER w AS fraction_delta,

    -- Rank change
    LAG(energy_rank) OVER w - energy_rank AS rank_improvement,

    -- Classification
    CASE
        WHEN total_energy - LAG(total_energy) OVER w > 0.01 THEN 'gaining'
        WHEN total_energy - LAG(total_energy) OVER w < -0.01 THEN 'losing'
        ELSE 'stable'
    END AS energy_trend,

    -- Energy role
    CASE
        WHEN energy_rank = 1 THEN 'dominant'
        WHEN energy_rank <= 3 THEN 'major'
        WHEN energy_fraction < 0.05 THEN 'minor'
        ELSE 'moderate'
    END AS energy_role

FROM v_energy_normalized
WINDOW w AS (PARTITION BY entity_id, signal_id ORDER BY I);


-- ============================================================================
-- SIGNAL ENERGY SUMMARY
-- ============================================================================
-- Summary statistics for each signal's energy behavior.

CREATE OR REPLACE VIEW v_signal_energy_summary AS
SELECT
    entity_id,
    signal_id,

    -- Energy statistics
    AVG(total_energy) AS mean_energy,
    MAX(total_energy) AS max_energy,
    MIN(total_energy) AS min_energy,
    STDDEV(total_energy) AS energy_volatility,

    -- Fraction statistics
    AVG(energy_fraction) AS mean_fraction,
    MAX(energy_fraction) AS max_fraction,

    -- Kinetic vs Potential balance
    AVG(kinetic_ratio) AS mean_kinetic_ratio,
    AVG(potential_ratio) AS mean_potential_ratio,

    -- Times as dominant
    SUM(CASE WHEN energy_rank = 1 THEN 1 ELSE 0 END) AS n_dominant,
    100.0 * SUM(CASE WHEN energy_rank = 1 THEN 1 ELSE 0 END) / COUNT(*) AS pct_dominant,

    -- Energy trend
    SUM(CASE WHEN energy_trend = 'gaining' THEN 1 ELSE 0 END) AS n_gaining,
    SUM(CASE WHEN energy_trend = 'losing' THEN 1 ELSE 0 END) AS n_losing,

    -- Signal role
    CASE
        WHEN AVG(energy_fraction) > 0.3 THEN 'primary_driver'
        WHEN AVG(kinetic_ratio) > 0.7 THEN 'high_activity'
        WHEN AVG(potential_ratio) > 0.7 THEN 'high_tension'
        WHEN AVG(energy_fraction) < 0.05 THEN 'low_contribution'
        ELSE 'balanced'
    END AS signal_character

FROM v_energy_dynamics
GROUP BY entity_id, signal_id;


-- ============================================================================
-- ENTITY ENERGY SUMMARY
-- ============================================================================
-- System-level energy statistics.

CREATE OR REPLACE VIEW v_entity_energy_summary AS
SELECT
    entity_id,
    COUNT(DISTINCT signal_id) AS n_signals,

    -- Total system energy
    AVG(SUM(total_energy) OVER (PARTITION BY entity_id, I)) AS mean_system_energy,

    -- Energy concentration
    AVG(MAX(energy_fraction) OVER (PARTITION BY entity_id, I)) AS mean_max_fraction,

    -- Number of signals that have been dominant
    COUNT(DISTINCT CASE WHEN energy_rank = 1 THEN signal_id END) AS n_ever_dominant,

    -- Energy distribution
    CASE
        WHEN AVG(MAX(energy_fraction) OVER (PARTITION BY entity_id, I)) > 0.5
        THEN 'concentrated'
        WHEN AVG(MAX(energy_fraction) OVER (PARTITION BY entity_id, I)) < 0.2
        THEN 'distributed'
        ELSE 'moderate'
    END AS energy_distribution

FROM v_energy_normalized
GROUP BY entity_id;


-- ============================================================================
-- ENERGY TRANSFER DETECTION
-- ============================================================================
-- Identify when energy appears to transfer between signals.

CREATE OR REPLACE VIEW v_energy_transfer AS
WITH ranked_changes AS (
    SELECT
        entity_id,
        I,
        signal_id,
        energy_delta,
        fraction_delta,
        RANK() OVER (PARTITION BY entity_id, I ORDER BY energy_delta DESC) AS gain_rank,
        RANK() OVER (PARTITION BY entity_id, I ORDER BY energy_delta ASC) AS loss_rank
    FROM v_energy_dynamics
    WHERE energy_delta IS NOT NULL
)
SELECT
    g.entity_id,
    g.I,
    g.signal_id AS gainer,
    l.signal_id AS loser,
    g.energy_delta AS gain_amount,
    l.energy_delta AS loss_amount,
    ABS(g.energy_delta + l.energy_delta) < 0.01 AS balanced_transfer

FROM ranked_changes g
JOIN ranked_changes l ON g.entity_id = l.entity_id AND g.I = l.I
WHERE g.gain_rank = 1
  AND l.loss_rank = 1
  AND g.energy_delta > 0.01
  AND l.energy_delta < -0.01;


-- ============================================================================
-- VERIFY
-- ============================================================================

.print ''
.print '=== VECTOR ENERGY ANALYSIS ==='
.print ''

SELECT
    entity_id,
    signal_id,
    signal_character,
    ROUND(mean_fraction * 100, 1) || '%' AS avg_share,
    ROUND(pct_dominant, 1) || '%' AS pct_dominant,
    ROUND(mean_kinetic_ratio * 100, 0) || '% kinetic' AS kinetic_share
FROM v_signal_energy_summary
ORDER BY entity_id, mean_fraction DESC
LIMIT 20;

.print ''
.print '=== ENERGY TRANSFERS (Recent) ==='

SELECT
    entity_id,
    I,
    loser || ' → ' || gainer AS transfer,
    ROUND(gain_amount, 4) AS amount,
    CASE WHEN balanced_transfer THEN 'balanced' ELSE 'net_change' END AS type
FROM v_energy_transfer
ORDER BY I DESC
LIMIT 10;
