-- ============================================================================
-- 24_incident_summary.sql
-- ============================================================================
-- CAPSTONE REPORT: Full Incident Synthesis
--
-- The one report that summarizes the entire pipeline. Pulls findings from
-- all upstream layers and reports into a 12-section diagnostic narrative.
--
-- Section  1: System Identity         <- observations, typology
-- Section  2: System Status           <- baseline deviation (report 23)
-- Section  3: Dynamics Classification <- FTLE, Lyapunov
-- Section  4: Drift Incidents         <- observations
-- Section  5: Canary Sequence         <- layer 13 canary views
-- Section  6: Geometry Health         <- layers 12, 17
-- Section  7: Coupling Events         <- layer 16
-- Section  8: Hunting / Oscillation   <- observations
-- Section  9: Energy Balance          <- physics (report 23), geometry_dynamics
-- Section 10: Topology Status         <- persistent_homology
-- Section 11: Confidence Assessment   <- baselines (report 23), observations
-- Section 12: Information Flow        <- information_flow
--
-- Sections 1-2: headline (act or don't)
-- Sections 3-8: what and where (investigate)
-- Sections 9-12: deeper context (understand)
--
-- Dependencies (views from upstream layers/reports):
--   Layers:    v_brittleness (12), v_canary_sequence (13),
--              v_coupling_ranked (16), v_dimension_trajectory (17)
--   Report 23: physics, baselines, v_deviation_entity_summary
--   Parquets:  ftle, ftle_backward, ftle_rolling, lyapunov,
--              thermodynamics, persistent_homology, information_flow,
--              state_geometry, geometry_dynamics
-- ============================================================================


.print ''
.print '============================================================================'
.print '                     INCIDENT SYNTHESIS REPORT                              '
.print '============================================================================'


-- ============================================================================
-- SECTION 1: SYSTEM IDENTITY
-- "What was analyzed?"
-- ============================================================================

.print ''
.print '-- Section 1: System Identity ---------------------------------------------'

-- 1a. System snapshot
WITH signal_counts AS (
    SELECT
        COUNT(DISTINCT signal_id) AS n_signals,
        COUNT(DISTINCT cohort) AS n_cohorts,
        COUNT(*) AS n_observations,
        ROUND(MAX(signal_0) - MIN(signal_0), 1) AS signal_0_range
    FROM observations
),
geometry_counts AS (
    SELECT COUNT(*) AS n_geometry_windows
    FROM state_geometry
)
SELECT
    sc.n_signals,
    sc.n_cohorts,
    sc.n_observations,
    sc.signal_0_range,
    gc.n_geometry_windows
FROM signal_counts sc
CROSS JOIN geometry_counts gc;

-- 1b. Signal typology card
SELECT
    t.signal_id,
    t.continuity,
    t.memory_class,
    t.temporal_primary,
    t.complexity_class
FROM typology t
ORDER BY t.signal_id;


-- ============================================================================
-- SECTION 2: SYSTEM STATUS
-- "Is the system healthy, shifted, or departed?"
-- ============================================================================

.print ''
.print '-- Section 2: System Status -----------------------------------------------'

SELECT
    CASE
        WHEN SUM(CASE WHEN departure_assessment = 'departed' THEN 1 ELSE 0 END) > 0
            THEN 'DEPARTED'
        WHEN SUM(CASE WHEN departure_assessment = 'unstable' THEN 1 ELSE 0 END) > 0
            THEN 'SHIFTED'
        WHEN SUM(CASE WHEN departure_assessment = 'insufficient_baseline' AND pct_abnormal > 50 THEN 1 ELSE 0 END) > 0
            THEN 'DEPARTED (insufficient baseline)'
        WHEN SUM(CASE WHEN departure_assessment = 'insufficient_baseline' THEN 1 ELSE 0 END) > 0
            THEN 'INSUFFICIENT_BASELINE'
        ELSE 'STABLE'
    END AS system_status,
    SUM(CASE WHEN departure_assessment = 'stable' THEN 1 ELSE 0 END) AS cohorts_stable,
    SUM(CASE WHEN departure_assessment IN ('noisy', 'unstable') THEN 1 ELSE 0 END) AS cohorts_watch,
    SUM(CASE WHEN departure_assessment IN ('departed', 'insufficient_baseline') THEN 1 ELSE 0 END) AS cohorts_alert,
    ROUND(AVG(pct_abnormal), 1) AS avg_pct_abnormal
FROM v_deviation_entity_summary;


-- ============================================================================
-- SECTION 3: DYNAMICS CLASSIFICATION
-- "What kind of dynamical system is this?"
-- ============================================================================

.print ''
.print '-- Section 3: Dynamics Classification -------------------------------------'

WITH ftle_trend AS (
    SELECT
        signal_id,
        cohort,
        CASE
            WHEN COUNT(CASE WHEN ftle IS NOT NULL AND NOT isnan(ftle) THEN 1 END) < 3
                THEN 'INSUFFICIENT_DATA'
            WHEN REGR_SLOPE(ftle, signal_0_center) > 0.0001 THEN 'DESTABILIZING'
            WHEN REGR_SLOPE(ftle, signal_0_center) < -0.0001 THEN 'STABILIZING'
            ELSE 'STATIONARY'
        END AS ftle_trend
    FROM ftle_rolling
    GROUP BY signal_id, cohort
)
SELECT
    f.signal_id,
    ROUND(f.ftle, 4) AS ftle,
    ROUND(l.lyapunov, 4) AS lyapunov_exponent,
    CASE
        WHEN f.ftle IS NULL OR isnan(f.ftle) THEN 'INSUFFICIENT_DATA'
        WHEN f.n_samples < 1000 THEN 'INSUFFICIENT_DATA'
        WHEN f.ftle > 0.05 THEN 'CHAOTIC'
        WHEN f.ftle > 0.01 THEN 'WEAKLY_CHAOTIC'
        WHEN f.ftle > -0.01 THEN 'MARGINAL'
        ELSE 'STABLE'
    END AS dynamics_class,
    CASE
        WHEN f.ftle IS NULL OR isnan(f.ftle) THEN 'UNKNOWN'
        WHEN f.n_samples < 1000 THEN 'UNKNOWN'
        WHEN f.ftle > 0 AND b.ftle < 0 THEN 'ATTRACTOR_WITH_SENSITIVITY'
        WHEN f.ftle > 0 AND b.ftle > 0 THEN 'FULLY_UNSTABLE'
        WHEN f.ftle < 0 AND b.ftle < 0 THEN 'STRONGLY_STABLE'
        ELSE 'MIXED'
    END AS stability_type,
    COALESCE(ft.ftle_trend, 'NO_DATA') AS ftle_trend
FROM ftle f
LEFT JOIN ftle_backward b ON f.signal_id = b.signal_id AND f.cohort = b.cohort
LEFT JOIN lyapunov l ON f.signal_id = l.signal_id AND f.cohort = l.cohort
LEFT JOIN ftle_trend ft ON f.signal_id = ft.signal_id AND f.cohort = ft.cohort
ORDER BY f.ftle DESC;


-- ============================================================================
-- SECTION 4: DRIFT INCIDENTS
-- "Which signals are trending, and is the trend persistent?"
-- ============================================================================

.print ''
.print '-- Section 4: Drift Incidents ---------------------------------------------'

WITH drift_windows AS (
    SELECT
        cohort, signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        value, signal_0
    FROM observations
),
drift_slopes AS (
    SELECT cohort, signal_id, window_id,
        REGR_SLOPE(value, signal_0) AS window_slope
    FROM drift_windows
    WHERE window_id > 1
    GROUP BY cohort, signal_id, window_id
),
drift_agg AS (
    SELECT
        cohort, signal_id,
        SUM(CASE WHEN window_slope > 0 THEN 1 ELSE 0 END) AS wp,
        SUM(CASE WHEN window_slope < 0 THEN 1 ELSE 0 END) AS wn,
        COUNT(*) AS tw,
        ROUND(AVG(window_slope), 6) AS avg_slope
    FROM drift_slopes
    GROUP BY cohort, signal_id
),
findings AS (
    SELECT
        cohort, signal_id,
        CASE
            WHEN wp = tw THEN 'PERSISTENT_ACCELERATION'
            WHEN wn = tw THEN 'PERSISTENT_DECELERATION'
            WHEN wp >= tw - 1 THEN 'MOSTLY_ACCELERATING'
            WHEN wn >= tw - 1 THEN 'MOSTLY_DECELERATING'
        END AS drift_type,
        wp || '/' || tw AS windows_positive,
        avg_slope
    FROM drift_agg
    WHERE wp >= tw - 1 OR wn >= tw - 1
)
SELECT * FROM (
    SELECT cohort, signal_id, drift_type, windows_positive, avg_slope
    FROM findings
    UNION ALL
    SELECT NULL, NULL, 'NONE_DETECTED', NULL, NULL
    WHERE NOT EXISTS (SELECT 1 FROM findings)
) combined
ORDER BY ABS(avg_slope) DESC NULLS LAST
LIMIT 20;


-- ============================================================================
-- SECTION 5: CANARY SEQUENCE
-- "What departed first, and how fast did it propagate?"
-- ============================================================================

.print ''
.print '-- Section 5: Canary Sequence ---------------------------------------------'

-- 5a. First movers
WITH canary_enriched AS (
    SELECT
        cohort,
        signal_id,
        canary_rank,
        ROUND(first_departure_I, 1) AS onset_I,
        ROUND(departure_magnitude, 6) AS departure_magnitude,
        ROUND(
            first_departure_I - MIN(first_departure_I) OVER (PARTITION BY cohort),
        1) AS delay_from_first
    FROM v_canary_sequence
),
findings AS (
    SELECT * FROM canary_enriched WHERE canary_rank <= 5
)
SELECT cohort, signal_id, canary_rank, onset_I, delay_from_first, departure_magnitude
FROM findings
UNION ALL
SELECT NULL, 'NONE_DEPARTED', NULL, NULL, NULL, NULL
WHERE NOT EXISTS (SELECT 1 FROM findings)
ORDER BY cohort NULLS LAST, canary_rank NULLS LAST;

-- 5b. Propagation speed (rank 1 to rank 2 delay)
SELECT
    cohort,
    MIN(CASE WHEN canary_rank = 1 THEN signal_id END) AS first_signal,
    MIN(CASE WHEN canary_rank = 2 THEN signal_id END) AS second_signal,
    ROUND(MIN(CASE WHEN canary_rank = 1 THEN first_departure_I END), 1) AS first_signal_I,
    ROUND(MIN(CASE WHEN canary_rank = 2 THEN first_departure_I END), 1) AS second_signal_I,
    ROUND(
        MIN(CASE WHEN canary_rank = 2 THEN first_departure_I END) -
        MIN(CASE WHEN canary_rank = 1 THEN first_departure_I END)
    , 1) AS propagation_delay
FROM v_canary_sequence
WHERE canary_rank <= 2
GROUP BY cohort
HAVING MIN(CASE WHEN canary_rank = 2 THEN first_departure_I END) IS NOT NULL;


-- ============================================================================
-- SECTION 6: GEOMETRY HEALTH
-- "Is the system's geometric structure intact?"
-- ============================================================================

.print ''
.print '-- Section 6: Geometry Health ---------------------------------------------'

-- 6a. Brittleness summary
SELECT
    cohort,
    ROUND(AVG(brittleness_score), 4) AS avg_brittleness,
    ROUND(MAX(brittleness_score), 4) AS max_brittleness,
    ROUND(AVG(effective_dim), 2) AS avg_effective_dim,
    ROUND(AVG(condition_number), 2) AS avg_condition_number,
    ROUND(AVG(temperature), 4) AS avg_temperature,
    CASE
        WHEN AVG(brittleness_score) > 100 THEN 'CRITICALLY_BRITTLE'
        WHEN AVG(brittleness_score) > 30  THEN 'ELEVATED'
        WHEN AVG(brittleness_score) > 10  THEN 'MODERATE'
        ELSE 'HEALTHY'
    END AS brittleness_status
FROM v_brittleness
GROUP BY cohort
ORDER BY AVG(brittleness_score) DESC NULLS LAST;

-- 6b. Dimension trajectory
SELECT
    cohort,
    early_dim,
    late_dim,
    dim_delta,
    trajectory_type
FROM v_dimension_trajectory
ORDER BY ABS(dim_delta) DESC;


-- ============================================================================
-- SECTION 7: COUPLING EVENTS
-- "Are signal relationships changing?"
-- ============================================================================

.print ''
.print '-- Section 7: Coupling Events ---------------------------------------------'

WITH coupling_findings AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        signal_0_end,
        ROUND(correlation, 4) AS correlation,
        ROUND(coupling_delta, 4) AS delta,
        CASE
            WHEN ABS(coupling_delta) > 0.5 THEN 'MAJOR_DECOUPLING'
            WHEN ABS(coupling_delta) > 0.3 THEN 'MODERATE_SHIFT'
        END AS event_severity
    FROM v_coupling_ranked
    WHERE coupling_delta IS NOT NULL
      AND is_sign_flip = FALSE
      AND ABS(coupling_delta) > 0.3
)
SELECT * FROM (
    SELECT cohort, signal_a, signal_b, signal_0_end, correlation, delta, event_severity
    FROM coupling_findings
    UNION ALL
    SELECT NULL, NULL, NULL, NULL, NULL, NULL, 'NONE_DETECTED'
    WHERE NOT EXISTS (SELECT 1 FROM coupling_findings)
) combined
ORDER BY ABS(delta) DESC NULLS LAST
LIMIT 10;


-- ============================================================================
-- SECTION 8: HUNTING / OSCILLATION ALERTS
-- "Any signals showing control instability?"
-- ============================================================================

.print ''
.print '-- Section 8: Hunting / Oscillation ---------------------------------------'

WITH centered AS (
    SELECT
        cohort, signal_id, signal_0,
        value - AVG(value) OVER (PARTITION BY cohort, signal_id) AS deviation,
        STDDEV_POP(value) OVER (PARTITION BY cohort, signal_id) AS std_val
    FROM observations
),
direction_changes AS (
    SELECT
        cohort, signal_id, signal_0, std_val,
        CASE
            WHEN SIGN(deviation) != LAG(SIGN(deviation))
                OVER (PARTITION BY cohort, signal_id ORDER BY signal_0)
            THEN 1 ELSE 0
        END AS is_reversal,
        ABS(deviation) AS abs_deviation
    FROM centered
),
hunting_stats AS (
    SELECT
        cohort, signal_id,
        ROUND(1.0 * SUM(is_reversal) / COUNT(*), 4) AS reversal_rate,
        AVG(abs_deviation) AS avg_excursion,
        MAX(std_val) AS std_val,
        CASE
            WHEN 1.0 * SUM(is_reversal) / COUNT(*) > 0.3 THEN 'SEVERE_HUNTING'
            WHEN 1.0 * SUM(is_reversal) / COUNT(*) > 0.2 THEN 'MODERATE_HUNTING'
            WHEN 1.0 * SUM(is_reversal) / COUNT(*) > 0.1 THEN 'MILD_HUNTING'
            ELSE 'WITHIN_BASELINE'
        END AS hunting_severity,
        CASE
            WHEN 1.0 * SUM(is_reversal) / COUNT(*) > 0.3 THEN 'HIGH_FREQUENCY_REVERSALS'
            WHEN 1.0 * SUM(is_reversal) / COUNT(*) > 0.2 THEN 'ELEVATED_REVERSAL_RATE'
            ELSE 'WITHIN_BASELINE'
        END AS recommendation
    FROM direction_changes
    GROUP BY cohort, signal_id
),
findings AS (
    SELECT h.cohort, h.signal_id, h.reversal_rate, h.hunting_severity, h.recommendation
    FROM hunting_stats h
    LEFT JOIN typology t ON h.signal_id = t.signal_id
    WHERE h.hunting_severity NOT IN ('WITHIN_BASELINE')
      AND (t.continuity IS NULL OR t.continuity NOT IN ('CONSTANT', 'DISCRETE', 'EVENT'))
      AND h.reversal_rate > 0.4
      AND h.avg_excursion > h.std_val
)
SELECT * FROM (
    SELECT cohort, signal_id, reversal_rate, hunting_severity, recommendation
    FROM findings
    UNION ALL
    SELECT NULL AS cohort, '' AS signal_id, 0.0 AS reversal_rate,
        'NONE_DETECTED' AS hunting_severity, '' AS recommendation
    WHERE NOT EXISTS (SELECT 1 FROM findings)
) combined
ORDER BY reversal_rate DESC NULLS LAST;


-- ============================================================================
-- SECTION 9: ENERGY BALANCE
-- "Is energy entering or leaving the system unmeasured?"
-- ============================================================================

.print ''
.print '-- Section 9: Energy Balance ----------------------------------------------'

-- 9a. Energy accounting
WITH energy_changes AS (
    SELECT
        cohort,
        signal_0_center,
        energy_proxy,
        energy_proxy - LAG(energy_proxy) OVER (
            PARTITION BY cohort ORDER BY signal_0_center
        ) AS energy_delta
    FROM physics
),
period_sums AS (
    SELECT
        cohort,
        SUM(CASE WHEN energy_delta > 0 THEN energy_delta ELSE 0 END) AS energy_injected,
        SUM(CASE WHEN energy_delta < 0 THEN ABS(energy_delta) ELSE 0 END) AS energy_dissipated,
        SUM(energy_delta) AS net_energy_change
    FROM energy_changes
    WHERE energy_delta IS NOT NULL
    GROUP BY cohort
)
SELECT
    cohort,
    ROUND(energy_injected, 4) AS energy_injected,
    ROUND(energy_dissipated, 4) AS energy_dissipated,
    ROUND(net_energy_change, 4) AS net_energy_change,
    ROUND(energy_dissipated - energy_injected, 4) AS energy_gap,
    CASE
        WHEN ABS(energy_dissipated - energy_injected) < 0.01 THEN 'balanced'
        WHEN energy_dissipated > energy_injected THEN 'deficit_unmeasured_sink'
        ELSE 'surplus_unmeasured_source'
    END AS energy_balance_status,
    CASE
        WHEN ABS(energy_dissipated - energy_injected) < 0.01 THEN 'Energy budget closed'
        WHEN energy_dissipated > energy_injected THEN 'Energy leaving system unmeasured'
        ELSE 'Energy entering system unmeasured'
    END AS interpretation
FROM period_sums;

-- 9b. Thermal and entropy trend
WITH energy_trajectory AS (
    SELECT
        cohort,
        signal_0_center,
        total_variance AS energy,
        total_variance / NULLIF(effective_dim, 0) AS temperature_inst
    FROM geometry_dynamics
    WHERE total_variance IS NOT NULL
),
thermal_agg AS (
    SELECT
        cohort,
        ROUND(AVG(temperature_inst), 4) AS avg_temperature,
        CASE
            WHEN REGR_SLOPE(energy, signal_0_center) > 0.001 THEN 'HEATING'
            WHEN REGR_SLOPE(energy, signal_0_center) < -0.001 THEN 'COOLING'
            ELSE 'THERMAL_EQUILIBRIUM'
        END AS thermal_trend
    FROM energy_trajectory
    GROUP BY cohort
),
entropy_agg AS (
    SELECT
        cohort,
        CASE
            WHEN REGR_SLOPE(eigenvalue_entropy_normalized, signal_0_center) > 0.0001
                THEN 'INCREASING_DISORDER'
            WHEN REGR_SLOPE(eigenvalue_entropy_normalized, signal_0_center) < -0.0001
                THEN 'INCREASING_ORDER'
            ELSE 'ENTROPY_STABLE'
        END AS entropy_trend_label
    FROM state_geometry
    WHERE eigenvalue_entropy_normalized IS NOT NULL
      AND NOT isnan(eigenvalue_entropy_normalized)
    GROUP BY cohort
)
SELECT
    t.cohort,
    t.thermal_trend,
    e.entropy_trend_label,
    t.avg_temperature
FROM thermal_agg t
LEFT JOIN entropy_agg e ON t.cohort = e.cohort;


-- ============================================================================
-- SECTION 10: TOPOLOGY STATUS
-- "Is the attractor's shape changing?"
-- ============================================================================

.print ''
.print '-- Section 10: Topology Status --------------------------------------------'

WITH lifecycle AS (
    SELECT cohort,
        MIN(signal_0_end) AS min_I,
        MAX(signal_0_end) AS max_I
    FROM persistent_homology
    GROUP BY cohort
),
early_late AS (
    SELECT
        ph.cohort,
        AVG(CASE WHEN ph.signal_0_end <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_0 END) AS early_b0,
        AVG(CASE WHEN ph.signal_0_end >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_0 END) AS late_b0,
        AVG(CASE WHEN ph.signal_0_end <= lc.min_I + (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_1 END) AS early_b1,
        AVG(CASE WHEN ph.signal_0_end >= lc.max_I - (lc.max_I - lc.min_I) * 0.3
            THEN ph.betti_1 END) AS late_b1
    FROM persistent_homology ph
    JOIN lifecycle lc ON ph.cohort = lc.cohort
    GROUP BY ph.cohort
)
SELECT
    cohort,
    ROUND(early_b0, 2) AS early_b0,
    ROUND(late_b0, 2) AS late_b0,
    ROUND(early_b1, 2) AS early_b1,
    ROUND(late_b1, 2) AS late_b1,
    CASE
        WHEN late_b1 - early_b1 > 0.5 THEN 'LOOPS_EMERGED'
        WHEN late_b1 - early_b1 < -0.5 THEN 'LOOPS_COLLAPSED'
        WHEN late_b0 - early_b0 > 0.5 THEN 'COMPONENTS_FRAGMENTING'
        WHEN late_b0 - early_b0 < -0.5 THEN 'COMPONENTS_MERGING'
        ELSE 'TOPOLOGY_PRESERVED'
    END AS topology_evolution
FROM early_late
WHERE early_b0 IS NOT NULL AND late_b0 IS NOT NULL
ORDER BY ABS(late_b1 - early_b1) DESC;


-- ============================================================================
-- SECTION 11: CONFIDENCE ASSESSMENT
-- "How much should we trust these findings?"
-- ============================================================================

.print ''
.print '-- Section 11: Confidence Assessment --------------------------------------'

-- 11a. Baseline quality
SELECT
    COUNT(*) AS total_cohorts,
    SUM(CASE WHEN n_baseline_points >= 50 THEN 1 ELSE 0 END) AS valid_baselines,
    ROUND(
        100.0 * SUM(CASE WHEN n_baseline_points >= 50 THEN 1 ELSE 0 END) / COUNT(*),
    1) AS pct_valid,
    CASE
        WHEN MIN(n_baseline_points) >= 50 THEN 'GOOD'
        WHEN MIN(n_baseline_points) >= 20 THEN 'ADEQUATE'
        ELSE 'POOR'
    END AS baseline_quality
FROM baselines;

-- 11b. Fleet validity
WITH fleet AS (
    SELECT COUNT(DISTINCT cohort) AS fleet_size
    FROM observations
)
SELECT
    fleet_size,
    CASE
        WHEN fleet_size >= 10 THEN 'LARGE'
        WHEN fleet_size >= 3 THEN 'MODERATE'
        WHEN fleet_size > 1 THEN 'SMALL'
        ELSE 'MINIMAL'
    END AS fleet_validity,
    CASE
        WHEN fleet_size >= 10 THEN 'Fleet-relative percentiles are reliable'
        WHEN fleet_size >= 3 THEN 'Fleet percentiles should be interpreted with caution'
        WHEN fleet_size > 1 THEN 'Limited fleet comparison available'
        ELSE 'No fleet comparison — self-referential baselines only'
    END AS recommendation
FROM fleet;


-- ============================================================================
-- SECTION 12: INFORMATION FLOW
-- "Where is information flowing in the system?"
-- ============================================================================

.print ''
.print '-- Section 12: Information Flow -------------------------------------------'

WITH flow_ranked AS (
    SELECT
        cohort,
        signal_a,
        signal_b,
        ROUND(transfer_entropy_a_to_b, 4) AS te_a_to_b,
        ROUND(transfer_entropy_b_to_a, 4) AS te_b_to_a,
        ROUND(transfer_entropy_a_to_b - transfer_entropy_b_to_a, 4) AS net_te,
        CASE
            WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05 THEN 'BIDIRECTIONAL'
            WHEN granger_p_a_to_b < 0.05 THEN 'A_DRIVES_B'
            WHEN granger_p_b_to_a < 0.05 THEN 'B_DRIVES_A'
            ELSE 'INDEPENDENT'
        END AS causal_direction,
        CASE
            WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05
                AND ABS(transfer_entropy_a_to_b - transfer_entropy_b_to_a) < 0.005
            THEN 'SYMMETRIC_FEEDBACK'
            WHEN granger_p_a_to_b < 0.05 AND granger_p_b_to_a < 0.05
            THEN 'ASYMMETRIC_FEEDBACK'
            ELSE NULL
        END AS feedback_type,
        CASE
            WHEN ABS(transfer_entropy_a_to_b - transfer_entropy_b_to_a) > 0.05
                THEN 'STRONG_ASYMMETRY'
            WHEN ABS(transfer_entropy_a_to_b - transfer_entropy_b_to_a) > 0.01
                THEN 'MODERATE_ASYMMETRY'
            ELSE 'BALANCED'
        END AS flow_balance
    FROM information_flow
    WHERE transfer_entropy_a_to_b > 0.01 OR transfer_entropy_b_to_a > 0.01
),
findings AS (
    SELECT cohort, signal_a, signal_b, causal_direction, feedback_type, net_te, flow_balance
    FROM flow_ranked
)
SELECT * FROM (
    SELECT cohort, signal_a, signal_b, causal_direction, feedback_type, net_te, flow_balance
    FROM findings
    UNION ALL
    SELECT NULL, NULL, NULL, 'NONE_DETECTED', NULL, NULL, NULL
    WHERE NOT EXISTS (SELECT 1 FROM findings)
) combined
ORDER BY ABS(net_te) DESC NULLS LAST;
