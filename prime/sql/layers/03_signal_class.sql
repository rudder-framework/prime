-- ============================================================================
-- ENGINES: SIGNAL CLASSIFICATION
-- ============================================================================
-- Classify signals into types: analog, digital, periodic, event
-- Based on units (if provided) and data characteristics (if not)
-- ============================================================================

-- ============================================================================
-- 001: UNIT-BASED CLASSIFICATION
-- ============================================================================
-- First pass: use units to determine signal class

CREATE OR REPLACE VIEW v_class_from_units AS
SELECT
    cohort,
    signal_id,
    value_unit,
    CASE
        -- Analog units (continuous physical quantities)
        WHEN value_unit IN ('PSI', 'kPa', 'bar', 'Pa', 'atm') THEN 'analog'
        WHEN value_unit IN ('°C', 'K', '°F', 'C', 'F') THEN 'analog'
        WHEN value_unit IN ('V', 'mV', 'kV') THEN 'analog'
        WHEN value_unit IN ('A', 'mA', 'μA') THEN 'analog'
        WHEN value_unit IN ('m', 'cm', 'mm', 'km', 'ft', 'in') THEN 'analog'
        WHEN value_unit IN ('m/s', 'km/h', 'mph', 'ft/s') THEN 'analog'
        WHEN value_unit IN ('m/s²', 'g') THEN 'analog'
        WHEN value_unit IN ('kg', 'g', 'lb', 'oz') THEN 'analog'
        WHEN value_unit IN ('N', 'kN', 'lbf') THEN 'analog'
        WHEN value_unit IN ('W', 'kW', 'MW', 'hp') THEN 'analog'
        WHEN value_unit IN ('J', 'kJ', 'MJ', 'cal', 'BTU') THEN 'analog'
        WHEN value_unit IN ('Hz', 'kHz', 'MHz', 'GHz') THEN 'analog'
        WHEN value_unit IN ('rad', 'deg', '°') THEN 'analog'
        WHEN value_unit IN ('m³/s', 'L/min', 'gpm', 'cfm') THEN 'analog'
        WHEN value_unit IN ('kg/m³', 'g/cm³', 'lb/ft³') THEN 'analog'
        WHEN value_unit IN ('Pa·s', 'cP', 'cSt') THEN 'analog'
        WHEN value_unit IN ('%', 'ppm', 'ppb') THEN 'analog'
        WHEN value_unit IN ('pH', 'dB', 'dBA') THEN 'analog'
        WHEN value_unit IN ('lux', 'lm', 'cd') THEN 'analog'
        WHEN value_unit IN ('Ω', 'kΩ', 'MΩ') THEN 'analog'
        WHEN value_unit IN ('F', 'μF', 'nF', 'pF') THEN 'analog'
        WHEN value_unit IN ('H', 'mH', 'μH') THEN 'analog'
        WHEN value_unit IN ('T', 'G', 'mT') THEN 'analog'
        WHEN value_unit IN ('J/K', 'kJ/K') THEN 'analog'
        WHEN value_unit IN ('mm/s', 'in/s', 'ips') THEN 'analog'  -- vibration

        -- Digital units (discrete states)
        WHEN value_unit IN ('state', 'mode', 'status') THEN 'digital'
        WHEN value_unit IN ('bool', 'boolean', 'binary') THEN 'digital'
        WHEN value_unit IN ('enum', 'category', 'class') THEN 'digital'
        WHEN value_unit IN ('level', 'grade', 'rank') THEN 'digital'

        -- Event units (sparse occurrences)
        WHEN value_unit IN ('count', 'events', 'occurrences') THEN 'event'
        WHEN value_unit IN ('alarms', 'alerts', 'triggers') THEN 'event'
        WHEN value_unit IN ('clicks', 'actions', 'transactions') THEN 'event'

        -- Unknown - will infer from data
        ELSE 'unknown'
    END AS unit_class,

    -- Is interpolation valid based on unit?
    CASE
        WHEN value_unit IN ('state', 'mode', 'status', 'bool', 'boolean', 'binary',
                           'enum', 'category', 'class', 'count', 'events',
                           'occurrences', 'alarms', 'alerts', 'triggers') THEN FALSE
        ELSE TRUE
    END AS interpolation_valid_from_unit
FROM v_base
GROUP BY cohort, signal_id, value_unit;


-- ============================================================================
-- 002: CONTINUITY DETECTION (analog vs digital from data)
-- ============================================================================

CREATE OR REPLACE VIEW v_class_continuity AS
SELECT
    cohort,
    signal_id,

    -- Unique value ratio (low = discrete)
    COUNT(DISTINCT ROUND(value, 4))::FLOAT / COUNT(*) AS unique_ratio,

    -- Is all integer?
    BOOL_AND(value = ROUND(value)) AS is_all_integer,

    -- Number of distinct values
    COUNT(DISTINCT ROUND(value, 2)) AS n_distinct_values,

    -- Range
    MAX(value) - MIN(value) AS value_range,

    -- Distinct values per unit range
    COUNT(DISTINCT ROUND(value, 2))::FLOAT / NULLIF(MAX(value) - MIN(value), 0) AS values_per_range

FROM v_base
GROUP BY cohort, signal_id;


-- ============================================================================
-- 003: SMOOTHNESS DETECTION (from curvature)
-- ============================================================================

CREATE OR REPLACE VIEW v_class_smoothness AS
SELECT
    cohort,
    signal_id,
    AVG(ABS(dvalue)) AS mean_abs_dvalue,
    STDDEV(dvalue) AS std_dvalue,
    AVG(kappa) AS mean_kappa,
    STDDEV(kappa) AS std_kappa,
    STDDEV(dvalue) / NULLIF(AVG(ABS(dvalue)), 0) AS roughness,
    1 - STDDEV(kappa) / NULLIF(AVG(kappa) + 0.001, 0) AS kappa_consistency
FROM v_curvature
WHERE dvalue IS NOT NULL AND kappa IS NOT NULL
GROUP BY cohort, signal_id;


-- ============================================================================
-- 004: PERIODICITY DETECTION (from d2value sign changes)
-- ============================================================================

CREATE OR REPLACE VIEW v_class_periodicity AS
WITH sign_changes AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        d2value,
        SIGN(d2value) AS d2value_sign,
        CASE
            WHEN SIGN(d2value) != SIGN(LAG(d2value) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0))
            THEN 1 ELSE 0
        END AS is_sign_change
    FROM v_d2value
    WHERE d2value IS NOT NULL
),
change_indices AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        ROW_NUMBER() OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS change_num
    FROM sign_changes
    WHERE is_sign_change = 1
),
periods AS (
    SELECT
        a.cohort,
        a.signal_id,
        b.signal_0 - a.signal_0 AS half_period
    FROM change_indices a
    JOIN change_indices b ON a.cohort = b.cohort AND a.signal_id = b.signal_id AND b.change_num = a.change_num + 1
)
SELECT
    cohort,
    signal_id,
    COUNT(*) AS n_half_periods,
    AVG(half_period) * 2 AS estimated_period,
    STDDEV(half_period) / NULLIF(AVG(half_period), 0) AS period_variability,
    CASE
        WHEN STDDEV(half_period) / NULLIF(AVG(half_period), 0) < 0.2
             AND COUNT(*) > 5
        THEN TRUE
        ELSE FALSE
    END AS is_periodic
FROM periods
GROUP BY cohort, signal_id;


-- ============================================================================
-- 005: SPARSITY DETECTION (for event signals)
-- ============================================================================

CREATE OR REPLACE VIEW v_class_sparsity AS
SELECT
    cohort,
    signal_id,
    COUNT(*) AS n_total,
    SUM(CASE WHEN value = 0 OR value IS NULL THEN 1 ELSE 0 END) AS n_zeros,
    SUM(CASE WHEN value != 0 AND value IS NOT NULL THEN 1 ELSE 0 END) AS n_nonzero,
    SUM(CASE WHEN value = 0 OR value IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) AS sparsity,
    CASE
        WHEN SUM(CASE WHEN value = 0 OR value IS NULL THEN 1 ELSE 0 END)::FLOAT / COUNT(*) > 0.9 THEN TRUE
        ELSE FALSE
    END AS is_sparse_event
FROM v_base
GROUP BY cohort, signal_id;


-- ============================================================================
-- 006: FINAL SIGNAL CLASSIFICATION
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_class AS
SELECT
    u.cohort,
    u.signal_id,
    u.value_unit,

    -- Classification priority: unit > periodicity > sparsity > continuity
    CASE
        -- Unit-based (highest priority if known)
        WHEN u.unit_class != 'unknown' THEN u.unit_class

        -- Event (sparse signals)
        WHEN sp.is_sparse_event THEN 'event'

        -- Periodic (consistent oscillation)
        WHEN p.is_periodic THEN 'periodic'

        -- Digital (discrete values, low unique ratio, all integer)
        WHEN c.is_all_integer AND c.unique_ratio < 0.05 THEN 'digital'
        WHEN c.n_distinct_values < 10 AND c.is_all_integer THEN 'digital'

        -- Analog (default for continuous)
        ELSE 'analog'
    END AS signal_class,

    -- Classification source
    CASE
        WHEN u.unit_class != 'unknown' THEN 'unit'
        WHEN sp.is_sparse_event THEN 'sparsity'
        WHEN p.is_periodic THEN 'periodicity'
        WHEN c.is_all_integer AND c.unique_ratio < 0.05 THEN 'continuity'
        WHEN c.n_distinct_values < 10 AND c.is_all_integer THEN 'continuity'
        ELSE 'default'
    END AS class_source,

    -- Derived properties
    CASE
        WHEN u.unit_class = 'digital' THEN FALSE
        WHEN u.unit_class = 'event' THEN FALSE
        WHEN sp.is_sparse_event THEN FALSE
        WHEN c.is_all_integer AND c.unique_ratio < 0.05 THEN FALSE
        ELSE TRUE
    END AS interpolation_valid,

    CASE
        WHEN p.is_periodic THEN TRUE
        ELSE FALSE
    END AS is_periodic,

    p.estimated_period,

    CASE
        WHEN u.unit_class IN ('analog', 'periodic') THEN 3
        WHEN p.is_periodic THEN 3
        WHEN c.is_all_integer THEN 0
        ELSE 2
    END AS max_derivative_order,

    -- Supporting metrics
    c.unique_ratio,
    c.is_all_integer,
    c.n_distinct_values,
    sp.sparsity,
    sm.kappa_consistency,
    p.period_variability

FROM v_class_from_units u
LEFT JOIN v_class_continuity c USING (cohort, signal_id)
LEFT JOIN v_class_periodicity p USING (cohort, signal_id)
LEFT JOIN v_class_sparsity sp USING (cohort, signal_id)
LEFT JOIN v_class_smoothness sm USING (cohort, signal_id);


-- ============================================================================
-- 007: INDEX DIMENSION PROPERTIES
-- ============================================================================

CREATE OR REPLACE VIEW v_index_properties AS
SELECT
    cohort,
    signal_id,
    index_dimension,

    -- What calculus means for this dimension
    CASE index_dimension
        WHEN 'time' THEN 'velocity/acceleration'
        WHEN 'space' THEN 'gradient/laplacian'
        WHEN 'frequency' THEN 'spectral_slope'
        WHEN 'scale' THEN 'scale_derivative'
        ELSE 'unknown'
    END AS derivative_meaning,

    -- What relationships mean
    CASE index_dimension
        WHEN 'time' THEN 'causal'
        WHEN 'space' THEN 'coupling'
        WHEN 'frequency' THEN 'harmonic'
        WHEN 'scale' THEN 'hierarchical'
        ELSE 'correlation'
    END AS relationship_type,

    -- Is causality directional?
    CASE index_dimension
        WHEN 'time' THEN TRUE
        WHEN 'space' THEN FALSE
        WHEN 'frequency' THEN FALSE
        WHEN 'scale' THEN FALSE
        ELSE FALSE
    END AS causality_directional

FROM v_base
GROUP BY cohort, signal_id, index_dimension;


-- ============================================================================
-- CLASSIFICATION SUMMARY
-- ============================================================================

CREATE OR REPLACE VIEW v_classification_complete AS
SELECT
    sc.cohort,
    sc.signal_id,
    sc.value_unit,
    sc.signal_class,
    sc.class_source,
    sc.interpolation_valid,
    sc.is_periodic,
    sc.estimated_period,
    sc.max_derivative_order,
    ip.index_dimension,
    ip.derivative_meaning,
    ip.relationship_type,
    ip.causality_directional,
    sc.unique_ratio,
    sc.sparsity,
    sc.kappa_consistency
FROM v_signal_class sc
LEFT JOIN v_index_properties ip USING (cohort, signal_id);
