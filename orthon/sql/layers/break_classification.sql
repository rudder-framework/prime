-- ============================================================
-- BREAK CLASSIFICATION (ORTHON)
-- ============================================================
-- Interprets Engines break detection output.
-- Classifies breaks as Heaviside (step), Dirac (impulse),
-- regime change, or gradual shift.
--
-- Input: breaks.parquet from Engines
-- ============================================================

-- ============================================================
-- BREAK TYPE CLASSIFICATION
-- ============================================================
-- Decision logic:
--   duration=1 + high sharpness → IMPULSE (Dirac)
--   duration=1 + returns to pre_level → IMPULSE (confirmed)
--   low duration + sustained level change → STEP (Heaviside)
--   high duration + sustained change → GRADUAL_SHIFT
--   clustered breaks → REGIME_CHANGE
-- ============================================================

CREATE OR REPLACE VIEW v_break_type AS
SELECT
    signal_id,
    I,
    magnitude,
    direction,
    sharpness,
    duration,
    pre_level,
    post_level,
    snr,

    -- Level change: did the signal stay at the new level?
    ABS(post_level - pre_level) AS level_change,

    -- Break type classification
    CASE
        -- Dirac impulse: sharp, short, returns to baseline
        WHEN duration <= 2
             AND sharpness > 3.0
             AND ABS(post_level - pre_level) / NULLIF(ABS(magnitude), 0) < 0.3
        THEN 'IMPULSE'

        -- Heaviside step: sharp transition to sustained new level
        WHEN duration <= 5
             AND ABS(post_level - pre_level) / NULLIF(ABS(magnitude), 0) > 0.5
        THEN 'STEP'

        -- Gradual shift: slow transition to new level
        WHEN duration > 5
             AND ABS(post_level - pre_level) / NULLIF(ABS(magnitude), 0) > 0.5
        THEN 'GRADUAL_SHIFT'

        -- Transient: brief deviation, doesn't clearly fit
        ELSE 'TRANSIENT'
    END AS break_type,

    -- Severity (same MAD-based scale as anomaly detection)
    CASE
        WHEN ABS(magnitude) > 5.0 THEN 'CRITICAL'
        WHEN ABS(magnitude) > 3.5 THEN 'SEVERE'
        WHEN ABS(magnitude) > 2.5 THEN 'MODERATE'
        WHEN ABS(magnitude) > 2.0 THEN 'MILD'
        ELSE 'MINOR'
    END AS severity

FROM breaks
ORDER BY signal_id, I;


-- ============================================================
-- REGIME DETECTION (from break clustering)
-- ============================================================
-- A regime is a period between consecutive STEP breaks.
-- If steps cluster tightly, that's a REGIME_CHANGE event.
-- ============================================================

CREATE OR REPLACE VIEW v_regimes AS
WITH steps AS (
    SELECT *
    FROM v_break_type
    WHERE break_type IN ('STEP', 'GRADUAL_SHIFT')
),
with_next AS (
    SELECT
        signal_id,
        I AS regime_start,
        LEAD(I) OVER (PARTITION BY signal_id ORDER BY I) AS regime_end,
        post_level AS regime_level,
        magnitude AS entry_magnitude,
        ROW_NUMBER() OVER (PARTITION BY signal_id ORDER BY I) AS regime_number
    FROM steps
)
SELECT
    signal_id,
    regime_number,
    regime_start,
    regime_end,
    regime_end - regime_start AS regime_duration,
    regime_level,
    entry_magnitude
FROM with_next
WHERE regime_end IS NOT NULL
ORDER BY signal_id, regime_start;


-- ============================================================
-- BREAK SUMMARY PER SIGNAL (for typology enrichment)
-- ============================================================

CREATE OR REPLACE VIEW v_break_summary AS
SELECT
    signal_id,
    COUNT(*) AS n_breaks,
    COUNT(*) FILTER (WHERE break_type = 'IMPULSE') AS n_impulses,
    COUNT(*) FILTER (WHERE break_type = 'STEP') AS n_steps,
    COUNT(*) FILTER (WHERE break_type = 'GRADUAL_SHIFT') AS n_gradual,
    COUNT(*) FILTER (WHERE break_type = 'TRANSIENT') AS n_transient,
    AVG(ABS(magnitude)) AS mean_magnitude,
    MAX(ABS(magnitude)) AS max_magnitude,
    AVG(sharpness) AS mean_sharpness,
    MAX(sharpness) AS max_sharpness,

    -- Dominant break character
    CASE
        WHEN COUNT(*) FILTER (WHERE break_type = 'IMPULSE') >
             COUNT(*) FILTER (WHERE break_type = 'STEP')
        THEN 'IMPULSE_DOMINANT'
        WHEN COUNT(*) FILTER (WHERE break_type = 'STEP') > 0
        THEN 'STEP_DOMINANT'
        ELSE 'MIXED'
    END AS break_character

FROM v_break_type
GROUP BY signal_id;


-- ============================================================
-- BREAK PATTERNS (temporal analysis)
-- ============================================================
-- Analyzes spacing and clustering of breaks

CREATE OR REPLACE VIEW v_break_patterns AS
WITH break_gaps AS (
    SELECT
        signal_id,
        I,
        break_type,
        I - LAG(I) OVER (PARTITION BY signal_id ORDER BY I) AS gap_to_prev,
        LEAD(I) OVER (PARTITION BY signal_id ORDER BY I) - I AS gap_to_next
    FROM v_break_type
)
SELECT
    signal_id,
    AVG(gap_to_prev) AS mean_break_spacing,
    STDDEV(gap_to_prev) AS std_break_spacing,
    MIN(gap_to_prev) AS min_break_spacing,
    MAX(gap_to_prev) AS max_break_spacing,

    -- Regularity: low CV = regular spacing (periodic breaks)
    STDDEV(gap_to_prev) / NULLIF(AVG(gap_to_prev), 0) AS spacing_cv,

    -- Pattern classification
    CASE
        WHEN STDDEV(gap_to_prev) / NULLIF(AVG(gap_to_prev), 0) < 0.3
        THEN 'PERIODIC'
        WHEN MIN(gap_to_prev) < 0.1 * AVG(gap_to_prev)
        THEN 'CLUSTERED'
        ELSE 'IRREGULAR'
    END AS break_pattern

FROM break_gaps
WHERE gap_to_prev IS NOT NULL
GROUP BY signal_id;
