-- ============================================================================
-- CANARY SEQUENCE
-- Which signal deviated first per cohort
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_deviation_onset AS
SELECT
    cohort,
    signal_id,
    I,
    value,
    ABS(value - AVG(value) OVER (PARTITION BY cohort, signal_id)) /
        NULLIF(STDDEV(value) OVER (PARTITION BY cohort, signal_id), 0) AS z_from_mean
FROM observations;

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH first_extremes AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS first_extreme_I
    FROM v_signal_deviation_onset
    WHERE z_from_mean > 2.0
    GROUP BY cohort, signal_id
)
SELECT
    cohort,
    signal_id,
    first_extreme_I,

    -- Canary rank: which signal deviated first in this engine
    RANK() OVER (PARTITION BY cohort ORDER BY first_extreme_I ASC) AS canary_rank,

    -- Fleet-wide: how often is this signal the first mover
    COUNT(*) OVER (PARTITION BY signal_id) AS times_canary_across_fleet

FROM first_extremes
WHERE first_extreme_I IS NOT NULL
ORDER BY cohort, canary_rank;

-- Top canary signals across the fleet
SELECT
    signal_id,
    COUNT(*) AS times_first_3_canary,
    ROUND(AVG(first_extreme_I), 1) AS avg_onset_I,
    ROUND(MIN(first_extreme_I), 0) AS earliest_onset,
    ROUND(MAX(first_extreme_I), 0) AS latest_onset
FROM v_canary_sequence
WHERE canary_rank <= 3
GROUP BY signal_id
ORDER BY times_first_3_canary DESC;

-- Canary sequence for each cohort (first 5 signals to deviate)
SELECT
    cohort,
    signal_id,
    first_extreme_I,
    canary_rank
FROM v_canary_sequence
WHERE canary_rank <= 5
ORDER BY cohort, canary_rank;
