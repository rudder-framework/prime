-- ============================================================================
-- CANARY SEQUENCE
-- Which signal deviated first per cohort
-- Excludes burn-in period (first 10% of lifecycle or first 5 windows)
-- Uses within-cohort z-score (not global) to avoid operating condition bias
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_deviation_onset AS
WITH cohort_lifecycle AS (
    SELECT cohort, MAX(I) AS max_I, MIN(I) AS min_I
    FROM observations
    GROUP BY cohort
),
signal_stats AS (
    -- Per-signal, per-cohort mean and std (within-cohort normalization)
    SELECT
        o.cohort,
        o.signal_id,
        AVG(o.value) AS sig_mean,
        STDDEV(o.value) AS sig_std
    FROM observations o
    GROUP BY o.cohort, o.signal_id
),
burn_in AS (
    -- Burn-in threshold: skip first 10% of lifecycle or first 5 windows (whichever is larger)
    SELECT
        cohort,
        GREATEST(min_I + CAST((max_I - min_I) * 0.1 AS INTEGER), min_I + 80) AS burn_in_I
    FROM cohort_lifecycle
)
SELECT
    o.cohort,
    o.signal_id,
    o.I,
    o.value,
    s.sig_mean,
    s.sig_std,
    CASE
        WHEN s.sig_std > 0 THEN ABS(o.value - s.sig_mean) / s.sig_std
        ELSE 0
    END AS z_within_cohort
FROM observations o
JOIN signal_stats s ON o.cohort = s.cohort AND o.signal_id = s.signal_id
JOIN burn_in b ON o.cohort = b.cohort
WHERE o.I > b.burn_in_I;

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH first_extremes AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS first_extreme_I
    FROM v_signal_deviation_onset
    WHERE z_within_cohort > 2.0
    GROUP BY cohort, signal_id
)
SELECT
    cohort,
    signal_id,
    first_extreme_I,

    -- Canary rank: which signal deviated first in this engine (post burn-in)
    RANK() OVER (PARTITION BY cohort ORDER BY first_extreme_I ASC) AS canary_rank,

    -- Fleet-wide: how often is this signal the first mover
    COUNT(*) OVER (PARTITION BY signal_id) AS times_canary_across_fleet

FROM first_extremes
WHERE first_extreme_I IS NOT NULL
ORDER BY cohort, canary_rank;

-- Top canary signals across the fleet (post burn-in)
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

-- Canary sequence for each cohort (first 5 signals to deviate, post burn-in)
SELECT
    cohort,
    signal_id,
    first_extreme_I,
    canary_rank
FROM v_canary_sequence
WHERE canary_rank <= 5
ORDER BY cohort, canary_rank;
