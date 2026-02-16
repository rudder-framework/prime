-- ============================================================
-- BREAK CLASSIFICATION — Ranked Views
-- ============================================================
-- Interprets Engines break detection output.
-- Ranks breaks by magnitude, sharpness, and spacing regularity.
-- No categorical gates — rank everything.
--
-- Input: breaks.parquet from Engines
-- ============================================================

-- ============================================================
-- BREAK RANKED
-- ============================================================
-- Every break gets ranked by magnitude, sharpness, level change.
-- The analyst queries WHERE magnitude_rank = 1 to see largest.
-- ============================================================

CREATE OR REPLACE VIEW v_break_type AS
SELECT
    signal_id,
    cohort,
    signal_0_center,
    magnitude,
    direction,
    sharpness,
    duration,
    pre_level,
    post_level,
    snr,

    -- Level change: did the signal stay at the new level?
    ABS(post_level - pre_level) AS level_change,

    -- Persistence ratio: how much of the magnitude was sustained
    ABS(post_level - pre_level) / NULLIF(ABS(magnitude), 0) AS persistence_ratio,

    -- Rank by magnitude within signal
    RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY ABS(magnitude) DESC
    ) AS magnitude_rank,

    -- Rank by sharpness within signal
    RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY sharpness DESC NULLS LAST
    ) AS sharpness_rank,

    -- Rank by magnitude fleet-wide
    RANK() OVER (
        ORDER BY ABS(magnitude) DESC
    ) AS fleet_magnitude_rank,

    -- Percentile within signal's break history
    PERCENT_RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY ABS(magnitude)
    ) AS magnitude_percentile,

    -- Percentile by sharpness
    PERCENT_RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY sharpness NULLS FIRST
    ) AS sharpness_percentile

FROM breaks
ORDER BY signal_id, signal_0_center;


-- ============================================================
-- REGIME DETECTION (from break clustering)
-- ============================================================
-- A regime is a period between consecutive significant breaks.
-- Rank regimes by duration and entry magnitude.
-- ============================================================

CREATE OR REPLACE VIEW v_regimes AS
WITH significant_breaks AS (
    SELECT *
    FROM v_break_type
    WHERE persistence_ratio > 0.3  -- Sustained level change
),
with_next AS (
    SELECT
        signal_id,
        cohort,
        signal_0_center AS regime_start,
        LEAD(signal_0_center) OVER (PARTITION BY signal_id, cohort ORDER BY signal_0_center) AS regime_end,
        post_level AS regime_level,
        magnitude AS entry_magnitude,
        ROW_NUMBER() OVER (PARTITION BY signal_id, cohort ORDER BY signal_0_center) AS regime_number
    FROM significant_breaks
)
SELECT
    signal_id,
    cohort,
    regime_number,
    regime_start,
    regime_end,
    regime_end - regime_start AS regime_duration,
    regime_level,
    entry_magnitude,

    -- Rank regimes by duration (longest first)
    RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY (regime_end - regime_start) DESC NULLS LAST
    ) AS duration_rank,

    -- Rank by entry magnitude
    RANK() OVER (
        PARTITION BY signal_id, cohort
        ORDER BY ABS(entry_magnitude) DESC
    ) AS entry_magnitude_rank

FROM with_next
WHERE regime_end IS NOT NULL
ORDER BY signal_id, regime_start;


-- ============================================================
-- BREAK SUMMARY PER SIGNAL (ranked)
-- ============================================================

CREATE OR REPLACE VIEW v_break_summary AS
SELECT
    signal_id,
    cohort,
    COUNT(*) AS n_breaks,
    AVG(ABS(magnitude)) AS mean_magnitude,
    MAX(ABS(magnitude)) AS max_magnitude,
    AVG(sharpness) AS mean_sharpness,
    MAX(sharpness) AS max_sharpness,
    AVG(persistence_ratio) AS mean_persistence,

    -- Rank signals by break activity
    RANK() OVER (ORDER BY COUNT(*) DESC) AS break_frequency_rank,
    RANK() OVER (ORDER BY MAX(ABS(magnitude)) DESC) AS max_magnitude_rank,
    RANK() OVER (ORDER BY AVG(ABS(magnitude)) DESC) AS mean_magnitude_rank

FROM v_break_type
GROUP BY signal_id, cohort;


-- ============================================================
-- BREAK PATTERNS RANKED (temporal analysis)
-- ============================================================
-- Ranks by spacing regularity and frequency

CREATE OR REPLACE VIEW v_break_patterns_ranked AS
WITH break_gaps AS (
    SELECT
        signal_id,
        cohort,
        signal_0_center,
        signal_0_center - LAG(signal_0_center) OVER (PARTITION BY signal_id, cohort ORDER BY signal_0_center) AS gap_to_prev
    FROM v_break_type
)
SELECT
    signal_id,
    cohort,
    AVG(gap_to_prev) AS mean_break_spacing,
    STDDEV(gap_to_prev) / NULLIF(AVG(gap_to_prev), 0) AS spacing_cv,
    COUNT(*) AS n_breaks,

    -- Rank by regularity (lowest CV = most periodic)
    RANK() OVER (
        ORDER BY STDDEV(gap_to_prev) / NULLIF(AVG(gap_to_prev), 0) ASC NULLS LAST
    ) AS regularity_rank,

    -- Rank by break frequency (most breaks = most active)
    RANK() OVER (
        ORDER BY COUNT(*) DESC
    ) AS frequency_rank,

    -- Percentile of regularity
    PERCENT_RANK() OVER (
        ORDER BY STDDEV(gap_to_prev) / NULLIF(AVG(gap_to_prev), 0) NULLS LAST
    ) AS regularity_percentile

FROM break_gaps
WHERE gap_to_prev IS NOT NULL
GROUP BY signal_id, cohort;
