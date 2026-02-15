-- ============================================================================
-- CANARY SEQUENCE v2
-- Which signal's trajectory changed first per cohort
-- Uses slope change detection, not z-score threshold
-- ============================================================================

CREATE OR REPLACE VIEW v_signal_trajectory AS
WITH signal_windows AS (
    -- Compute per-signal rolling statistics using observation windows
    SELECT
        cohort,
        signal_id,
        I,
        value,
        -- Rolling mean over last 3 observations
        AVG(value) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY I
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_mean,
        -- Rolling mean over the 3 observations before that
        AVG(value) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY I
            ROWS BETWEEN 7 PRECEDING AND 4 PRECEDING
        ) AS prior_rolling_mean
    FROM observations
),
slopes AS (
    SELECT
        cohort,
        signal_id,
        I,
        rolling_mean,
        prior_rolling_mean,
        -- Slope: direction of recent trajectory
        rolling_mean - prior_rolling_mean AS trajectory_delta,
        -- Baseline slope: average delta in first 20% of life
        AVG(rolling_mean - prior_rolling_mean) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY I
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_avg_delta
    FROM signal_windows
    WHERE prior_rolling_mean IS NOT NULL
),
baseline_slopes AS (
    -- Establish baseline trajectory from first 20% of each signal's life
    SELECT
        cohort,
        signal_id,
        AVG(trajectory_delta) AS baseline_slope,
        STDDEV(trajectory_delta) AS baseline_slope_std
    FROM slopes s
    JOIN (
        SELECT cohort, MIN(I) AS min_I, MAX(I) AS max_I FROM observations GROUP BY cohort
    ) life ON s.cohort = life.cohort
    WHERE s.I < life.min_I + (life.max_I - life.min_I) * 0.2
    GROUP BY cohort, signal_id
),
trajectory_departures AS (
    SELECT
        s.cohort,
        s.signal_id,
        s.I,
        s.trajectory_delta,
        b.baseline_slope,
        b.baseline_slope_std,
        -- How much has the slope changed from baseline?
        -- Normalized by baseline variability to handle different signal scales
        CASE
            WHEN b.baseline_slope_std > 0
            THEN ABS(s.trajectory_delta - b.baseline_slope) / b.baseline_slope_std
            ELSE ABS(s.trajectory_delta - b.baseline_slope)
        END AS slope_departure
    FROM slopes s
    JOIN baseline_slopes b ON s.cohort = b.cohort AND s.signal_id = b.signal_id
    -- Skip the baseline period itself
    JOIN (
        SELECT cohort, MIN(I) AS min_I, MAX(I) AS max_I FROM observations GROUP BY cohort
    ) life ON s.cohort = life.cohort
    WHERE s.I > life.min_I + (life.max_I - life.min_I) * 0.2
)
SELECT * FROM trajectory_departures;

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH first_departure AS (
    SELECT
        cohort,
        signal_id,
        MIN(I) AS first_departure_I
    FROM v_signal_trajectory
    -- Slope departure > 3x baseline variability = trajectory changed
    WHERE slope_departure > 3.0
    GROUP BY cohort, signal_id
)
SELECT
    cohort,
    signal_id,
    first_departure_I,
    RANK() OVER (PARTITION BY cohort ORDER BY first_departure_I ASC) AS canary_rank,
    COUNT(*) OVER (PARTITION BY signal_id) AS times_canary_across_fleet
FROM first_departure
WHERE first_departure_I IS NOT NULL
ORDER BY cohort, canary_rank;

-- Top canary signals across fleet
SELECT
    signal_id,
    COUNT(*) AS times_first_3_canary,
    ROUND(AVG(first_departure_I), 1) AS avg_onset_I,
    ROUND(MIN(first_departure_I), 0) AS earliest_onset,
    ROUND(MAX(first_departure_I), 0) AS latest_onset
FROM v_canary_sequence
WHERE canary_rank <= 3
GROUP BY signal_id
ORDER BY times_first_3_canary DESC;

-- Per-cohort canary sequence (first 5 trajectory departures)
SELECT
    cohort,
    signal_id,
    first_departure_I,
    canary_rank
FROM v_canary_sequence
WHERE canary_rank <= 5
ORDER BY cohort, canary_rank;
