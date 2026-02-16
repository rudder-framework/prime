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
        signal_0,
        value,
        -- Rolling mean over last 3 observations
        AVG(value) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY signal_0
            ROWS BETWEEN 3 PRECEDING AND CURRENT ROW
        ) AS rolling_mean,
        -- Rolling mean over the 3 observations before that
        AVG(value) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY signal_0
            ROWS BETWEEN 7 PRECEDING AND 4 PRECEDING
        ) AS prior_rolling_mean
    FROM observations
),
slopes AS (
    SELECT
        cohort,
        signal_id,
        signal_0,
        rolling_mean,
        prior_rolling_mean,
        -- Slope: direction of recent trajectory
        rolling_mean - prior_rolling_mean AS trajectory_delta,
        -- Baseline slope: average delta in first 20% of life
        AVG(rolling_mean - prior_rolling_mean) OVER (
            PARTITION BY cohort, signal_id
            ORDER BY signal_0
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS cumulative_avg_delta
    FROM signal_windows
    WHERE prior_rolling_mean IS NOT NULL
),
baseline_slopes AS (
    -- Establish baseline trajectory from first 20% of each signal's life
    SELECT
        s.cohort,
        s.signal_id,
        AVG(s.trajectory_delta) AS baseline_slope,
        STDDEV(s.trajectory_delta) AS baseline_slope_std
    FROM slopes s
    JOIN (
        SELECT cohort, MIN(signal_0) AS min_I, MAX(signal_0) AS max_I FROM observations GROUP BY cohort
    ) life ON s.cohort = life.cohort
    WHERE s.signal_0 < life.min_I + (life.max_I - life.min_I) * 0.2
    GROUP BY s.cohort, s.signal_id
),
trajectory_departures AS (
    SELECT
        s.cohort,
        s.signal_id,
        s.signal_0,
        s.trajectory_delta,
        b.baseline_slope,
        -- Absolute departure from baseline slope
        ABS(s.trajectory_delta - b.baseline_slope) AS slope_departure_abs
    FROM slopes s
    JOIN baseline_slopes b ON s.cohort = b.cohort AND s.signal_id = b.signal_id
    -- Skip the baseline period itself
    JOIN (
        SELECT cohort, MIN(signal_0) AS min_I, MAX(signal_0) AS max_I FROM observations GROUP BY cohort
    ) life ON s.cohort = life.cohort
    WHERE s.signal_0 > life.min_I + (life.max_I - life.min_I) * 0.2
)
SELECT
    *,
    -- Percentile rank of departure within this signal's history
    PERCENT_RANK() OVER (
        PARTITION BY cohort, signal_id
        ORDER BY slope_departure_abs
    ) AS slope_departure_pctile
FROM trajectory_departures;

CREATE OR REPLACE VIEW v_canary_sequence AS
WITH first_departure AS (
    SELECT
        cohort,
        signal_id,
        MIN(signal_0) AS first_departure_I
    FROM v_signal_trajectory
    -- Slope departure above 99th percentile = trajectory changed
    WHERE slope_departure_pctile > 0.99
    GROUP BY cohort, signal_id
),
-- Get departure magnitude at the first departure window for tie-breaking
first_departure_magnitude AS (
    SELECT
        fd.cohort,
        fd.signal_id,
        fd.first_departure_I,
        t.slope_departure_abs AS departure_magnitude
    FROM first_departure fd
    JOIN v_signal_trajectory t
        ON fd.cohort = t.cohort
        AND fd.signal_id = t.signal_id
        AND fd.first_departure_I = t.signal_0
)
SELECT
    cohort,
    signal_id,
    first_departure_I,
    departure_magnitude,
    -- Break ties by magnitude: same window, but signal that moved hardest ranks lower
    ROW_NUMBER() OVER (
        PARTITION BY cohort
        ORDER BY first_departure_I ASC, departure_magnitude DESC
    ) AS canary_rank,
    COUNT(*) OVER (PARTITION BY signal_id) AS times_canary_across_fleet
FROM first_departure_magnitude
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
    ROUND(departure_magnitude, 6) AS departure_magnitude,
    canary_rank
FROM v_canary_sequence
WHERE canary_rank <= 5
ORDER BY cohort, canary_rank;

-- Lead canaries: rank 1 signals where no other signal shares their departure window
WITH lead AS (
    SELECT cohort, signal_id, first_departure_I, canary_rank
    FROM v_canary_sequence
    WHERE canary_rank = 1
),
tied AS (
    SELECT cohort, first_departure_I, COUNT(*) AS n_at_window
    FROM v_canary_sequence
    WHERE first_departure_I IN (SELECT first_departure_I FROM lead)
      AND cohort IN (SELECT cohort FROM lead)
    GROUP BY cohort, first_departure_I
)
SELECT
    l.signal_id,
    COUNT(*) AS times_sole_leader,
    ROUND(AVG(l.first_departure_I), 1) AS avg_onset_I,
    MIN(l.first_departure_I) AS earliest_onset,
    MAX(l.first_departure_I) AS latest_onset
FROM lead l
JOIN tied t ON l.cohort = t.cohort AND l.first_departure_I = t.first_departure_I
WHERE t.n_at_window = 1
GROUP BY l.signal_id
ORDER BY times_sole_leader DESC;

-- Average delay between rank 1 and rank 2 canary per engine
SELECT
    cohort,
    MIN(CASE WHEN canary_rank = 1 THEN first_departure_I END) AS first_signal_I,
    MIN(CASE WHEN canary_rank = 2 THEN first_departure_I END) AS second_signal_I,
    MIN(CASE WHEN canary_rank = 2 THEN first_departure_I END) -
        MIN(CASE WHEN canary_rank = 1 THEN first_departure_I END) AS propagation_delay
FROM v_canary_sequence
GROUP BY cohort
HAVING propagation_delay IS NOT NULL AND propagation_delay > 0
ORDER BY propagation_delay DESC;
