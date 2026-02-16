-- ============================================================================
-- Engines CORRELATION CHANGE REPORTS
-- ============================================================================
--
-- Detects when relationships between signals change over time.
-- A change in correlation suggests process dynamics are shifting.
--
-- Usage: Run against observations table with columns [cohort, signal_id, signal_0, value]
-- ============================================================================


-- ============================================================================
-- REPORT 1: CORRELATION BASELINE VS LATE
-- Compares pairwise correlations between early and late periods
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        MIN(signal_0) AS i_min,
        MAX(signal_0) AS i_max,
        MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end,
        MIN(signal_0) + 0.8 * (MAX(signal_0) - MIN(signal_0)) AS late_start
    FROM observations
    GROUP BY cohort
),

-- Pivot to wide format for baseline period
baseline_wide AS (
    SELECT
        o.cohort,
        o.signal_0,
        o.signal_id,
        o.value
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 <= t.baseline_end
),

-- Pivot to wide format for late period
late_wide AS (
    SELECT
        o.cohort,
        o.signal_0,
        o.signal_id,
        o.value
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 >= t.late_start
),

-- Baseline correlations
baseline_corr AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM baseline_wide a
    JOIN baseline_wide b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.signal_id, b.signal_id
),

-- Late period correlations
late_corr AS (
    SELECT
        a.cohort,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM late_wide a
    JOIN late_wide b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.signal_id, b.signal_id
)

SELECT
    b.cohort,
    b.signal_a,
    b.signal_b,
    ROUND(b.correlation, 3) AS baseline_corr,
    ROUND(l.correlation, 3) AS late_corr,
    ROUND(l.correlation - b.correlation, 3) AS corr_change,
    CASE
        WHEN ABS(l.correlation - b.correlation) > 0.3 THEN 'MAJOR_CHANGE'
        WHEN ABS(l.correlation - b.correlation) > 0.15 THEN 'MODERATE_CHANGE'
        ELSE 'STABLE'
    END AS status,
    CASE
        WHEN b.correlation > 0.5 AND l.correlation < 0.3 THEN 'DECOUPLED'
        WHEN b.correlation < 0.3 AND l.correlation > 0.5 THEN 'COUPLED'
        WHEN b.correlation > 0 AND l.correlation < 0 THEN 'SIGN_FLIP'
        WHEN b.correlation < 0 AND l.correlation > 0 THEN 'SIGN_FLIP'
        ELSE 'SHIFT'
    END AS change_type
FROM baseline_corr b
JOIN late_corr l
    ON b.cohort = l.cohort
    AND b.signal_a = l.signal_a
    AND b.signal_b = l.signal_b
WHERE ABS(l.correlation - b.correlation) > 0.1
ORDER BY ABS(l.correlation - b.correlation) DESC;


-- ============================================================================
-- REPORT 2: ROLLING CORRELATION (detect when correlation changed)
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        signal_0,
        value
    FROM observations
),

window_corr AS (
    SELECT
        a.cohort,
        a.window_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM windowed a
    JOIN windowed b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.window_id = b.window_id
        AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.window_id, a.signal_id, b.signal_id
)

SELECT
    cohort,
    signal_a,
    signal_b,
    MAX(CASE WHEN window_id = 1 THEN ROUND(correlation, 3) END) AS w1_corr,
    MAX(CASE WHEN window_id = 2 THEN ROUND(correlation, 3) END) AS w2_corr,
    MAX(CASE WHEN window_id = 3 THEN ROUND(correlation, 3) END) AS w3_corr,
    MAX(CASE WHEN window_id = 4 THEN ROUND(correlation, 3) END) AS w4_corr,
    MAX(CASE WHEN window_id = 5 THEN ROUND(correlation, 3) END) AS w5_corr,
    ROUND(MAX(correlation) - MIN(correlation), 3) AS corr_range,
    CASE
        WHEN MAX(correlation) - MIN(correlation) > 0.3 THEN 'UNSTABLE'
        ELSE 'STABLE'
    END AS stability
FROM window_corr
GROUP BY cohort, signal_a, signal_b
HAVING MAX(correlation) - MIN(correlation) > 0.15
ORDER BY MAX(correlation) - MIN(correlation) DESC;


-- ============================================================================
-- REPORT 3: DECOUPLING EVENTS
-- Identifies signal pairs that were correlated but became uncorrelated
-- ============================================================================

WITH
time_bounds AS (
    SELECT
        cohort,
        MIN(signal_0) + 0.2 * (MAX(signal_0) - MIN(signal_0)) AS baseline_end,
        MIN(signal_0) + 0.8 * (MAX(signal_0) - MIN(signal_0)) AS late_start
    FROM observations
    GROUP BY cohort
),

baseline_wide AS (
    SELECT o.cohort, o.signal_0, o.signal_id, o.value
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 <= t.baseline_end
),

late_wide AS (
    SELECT o.cohort, o.signal_0, o.signal_id, o.value
    FROM observations o
    JOIN time_bounds t ON o.cohort = t.cohort
    WHERE o.signal_0 >= t.late_start
),

baseline_corr AS (
    SELECT
        a.cohort, a.signal_id AS signal_a, b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM baseline_wide a
    JOIN baseline_wide b ON a.cohort = b.cohort AND a.signal_0 = b.signal_0 AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.signal_id, b.signal_id
),

late_corr AS (
    SELECT
        a.cohort, a.signal_id AS signal_a, b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM late_wide a
    JOIN late_wide b ON a.cohort = b.cohort AND a.signal_0 = b.signal_0 AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.signal_id, b.signal_id
)

SELECT
    b.cohort,
    b.signal_a,
    b.signal_b,
    ROUND(b.correlation, 3) AS baseline_corr,
    ROUND(l.correlation, 3) AS late_corr,
    'DECOUPLED' AS event_type,
    'Signals were strongly correlated, now uncorrelated - check for broken linkage' AS interpretation
FROM baseline_corr b
JOIN late_corr l USING (cohort, signal_a, signal_b)
WHERE ABS(b.correlation) > 0.6 AND ABS(l.correlation) < 0.3

UNION ALL

SELECT
    b.cohort,
    b.signal_a,
    b.signal_b,
    ROUND(b.correlation, 3) AS baseline_corr,
    ROUND(l.correlation, 3) AS late_corr,
    'NEWLY_COUPLED' AS event_type,
    'Signals were uncorrelated, now strongly correlated - new interaction detected' AS interpretation
FROM baseline_corr b
JOIN late_corr l USING (cohort, signal_a, signal_b)
WHERE ABS(b.correlation) < 0.3 AND ABS(l.correlation) > 0.6

ORDER BY ABS(baseline_corr - late_corr) DESC;


-- ============================================================================
-- REPORT 4: CORRELATION NETWORK DENSITY
-- Tracks how interconnected signals are over time
-- ============================================================================

WITH
windowed AS (
    SELECT
        cohort,
        signal_id,
        NTILE(5) OVER (PARTITION BY cohort, signal_id ORDER BY signal_0) AS window_id,
        signal_0,
        value
    FROM observations
),

window_corr AS (
    SELECT
        a.cohort,
        a.window_id,
        a.signal_id AS signal_a,
        b.signal_id AS signal_b,
        CORR(a.value, b.value) AS correlation
    FROM windowed a
    JOIN windowed b
        ON a.cohort = b.cohort
        AND a.signal_0 = b.signal_0
        AND a.window_id = b.window_id
        AND a.signal_id < b.signal_id
    GROUP BY a.cohort, a.window_id, a.signal_id, b.signal_id
)

SELECT
    cohort,
    window_id,
    COUNT(*) AS n_pairs,
    SUM(CASE WHEN ABS(correlation) > 0.5 THEN 1 ELSE 0 END) AS n_strong_corr,
    ROUND(100.0 * SUM(CASE WHEN ABS(correlation) > 0.5 THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_strong,
    ROUND(AVG(ABS(correlation)), 3) AS avg_abs_corr,
    CASE
        WHEN AVG(ABS(correlation)) > 0.4 THEN 'HIGHLY_COUPLED'
        WHEN AVG(ABS(correlation)) > 0.25 THEN 'MODERATELY_COUPLED'
        ELSE 'LOOSELY_COUPLED'
    END AS coupling_level
FROM window_corr
GROUP BY cohort, window_id
ORDER BY cohort, window_id;
