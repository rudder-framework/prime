-- ============================================================================
-- 13_canary_sequence.sql
-- ============================================================================
-- Canary sequence analysis report
-- Depends on: v_signal_trajectory, v_canary_sequence (layer 13)
-- ============================================================================

.print ''
.print '============================================================================'
.print '                     CANARY SEQUENCE ANALYSIS                              '
.print '============================================================================'

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
