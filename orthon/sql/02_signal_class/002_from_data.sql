-- ============================================================================
-- ORTHON SQL Engine: 02_signal_class/002_from_data.sql
-- ============================================================================
-- Infer signal properties from data when units don't tell us
-- ============================================================================

-- Data value properties
CREATE OR REPLACE VIEW v_data_props AS
SELECT
    signal_id,
    -- Unique ratio (low = discrete values)
    COUNT(DISTINCT ROUND(y, 4))::FLOAT / COUNT(*) AS unique_ratio,
    -- All integers check
    BOOL_AND(y = ROUND(y, 0)) AS is_integer,
    -- Sparsity (fraction of zeros)
    COUNT(*) FILTER (WHERE ABS(y) < 1e-10)::FLOAT / COUNT(*) AS sparsity
FROM v_base
GROUP BY signal_id;

-- Periodicity detection via sign changes in d2y
-- Periodic signals have regular, predictable zero-crossings
CREATE OR REPLACE VIEW v_periodicity AS
WITH sign_changes AS (
    SELECT
        signal_id,
        CASE
            WHEN SIGN(d2y) != SIGN(LAG(d2y) OVER (PARTITION BY signal_id ORDER BY I))
             AND d2y IS NOT NULL
            THEN 1
            ELSE 0
        END AS sign_change
    FROM v_d2y
)
SELECT
    signal_id,
    SUM(sign_change) AS n_sign_changes,
    COUNT(*) AS n_points,
    SUM(sign_change)::FLOAT / COUNT(*) * 1000 AS changes_per_1000,
    -- Estimate period from sign changes (2 changes per period)
    CASE
        WHEN SUM(sign_change) > 0
        THEN COUNT(*)::FLOAT / (SUM(sign_change)::FLOAT / 2)
        ELSE NULL
    END AS est_period
FROM sign_changes
GROUP BY signal_id;
