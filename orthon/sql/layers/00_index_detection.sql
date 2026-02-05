-- ============================================================================
-- ORTHON SQL: 00_index_detection.sql
-- ============================================================================
-- Detect index column and type
--
-- TIME: Auto-detectable (ISO 8601, Unix epoch, date strings, Excel serial)
-- OTHER: Requires user input (space, frequency, scale, cycle)
--
-- "Time is the one index dimension where ORTHON can be smart."
-- ============================================================================

-- ============================================================================
-- INDEX COLUMN DETECTION
-- ============================================================================

CREATE OR REPLACE VIEW v_index_detection AS
WITH column_stats AS (
    SELECT
        column_name,
        data_type,
        -- Sample first 5 values
        (SELECT LIST(value ORDER BY rownum LIMIT 5)
         FROM (SELECT value, ROW_NUMBER() OVER () as rownum
               FROM observations UNPIVOT (value FOR col IN (column_name)))) AS sample_values
    FROM information_schema.columns
    WHERE table_name = 'observations'
),

-- Analyze each column for index potential
analysis AS (
    SELECT
        column_name,
        data_type,
        sample_values,

        -- Check for time-related column names
        LOWER(column_name) IN (
            'timestamp', 'time', 'datetime', 'date', 't', 'ts',
            'time_stamp', 'sample_time', 'record_time', 'created_at',
            'measured_at', 'observation_time', 'recorded_at', 'dt'
        ) AS is_time_named,

        -- Check for cycle/sequence names
        LOWER(column_name) IN (
            'cycle', 'cycles', 'sample', 'index', 'i', 'n',
            'step', 'iteration', 'sequence', 'row', 'sample_id'
        ) AS is_cycle_named,

        -- Check for spatial names
        LOWER(column_name) IN (
            'x', 'y', 'z', 'position', 'distance', 'depth', 'height',
            'length', 'radius', 'r', 'location', 'station', 'chainage'
        ) AS is_spatial_named,

        -- Check for frequency names
        LOWER(column_name) IN (
            'frequency', 'freq', 'f', 'hz', 'khz', 'bin', 'band'
        ) AS is_frequency_named

    FROM column_stats
)

SELECT
    column_name,
    data_type,
    sample_values,

    -- Determine detected type
    CASE
        -- Explicit timestamp column name
        WHEN is_time_named AND data_type IN ('TIMESTAMP', 'DATE', 'DATETIME', 'VARCHAR')
        THEN 'timestamp'

        -- Spatial column
        WHEN is_spatial_named
        THEN 'spatial'

        -- Frequency column
        WHEN is_frequency_named
        THEN 'frequency'

        -- Cycle/sequence
        WHEN is_cycle_named
        THEN 'cycle'

        -- Needs further analysis
        ELSE 'unknown'
    END AS detected_type,

    -- Determine dimension
    CASE
        WHEN is_time_named THEN 'time'
        WHEN is_spatial_named THEN 'space'
        WHEN is_frequency_named THEN 'frequency'
        WHEN is_cycle_named THEN 'time'  -- Cycles are time-like
        ELSE 'unknown'
    END AS dimension,

    -- Confidence level
    CASE
        WHEN is_time_named AND data_type IN ('TIMESTAMP', 'DATE') THEN 'high'
        WHEN is_time_named THEN 'medium'
        WHEN is_cycle_named THEN 'high'
        WHEN is_spatial_named OR is_frequency_named THEN 'medium'
        ELSE 'low'
    END AS confidence,

    -- Needs user input?
    CASE
        -- Time columns: auto-detect
        WHEN is_time_named AND data_type IN ('TIMESTAMP', 'DATE', 'DATETIME') THEN FALSE
        -- Other: need confirmation
        ELSE TRUE
    END AS needs_user_input

FROM analysis;


-- ============================================================================
-- SAMPLING INTERVAL DETECTION (for time indices)
-- ============================================================================

CREATE OR REPLACE VIEW v_sampling_interval AS
WITH ordered_data AS (
    SELECT
        I AS index_value,
        ROW_NUMBER() OVER (ORDER BY I) AS row_num
    FROM observations
    LIMIT 10000  -- Limit for performance
),

diffs AS (
    SELECT
        index_value - LAG(index_value) OVER (ORDER BY row_num) AS dt
    FROM ordered_data
),

stats AS (
    SELECT
        MODE(dt) AS modal_dt,          -- Most common interval
        AVG(dt) AS mean_dt,
        STDDEV(dt) AS std_dt,
        MIN(dt) AS min_dt,
        MAX(dt) AS max_dt,
        COUNT(*) AS n_intervals
    FROM diffs
    WHERE dt IS NOT NULL AND dt > 0
)

SELECT
    modal_dt AS interval_value,

    -- Classify the interval
    CASE
        -- Milliseconds (< 1 second)
        WHEN modal_dt < 1 THEN 'milliseconds'
        -- Seconds (< 1 minute)
        WHEN modal_dt < 60 THEN 'seconds'
        -- Minutes (< 1 hour)
        WHEN modal_dt < 3600 THEN 'minutes'
        -- Hours (< 1 day)
        WHEN modal_dt < 86400 THEN 'hours'
        -- Days (< 1 week)
        WHEN modal_dt < 604800 THEN 'days'
        -- Weeks (< 1 month)
        WHEN modal_dt < 2592000 THEN 'weeks'
        -- Months
        ELSE 'months'
    END AS interval_unit,

    -- Value in detected unit
    CASE
        WHEN modal_dt < 1 THEN ROUND(modal_dt * 1000, 2)
        WHEN modal_dt < 60 THEN ROUND(modal_dt, 2)
        WHEN modal_dt < 3600 THEN ROUND(modal_dt / 60, 2)
        WHEN modal_dt < 86400 THEN ROUND(modal_dt / 3600, 2)
        WHEN modal_dt < 604800 THEN ROUND(modal_dt / 86400, 2)
        WHEN modal_dt < 2592000 THEN ROUND(modal_dt / 604800, 2)
        ELSE ROUND(modal_dt / 2592000, 2)
    END AS interval_in_unit,

    -- Is it regular?
    CASE
        WHEN std_dt / NULLIF(mean_dt, 0) < 0.01 THEN 'regular'
        WHEN std_dt / NULLIF(mean_dt, 0) < 0.1 THEN 'mostly_regular'
        ELSE 'irregular'
    END AS regularity,

    -- Any gaps? (max > 2x modal)
    CASE
        WHEN max_dt > modal_dt * 2 THEN TRUE
        ELSE FALSE
    END AS has_gaps,

    -- Gap ratio
    ROUND(max_dt / NULLIF(modal_dt, 0), 1) AS max_gap_ratio,

    n_intervals

FROM stats;


-- ============================================================================
-- TIMESTAMP FORMAT DETECTION
-- ============================================================================

CREATE OR REPLACE VIEW v_timestamp_format AS
WITH sample_values AS (
    SELECT I AS value
    FROM observations
    LIMIT 10
)
SELECT
    value,

    -- Try different formats
    CASE
        -- ISO 8601 with Z
        WHEN value LIKE '____-__-__T__:__:__Z' THEN 'ISO 8601 (Z)'
        WHEN value LIKE '____-__-__T__:__:__.___Z' THEN 'ISO 8601 with ms (Z)'

        -- ISO 8601 without Z
        WHEN value LIKE '____-__-__T__:__:__' THEN 'ISO 8601'
        WHEN value LIKE '____-__-__T__:__:__.___' THEN 'ISO 8601 with ms'

        -- Standard datetime
        WHEN value LIKE '____-__-__ __:__:__' THEN 'YYYY-MM-DD HH:MM:SS'
        WHEN value LIKE '____-__-__ __:__:__.___' THEN 'YYYY-MM-DD HH:MM:SS.fff'

        -- Date only
        WHEN value LIKE '____-__-__' THEN 'YYYY-MM-DD'
        WHEN value LIKE '__/__/____' THEN 'MM/DD/YYYY or DD/MM/YYYY'

        -- Unix timestamp
        WHEN TRY_CAST(value AS BIGINT) BETWEEN 1000000000 AND 4100000000 THEN 'Unix seconds'
        WHEN TRY_CAST(value AS BIGINT) BETWEEN 1000000000000 AND 4100000000000 THEN 'Unix milliseconds'

        -- Excel serial date
        WHEN TRY_CAST(value AS DOUBLE) BETWEEN 1 AND 72000 THEN 'Excel serial (possible)'

        ELSE 'Unknown format'
    END AS detected_format,

    -- Try parsing
    TRY_CAST(value AS TIMESTAMP) AS parsed_timestamp

FROM sample_values;


-- ============================================================================
-- COMBINED INDEX DETECTION RESULT
-- ============================================================================

CREATE OR REPLACE VIEW v_index_result AS
WITH best_candidate AS (
    SELECT *
    FROM v_index_detection
    WHERE detected_type != 'unknown'
    ORDER BY
        CASE confidence
            WHEN 'high' THEN 1
            WHEN 'medium' THEN 2
            ELSE 3
        END
    LIMIT 1
)
SELECT
    c.column_name,
    c.detected_type AS index_type,
    c.dimension,
    c.confidence,
    c.needs_user_input,
    s.interval_value AS sampling_interval_seconds,
    s.interval_unit AS sampling_unit,
    s.interval_in_unit AS sampling_value,
    s.regularity,
    s.has_gaps,

    -- Message for user (if needed)
    CASE
        WHEN c.needs_user_input AND c.detected_type = 'cycle'
        THEN 'Column "' || c.column_name || '" is a cycle counter. What is the duration per cycle?'

        WHEN c.needs_user_input AND c.detected_type = 'spatial'
        THEN 'Column "' || c.column_name || '" is spatial. What is the unit? (m, ft, km)'

        WHEN c.needs_user_input AND c.detected_type = 'frequency'
        THEN 'Column "' || c.column_name || '" is frequency bins. What is the unit? (Hz, kHz)'

        WHEN c.needs_user_input
        THEN 'Column "' || c.column_name || '" type is unclear. What does it represent?'

        ELSE NULL
    END AS user_prompt

FROM best_candidate c
LEFT JOIN v_sampling_interval s ON TRUE;
