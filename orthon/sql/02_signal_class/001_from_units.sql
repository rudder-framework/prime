-- ============================================================================
-- ORTHON SQL Engine: 02_signal_class/001_from_units.sql
-- ============================================================================
-- Classify signals by value_unit
-- Units determine if calculus applies (analog vs digital vs event)
-- ============================================================================

CREATE OR REPLACE VIEW v_class_from_units AS
SELECT
    signal_id,
    value_unit,
    CASE
        -- Pressure (analog)
        WHEN value_unit IN ('PSI', 'psi', 'kPa', 'bar', 'Pa', 'atm') THEN 'analog'
        -- Temperature (analog)
        WHEN value_unit IN ('°C', 'C', 'degC', '°F', 'F', 'degF', 'K') THEN 'analog'
        -- Velocity (analog)
        WHEN value_unit IN ('m/s', 'ft/s', 'km/h', 'mph') THEN 'analog'
        -- Voltage/Current (analog)
        WHEN value_unit IN ('V', 'mV', 'A', 'mA', 'W', 'kW') THEN 'analog'
        -- Flow (analog)
        WHEN value_unit IN ('gpm', 'lpm', 'cfm', 'm3/s', 'kg/s') THEN 'analog'
        -- Digital states
        WHEN value_unit IN ('state', 'mode', 'status', 'level') THEN 'digital'
        -- Event counts
        WHEN value_unit IN ('count', 'events', 'occurrences') THEN 'event'
        -- Unknown - will check data properties
        ELSE 'check_data'
    END AS base_class
FROM v_signal_meta;
