-- ============================================================================
-- ORTHON SQL: 01_classification_units.sql
-- ============================================================================
-- Classify signals by UNIT only - NO COMPUTE
-- Uses the comprehensive unit mapping from constants_units.sql
--
-- ORTHON classifies by units to generate PRISM work orders.
-- PRISM will do data-based classification (sparsity, periodicity, etc.)
-- ============================================================================

-- Unit-to-signal-class mapping (embedded from constants_units.sql)
CREATE OR REPLACE TABLE unit_signal_class (
    unit VARCHAR PRIMARY KEY,
    signal_class VARCHAR,  -- analog, digital, periodic, event
    quantity VARCHAR,      -- pressure, temperature, flow, etc.
    notes VARCHAR
);

INSERT INTO unit_signal_class VALUES
-- Pressure (analog)
('PSI', 'analog', 'pressure', 'pounds per square inch'),
('psi', 'analog', 'pressure', 'pounds per square inch'),
('kPa', 'analog', 'pressure', 'kilopascal'),
('Pa', 'analog', 'pressure', 'pascal'),
('bar', 'analog', 'pressure', 'bar'),
('atm', 'analog', 'pressure', 'atmosphere'),
('mmHg', 'analog', 'pressure', 'millimeters of mercury'),
('inHg', 'analog', 'pressure', 'inches of mercury'),
('torr', 'analog', 'pressure', 'torr'),
('inH2O', 'analog', 'pressure', 'inches of water'),

-- Temperature (analog)
('°C', 'analog', 'temperature', 'degrees Celsius'),
('C', 'analog', 'temperature', 'degrees Celsius'),
('degC', 'analog', 'temperature', 'degrees Celsius'),
('°F', 'analog', 'temperature', 'degrees Fahrenheit'),
('F', 'analog', 'temperature', 'degrees Fahrenheit'),
('degF', 'analog', 'temperature', 'degrees Fahrenheit'),
('K', 'analog', 'temperature', 'kelvin'),

-- Flow (analog)
('gpm', 'analog', 'flow', 'gallons per minute'),
('lpm', 'analog', 'flow', 'liters per minute'),
('cfm', 'analog', 'flow', 'cubic feet per minute'),
('m3/s', 'analog', 'flow', 'cubic meters per second'),
('kg/s', 'analog', 'mass_flow', 'kilograms per second'),
('lb/h', 'analog', 'mass_flow', 'pounds per hour'),
('SCFM', 'analog', 'flow', 'standard cubic feet per minute'),

-- Velocity (analog)
('m/s', 'analog', 'velocity', 'meters per second'),
('ft/s', 'analog', 'velocity', 'feet per second'),
('km/h', 'analog', 'velocity', 'kilometers per hour'),
('mph', 'analog', 'velocity', 'miles per hour'),
('RPM', 'analog', 'angular_velocity', 'revolutions per minute'),
('rpm', 'analog', 'angular_velocity', 'revolutions per minute'),
('rad/s', 'analog', 'angular_velocity', 'radians per second'),

-- Electrical (analog)
('V', 'analog', 'voltage', 'volts'),
('mV', 'analog', 'voltage', 'millivolts'),
('kV', 'analog', 'voltage', 'kilovolts'),
('A', 'analog', 'current', 'amperes'),
('mA', 'analog', 'current', 'milliamperes'),
('W', 'analog', 'power', 'watts'),
('kW', 'analog', 'power', 'kilowatts'),
('MW', 'analog', 'power', 'megawatts'),
('VA', 'analog', 'apparent_power', 'volt-amperes'),
('kVA', 'analog', 'apparent_power', 'kilovolt-amperes'),
('Ohm', 'analog', 'resistance', 'ohms'),
('Hz', 'periodic', 'frequency', 'hertz'),

-- Mass/Weight (analog)
('kg', 'analog', 'mass', 'kilograms'),
('g', 'analog', 'mass', 'grams'),
('lb', 'analog', 'mass', 'pounds'),
('ton', 'analog', 'mass', 'metric tons'),

-- Level/Distance (analog)
('m', 'analog', 'length', 'meters'),
('cm', 'analog', 'length', 'centimeters'),
('mm', 'analog', 'length', 'millimeters'),
('ft', 'analog', 'length', 'feet'),
('in', 'analog', 'length', 'inches'),
('%', 'analog', 'percentage', 'percent'),
('pct', 'analog', 'percentage', 'percent'),

-- Concentration (analog)
('ppm', 'analog', 'concentration', 'parts per million'),
('ppb', 'analog', 'concentration', 'parts per billion'),
('mol/L', 'analog', 'concentration', 'moles per liter'),
('mg/L', 'analog', 'concentration', 'milligrams per liter'),

-- Digital states
('state', 'digital', 'state', 'discrete state'),
('mode', 'digital', 'mode', 'operating mode'),
('status', 'digital', 'status', 'status indicator'),
('level', 'digital', 'level', 'discrete level'),
('bool', 'digital', 'boolean', 'true/false'),
('boolean', 'digital', 'boolean', 'true/false'),

-- Event/Count
('count', 'event', 'count', 'event count'),
('events', 'event', 'count', 'event count'),
('occurrences', 'event', 'count', 'occurrences'),
('alarm', 'event', 'alarm', 'alarm event'),
('trip', 'event', 'trip', 'trip event'),

-- Dimensionless (analog by default)
('ratio', 'analog', 'ratio', 'dimensionless ratio'),
('factor', 'analog', 'factor', 'dimensionless factor'),
('unitless', 'analog', 'dimensionless', 'no unit'),
('1', 'analog', 'dimensionless', 'unity'),
('unknown', 'analog', 'unknown', 'unknown unit - assume analog');

-- Classify signals from observations
CREATE OR REPLACE VIEW v_signal_class_unit AS
SELECT
    o.signal_id,
    o.unit,
    COALESCE(u.signal_class, 'analog') AS signal_class,
    COALESCE(u.quantity, 'unknown') AS quantity,
    -- Is calculus valid for this signal?
    CASE
        WHEN COALESCE(u.signal_class, 'analog') IN ('analog', 'periodic') THEN TRUE
        ELSE FALSE
    END AS calculus_valid,
    -- Is interpolation valid?
    CASE
        WHEN COALESCE(u.signal_class, 'analog') IN ('analog', 'periodic') THEN TRUE
        ELSE FALSE
    END AS interpolation_valid
FROM (
    SELECT DISTINCT signal_id, unit FROM observations
) o
LEFT JOIN unit_signal_class u ON LOWER(o.unit) = LOWER(u.unit);

-- Summary for UI
CREATE OR REPLACE VIEW v_signal_class_summary AS
SELECT
    signal_class,
    COUNT(*) AS n_signals,
    ARRAY_AGG(signal_id ORDER BY signal_id) AS signals
FROM v_signal_class_unit
GROUP BY signal_class
ORDER BY
    CASE signal_class
        WHEN 'analog' THEN 1
        WHEN 'periodic' THEN 2
        WHEN 'digital' THEN 3
        WHEN 'event' THEN 4
    END;

-- Show classification
SELECT * FROM v_signal_class_summary;
