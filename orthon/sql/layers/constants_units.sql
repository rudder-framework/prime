-- ============================================================================
-- ORTHON Constants & Units System
-- ============================================================================
-- For: validation, physics checks, unit conversion, domain-specific calculations
-- ============================================================================

-- ============================================================================
-- FUNDAMENTAL CONSTANTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS constants_fundamental (
    name VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    value DOUBLE,
    unit VARCHAR,
    uncertainty DOUBLE,
    source VARCHAR
);

INSERT OR REPLACE INTO constants_fundamental VALUES
-- Universal
('speed_of_light',           'c',      299792458,           'm/s',      0,           'CODATA 2018'),
('planck_constant',          'h',      6.62607015e-34,      'J·s',      0,           'CODATA 2018'),
('boltzmann_constant',       'k_B',    1.380649e-23,        'J/K',      0,           'CODATA 2018'),
('avogadro_constant',        'N_A',    6.02214076e23,       '1/mol',    0,           'CODATA 2018'),
('gas_constant',             'R',      8.314462618,         'J/(mol·K)', 0,          'CODATA 2018'),
('stefan_boltzmann',         'σ',      5.670374419e-8,      'W/(m²·K⁴)', 0,          'CODATA 2018'),
('faraday_constant',         'F',      96485.33212,         'C/mol',    0,           'CODATA 2018'),
('gravitational_constant',   'G',      6.67430e-11,         'N·m²/kg²', 1.5e-15,     'CODATA 2018'),

-- Electromagnetic
('vacuum_permittivity',      'ε₀',     8.8541878128e-12,    'F/m',      0,           'CODATA 2018'),
('vacuum_permeability',      'μ₀',     1.25663706212e-6,    'H/m',      0,           'CODATA 2018'),
('elementary_charge',        'e',      1.602176634e-19,     'C',        0,           'CODATA 2018'),

-- Atomic
('electron_mass',            'm_e',    9.1093837015e-31,    'kg',       0,           'CODATA 2018'),
('proton_mass',              'm_p',    1.67262192369e-27,   'kg',       0,           'CODATA 2018'),

-- Standard conditions
('standard_atmosphere',      'atm',    101325,              'Pa',       0,           'definition'),
('standard_gravity',         'g',      9.80665,             'm/s²',     0,           'definition'),
('standard_temperature',     'T_std',  273.15,              'K',        0,           'definition'),
('standard_temperature_25',  'T_25',   298.15,              'K',        0,           'definition');


-- ============================================================================
-- IDEAL GAS SPECIFIC CONSTANTS (R_specific = R / M)
-- ============================================================================

CREATE TABLE IF NOT EXISTS constants_gas (
    species VARCHAR PRIMARY KEY,
    formula VARCHAR,
    molar_mass DOUBLE,           -- kg/mol
    R_specific DOUBLE,           -- J/(kg·K)
    gamma DOUBLE,                -- Cp/Cv ratio
    Cp DOUBLE,                   -- J/(kg·K) at 298K
    Cv DOUBLE                    -- J/(kg·K) at 298K
);

INSERT OR REPLACE INTO constants_gas VALUES
('air',              'N2/O2',   0.02897,    287.05,   1.400,  1005,   718),
('nitrogen',         'N2',      0.02801,    296.80,   1.400,  1040,   743),
('oxygen',           'O2',      0.03200,    259.84,   1.395,  919,    659),
('hydrogen',         'H2',      0.00202,    4124.2,   1.405,  14300,  10180),
('helium',           'He',      0.00400,    2077.1,   1.667,  5193,   3116),
('carbon_dioxide',   'CO2',     0.04401,    188.92,   1.289,  844,    655),
('water_vapor',      'H2O',     0.01802,    461.52,   1.330,  1864,   1403),
('methane',          'CH4',     0.01604,    518.28,   1.307,  2226,   1703),
('ammonia',          'NH3',     0.01703,    488.21,   1.310,  2175,   1661),
('argon',            'Ar',      0.03995,    208.13,   1.667,  520,    312),
('propane',          'C3H8',    0.04410,    188.56,   1.131,  1679,   1485),
('natural_gas',      'mix',     0.01800,    461.9,    1.270,  2100,   1654);


-- ============================================================================
-- COMMON FLUID PROPERTIES (at standard conditions unless noted)
-- ============================================================================

CREATE TABLE IF NOT EXISTS constants_fluid (
    fluid VARCHAR PRIMARY KEY,
    density DOUBLE,              -- kg/m³
    viscosity_dynamic DOUBLE,    -- Pa·s
    viscosity_kinematic DOUBLE,  -- m²/s
    thermal_conductivity DOUBLE, -- W/(m·K)
    specific_heat DOUBLE,        -- J/(kg·K)
    surface_tension DOUBLE,      -- N/m (for liquids)
    vapor_pressure DOUBLE,       -- Pa (at T_ref)
    T_ref DOUBLE                 -- K
);

INSERT OR REPLACE INTO constants_fluid VALUES
-- Liquids at 25°C
('water',           998.2,    0.000890,   8.92e-7,   0.606,    4182,   0.0728,   3169,    298.15),
('seawater',        1025,     0.00108,    1.05e-6,   0.596,    3993,   0.0735,   3100,    298.15),
('ethanol',         789,      0.00109,    1.38e-6,   0.171,    2440,   0.0223,   7870,    298.15),
('methanol',        791,      0.000544,   6.87e-7,   0.203,    2530,   0.0226,   16900,   298.15),
('acetone',         784,      0.000306,   3.90e-7,   0.161,    2150,   0.0237,   30800,   298.15),
('glycerol',        1261,     0.934,      7.41e-4,   0.285,    2430,   0.0634,   0.03,    298.15),
('hydraulic_oil',   870,      0.032,      3.68e-5,   0.140,    1880,   0.029,    10,      298.15),
('diesel',          830,      0.00245,    2.95e-6,   0.130,    2090,   0.025,    300,     298.15),
('gasoline',        750,      0.00045,    6.0e-7,    0.115,    2220,   0.022,    35000,   298.15),

-- Gases at 25°C, 1 atm
('air_25C',         1.184,    1.849e-5,   1.56e-5,   0.0261,   1005,   NULL,     NULL,    298.15),
('steam_100C',      0.598,    1.27e-5,    2.12e-5,   0.0248,   2080,   NULL,     NULL,    373.15);


-- ============================================================================
-- UNIT SYSTEM DEFINITIONS
-- ============================================================================

CREATE TABLE IF NOT EXISTS unit_systems (
    system_name VARCHAR,
    quantity VARCHAR,
    base_unit VARCHAR,
    PRIMARY KEY (system_name, quantity)
);

INSERT OR REPLACE INTO unit_systems VALUES
-- SI (base)
('SI', 'length',        'm'),
('SI', 'mass',          'kg'),
('SI', 'time',          's'),
('SI', 'temperature',   'K'),
('SI', 'amount',        'mol'),
('SI', 'current',       'A'),
('SI', 'luminosity',    'cd'),
('SI', 'force',         'N'),
('SI', 'pressure',      'Pa'),
('SI', 'energy',        'J'),
('SI', 'power',         'W'),
('SI', 'frequency',     'Hz'),
('SI', 'voltage',       'V'),
('SI', 'flow_volume',   'm³/s'),
('SI', 'flow_mass',     'kg/s'),
('SI', 'viscosity_dyn', 'Pa·s'),
('SI', 'viscosity_kin', 'm²/s'),
('SI', 'density',       'kg/m³'),

-- US Customary
('US', 'length',        'ft'),
('US', 'mass',          'lb'),
('US', 'time',          's'),
('US', 'temperature',   '°F'),
('US', 'force',         'lbf'),
('US', 'pressure',      'psi'),
('US', 'energy',        'BTU'),
('US', 'power',         'hp'),
('US', 'flow_volume',   'gpm'),
('US', 'flow_mass',     'lb/hr'),

-- CGS
('CGS', 'length',       'cm'),
('CGS', 'mass',         'g'),
('CGS', 'force',        'dyn'),
('CGS', 'pressure',     'Ba'),
('CGS', 'energy',       'erg'),
('CGS', 'viscosity_dyn', 'P'),
('CGS', 'viscosity_kin', 'St');


-- ============================================================================
-- UNIT CONVERSION FACTORS (multiply to convert TO SI base unit)
-- ============================================================================

CREATE TABLE IF NOT EXISTS unit_conversions (
    unit VARCHAR PRIMARY KEY,
    quantity VARCHAR,
    to_si_factor DOUBLE,
    to_si_offset DOUBLE DEFAULT 0,  -- for temperature
    si_unit VARCHAR
);

INSERT OR REPLACE INTO unit_conversions VALUES
-- Length
('m',       'length',    1,              0,    'm'),
('km',      'length',    1000,           0,    'm'),
('cm',      'length',    0.01,           0,    'm'),
('mm',      'length',    0.001,          0,    'm'),
('μm',      'length',    1e-6,           0,    'm'),
('nm',      'length',    1e-9,           0,    'm'),
('in',      'length',    0.0254,         0,    'm'),
('ft',      'length',    0.3048,         0,    'm'),
('yd',      'length',    0.9144,         0,    'm'),
('mi',      'length',    1609.344,       0,    'm'),
('nmi',     'length',    1852,           0,    'm'),

-- Mass
('kg',      'mass',      1,              0,    'kg'),
('g',       'mass',      0.001,          0,    'kg'),
('mg',      'mass',      1e-6,           0,    'kg'),
('μg',      'mass',      1e-9,           0,    'kg'),
('lb',      'mass',      0.45359237,     0,    'kg'),
('oz',      'mass',      0.028349523,    0,    'kg'),
('ton',     'mass',      907.18474,      0,    'kg'),
('tonne',   'mass',      1000,           0,    'kg'),
('slug',    'mass',      14.593903,      0,    'kg'),

-- Time
('s',       'time',      1,              0,    's'),
('ms',      'time',      0.001,          0,    's'),
('μs',      'time',      1e-6,           0,    's'),
('ns',      'time',      1e-9,           0,    's'),
('min',     'time',      60,             0,    's'),
('hr',      'time',      3600,           0,    's'),
('day',     'time',      86400,          0,    's'),

-- Temperature (special: T_SI = factor * T + offset)
('K',       'temperature', 1,            0,       'K'),
('°C',      'temperature', 1,            273.15,  'K'),
('C',       'temperature', 1,            273.15,  'K'),
('°F',      'temperature', 0.5555556,    255.372, 'K'),
('F',       'temperature', 0.5555556,    255.372, 'K'),
('°R',      'temperature', 0.5555556,    0,       'K'),
('R',       'temperature', 0.5555556,    0,       'K'),

-- Pressure
('Pa',      'pressure',  1,              0,    'Pa'),
('kPa',     'pressure',  1000,           0,    'Pa'),
('MPa',     'pressure',  1e6,            0,    'Pa'),
('GPa',     'pressure',  1e9,            0,    'Pa'),
('bar',     'pressure',  1e5,            0,    'Pa'),
('mbar',    'pressure',  100,            0,    'Pa'),
('atm',     'pressure',  101325,         0,    'Pa'),
('psi',     'pressure',  6894.757,       0,    'Pa'),
('PSI',     'pressure',  6894.757,       0,    'Pa'),
('psig',    'pressure',  6894.757,       0,    'Pa'),
('psia',    'pressure',  6894.757,       0,    'Pa'),
('ksi',     'pressure',  6894757,        0,    'Pa'),
('mmHg',    'pressure',  133.322,        0,    'Pa'),
('torr',    'pressure',  133.322,        0,    'Pa'),
('inHg',    'pressure',  3386.39,        0,    'Pa'),
('inH2O',   'pressure',  249.089,        0,    'Pa'),
('ftH2O',   'pressure',  2989.07,        0,    'Pa'),

-- Force
('N',       'force',     1,              0,    'N'),
('kN',      'force',     1000,           0,    'N'),
('MN',      'force',     1e6,            0,    'N'),
('dyn',     'force',     1e-5,           0,    'N'),
('lbf',     'force',     4.448222,       0,    'N'),
('kgf',     'force',     9.80665,        0,    'N'),
('ozf',     'force',     0.278014,       0,    'N'),

-- Energy
('J',       'energy',    1,              0,    'J'),
('kJ',      'energy',    1000,           0,    'J'),
('MJ',      'energy',    1e6,            0,    'J'),
('GJ',      'energy',    1e9,            0,    'J'),
('Wh',      'energy',    3600,           0,    'J'),
('kWh',     'energy',    3.6e6,          0,    'J'),
('MWh',     'energy',    3.6e9,          0,    'J'),
('cal',     'energy',    4.184,          0,    'J'),
('kcal',    'energy',    4184,           0,    'J'),
('BTU',     'energy',    1055.06,        0,    'J'),
('therm',   'energy',    1.055e8,        0,    'J'),
('erg',     'energy',    1e-7,           0,    'J'),
('eV',      'energy',    1.60218e-19,    0,    'J'),
('ft·lbf',  'energy',    1.355818,       0,    'J'),

-- Power
('W',       'power',     1,              0,    'W'),
('kW',      'power',     1000,           0,    'W'),
('MW',      'power',     1e6,            0,    'W'),
('GW',      'power',     1e9,            0,    'W'),
('hp',      'power',     745.7,          0,    'W'),
('PS',      'power',     735.499,        0,    'W'),
('BTU/hr',  'power',     0.29307,        0,    'W'),
('ton_ref', 'power',     3516.85,        0,    'W'),

-- Volume
('m³',      'volume',    1,              0,    'm³'),
('L',       'volume',    0.001,          0,    'm³'),
('mL',      'volume',    1e-6,           0,    'm³'),
('cm³',     'volume',    1e-6,           0,    'm³'),
('cc',      'volume',    1e-6,           0,    'm³'),
('gal',     'volume',    0.003785412,    0,    'm³'),
('gal_uk',  'volume',    0.00454609,     0,    'm³'),
('qt',      'volume',    0.000946353,    0,    'm³'),
('pt',      'volume',    0.000473176,    0,    'm³'),
('fl_oz',   'volume',    2.95735e-5,     0,    'm³'),
('bbl',     'volume',    0.158987,       0,    'm³'),
('ft³',     'volume',    0.028316847,    0,    'm³'),
('in³',     'volume',    1.6387e-5,      0,    'm³'),

-- Volume flow rate
('m³/s',    'flow_volume', 1,            0,    'm³/s'),
('m³/h',    'flow_volume', 2.7778e-4,    0,    'm³/s'),
('m³/hr',   'flow_volume', 2.7778e-4,    0,    'm³/s'),
('L/s',     'flow_volume', 0.001,        0,    'm³/s'),
('L/min',   'flow_volume', 1.6667e-5,    0,    'm³/s'),
('L/h',     'flow_volume', 2.7778e-7,    0,    'm³/s'),
('gpm',     'flow_volume', 6.309e-5,     0,    'm³/s'),
('GPM',     'flow_volume', 6.309e-5,     0,    'm³/s'),
('gph',     'flow_volume', 1.0515e-6,    0,    'm³/s'),
('cfm',     'flow_volume', 4.7195e-4,    0,    'm³/s'),
('CFM',     'flow_volume', 4.7195e-4,    0,    'm³/s'),
('cfs',     'flow_volume', 0.028316847,  0,    'm³/s'),
('bbl/day', 'flow_volume', 1.8401e-6,    0,    'm³/s'),
('SCFM',    'flow_volume', 4.7195e-4,    0,    'm³/s'),

-- Mass flow rate
('kg/s',    'flow_mass', 1,              0,    'kg/s'),
('kg/h',    'flow_mass', 2.7778e-4,      0,    'kg/s'),
('kg/hr',   'flow_mass', 2.7778e-4,      0,    'kg/s'),
('g/s',     'flow_mass', 0.001,          0,    'kg/s'),
('lb/s',    'flow_mass', 0.45359237,     0,    'kg/s'),
('lb/h',    'flow_mass', 1.26e-4,        0,    'kg/s'),
('lb/hr',   'flow_mass', 1.26e-4,        0,    'kg/s'),
('ton/h',   'flow_mass', 0.2520,         0,    'kg/s'),
('tonne/h', 'flow_mass', 0.2778,         0,    'kg/s'),

-- Velocity
('m/s',     'velocity',  1,              0,    'm/s'),
('km/h',    'velocity',  0.27778,        0,    'm/s'),
('km/hr',   'velocity',  0.27778,        0,    'm/s'),
('mph',     'velocity',  0.44704,        0,    'm/s'),
('ft/s',    'velocity',  0.3048,         0,    'm/s'),
('ft/min',  'velocity',  0.00508,        0,    'm/s'),
('knot',    'velocity',  0.51444,        0,    'm/s'),
('in/s',    'velocity',  0.0254,         0,    'm/s'),
('mm/s',    'velocity',  0.001,          0,    'm/s'),

-- Acceleration
('m/s²',    'acceleration', 1,           0,    'm/s²'),
('ft/s²',   'acceleration', 0.3048,      0,    'm/s²'),
('g',       'acceleration', 9.80665,     0,    'm/s²'),
('Gal',     'acceleration', 0.01,        0,    'm/s²'),

-- Frequency
('Hz',      'frequency', 1,              0,    'Hz'),
('kHz',     'frequency', 1000,           0,    'Hz'),
('MHz',     'frequency', 1e6,            0,    'Hz'),
('GHz',     'frequency', 1e9,            0,    'Hz'),
('rpm',     'frequency', 0.01667,        0,    'Hz'),
('RPM',     'frequency', 0.01667,        0,    'Hz'),
('rad/s',   'frequency', 0.15915,        0,    'Hz'),
('cpm',     'frequency', 0.01667,        0,    'Hz'),

-- Dynamic viscosity
('Pa·s',    'viscosity_dyn', 1,          0,    'Pa·s'),
('mPa·s',   'viscosity_dyn', 0.001,      0,    'Pa·s'),
('P',       'viscosity_dyn', 0.1,        0,    'Pa·s'),
('cP',      'viscosity_dyn', 0.001,      0,    'Pa·s'),
('lb/(ft·s)', 'viscosity_dyn', 1.488,    0,    'Pa·s'),

-- Kinematic viscosity
('m²/s',    'viscosity_kin', 1,          0,    'm²/s'),
('mm²/s',   'viscosity_kin', 1e-6,       0,    'm²/s'),
('St',      'viscosity_kin', 1e-4,       0,    'm²/s'),
('cSt',     'viscosity_kin', 1e-6,       0,    'm²/s'),
('ft²/s',   'viscosity_kin', 0.0929,     0,    'm²/s'),

-- Density
('kg/m³',   'density',   1,              0,    'kg/m³'),
('g/cm³',   'density',   1000,           0,    'kg/m³'),
('g/mL',    'density',   1000,           0,    'kg/m³'),
('g/L',     'density',   1,              0,    'kg/m³'),
('kg/L',    'density',   1000,           0,    'kg/m³'),
('lb/ft³',  'density',   16.0185,        0,    'kg/m³'),
('lb/gal',  'density',   119.826,        0,    'kg/m³'),
('slug/ft³','density',   515.379,        0,    'kg/m³'),

-- Concentration
('mol/L',   'concentration', 1000,       0,    'mol/m³'),
('M',       'concentration', 1000,       0,    'mol/m³'),
('mM',      'concentration', 1,          0,    'mol/m³'),
('mol/m³',  'concentration', 1,          0,    'mol/m³'),
('ppm',     'concentration', 1,          0,    'ppm'),
('ppb',     'concentration', 0.001,      0,    'ppm'),
('%',       'concentration', 10000,      0,    'ppm'),
('wt%',     'concentration', 10000,      0,    'ppm'),
('vol%',    'concentration', 10000,      0,    'ppm'),

-- Electrical
('V',       'voltage',   1,              0,    'V'),
('mV',      'voltage',   0.001,          0,    'V'),
('kV',      'voltage',   1000,           0,    'V'),
('A',       'current',   1,              0,    'A'),
('mA',      'current',   0.001,          0,    'A'),
('μA',      'current',   1e-6,           0,    'A'),
('Ω',       'resistance', 1,             0,    'Ω'),
('kΩ',      'resistance', 1000,          0,    'Ω'),
('MΩ',      'resistance', 1e6,           0,    'Ω'),
('mΩ',      'resistance', 0.001,         0,    'Ω'),
('S',       'conductance', 1,            0,    'S'),
('mS',      'conductance', 0.001,        0,    'S'),

-- Thermal conductivity
('W/(m·K)', 'thermal_conductivity', 1,   0,    'W/(m·K)'),
('BTU/(hr·ft·°F)', 'thermal_conductivity', 1.7307, 0, 'W/(m·K)'),

-- Heat transfer coefficient
('W/(m²·K)', 'heat_transfer_coeff', 1,   0,    'W/(m²·K)'),
('BTU/(hr·ft²·°F)', 'heat_transfer_coeff', 5.678, 0, 'W/(m²·K)'),

-- Specific heat
('J/(kg·K)', 'specific_heat', 1,         0,    'J/(kg·K)'),
('kJ/(kg·K)', 'specific_heat', 1000,     0,    'J/(kg·K)'),
('BTU/(lb·°F)', 'specific_heat', 4186.8, 0,    'J/(kg·K)'),
('cal/(g·°C)', 'specific_heat', 4184,    0,    'J/(kg·K)'),

-- Angle
('rad',     'angle',     1,              0,    'rad'),
('deg',     'angle',     0.01745329,     0,    'rad'),
('°',       'angle',     0.01745329,     0,    'rad'),
('arcmin',  'angle',     2.9089e-4,      0,    'rad'),
('arcsec',  'angle',     4.8481e-6,      0,    'rad'),
('grad',    'angle',     0.01570796,     0,    'rad'),
('rev',     'angle',     6.2831853,      0,    'rad'),

-- Torque
('N·m',     'torque',    1,              0,    'N·m'),
('kN·m',    'torque',    1000,           0,    'N·m'),
('lbf·ft',  'torque',    1.35582,        0,    'N·m'),
('lbf·in',  'torque',    0.112985,       0,    'N·m'),
('kgf·m',   'torque',    9.80665,        0,    'N·m');


-- ============================================================================
-- UNIT TO SIGNAL CLASS MAPPING
-- ============================================================================

CREATE TABLE IF NOT EXISTS unit_signal_class (
    unit VARCHAR PRIMARY KEY,
    signal_class VARCHAR,          -- analog, digital, periodic, event
    quantity VARCHAR,
    interpolation_valid BOOLEAN,
    typical_range_min DOUBLE,
    typical_range_max DOUBLE,
    notes VARCHAR
);

INSERT OR REPLACE INTO unit_signal_class VALUES
-- Continuous physical quantities (ANALOG)
('PSI',     'analog', 'pressure',      TRUE,  0,      10000,  'Industrial pressure'),
('kPa',     'analog', 'pressure',      TRUE,  0,      100000, NULL),
('bar',     'analog', 'pressure',      TRUE,  0,      1000,   NULL),
('Pa',      'analog', 'pressure',      TRUE,  0,      1e8,    NULL),
('atm',     'analog', 'pressure',      TRUE,  0,      100,    NULL),
('K',       'analog', 'temperature',   TRUE,  0,      10000,  'Absolute temperature'),
('°C',      'analog', 'temperature',   TRUE,  -273,   5000,   NULL),
('°F',      'analog', 'temperature',   TRUE,  -460,   9000,   NULL),
('m/s',     'analog', 'velocity',      TRUE,  0,      1000,   NULL),
('ft/s',    'analog', 'velocity',      TRUE,  0,      3000,   NULL),
('m³/s',    'analog', 'flow_volume',   TRUE,  0,      10000,  NULL),
('gpm',     'analog', 'flow_volume',   TRUE,  0,      100000, 'US gallons per minute'),
('L/min',   'analog', 'flow_volume',   TRUE,  0,      100000, NULL),
('kg/s',    'analog', 'flow_mass',     TRUE,  0,      10000,  NULL),
('lb/hr',   'analog', 'flow_mass',     TRUE,  0,      1e6,    NULL),
('V',       'analog', 'voltage',       TRUE,  -1e6,   1e6,    NULL),
('mV',      'analog', 'voltage',       TRUE,  -1e6,   1e6,    NULL),
('A',       'analog', 'current',       TRUE,  -1e6,   1e6,    NULL),
('mA',      'analog', 'current',       TRUE,  -1e6,   1e6,    '4-20mA common'),
('W',       'analog', 'power',         TRUE,  0,      1e12,   NULL),
('kW',      'analog', 'power',         TRUE,  0,      1e9,    NULL),
('MW',      'analog', 'power',         TRUE,  0,      1e6,    NULL),
('Hz',      'analog', 'frequency',     TRUE,  0,      1e12,   NULL),
('rpm',     'analog', 'frequency',     TRUE,  0,      1e6,    NULL),
('mm/s',    'analog', 'velocity',      TRUE,  0,      10000,  'Vibration velocity'),
('g',       'analog', 'acceleration',  TRUE,  -1000,  1000,   'Vibration'),
('%',       'analog', 'percentage',    TRUE,  0,      100,    NULL),
('pH',      'analog', 'acidity',       TRUE,  0,      14,     NULL),
('dB',      'analog', 'level',         TRUE,  -200,   200,    NULL),

-- Discrete states (DIGITAL)
('state',   'digital', 'state',        FALSE, NULL,   NULL,   'Enumerated states'),
('mode',    'digital', 'state',        FALSE, NULL,   NULL,   'Operating modes'),
('status',  'digital', 'state',        FALSE, NULL,   NULL,   'On/off status'),
('bool',    'digital', 'state',        FALSE, 0,      1,      'Boolean'),
('binary',  'digital', 'state',        FALSE, 0,      1,      NULL),
('enum',    'digital', 'state',        FALSE, NULL,   NULL,   'Enumeration'),
('level',   'digital', 'state',        FALSE, NULL,   NULL,   'Discrete levels'),
('grade',   'digital', 'state',        FALSE, NULL,   NULL,   'Quality grades'),
('position','digital', 'state',        FALSE, NULL,   NULL,   'Valve positions'),

-- Sparse events (EVENT)
('count',   'event',  'count',         FALSE, 0,      NULL,   'Event counts'),
('events',  'event',  'count',         FALSE, 0,      NULL,   NULL),
('alarms',  'event',  'count',         FALSE, 0,      NULL,   'Alarm events'),
('alerts',  'event',  'count',         FALSE, 0,      NULL,   NULL),
('trips',   'event',  'count',         FALSE, 0,      NULL,   'Trip events'),
('faults',  'event',  'count',         FALSE, 0,      NULL,   'Fault events'),
('clicks',  'event',  'count',         FALSE, 0,      NULL,   'User interactions'),
('transactions', 'event', 'count',     FALSE, 0,      NULL,   NULL);


-- ============================================================================
-- DIMENSIONLESS NUMBERS (for physics validation)
-- ============================================================================

CREATE TABLE IF NOT EXISTS dimensionless_numbers (
    name VARCHAR PRIMARY KEY,
    symbol VARCHAR,
    formula VARCHAR,
    interpretation VARCHAR,
    typical_range_min DOUBLE,
    typical_range_max DOUBLE,
    transition_value DOUBLE,
    notes VARCHAR
);

INSERT OR REPLACE INTO dimensionless_numbers VALUES
('Reynolds',        'Re',   'ρvL/μ',         'Inertia vs viscosity',          0,    1e9,    2300,   'Laminar-turbulent transition'),
('Prandtl',         'Pr',   'Cpμ/k',         'Momentum vs thermal diffusivity', 0.001, 10000, NULL,  'Pr≈0.7 for air, ≈7 for water'),
('Nusselt',         'Nu',   'hL/k',          'Convection vs conduction',      1,    10000,  NULL,   NULL),
('Mach',            'Ma',   'v/c',           'Flow vs sound speed',           0,    25,     1,      'Supersonic transition'),
('Froude',          'Fr',   'v/√(gL)',       'Inertia vs gravity',            0,    100,    1,      'Subcritical/supercritical'),
('Weber',           'We',   'ρv²L/σ',        'Inertia vs surface tension',    0,    10000,  NULL,   'Droplet breakup'),
('Grashof',         'Gr',   'gβΔTL³/ν²',     'Buoyancy vs viscosity',         0,    1e12,   1e9,    'Natural convection transition'),
('Rayleigh',        'Ra',   'GrPr',          'Buoyancy-driven convection',    0,    1e15,   1708,   'Convection onset'),
('Schmidt',         'Sc',   'ν/D',           'Momentum vs mass diffusivity',  0.1,  10000,  NULL,   'Sc≈0.7 gases, ≈1000 liquids'),
('Sherwood',        'Sh',   'k_mL/D',        'Convective vs diffusive mass',  1,    10000,  NULL,   NULL),
('Peclet_thermal',  'Pe_t', 'vL/α',          'Advection vs thermal diffusion', 0,   1e6,    NULL,   NULL),
('Peclet_mass',     'Pe_m', 'vL/D',          'Advection vs mass diffusion',   0,    1e8,    NULL,   NULL),
('Strouhal',        'St',   'fL/v',          'Oscillation frequency',         0,    10,     0.2,    'Vortex shedding ≈0.2'),
('Euler',           'Eu',   'ΔP/(ρv²)',      'Pressure vs inertia',           0,    100,    NULL,   NULL),
('Damköhler',       'Da',   'τ_flow/τ_rxn',  'Residence vs reaction time',    0,    1e6,    1,      'Reaction-limited vs diffusion-limited'),
('Biot',            'Bi',   'hL/k_s',        'Surface vs internal resistance', 0,   100,    0.1,    'Bi<0.1: lumped capacitance valid'),
('Fourier',         'Fo',   'αt/L²',         'Thermal diffusion time',        0,    10,     NULL,   'Fo>0.2: steady state');


-- ============================================================================
-- UNIT VALIDATION VIEW
-- ============================================================================

CREATE OR REPLACE VIEW v_unit_validation AS
SELECT
    o.signal_id,
    o.unit,

    -- Is this a known unit?
    (SELECT COUNT(*) > 0 FROM unit_conversions WHERE unit = o.unit) AS unit_recognized,

    -- Signal class from unit
    COALESCE(usc.signal_class, 'unknown') AS inferred_signal_class,

    -- Interpolation valid?
    COALESCE(usc.interpolation_valid, TRUE) AS interpolation_allowed,

    -- Physical quantity
    COALESCE(usc.quantity, uc.quantity, 'unknown') AS quantity,

    -- Typical ranges
    usc.typical_range_min,
    usc.typical_range_max

FROM (SELECT DISTINCT signal_id, unit FROM observations) o
LEFT JOIN unit_signal_class usc ON o.unit = usc.unit
LEFT JOIN unit_conversions uc ON o.unit = uc.unit;


.print '✓ Constants and units system loaded'
