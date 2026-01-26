"""
File Inspector
==============

ORTHON's gatekeeper for uploaded observations.

Scans files to extract:
- Constants (header comments, constant columns, metadata)
- Units (from column name suffixes)
- Physical quantities (velocity, pressure, temperature, etc.)
- Signal classification for engine routing

This is the authoritative source - PRISM trusts whatever ORTHON sends.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import pandas as pd
import numpy as np


# =============================================================================
# UNIT SYSTEM - Comprehensive unit detection and classification
# =============================================================================

@dataclass
class UnitInfo:
    """Parsed unit information."""
    symbol: str           # Display symbol (e.g., "°F", "psi")
    quantity: str         # Physical quantity (e.g., "temperature", "pressure")
    si_unit: str          # SI equivalent (e.g., "K", "Pa")
    to_si: float          # Multiplication factor to convert to SI
    to_si_offset: float = 0.0  # Additive offset (for temperature)


# Unit patterns - suffix -> UnitInfo
# Format: (symbol, quantity, si_unit, to_si_factor, to_si_offset)
UNIT_DATABASE = {
    # Pressure
    r'_psi$':     UnitInfo('psi', 'pressure', 'Pa', 6894.76),
    r'_psia$':    UnitInfo('psia', 'pressure', 'Pa', 6894.76),
    r'_psig$':    UnitInfo('psig', 'pressure', 'Pa', 6894.76),
    r'_bar$':     UnitInfo('bar', 'pressure', 'Pa', 100000.0),
    r'_kpa$':     UnitInfo('kPa', 'pressure', 'Pa', 1000.0),
    r'_pa$':      UnitInfo('Pa', 'pressure', 'Pa', 1.0),
    r'_atm$':     UnitInfo('atm', 'pressure', 'Pa', 101325.0),
    r'_torr$':    UnitInfo('torr', 'pressure', 'Pa', 133.322),
    r'_mmhg$':    UnitInfo('mmHg', 'pressure', 'Pa', 133.322),

    # Temperature
    r'_F$':       UnitInfo('°F', 'temperature', 'K', 5/9, 255.372),
    r'_degF$':    UnitInfo('°F', 'temperature', 'K', 5/9, 255.372),
    r'_C$':       UnitInfo('°C', 'temperature', 'K', 1.0, 273.15),
    r'_degC$':    UnitInfo('°C', 'temperature', 'K', 1.0, 273.15),
    r'_K$':       UnitInfo('K', 'temperature', 'K', 1.0),
    r'_degR$':    UnitInfo('°R', 'temperature', 'K', 5/9),
    r'_R$':       UnitInfo('°R', 'temperature', 'K', 5/9),

    # Velocity / Speed
    r'_m_s$':     UnitInfo('m/s', 'velocity', 'm/s', 1.0),
    r'_mps$':     UnitInfo('m/s', 'velocity', 'm/s', 1.0),
    r'_ft_s$':    UnitInfo('ft/s', 'velocity', 'm/s', 0.3048),
    r'_fps$':     UnitInfo('ft/s', 'velocity', 'm/s', 0.3048),
    r'_km_h$':    UnitInfo('km/h', 'velocity', 'm/s', 0.27778),
    r'_kph$':     UnitInfo('km/h', 'velocity', 'm/s', 0.27778),
    r'_mph$':     UnitInfo('mph', 'velocity', 'm/s', 0.44704),
    r'_knots$':   UnitInfo('knots', 'velocity', 'm/s', 0.51444),

    # Flow rate (volumetric)
    r'_gpm$':     UnitInfo('gpm', 'volumetric_flow', 'm³/s', 6.309e-5),
    r'_lpm$':     UnitInfo('L/min', 'volumetric_flow', 'm³/s', 1.667e-5),
    r'_m3_s$':    UnitInfo('m³/s', 'volumetric_flow', 'm³/s', 1.0),
    r'_m3_h$':    UnitInfo('m³/h', 'volumetric_flow', 'm³/s', 2.778e-4),
    r'_cfm$':     UnitInfo('cfm', 'volumetric_flow', 'm³/s', 4.719e-4),
    r'_scfm$':    UnitInfo('scfm', 'volumetric_flow', 'm³/s', 4.719e-4),

    # Flow rate (mass)
    r'_kg_s$':    UnitInfo('kg/s', 'mass_flow', 'kg/s', 1.0),
    r'_kg_h$':    UnitInfo('kg/h', 'mass_flow', 'kg/s', 2.778e-4),
    r'_lbm_s$':   UnitInfo('lbm/s', 'mass_flow', 'kg/s', 0.4536),
    r'_lbm_h$':   UnitInfo('lbm/h', 'mass_flow', 'kg/s', 1.26e-4),

    # Length / Position
    r'_m$':       UnitInfo('m', 'length', 'm', 1.0),
    r'_mm$':      UnitInfo('mm', 'length', 'm', 0.001),
    r'_cm$':      UnitInfo('cm', 'length', 'm', 0.01),
    r'_km$':      UnitInfo('km', 'length', 'm', 1000.0),
    r'_in$':      UnitInfo('in', 'length', 'm', 0.0254),
    r'_ft$':      UnitInfo('ft', 'length', 'm', 0.3048),
    r'_yd$':      UnitInfo('yd', 'length', 'm', 0.9144),
    r'_mi$':      UnitInfo('mi', 'length', 'm', 1609.34),

    # Rotational speed
    r'_rpm$':     UnitInfo('rpm', 'angular_velocity', 'rad/s', 0.10472),
    r'_rps$':     UnitInfo('rps', 'angular_velocity', 'rad/s', 6.2832),
    r'_rad_s$':   UnitInfo('rad/s', 'angular_velocity', 'rad/s', 1.0),
    r'_deg_s$':   UnitInfo('°/s', 'angular_velocity', 'rad/s', 0.01745),

    # Frequency
    r'_hz$':      UnitInfo('Hz', 'frequency', 'Hz', 1.0),
    r'_khz$':     UnitInfo('kHz', 'frequency', 'Hz', 1000.0),
    r'_mhz$':     UnitInfo('MHz', 'frequency', 'Hz', 1e6),

    # Electrical
    r'_V$':       UnitInfo('V', 'voltage', 'V', 1.0),
    r'_mV$':      UnitInfo('mV', 'voltage', 'V', 0.001),
    r'_kV$':      UnitInfo('kV', 'voltage', 'V', 1000.0),
    r'_A$':       UnitInfo('A', 'current', 'A', 1.0),
    r'_mA$':      UnitInfo('mA', 'current', 'A', 0.001),
    r'_W$':       UnitInfo('W', 'power', 'W', 1.0),
    r'_kW$':      UnitInfo('kW', 'power', 'W', 1000.0),
    r'_MW$':      UnitInfo('MW', 'power', 'W', 1e6),
    r'_hp$':      UnitInfo('hp', 'power', 'W', 745.7),
    r'_ohm$':     UnitInfo('Ω', 'resistance', 'Ω', 1.0),
    r'_kohm$':    UnitInfo('kΩ', 'resistance', 'Ω', 1000.0),

    # Mass
    r'_kg$':      UnitInfo('kg', 'mass', 'kg', 1.0),
    r'_g$':       UnitInfo('g', 'mass', 'kg', 0.001),
    r'_mg$':      UnitInfo('mg', 'mass', 'kg', 1e-6),
    r'_lbm$':     UnitInfo('lbm', 'mass', 'kg', 0.4536),
    r'_ton$':     UnitInfo('ton', 'mass', 'kg', 907.185),
    r'_tonne$':   UnitInfo('tonne', 'mass', 'kg', 1000.0),

    # Force
    r'_N$':       UnitInfo('N', 'force', 'N', 1.0),
    r'_kN$':      UnitInfo('kN', 'force', 'N', 1000.0),
    r'_lbf$':     UnitInfo('lbf', 'force', 'N', 4.4482),

    # Torque
    r'_Nm$':      UnitInfo('N·m', 'torque', 'N·m', 1.0),
    r'_ftlb$':    UnitInfo('ft·lbf', 'torque', 'N·m', 1.3558),
    r'_inlb$':    UnitInfo('in·lbf', 'torque', 'N·m', 0.113),

    # Energy
    r'_J$':       UnitInfo('J', 'energy', 'J', 1.0),
    r'_kJ$':      UnitInfo('kJ', 'energy', 'J', 1000.0),
    r'_MJ$':      UnitInfo('MJ', 'energy', 'J', 1e6),
    r'_cal$':     UnitInfo('cal', 'energy', 'J', 4.184),
    r'_kcal$':    UnitInfo('kcal', 'energy', 'J', 4184.0),
    r'_btu$':     UnitInfo('BTU', 'energy', 'J', 1055.06),
    r'_kwh$':     UnitInfo('kWh', 'energy', 'J', 3.6e6),

    # Density
    r'_kg_m3$':   UnitInfo('kg/m³', 'density', 'kg/m³', 1.0),
    r'_g_cm3$':   UnitInfo('g/cm³', 'density', 'kg/m³', 1000.0),
    r'_lb_ft3$':  UnitInfo('lb/ft³', 'density', 'kg/m³', 16.0185),

    # Viscosity (dynamic)
    r'_Pa_s$':    UnitInfo('Pa·s', 'dynamic_viscosity', 'Pa·s', 1.0),
    r'_cP$':      UnitInfo('cP', 'dynamic_viscosity', 'Pa·s', 0.001),
    r'_mPa_s$':   UnitInfo('mPa·s', 'dynamic_viscosity', 'Pa·s', 0.001),
    r'_poise$':   UnitInfo('P', 'dynamic_viscosity', 'Pa·s', 0.1),

    # Viscosity (kinematic)
    r'_m2_s$':    UnitInfo('m²/s', 'kinematic_viscosity', 'm²/s', 1.0),
    r'_cSt$':     UnitInfo('cSt', 'kinematic_viscosity', 'm²/s', 1e-6),
    r'_St$':      UnitInfo('St', 'kinematic_viscosity', 'm²/s', 1e-4),

    # Concentration
    r'_mol_L$':   UnitInfo('mol/L', 'concentration', 'mol/m³', 1000.0),
    r'_mol_m3$':  UnitInfo('mol/m³', 'concentration', 'mol/m³', 1.0),
    r'_ppm$':     UnitInfo('ppm', 'concentration', 'ppm', 1.0),
    r'_ppb$':     UnitInfo('ppb', 'concentration', 'ppm', 0.001),

    # Angle
    r'_deg$':     UnitInfo('°', 'angle', 'rad', 0.01745),
    r'_rad$':     UnitInfo('rad', 'angle', 'rad', 1.0),

    # Acceleration
    r'_m_s2$':    UnitInfo('m/s²', 'acceleration', 'm/s²', 1.0),
    r'_g_acc$':   UnitInfo('g', 'acceleration', 'm/s²', 9.80665),
    r'_ft_s2$':   UnitInfo('ft/s²', 'acceleration', 'm/s²', 0.3048),

    # Dimensionless / Ratio
    r'_pct$':     UnitInfo('%', 'ratio', '', 0.01),
    r'_percent$': UnitInfo('%', 'ratio', '', 0.01),
    r'_ratio$':   UnitInfo('', 'ratio', '', 1.0),
}


# Physical quantity to potential role mapping (for engine routing)
QUANTITY_ROLES = {
    'velocity': ['velocity', 'speed', 'flow_velocity'],
    'pressure': ['pressure', 'head'],
    'temperature': ['temperature', 'thermal'],
    'length': ['position', 'displacement', 'height', 'depth'],
    'mass': ['mass', 'weight'],
    'force': ['force', 'load', 'thrust'],
    'power': ['power', 'work_rate'],
    'energy': ['energy', 'work', 'heat'],
    'volumetric_flow': ['flow_rate', 'discharge'],
    'mass_flow': ['mass_flow', 'feed_rate'],
    'angular_velocity': ['rotation', 'spin', 'rpm'],
    'density': ['density'],
    'dynamic_viscosity': ['viscosity'],
    'kinematic_viscosity': ['kinematic_viscosity'],
    'current': ['current', 'amperage'],
    'voltage': ['voltage', 'potential'],
    'torque': ['torque', 'moment'],
}


# =============================================================================
# CONSTANT DETECTION
# =============================================================================

@dataclass
class ConstantInfo:
    """Detected constant with metadata."""
    name: str
    value: float
    unit: Optional[str]
    quantity: Optional[str]
    source: str  # 'header', 'column', 'metadata'
    si_value: Optional[float] = None
    si_unit: Optional[str] = None


# Known constant name patterns -> physical quantity
CONSTANT_PATTERNS = {
    r'density|rho': 'density',
    r'viscosity|mu|nu': 'viscosity',
    r'diameter|dia|d_pipe': 'length',
    r'length|l_pipe': 'length',
    r'temperature|temp|t_ref': 'temperature',
    r'pressure|p_ref|p_atm': 'pressure',
    r'mass|m_total': 'mass',
    r'area|a_cross': 'area',
    r'velocity|v_ref': 'velocity',
    r'gravity|g': 'acceleration',
    r'specific_heat|cp|cv': 'specific_heat',
    r'thermal_conductivity|k_thermal': 'thermal_conductivity',
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ColumnInfo:
    """Detailed column information."""
    name: str
    dtype: str
    unit: Optional[str]
    unit_info: Optional[UnitInfo]
    quantity: Optional[str]
    is_constant: bool
    constant_value: Optional[float]
    is_entity_id: bool
    is_sequence: bool
    sample_values: List[Any]
    null_pct: float
    stats: Dict[str, float] = field(default_factory=dict)

    def to_dict(self):
        return {
            "name": str(self.name),
            "dtype": str(self.dtype),
            "unit": self.unit,
            "quantity": self.quantity,
            "si_unit": self.unit_info.si_unit if self.unit_info else None,
            "to_si": self.unit_info.to_si if self.unit_info else None,
            "is_constant": bool(self.is_constant),
            "constant_value": self.constant_value,
            "is_entity_id": bool(self.is_entity_id),
            "is_sequence": bool(self.is_sequence),
            "sample_values": [
                float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v)
                for v in self.sample_values
            ],
            "null_pct": round(self.null_pct, 2),
            "stats": self.stats,
        }


@dataclass
class FileInspection:
    """Complete file inspection result - ORTHON's source of truth."""

    # File info
    filename: str
    row_count: int
    column_count: int

    # Detected structure
    entity_column: Optional[str]
    sequence_column: Optional[str]
    entities: List[str]

    # Signals (time-varying columns)
    signals: List[ColumnInfo]

    # Constants (from all sources)
    constants: Dict[str, ConstantInfo]

    # Summary for quick access
    units_detected: int
    quantities_detected: List[str]

    # Validation
    errors: List[str]
    warnings: List[str]

    def to_dict(self):
        return {
            "filename": self.filename,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "entity_column": self.entity_column,
            "sequence_column": self.sequence_column,
            "entities": self.entities,
            "signals": [s.to_dict() for s in self.signals],
            "constants": {k: {
                "name": v.name,
                "value": v.value,
                "unit": v.unit,
                "quantity": v.quantity,
                "source": v.source,
                "si_value": v.si_value,
                "si_unit": v.si_unit,
            } for k, v in self.constants.items()},
            "units_detected": self.units_detected,
            "quantities_detected": self.quantities_detected,
            "errors": self.errors,
            "warnings": self.warnings,
        }

    def get_signals_by_quantity(self, quantity: str) -> List[ColumnInfo]:
        """Get all signals of a specific physical quantity."""
        return [s for s in self.signals if s.quantity == quantity]

    def get_constant(self, name: str) -> Optional[ConstantInfo]:
        """Get a constant by name (case-insensitive)."""
        name_lower = name.lower()
        for k, v in self.constants.items():
            if k.lower() == name_lower:
                return v
        return None

    def has_requirements(self, requirements: List[str]) -> Tuple[bool, List[str]]:
        """Check if inspection has required quantities/constants."""
        missing = []
        quantities = set(self.quantities_detected)
        constant_names = set(k.lower() for k in self.constants.keys())

        for req in requirements:
            req_lower = req.lower()
            if req_lower not in quantities and req_lower not in constant_names:
                missing.append(req)

        return len(missing) == 0, missing


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

# Entity column patterns (order matters - more specific first)
ENTITY_PATTERNS = [
    r'^entity_id$', r'^entity$', r'^signal_id$', r'^sensor_id$',
    r'^equipment_id$', r'^asset_id$', r'^machine_id$', r'^device_id$',
    r'^pump_id$', r'^engine_id$', r'^run_id$', r'^batch_id$',
    r'^unit_id$', r'^id$',
]

# Sequence column patterns
SEQUENCE_PATTERNS = [
    r'^timestamp', r'^time$', r'^datetime', r'^date$',
    r'^cycle', r'^index$', r'^idx$', r'^step', r'^t$', r'^i$', r'^n$',
    r'^seq$', r'^row$', r'^sample$', r'^tick$', r'^frame$',
    r'_min$', r'_sec$', r'_hr$', r'_ms$',
]


def detect_unit(column_name: str) -> Optional[UnitInfo]:
    """Detect unit from column name suffix."""
    for pattern, unit_info in UNIT_DATABASE.items():
        if re.search(pattern, column_name, re.IGNORECASE):
            return unit_info
    return None


def detect_entity_column(columns: List[str]) -> Optional[str]:
    """Find entity ID column."""
    for col in columns:
        for pattern in ENTITY_PATTERNS:
            if re.match(pattern, col, re.IGNORECASE):
                return col
    return None


def detect_sequence_column(columns: List[str]) -> Optional[str]:
    """Find timestamp/sequence column."""
    for col in columns:
        for pattern in SEQUENCE_PATTERNS:
            if re.search(pattern, col, re.IGNORECASE):
                return col
    return None


def detect_constant_quantity(name: str) -> Optional[str]:
    """Detect physical quantity from constant name."""
    name_lower = name.lower()
    for pattern, quantity in CONSTANT_PATTERNS.items():
        if re.search(pattern, name_lower):
            return quantity
    return None


def parse_header_constants(filepath: Path) -> Dict[str, ConstantInfo]:
    """Parse constants from CSV header comments."""
    constants = {}

    if filepath.suffix != '.csv':
        return constants

    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('#'):
                    # Parse: # name = value or # name_unit = value
                    # Also: # name = value unit
                    match = re.match(r'#\s*(\w+)\s*=\s*([0-9.eE+-]+)\s*(\w+)?', line)
                    if match:
                        name, value_str, unit = match.groups()
                        try:
                            value = float(value_str)
                            quantity = detect_constant_quantity(name)

                            # Try to get SI conversion if unit provided
                            si_value = None
                            si_unit = None
                            if unit:
                                for pattern, unit_info in UNIT_DATABASE.items():
                                    if re.search(pattern, f"_{unit}", re.IGNORECASE):
                                        si_value = value * unit_info.to_si + unit_info.to_si_offset
                                        si_unit = unit_info.si_unit
                                        if not quantity:
                                            quantity = unit_info.quantity
                                        break

                            constants[name.lower()] = ConstantInfo(
                                name=name,
                                value=value,
                                unit=unit,
                                quantity=quantity,
                                source='header',
                                si_value=si_value,
                                si_unit=si_unit,
                            )
                        except ValueError:
                            pass
                else:
                    break
    except Exception:
        pass

    return constants


def extract_column_constant(df: pd.DataFrame, col: str, entity_column: Optional[str]) -> Optional[float]:
    """Extract constant value from a column."""
    if entity_column and entity_column in df.columns:
        # Check if constant within each entity
        grouped = df.groupby(entity_column)[col].nunique()
        if (grouped == 1).all():
            return float(df[col].iloc[0])
    else:
        if df[col].nunique() == 1:
            return float(df[col].iloc[0])
    return None


# =============================================================================
# MAIN INSPECTION FUNCTION
# =============================================================================

def inspect_file(filepath: str) -> FileInspection:
    """
    Inspect an uploaded file and extract structure.

    This is ORTHON's gatekeeper - the authoritative source of truth for:
    - What signals exist and their units
    - What constants are available
    - What physical quantities are present

    PRISM trusts whatever ORTHON tells it.

    Args:
        filepath: Path to CSV, XLSX, or Parquet file

    Returns:
        FileInspection with complete metadata
    """
    filepath = Path(filepath)
    errors = []
    warnings = []

    # Parse header constants (CSV only)
    constants = parse_header_constants(filepath)

    # Read data
    df = None
    try:
        if filepath.suffix == '.csv':
            df = pd.read_csv(filepath, comment='#')
        elif filepath.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(filepath)
        elif filepath.suffix == '.parquet':
            df = pd.read_parquet(filepath)
        else:
            errors.append(f"Unsupported file type: {filepath.suffix}")
    except Exception as e:
        errors.append(f"Failed to read file: {str(e)}")

    if df is None or df.empty:
        return FileInspection(
            filename=filepath.name,
            row_count=0,
            column_count=0,
            entity_column=None,
            sequence_column=None,
            entities=[],
            signals=[],
            constants=constants,
            units_detected=0,
            quantities_detected=[],
            errors=errors or ["File is empty"],
            warnings=warnings,
        )

    # Detect entity column
    entity_column = detect_entity_column(df.columns.tolist())
    if not entity_column:
        warnings.append("No entity column detected. Treating as single entity.")

    # Detect sequence column
    sequence_column = detect_sequence_column(df.columns.tolist())
    if not sequence_column:
        warnings.append("No timestamp/sequence column detected.")

    # Get entities
    if entity_column and entity_column in df.columns:
        entities = df[entity_column].unique().tolist()
        entities = [str(e) for e in entities]
    else:
        entities = ['default']

    # Check for dedicated unit column (e.g., "unit" or "units" column with values like "V", "PSI")
    unit_column = None
    unit_column_values = []
    for col in df.columns:
        if col.lower() in ('unit', 'units', 'uom', 'unit_of_measure'):
            unit_column = col
            unit_column_values = df[col].dropna().unique().tolist()
            break

    # Analyze each column
    signals = []
    quantities_detected = set()
    units_detected = 0

    for col in df.columns:
        # Detect unit
        unit_info = detect_unit(str(col))
        unit = unit_info.symbol if unit_info else None
        quantity = unit_info.quantity if unit_info else None

        if unit:
            units_detected += 1
        if quantity:
            quantities_detected.add(quantity)

        # Check if numeric
        is_numeric = df[col].dtype in ['float64', 'int64', 'float32', 'int32', 'float16']

        # Check if constant
        is_constant = False
        constant_value = None
        if is_numeric:
            constant_value = extract_column_constant(df, col, entity_column)
            is_constant = constant_value is not None

            # Add to constants dict if it's a constant column
            if is_constant:
                const_quantity = quantity or detect_constant_quantity(col)

                # Calculate SI value if we have unit info
                si_value = None
                si_unit = None
                if unit_info:
                    si_value = constant_value * unit_info.to_si + unit_info.to_si_offset
                    si_unit = unit_info.si_unit

                constants[col.lower()] = ConstantInfo(
                    name=col,
                    value=constant_value,
                    unit=unit,
                    quantity=const_quantity,
                    source='column',
                    si_value=si_value,
                    si_unit=si_unit,
                )

        # Calculate null percentage
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100

        # Get sample values
        try:
            sample_values = df[col].dropna().head(3).tolist()
        except Exception:
            sample_values = []

        # Get basic stats for numeric columns
        stats = {}
        if is_numeric and not is_constant:
            try:
                stats = {
                    'min': float(df[col].min()),
                    'max': float(df[col].max()),
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                }
            except Exception:
                pass

        col_info = ColumnInfo(
            name=col,
            dtype=str(df[col].dtype),
            unit=unit,
            unit_info=unit_info,
            quantity=quantity,
            is_constant=is_constant,
            constant_value=constant_value,
            is_entity_id=(col == entity_column),
            is_sequence=(col == sequence_column),
            sample_values=sample_values,
            null_pct=null_pct,
            stats=stats,
        )
        signals.append(col_info)

    # Generate warnings for potential issues
    if units_detected == 0 and not unit_column_values:
        warnings.append("No units detected in column names. Consider using suffixes like _psi, _degF, _m_s")
    elif unit_column:
        units_detected = len(unit_column_values)  # Count unique units from unit column

    high_null_cols = [s.name for s in signals if s.null_pct > 20 and not s.is_constant]
    if high_null_cols:
        warnings.append(f"High null rate (>20%) in: {', '.join(high_null_cols[:3])}")

    return FileInspection(
        filename=filepath.name,
        row_count=len(df),
        column_count=len(df.columns),
        entity_column=entity_column,
        sequence_column=sequence_column,
        entities=entities,
        signals=signals,
        constants=constants,
        units_detected=units_detected,
        quantities_detected=sorted(quantities_detected),
        errors=errors,
        warnings=warnings,
    )
