"""
ORTHON Data Validation
======================

Pre-flight checks and data profiling.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# Unit suffix detection
SUFFIX_TO_UNIT = {
    # Pressure
    '_psi': ('psi', 'pressure'),
    '_bar': ('bar', 'pressure'),
    '_kpa': ('kPa', 'pressure'),
    '_mpa': ('MPa', 'pressure'),
    '_pa': ('Pa', 'pressure'),
    '_atm': ('atm', 'pressure'),

    # Temperature
    '_f': ('°F', 'temperature'),
    '_c': ('°C', 'temperature'),
    '_k': ('K', 'temperature'),
    '_degf': ('°F', 'temperature'),
    '_degc': ('°C', 'temperature'),
    '_degr': ('°R', 'temperature'),

    # Flow
    '_gpm': ('gpm', 'flow'),
    '_lpm': ('L/min', 'flow'),
    '_m3h': ('m³/h', 'flow'),
    '_cfm': ('cfm', 'flow'),

    # Velocity
    '_mps': ('m/s', 'velocity'),
    '_fps': ('ft/s', 'velocity'),
    '_mph': ('mph', 'velocity'),
    '_kph': ('km/h', 'velocity'),

    # Length
    '_m': ('m', 'length'),
    '_mm': ('mm', 'length'),
    '_cm': ('cm', 'length'),
    '_in': ('in', 'length'),
    '_ft': ('ft', 'length'),

    # Mass
    '_kg': ('kg', 'mass'),
    '_lb': ('lb', 'mass'),
    '_g': ('g', 'mass'),

    # Frequency / Speed
    '_rpm': ('rpm', 'frequency'),
    '_hz': ('Hz', 'frequency'),
    '_khz': ('kHz', 'frequency'),

    # Electrical
    '_v': ('V', 'voltage'),
    '_mv': ('mV', 'voltage'),
    '_a': ('A', 'current'),
    '_ma': ('mA', 'current'),
    '_w': ('W', 'power'),
    '_kw': ('kW', 'power'),
    '_mw': ('MW', 'power'),
    '_ohm': ('Ω', 'resistance'),

    # Ratio
    '_pct': ('%', 'ratio'),
    '_percent': ('%', 'ratio'),

    # Time
    '_s': ('s', 'time'),
    '_ms': ('ms', 'time'),
    '_min': ('min', 'time'),
    '_hr': ('hr', 'time'),
}

# Known column names for entity/time
ENTITY_COLUMNS = [
    'entity_id', 'unit_id', 'equipment_id', 'asset_id', 'machine_id',
    'engine_id', 'pump_id', 'id', 'unit', 'device_id', 'sensor_id',
]

TIME_COLUMNS = [
    'timestamp', 'time', 'datetime', 'date', 'cycle', 't', 'ts',
    'time_stamp', 'date_time',
]


def detect_units(column_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect unit and category from column name suffix.

    Returns:
        (unit, category) or (None, None) if not detected
    """
    name_lower = column_name.lower()

    for suffix, (unit, category) in SUFFIX_TO_UNIT.items():
        if name_lower.endswith(suffix):
            return unit, category

    return None, None


def detect_columns(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Auto-detect column roles (entity, time, signals, constants).

    Returns:
        {
            'entity_col': str or None,
            'time_col': str or None,
            'signals': list of column names,
            'constants': list of column names,
            'other': list of column names,
        }
    """
    result = {
        'entity_col': None,
        'time_col': None,
        'signals': [],
        'constants': [],
        'other': [],
    }

    for col in df.columns:
        name_lower = col.lower()

        # Check for entity column
        if name_lower in ENTITY_COLUMNS:
            result['entity_col'] = col
            continue

        # Check for time column
        if name_lower in TIME_COLUMNS:
            result['time_col'] = col
            continue

        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            # Constant if single unique value
            if df[col].nunique() == 1:
                result['constants'].append(col)
            else:
                result['signals'].append(col)
        else:
            result['other'].append(col)

    return result


def validate(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Full validation of uploaded data.

    Returns:
        {
            'valid': bool,
            'rows': int,
            'columns': int,
            'issues': list of critical problems,
            'warnings': list of potential problems,
            'column_info': list of column details,
            'structure': detected column roles,
        }
    """
    result = {
        'valid': True,
        'rows': len(df),
        'columns': len(df.columns),
        'issues': [],
        'warnings': [],
        'column_info': [],
        'structure': detect_columns(df),
    }

    # Check for empty data
    if len(df) == 0:
        result['valid'] = False
        result['issues'].append("Data is empty (0 rows)")
        return result

    if len(df.columns) == 0:
        result['valid'] = False
        result['issues'].append("Data has no columns")
        return result

    # Analyze each column
    for col in df.columns:
        unit, category = detect_units(col)

        info = {
            'name': col,
            'dtype': str(df[col].dtype),
            'unit': unit,
            'category': category,
            'nulls': int(df[col].isna().sum()),
            'null_pct': float(df[col].isna().mean() * 100),
        }

        # Numeric stats
        if pd.api.types.is_numeric_dtype(df[col]):
            valid_data = df[col].dropna()
            if len(valid_data) > 0:
                info.update({
                    'min': float(valid_data.min()),
                    'max': float(valid_data.max()),
                    'mean': float(valid_data.mean()),
                    'std': float(valid_data.std()) if len(valid_data) > 1 else 0.0,
                    'unique': int(df[col].nunique()),
                })

        result['column_info'].append(info)

        # Validation checks
        _check_column(col, info, category, result)

    # Overall checks
    if not result['structure']['signals']:
        result['warnings'].append("No signal columns detected")

    return result


def _check_column(col: str, info: dict, category: Optional[str], result: dict):
    """Run validation checks on a single column."""

    # Entirely null
    if info['nulls'] == result['rows']:
        result['issues'].append(f"`{col}` is entirely null")
        result['valid'] = False
        return

    # Partial nulls
    if info['nulls'] > 0:
        result['warnings'].append(
            f"`{col}` has {info['nulls']} nulls ({info['null_pct']:.1f}%)"
        )

    # Category-specific checks
    if category == 'temperature':
        min_val = info.get('min', 0)
        if min_val < -273.15:
            result['issues'].append(
                f"`{col}` has temperature below absolute zero ({min_val:.1f})"
            )

    if category == 'pressure':
        min_val = info.get('min', 0)
        if min_val < 0:
            result['warnings'].append(
                f"`{col}` has negative pressure ({min_val:.2f})"
            )

    if category == 'flow':
        min_val = info.get('min', 0)
        if min_val < 0:
            result['warnings'].append(
                f"`{col}` has negative flow ({min_val:.2f})"
            )

    # Constant column
    if info.get('unique') == 1:
        result['warnings'].append(
            f"`{col}` is constant (value: {info.get('mean', 'N/A')})"
        )


def get_entities(df: pd.DataFrame, entity_col: Optional[str] = None) -> List[Any]:
    """Get list of unique entities in data."""
    if entity_col is None:
        structure = detect_columns(df)
        entity_col = structure['entity_col']

    if entity_col and entity_col in df.columns:
        return df[entity_col].unique().tolist()

    return []
