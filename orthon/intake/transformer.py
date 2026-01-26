"""
ORTHON Intake — Transform user data to PRISM format

User uploads whatever → ORTHON produces:
  1. observations.parquet (tidy long format)
  2. config.json (constants, metadata)

CANONICAL SCHEMA - THE RULE:
  ┌───────────┬─────────┬──────────┬───────────────────────────────────────────────┐
  │  Column   │  Type   │ Required │                  Description                  │
  ├───────────┼─────────┼──────────┼───────────────────────────────────────────────┤
  │ entity_id │ Utf8    │ Yes      │ Entity identifier                             │
  │ signal_id │ Utf8    │ Yes      │ Signal identifier                             │
  │ I         │ Float64 │ Yes      │ Index (time, cycle, depth, sample)            │
  │ y         │ Float64 │ Yes      │ Value (the measurement)                       │
  │ unit      │ Utf8    │ Yes      │ Unit string (enables physics calculations)    │
  └───────────┴─────────┴──────────┴───────────────────────────────────────────────┘

I means I. y means y. No aliases after intake. This is the rule.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime

from ..shared.config_schema import PrismConfig, SignalInfo, DISCIPLINES, DisciplineType


# =============================================================================
# UNIT DETECTION
# =============================================================================

SUFFIX_TO_UNIT = {
    # Pressure
    '_psi': 'psi', '_bar': 'bar', '_kpa': 'kPa', '_pa': 'Pa', '_mpa': 'MPa',
    '_psia': 'psia', '_psig': 'psig',
    # Temperature
    '_f': '°F', '_c': '°C', '_k': 'K',
    '_degf': '°F', '_degc': '°C', '_degk': 'K', '_degr': '°R',
    # Flow
    '_gpm': 'gpm', '_lpm': 'L/min', '_m3s': 'm³/s', '_cfm': 'cfm',
    '_kg_s': 'kg/s', '_kg_h': 'kg/h', '_lb_h': 'lb/h',
    # Length
    '_m': 'm', '_mm': 'mm', '_cm': 'cm', '_ft': 'ft', '_in': 'in',
    # Velocity
    '_mps': 'm/s', '_fps': 'ft/s', '_mph': 'mph', '_kph': 'km/h',
    # Mass
    '_kg': 'kg', '_g': 'g', '_lb': 'lb', '_mg': 'mg',
    # Frequency/Speed
    '_rpm': 'rpm', '_hz': 'Hz', '_khz': 'kHz',
    # Electrical
    '_v': 'V', '_mv': 'mV', '_a': 'A', '_ma': 'mA', '_w': 'W', '_kw': 'kW',
    '_ohm': 'Ω', '_ohm_m': 'Ω·m',
    # Ratio/Percent
    '_pct': '%', '_percent': '%', '_ratio': '',
    # Density
    '_kg_m3': 'kg/m³', '_g_cc': 'g/cm³', '_ppg': 'ppg',
    # Viscosity
    '_cp': 'cP', '_pa_s': 'Pa·s',
    # Energy
    '_j': 'J', '_kj': 'kJ', '_btu': 'BTU',
    # Other
    '_api': 'API', '_cc': 'cm³',
}

# Columns that indicate sequence (x-axis)
SEQUENCE_PATTERNS = [
    'timestamp', 'time', 'datetime', 'date', 't', 'ts',  # Time
    'cycle', 'cycles',  # Cycles
    'depth', 'depth_m', 'depth_ft', 'tvd', 'md', 'z',  # Depth
    'distance', 'chainage', 'station', 'position', 'x',  # Distance
    'index', 'sequence', 'sample', 'n', 'step', 'i',  # Generic
]

# Columns that indicate entity grouping
ENTITY_PATTERNS = [
    'entity_id', 'unit_id', 'equipment_id', 'asset_id', 'machine_id',
    'engine_id', 'pump_id', 'well_id', 'tag', 'id', 'unit',
]


def detect_unit(col_name: str) -> Optional[str]:
    """Detect unit from column name suffix"""
    name_lower = str(col_name).lower()
    for suffix, unit in SUFFIX_TO_UNIT.items():
        if name_lower.endswith(suffix):
            return unit
    return None


def strip_unit_suffix(col_name: str) -> str:
    """Remove unit suffix from column name to get signal_id"""
    col_name = str(col_name)
    name_lower = col_name.lower()
    for suffix in SUFFIX_TO_UNIT.keys():
        if name_lower.endswith(suffix):
            return col_name[:len(col_name) - len(suffix)]
    return col_name


def is_sequence_column(col_name: str) -> bool:
    """Check if column is a sequence (x-axis) column"""
    return str(col_name).lower() in SEQUENCE_PATTERNS


def is_entity_column(col_name: str) -> bool:
    """Check if column is an entity grouping column"""
    return str(col_name).lower() in ENTITY_PATTERNS


# =============================================================================
# MAIN TRANSFORMER
# =============================================================================

class IntakeTransformer:
    """
    Transform user data to PRISM format.

    Input: CSV/Parquet/TSV (wide format, any structure)
    Output: observations.parquet + config.json
    """

    def __init__(self, discipline: DisciplineType = None):
        """
        Initialize transformer.

        Args:
            discipline: Optional discipline for specialized physics engines
        """
        self.discipline = discipline
        self._config: Optional[PrismConfig] = None
        self._header_constants: Dict[str, Any] = {}

    @property
    def config(self) -> Optional[PrismConfig]:
        """Get the config after transformation"""
        return self._config

    def transform(
        self,
        source: Union[str, Path, pd.DataFrame],
        output_dir: Union[str, Path] = ".",
    ) -> Tuple[Path, Path]:
        """
        Transform user data to PRISM format.

        Args:
            source: Path to file or DataFrame
            output_dir: Where to write observations.parquet and config.json

        Returns:
            Tuple of (observations_path, config_path)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load data
        df, header_constants = self._load_data(source)
        self._header_constants = header_constants

        # Detect structure
        structure = self._detect_structure(df)

        # Build config
        self._build_config(source, df, structure, header_constants)

        # Transform to long format
        observations_df = self._to_long_format(df, structure)

        # Write outputs
        obs_path = output_dir / "observations.parquet"
        config_path = output_dir / "config.json"

        observations_df.to_parquet(obs_path, index=False)
        self._config.to_json(config_path)

        return obs_path, config_path

    def transform_df(
        self,
        source: Union[str, Path, pd.DataFrame],
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Transform to PRISM format, return DataFrames (no file I/O).

        Returns:
            Tuple of (observations_df, config_dict)
        """
        # Load data
        df, header_constants = self._load_data(source)
        self._header_constants = header_constants

        # Detect structure
        structure = self._detect_structure(df)

        # Build config
        self._build_config(source, df, structure, header_constants)

        # Transform to long format
        observations_df = self._to_long_format(df, structure)

        return observations_df, self._config.model_dump()

    def _load_data(self, source: Union[str, Path, pd.DataFrame]) -> Tuple[pd.DataFrame, dict]:
        """Load data and extract header constants"""

        if isinstance(source, pd.DataFrame):
            return source, {}

        source = Path(source)
        header_constants = {}

        if source.suffix == '.parquet':
            return pd.read_parquet(source), {}

        # Read file to extract header comments
        with open(source, 'r') as f:
            lines = f.readlines()

        data_start = 0
        for i, line in enumerate(lines):
            line = line.strip()
            if line.startswith('#'):
                # Parse constant: # name: value
                match = re.match(r'#\s*(\w+)\s*:\s*(.+)', line)
                if match:
                    key, value = match.groups()
                    # Try to parse as number
                    try:
                        if '.' in value:
                            value = float(value.split()[0])  # Handle "850 kg/m³"
                        else:
                            value = int(value.split()[0])
                    except ValueError:
                        value = value.strip()
                    header_constants[key.lower()] = value
            elif line:
                data_start = i
                break

        # Read CSV skipping header comments
        if source.suffix == '.tsv':
            df = pd.read_csv(source, sep='\t', skiprows=data_start)
        else:
            df = pd.read_csv(source, skiprows=data_start)

        return df, header_constants

    def _detect_structure(self, df: pd.DataFrame) -> dict:
        """Detect data structure: sequence column, entity column, signals"""

        structure = {
            'sequence_column': None,
            'sequence_unit': None,
            'entity_column': None,
            'signal_columns': [],
            'constant_columns': [],
        }

        # Find sequence column
        for col in df.columns:
            if is_sequence_column(col):
                structure['sequence_column'] = col
                structure['sequence_unit'] = detect_unit(col)
                break

        # If no sequence column found, check for monotonic numeric
        if structure['sequence_column'] is None:
            for col in df.columns:
                if pd.api.types.is_numeric_dtype(df[col]):
                    if df[col].is_monotonic_increasing or df[col].is_monotonic_decreasing:
                        structure['sequence_column'] = col
                        structure['sequence_unit'] = detect_unit(col)
                        break

        # Find entity column
        for col in df.columns:
            if is_entity_column(col):
                structure['entity_column'] = col
                break

        # Classify remaining columns
        for col in df.columns:
            if col == structure['sequence_column'] or col == structure['entity_column']:
                continue

            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            # Check if constant (same value throughout, or per entity)
            if df[col].nunique() == 1:
                structure['constant_columns'].append(col)
            elif structure['entity_column']:
                # Check if constant per entity
                nunique_per_entity = df.groupby(structure['entity_column'])[col].nunique()
                if (nunique_per_entity == 1).all() and df[col].nunique() > 1:
                    structure['constant_columns'].append(col)
                else:
                    structure['signal_columns'].append(col)
            else:
                structure['signal_columns'].append(col)

        return structure

    def _build_config(
        self,
        source: Union[str, Path, pd.DataFrame],
        df: pd.DataFrame,
        structure: dict,
        header_constants: dict,
    ):
        """Build PRISM config"""

        # Determine sequence name
        sequence_name = "index"
        if structure['sequence_column']:
            seq_lower = str(structure['sequence_column']).lower()
            if any(t in seq_lower for t in ['time', 'date', 'ts']):
                sequence_name = "time"
            elif any(t in seq_lower for t in ['depth', 'tvd', 'md']):
                sequence_name = "depth"
            elif any(t in seq_lower for t in ['cycle']):
                sequence_name = "cycle"
            elif any(t in seq_lower for t in ['distance', 'chainage', 'station']):
                sequence_name = "distance"

        # Get entities
        if structure['entity_column']:
            entities = [str(e) for e in df[structure['entity_column']].unique().tolist()]
        else:
            entities = ["default"]

        # Build global constants
        global_constants = header_constants.copy()

        # Add global constants from constant columns with single value
        for col in structure['constant_columns']:
            if df[col].nunique() == 1:
                unit = detect_unit(col)
                key = strip_unit_suffix(col).lower()
                value = df[col].iloc[0]
                # Convert numpy types to Python types
                if hasattr(value, 'item'):
                    value = value.item()
                if unit:
                    global_constants[f"{key}_{unit}"] = value
                else:
                    global_constants[key] = value

        # Build per-entity constants
        per_entity_constants: Dict[str, Dict[str, Any]] = {}
        if structure['entity_column']:
            for col in structure['constant_columns']:
                nunique_per_entity = df.groupby(structure['entity_column'])[col].nunique()
                if (nunique_per_entity == 1).all() and df[col].nunique() > 1:
                    unit = detect_unit(col)
                    key = strip_unit_suffix(col).lower()

                    for entity in entities:
                        if entity not in per_entity_constants:
                            per_entity_constants[entity] = {}

                        value = df[df[structure['entity_column']] == entity][col].iloc[0]
                        if hasattr(value, 'item'):
                            value = value.item()
                        if unit:
                            per_entity_constants[entity][f"{key}_{unit}"] = value
                        else:
                            per_entity_constants[entity][key] = value

        # Build signals list
        signals = []
        for col in structure['signal_columns']:
            unit = detect_unit(col)
            signal_id = strip_unit_suffix(col)
            signals.append(SignalInfo(
                column=col,
                signal_id=signal_id,
                unit=unit,
            ))

        # Create config
        self._config = PrismConfig(
            source_file=str(source) if not isinstance(source, pd.DataFrame) else "<DataFrame>",
            created_at=datetime.now().isoformat(),
            orthon_version="0.1.0",
            discipline=self.discipline,
            sequence_column=structure['sequence_column'],
            sequence_unit=structure['sequence_unit'],
            sequence_name=sequence_name,
            entity_column=structure['entity_column'],
            entities=entities,
            global_constants=global_constants,
            per_entity_constants=per_entity_constants,
            signals=signals,
            row_count=len(df),
            observation_count=len(df) * len(structure['signal_columns']),
        )

    def _to_long_format(self, df: pd.DataFrame, structure: dict) -> pd.DataFrame:
        """Transform wide format to observations.parquet schema"""

        records = []

        # Get sequence values
        if structure['sequence_column']:
            seq_col = df[structure['sequence_column']]

            # Handle datetime columns
            if pd.api.types.is_datetime64_any_dtype(seq_col):
                # Convert to seconds since start
                sequence_values = (seq_col - seq_col.min()).dt.total_seconds().values
                self._config.sequence_unit = 's'
                self._config.sequence_name = 'time'
            elif seq_col.dtype == object:
                # Try parsing as datetime
                try:
                    dt_col = pd.to_datetime(seq_col)
                    sequence_values = (dt_col - dt_col.min()).dt.total_seconds().values
                    self._config.sequence_unit = 's'
                    self._config.sequence_name = 'time'
                except Exception:
                    # Try numeric conversion, fallback to row index
                    try:
                        sequence_values = seq_col.astype(float).values
                    except Exception:
                        sequence_values = np.arange(len(df), dtype=float)
            else:
                sequence_values = seq_col.values.astype(float)
        else:
            sequence_values = np.arange(len(df), dtype=float)

        # Get entity values
        if structure['entity_column']:
            entity_values = df[structure['entity_column']].values
        else:
            entity_values = np.full(len(df), "default")

        # Transform each signal
        for sig in self._config.signals:
            col = sig.column
            signal_id = sig.signal_id
            unit = sig.unit

            values = df[col].values

            for i in range(len(df)):
                if pd.notna(values[i]):  # Skip nulls
                    records.append({
                        'entity_id': str(entity_values[i]),
                        'signal_id': signal_id,
                        'I': float(sequence_values[i]),  # CANONICAL: I not index
                        'y': float(values[i]),           # CANONICAL: y not value
                        'unit': unit,
                    })

        observations_df = pd.DataFrame(records)

        # Ensure CANONICAL schema: entity_id, signal_id, I, y, unit
        observations_df = observations_df.astype({
            'entity_id': 'string',
            'signal_id': 'string',
            'I': 'float64',      # CANONICAL: I not index
            'y': 'float64',      # CANONICAL: y not value
            'unit': 'string',
        })

        return observations_df


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def prepare_for_prism(
    source: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path] = ".",
    discipline: DisciplineType = None,
) -> Tuple[Path, Path]:
    """
    Prepare user data for PRISM analysis.

    Args:
        source: Path to CSV/Parquet/TSV or DataFrame
        output_dir: Where to write outputs
        discipline: Optional discipline for specialized physics engines

    Returns:
        (observations.parquet path, config.json path)

    Example:
        obs_path, config_path = prepare_for_prism("pump_data.csv", "output/")
        # Now run PRISM:
        # prism.analyze(obs_path, config_path)
    """
    transformer = IntakeTransformer(discipline=discipline)
    return transformer.transform(source, output_dir)


def transform_for_prism(
    source: Union[str, Path, pd.DataFrame],
    discipline: DisciplineType = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    Transform user data to PRISM format (in memory, no files).

    Args:
        source: Path to CSV/Parquet/TSV or DataFrame
        discipline: Optional discipline for specialized physics engines

    Returns:
        (observations_df, config_dict)
    """
    transformer = IntakeTransformer(discipline=discipline)
    return transformer.transform_df(source)


# Re-export for convenience
__all__ = [
    'IntakeTransformer',
    'prepare_for_prism',
    'transform_for_prism',
    'PrismConfig',
    'SignalInfo',
    'DISCIPLINES',
    'detect_unit',
    'strip_unit_suffix',
]
