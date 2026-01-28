"""
ORTHON Config Generator
=======================

Generates default PRISM configuration based on detected units and signal types.
Uses the engine gating system from prism_config.yaml.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
import json


# =============================================================================
# UNIT → CATEGORY MAPPING (from prism_config.yaml)
# =============================================================================

UNIT_TO_CATEGORY: Dict[str, str] = {
    # Vibration
    'g': 'vibration', 'm/s²': 'vibration', 'mm/s': 'vibration',
    'in/s': 'vibration', 'ips': 'vibration', 'mil': 'vibration',
    'μm': 'vibration', 'um': 'vibration',

    # Rotation
    'RPM': 'rotation', 'rpm': 'rotation', 'rad/s': 'rotation',

    # Force
    'N': 'force', 'kN': 'force', 'Nm': 'force', 'lbf': 'force',
    'MPa': 'force', 'GPa': 'force',

    # Electrical
    'A': 'electrical_current', 'mA': 'electrical_current',
    'μA': 'electrical_current', 'uA': 'electrical_current',
    'V': 'electrical_voltage', 'mV': 'electrical_voltage', 'kV': 'electrical_voltage',
    'W': 'electrical_power', 'kW': 'electrical_power', 'MW': 'electrical_power',
    'VA': 'electrical_power', 'VAR': 'electrical_power', 'PF': 'electrical_power',
    'Ω': 'electrical_impedance', 'ohm': 'electrical_impedance',

    # Flow
    'm³/s': 'flow_volume', 'L/s': 'flow_volume', 'L/min': 'flow_volume',
    'GPM': 'flow_volume', 'gpm': 'flow_volume', 'CFM': 'flow_volume',
    'kg/s': 'flow_mass', 'kg/hr': 'flow_mass', 'lb/hr': 'flow_mass', 'g/s': 'flow_mass',

    # Velocity
    'm/s': 'velocity', 'ft/s': 'velocity', 'km/h': 'velocity', 'mph': 'velocity',

    # Pressure
    'Pa': 'pressure', 'kPa': 'pressure', 'MPa': 'pressure', 'bar': 'pressure',
    'psi': 'pressure', 'PSI': 'pressure', 'atm': 'pressure',
    'mmHg': 'pressure', 'inH2O': 'pressure',

    # Temperature
    '°C': 'temperature', 'C': 'temperature', '°F': 'temperature',
    'F': 'temperature', 'K': 'temperature', 'degC': 'temperature', 'degF': 'temperature',

    # Heat Transfer
    'W/m²': 'heat_transfer', 'W/(m·K)': 'heat_transfer', 'BTU/hr': 'heat_transfer',

    # Chemical
    'mol/L': 'concentration', 'M': 'concentration', 'mmol/L': 'concentration',
    'ppm': 'concentration', 'ppb': 'concentration', 'mg/L': 'concentration',
    'wt%': 'concentration', 'mol%': 'concentration',
    'pH': 'ph',

    # Thermodynamic
    'J/mol': 'molar_properties', 'kJ/mol': 'molar_properties',
    'J/kg': 'specific_properties', 'kJ/kg': 'specific_properties',

    # Control
    '%': 'control', 'percent': 'control',

    # Dimensionless
    'dimensionless': 'dimensionless', 'ratio': 'dimensionless',
    'unitless': 'dimensionless', 'count': 'dimensionless',
}


# =============================================================================
# ENGINE GATING (from prism_config.yaml)
# =============================================================================

# Engines that always run (no unit requirements)
CORE_ENGINES = [
    'hurst', 'entropy', 'garch', 'lyapunov', 'lof', 'clustering', 'pca',
    'granger', 'transfer_entropy', 'cointegration', 'dmd', 'fft', 'wavelet',
    'hilbert', 'rqa', 'mst', 'mutual_info', 'copula', 'dtw', 'embedding',
    'attractor', 'basin', 'acf_decay', 'spectral_slope', 'entropy_rate',
    'convex_hull', 'divergence', 'umap', 'modes',
]

# Domain engines with their required categories
DOMAIN_ENGINES: Dict[str, List[str]] = {
    # Mechanical
    'bearing_fault': ['vibration'],
    'gear_mesh': ['vibration', 'rotation'],
    'modal_analysis': ['vibration'],
    'rotor_dynamics': ['vibration', 'rotation'],
    'fatigue': ['force'],

    # Electrical
    'motor_signature': ['electrical_current'],
    'power_quality': ['electrical_voltage'],
    'impedance': ['electrical_voltage', 'electrical_current'],

    # Fluids
    'navier_stokes': ['velocity'],
    'turbulence_spectrum': ['velocity'],
    'reynolds_stress': ['velocity'],
    'vorticity': ['velocity'],
    'two_phase_flow': ['flow_volume'],

    # Thermal
    'heat_equation': ['temperature'],
    'convection': ['temperature', 'velocity'],
    'radiation': ['temperature'],
    'stefan_problem': ['temperature'],
    'heat_exchanger': ['temperature', 'flow_mass'],

    # Thermo
    'phase_equilibria': ['temperature', 'pressure'],
    'equation_of_state': ['temperature', 'pressure'],
    'fugacity': ['temperature', 'pressure'],
    'exergy': ['temperature', 'pressure'],
    'activity_models': ['concentration', 'temperature'],

    # Chemical
    'reaction_kinetics': ['concentration'],
    'electrochemistry': ['electrical_voltage', 'concentration'],
    'separations': ['concentration'],

    # Control
    'transfer_function': ['control'],
    'kalman': ['control'],
    'stability': ['control'],

    # Process
    'reactor_ode': ['concentration', 'temperature'],
    'distillation': ['concentration', 'temperature', 'pressure'],
    'crystallization': ['concentration', 'temperature'],
}


@dataclass
class SignalConfig:
    """Configuration for a single signal."""
    name: str
    unit: str
    category: str
    description: Optional[str] = None


@dataclass
class PrismJobConfig:
    """Complete PRISM job configuration."""
    # Dataset info
    dataset_name: str
    entity_count: int
    signal_count: int
    observation_count: int

    # Signals
    signals: List[SignalConfig] = field(default_factory=list)

    # Detected categories
    categories: Set[str] = field(default_factory=set)

    # Engines to run
    core_engines: List[str] = field(default_factory=list)
    domain_engines: List[str] = field(default_factory=list)

    # Index info
    index_type: str = 'integer_sequence'
    index_unit: Optional[str] = None
    sampling_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = asdict(self)
        d['categories'] = list(d['categories'])
        return d

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def detect_category(unit: str) -> str:
    """Map a unit to its category."""
    if unit is None:
        return 'unknown'
    return UNIT_TO_CATEGORY.get(unit, 'unknown')


def get_enabled_engines(categories: Set[str]) -> Dict[str, List[str]]:
    """
    Determine which engines to run based on detected categories.

    Returns:
        Dict with 'core' and 'domain' engine lists
    """
    # Core engines always run
    core = CORE_ENGINES.copy()

    # Check domain engines
    domain = []
    for engine, required_cats in DOMAIN_ENGINES.items():
        if all(cat in categories for cat in required_cats):
            domain.append(engine)

    return {'core': core, 'domain': domain}


def generate_config(
    observations_path: Path,
    dataset_name: Optional[str] = None,
) -> PrismJobConfig:
    """
    Generate PRISM configuration from observations.parquet.

    Args:
        observations_path: Path to observations.parquet
        dataset_name: Name for the dataset (defaults to parent directory name)

    Returns:
        PrismJobConfig with all settings
    """
    import pandas as pd

    observations_path = Path(observations_path)
    df = pd.read_parquet(observations_path)

    # Validate canonical schema
    required = {'entity_id', 'signal_id', 'I', 'y', 'unit'}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"Not canonical schema. Has: {df.columns.tolist()}, needs: {required}")

    # Dataset info
    if dataset_name is None:
        dataset_name = observations_path.parent.name

    entity_count = df['entity_id'].nunique()
    signal_count = df['signal_id'].nunique()
    observation_count = len(df)

    # Build signal configs
    signals = []
    categories = set()

    for signal_id in df['signal_id'].unique():
        signal_df = df[df['signal_id'] == signal_id]
        unit = signal_df['unit'].iloc[0]
        category = detect_category(unit)

        signals.append(SignalConfig(
            name=signal_id,
            unit=unit if unit else 'unknown',
            category=category,
        ))

        if category != 'unknown':
            categories.add(category)

    # Determine engines
    engines = get_enabled_engines(categories)

    return PrismJobConfig(
        dataset_name=dataset_name,
        entity_count=entity_count,
        signal_count=signal_count,
        observation_count=observation_count,
        signals=signals,
        categories=categories,
        core_engines=engines['core'],
        domain_engines=engines['domain'],
    )


def save_config(config: PrismJobConfig, output_path: Path) -> Path:
    """Save configuration to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(config.to_json())

    return output_path


def generate_and_save_config(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    dataset_name: Optional[str] = None,
) -> Path:
    """
    Generate config and save to config.json in output directory.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory (defaults to same as observations)
        dataset_name: Dataset name

    Returns:
        Path to config.json
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    config = generate_config(observations_path, dataset_name)
    config_path = output_dir / 'config.json'

    return save_config(config, config_path)
