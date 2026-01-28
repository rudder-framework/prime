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


# =============================================================================
# MANIFEST GENERATION (Orthon as Brain)
# =============================================================================

from ..shared.engine_registry import (
    Granularity,
    EngineSpec,
    ENGINE_SPECS,
    UNIT_TO_CATEGORY as REGISTRY_UNIT_TO_CATEGORY,
    get_engines_for_categories,
    get_category_for_unit,
)
from .manifest_schema import (
    PrismManifest,
    EngineManifestEntry,
    ManifestMetadata,
    WindowManifest,
)


@dataclass
class DataAnalysis:
    """Results from analyzing observations.parquet."""

    # Counts
    entity_count: int = 0
    signal_count: int = 0
    observation_count: int = 0

    # Lists
    entities: List[str] = field(default_factory=list)
    signals: List[str] = field(default_factory=list)
    units: List[str] = field(default_factory=list)
    categories: Set[str] = field(default_factory=set)

    # Signal → unit mapping
    signal_units: Dict[str, str] = field(default_factory=dict)
    # Signal → category mapping
    signal_categories: Dict[str, str] = field(default_factory=dict)
    # Category → signals mapping
    category_signals: Dict[str, List[str]] = field(default_factory=dict)
    # Category → units mapping
    category_units: Dict[str, List[str]] = field(default_factory=dict)

    # I (index) statistics
    I_min: Optional[float] = None
    I_max: Optional[float] = None
    I_range: Optional[float] = None
    sampling_rate: Optional[float] = None

    # y (value) statistics
    y_min: Optional[float] = None
    y_max: Optional[float] = None


class DataAnalyzer:
    """
    Analyze observations.parquet to extract metadata for manifest generation.

    Uses Polars for efficient large-file handling.
    """

    def __init__(self, observations_path: Path):
        self.observations_path = Path(observations_path)

    def analyze(self) -> DataAnalysis:
        """
        Analyze the observations file and return DataAnalysis.

        Returns:
            DataAnalysis with all extracted metadata
        """
        import polars as pl

        # Read parquet with Polars
        df = pl.read_parquet(self.observations_path)

        # Validate canonical schema
        required = {'entity_id', 'signal_id', 'I', 'y', 'unit'}
        if not required.issubset(set(df.columns)):
            raise ValueError(
                f"Not canonical schema. Has: {df.columns}, needs: {required}"
            )

        analysis = DataAnalysis()

        # Basic counts
        analysis.entity_count = df['entity_id'].n_unique()
        analysis.signal_count = df['signal_id'].n_unique()
        analysis.observation_count = len(df)

        # Entity and signal lists
        analysis.entities = df['entity_id'].unique().sort().to_list()
        analysis.signals = df['signal_id'].unique().sort().to_list()

        # Unit analysis per signal
        signal_unit_df = (
            df.group_by('signal_id')
            .agg(pl.col('unit').first())
            .sort('signal_id')
        )

        for row in signal_unit_df.iter_rows(named=True):
            signal_id = row['signal_id']
            unit = row['unit']
            analysis.signal_units[signal_id] = unit

            # Get category for unit
            category = get_category_for_unit(unit)
            analysis.signal_categories[signal_id] = category

            if category != 'unknown':
                analysis.categories.add(category)

                # Track signals per category
                if category not in analysis.category_signals:
                    analysis.category_signals[category] = []
                analysis.category_signals[category].append(signal_id)

                # Track units per category
                if category not in analysis.category_units:
                    analysis.category_units[category] = []
                if unit and unit not in analysis.category_units[category]:
                    analysis.category_units[category].append(unit)

        # Unique units
        analysis.units = [u for u in df['unit'].unique().to_list() if u is not None]

        # I statistics
        I_stats = df.select([
            pl.col('I').min().alias('I_min'),
            pl.col('I').max().alias('I_max'),
        ]).row(0, named=True)

        analysis.I_min = I_stats['I_min']
        analysis.I_max = I_stats['I_max']
        if analysis.I_min is not None and analysis.I_max is not None:
            analysis.I_range = analysis.I_max - analysis.I_min

        # Estimate sampling rate from first entity/signal
        if analysis.entities and analysis.signals:
            sample_df = df.filter(
                (pl.col('entity_id') == analysis.entities[0]) &
                (pl.col('signal_id') == analysis.signals[0])
            ).sort('I')

            if len(sample_df) > 1:
                I_vals = sample_df['I'].to_list()
                diffs = [I_vals[i+1] - I_vals[i] for i in range(min(100, len(I_vals)-1))]
                if diffs:
                    median_diff = sorted(diffs)[len(diffs)//2]
                    if median_diff > 0:
                        analysis.sampling_rate = 1.0 / median_diff

        # y statistics
        y_stats = df.select([
            pl.col('y').min().alias('y_min'),
            pl.col('y').max().alias('y_max'),
        ]).row(0, named=True)

        analysis.y_min = y_stats['y_min']
        analysis.y_max = y_stats['y_max']

        return analysis


class ManifestBuilder:
    """
    Build a PrismManifest from DataAnalysis.

    This is where Orthon makes decisions about what engines to run.
    """

    # Granularity → output parquet mapping
    GRANULARITY_TO_OUTPUT = {
        Granularity.SIGNAL: 'vector',
        Granularity.OBSERVATION: 'dynamics',
        Granularity.PAIR_DIRECTIONAL: 'pairs',
        Granularity.PAIR_SYMMETRIC: 'pairs',
        Granularity.OBSERVATION_CROSS_SIGNAL: 'geometry',
    }

    # Granularity → groupby columns
    GRANULARITY_TO_GROUPBY = {
        Granularity.SIGNAL: ['entity_id', 'signal_id'],
        Granularity.OBSERVATION: ['entity_id'],
        Granularity.PAIR_DIRECTIONAL: ['entity_id'],  # pairs enumerated internally
        Granularity.PAIR_SYMMETRIC: ['entity_id'],    # pairs enumerated internally
        Granularity.OBSERVATION_CROSS_SIGNAL: ['entity_id'],
    }

    def __init__(
        self,
        analysis: DataAnalysis,
        input_file: str,
        output_dir: str,
    ):
        self.analysis = analysis
        self.input_file = input_file
        self.output_dir = output_dir

    def build(
        self,
        window_size: int = 100,
        window_stride: int = 50,
        min_samples: int = 50,
        constants: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> PrismManifest:
        """
        Build the complete manifest.

        Args:
            window_size: Window size for analysis
            window_stride: Stride between windows
            min_samples: Minimum samples per window
            constants: Global constants for physics calculations
            callback_url: Optional callback URL for job completion

        Returns:
            PrismManifest ready for PRISM execution
        """
        # Build metadata
        metadata = ManifestMetadata(
            entity_count=self.analysis.entity_count,
            signal_count=self.analysis.signal_count,
            observation_count=self.analysis.observation_count,
            entities=self.analysis.entities,
            signals=self.analysis.signals,
            units_present=self.analysis.units,
            unit_categories=list(self.analysis.categories),
            sampling_rate=self.analysis.sampling_rate,
            I_min=self.analysis.I_min,
            I_max=self.analysis.I_max,
            I_range=self.analysis.I_range,
            y_min=self.analysis.y_min,
            y_max=self.analysis.y_max,
        )

        # Build window config
        window = WindowManifest(
            size=window_size,
            stride=window_stride,
            min_samples=min_samples,
        )

        # Select engines based on detected categories
        selected_engines = get_engines_for_categories(self.analysis.categories)

        # Build engine manifest entries
        engine_entries = []
        for spec in selected_engines:
            entry = self._build_engine_entry(spec)
            engine_entries.append(entry)

        # Sort engines: universal first, then by name
        engine_entries.sort(key=lambda e: (not self._is_universal(e.name), e.name))

        return PrismManifest(
            input_file=self.input_file,
            output_dir=self.output_dir,
            metadata=metadata,
            engines=engine_entries,
            window=window,
            constants=constants or {},
            callback_url=callback_url,
        )

    def _is_universal(self, engine_name: str) -> bool:
        """Check if engine is universal (no category restrictions)."""
        spec = ENGINE_SPECS.get(engine_name)
        return spec.is_universal() if spec else True

    def _build_engine_entry(self, spec: EngineSpec) -> EngineManifestEntry:
        """Build manifest entry for a single engine."""

        # Determine output parquet
        output = self.GRANULARITY_TO_OUTPUT.get(spec.granularity, 'dynamics')

        # Determine groupby
        groupby = self.GRANULARITY_TO_GROUPBY.get(spec.granularity, ['entity_id'])

        # Build filter for category-specific engines
        filter_expr = None
        if not spec.is_universal():
            # Get all units that match the engine's categories
            matching_units = []
            for cat in spec.categories:
                if cat in self.analysis.category_units:
                    matching_units.extend(self.analysis.category_units[cat])

            if matching_units:
                # Remove duplicates while preserving order
                seen = set()
                unique_units = []
                for u in matching_units:
                    if u not in seen:
                        seen.add(u)
                        unique_units.append(u)

                # Build Polars filter expression
                units_list = ', '.join(f'"{u}"' for u in unique_units)
                filter_expr = f'col("unit").is_in([{units_list}])'

        return EngineManifestEntry(
            name=spec.name,
            output=output,
            granularity=spec.granularity.value,
            groupby=groupby,
            orderby=['I'],
            input_columns=spec.input_columns,
            output_columns=spec.output_columns,
            function=spec.function or f"prism.engines.{spec.name}.compute",
            params=spec.params,
            filter=filter_expr,
            min_rows=spec.min_rows,
            enabled=True,
        )


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_manifest(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    window_size: int = 100,
    window_stride: int = 50,
    min_samples: int = 50,
    constants: Optional[Dict[str, Any]] = None,
    callback_url: Optional[str] = None,
) -> PrismManifest:
    """
    Generate a PRISM manifest from observations.parquet.

    This is the main entry point for manifest generation.
    Orthon analyzes the data and decides what engines to run.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory for results (defaults to same as observations)
        window_size: Window size for analysis
        window_stride: Stride between windows
        min_samples: Minimum samples per window
        constants: Global constants for physics calculations
        callback_url: Optional callback URL for job completion

    Returns:
        PrismManifest ready for PRISM execution

    Example:
        manifest = generate_manifest(
            "output/observations.parquet",
            output_dir="output/",
            window_size=100,
            constants={'density_kg_m3': 1000},
        )
        manifest.to_json("output/manifest.json")
        print(manifest.summary())
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    # Analyze data
    analyzer = DataAnalyzer(observations_path)
    analysis = analyzer.analyze()

    # Build manifest
    builder = ManifestBuilder(
        analysis=analysis,
        input_file=str(observations_path),
        output_dir=str(output_dir),
    )

    return builder.build(
        window_size=window_size,
        window_stride=window_stride,
        min_samples=min_samples,
        constants=constants,
        callback_url=callback_url,
    )


def generate_and_save_manifest(
    observations_path: Path,
    output_dir: Optional[Path] = None,
    manifest_name: str = "manifest.json",
    **kwargs,
) -> Path:
    """
    Generate manifest and save to file.

    Args:
        observations_path: Path to observations.parquet
        output_dir: Output directory (defaults to same as observations)
        manifest_name: Name of manifest file (default: manifest.json)
        **kwargs: Additional arguments passed to generate_manifest

    Returns:
        Path to saved manifest.json
    """
    observations_path = Path(observations_path)
    output_dir = Path(output_dir) if output_dir else observations_path.parent

    manifest = generate_manifest(
        observations_path,
        output_dir=output_dir,
        **kwargs,
    )

    manifest_path = output_dir / manifest_name
    manifest.to_json(manifest_path)

    return manifest_path
