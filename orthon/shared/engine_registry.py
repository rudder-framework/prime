"""
ORTHON Engine Registry
======================

Centralized engine metadata with granularity, categories, and specifications.
Orthon uses this to decide what engines to run; PRISM just executes.

Granularity determines output parquet:
- SIGNAL: vector.parquet (entity + signal + window)
- OBSERVATION: dynamics.parquet (entity + window)
- PAIR_DIRECTIONAL: pairs.parquet (entity + signal_i + signal_j + window)
- PAIR_SYMMETRIC: pairs.parquet (entity + pair + window)
- OBSERVATION_CROSS_SIGNAL: geometry.parquet (entity + window)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Set


# =============================================================================
# GRANULARITY ENUM
# =============================================================================

class Granularity(str, Enum):
    """Output granularity determines groupby and output parquet."""

    SIGNAL = "signal"
    # Per-signal metrics (e.g., entropy, hurst)
    # Output: vector.parquet
    # Groupby: [entity_id, signal_id]

    OBSERVATION = "observation"
    # Per-entity metrics across all signals (e.g., PCA, clustering)
    # Output: dynamics.parquet
    # Groupby: [entity_id]

    PAIR_DIRECTIONAL = "pair_directional"
    # Directional pair metrics (e.g., granger, transfer_entropy)
    # Output: pairs.parquet
    # Groupby: [entity_id, signal_i, signal_j]

    PAIR_SYMMETRIC = "pair_symmetric"
    # Symmetric pair metrics (e.g., correlation, mutual_info, dtw)
    # Output: pairs.parquet
    # Groupby: [entity_id] with pair enumeration

    OBSERVATION_CROSS_SIGNAL = "observation_cross_signal"
    # Cross-signal entity metrics (e.g., convex_hull, attractor)
    # Output: geometry.parquet
    # Groupby: [entity_id]


# =============================================================================
# ENGINE SPECIFICATION
# =============================================================================

@dataclass
class EngineSpec:
    """Full specification for a PRISM engine."""

    name: str
    granularity: Granularity
    categories: List[str] = field(default_factory=list)
    # Empty = universal (runs on all data)
    # Non-empty = only runs when ANY listed category is present

    input_columns: List[str] = field(default_factory=lambda: ["I", "y"])
    output_columns: List[str] = field(default_factory=list)
    function: str = ""  # PRISM function path
    params: Dict[str, Any] = field(default_factory=dict)
    min_rows: int = 10
    description: str = ""

    def is_universal(self) -> bool:
        """Check if engine runs on all data (no category restrictions)."""
        return len(self.categories) == 0

    def matches_categories(self, detected_categories: Set[str]) -> bool:
        """Check if engine should run given detected categories."""
        if self.is_universal():
            return True
        return bool(set(self.categories) & detected_categories)


# =============================================================================
# UNIT → CATEGORY MAPPING (consolidated)
# =============================================================================

UNIT_TO_CATEGORY: Dict[str, str] = {
    # -------------------------------------------------------------------------
    # Vibration
    # -------------------------------------------------------------------------
    'g': 'vibration',
    'm/s²': 'vibration',
    'mm/s': 'vibration',
    'in/s': 'vibration',
    'ips': 'vibration',
    'mil': 'vibration',
    'μm': 'vibration',
    'um': 'vibration',

    # -------------------------------------------------------------------------
    # Rotation
    # -------------------------------------------------------------------------
    'RPM': 'rotation',
    'rpm': 'rotation',
    'rad/s': 'rotation',
    'Hz': 'rotation',
    'hz': 'rotation',

    # -------------------------------------------------------------------------
    # Force / Stress
    # -------------------------------------------------------------------------
    'N': 'force',
    'kN': 'force',
    'Nm': 'torque',
    'lbf': 'force',
    'MPa': 'stress',
    'GPa': 'stress',
    'ksi': 'stress',

    # -------------------------------------------------------------------------
    # Electrical - Current
    # -------------------------------------------------------------------------
    'A': 'electrical_current',
    'mA': 'electrical_current',
    'μA': 'electrical_current',
    'uA': 'electrical_current',

    # -------------------------------------------------------------------------
    # Electrical - Voltage
    # -------------------------------------------------------------------------
    'V': 'electrical_voltage',
    'mV': 'electrical_voltage',
    'kV': 'electrical_voltage',

    # -------------------------------------------------------------------------
    # Electrical - Power
    # -------------------------------------------------------------------------
    'W': 'electrical_power',
    'kW': 'electrical_power',
    'MW': 'electrical_power',
    'VA': 'electrical_power',
    'VAR': 'electrical_power',
    'PF': 'electrical_power',

    # -------------------------------------------------------------------------
    # Electrical - Impedance
    # -------------------------------------------------------------------------
    'Ω': 'electrical_impedance',
    'ohm': 'electrical_impedance',
    'Ω·m': 'electrical_impedance',

    # -------------------------------------------------------------------------
    # Flow - Volume
    # -------------------------------------------------------------------------
    'm³/s': 'flow_volume',
    'L/s': 'flow_volume',
    'L/min': 'flow_volume',
    'GPM': 'flow_volume',
    'gpm': 'flow_volume',
    'CFM': 'flow_volume',
    'cfm': 'flow_volume',

    # -------------------------------------------------------------------------
    # Flow - Mass
    # -------------------------------------------------------------------------
    'kg/s': 'flow_mass',
    'kg/hr': 'flow_mass',
    'kg/h': 'flow_mass',
    'lb/hr': 'flow_mass',
    'lb/h': 'flow_mass',
    'g/s': 'flow_mass',

    # -------------------------------------------------------------------------
    # Velocity
    # -------------------------------------------------------------------------
    'm/s': 'velocity',
    'ft/s': 'velocity',
    'km/h': 'velocity',
    'mph': 'velocity',
    'fps': 'velocity',
    'mps': 'velocity',

    # -------------------------------------------------------------------------
    # Pressure
    # -------------------------------------------------------------------------
    'Pa': 'pressure',
    'kPa': 'pressure',
    'MPa': 'pressure',
    'bar': 'pressure',
    'psi': 'pressure',
    'PSI': 'pressure',
    'psia': 'pressure',
    'psig': 'pressure',
    'atm': 'pressure',
    'mmHg': 'pressure',
    'inH2O': 'pressure',

    # -------------------------------------------------------------------------
    # Temperature
    # -------------------------------------------------------------------------
    '°C': 'temperature',
    'C': 'temperature',
    '°F': 'temperature',
    'F': 'temperature',
    'K': 'temperature',
    'degC': 'temperature',
    'degF': 'temperature',
    'degK': 'temperature',
    '°R': 'temperature',

    # -------------------------------------------------------------------------
    # Heat Transfer
    # -------------------------------------------------------------------------
    'W/m²': 'heat_transfer',
    'W/(m·K)': 'heat_transfer',
    'W/m²K': 'heat_transfer',
    'BTU/hr': 'heat_transfer',

    # -------------------------------------------------------------------------
    # Concentration
    # -------------------------------------------------------------------------
    'mol/L': 'concentration',
    'M': 'concentration',
    'mmol/L': 'concentration',
    'ppm': 'concentration',
    'ppb': 'concentration',
    'mg/L': 'concentration',
    'wt%': 'concentration',
    'mol%': 'concentration',

    # -------------------------------------------------------------------------
    # pH
    # -------------------------------------------------------------------------
    'pH': 'ph',

    # -------------------------------------------------------------------------
    # Thermodynamic - Molar
    # -------------------------------------------------------------------------
    'J/mol': 'molar_properties',
    'kJ/mol': 'molar_properties',
    'J/(mol·K)': 'molar_properties',

    # -------------------------------------------------------------------------
    # Thermodynamic - Specific
    # -------------------------------------------------------------------------
    'J/kg': 'specific_properties',
    'kJ/kg': 'specific_properties',
    'J/(kg·K)': 'specific_properties',

    # -------------------------------------------------------------------------
    # Control
    # -------------------------------------------------------------------------
    '%': 'control',
    'percent': 'control',

    # -------------------------------------------------------------------------
    # Length
    # -------------------------------------------------------------------------
    'm': 'length',
    'mm': 'length',
    'cm': 'length',
    'ft': 'length',
    'in': 'length',

    # -------------------------------------------------------------------------
    # Mass
    # -------------------------------------------------------------------------
    'kg': 'mass',
    'lb': 'mass',
    'mg': 'mass',
    # Note: 'g' is mapped to 'vibration' (accelerometer units), not mass

    # -------------------------------------------------------------------------
    # Density
    # -------------------------------------------------------------------------
    'kg/m³': 'density',
    'g/cm³': 'density',
    'ppg': 'density',

    # -------------------------------------------------------------------------
    # Viscosity
    # -------------------------------------------------------------------------
    'cP': 'viscosity',
    'Pa·s': 'viscosity',

    # -------------------------------------------------------------------------
    # Energy
    # -------------------------------------------------------------------------
    'J': 'energy',
    'kJ': 'energy',
    'BTU': 'energy',

    # -------------------------------------------------------------------------
    # Dimensionless
    # -------------------------------------------------------------------------
    'dimensionless': 'dimensionless',
    'ratio': 'dimensionless',
    'unitless': 'dimensionless',
    'count': 'dimensionless',
    '': 'dimensionless',
}


# =============================================================================
# CATEGORY → ENGINES MAPPING
# =============================================================================

# Categories and which domain engines they enable
CATEGORY_TO_ENGINES: Dict[str, List[str]] = {
    'vibration': [
        'bearing_fault', 'gear_mesh', 'modal_analysis', 'rotor_dynamics',
    ],
    'rotation': [
        'gear_mesh', 'rotor_dynamics',
    ],
    'force': [
        'fatigue',
    ],
    'stress': [
        'fatigue', 'stress_concentration',
    ],
    'torque': [
        'fatigue',
    ],
    'electrical_current': [
        'motor_signature', 'impedance',
    ],
    'electrical_voltage': [
        'power_quality', 'impedance',
    ],
    'electrical_power': [
        'power_quality',
    ],
    'velocity': [
        'navier_stokes', 'turbulence_spectrum', 'reynolds_stress', 'vorticity',
    ],
    'flow_volume': [
        'two_phase_flow',
    ],
    'flow_mass': [
        'heat_exchanger',
    ],
    'temperature': [
        'heat_equation', 'convection', 'radiation', 'stefan_problem',
        'heat_exchanger', 'phase_equilibria', 'equation_of_state',
        'activity_models', 'reactor_ode', 'distillation', 'crystallization',
    ],
    'pressure': [
        'phase_equilibria', 'equation_of_state', 'fugacity', 'exergy',
        'distillation',
    ],
    'concentration': [
        'reaction_kinetics', 'electrochemistry', 'separations',
        'activity_models', 'reactor_ode', 'distillation', 'crystallization',
    ],
    'control': [
        'transfer_function', 'kalman', 'stability',
    ],
}


# =============================================================================
# ENGINE SPECIFICATIONS
# =============================================================================

ENGINE_SPECS: Dict[str, EngineSpec] = {
    # =========================================================================
    # UNIVERSAL ENGINES (run on all data)
    # =========================================================================

    # --- Signal-level metrics ---
    'hurst': EngineSpec(
        name='hurst',
        granularity=Granularity.SIGNAL,
        description='Hurst exponent for long-range dependence',
        output_columns=['hurst_exponent', 'hurst_confidence'],
        min_rows=50,
    ),
    'entropy': EngineSpec(
        name='entropy',
        granularity=Granularity.SIGNAL,
        description='Sample entropy and permutation entropy',
        output_columns=['sample_entropy', 'permutation_entropy'],
        min_rows=30,
    ),
    'garch': EngineSpec(
        name='garch',
        granularity=Granularity.SIGNAL,
        description='GARCH volatility modeling',
        output_columns=['garch_omega', 'garch_alpha', 'garch_beta', 'volatility'],
        min_rows=100,
    ),
    'lyapunov': EngineSpec(
        name='lyapunov',
        granularity=Granularity.SIGNAL,
        description='Largest Lyapunov exponent',
        output_columns=['lyapunov_exponent'],
        min_rows=100,
    ),
    'fft': EngineSpec(
        name='fft',
        granularity=Granularity.SIGNAL,
        description='Fast Fourier Transform spectrum',
        output_columns=['dominant_freq', 'spectral_centroid', 'spectral_bandwidth'],
        min_rows=32,
    ),
    'wavelet': EngineSpec(
        name='wavelet',
        granularity=Granularity.SIGNAL,
        description='Wavelet decomposition',
        output_columns=['wavelet_energy', 'wavelet_entropy'],
        min_rows=32,
    ),
    'hilbert': EngineSpec(
        name='hilbert',
        granularity=Granularity.SIGNAL,
        description='Hilbert transform for instantaneous frequency',
        output_columns=['inst_freq_mean', 'inst_freq_std', 'inst_amp_mean'],
        min_rows=20,
    ),
    'rqa': EngineSpec(
        name='rqa',
        granularity=Granularity.SIGNAL,
        description='Recurrence quantification analysis',
        output_columns=['recurrence_rate', 'determinism', 'laminarity', 'trapping_time'],
        min_rows=50,
    ),
    'acf_decay': EngineSpec(
        name='acf_decay',
        granularity=Granularity.SIGNAL,
        description='Autocorrelation decay rate',
        output_columns=['acf_decay_rate', 'acf_half_life'],
        min_rows=30,
    ),
    'spectral_slope': EngineSpec(
        name='spectral_slope',
        granularity=Granularity.SIGNAL,
        description='Power law exponent of spectrum',
        output_columns=['spectral_slope', 'spectral_slope_r2'],
        min_rows=32,
    ),
    'entropy_rate': EngineSpec(
        name='entropy_rate',
        granularity=Granularity.SIGNAL,
        description='Rate of entropy production',
        output_columns=['entropy_rate'],
        min_rows=50,
    ),
    'modes': EngineSpec(
        name='modes',
        granularity=Granularity.SIGNAL,
        description='Empirical mode decomposition',
        output_columns=['n_imfs', 'imf_energies'],
        min_rows=50,
    ),

    # --- Observation-level metrics ---
    'lof': EngineSpec(
        name='lof',
        granularity=Granularity.OBSERVATION,
        description='Local outlier factor',
        output_columns=['lof_score', 'lof_neighbors'],
        min_rows=20,
    ),
    'clustering': EngineSpec(
        name='clustering',
        granularity=Granularity.OBSERVATION,
        description='Clustering analysis (DBSCAN, KMeans)',
        output_columns=['cluster_id', 'cluster_distance', 'silhouette'],
        min_rows=20,
    ),
    'pca': EngineSpec(
        name='pca',
        granularity=Granularity.OBSERVATION,
        description='Principal component analysis',
        output_columns=['pc1', 'pc2', 'pc3', 'explained_variance'],
        min_rows=10,
    ),
    'dmd': EngineSpec(
        name='dmd',
        granularity=Granularity.OBSERVATION,
        description='Dynamic mode decomposition',
        output_columns=['dmd_modes', 'dmd_eigenvalues', 'dmd_amplitudes'],
        min_rows=50,
    ),
    'umap': EngineSpec(
        name='umap',
        granularity=Granularity.OBSERVATION,
        description='UMAP dimensionality reduction',
        output_columns=['umap_1', 'umap_2'],
        min_rows=15,
    ),
    'embedding': EngineSpec(
        name='embedding',
        granularity=Granularity.OBSERVATION,
        description='Takens embedding analysis',
        output_columns=['embedding_dim', 'embedding_delay', 'embedding_quality'],
        min_rows=50,
    ),

    # --- Cross-signal observation metrics ---
    'attractor': EngineSpec(
        name='attractor',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Strange attractor reconstruction',
        output_columns=['attractor_dim', 'correlation_dim', 'lyapunov_spectrum'],
        min_rows=100,
    ),
    'basin': EngineSpec(
        name='basin',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Basin of attraction analysis',
        output_columns=['n_basins', 'basin_volumes', 'basin_stability'],
        min_rows=100,
    ),
    'convex_hull': EngineSpec(
        name='convex_hull',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='State space convex hull',
        output_columns=['hull_volume', 'hull_area', 'hull_vertices'],
        min_rows=10,
    ),
    'divergence': EngineSpec(
        name='divergence',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Trajectory divergence from baseline',
        output_columns=['divergence_rate', 'max_divergence'],
        min_rows=20,
    ),

    # --- Pair metrics (directional) ---
    'granger': EngineSpec(
        name='granger',
        granularity=Granularity.PAIR_DIRECTIONAL,
        description='Granger causality',
        output_columns=['granger_f', 'granger_p', 'granger_lag'],
        min_rows=50,
    ),
    'transfer_entropy': EngineSpec(
        name='transfer_entropy',
        granularity=Granularity.PAIR_DIRECTIONAL,
        description='Transfer entropy (information flow)',
        output_columns=['transfer_entropy', 'effective_te'],
        min_rows=50,
    ),

    # --- Pair metrics (symmetric) ---
    'cointegration': EngineSpec(
        name='cointegration',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Cointegration test',
        output_columns=['coint_stat', 'coint_pvalue', 'coint_vector'],
        min_rows=50,
    ),
    'mutual_info': EngineSpec(
        name='mutual_info',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Mutual information',
        output_columns=['mutual_info', 'normalized_mi'],
        min_rows=30,
    ),
    'copula': EngineSpec(
        name='copula',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Copula dependence analysis',
        output_columns=['copula_type', 'copula_param', 'tail_dependence'],
        min_rows=50,
    ),
    'dtw': EngineSpec(
        name='dtw',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Dynamic time warping distance',
        output_columns=['dtw_distance', 'dtw_path_length'],
        min_rows=20,
    ),
    'mst': EngineSpec(
        name='mst',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Minimum spanning tree of correlations',
        output_columns=['mst_edges', 'mst_centrality'],
        min_rows=10,
    ),

    # =========================================================================
    # DOMAIN ENGINES (category-gated)
    # =========================================================================

    # --- Vibration ---
    'bearing_fault': EngineSpec(
        name='bearing_fault',
        granularity=Granularity.SIGNAL,
        categories=['vibration'],
        description='Bearing fault detection (BPFO, BPFI, BSF, FTF)',
        output_columns=['bpfo', 'bpfi', 'bsf', 'ftf', 'fault_indicator'],
        min_rows=256,
    ),
    'gear_mesh': EngineSpec(
        name='gear_mesh',
        granularity=Granularity.SIGNAL,
        categories=['vibration', 'rotation'],
        description='Gear mesh frequency analysis',
        output_columns=['gmf', 'gmf_harmonics', 'sideband_energy'],
        min_rows=256,
    ),
    'modal_analysis': EngineSpec(
        name='modal_analysis',
        granularity=Granularity.SIGNAL,
        categories=['vibration'],
        description='Modal frequencies and damping',
        output_columns=['natural_freqs', 'damping_ratios', 'mode_shapes'],
        min_rows=256,
    ),
    'rotor_dynamics': EngineSpec(
        name='rotor_dynamics',
        granularity=Granularity.SIGNAL,
        categories=['vibration', 'rotation'],
        description='Rotor dynamics analysis',
        output_columns=['orbit_shape', 'critical_speeds', 'unbalance'],
        min_rows=256,
    ),

    # --- Force/Stress ---
    'fatigue': EngineSpec(
        name='fatigue',
        granularity=Granularity.SIGNAL,
        categories=['force', 'stress', 'torque'],
        description='Fatigue life estimation',
        output_columns=['rainflow_counts', 'damage_accumulation', 'remaining_life'],
        min_rows=100,
    ),

    # --- Electrical ---
    'motor_signature': EngineSpec(
        name='motor_signature',
        granularity=Granularity.SIGNAL,
        categories=['electrical_current'],
        description='Motor current signature analysis (MCSA)',
        output_columns=['rotor_bar_freq', 'eccentricity', 'bearing_freq'],
        min_rows=256,
    ),
    'power_quality': EngineSpec(
        name='power_quality',
        granularity=Granularity.SIGNAL,
        categories=['electrical_voltage', 'electrical_power'],
        description='Power quality analysis (THD, harmonics)',
        output_columns=['thd', 'harmonics', 'power_factor', 'crest_factor'],
        min_rows=128,
    ),
    'impedance': EngineSpec(
        name='impedance',
        granularity=Granularity.PAIR_SYMMETRIC,
        categories=['electrical_voltage', 'electrical_current'],
        description='Impedance spectroscopy',
        output_columns=['impedance_real', 'impedance_imag', 'phase_angle'],
        min_rows=50,
    ),

    # --- Fluids ---
    'navier_stokes': EngineSpec(
        name='navier_stokes',
        granularity=Granularity.OBSERVATION,
        categories=['velocity'],
        description='Navier-Stokes residual analysis',
        output_columns=['ns_residual', 'reynolds_number'],
        min_rows=50,
    ),
    'turbulence_spectrum': EngineSpec(
        name='turbulence_spectrum',
        granularity=Granularity.SIGNAL,
        categories=['velocity'],
        description='Turbulence energy spectrum (Kolmogorov)',
        output_columns=['kolmogorov_slope', 'integral_scale', 'taylor_scale'],
        min_rows=128,
    ),
    'reynolds_stress': EngineSpec(
        name='reynolds_stress',
        granularity=Granularity.OBSERVATION,
        categories=['velocity'],
        description='Reynolds stress tensor',
        output_columns=['reynolds_stress_tensor', 'tke', 'anisotropy'],
        min_rows=50,
    ),
    'vorticity': EngineSpec(
        name='vorticity',
        granularity=Granularity.OBSERVATION,
        categories=['velocity'],
        description='Vorticity and Q-criterion',
        output_columns=['vorticity_mag', 'q_criterion', 'lambda2'],
        min_rows=50,
    ),
    'two_phase_flow': EngineSpec(
        name='two_phase_flow',
        granularity=Granularity.SIGNAL,
        categories=['flow_volume'],
        description='Two-phase flow pattern detection',
        output_columns=['flow_regime', 'void_fraction', 'slip_ratio'],
        min_rows=100,
    ),

    # --- Thermal ---
    'heat_equation': EngineSpec(
        name='heat_equation',
        granularity=Granularity.SIGNAL,
        categories=['temperature'],
        description='Heat diffusion analysis',
        output_columns=['thermal_diffusivity', 'heat_flux'],
        min_rows=50,
    ),
    'convection': EngineSpec(
        name='convection',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'velocity'],
        description='Convective heat transfer',
        output_columns=['nusselt', 'heat_transfer_coeff'],
        min_rows=50,
    ),
    'radiation': EngineSpec(
        name='radiation',
        granularity=Granularity.SIGNAL,
        categories=['temperature'],
        description='Radiative heat transfer',
        output_columns=['emissivity_eff', 'radiative_flux'],
        min_rows=20,
    ),
    'stefan_problem': EngineSpec(
        name='stefan_problem',
        granularity=Granularity.SIGNAL,
        categories=['temperature'],
        description='Phase change (Stefan problem)',
        output_columns=['interface_position', 'solidification_rate'],
        min_rows=50,
    ),
    'heat_exchanger': EngineSpec(
        name='heat_exchanger',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'flow_mass'],
        description='Heat exchanger performance',
        output_columns=['effectiveness', 'ntu', 'lmtd', 'ua'],
        min_rows=20,
    ),

    # --- Thermodynamics ---
    'phase_equilibria': EngineSpec(
        name='phase_equilibria',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'pressure'],
        description='Phase equilibria calculations',
        output_columns=['phase_fraction', 'bubble_point', 'dew_point'],
        min_rows=10,
    ),
    'equation_of_state': EngineSpec(
        name='equation_of_state',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'pressure'],
        description='Equation of state analysis',
        output_columns=['compressibility', 'fugacity_coeff', 'z_factor'],
        min_rows=10,
    ),
    'fugacity': EngineSpec(
        name='fugacity',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'pressure'],
        description='Fugacity calculations',
        output_columns=['fugacity', 'fugacity_coeff', 'activity'],
        min_rows=10,
    ),
    'exergy': EngineSpec(
        name='exergy',
        granularity=Granularity.OBSERVATION,
        categories=['temperature', 'pressure'],
        description='Exergy analysis',
        output_columns=['exergy', 'exergy_destruction', 'exergy_efficiency'],
        min_rows=10,
    ),
    'activity_models': EngineSpec(
        name='activity_models',
        granularity=Granularity.OBSERVATION,
        categories=['concentration', 'temperature'],
        description='Activity coefficient models',
        output_columns=['activity_coeff', 'excess_gibbs'],
        min_rows=10,
    ),

    # --- Chemical ---
    'reaction_kinetics': EngineSpec(
        name='reaction_kinetics',
        granularity=Granularity.SIGNAL,
        categories=['concentration'],
        description='Reaction rate analysis',
        output_columns=['rate_constant', 'reaction_order', 'activation_energy'],
        min_rows=20,
    ),
    'electrochemistry': EngineSpec(
        name='electrochemistry',
        granularity=Granularity.OBSERVATION,
        categories=['electrical_voltage', 'concentration'],
        description='Electrochemical analysis',
        output_columns=['nernst_potential', 'overpotential', 'exchange_current'],
        min_rows=20,
    ),
    'separations': EngineSpec(
        name='separations',
        granularity=Granularity.OBSERVATION,
        categories=['concentration'],
        description='Separation process analysis',
        output_columns=['separation_factor', 'recovery', 'purity'],
        min_rows=20,
    ),

    # --- Process ---
    'reactor_ode': EngineSpec(
        name='reactor_ode',
        granularity=Granularity.OBSERVATION,
        categories=['concentration', 'temperature'],
        description='Reactor ODE system analysis',
        output_columns=['conversion', 'selectivity', 'yield'],
        min_rows=20,
    ),
    'distillation': EngineSpec(
        name='distillation',
        granularity=Granularity.OBSERVATION,
        categories=['concentration', 'temperature', 'pressure'],
        description='Distillation column analysis',
        output_columns=['n_stages', 'reflux_ratio', 'reboil_ratio'],
        min_rows=20,
    ),
    'crystallization': EngineSpec(
        name='crystallization',
        granularity=Granularity.OBSERVATION,
        categories=['concentration', 'temperature'],
        description='Crystallization process analysis',
        output_columns=['supersaturation', 'nucleation_rate', 'growth_rate'],
        min_rows=20,
    ),

    # --- Control ---
    'transfer_function': EngineSpec(
        name='transfer_function',
        granularity=Granularity.PAIR_DIRECTIONAL,
        categories=['control'],
        description='Transfer function identification',
        output_columns=['gain', 'time_constant', 'dead_time', 'poles', 'zeros'],
        min_rows=50,
    ),
    'kalman': EngineSpec(
        name='kalman',
        granularity=Granularity.OBSERVATION,
        categories=['control'],
        description='Kalman filter state estimation',
        output_columns=['state_estimate', 'covariance', 'innovation'],
        min_rows=20,
    ),
    'stability': EngineSpec(
        name='stability',
        granularity=Granularity.OBSERVATION,
        categories=['control'],
        description='Stability analysis (eigenvalues)',
        output_columns=['eigenvalues', 'stability_margin', 'damping'],
        min_rows=20,
    ),
}


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_engines_for_categories(categories: Set[str]) -> List[EngineSpec]:
    """
    Get all engines that should run given detected categories.

    Includes all universal engines plus category-specific engines.
    """
    engines = []
    for spec in ENGINE_SPECS.values():
        if spec.matches_categories(categories):
            engines.append(spec)
    return engines


def get_universal_engines() -> List[EngineSpec]:
    """Get all universal engines (no category restrictions)."""
    return [spec for spec in ENGINE_SPECS.values() if spec.is_universal()]


def get_domain_engines() -> List[EngineSpec]:
    """Get all domain-specific engines (have category restrictions)."""
    return [spec for spec in ENGINE_SPECS.values() if not spec.is_universal()]


def get_category_for_unit(unit: str) -> str:
    """Get category for a unit string."""
    if unit is None:
        return 'unknown'
    return UNIT_TO_CATEGORY.get(unit, 'unknown')


def get_all_categories() -> Set[str]:
    """Get set of all known categories."""
    return set(UNIT_TO_CATEGORY.values())


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Granularity',
    'EngineSpec',
    'UNIT_TO_CATEGORY',
    'CATEGORY_TO_ENGINES',
    'ENGINE_SPECS',
    'get_engines_for_categories',
    'get_universal_engines',
    'get_domain_engines',
    'get_category_for_unit',
    'get_all_categories',
]
