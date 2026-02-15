"""
RUDDER Engine Registry
======================

Centralized engine metadata with granularity, categories, and specifications.
Rudder uses this to decide what engines to run; PRISM just executes.

Covers all 116 primitives (Y1-Y9) and 18 engines (Y10-Y13).

Granularity determines output parquet:
- SIGNAL: vector.parquet (entity + signal + window)
- OBSERVATION: dynamics.parquet (entity + window)
- PAIR_DIRECTIONAL: pairs.parquet (entity + signal_i + signal_j + window)
- PAIR_SYMMETRIC: pairs.parquet (entity + pair + window)
- OBSERVATION_CROSS_SIGNAL: geometry.parquet (entity + window)
- TOPOLOGY: topology.parquet (entity + window)
- INFORMATION: information_flow.parquet (entity + window)
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
    # Cross-signal entity metrics (e.g., covariance, eigenvalues)
    # Output: geometry.parquet
    # Groupby: [entity_id]

    TOPOLOGY = "topology"
    # Topological metrics (Betti numbers, persistence)
    # Output: topology.parquet
    # Groupby: [entity_id]

    INFORMATION = "information"
    # Information flow metrics (transfer entropy, hierarchy)
    # Output: information_flow.parquet
    # Groupby: [entity_id]


# =============================================================================
# PILLAR ENUM
# =============================================================================

class Pillar(str, Enum):
    """The four pillars of structural health analysis."""
    GEOMETRY = "geometry"
    DYNAMICS = "dynamics"
    TOPOLOGY = "topology"
    INFORMATION = "information"
    PHYSICS = "physics"  # Cross-pillar


# =============================================================================
# ENGINE SPECIFICATION
# =============================================================================

@dataclass
class EngineSpec:
    """Full specification for a PRISM engine or primitive."""

    name: str
    granularity: Granularity
    pillar: Optional[Pillar] = None
    categories: List[str] = field(default_factory=list)
    # Empty = universal (runs on all data)
    # Non-empty = only runs when ANY listed category is present

    input_columns: List[str] = field(default_factory=lambda: ["I", "value"])
    output_columns: List[str] = field(default_factory=list)
    function: str = ""  # PRISM function path
    params: Dict[str, Any] = field(default_factory=dict)
    min_rows: int = 10
    description: str = ""
    equation: str = ""  # LaTeX equation

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
# Y1: INDIVIDUAL PRIMITIVES (35)
# =============================================================================

Y1_PRIMITIVES: Dict[str, EngineSpec] = {
    # --- Statistical Moments ---
    'mean': EngineSpec(
        name='mean',
        granularity=Granularity.SIGNAL,
        description='Arithmetic mean',
        equation=r'\bar{x} = \frac{1}{N}\sum_{i=1}^{N} x_i',
        output_columns=['mean'],
        min_rows=1,
    ),
    'std': EngineSpec(
        name='std',
        granularity=Granularity.SIGNAL,
        description='Standard deviation',
        equation=r'\sigma = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \bar{x})^2}',
        output_columns=['std'],
        min_rows=2,
    ),
    'var': EngineSpec(
        name='var',
        granularity=Granularity.SIGNAL,
        description='Variance',
        equation=r'\sigma^2',
        output_columns=['var'],
        min_rows=2,
    ),
    'skew': EngineSpec(
        name='skew',
        granularity=Granularity.SIGNAL,
        description='Skewness (asymmetry)',
        equation=r'\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3}',
        output_columns=['skew'],
        min_rows=3,
    ),
    'kurtosis': EngineSpec(
        name='kurtosis',
        granularity=Granularity.SIGNAL,
        description='Excess kurtosis (tail heaviness)',
        equation=r'\gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4} - 3',
        output_columns=['kurtosis'],
        min_rows=4,
    ),
    'rms': EngineSpec(
        name='rms',
        granularity=Granularity.SIGNAL,
        description='Root mean square',
        equation=r'x_{rms} = \sqrt{\frac{1}{N}\sum_{i=1}^{N} x_i^2}',
        output_columns=['rms'],
        min_rows=1,
    ),
    'crest_factor': EngineSpec(
        name='crest_factor',
        granularity=Granularity.SIGNAL,
        description='Peak-to-RMS ratio (impulsiveness)',
        equation=r'CF = \frac{|x|_{max}}{x_{rms}}',
        output_columns=['crest_factor'],
        min_rows=1,
    ),
    'peak_to_peak': EngineSpec(
        name='peak_to_peak',
        granularity=Granularity.SIGNAL,
        description='Total range',
        equation=r'x_{pp} = x_{max} - x_{min}',
        output_columns=['peak_to_peak'],
        min_rows=1,
    ),

    # --- Trend Analysis ---
    'trend_slope': EngineSpec(
        name='trend_slope',
        granularity=Granularity.SIGNAL,
        description='Linear trend slope',
        equation=r'm = \frac{\sum(x_i - \bar{x})(t_i - \bar{t})}{\sum(t_i - \bar{t})^2}',
        output_columns=['trend_slope'],
        min_rows=3,
    ),
    'trend_intercept': EngineSpec(
        name='trend_intercept',
        granularity=Granularity.SIGNAL,
        description='Linear trend intercept',
        equation=r'b = \bar{x} - m\bar{t}',
        output_columns=['trend_intercept'],
        min_rows=3,
    ),
    'trend_r2': EngineSpec(
        name='trend_r2',
        granularity=Granularity.SIGNAL,
        description='Trend fit R-squared',
        equation=r'R^2 = 1 - \frac{SS_{res}}{SS_{tot}}',
        output_columns=['trend_r2'],
        min_rows=3,
    ),
    'detrend_std': EngineSpec(
        name='detrend_std',
        granularity=Granularity.SIGNAL,
        description='Residual std after detrending',
        output_columns=['detrend_std'],
        min_rows=3,
    ),

    # --- Long-Range Dependence ---
    'hurst_rs': EngineSpec(
        name='hurst_rs',
        granularity=Granularity.SIGNAL,
        description='Hurst exponent (R/S method)',
        equation=r'E[R(n)/S(n)] = Cn^H',
        output_columns=['hurst_rs'],
        min_rows=50,
    ),
    'hurst_dfa': EngineSpec(
        name='hurst_dfa',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Hurst exponent (DFA method)',
        equation=r'F(n) \sim n^H',
        output_columns=['hurst_dfa'],
        min_rows=50,
    ),
    'hurst_wavelet': EngineSpec(
        name='hurst_wavelet',
        granularity=Granularity.SIGNAL,
        description='Hurst exponent (wavelet method)',
        equation=r'H = \frac{1}{2}(\beta + 1)',
        output_columns=['hurst_wavelet'],
        min_rows=32,
    ),

    # --- Entropy & Complexity ---
    'sample_entropy': EngineSpec(
        name='sample_entropy',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Sample entropy (regularity)',
        equation=r'SampEn = -\ln\frac{A}{B}',
        output_columns=['sample_entropy'],
        min_rows=30,
    ),
    'permutation_entropy': EngineSpec(
        name='permutation_entropy',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Permutation entropy',
        equation=r'H_p = -\sum p(\pi)\ln p(\pi)',
        output_columns=['permutation_entropy'],
        min_rows=20,
    ),
    'approximate_entropy': EngineSpec(
        name='approximate_entropy',
        granularity=Granularity.SIGNAL,
        description='Approximate entropy',
        equation=r'ApEn = \phi^m(r) - \phi^{m+1}(r)',
        output_columns=['approximate_entropy'],
        min_rows=30,
    ),
    'spectral_entropy': EngineSpec(
        name='spectral_entropy',
        granularity=Granularity.SIGNAL,
        description='Spectral entropy',
        equation=r'H_s = -\sum P(f)\ln P(f)',
        output_columns=['spectral_entropy'],
        min_rows=32,
    ),
    'svd_entropy': EngineSpec(
        name='svd_entropy',
        granularity=Granularity.SIGNAL,
        description='SVD entropy',
        equation=r'H_{svd} = -\sum \sigma_i \ln \sigma_i',
        output_columns=['svd_entropy'],
        min_rows=20,
    ),

    # --- Autocorrelation ---
    'acf_lag1': EngineSpec(
        name='acf_lag1',
        granularity=Granularity.SIGNAL,
        description='Lag-1 autocorrelation',
        equation=r'\rho_1 = \frac{E[(X_t - \mu)(X_{t-1} - \mu)]}{\sigma^2}',
        output_columns=['acf_lag1'],
        min_rows=3,
    ),
    'acf_decay_rate': EngineSpec(
        name='acf_decay_rate',
        granularity=Granularity.SIGNAL,
        description='ACF exponential decay rate',
        equation=r'\rho_k \sim e^{-k/\tau}',
        output_columns=['acf_decay_rate'],
        min_rows=20,
    ),
    'acf_zero_crossing': EngineSpec(
        name='acf_zero_crossing',
        granularity=Granularity.SIGNAL,
        description='First ACF zero-crossing lag',
        output_columns=['acf_zero_crossing'],
        min_rows=10,
    ),
    'partial_acf_lag1': EngineSpec(
        name='partial_acf_lag1',
        granularity=Granularity.SIGNAL,
        description='Partial ACF lag-1',
        output_columns=['partial_acf_lag1'],
        min_rows=5,
    ),

    # --- Spectral Features ---
    'dominant_freq': EngineSpec(
        name='dominant_freq',
        granularity=Granularity.SIGNAL,
        description='Dominant frequency',
        equation=r'f_{dom} = \arg\max_f |X(f)|^2',
        output_columns=['dominant_freq'],
        min_rows=32,
    ),
    'spectral_centroid': EngineSpec(
        name='spectral_centroid',
        granularity=Granularity.SIGNAL,
        description='Spectral center of mass',
        equation=r'f_c = \frac{\sum f \cdot P(f)}{\sum P(f)}',
        output_columns=['spectral_centroid'],
        min_rows=32,
    ),
    'spectral_bandwidth': EngineSpec(
        name='spectral_bandwidth',
        granularity=Granularity.SIGNAL,
        description='Spectral spread',
        output_columns=['spectral_bandwidth'],
        min_rows=32,
    ),
    'spectral_slope': EngineSpec(
        name='spectral_slope',
        granularity=Granularity.SIGNAL,
        description='Power law exponent (1/f noise)',
        equation=r'P(f) \sim f^{-\beta}',
        output_columns=['spectral_slope', 'spectral_slope_r2'],
        min_rows=32,
    ),
    'spectral_rolloff': EngineSpec(
        name='spectral_rolloff',
        granularity=Granularity.SIGNAL,
        description='85% spectral energy frequency',
        output_columns=['spectral_rolloff'],
        min_rows=32,
    ),
    'band_power_low': EngineSpec(
        name='band_power_low',
        granularity=Granularity.SIGNAL,
        description='Low-frequency band power',
        output_columns=['band_power_low'],
        min_rows=32,
    ),
    'band_power_mid': EngineSpec(
        name='band_power_mid',
        granularity=Granularity.SIGNAL,
        description='Mid-frequency band power',
        output_columns=['band_power_mid'],
        min_rows=32,
    ),
    'band_power_high': EngineSpec(
        name='band_power_high',
        granularity=Granularity.SIGNAL,
        description='High-frequency band power',
        output_columns=['band_power_high'],
        min_rows=32,
    ),

    # --- Nonlinear Features ---
    'zero_crossing_rate': EngineSpec(
        name='zero_crossing_rate',
        granularity=Granularity.SIGNAL,
        description='Sign change frequency',
        output_columns=['zero_crossing_rate'],
        min_rows=5,
    ),
    'mean_crossing_rate': EngineSpec(
        name='mean_crossing_rate',
        granularity=Granularity.SIGNAL,
        description='Mean crossing frequency',
        output_columns=['mean_crossing_rate'],
        min_rows=5,
    ),
    'turning_points': EngineSpec(
        name='turning_points',
        granularity=Granularity.SIGNAL,
        description='Local extrema count',
        output_columns=['turning_points'],
        min_rows=5,
    ),
}


# =============================================================================
# Y2: PAIRWISE PRIMITIVES (20)
# =============================================================================

Y2_PRIMITIVES: Dict[str, EngineSpec] = {
    # --- Correlation Measures ---
    'pearson_corr': EngineSpec(
        name='pearson_corr',
        granularity=Granularity.PAIR_SYMMETRIC,
        pillar=Pillar.GEOMETRY,
        description='Pearson correlation',
        equation=r'\rho_{XY} = \frac{Cov(X,Y)}{\sigma_X \sigma_Y}',
        output_columns=['pearson_corr'],
        min_rows=10,
    ),
    'spearman_corr': EngineSpec(
        name='spearman_corr',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Spearman rank correlation',
        equation=r'\rho_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}',
        output_columns=['spearman_corr'],
        min_rows=10,
    ),
    'kendall_tau': EngineSpec(
        name='kendall_tau',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Kendall tau concordance',
        equation=r'\tau = \frac{C - D}{\binom{n}{2}}',
        output_columns=['kendall_tau'],
        min_rows=10,
    ),
    'partial_corr': EngineSpec(
        name='partial_corr',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Partial correlation',
        output_columns=['partial_corr'],
        min_rows=15,
    ),
    'cross_corr_max': EngineSpec(
        name='cross_corr_max',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Maximum cross-correlation',
        output_columns=['cross_corr_max'],
        min_rows=20,
    ),
    'cross_corr_lag': EngineSpec(
        name='cross_corr_lag',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Lag at max cross-correlation',
        output_columns=['cross_corr_lag'],
        min_rows=20,
    ),

    # --- Information-Theoretic ---
    'mutual_info': EngineSpec(
        name='mutual_info',
        granularity=Granularity.PAIR_SYMMETRIC,
        pillar=Pillar.INFORMATION,
        description='Mutual information',
        equation=r'I(X;Y) = \sum p(x,y)\log\frac{p(x,y)}{p(x)p(y)}',
        output_columns=['mutual_info'],
        min_rows=30,
    ),
    'normalized_mi': EngineSpec(
        name='normalized_mi',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Normalized mutual information',
        equation=r'NMI = \frac{I(X;Y)}{\sqrt{H(X)H(Y)}}',
        output_columns=['normalized_mi'],
        min_rows=30,
    ),
    'conditional_entropy': EngineSpec(
        name='conditional_entropy',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Conditional entropy H(Y|X)',
        equation=r'H(Y|X) = -\sum p(x,y)\log p(y|x)',
        output_columns=['conditional_entropy'],
        min_rows=30,
    ),

    # --- Distance Measures ---
    'dtw_distance': EngineSpec(
        name='dtw_distance',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Dynamic time warping distance',
        equation=r'DTW(X,Y) = \min_{\pi} \sum d(x_{\pi_x(k)}, y_{\pi_y(k)})',
        output_columns=['dtw_distance', 'dtw_path_length'],
        min_rows=10,
    ),
    'euclidean_dist': EngineSpec(
        name='euclidean_dist',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Euclidean distance',
        equation=r'd = \sqrt{\sum (x_i - y_i)^2}',
        output_columns=['euclidean_dist'],
        min_rows=5,
    ),
    'cosine_similarity': EngineSpec(
        name='cosine_similarity',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Cosine similarity',
        equation=r'\cos\theta = \frac{X \cdot Y}{||X|| ||Y||}',
        output_columns=['cosine_similarity'],
        min_rows=5,
    ),

    # --- Coherence & Phase ---
    'coherence': EngineSpec(
        name='coherence',
        granularity=Granularity.PAIR_SYMMETRIC,
        pillar=Pillar.GEOMETRY,
        description='Spectral coherence',
        equation=r'C_{xy}(f) = \frac{|S_{xy}(f)|^2}{S_{xx}(f)S_{yy}(f)}',
        output_columns=['coherence'],
        min_rows=32,
    ),
    'phase_lag': EngineSpec(
        name='phase_lag',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Phase difference',
        equation=r'\phi = \arg(S_{xy}(f))',
        output_columns=['phase_lag'],
        min_rows=32,
    ),
    'phase_sync': EngineSpec(
        name='phase_sync',
        granularity=Granularity.PAIR_SYMMETRIC,
        description='Instantaneous phase synchronization',
        equation=r'R = |\frac{1}{N}\sum e^{i(\phi_x(t) - \phi_y(t))}|',
        output_columns=['phase_sync'],
        min_rows=32,
    ),

    # --- Causality (Basic) ---
    'granger_f_stat': EngineSpec(
        name='granger_f_stat',
        granularity=Granularity.PAIR_DIRECTIONAL,
        pillar=Pillar.INFORMATION,
        description='Granger causality F-statistic',
        equation=r'F = \frac{(RSS_r - RSS_u)/p}{RSS_u/(n-2p-1)}',
        output_columns=['granger_f_stat'],
        min_rows=50,
    ),
    'granger_p_value': EngineSpec(
        name='granger_p_value',
        granularity=Granularity.PAIR_DIRECTIONAL,
        description='Granger causality p-value',
        output_columns=['granger_p_value'],
        min_rows=50,
    ),
    'ccm_score': EngineSpec(
        name='ccm_score',
        granularity=Granularity.PAIR_DIRECTIONAL,
        description='Convergent cross-mapping score',
        output_columns=['ccm_score'],
        min_rows=100,
    ),
    'optimal_lag': EngineSpec(
        name='optimal_lag',
        granularity=Granularity.PAIR_DIRECTIONAL,
        description='Optimal causal lag (AIC)',
        output_columns=['optimal_lag'],
        min_rows=30,
    ),
}


# =============================================================================
# Y3: MATRIX PRIMITIVES (10)
# =============================================================================

Y3_PRIMITIVES: Dict[str, EngineSpec] = {
    'covariance_matrix': EngineSpec(
        name='covariance_matrix',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Signal covariance matrix',
        equation=r'\Sigma_{ij} = Cov(X_i, X_j)',
        output_columns=['covariance_matrix'],
        min_rows=10,
    ),
    'correlation_matrix': EngineSpec(
        name='correlation_matrix',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Signal correlation matrix',
        equation=r'R_{ij} = \frac{\Sigma_{ij}}{\sigma_i \sigma_j}',
        output_columns=['correlation_matrix'],
        min_rows=10,
    ),
    'eigenvalues': EngineSpec(
        name='eigenvalues',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Covariance eigenvalues',
        equation=r'\Sigma v = \lambda v',
        output_columns=['eigenvalues', 'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3'],
        min_rows=10,
    ),
    'eigenvectors': EngineSpec(
        name='eigenvectors',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Covariance eigenvectors (mode shapes)',
        output_columns=['eigenvectors'],
        min_rows=10,
    ),
    'effective_dimension': EngineSpec(
        name='effective_dimension',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Participation ratio',
        equation=r'd_{eff} = \frac{(\sum \lambda_i)^2}{\sum \lambda_i^2}',
        output_columns=['effective_dimension', 'eff_dim'],
        min_rows=10,
    ),
    'coherence_ratio': EngineSpec(
        name='coherence_ratio',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='First eigenvalue dominance',
        equation=r'\frac{\lambda_1}{\sum \lambda_i}',
        output_columns=['coherence_ratio', 'coherence'],
        min_rows=10,
    ),
    'condition_number': EngineSpec(
        name='condition_number',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Matrix condition number',
        equation=r'\kappa = \frac{\lambda_{max}}{\lambda_{min}}',
        output_columns=['condition_number'],
        min_rows=10,
    ),
    'trace': EngineSpec(
        name='trace',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Matrix trace (total variance)',
        equation=r'tr(\Sigma) = \sum \lambda_i',
        output_columns=['trace'],
        min_rows=10,
    ),
    'determinant': EngineSpec(
        name='determinant',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Matrix determinant (generalized variance)',
        equation=r'det(\Sigma) = \prod \lambda_i',
        output_columns=['determinant'],
        min_rows=10,
    ),
    'frobenius_norm': EngineSpec(
        name='frobenius_norm',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        description='Frobenius norm',
        equation=r'||\Sigma||_F = \sqrt{\sum \sigma_{ij}^2}',
        output_columns=['frobenius_norm'],
        min_rows=10,
    ),
}


# =============================================================================
# Y4: EMBEDDING PRIMITIVES (4)
# =============================================================================

Y4_PRIMITIVES: Dict[str, EngineSpec] = {
    'embedding_delay': EngineSpec(
        name='embedding_delay',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Optimal time delay (MI method)',
        equation=r'\tau = \arg\min_\tau MI(x(t), x(t+\tau))',
        output_columns=['embedding_delay', 'time_delay'],
        min_rows=50,
    ),
    'embedding_dimension': EngineSpec(
        name='embedding_dimension',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Embedding dimension (FNN method)',
        output_columns=['embedding_dimension', 'embedding_dim'],
        min_rows=50,
    ),
    'delay_vector': EngineSpec(
        name='delay_vector',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Reconstructed state vectors',
        equation=r'\vec{x}(t) = [x(t), x(t-\tau), ..., x(t-(d-1)\tau)]',
        output_columns=['delay_vectors'],
        min_rows=50,
    ),
    'reconstruction_quality': EngineSpec(
        name='reconstruction_quality',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Embedding quality score',
        equation=r'Q = 1 - \frac{FNN(d_E)}{FNN(1)}',
        output_columns=['reconstruction_quality'],
        min_rows=50,
    ),
}


# =============================================================================
# Y5: TOPOLOGY PRIMITIVES (5)
# =============================================================================

Y5_PRIMITIVES: Dict[str, EngineSpec] = {
    'betti_0': EngineSpec(
        name='betti_0',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        description='Connected components count',
        equation=r'\beta_0',
        output_columns=['betti_0'],
        min_rows=50,
    ),
    'betti_1': EngineSpec(
        name='betti_1',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        description='1-dimensional holes (loops)',
        equation=r'\beta_1',
        output_columns=['betti_1'],
        min_rows=50,
    ),
    'betti_2': EngineSpec(
        name='betti_2',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        description='2-dimensional voids',
        equation=r'\beta_2',
        output_columns=['betti_2'],
        min_rows=100,
    ),
    'persistence_diagram': EngineSpec(
        name='persistence_diagram',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        description='Birth-death persistence pairs',
        output_columns=['persistence_diagram', 'h1_max_persistence'],
        min_rows=50,
    ),
    'persistence_entropy': EngineSpec(
        name='persistence_entropy',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        description='Persistence distribution entropy',
        equation=r'H_p = -\sum p_i \ln p_i',
        output_columns=['persistence_entropy', 'h1_persistence_entropy'],
        min_rows=50,
    ),
}


# =============================================================================
# Y6: NETWORK PRIMITIVES (11)
# =============================================================================

Y6_PRIMITIVES: Dict[str, EngineSpec] = {
    'degree_centrality': EngineSpec(
        name='degree_centrality',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Node connection count',
        equation=r'C_D(v) = \frac{deg(v)}{n-1}',
        output_columns=['degree_centrality'],
        min_rows=10,
    ),
    'betweenness_centrality': EngineSpec(
        name='betweenness_centrality',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Path intermediary measure',
        equation=r'C_B(v) = \sum_{s \neq v \neq t} \frac{\sigma_{st}(v)}{\sigma_{st}}',
        output_columns=['betweenness_centrality'],
        min_rows=10,
    ),
    'closeness_centrality': EngineSpec(
        name='closeness_centrality',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Average distance to others',
        equation=r'C_C(v) = \frac{n-1}{\sum_u d(v,u)}',
        output_columns=['closeness_centrality'],
        min_rows=10,
    ),
    'eigenvector_centrality': EngineSpec(
        name='eigenvector_centrality',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Influence propagation',
        equation=r'Ax = \lambda x',
        output_columns=['eigenvector_centrality'],
        min_rows=10,
    ),
    'clustering_coefficient': EngineSpec(
        name='clustering_coefficient',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Local clustering density',
        equation=r'C_i = \frac{2e_i}{k_i(k_i-1)}',
        output_columns=['clustering_coefficient'],
        min_rows=10,
    ),
    'global_efficiency': EngineSpec(
        name='global_efficiency',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Network integration',
        equation=r'E = \frac{1}{n(n-1)}\sum_{i \neq j} \frac{1}{d_{ij}}',
        output_columns=['global_efficiency'],
        min_rows=10,
    ),
    'modularity': EngineSpec(
        name='modularity',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Community structure',
        equation=r'Q = \frac{1}{2m}\sum_{ij}[A_{ij} - \frac{k_i k_j}{2m}]\delta(c_i, c_j)',
        output_columns=['modularity'],
        min_rows=10,
    ),
    'assortativity': EngineSpec(
        name='assortativity',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Degree correlation',
        output_columns=['assortativity'],
        min_rows=10,
    ),
    'average_path_length': EngineSpec(
        name='average_path_length',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Mean shortest path',
        equation=r'L = \frac{1}{n(n-1)}\sum_{i \neq j} d_{ij}',
        output_columns=['average_path_length'],
        min_rows=10,
    ),
    'network_density': EngineSpec(
        name='network_density',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Edge fraction',
        equation=r'\rho = \frac{2m}{n(n-1)}',
        output_columns=['network_density'],
        min_rows=10,
    ),
    'rich_club_coefficient': EngineSpec(
        name='rich_club_coefficient',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        description='Hub interconnection',
        equation=r'\phi(k) = \frac{2E_{>k}}{N_{>k}(N_{>k}-1)}',
        output_columns=['rich_club_coefficient'],
        min_rows=10,
    ),
}


# =============================================================================
# Y7: DYNAMICAL PRIMITIVES (10)
# =============================================================================

Y7_PRIMITIVES: Dict[str, EngineSpec] = {
    # --- Lyapunov ---
    'lyapunov_max': EngineSpec(
        name='lyapunov_max',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Maximum Lyapunov exponent',
        equation=r'\lambda_{max} = \lim_{t \to \infty} \frac{1}{t} \ln \frac{||\delta x(t)||}{||\delta x(0)||}',
        output_columns=['lyapunov_max'],
        min_rows=100,
    ),
    'lyapunov_spectrum': EngineSpec(
        name='lyapunov_spectrum',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Full Lyapunov exponent spectrum',
        output_columns=['lyapunov_spectrum'],
        min_rows=100,
    ),
    'kaplan_yorke_dim': EngineSpec(
        name='kaplan_yorke_dim',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Kaplan-Yorke dimension',
        equation=r'D_{KY} = j + \frac{\sum_{i=1}^j \lambda_i}{|\lambda_{j+1}|}',
        output_columns=['kaplan_yorke_dim'],
        min_rows=100,
    ),

    # --- RQA ---
    'recurrence_rate': EngineSpec(
        name='recurrence_rate',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Recurrence rate',
        equation=r'RR = \frac{1}{N^2}\sum_{i,j} R_{ij}',
        output_columns=['recurrence_rate'],
        min_rows=50,
    ),
    'determinism': EngineSpec(
        name='determinism',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Determinism (diagonal lines)',
        equation=r'DET = \frac{\sum_{l=l_{min}}^N l \cdot P(l)}{\sum_{l=1}^N l \cdot P(l)}',
        output_columns=['determinism'],
        min_rows=50,
    ),
    'laminarity': EngineSpec(
        name='laminarity',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Laminarity (vertical lines)',
        equation=r'LAM = \frac{\sum_{v=v_{min}}^N v \cdot P(v)}{\sum_{v=1}^N v \cdot P(v)}',
        output_columns=['laminarity'],
        min_rows=50,
    ),
    'trapping_time': EngineSpec(
        name='trapping_time',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Average trapping time',
        equation=r'TT = \frac{\sum_{v=v_{min}}^N v \cdot P(v)}{\sum_{v=v_{min}}^N P(v)}',
        output_columns=['trapping_time'],
        min_rows=50,
    ),
    'rqa_entropy': EngineSpec(
        name='rqa_entropy',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Recurrence entropy',
        equation=r'ENTR = -\sum P(l) \ln P(l)',
        output_columns=['rqa_entropy'],
        min_rows=50,
    ),

    # --- Attractor ---
    'correlation_dimension': EngineSpec(
        name='correlation_dimension',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Grassberger-Procaccia correlation dimension',
        equation=r'D_2 = \lim_{r \to 0} \frac{\ln C(r)}{\ln r}',
        output_columns=['correlation_dimension', 'correlation_dim'],
        min_rows=100,
    ),
    'largest_lyapunov': EngineSpec(
        name='largest_lyapunov',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Largest Lyapunov (Rosenstein method)',
        output_columns=['largest_lyapunov'],
        min_rows=100,
    ),
}


# =============================================================================
# Y8: TEST PRIMITIVES (12)
# =============================================================================

Y8_PRIMITIVES: Dict[str, EngineSpec] = {
    'adf_statistic': EngineSpec(
        name='adf_statistic',
        granularity=Granularity.SIGNAL,
        description='Augmented Dickey-Fuller statistic',
        output_columns=['adf_statistic'],
        min_rows=20,
    ),
    'adf_pvalue': EngineSpec(
        name='adf_pvalue',
        granularity=Granularity.SIGNAL,
        description='ADF p-value (stationarity test)',
        output_columns=['adf_pvalue'],
        min_rows=20,
    ),
    'kpss_statistic': EngineSpec(
        name='kpss_statistic',
        granularity=Granularity.SIGNAL,
        description='KPSS test statistic',
        output_columns=['kpss_statistic'],
        min_rows=20,
    ),
    'kpss_pvalue': EngineSpec(
        name='kpss_pvalue',
        granularity=Granularity.SIGNAL,
        description='KPSS p-value (trend stationarity)',
        output_columns=['kpss_pvalue'],
        min_rows=20,
    ),
    'ljung_box_stat': EngineSpec(
        name='ljung_box_stat',
        granularity=Granularity.SIGNAL,
        description='Ljung-Box test statistic',
        equation=r'Q = n(n+2)\sum_{k=1}^h \frac{\hat{\rho}_k^2}{n-k}',
        output_columns=['ljung_box_stat'],
        min_rows=20,
    ),
    'ljung_box_pvalue': EngineSpec(
        name='ljung_box_pvalue',
        granularity=Granularity.SIGNAL,
        description='Ljung-Box p-value (white noise test)',
        output_columns=['ljung_box_pvalue'],
        min_rows=20,
    ),
    'jarque_bera_stat': EngineSpec(
        name='jarque_bera_stat',
        granularity=Granularity.SIGNAL,
        description='Jarque-Bera test statistic',
        equation=r'JB = \frac{n}{6}(S^2 + \frac{(K-3)^2}{4})',
        output_columns=['jarque_bera_stat'],
        min_rows=20,
    ),
    'jarque_bera_pvalue': EngineSpec(
        name='jarque_bera_pvalue',
        granularity=Granularity.SIGNAL,
        description='Jarque-Bera p-value (normality test)',
        output_columns=['jarque_bera_pvalue'],
        min_rows=20,
    ),
    'levene_stat': EngineSpec(
        name='levene_stat',
        granularity=Granularity.SIGNAL,
        description='Levene test statistic',
        output_columns=['levene_stat'],
        min_rows=20,
    ),
    'levene_pvalue': EngineSpec(
        name='levene_pvalue',
        granularity=Granularity.SIGNAL,
        description='Levene p-value (homoscedasticity)',
        output_columns=['levene_pvalue'],
        min_rows=20,
    ),
    'runs_test_stat': EngineSpec(
        name='runs_test_stat',
        granularity=Granularity.SIGNAL,
        description='Runs test statistic',
        output_columns=['runs_test_stat'],
        min_rows=20,
    ),
    'runs_test_pvalue': EngineSpec(
        name='runs_test_pvalue',
        granularity=Granularity.SIGNAL,
        description='Runs test p-value (randomness)',
        output_columns=['runs_test_pvalue'],
        min_rows=20,
    ),
}


# =============================================================================
# Y9: INFORMATION PRIMITIVES (9)
# =============================================================================

Y9_PRIMITIVES: Dict[str, EngineSpec] = {
    # --- Transfer Entropy ---
    'transfer_entropy_xy': EngineSpec(
        name='transfer_entropy_xy',
        granularity=Granularity.PAIR_DIRECTIONAL,
        pillar=Pillar.INFORMATION,
        description='Transfer entropy X→Y',
        equation=r'TE_{X \to Y} = \sum p(y_{t+1}, y_t^{(k)}, x_t^{(l)}) \log \frac{p(y_{t+1}|y_t^{(k)}, x_t^{(l)})}{p(y_{t+1}|y_t^{(k)})}',
        output_columns=['transfer_entropy'],
        min_rows=50,
    ),
    'transfer_entropy_yx': EngineSpec(
        name='transfer_entropy_yx',
        granularity=Granularity.PAIR_DIRECTIONAL,
        pillar=Pillar.INFORMATION,
        description='Transfer entropy Y→X',
        output_columns=['reverse_transfer_entropy'],
        min_rows=50,
    ),
    'net_transfer_entropy': EngineSpec(
        name='net_transfer_entropy',
        granularity=Granularity.PAIR_DIRECTIONAL,
        pillar=Pillar.INFORMATION,
        description='Net information flow',
        equation=r'nTE = TE_{X \to Y} - TE_{Y \to X}',
        output_columns=['net_transfer_entropy'],
        min_rows=50,
    ),
    'effective_te': EngineSpec(
        name='effective_te',
        granularity=Granularity.PAIR_DIRECTIONAL,
        pillar=Pillar.INFORMATION,
        description='Bias-corrected transfer entropy',
        output_columns=['effective_te'],
        min_rows=50,
    ),

    # --- Complexity ---
    'lempel_ziv_complexity': EngineSpec(
        name='lempel_ziv_complexity',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Lempel-Ziv complexity',
        output_columns=['lempel_ziv_complexity'],
        min_rows=50,
    ),
    'kolmogorov_complexity': EngineSpec(
        name='kolmogorov_complexity',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Approximated Kolmogorov complexity',
        output_columns=['kolmogorov_complexity'],
        min_rows=50,
    ),
    'multiscale_entropy': EngineSpec(
        name='multiscale_entropy',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Multi-scale sample entropy',
        equation=r'MSE(\tau) = SampEn(y^{(\tau)})',
        output_columns=['multiscale_entropy'],
        min_rows=100,
    ),
    'fisher_information': EngineSpec(
        name='fisher_information',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Fisher information',
        equation=r'F = E[(\frac{\partial}{\partial \theta} \ln p)^2]',
        output_columns=['fisher_information'],
        min_rows=50,
    ),
    'active_information': EngineSpec(
        name='active_information',
        granularity=Granularity.SIGNAL,
        pillar=Pillar.INFORMATION,
        description='Active information storage',
        equation=r'A = I(X_{t-1}; X_t)',
        output_columns=['active_information'],
        min_rows=30,
    ),
}


# =============================================================================
# Y10-Y13: COMPOSED ENGINES
# =============================================================================

Y10_STRUCTURE_ENGINES: Dict[str, EngineSpec] = {
    'covariance_engine': EngineSpec(
        name='covariance_engine',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Covariance analysis engine',
        output_columns=['covariance_matrix', 'eigenvalues', 'eigenvectors'],
        min_rows=10,
    ),
    'eigenvalue_engine': EngineSpec(
        name='eigenvalue_engine',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Eigenvalue decomposition engine',
        output_columns=['eigenvalues', 'eff_dim', 'coherence'],
        min_rows=10,
    ),
    'koopman_engine': EngineSpec(
        name='koopman_engine',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Koopman operator analysis',
        output_columns=['koopman_modes', 'koopman_eigenvalues'],
        min_rows=50,
    ),
    'spectral_engine': EngineSpec(
        name='spectral_engine',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Cross-spectral analysis',
        output_columns=['cross_spectrum', 'coherence_matrix'],
        min_rows=32,
    ),
    'wavelet_engine': EngineSpec(
        name='wavelet_engine',
        granularity=Granularity.OBSERVATION_CROSS_SIGNAL,
        pillar=Pillar.GEOMETRY,
        description='Wavelet decomposition engine',
        output_columns=['wavelet_coeffs', 'scale_energy'],
        min_rows=32,
    ),
}

Y11_PHYSICS_ENGINES: Dict[str, EngineSpec] = {
    'energy_engine': EngineSpec(
        name='energy_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.PHYSICS,
        categories=['velocity', 'mass'],
        description='System energy computation',
        output_columns=['kinetic_energy', 'potential_energy', 'total_energy'],
        min_rows=10,
    ),
    'mass_engine': EngineSpec(
        name='mass_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.PHYSICS,
        categories=['flow_mass', 'density'],
        description='Mass balance computation',
        output_columns=['mass_flow', 'accumulation'],
        min_rows=10,
    ),
    'momentum_engine': EngineSpec(
        name='momentum_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.PHYSICS,
        categories=['velocity', 'mass'],
        description='Momentum computation',
        output_columns=['momentum', 'angular_momentum'],
        min_rows=10,
    ),
    'constitutive_engine': EngineSpec(
        name='constitutive_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.PHYSICS,
        categories=['stress', 'force'],
        description='Constitutive relation analysis',
        output_columns=['youngs_modulus', 'viscosity'],
        min_rows=10,
    ),
}

Y12_DYNAMICS_ENGINES: Dict[str, EngineSpec] = {
    'lyapunov_engine': EngineSpec(
        name='lyapunov_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Lyapunov stability analysis',
        output_columns=['lyapunov_max', 'lyapunov_spectrum', 'stability_class'],
        min_rows=100,
    ),
    'attractor_engine': EngineSpec(
        name='attractor_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Attractor characterization',
        output_columns=['correlation_dim', 'kaplan_yorke_dim', 'attractor_type'],
        min_rows=100,
    ),
    'recurrence_engine': EngineSpec(
        name='recurrence_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Recurrence quantification analysis',
        output_columns=['determinism', 'laminarity', 'trapping_time', 'rqa_entropy'],
        min_rows=50,
    ),
    'bifurcation_engine': EngineSpec(
        name='bifurcation_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.DYNAMICS,
        description='Bifurcation detection',
        output_columns=['bifurcation_points', 'regime_changes'],
        min_rows=100,
    ),
}

Y13_ADVANCED_ENGINES: Dict[str, EngineSpec] = {
    # -------------------------------------------------------------------------
    # Engine 130: Causality Engine
    # Computes causal networks using Granger causality and Transfer Entropy
    # -------------------------------------------------------------------------
    'causality_engine': EngineSpec(
        name='causality_engine',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        function='prism.engines.python.advanced.causality_engine.run_causality_engine',
        description='Causal network analysis using Granger causality and Transfer Entropy. '
                    'Identifies drivers, sinks, feedback loops, and causal hierarchy.',
        equation=r'''
Granger: F_{i \to j} = \frac{(RSS_{reduced} - RSS_{full}) / p}{RSS_{full} / (n - 2p - 1)}
Transfer Entropy: T_{X \to Y} = \sum p(y_{t+1}, y_t^k, x_t^l) \log \frac{p(y_{t+1} | y_t^k, x_t^l)}{p(y_{t+1} | y_t^k)}
Hierarchy: H = 1 - \frac{\text{reciprocal edges}}{\text{total edges}}
''',
        output_columns=[
            'granger_f', 'granger_p', 'transfer_entropy', 'is_significant',
            'density', 'hierarchy', 'n_feedback_loops',
            'top_driver', 'top_driver_flow', 'top_sink',
            'bottleneck', 'bottleneck_centrality', 'mean_te',
        ],
        params={'max_lag': 5, 'threshold_percentile': 75},
        min_rows=50,
    ),

    # -------------------------------------------------------------------------
    # Engine 131: Topology Engine
    # Computes persistent homology metrics for attractor analysis
    # -------------------------------------------------------------------------
    'topology_engine': EngineSpec(
        name='topology_engine',
        granularity=Granularity.TOPOLOGY,
        pillar=Pillar.TOPOLOGY,
        function='prism.engines.python.advanced.topology_engine.run_topology_engine',
        description='Persistent homology analysis: Betti numbers, persistence entropy, '
                    'topological complexity. Detects attractor fragmentation and structural changes.',
        equation=r'''
\beta_0 = \text{connected components}, \quad \beta_1 = \text{loops/holes}, \quad \beta_2 = \text{voids}
H_{persist} = -\sum_i \frac{p_i}{\sum p} \log \frac{p_i}{\sum p} \quad (p_i = \text{persistence of feature } i)
W_q(D_1, D_2) = \inf_\gamma \left( \sum_{i} ||x_i - \gamma(x_i)||_\infty^q \right)^{1/q}
''',
        output_columns=[
            'betti_0', 'betti_1', 'betti_2',
            'persistence_entropy_h0', 'persistence_entropy_h1',
            'total_persistence_h1', 'max_persistence_h1',
            'topological_complexity', 'fragmentation', 'topology_change',
        ],
        params={'max_dimension': 2},
        min_rows=100,
    ),

    # -------------------------------------------------------------------------
    # Engine 132: Emergence Engine
    # Computes synergy, redundancy, unique information via PID
    # -------------------------------------------------------------------------
    'emergence_engine': EngineSpec(
        name='emergence_engine',
        granularity=Granularity.INFORMATION,
        pillar=Pillar.INFORMATION,
        function='prism.engines.python.advanced.emergence_engine.run_emergence_engine',
        description='Emergence/synergy analysis via Partial Information Decomposition. '
                    'Identifies multi-signal interactions that pairwise analysis misses.',
        equation=r'''
I(X_1, X_2; Y) = \underbrace{R(X_1, X_2 \to Y)}_{\text{redundancy}} +
                 \underbrace{U_1(X_1 \to Y)}_{\text{unique}_1} +
                 \underbrace{U_2(X_2 \to Y)}_{\text{unique}_2} +
                 \underbrace{S(X_1, X_2 \to Y)}_{\text{synergy}}
\text{Emergence ratio} = \frac{S}{R + U_1 + U_2 + S}
''',
        output_columns=[
            'mutual_information',  # pairwise
            'redundancy', 'unique_1', 'unique_2', 'synergy',  # triplet PID
            'total_info', 'synergy_ratio',
        ],
        params={'n_bins': 10},
        min_rows=50,
    ),

    # -------------------------------------------------------------------------
    # Engine 133: Integration Engine
    # Combines all metrics into unified health assessment
    # -------------------------------------------------------------------------
    'integration_engine': EngineSpec(
        name='integration_engine',
        granularity=Granularity.OBSERVATION,
        pillar=Pillar.PHYSICS,  # Cross-pillar integration
        function='prism.engines.python.advanced.integration_engine.run_integration_engine',
        description='Master integration engine. Combines all metrics into unified health assessment. '
                    'Computes composite health score, risk level, and recommendations.',
        equation=r'''
\text{Health} = 100 \times (1 - \text{Risk})
\text{Risk} = w_s \cdot S_{stability} + w_p \cdot S_{predictability} + w_{ph} \cdot S_{physics} + w_t \cdot S_{topology} + w_c \cdot S_{causality}
\text{where } w_s = 0.25, w_p = 0.20, w_{ph} = 0.25, w_t = 0.15, w_c = 0.15
''',
        output_columns=[
            'health_score', 'risk_level',
            'stability_score', 'predictability_score', 'physics_score',
            'topology_score', 'causality_score',
            'lyapunov', 'effective_dimension', 'determinism', 'csd_score',
            'efficiency', 'balance_residual_pct',
            'primary_concern', 'secondary_concern', 'recommendation',
        ],
        params={
            'weights': {
                'stability': 0.25, 'predictability': 0.20, 'physics': 0.25,
                'topology': 0.15, 'causality': 0.15,
            },
            'thresholds': {
                'lyapunov_critical': 0.1, 'determinism_low': 0.5,
                'csd_high': 2.0, 'efficiency_low': 0.7, 'balance_error': 10.0,
            },
        },
        min_rows=1,  # Can run on any number of windows
    ),
}

# =============================================================================
# Y14: STATISTICS ENGINES (PR #14 - FINAL)
# Fleet-wide analytics, baselines, anomaly scoring, report generation
# =============================================================================

Y14_STATISTICS_ENGINES: Dict[str, EngineSpec] = {
    # -------------------------------------------------------------------------
    # Engine 134: Baseline Engine
    # Computes fleet-wide and per-entity baselines for all metrics
    # -------------------------------------------------------------------------
    'baseline_engine': EngineSpec(
        name='baseline_engine',
        granularity=Granularity.OBSERVATION,
        pillar=None,  # Cross-pillar statistics
        function='prism.engines.python.statistics.baseline_engine.run_baseline_engine',
        description='Compute fleet-wide and per-entity baselines for all metrics. '
                    'Establishes what "normal" looks like for comparison.',
        equation=r'''
\mu_{baseline} = \frac{1}{n}\sum_{i=1}^{n} x_i \quad \text{(first } n \text{ windows)}
\sigma_{baseline} = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(x_i - \mu)^2}
CV = \frac{\sigma}{\mu} \quad \text{(coefficient of variation)}
''',
        output_columns=[
            'metric_source', 'metric_name', 'entity_id',
            'mean', 'std', 'median', 'min', 'max',
            'p5', 'p25', 'p75', 'p95', 'n_samples',
        ],
        params={'baseline_windows': 10, 'group_by': 'fleet'},
        min_rows=3,
    ),

    # -------------------------------------------------------------------------
    # Engine 135: Anomaly Engine
    # Scores deviations from baseline using range exceedance
    # -------------------------------------------------------------------------
    'anomaly_engine': EngineSpec(
        name='anomaly_engine',
        granularity=Granularity.OBSERVATION,
        pillar=None,  # Cross-pillar statistics
        function='prism.engines.python.statistics.anomaly_engine.run_anomaly_engine',
        description='Compute anomaly scores by comparing current values to baselines. '
                    'Uses percentile rankings, range exceedance, and multi-metric fusion.',
        equation=r'''
\text{exceedance} = \frac{x - p_{95}}{p_{95} - p_{05}} \quad (x > p_{95})
\text{Severity} = \begin{cases}
    \text{CRITICAL} & \text{exceedance} \geq 1.0 \\
    \text{WARNING} & \text{exceedance} \geq 0.5 \\
    \text{ELEVATED} & \text{exceedance} > 0 \\
    \text{NORMAL} & \text{otherwise}
\end{cases}
''',
        output_columns=[
            'entity_id', 'window_id', 'timestamp_start',
            'metric_source', 'metric_name', 'value',
            'baseline_mean', 'baseline_p05', 'baseline_p95',
            'percentile_rank', 'is_anomaly', 'anomaly_severity',
        ],
        params={'deviation_threshold': 0.5, 'critical_threshold': 1.0},
        min_rows=1,
    ),

    # -------------------------------------------------------------------------
    # Engine 136: Fleet Engine
    # Rankings, clustering, cohort analysis across entities
    # -------------------------------------------------------------------------
    'fleet_engine': EngineSpec(
        name='fleet_engine',
        granularity=Granularity.OBSERVATION,
        pillar=None,  # Cross-pillar statistics
        function='prism.engines.python.statistics.fleet_engine.run_fleet_engine',
        description='Fleet-wide analytics: entity rankings, clustering, cohort analysis, '
                    'and comparative statistics across all entities.',
        equation=r'''
\text{Rank}_i = \text{order}(\bar{h}_i) \quad \text{(by avg health)}
\text{Cluster} = k\text{-means}(\text{health}, \text{volatility}, \text{critical\_events})
\text{Tier} = \begin{cases}
    \text{HEALTHY} & \bar{h} \geq 80 \\
    \text{MODERATE} & \bar{h} \geq 60 \\
    \text{AT\_RISK} & \bar{h} \geq 40 \\
    \text{CRITICAL} & \bar{h} < 40
\end{cases}
''',
        output_columns=[
            'entity_id', 'health_rank', 'risk_rank',
            'avg_health', 'min_health', 'max_health', 'latest_health',
            'health_volatility', 'critical_events', 'high_events',
            'cluster', 'health_tier',
            'total_anomalies', 'critical_anomalies', 'warning_anomalies',
        ],
        params={'n_clusters': 3, 'ranking_metric': 'health_score'},
        min_rows=1,
    ),

    # -------------------------------------------------------------------------
    # Engine 137: Summary Engine
    # Generates executive summaries and reports
    # -------------------------------------------------------------------------
    'summary_engine': EngineSpec(
        name='summary_engine',
        granularity=Granularity.OBSERVATION,
        pillar=None,  # Cross-pillar statistics
        function='prism.engines.python.statistics.summary_engine.run_summary_engine',
        description='Generate executive summaries and reports. '
                    'Produces text summaries, key findings, and recommendations.',
        equation=r'''
\text{Report} = \text{Executive Summary} + \text{Critical Alerts} +
\text{Top Concerns} + \text{Anomaly Summary} + \text{Rankings} +
\text{Recommendations} + \text{Trends}
''',
        output_columns=[
            'report_section', 'content', 'priority',
        ],
        params={'include_recommendations': True},
        min_rows=1,
    ),
}


# =============================================================================
# COMBINED ENGINE SPECS
# =============================================================================

ENGINE_SPECS: Dict[str, EngineSpec] = {
    **Y1_PRIMITIVES,
    **Y2_PRIMITIVES,
    **Y3_PRIMITIVES,
    **Y4_PRIMITIVES,
    **Y5_PRIMITIVES,
    **Y6_PRIMITIVES,
    **Y7_PRIMITIVES,
    **Y8_PRIMITIVES,
    **Y9_PRIMITIVES,
    **Y10_STRUCTURE_ENGINES,
    **Y11_PHYSICS_ENGINES,
    **Y12_DYNAMICS_ENGINES,
    **Y13_ADVANCED_ENGINES,
    **Y14_STATISTICS_ENGINES,
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


def get_engines_by_pillar(pillar: Pillar) -> List[EngineSpec]:
    """Get all engines for a specific pillar."""
    return [spec for spec in ENGINE_SPECS.values() if spec.pillar == pillar]


def get_engines_by_granularity(granularity: Granularity) -> List[EngineSpec]:
    """Get all engines with a specific granularity."""
    return [spec for spec in ENGINE_SPECS.values() if spec.granularity == granularity]


def get_primitives() -> List[EngineSpec]:
    """Get all primitives (Y1-Y9)."""
    primitives = {}
    primitives.update(Y1_PRIMITIVES)
    primitives.update(Y2_PRIMITIVES)
    primitives.update(Y3_PRIMITIVES)
    primitives.update(Y4_PRIMITIVES)
    primitives.update(Y5_PRIMITIVES)
    primitives.update(Y6_PRIMITIVES)
    primitives.update(Y7_PRIMITIVES)
    primitives.update(Y8_PRIMITIVES)
    primitives.update(Y9_PRIMITIVES)
    return list(primitives.values())


def get_composed_engines() -> List[EngineSpec]:
    """Get all composed engines (Y10-Y14)."""
    engines = {}
    engines.update(Y10_STRUCTURE_ENGINES)
    engines.update(Y11_PHYSICS_ENGINES)
    engines.update(Y12_DYNAMICS_ENGINES)
    engines.update(Y13_ADVANCED_ENGINES)
    engines.update(Y14_STATISTICS_ENGINES)
    return list(engines.values())


def get_category_for_unit(unit: str) -> str:
    """Get category for a unit string."""
    if unit is None:
        return 'unknown'
    return UNIT_TO_CATEGORY.get(unit, 'unknown')


def get_all_categories() -> Set[str]:
    """Get set of all known categories."""
    return set(UNIT_TO_CATEGORY.values())


def engine_count() -> Dict[str, int]:
    """Get count of engines by category."""
    return {
        'Y1_individual': len(Y1_PRIMITIVES),
        'Y2_pairwise': len(Y2_PRIMITIVES),
        'Y3_matrix': len(Y3_PRIMITIVES),
        'Y4_embedding': len(Y4_PRIMITIVES),
        'Y5_topology': len(Y5_PRIMITIVES),
        'Y6_network': len(Y6_PRIMITIVES),
        'Y7_dynamical': len(Y7_PRIMITIVES),
        'Y8_test': len(Y8_PRIMITIVES),
        'Y9_information': len(Y9_PRIMITIVES),
        'Y10_structure': len(Y10_STRUCTURE_ENGINES),
        'Y11_physics': len(Y11_PHYSICS_ENGINES),
        'Y12_dynamics': len(Y12_DYNAMICS_ENGINES),
        'Y13_advanced': len(Y13_ADVANCED_ENGINES),
        'Y14_statistics': len(Y14_STATISTICS_ENGINES),
        'total_primitives': sum([
            len(Y1_PRIMITIVES), len(Y2_PRIMITIVES), len(Y3_PRIMITIVES),
            len(Y4_PRIMITIVES), len(Y5_PRIMITIVES), len(Y6_PRIMITIVES),
            len(Y7_PRIMITIVES), len(Y8_PRIMITIVES), len(Y9_PRIMITIVES),
        ]),
        'total_engines': sum([
            len(Y10_STRUCTURE_ENGINES), len(Y11_PHYSICS_ENGINES),
            len(Y12_DYNAMICS_ENGINES), len(Y13_ADVANCED_ENGINES),
            len(Y14_STATISTICS_ENGINES),
        ]),
        'total': len(ENGINE_SPECS),
    }


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'Granularity',
    'Pillar',
    'EngineSpec',
    'UNIT_TO_CATEGORY',
    'ENGINE_SPECS',
    'Y1_PRIMITIVES',
    'Y2_PRIMITIVES',
    'Y3_PRIMITIVES',
    'Y4_PRIMITIVES',
    'Y5_PRIMITIVES',
    'Y6_PRIMITIVES',
    'Y7_PRIMITIVES',
    'Y8_PRIMITIVES',
    'Y9_PRIMITIVES',
    'Y10_STRUCTURE_ENGINES',
    'Y11_PHYSICS_ENGINES',
    'Y12_DYNAMICS_ENGINES',
    'Y13_ADVANCED_ENGINES',
    'Y14_STATISTICS_ENGINES',
    'get_engines_for_categories',
    'get_universal_engines',
    'get_domain_engines',
    'get_engines_by_pillar',
    'get_engines_by_granularity',
    'get_primitives',
    'get_composed_engines',
    'get_category_for_unit',
    'get_all_categories',
    'engine_count',
]
