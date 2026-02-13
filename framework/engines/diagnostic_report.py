"""
Diagnostic Report Generator

Combines all RUDDER engines into a unified diagnostic assessment.

Process flow:
1. System Typology (Level 0)
2. Stationarity (Level 1)
3. Signal Classification (Level 2)
4. Geometry
5. Mass
6. Structure = Geometry × Mass
7. Stability
8. Tipping Classification
9. Spin Glass Mapping
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
import polars as pl

from .typology_engine import classify_system_type, SystemType, interpret_typology_for_domain
from .stationarity_engine import test_stationarity
from .classification_engine import classify_signal
from .signal_geometry import compute_geometry, compute_geometry_trajectory, classify_geometry_state
from .mass_engine import compute_mass, compute_mass_trajectory
from .structure_engine import compute_structure, compute_structure_trajectory, interpret_coupling
from .stability_engine import compute_stability, compute_csd_indicators
from .tipping_engine import classify_tipping, interpret_tipping_for_domain
from .spin_glass import compute_spin_glass, generate_spin_glass_report


@dataclass
class DiagnosticResult:
    """Complete diagnostic assessment."""
    # Identity
    domain: str
    n_signals: int
    n_windows: int

    # Level 0: Typology
    system_type: str
    typology_confidence: float
    typology_description: str

    # Level 1/2: Signal summary
    n_stationary: int
    n_nonstationary: int
    signal_classes: Dict[str, int]

    # Geometry
    eff_dim: float
    eff_dim_ratio: float
    alignment: float
    geometry_zone: str
    geometry_description: str
    geometry_trend: float
    is_collapsing: bool

    # Mass
    total_variance: float
    mass_trend: str
    geometry_mass_coupling: float

    # Structure
    compression: float
    absorption: float
    structure_state: str
    structure_description: str

    # Stability
    n_unstable: int
    n_csd_detected: int
    mean_lyapunov: float
    bifurcation_proximity: float

    # Tipping
    tipping_type: str
    warning_available: bool
    warning_mechanism: str

    # Spin Glass
    phase: str
    overlap_q: float
    magnetization_m: float
    resilience: str
    shock_response: str

    # Overall
    health_score: float  # 0-1
    risk_level: str      # low/moderate/high/critical


def run_diagnostic(
    X: np.ndarray,
    domain: str = "general",
    window_size: int = 100,
    stride: Optional[int] = None,
) -> DiagnosticResult:
    """
    Run complete RUDDER diagnostic on signal matrix.

    Args:
        X: Signal matrix (n_timepoints × n_signals)
        domain: Domain identifier for interpretation
        window_size: Window size for rolling analysis
        stride: Stride between windows (default: window_size // 2)

    Returns:
        DiagnosticResult with complete assessment
    """
    if stride is None:
        stride = window_size // 2

    n_timepoints, n_signals = X.shape

    # === CREATE WINDOWS ===
    windows = []
    for start in range(0, n_timepoints - window_size + 1, stride):
        windows.append(X[start:start + window_size])

    n_windows = len(windows)
    if n_windows < 2:
        raise ValueError(f"Insufficient data: need at least 2 windows, got {n_windows}")

    # === LEVEL 1/2: Per-signal analysis ===
    n_stationary = 0
    n_nonstationary = 0
    signal_classes = {}
    signal_trends = []

    for j in range(n_signals):
        sig = X[:, j]

        # Stationarity
        stat_result = test_stationarity(sig)
        if stat_result.classification.value == "stationary":
            n_stationary += 1
        else:
            n_nonstationary += 1

        # Classification
        class_result = classify_signal(sig, stat_result.classification.value == "stationary")
        signal_class = class_result.signal_class.value
        signal_classes[signal_class] = signal_classes.get(signal_class, 0) + 1

        # Trend for magnetization
        t = np.arange(len(sig))
        slope = np.polyfit(t, sig, 1)[0] if len(t) > 1 else 0.0
        signal_trends.append(slope)

    # === GEOMETRY AND MASS PER WINDOW ===
    geometries = []
    masses = []
    ar1s = []
    variances = []

    for window in windows:
        geo = compute_geometry(window)
        geometries.append(geo)

        mass = compute_mass(window)
        masses.append(mass)

        # AR1 for CSD
        for j in range(n_signals):
            sig = window[:, j]
            if np.std(sig) > 1e-10 and len(sig) > 1:
                ar1 = np.corrcoef(sig[:-1], sig[1:])[0, 1]
                if np.isfinite(ar1):
                    ar1s.append(ar1)

        variances.append(mass.total_variance)

    # === GEOMETRY TRAJECTORY ===
    geo_trajectory = compute_geometry_trajectory(geometries)

    # === MASS TRAJECTORY ===
    mass_trajectory = compute_mass_trajectory(masses)

    # === STRUCTURE ===
    latest_geo = geometries[-1]
    latest_mass = masses[-1]
    structure = compute_structure(latest_geo, latest_mass)

    structure_trajectory = compute_structure_trajectory(geometries, masses)

    # === LEVEL 0: TYPOLOGY ===
    eff_dim_series = np.array([g.eff_dim for g in geometries])
    mass_series = np.array([m.total_variance for m in masses])

    typology = classify_system_type(mass_series, eff_dim_series)

    # === STABILITY ===
    # Aggregate stability metrics
    stability_results = []
    for j in range(n_signals):
        sig = X[:, j]
        stab = compute_stability(sig)
        stability_results.append(stab)

    n_unstable = sum(1 for s in stability_results if not s.is_stable)
    n_csd = sum(1 for s in stability_results if s.csd_detected)
    mean_lyapunov = np.mean([s.lyapunov_exponent for s in stability_results])

    # CSD indicators
    csd_result = compute_csd_indicators(ar1s[-min(20, len(ar1s)):], variances[-min(20, len(variances)):])

    # Bifurcation proximity (based on AR1 approaching 1)
    latest_ar1 = csd_result.get('latest_ar1', 0.0)
    bifurcation_proximity = max(0, (latest_ar1 - 0.5) / 0.5) if latest_ar1 else 0.0

    # === TIPPING CLASSIFICATION ===
    tipping = classify_tipping(
        eff_dim_series,
        mass_series,
        csd_detected=n_csd > 0 or csd_result.get('csd_detected', False),
    )

    # === SPIN GLASS ===
    # Compute correlation matrix from latest window
    corr_matrix = np.corrcoef(windows[-1].T)
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

    spin_glass = compute_spin_glass(
        geometry_eff_dim=latest_geo.eff_dim,
        n_signals=n_signals,
        signal_trends=signal_trends,
        correlation_matrix=corr_matrix,
        csd_detected=n_csd > 0 or csd_result.get('csd_detected', False),
    )

    # === HEALTH SCORE ===
    # Combine multiple factors
    geo_score = latest_geo.eff_dim_ratio
    stability_score = 1 - (n_unstable + n_csd) / (2 * n_signals)
    structure_score = structure.absorption
    pattern_score = (spin_glass.absorption + 1) / 2

    health_score = 0.3 * geo_score + 0.25 * stability_score + 0.25 * structure_score + 0.2 * pattern_score
    health_score = np.clip(health_score, 0, 1)

    # Risk level
    if health_score >= 0.7:
        risk_level = "low"
    elif health_score >= 0.5:
        risk_level = "moderate"
    elif health_score >= 0.3:
        risk_level = "high"
    else:
        risk_level = "critical"

    # === GEOMETRY STATE ===
    geo_state = classify_geometry_state(latest_geo)

    return DiagnosticResult(
        domain=domain,
        n_signals=n_signals,
        n_windows=n_windows,
        # Typology
        system_type=typology.system_type.value,
        typology_confidence=typology.confidence,
        typology_description=interpret_typology_for_domain(typology, domain),
        # Signals
        n_stationary=n_stationary,
        n_nonstationary=n_nonstationary,
        signal_classes=signal_classes,
        # Geometry
        eff_dim=latest_geo.eff_dim,
        eff_dim_ratio=latest_geo.eff_dim_ratio,
        alignment=latest_geo.alignment,
        geometry_zone=geo_state['zone'],
        geometry_description=geo_state['description'],
        geometry_trend=geo_trajectory['eff_dim_trend'],
        is_collapsing=geo_trajectory['is_collapsing'],
        # Mass
        total_variance=latest_mass.total_variance,
        mass_trend=mass_trajectory['mass_trend'],
        geometry_mass_coupling=structure_trajectory['geometry_mass_coupling'],
        # Structure
        compression=structure.compression,
        absorption=structure.absorption,
        structure_state=structure.state.value,
        structure_description=structure.description,
        # Stability
        n_unstable=n_unstable,
        n_csd_detected=n_csd,
        mean_lyapunov=float(mean_lyapunov),
        bifurcation_proximity=float(bifurcation_proximity),
        # Tipping
        tipping_type=tipping.tipping_type.value,
        warning_available=tipping.warning_available,
        warning_mechanism=tipping.warning_mechanism,
        # Spin Glass
        phase=spin_glass.phase.value,
        overlap_q=spin_glass.overlap_q,
        magnetization_m=spin_glass.magnetization_m,
        resilience=spin_glass.resilience,
        shock_response=spin_glass.shock_response,
        # Overall
        health_score=float(health_score),
        risk_level=risk_level,
    )


def generate_report(result: DiagnosticResult) -> str:
    """
    Generate human-readable diagnostic report.

    Args:
        result: DiagnosticResult from run_diagnostic

    Returns:
        Formatted report string
    """
    # Geometry zone colors
    zone_colors = {
        'healthy': '🟢',
        'caution': '🟡',
        'stress': '🟠',
        'crisis': '🔴',
    }
    geo_color = zone_colors.get(result.geometry_zone, '⚪')

    # Risk colors
    risk_colors = {
        'low': '🟢',
        'moderate': '🟡',
        'high': '🟠',
        'critical': '🔴',
    }
    risk_color = risk_colors.get(result.risk_level, '⚪')

    report = f"""
════════════════════════════════════════════════════════════════════════════════
RUDDER DIAGNOSTIC ASSESSMENT
════════════════════════════════════════════════════════════════════════════════

SYSTEM IDENTIFICATION
├── Domain:           {result.domain}
├── Type:             {result.system_type.upper()}
├── Signals:          N = {result.n_signals}
├── Windows:          {result.n_windows}
└── Confidence:       {result.typology_confidence:.0%}

{result.typology_description}

────────────────────────────────────────────────────────────────────────────────

SIGNAL ANALYSIS
├── Stationary:       {result.n_stationary} / {result.n_signals}
├── Non-stationary:   {result.n_nonstationary} / {result.n_signals}
└── Classes:          {', '.join(f'{k}({v})' for k, v in result.signal_classes.items())}

GEOMETRY (Eigenstructure)
├── eff_dim:          {result.eff_dim:.2f} / {result.n_signals}  ({result.eff_dim_ratio:.0%})
├── Alignment:        {result.alignment:.2f} (PC1 dominance)
├── Trend:            {result.geometry_trend:+.3f} per window
├── Zone:             {geo_color} {result.geometry_zone.upper()}
└── Status:           {"⚠️ COLLAPSING" if result.is_collapsing else "Stable"}

{result.geometry_description}

MASS (Energy)
├── Total:            {result.total_variance:.4e}
├── Trend:            {result.mass_trend.upper()}
└── Geo-Mass r:       {result.geometry_mass_coupling:+.3f}

STRUCTURE (Geometry × Mass)
├── Compression:      {result.compression:.4e} (energy per DOF)
├── Absorption:       {result.absorption:.0%} remaining capacity
├── State:            {result.structure_state.upper()}
└── Assessment:       {result.structure_description}

────────────────────────────────────────────────────────────────────────────────

STABILITY
├── Unstable signals: {result.n_unstable} / {result.n_signals}
├── CSD detected:     {result.n_csd_detected} / {result.n_signals}
├── Mean λ:           {result.mean_lyapunov:+.4f}
└── Bifurcation:      {result.bifurcation_proximity:.0%} proximity

TIPPING CLASSIFICATION
├── Type:             {result.tipping_type.upper().replace('_', '-')}
├── Warning:          {"YES" if result.warning_available else "NO"}
└── Mechanism:        {result.warning_mechanism}

────────────────────────────────────────────────────────────────────────────────

SPIN GLASS INTERPRETATION
├── Phase:            {result.phase.upper().replace('_', ' ')}
├── Overlap q:        {result.overlap_q:.3f}
├── Magnetization m:  {result.magnetization_m:+.3f}
├── Resilience:       {result.resilience}
└── Shock response:   {result.shock_response}

────────────────────────────────────────────────────────────────────────────────

OVERALL ASSESSMENT
├── Health Score:     {result.health_score:.0%}
└── Risk Level:       {risk_color} {result.risk_level.upper()}

════════════════════════════════════════════════════════════════════════════════
"""
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RUDDER Diagnostic Report")
    parser.add_argument('input', help='Input observations parquet')
    parser.add_argument('--domain', default='general', help='Domain for interpretation')
    parser.add_argument('--window-size', type=int, default=100, help='Window size')
    parser.add_argument('--output', help='Output report file')

    args = parser.parse_args()

    # Load data
    df = pl.read_parquet(args.input)

    # Pivot to matrix
    if 'signal_id' in df.columns and 'value' in df.columns and 'I' in df.columns:
        matrix = df.pivot(on='signal_id', index='I', values='value')
        signals = [c for c in matrix.columns if c != 'I']
        X = matrix.select(signals).to_numpy()
    else:
        raise ValueError("Expected columns: signal_id, value, I")

    print(f"Loaded: {X.shape[0]} timepoints × {X.shape[1]} signals")

    # Run diagnostic
    result = run_diagnostic(X, domain=args.domain, window_size=args.window_size)

    # Generate report
    report = generate_report(result)
    print(report)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
