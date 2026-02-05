"""
Cohort Discovery Module v2

Automatically detects independent subdomains from geometry.
Identifies operational constants, system-level coupling, and component-level signals.

CANONICAL PRINCIPLE:
    "Run PRISM FIRST - geometry reveals physics you didn't know."

NEW IN V2:
    - Constant detection (operational settings with rho > 0.99)
    - Four-class classification: CONSTANT, SYSTEM, COMPONENT, ORPHAN
    - ML-ready output (get_ml_signals, get_exclude_list)
    - Within-unit coupling analysis (coupled vs decoupled units)

Example findings:
    FEMTO: acc_y coupled across bearings (test rig), acc_x independent
    C-MAPSS: 3 of 21 "sensors" are operational constants

Usage:
    from orthon.cohorts.detection import CohortDiscovery, process_observations

    cd = CohortDiscovery(observations_path)
    result = cd.discover()

    # Get ML-ready signal list
    ml_signals = result.get_ml_signals()  # excludes constants
    exclude_list = result.get_exclude_list()  # constants + orphans
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import polars as pl
import numpy as np
import yaml


# =============================================================================
# THRESHOLDS
# =============================================================================

CONSTANT_THRESHOLD = 0.99    # Above this = operational constant (exclude)
SYSTEM_THRESHOLD = 0.70      # Above this = system-level coupling
COMPONENT_THRESHOLD = 0.50   # Below this = component-level (independent)
COUPLING_THRESHOLD = 0.70    # For backward compatibility
WITHIN_UNIT_THRESHOLD = 0.70 # For coupled/decoupled unit classification


# =============================================================================
# ENUMS AND TYPES
# =============================================================================

class CohortType(Enum):
    """Classification of signal/cohort coupling."""
    CONSTANT = "constant"
    """Operational constant - same value across all units, exclude from analysis."""

    SYSTEM = "system"
    """Strongly coupled signals across units - use as fleet baseline."""

    COMPONENT = "component"
    """Unit-specific signals - USE FOR RUL/FAULT DETECTION."""

    ORPHAN = "orphan"
    """Uncorrelated with everything - investigate."""

    COUPLED_PAIR = "coupled_pair"
    """Two signals that move together within a unit (isotropic degradation)."""

    DECOUPLED_PAIR = "decoupled_pair"
    """Two signals that move independently within a unit (anisotropic fault)."""


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SignalClassification:
    """Classification of a single signal."""
    signal_id: str
    classification: str  # 'constant', 'system', 'component', 'orphan'
    cross_unit_correlation: float
    within_unit_correlation: Optional[float] = None
    cohort_id: Optional[int] = None
    notes: str = ""


@dataclass
class ConstantDetectionResult:
    """Result of constant/operational setting detection (v1 compatibility)."""
    constants: List[str]
    """Signals identified as constants (rho > 0.99 across units)."""

    valid_signals: List[str]
    """Signals that vary across units (use for analysis)."""

    cross_unit_correlations: Dict[str, float]
    """Mean cross-unit correlation per signal."""

    threshold: float
    """Threshold used for constant detection."""


@dataclass
class CohortResult:
    """Result of cohort discovery."""

    # Signal classifications (v2)
    constants: List[str] = field(default_factory=list)
    """Signals with rho >= 0.99 across units (operational settings)."""

    system_signals: List[str] = field(default_factory=list)
    """Signals with rho >= 0.70 across units (fleet-coupled)."""

    component_signals: List[str] = field(default_factory=list)
    """Signals with rho < 0.50 across units (unit-specific, USE FOR RUL)."""

    orphan_signals: List[str] = field(default_factory=list)
    """Signals with rho ~ 0 (uncorrelated, investigate)."""

    # Detailed classifications
    signal_details: Dict[str, SignalClassification] = field(default_factory=dict)

    # Unit-level analysis (for multi-signal units like FEMTO)
    coupled_units: List[str] = field(default_factory=list)
    """Units with high within-unit signal correlation (isotropic degradation)."""

    decoupled_units: List[str] = field(default_factory=list)
    """Units with low within-unit signal correlation (anisotropic faults)."""

    # Graph structure (v1 compatibility)
    n_cohorts: int = 0
    cohort_membership: Dict[str, int] = field(default_factory=dict)
    cohort_contents: Dict[int, List[str]] = field(default_factory=dict)
    cohorts: List[set] = field(default_factory=list)
    cohort_types: Dict[int, CohortType] = field(default_factory=dict)

    # Correlations
    cross_unit_corr_matrix: Optional[Dict[str, float]] = None
    cross_signal_correlation: Dict[tuple, float] = field(default_factory=dict)
    within_unit_coupling: Dict[str, float] = field(default_factory=dict)

    # Thresholds used
    threshold: float = COUPLING_THRESHOLD

    # Metadata
    metadata: Dict = field(default_factory=dict)

    def get_ml_signals(self) -> List[str]:
        """Get signals suitable for ML (excludes constants and orphans)."""
        return self.component_signals + self.system_signals

    def get_exclude_list(self) -> List[str]:
        """Get signals to exclude from ML."""
        return self.constants + self.orphan_signals

    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'constants': self.constants,
            'system_signals': self.system_signals,
            'component_signals': self.component_signals,
            'orphan_signals': self.orphan_signals,
            'coupled_units': self.coupled_units,
            'decoupled_units': self.decoupled_units,
            'n_cohorts': self.n_cohorts,
            'ml_signals': self.get_ml_signals(),
            'exclude_list': self.get_exclude_list(),
            'signal_details': {
                k: {
                    'classification': v.classification,
                    'cross_unit_correlation': v.cross_unit_correlation,
                    'notes': v.notes,
                }
                for k, v in self.signal_details.items()
            },
            'metadata': self.metadata,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "COHORT DISCOVERY RESULTS",
            "=" * 60,
            "",
            f"CONSTANTS ({len(self.constants)}): {self.constants}",
            f"  -> Operational settings, EXCLUDE from analysis",
            "",
            f"SYSTEM SIGNALS ({len(self.system_signals)}): {self.system_signals[:5]}{'...' if len(self.system_signals) > 5 else ''}",
            f"  -> Coupled across units, use for fleet baseline",
            "",
            f"COMPONENT SIGNALS ({len(self.component_signals)}): {self.component_signals[:5]}{'...' if len(self.component_signals) > 5 else ''}",
            f"  -> Unit-specific, USE FOR RUL/FAULT DETECTION",
            "",
            f"ORPHANS ({len(self.orphan_signals)}): {self.orphan_signals}",
            f"  -> Low correlation with everything, investigate",
            "",
            "=" * 60,
            "ML RECOMMENDATIONS",
            "=" * 60,
            f"Use these {len(self.get_ml_signals())} signals: {self.get_ml_signals()[:10]}{'...' if len(self.get_ml_signals()) > 10 else ''}",
            f"Exclude these {len(self.get_exclude_list())} signals: {self.get_exclude_list()}",
            "",
        ]

        if self.coupled_units or self.decoupled_units:
            lines.extend([
                "=" * 60,
                "UNIT COUPLING (within-unit signal correlation)",
                "=" * 60,
                f"COUPLED UNITS ({len(self.coupled_units)}): {self.coupled_units[:5]}{'...' if len(self.coupled_units) > 5 else ''}",
                f"  -> Isotropic degradation, signals move together",
                f"DECOUPLED UNITS ({len(self.decoupled_units)}): {self.decoupled_units[:5]}{'...' if len(self.decoupled_units) > 5 else ''}",
                f"  -> Anisotropic/localized faults, signals independent",
                "",
            ])

        return "\n".join(lines)


# =============================================================================
# MAIN CLASS
# =============================================================================

class CohortDiscovery:
    """
    Discover signal structure from observations.

    Identifies:
        1. CONSTANTS: Operational settings (rho > 0.99 across units)
        2. SYSTEM: Signals coupled across units (rho > 0.7)
        3. COMPONENT: Unit-specific signals (rho < 0.5) - USE FOR ML
        4. ORPHANS: Uncorrelated signals (investigate)

    For multi-signal units (like FEMTO bearings):
        - COUPLED: Signals within unit move together (isotropic)
        - DECOUPLED: Signals within unit are independent (anisotropic/localized fault)
    """

    def __init__(
        self,
        observations_path: Optional[Path] = None,
        manifest_path: Optional[Path] = None,
        constant_threshold: float = CONSTANT_THRESHOLD,
        system_threshold: float = SYSTEM_THRESHOLD,
        component_threshold: float = COMPONENT_THRESHOLD,
    ):
        """
        Initialize cohort discovery.

        Args:
            observations_path: Path to observations.parquet
            manifest_path: Path to manifest.yaml (optional, for metadata)
            constant_threshold: Correlation threshold for constants (default 0.99)
            system_threshold: Correlation threshold for system signals (default 0.70)
            component_threshold: Correlation threshold for component signals (default 0.50)
        """
        self.observations_path = Path(observations_path) if observations_path else None
        self.manifest_path = Path(manifest_path) if manifest_path else None

        self.constant_threshold = constant_threshold
        self.system_threshold = system_threshold
        self.component_threshold = component_threshold

        self._observations: Optional[pl.DataFrame] = None
        self._manifest: Optional[Dict] = None
        self._data_format: Optional[Dict] = None

    def load_observations(self) -> pl.DataFrame:
        """Load observations data."""
        if self._observations is None:
            if self.observations_path is None:
                raise ValueError("No observations_path provided")
            self._observations = pl.read_parquet(self.observations_path)
        return self._observations

    def load_manifest(self) -> Optional[Dict]:
        """Load manifest if available."""
        if self._manifest is None and self.manifest_path and self.manifest_path.exists():
            with open(self.manifest_path) as f:
                self._manifest = yaml.safe_load(f)
        return self._manifest

    def _detect_data_format(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Detect the format of the observations data.

        Handles various column naming conventions:
            - unit_id, entity_id, unit, engine_id
            - signal_id, signal, sensor
            - I, timestamp, time, cycle
            - value, reading

        Returns dict with:
            - format: 'long' or 'wide'
            - signal_col: column containing signal names (if long)
            - value_col: column containing values (if long)
            - unit_col: column containing unit IDs (if present)
            - time_col: column containing timestamps/indices
            - signal_cols: list of signal columns (if wide)
        """
        if self._data_format is not None:
            return self._data_format

        cols = set(df.columns)

        # Detect unit column (various naming conventions)
        unit_candidates = ['unit_id', 'entity_id', 'unit', 'engine_id', 'bearing_id']
        unit_col = next((c for c in unit_candidates if c in cols), None)

        # Detect time/index column
        time_candidates = ['I', 'timestamp', 'time', 'cycle', 'index']
        time_col = next((c for c in time_candidates if c in cols), None)

        # Check for long format indicators
        signal_candidates = ['signal_id', 'signal', 'sensor', 'sensor_id', 'variable']
        signal_col = next((c for c in signal_candidates if c in cols), None)

        value_candidates = ['value', 'reading', 'measurement']
        value_col = next((c for c in value_candidates if c in cols), None)

        if signal_col and value_col:
            self._data_format = {
                'format': 'long',
                'signal_col': signal_col,
                'value_col': value_col,
                'unit_col': unit_col,
                'time_col': time_col,
            }
            return self._data_format

        # Assume wide format
        meta_cols = {'timestamp', 'time', 'cycle', 'I', 'unit_id', 'entity_id',
                     'cohort_id', 'window', 'observation_id', 'RUL', 'rul'}
        signal_cols = [c for c in df.columns
                      if c not in meta_cols
                      and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        self._data_format = {
            'format': 'wide',
            'signal_cols': signal_cols,
            'unit_col': unit_col,
            'time_col': time_col,
        }
        return self._data_format

    def compute_cross_unit_correlations(self) -> Dict[str, float]:
        """
        Compute mean cross-unit correlation for each signal.

        For each signal, computes correlation between all pairs of units
        and returns the mean. High correlation (> 0.99) indicates an
        operational constant, not a sensor.

        Returns:
            Dict mapping signal_id to mean cross-unit correlation
        """
        df = self.load_observations()
        fmt = self._detect_data_format(df)

        if fmt['unit_col'] is None:
            # No unit column - can't compute cross-unit correlation
            return {}

        unit_col = fmt['unit_col']
        units = df[unit_col].unique().to_list()

        if len(units) < 2:
            return {}

        cross_unit_corrs = {}

        if fmt['format'] == 'long':
            signal_col = fmt['signal_col']
            value_col = fmt['value_col']
            time_col = fmt['time_col']

            signals = df[signal_col].unique().to_list()

            for signal in signals:
                signal_data = df.filter(pl.col(signal_col) == signal)

                # Pivot to get units as columns
                try:
                    pivoted = signal_data.pivot(
                        index=time_col,
                        on=unit_col,
                        values=value_col
                    ).drop_nulls()
                except Exception:
                    continue

                if len(pivoted) < 10:
                    cross_unit_corrs[signal] = 0.0
                    continue

                # Compute pairwise correlations between units
                unit_cols = [c for c in pivoted.columns if c != time_col]
                if len(unit_cols) < 2:
                    continue

                data = pivoted.select(unit_cols).to_numpy()
                data = np.nan_to_num(data, nan=0.0)

                # Check for constant columns (zero variance)
                stds = np.std(data, axis=0)
                if np.all(stds == 0):
                    # All constant = perfectly correlated
                    cross_unit_corrs[signal] = 1.0
                    continue

                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_matrix = np.abs(np.corrcoef(data.T))
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                # Extract upper triangle (excluding diagonal)
                n = len(unit_cols)
                correlations = []
                for i in range(n):
                    for j in range(i + 1, n):
                        correlations.append(corr_matrix[i, j])

                if correlations:
                    cross_unit_corrs[signal] = float(np.mean(correlations))
                else:
                    cross_unit_corrs[signal] = 0.0

        else:  # wide format
            signal_cols = fmt['signal_cols']
            time_col = fmt['time_col']

            for signal in signal_cols:
                try:
                    pivoted = df.select([time_col, unit_col, signal]).pivot(
                        index=time_col,
                        on=unit_col,
                        values=signal
                    ).drop_nulls()
                except Exception:
                    continue

                if len(pivoted) < 10:
                    cross_unit_corrs[signal] = 0.0
                    continue

                unit_cols_piv = [c for c in pivoted.columns if c != time_col]
                if len(unit_cols_piv) < 2:
                    continue

                data = pivoted.select(unit_cols_piv).to_numpy()
                data = np.nan_to_num(data, nan=0.0)

                stds = np.std(data, axis=0)
                if np.all(stds == 0):
                    cross_unit_corrs[signal] = 1.0
                    continue

                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_matrix = np.abs(np.corrcoef(data.T))
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                correlations = []
                n = len(unit_cols_piv)
                for i in range(n):
                    for j in range(i + 1, n):
                        correlations.append(corr_matrix[i, j])

                if correlations:
                    cross_unit_corrs[signal] = float(np.mean(correlations))
                else:
                    cross_unit_corrs[signal] = 0.0

        return cross_unit_corrs

    def compute_within_unit_correlations(self) -> Dict[str, float]:
        """
        Compute within-unit signal correlations.

        For datasets like FEMTO where each unit has multiple signals
        (e.g., acc_x and acc_y per bearing).

        Returns:
            Dict mapping unit_id to mean within-unit correlation
        """
        df = self.load_observations()
        fmt = self._detect_data_format(df)

        if fmt['format'] != 'long':
            return {}

        unit_col = fmt['unit_col']
        if unit_col is None:
            return {}

        signal_col = fmt['signal_col']
        value_col = fmt['value_col']
        time_col = fmt['time_col']

        within_corrs = {}

        for unit in df[unit_col].unique().to_list():
            unit_data = df.filter(pl.col(unit_col) == unit)
            signals = unit_data[signal_col].unique().to_list()

            if len(signals) < 2:
                continue

            # Pivot to wide format
            try:
                pivoted = unit_data.pivot(
                    index=time_col,
                    on=signal_col,
                    values=value_col
                ).drop_nulls()
            except Exception:
                continue

            if len(pivoted) < 10:
                continue

            sig_cols = [c for c in pivoted.columns if c != time_col]
            data = pivoted.select(sig_cols).to_numpy()
            data = np.nan_to_num(data, nan=0.0)

            with np.errstate(divide='ignore', invalid='ignore'):
                corr_matrix = np.abs(np.corrcoef(data.T))
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            correlations = []
            n = len(sig_cols)
            for i in range(n):
                for j in range(i + 1, n):
                    correlations.append(corr_matrix[i, j])

            if correlations:
                within_corrs[unit] = float(np.mean(correlations))

        return within_corrs

    def classify_signals(self, cross_unit_corrs: Dict[str, float]) -> CohortResult:
        """
        Classify signals based on cross-unit correlation.

        | Class     | Cross-Unit rho | Action          |
        |-----------|---------------|-----------------|
        | CONSTANT  | >= 0.99       | EXCLUDE         |
        | SYSTEM    | >= 0.70       | Fleet baseline  |
        | COMPONENT | < 0.50        | USE FOR RUL     |
        | ORPHAN    | ~ 0           | Investigate     |

        Args:
            cross_unit_corrs: Dict mapping signal to cross-unit correlation

        Returns:
            CohortResult with classified signals
        """
        result = CohortResult()
        result.cross_unit_corr_matrix = cross_unit_corrs

        for signal, corr in cross_unit_corrs.items():
            classification = SignalClassification(
                signal_id=signal,
                classification='unknown',
                cross_unit_correlation=corr,
            )

            if corr >= self.constant_threshold:
                classification.classification = 'constant'
                classification.notes = f"rho={corr:.3f} >= {self.constant_threshold} (operational setting)"
                result.constants.append(signal)

            elif corr >= self.system_threshold:
                classification.classification = 'system'
                classification.notes = f"rho={corr:.3f} >= {self.system_threshold} (coupled across units)"
                result.system_signals.append(signal)

            elif corr >= self.component_threshold:
                classification.classification = 'partial'
                classification.notes = f"{self.component_threshold} <= rho={corr:.3f} < {self.system_threshold} (partially coupled)"
                result.component_signals.append(signal)  # Still usable for ML

            elif corr > 0.1:
                classification.classification = 'component'
                classification.notes = f"rho={corr:.3f} < {self.component_threshold} (unit-specific)"
                result.component_signals.append(signal)

            else:
                classification.classification = 'orphan'
                classification.notes = f"rho={corr:.3f} ~ 0 (uncorrelated, investigate)"
                result.orphan_signals.append(signal)

            result.signal_details[signal] = classification

        return result

    def classify_units(self, within_unit_corrs: Dict[str, float]) -> Tuple[List[str], List[str]]:
        """
        Classify units as coupled or decoupled.

        Args:
            within_unit_corrs: Dict mapping unit to within-unit correlation

        Returns:
            Tuple of (coupled_units, decoupled_units)
        """
        coupled = []
        decoupled = []

        for unit, corr in within_unit_corrs.items():
            if corr >= WITHIN_UNIT_THRESHOLD:
                coupled.append(unit)
            else:
                decoupled.append(unit)

        return coupled, decoupled

    def discover(self) -> CohortResult:
        """
        Run full cohort discovery.

        Returns:
            CohortResult with all findings
        """
        # Step 1: Compute cross-unit correlations
        cross_unit_corrs = self.compute_cross_unit_correlations()

        if not cross_unit_corrs:
            # Fall back to simpler analysis if no unit structure
            return self._discover_no_units()

        # Step 2: Classify signals
        result = self.classify_signals(cross_unit_corrs)

        # Step 3: Compute within-unit correlations (for multi-signal units)
        within_unit_corrs = self.compute_within_unit_correlations()

        if within_unit_corrs:
            result.coupled_units, result.decoupled_units = self.classify_units(within_unit_corrs)
            result.within_unit_coupling = within_unit_corrs

        # Add metadata
        df = self.load_observations()
        fmt = self._detect_data_format(df)
        result.metadata = {
            'n_signals': len(cross_unit_corrs),
            'n_units': df[fmt['unit_col']].n_unique() if fmt['unit_col'] else 1,
            'n_rows': len(df),
            'data_format': fmt['format'],
            'thresholds': {
                'constant': self.constant_threshold,
                'system': self.system_threshold,
                'component': self.component_threshold,
            }
        }

        return result

    def _discover_no_units(self) -> CohortResult:
        """
        Fallback discovery when no unit structure exists.

        Computes signal-to-signal correlations and finds clusters.
        """
        df = self.load_observations()
        fmt = self._detect_data_format(df)

        result = CohortResult()

        if fmt['format'] == 'wide':
            signal_cols = fmt['signal_cols']

            # Check for constant signals
            for col in signal_cols:
                std = df[col].std()
                if std is not None and std < 1e-10:
                    result.constants.append(col)
                    result.signal_details[col] = SignalClassification(
                        signal_id=col,
                        classification='constant',
                        cross_unit_correlation=1.0,
                        notes="Zero variance (constant)"
                    )
                else:
                    result.component_signals.append(col)
                    result.signal_details[col] = SignalClassification(
                        signal_id=col,
                        classification='component',
                        cross_unit_correlation=0.0,
                        notes="No unit structure detected"
                    )

        return result

    def generate_manifest_update(self, result: CohortResult) -> Dict:
        """
        Generate manifest updates based on discovery.

        Returns dict to merge into existing manifest.
        """
        return {
            'cohort_discovery': {
                'version': '2.0',
                'findings': result.to_dict(),
                'recommendations': {
                    'exclude_signals': result.get_exclude_list(),
                    'ml_signals': result.get_ml_signals(),
                    'coupled_units': result.coupled_units,
                    'decoupled_units': result.decoupled_units,
                },
                'thresholds_used': {
                    'constant': self.constant_threshold,
                    'system': self.system_threshold,
                    'component': self.component_threshold,
                    'within_unit': WITHIN_UNIT_THRESHOLD,
                }
            }
        }

    def save_results(self, result: CohortResult, output_dir: Path):
        """
        Save discovery results to output directory.

        Creates:
            - cohort_discovery.yaml (full results)
            - ml_signals.txt (one signal per line, for easy loading)
            - exclude_signals.txt (signals to exclude)
            - cohort_discovery_summary.txt (human-readable)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full results
        with open(output_dir / 'cohort_discovery.yaml', 'w') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False)

        # ML signals (one per line)
        with open(output_dir / 'ml_signals.txt', 'w') as f:
            for sig in result.get_ml_signals():
                f.write(f"{sig}\n")

        # Exclude signals
        with open(output_dir / 'exclude_signals.txt', 'w') as f:
            for sig in result.get_exclude_list():
                f.write(f"{sig}\n")

        # Human-readable summary
        with open(output_dir / 'cohort_discovery_summary.txt', 'w') as f:
            f.write(result.summary())

        print(f"Results saved to {output_dir}")
        print(result.summary())


# =============================================================================
# V1 COMPATIBILITY FUNCTIONS
# =============================================================================

def should_run_cohort_discovery(
    observations: pl.DataFrame,
    signal_col: str = "signal_id",
    unit_col: str = "unit_id",
) -> Tuple[bool, str]:
    """
    Determine if cohort discovery should be run on this dataset.

    ALWAYS RUN WHEN:
    - > 20 signals
    - > 1M rows
    - > 5 units
    - Unknown data source

    Args:
        observations: DataFrame with observations
        signal_col: Column name for signal identifier
        unit_col: Column name for unit identifier

    Returns:
        (should_run, reason)
    """
    # Try to detect actual column names
    cols = set(observations.columns)
    signal_col = signal_col if signal_col in cols else next(
        (c for c in ['signal_id', 'signal', 'sensor'] if c in cols), signal_col
    )
    unit_col = unit_col if unit_col in cols else next(
        (c for c in ['unit_id', 'entity_id', 'unit'] if c in cols), unit_col
    )

    n_rows = len(observations)
    n_signals = observations[signal_col].n_unique() if signal_col in cols else 0
    n_units = observations[unit_col].n_unique() if unit_col in cols else 1

    reasons = []

    if n_signals > 20:
        reasons.append(f"{n_signals} signals (> 20)")

    if n_rows > 1_000_000:
        reasons.append(f"{n_rows:,} rows (> 1M)")

    if n_units > 5:
        reasons.append(f"{n_units} units (> 5)")

    if reasons:
        return True, f"RECOMMENDED: {', '.join(reasons)}"
    else:
        return False, f"Optional: {n_signals} signals, {n_rows:,} rows, {n_units} units"


def detect_constants(
    observations: pl.DataFrame,
    threshold: float = CONSTANT_THRESHOLD,
    signal_col: str = "signal_id",
    unit_col: str = "unit_id",
    value_col: str = "value",
    index_col: str = "I",
    **kwargs,
) -> ConstantDetectionResult:
    """
    Detect signals that are constants/operational settings.

    V1 compatibility wrapper around CohortDiscovery.
    """
    # Create temporary parquet for CohortDiscovery
    cd = CohortDiscovery(constant_threshold=threshold)
    cd._observations = observations
    cd._data_format = None  # Reset to auto-detect

    cross_unit_corrs = cd.compute_cross_unit_correlations()

    constants = []
    valid_signals = []

    for signal, corr in cross_unit_corrs.items():
        if corr >= threshold:
            constants.append(signal)
        else:
            valid_signals.append(signal)

    return ConstantDetectionResult(
        constants=constants,
        valid_signals=valid_signals,
        cross_unit_correlations=cross_unit_corrs,
        threshold=threshold,
    )


def detect_cohorts(
    observations: pl.DataFrame,
    threshold: float = COUPLING_THRESHOLD,
    signal_col: str = "signal_id",
    unit_col: str = "unit_id",
    value_col: str = "value",
    index_col: str = "I",
    auto_exclude_constants: bool = True,
    **kwargs,
) -> CohortResult:
    """
    Detect independent cohorts from observation correlations.

    V1 compatibility wrapper around CohortDiscovery.
    """
    cd = CohortDiscovery(system_threshold=threshold)
    cd._observations = observations
    cd._data_format = None  # Reset to auto-detect

    return cd.discover()


def classify_coupling_trajectory(
    observations: pl.DataFrame,
    unit_id: str,
    signal_a: str = "acc_x",
    signal_b: str = "acc_y",
    window_size: int = 2000,
    stride: int = 500,
) -> dict:
    """
    Track correlation between two signals over time for a single unit.

    Classifies trajectory as:
    - STABLE_COUPLED: starts high, stays high
    - STABLE_DECOUPLED: starts low, stays low
    - DECOUPLING: starts high, ends low (fault localizing)
    - COUPLING: starts low, ends high
    - TRANSITIONAL: other patterns
    """
    # Detect column names
    cols = set(observations.columns)
    unit_col = next((c for c in ['unit_id', 'entity_id'] if c in cols), 'unit_id')
    signal_col = next((c for c in ['signal_id', 'signal'] if c in cols), 'signal_id')
    value_col = next((c for c in ['value', 'reading'] if c in cols), 'value')
    index_col = next((c for c in ['I', 'timestamp', 'time', 'cycle'] if c in cols), 'I')

    unit_data = observations.filter(pl.col(unit_col) == unit_id)

    a_data = unit_data.filter(pl.col(signal_col) == signal_a).sort(index_col)
    b_data = unit_data.filter(pl.col(signal_col) == signal_b).sort(index_col)

    if a_data.height == 0 or b_data.height == 0:
        return {"error": f"Missing signal data for {unit_id}"}

    merged = a_data.select([index_col, pl.col(value_col).alias("a")]).join(
        b_data.select([index_col, pl.col(value_col).alias("b")]),
        on=index_col,
        how="inner"
    ).sort(index_col)

    if merged.height < window_size:
        return {"error": f"Insufficient aligned data for {unit_id}"}

    a_vals = merged["a"].to_numpy()
    b_vals = merged["b"].to_numpy()
    i_vals = merged[index_col].to_numpy()

    correlations = []
    for i in range(0, len(a_vals) - window_size, stride):
        a_chunk = a_vals[i:i+window_size]
        b_chunk = b_vals[i:i+window_size]

        corr = np.corrcoef(a_chunk, b_chunk)[0, 1]
        lifecycle = i / (len(a_vals) - window_size)

        correlations.append({
            "I": i_vals[i + window_size // 2],
            "correlation": float(corr) if not np.isnan(corr) else 0.0,
            "lifecycle_position": lifecycle,
        })

    if not correlations:
        return {"error": f"No windows computed for {unit_id}"}

    corr_df = pl.DataFrame(correlations)

    early = [c["correlation"] for c in correlations if c["lifecycle_position"] < 0.25]
    late = [c["correlation"] for c in correlations if c["lifecycle_position"] > 0.75]

    early_mean = np.mean(early) if early else 0.0
    late_mean = np.mean(late) if late else 0.0

    if early_mean > 0.7 and late_mean > 0.7:
        trajectory = "STABLE_COUPLED"
    elif early_mean > 0.7 and late_mean < 0.5:
        trajectory = "DECOUPLING"
    elif early_mean < 0.5 and late_mean < 0.5:
        trajectory = "STABLE_DECOUPLED"
    elif early_mean < 0.5 and late_mean > 0.7:
        trajectory = "COUPLING"
    else:
        trajectory = "TRANSITIONAL"

    return {
        "unit_id": unit_id,
        "trajectory": trajectory,
        "early_correlation": early_mean,
        "late_correlation": late_mean,
        "delta_correlation": late_mean - early_mean,
        "n_windows": len(correlations),
        "time_series": corr_df,
    }


def generate_cohort_report(result: CohortResult) -> str:
    """Generate markdown report from cohort detection result."""
    return result.summary()


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def process_observations(
    observations_path: str,
    manifest_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    constant_threshold: float = CONSTANT_THRESHOLD,
    system_threshold: float = SYSTEM_THRESHOLD,
    component_threshold: float = COMPONENT_THRESHOLD,
) -> CohortResult:
    """
    Process observations.parquet and discover cohorts.

    Main entry point for ORTHON integration.

    Args:
        observations_path: Path to observations.parquet
        manifest_path: Optional path to manifest.yaml
        output_dir: Optional output directory for results
        constant_threshold: Threshold for constant detection (default 0.99)
        system_threshold: Threshold for system signals (default 0.70)
        component_threshold: Threshold for component signals (default 0.50)

    Returns:
        CohortResult with all findings
    """
    cd = CohortDiscovery(
        observations_path=observations_path,
        manifest_path=manifest_path,
        constant_threshold=constant_threshold,
        system_threshold=system_threshold,
        component_threshold=component_threshold,
    )

    result = cd.discover()

    if output_dir:
        cd.save_results(result, Path(output_dir))

    return result


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Discover signal cohorts from observations'
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--manifest', '-m', help='Path to manifest.yaml')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--constant-threshold', type=float, default=CONSTANT_THRESHOLD)
    parser.add_argument('--system-threshold', type=float, default=SYSTEM_THRESHOLD)
    parser.add_argument('--component-threshold', type=float, default=COMPONENT_THRESHOLD)

    args = parser.parse_args()

    cd = CohortDiscovery(
        observations_path=args.observations,
        manifest_path=args.manifest,
        constant_threshold=args.constant_threshold,
        system_threshold=args.system_threshold,
        component_threshold=args.component_threshold,
    )

    result = cd.discover()

    if args.output:
        cd.save_results(result, Path(args.output))
    else:
        print(result.summary())


if __name__ == '__main__':
    main()
