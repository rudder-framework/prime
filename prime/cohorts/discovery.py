"""
Cohort Discovery Module v2

Automatically detects independent subdomains from geometry.
Identifies operational constants, system-level coupling, and component-level signals.

CANONICAL PRINCIPLE:
    "Run Manifold FIRST - geometry reveals physics you didn't know."

NEW IN V2:
    - Constant detection (operational settings with ρ > 0.99)
    - Improved cohort classification
    - ML-ready output (exclude list for preprocessing)

Usage:
    from prime.cohorts.discovery import CohortDiscovery
    
    cd = CohortDiscovery(observations_path)
    result = cd.discover()
    
    # Get ML-ready signal list
    ml_signals = result.get_ml_signals()  # excludes constants
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import defaultdict
import json
import yaml

import polars as pl
import numpy as np


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
class CohortResult:
    """Result of cohort discovery."""
    
    # Signal classifications
    constants: List[str] = field(default_factory=list)        # ρ > 0.99 across units
    system_signals: List[str] = field(default_factory=list)   # ρ > 0.7 across units
    component_signals: List[str] = field(default_factory=list) # ρ < 0.5 across units (use for RUL)
    orphan_signals: List[str] = field(default_factory=list)   # Uncorrelated with anything
    
    # Detailed classifications
    signal_details: Dict[str, SignalClassification] = field(default_factory=dict)
    
    # Unit-level analysis (for multi-signal units like FEMTO)
    coupled_units: List[str] = field(default_factory=list)
    decoupled_units: List[str] = field(default_factory=list)
    
    # Graph structure
    n_cohorts: int = 0
    cohort_membership: Dict[str, int] = field(default_factory=dict)
    cohort_contents: Dict[int, List[str]] = field(default_factory=dict)
    
    # Metrics
    cross_unit_corr_matrix: Optional[Dict[str, float]] = None
    
    def get_ml_signals(self) -> List[str]:
        """Get signals suitable for ML (excludes constants)."""
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
            }
        }
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "COHORT DISCOVERY RESULTS",
            "=" * 60,
            "",
            f"CONSTANTS ({len(self.constants)}): {self.constants}",
            f"  → Operational settings, exclude from analysis",
            "",
            f"SYSTEM SIGNALS ({len(self.system_signals)}): {self.system_signals[:5]}{'...' if len(self.system_signals) > 5 else ''}",
            f"  → Coupled across units, use for fleet baseline",
            "",
            f"COMPONENT SIGNALS ({len(self.component_signals)}): {self.component_signals[:5]}{'...' if len(self.component_signals) > 5 else ''}",
            f"  → Unit-specific, USE FOR RUL/FAULT DETECTION",
            "",
            f"ORPHANS ({len(self.orphan_signals)}): {self.orphan_signals}",
            f"  → Low correlation with everything, investigate",
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
                f"COUPLED UNITS ({len(self.coupled_units)}): {self.coupled_units[:5]}",
                f"  → Isotropic degradation, signals move together",
                f"DECOUPLED UNITS ({len(self.decoupled_units)}): {self.decoupled_units[:5]}",
                f"  → Anisotropic/localized faults, signals independent",
                "",
            ])
        
        return "\n".join(lines)


class CohortDiscovery:
    """
    Discover signal structure from observations.
    
    Identifies:
        1. CONSTANTS: Operational settings (ρ > 0.99 across units)
        2. SYSTEM: Signals coupled across units (ρ > 0.7)
        3. COMPONENT: Unit-specific signals (ρ < 0.5) - USE FOR ML
        4. ORPHANS: Uncorrelated signals (investigate)
    
    For multi-signal units (like FEMTO bearings):
        - COUPLED: Signals within unit move together
        - DECOUPLED: Signals within unit are independent (localized fault)
    """
    
    # Thresholds
    CONSTANT_THRESHOLD = 0.99    # Above this = operational constant
    SYSTEM_THRESHOLD = 0.70      # Above this = system-level coupling
    COMPONENT_THRESHOLD = 0.50   # Below this = component-level (independent)
    WITHIN_UNIT_THRESHOLD = 0.70 # For coupled/decoupled unit classification
    
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
        
        Returns dict with:
            - format: 'long' or 'wide'
            - signal_col: column containing signal names (if long)
            - value_col: column containing values (if long)
            - unit_col: column containing unit IDs (if present)
            - time_col: column containing timestamps
            - signal_cols: list of signal columns (if wide)
        """
        cols = set(df.columns)
        
        # Check for long format indicators
        if 'signal_id' in cols and 'value' in cols:
            return {
                'format': 'long',
                'signal_col': 'signal_id',
                'value_col': 'value',
                'unit_col': 'cohort' if 'cohort' in cols else None,
                'time_col': next((c for c in ['signal_0', 'I', 'timestamp', 'time', 'cycle'] if c in cols), None),
            }

        if 'signal' in cols and 'value' in cols:
            return {
                'format': 'long',
                'signal_col': 'signal',
                'value_col': 'value',
                'unit_col': 'cohort' if 'cohort' in cols else None,
                'time_col': next((c for c in ['signal_0', 'I', 'timestamp', 'time', 'cycle'] if c in cols), None),
            }

        # Assume wide format
        meta_cols = {'timestamp', 'time', 'cycle', 'signal_0', 'cohort', 'cohort_id', 'window', 'observation_id'}
        signal_cols = [c for c in df.columns if c not in meta_cols and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

        return {
            'format': 'wide',
            'signal_cols': signal_cols,
            'unit_col': 'cohort' if 'cohort' in cols else None,
            'time_col': next((c for c in ['signal_0', 'I', 'timestamp', 'time', 'cycle'] if c in cols), None),
        }
    
    def compute_cross_unit_correlations(self) -> Dict[str, float]:
        """
        Compute mean cross-unit correlation for each signal.
        
        For each signal, computes correlation between all pairs of units
        and returns the mean.
        
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
                        columns=unit_col,
                        values=value_col
                    ).drop_nulls()
                except:
                    continue
                
                if len(pivoted) < 10:
                    cross_unit_corrs[signal] = 0.0
                    continue
                
                # Compute pairwise correlations between units
                unit_cols = [c for c in pivoted.columns if c != time_col]
                if len(unit_cols) < 2:
                    continue
                
                correlations = []
                data = pivoted.select(unit_cols).to_numpy()
                data = np.nan_to_num(data, nan=0.0)
                
                # Check for constant columns
                stds = np.std(data, axis=0)
                if np.any(stds == 0):
                    # Has constant columns - check if ALL are constant
                    if np.all(stds == 0):
                        cross_unit_corrs[signal] = 1.0  # All constant = perfectly correlated
                        continue
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    corr_matrix = np.abs(np.corrcoef(data.T))
                    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
                
                # Extract upper triangle (excluding diagonal)
                n = len(unit_cols)
                for i in range(n):
                    for j in range(i + 1, n):
                        correlations.append(corr_matrix[i, j])
                
                if correlations:
                    cross_unit_corrs[signal] = float(np.mean(correlations))
                else:
                    cross_unit_corrs[signal] = 0.0
        
        else:  # wide format
            # For wide format with cohort, need different approach
            signal_cols = fmt['signal_cols']
            time_col = fmt['time_col']
            
            for signal in signal_cols:
                # Pivot to get units as columns for this signal
                try:
                    pivoted = df.select([time_col, unit_col, signal]).pivot(
                        index=time_col,
                        columns=unit_col,
                        values=signal
                    ).drop_nulls()
                except:
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
    
    def compute_within_unit_correlations(self, unit_pattern: str = None) -> Dict[str, float]:
        """
        Compute within-unit signal correlations.
        
        For datasets like FEMTO where each unit has multiple signals
        (e.g., acc_x and acc_y per bearing).
        
        Returns:
            Dict mapping cohort to mean within-unit correlation
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
                    columns=signal_col,
                    values=value_col
                ).drop_nulls()
            except:
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
                classification.notes = f"ρ={corr:.3f} ≥ {self.constant_threshold} (operational setting)"
                result.constants.append(signal)
                
            elif corr >= self.system_threshold:
                classification.classification = 'system'
                classification.notes = f"ρ={corr:.3f} ≥ {self.system_threshold} (coupled across units)"
                result.system_signals.append(signal)
                
            elif corr >= self.component_threshold:
                classification.classification = 'partial'
                classification.notes = f"{self.component_threshold} ≤ ρ={corr:.3f} < {self.system_threshold} (partially coupled)"
                result.component_signals.append(signal)  # Still usable for ML
                
            elif corr > 0.1:
                classification.classification = 'component'
                classification.notes = f"ρ={corr:.3f} < {self.component_threshold} (unit-specific)"
                result.component_signals.append(signal)
                
            else:
                classification.classification = 'orphan'
                classification.notes = f"ρ={corr:.3f} ≈ 0 (uncorrelated, investigate)"
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
            if corr >= self.WITHIN_UNIT_THRESHOLD:
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
                    'within_unit': self.WITHIN_UNIT_THRESHOLD,
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


def process_observations(
    observations_path: str,
    manifest_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> CohortResult:
    """
    Process observations.parquet and discover cohorts.
    
    Main entry point for Prime integration.
    
    Args:
        observations_path: Path to observations.parquet
        manifest_path: Optional path to manifest.yaml
        output_dir: Optional output directory for results
        
    Returns:
        CohortResult with all findings
    """
    cd = CohortDiscovery(
        observations_path=observations_path,
        manifest_path=manifest_path,
    )
    
    result = cd.discover()
    
    if output_dir:
        cd.save_results(result, Path(output_dir))
    
    return result


# CLI
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Discover signal cohorts from observations'
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--manifest', '-m', help='Path to manifest.yaml')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--constant-threshold', type=float, default=0.99)
    parser.add_argument('--system-threshold', type=float, default=0.70)
    parser.add_argument('--component-threshold', type=float, default=0.50)
    
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
