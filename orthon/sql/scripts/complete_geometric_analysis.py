"""
ORTHON Complete Geometric Analysis

Five-layer analysis:
1. DIMENSIONAL    - Effective dimension over time
2. PROPAGATION    - Lead-lag causal chain
3. CAUSALITY      - Who drives structural changes
4. RESTRUCTURE    - Eigenvector rotation events
5. EMERGENCE      - Dimensional increase (true regime change)

Output: Full diagnostic report + regime change detection

Usage:
    python complete_geometric_analysis.py [dataset_name]

    # Default: building_vibration
    python complete_geometric_analysis.py

    # Specify dataset:
    python complete_geometric_analysis.py skab

Outputs:
    complete_analysis.parquet - Per-window metrics
    regime_changes.parquet - Detected regime change events
    complete_analysis.png - 5-panel visualization
"""

import polars as pl
import numpy as np
from scipy import signal as sig
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum
import matplotlib.pyplot as plt


class RegimeChangeType(str, Enum):
    """Types of regime change detected."""
    DIMENSIONAL_COLLAPSE = "DIMENSIONAL_COLLAPSE"      # eff_dim decreasing
    DIMENSIONAL_EMERGENCE = "DIMENSIONAL_EMERGENCE"    # eff_dim increasing ← TRUE REGIME CHANGE
    STRUCTURAL_ROTATION = "STRUCTURAL_ROTATION"        # eigenvectors rotating
    OPERATING_SHIFT = "OPERATING_SHIFT"                # mean values shifted
    COUPLING_CHANGE = "COUPLING_CHANGE"                # correlation structure changed
    NONE = "NONE"


@dataclass
class RegimeChangeEvent:
    """Detected regime change event."""
    I: int
    change_type: RegimeChangeType
    magnitude: float
    details: Dict


@dataclass
class AnalysisWindow:
    """Metrics for a single analysis window."""
    I: int

    # Dimensional
    effective_dim: float
    eff_dim_delta: float
    eigenvalues: np.ndarray
    variance_explained: np.ndarray

    # Structural
    eigenvec_alignment: float
    subspace_alignment: float
    condition_number: float

    # Operating point
    operating_point: np.ndarray
    op_deviation: float

    # Coupling
    mean_abs_correlation: float
    correlation_matrix: np.ndarray

    # Causality (filled by propagation analysis)
    driver_by_loading: Optional[str] = None
    driver_by_innovation: Optional[str] = None


def compute_effective_dim(eigenvalues: np.ndarray) -> float:
    """Participation ratio."""
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    total = np.sum(eigenvalues)
    p = eigenvalues / total
    return 1.0 / np.sum(p ** 2)


def compute_subspace_alignment(V1: np.ndarray, V2: np.ndarray, k: int = 3) -> float:
    """How much do top-k eigenvectors span the same subspace?"""
    overlap = 0
    for i in range(min(k, V1.shape[1])):
        for j in range(min(k, V2.shape[1])):
            overlap += np.dot(V1[:, i], V2[:, j]) ** 2
    return overlap / k


class GeometricAnalyzer:
    """Complete geometric analysis of multivariate time series."""

    def __init__(
        self,
        data_dir: Path,
        window_size: int = 128,
        stride: int = 32
    ):
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride

        # Load data
        obs = pl.read_parquet(self.data_dir / 'observations.parquet')
        wide = obs.pivot(values='value', index='I', on='signal_id').sort('I')

        self.signals = [c for c in wide.columns if c != 'I']
        self.n_signals = len(self.signals)
        self.I_values = wide['I'].to_numpy()
        self.X = wide.select(self.signals).to_numpy()

        # Normalize
        self.X_mean = np.nanmean(self.X, axis=0)
        self.X_std = np.nanstd(self.X, axis=0)
        self.X_std[self.X_std == 0] = 1
        self.X_norm = (self.X - self.X_mean) / self.X_std

        # Baseline (first 20%)
        baseline_end = int(len(self.X) * 0.2)
        self.baseline_mean = np.nanmean(self.X[:baseline_end], axis=0)
        self.baseline_std = np.nanstd(self.X[:baseline_end], axis=0)
        self.baseline_std[self.baseline_std == 0] = 1

        baseline_cov = np.cov(self.X_norm[:baseline_end], rowvar=False)
        baseline_eig, _ = np.linalg.eigh(baseline_cov)
        self.baseline_eff_dim = compute_effective_dim(np.sort(baseline_eig)[::-1])

        # Storage
        self.windows: List[AnalysisWindow] = []
        self.regime_changes: List[RegimeChangeEvent] = []

        print(f"Loaded {len(self.signals)} signals: {self.signals}")
        print(f"Samples: {len(self.X)}, Windows: {(len(self.X) - window_size) // stride + 1}")
        print(f"Baseline effective dimension: {self.baseline_eff_dim:.2f}")

    # ================================================================
    # LAYER 1: DIMENSIONAL ANALYSIS
    # ================================================================

    def analyze_dimensions(self):
        """Compute eigenstructure per window."""
        print("\n" + "="*60)
        print("LAYER 1: DIMENSIONAL ANALYSIS")
        print("="*60)

        prev_eigenvectors = None
        prev_eff_dim = None

        for start in range(0, len(self.X_norm) - self.window_size + 1, self.stride):
            end = start + self.window_size
            window_I = self.I_values[end - 1]
            W = self.X_norm[start:end]

            if np.any(np.isnan(W)):
                continue

            # Eigendecomposition
            cov = np.cov(W, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues[idx], 1e-10)
            eigenvectors = eigenvectors[:, idx]

            # Effective dimension
            eff_dim = compute_effective_dim(eigenvalues)
            eff_dim_delta = eff_dim - prev_eff_dim if prev_eff_dim else 0

            # Variance explained
            total_var = np.sum(eigenvalues)
            var_explained = np.cumsum(eigenvalues) / total_var

            # Eigenvector alignment
            if prev_eigenvectors is not None:
                sign = np.sign(np.dot(eigenvectors[:, 0], prev_eigenvectors[:, 0]))
                alignment = abs(np.dot(eigenvectors[:, 0], prev_eigenvectors[:, 0]))
                subspace_align = compute_subspace_alignment(eigenvectors, prev_eigenvectors)
            else:
                alignment = 1.0
                subspace_align = 1.0

            # Condition number
            condition = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 0 else np.inf

            # Operating point deviation
            window_mean = np.mean(self.X[start:end], axis=0)
            op_deviation = np.mean(np.abs((window_mean - self.baseline_mean) / self.baseline_std))

            # Correlation matrix
            corr_matrix = np.corrcoef(W, rowvar=False)
            upper_tri = corr_matrix[np.triu_indices(self.n_signals, k=1)]
            mean_abs_corr = np.mean(np.abs(upper_tri))

            window = AnalysisWindow(
                I=int(window_I),
                effective_dim=eff_dim,
                eff_dim_delta=eff_dim_delta,
                eigenvalues=eigenvalues,
                variance_explained=var_explained,
                eigenvec_alignment=alignment,
                subspace_alignment=subspace_align,
                condition_number=condition,
                operating_point=window_mean,
                op_deviation=op_deviation,
                mean_abs_correlation=mean_abs_corr,
                correlation_matrix=corr_matrix,
            )

            self.windows.append(window)
            prev_eigenvectors = eigenvectors.copy()
            prev_eff_dim = eff_dim

        print(f"Computed {len(self.windows)} windows")

        # Summary
        eff_dims = [w.effective_dim for w in self.windows]
        alignments = [w.eigenvec_alignment for w in self.windows]

        print(f"\nEffective Dimension: {np.mean(eff_dims):.2f} ± {np.std(eff_dims):.2f}")
        print(f"  Range: {np.min(eff_dims):.2f} to {np.max(eff_dims):.2f}")
        print(f"  Baseline: {self.baseline_eff_dim:.2f}")

        print(f"\nEigenvector Alignment: {np.mean(alignments):.3f} ± {np.std(alignments):.3f}")
        print(f"  Windows < 0.95: {sum(1 for a in alignments if a < 0.95)} ({100*sum(1 for a in alignments if a < 0.95)/len(alignments):.1f}%)")

    # ================================================================
    # LAYER 2: PROPAGATION ANALYSIS
    # ================================================================

    def analyze_propagation(self, max_lag: int = 50):
        """Determine lead-lag relationships between signals."""
        print("\n" + "="*60)
        print("LAYER 2: PROPAGATION ANALYSIS")
        print("="*60)

        data = {s: self.X_norm[:, i] for i, s in enumerate(self.signals)}

        # Lead-lag matrix
        lead_matrix = np.zeros((self.n_signals, self.n_signals))
        corr_at_peak = np.zeros((self.n_signals, self.n_signals))

        for i, sig_a in enumerate(self.signals):
            for j, sig_b in enumerate(self.signals):
                if i == j:
                    continue

                best_lag = 0
                best_corr = 0

                for lag in range(-max_lag, max_lag + 1):
                    if lag < 0:
                        corr = np.corrcoef(data[sig_a][:lag], data[sig_b][-lag:])[0, 1]
                    elif lag > 0:
                        corr = np.corrcoef(data[sig_a][lag:], data[sig_b][:-lag])[0, 1]
                    else:
                        corr = np.corrcoef(data[sig_a], data[sig_b])[0, 1]

                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_lag = lag

                lead_matrix[i, j] = best_lag
                corr_at_peak[i, j] = best_corr

        # Determine propagation chain
        mean_lead = np.mean(lead_matrix, axis=1)
        chain_order = np.argsort(mean_lead)[::-1]

        print("\nPROPAGATION CHAIN (earliest to latest):")
        for rank, idx in enumerate(chain_order):
            sig = self.signals[idx]
            print(f"  {rank+1}. {sig} (mean lead: {mean_lead[idx]:+.1f} samples)")

        # Store results
        self.lead_matrix = lead_matrix
        self.propagation_chain = [self.signals[i] for i in chain_order]
        self.primary_source = self.propagation_chain[0]

        print(f"\nPRIMARY SOURCE: {self.primary_source}")

        # Check consistency
        primary_idx = self.signals.index(self.primary_source)
        primary_leads_all = all(lead_matrix[primary_idx, j] >= 0
                                for j in range(self.n_signals) if j != primary_idx)
        print(f"Consistently leads all others: {primary_leads_all}")

        return self.propagation_chain

    # ================================================================
    # LAYER 3: CAUSALITY ANALYSIS
    # ================================================================

    def analyze_causality(self):
        """Identify what drives structural changes."""
        print("\n" + "="*60)
        print("LAYER 3: CAUSALITY ANALYSIS")
        print("="*60)

        prev_eigenvectors = None
        driver_counts_loading = {s: 0 for s in self.signals}
        driver_counts_innovation = {s: 0 for s in self.signals}

        for i, window in enumerate(self.windows):
            if i == 0:
                continue

            # Recompute for this window to get eigenvectors
            start = i * self.stride
            end = start + self.window_size
            if end > len(self.X_norm):
                continue

            W = self.X_norm[start:end]
            cov = np.cov(W, rowvar=False)
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            idx = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, idx]

            if prev_eigenvectors is not None:
                # Loading change per signal
                sign = np.sign(np.dot(eigenvectors[:, 0], prev_eigenvectors[:, 0]))
                loading_change = np.abs(eigenvectors[:, 0] * sign - prev_eigenvectors[:, 0])

                # Innovation per signal
                last_point = W[-1]
                projection = np.dot(last_point, prev_eigenvectors)
                reconstructed = np.dot(projection, prev_eigenvectors.T)
                innovation = np.abs(last_point - reconstructed)

                driver_loading = self.signals[np.argmax(loading_change)]
                driver_innovation = self.signals[np.argmax(innovation)]

                driver_counts_loading[driver_loading] += 1
                driver_counts_innovation[driver_innovation] += 1

                window.driver_by_loading = driver_loading
                window.driver_by_innovation = driver_innovation

            prev_eigenvectors = eigenvectors.copy()

        print("\nDRIVER BY LOADING CHANGE (who changes the eigenstructure):")
        for sig, count in sorted(driver_counts_loading.items(), key=lambda x: -x[1]):
            pct = 100 * count / (len(self.windows) - 1)
            print(f"  {sig}: {count} ({pct:.1f}%)")

        print("\nDRIVER BY INNOVATION (who moves unexpectedly first):")
        for sig, count in sorted(driver_counts_innovation.items(), key=lambda x: -x[1]):
            pct = 100 * count / (len(self.windows) - 1)
            print(f"  {sig}: {count} ({pct:.1f}%)")

        self.structural_driver = max(driver_counts_loading, key=driver_counts_loading.get)
        self.innovation_driver = max(driver_counts_innovation, key=driver_counts_innovation.get)

        print(f"\nPRIMARY STRUCTURAL DRIVER: {self.structural_driver}")
        print(f"PRIMARY INNOVATION DRIVER: {self.innovation_driver}")

    # ================================================================
    # LAYER 4: RESTRUCTURE ANALYSIS
    # ================================================================

    def analyze_restructure(self, rotation_threshold: float = 0.8):
        """Detect eigenvector rotation events."""
        print("\n" + "="*60)
        print("LAYER 4: RESTRUCTURE ANALYSIS")
        print("="*60)

        rotation_events = []

        for i, window in enumerate(self.windows):
            if window.eigenvec_alignment < rotation_threshold:
                rotation_events.append({
                    'I': window.I,
                    'alignment': window.eigenvec_alignment,
                    'subspace_alignment': window.subspace_alignment,
                    'eff_dim': window.effective_dim,
                })

                self.regime_changes.append(RegimeChangeEvent(
                    I=window.I,
                    change_type=RegimeChangeType.STRUCTURAL_ROTATION,
                    magnitude=1 - window.eigenvec_alignment,
                    details={'alignment': window.eigenvec_alignment}
                ))

        print(f"\nStructural rotation events (alignment < {rotation_threshold}): {len(rotation_events)}")

        if rotation_events:
            print("\nTop 10 rotation events:")
            sorted_events = sorted(rotation_events, key=lambda x: x['alignment'])
            for e in sorted_events[:10]:
                print(f"  I={e['I']}: alignment={e['alignment']:.3f}, eff_dim={e['eff_dim']:.2f}")

    # ================================================================
    # LAYER 5: EMERGENCE ANALYSIS (TRUE REGIME CHANGE)
    # ================================================================

    def analyze_emergence(self, emergence_threshold: float = 0.5):
        """
        Detect dimensional emergence - TRUE REGIME CHANGE.

        When effective_dim INCREASES significantly, new independent
        modes are appearing. The system has fundamentally restructured.
        """
        print("\n" + "="*60)
        print("LAYER 5: EMERGENCE ANALYSIS (TRUE REGIME CHANGE)")
        print("="*60)

        emergence_events = []
        collapse_events = []

        for i, window in enumerate(self.windows):
            if i == 0:
                continue

            prev_eff_dim = self.windows[i-1].effective_dim
            delta = window.effective_dim - prev_eff_dim

            # EMERGENCE: New dimensions appearing
            if delta > emergence_threshold:
                emergence_events.append({
                    'I': window.I,
                    'prev_dim': prev_eff_dim,
                    'new_dim': window.effective_dim,
                    'delta': delta,
                    'alignment': window.eigenvec_alignment,
                })

                self.regime_changes.append(RegimeChangeEvent(
                    I=window.I,
                    change_type=RegimeChangeType.DIMENSIONAL_EMERGENCE,
                    magnitude=delta,
                    details={
                        'prev_dim': prev_eff_dim,
                        'new_dim': window.effective_dim,
                    }
                ))

            # COLLAPSE: Dimensions merging
            elif delta < -emergence_threshold:
                collapse_events.append({
                    'I': window.I,
                    'prev_dim': prev_eff_dim,
                    'new_dim': window.effective_dim,
                    'delta': delta,
                    'alignment': window.eigenvec_alignment,
                })

                self.regime_changes.append(RegimeChangeEvent(
                    I=window.I,
                    change_type=RegimeChangeType.DIMENSIONAL_COLLAPSE,
                    magnitude=abs(delta),
                    details={
                        'prev_dim': prev_eff_dim,
                        'new_dim': window.effective_dim,
                    }
                ))

        print(f"\n>>> DIMENSIONAL EMERGENCE EVENTS: {len(emergence_events)} <<<")
        print("(New independent modes appearing - TRUE REGIME CHANGE)")

        if emergence_events:
            for e in sorted(emergence_events, key=lambda x: -x['delta'])[:10]:
                print(f"  I={e['I']}: {e['prev_dim']:.2f} → {e['new_dim']:.2f} (Δ=+{e['delta']:.2f})")

        print(f"\nDimensional collapse events: {len(collapse_events)}")
        print("(Modes coupling together)")

        if collapse_events:
            for e in sorted(collapse_events, key=lambda x: x['delta'])[:10]:
                print(f"  I={e['I']}: {e['prev_dim']:.2f} → {e['new_dim']:.2f} (Δ={e['delta']:.2f})")

        # Track sustained changes
        print("\n" + "-"*40)
        print("SUSTAINED DIMENSIONAL SHIFTS")
        print("-"*40)

        # Compare first 20% vs last 20%
        n_windows = len(self.windows)
        early_dims = [w.effective_dim for w in self.windows[:n_windows//5]]
        late_dims = [w.effective_dim for w in self.windows[-n_windows//5:]]

        early_mean = np.mean(early_dims)
        late_mean = np.mean(late_dims)
        shift = late_mean - early_mean

        print(f"\nEarly period (first 20%): eff_dim = {early_mean:.2f} ± {np.std(early_dims):.2f}")
        print(f"Late period (last 20%):   eff_dim = {late_mean:.2f} ± {np.std(late_dims):.2f}")
        print(f"Sustained shift: {shift:+.2f}")

        if shift > 0.3:
            print("\n>>> SUSTAINED DIMENSIONAL EMERGENCE DETECTED <<<")
            print("System has MORE independent modes than it started with.")
            print("This indicates FUNDAMENTAL REGIME CHANGE.")
        elif shift < -0.3:
            print("\n>>> SUSTAINED DIMENSIONAL COLLAPSE DETECTED <<<")
            print("System has FEWER independent modes than it started with.")
            print("Modes have become coupled.")

    # ================================================================
    # FULL ANALYSIS
    # ================================================================

    def run_full_analysis(self):
        """Run all five analysis layers."""
        self.analyze_dimensions()
        self.analyze_propagation()
        self.analyze_causality()
        self.analyze_restructure()
        self.analyze_emergence()

        # Final summary
        print("\n" + "="*60)
        print("REGIME CHANGE SUMMARY")
        print("="*60)

        # Count by type
        by_type = {}
        for rc in self.regime_changes:
            by_type[rc.change_type] = by_type.get(rc.change_type, 0) + 1

        print("\nRegime change events by type:")
        for change_type, count in sorted(by_type.items(), key=lambda x: -x[1]):
            print(f"  {change_type.value}: {count}")

        # Key findings
        print("\n" + "-"*40)
        print("KEY FINDINGS")
        print("-"*40)
        print(f"Propagation source: {self.primary_source}")
        print(f"Structural driver: {self.structural_driver}")
        print(f"Innovation driver: {self.innovation_driver}")

        emergence_count = by_type.get(RegimeChangeType.DIMENSIONAL_EMERGENCE, 0)
        if emergence_count > 0:
            print(f"\n>>> {emergence_count} DIMENSIONAL EMERGENCE EVENTS <<<")
            print("True regime changes detected - new independent modes appeared.")

        return self.windows, self.regime_changes

    # ================================================================
    # VISUALIZATION
    # ================================================================

    def plot_analysis(self):
        """Generate visualization plots."""
        fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)

        I_vals = [w.I for w in self.windows]

        # 1. Effective Dimension
        ax = axes[0]
        eff_dims = [w.effective_dim for w in self.windows]
        ax.plot(I_vals, eff_dims, 'b-', linewidth=1)
        ax.fill_between(I_vals, 0, eff_dims, alpha=0.3)
        ax.axhline(y=self.baseline_eff_dim, color='r', linestyle='--',
                   label=f'Baseline: {self.baseline_eff_dim:.2f}')
        ax.set_ylabel('Effective\nDimension')
        ax.set_title('DIMENSIONAL ANALYSIS')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # Mark emergence events
        for rc in self.regime_changes:
            if rc.change_type == RegimeChangeType.DIMENSIONAL_EMERGENCE:
                ax.axvline(x=rc.I, color='green', alpha=0.5, linewidth=2)

        # 2. Dimensional Delta
        ax = axes[1]
        deltas = [w.eff_dim_delta for w in self.windows]
        colors = ['green' if d > 0 else 'red' for d in deltas]
        ax.bar(I_vals, deltas, color=colors, alpha=0.7, width=self.stride*0.8)
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Emergence threshold')
        ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5, label='Collapse threshold')
        ax.set_ylabel('Δ Effective\nDimension')
        ax.set_title('DIMENSIONAL CHANGE (green=emergence, red=collapse)')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # 3. Eigenvector Alignment
        ax = axes[2]
        alignments = [w.eigenvec_alignment for w in self.windows]
        ax.plot(I_vals, alignments, 'purple', linewidth=1)
        ax.fill_between(I_vals, 0, alignments, alpha=0.3, color='purple')
        ax.axhline(y=0.95, color='orange', linestyle='--', label='Stability threshold')
        ax.axhline(y=0.8, color='red', linestyle='--', label='Rotation threshold')
        ax.set_ylabel('Eigenvector\nAlignment')
        ax.set_title('STRUCTURAL STABILITY')
        ax.set_ylim(0, 1.05)
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # 4. Mean Correlation (Coupling)
        ax = axes[3]
        correlations = [w.mean_abs_correlation for w in self.windows]
        ax.plot(I_vals, correlations, 'orange', linewidth=1)
        ax.fill_between(I_vals, 0, correlations, alpha=0.3, color='orange')
        ax.set_ylabel('Mean |Correlation|')
        ax.set_title('COUPLING STRENGTH')
        ax.grid(True, alpha=0.3)

        # 5. Operating Point Deviation
        ax = axes[4]
        op_devs = [w.op_deviation for w in self.windows]
        ax.plot(I_vals, op_devs, 'brown', linewidth=1)
        ax.fill_between(I_vals, 0, op_devs, alpha=0.3, color='brown')
        ax.axhline(y=2.0, color='red', linestyle='--', label='Shift threshold (2σ)')
        ax.set_ylabel('Operating Point\nDeviation (σ)')
        ax.set_xlabel('Time (I)')
        ax.set_title('OPERATING POINT DRIFT')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.data_dir / 'complete_analysis.png', dpi=150)
        print(f"\nSaved: {self.data_dir / 'complete_analysis.png'}")

        return fig


# ================================================================
# MAIN EXECUTION
# ================================================================

def main():
    """CLI entry point."""
    import sys

    # Default to building_vibration, or accept command line arg
    if len(sys.argv) > 1:
        dataset = sys.argv[1]
    else:
        dataset = "building_vibration"

    data_dir = Path.home() / 'domains' / dataset

    print(f"\n{'='*60}")
    print(f"ORTHON COMPLETE GEOMETRIC ANALYSIS")
    print(f"Dataset: {dataset}")
    print(f"{'='*60}")

    analyzer = GeometricAnalyzer(data_dir, window_size=128, stride=32)
    windows, regime_changes = analyzer.run_full_analysis()
    analyzer.plot_analysis()

    # Save results
    results = []
    for w in windows:
        results.append({
            'I': w.I,
            'effective_dim': w.effective_dim,
            'eff_dim_delta': w.eff_dim_delta,
            'eigenvec_alignment': w.eigenvec_alignment,
            'subspace_alignment': w.subspace_alignment,
            'condition_number': w.condition_number,
            'op_deviation': w.op_deviation,
            'mean_abs_correlation': w.mean_abs_correlation,
            'lambda_1': w.eigenvalues[0],
            'lambda_2': w.eigenvalues[1] if len(w.eigenvalues) > 1 else None,
            'var_explained_1': w.variance_explained[0],
            'var_explained_2': w.variance_explained[1] if len(w.variance_explained) > 1 else None,
            'driver_by_loading': w.driver_by_loading,
            'driver_by_innovation': w.driver_by_innovation,
        })

    df = pl.DataFrame(results)
    df.write_parquet(data_dir / 'complete_analysis.parquet')
    print(f"Saved: {data_dir / 'complete_analysis.parquet'}")

    # Save regime changes
    rc_data = [{
        'I': rc.I,
        'change_type': rc.change_type.value,
        'magnitude': rc.magnitude,
    } for rc in regime_changes]

    if rc_data:
        rc_df = pl.DataFrame(rc_data)
        rc_df.write_parquet(data_dir / 'regime_changes.parquet')
        print(f"Saved: {data_dir / 'regime_changes.parquet'}")


if __name__ == '__main__':
    main()
