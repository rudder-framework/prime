#!/usr/bin/env python3
"""
Framework Analysis: Thermodynamic Interpretation

The eigenvalue spectrum IS statistical mechanics. This script reads
Manifold's state_geometry.parquet and computes thermodynamic quantities
from existing columns. No new compute — just arithmetic on what's there.

Mapping:
    total_variance  (Σλᵢ)        →  Energy (E)
    effective_dim   (exp H(λ))   →  Entropy (S)
    ΔS / ΔI                      →  Temperature (T)
    E - T·S                      →  Free energy (F)
    ΔE / ΔT                      →  Heat capacity (Cᵥ)

If Manifold already ran stage_09a (cohort_thermodynamics.parquet),
this reads that. Otherwise, computes directly from state_geometry.

Input:
    state_geometry.parquet  — per-cohort eigenvalue trajectories

Output:
    thermodynamics.parquet  — per-cohort thermodynamic trajectories
    Console summary with phase transition detection

Usage:
    python -m prime.analysis.thermodynamics \
        --geometry path/to/state_geometry.parquet \
        --output path/to/thermodynamics.parquet
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats


def compute_thermodynamics(geometry: pl.DataFrame, engine_name: str = None) -> pl.DataFrame:
    """
    Compute thermodynamic quantities per cohort per window.
    
    All quantities derived from existing state_geometry columns.
    This is interpretation, not new computation.
    """
    if engine_name and 'engine' in geometry.columns:
        geo = geometry.filter(pl.col('engine') == engine_name)
    else:
        geo = geometry
    
    cohorts = sorted(geo['cohort'].unique().to_list())
    
    rows = []
    
    for cohort in cohorts:
        cg = geo.filter(pl.col('cohort') == cohort).sort('I')
        
        if len(cg) < 3:
            continue
        
        I_vals = cg['I'].to_numpy().astype(float)
        E = cg['total_variance'].to_numpy()   # Energy
        S = cg['effective_dim'].to_numpy()      # Entropy (exp of Shannon entropy)
        
        # Handle NaN
        valid = ~(np.isnan(E) | np.isnan(S))
        if valid.sum() < 3:
            continue
        
        n = len(I_vals)
        
        # Temperature: T = ΔS/ΔI (rate of entropy change)
        # Central differences for interior, forward/backward at edges
        T = np.full(n, np.nan)
        if n >= 3:
            T[1:-1] = (S[2:] - S[:-2]) / (I_vals[2:] - I_vals[:-2] + 1e-10)
            T[0] = (S[1] - S[0]) / (I_vals[1] - I_vals[0] + 1e-10)
            T[-1] = (S[-1] - S[-2]) / (I_vals[-1] - I_vals[-2] + 1e-10)
        
        # Free energy: F = E - T·S
        F = E - T * S
        
        # Heat capacity: Cᵥ = ΔE/ΔT
        Cv = np.full(n, np.nan)
        dE = np.gradient(E, I_vals)
        dT = np.gradient(T, I_vals)
        # Avoid division by near-zero temperature change
        safe_dT = np.where(np.abs(dT) > 1e-10, dT, np.nan)
        Cv = dE / safe_dT
        
        # Energy concentration (eigenvalue_1 / total_variance)
        if 'explained_1' in cg.columns:
            concentration = cg['explained_1'].to_numpy()
        elif 'eigenvalue_1' in cg.columns:
            ev1 = cg['eigenvalue_1'].to_numpy()
            concentration = np.where(E > 0, ev1 / E, np.nan)
        else:
            concentration = np.full(n, np.nan)
        
        for i in range(n):
            row = {
                'cohort': cohort,
                'I': int(I_vals[i]),
                'energy': float(E[i]),
                'entropy': float(S[i]),
                'temperature': float(T[i]) if not np.isnan(T[i]) else None,
                'free_energy': float(F[i]) if not np.isnan(F[i]) else None,
                'heat_capacity': float(Cv[i]) if not np.isnan(Cv[i]) else None,
                'concentration': float(concentration[i]) if not np.isnan(concentration[i]) else None,
            }
            rows.append(row)
    
    return pl.DataFrame(rows)


def detect_phase_transitions(thermo: pl.DataFrame) -> pl.DataFrame:
    """
    Detect phase transitions per cohort.
    
    Signatures:
      - Temperature spike (|T| > 2σ from cohort mean)
      - Free energy approaching zero
      - Heat capacity divergence (|Cᵥ| > 3σ)
    """
    cohorts = sorted(thermo['cohort'].unique().to_list())
    
    rows = []
    
    for cohort in cohorts:
        ct = thermo.filter(pl.col('cohort') == cohort).sort('I')
        
        if len(ct) < 5:
            continue
        
        T = ct['temperature'].to_numpy()
        F = ct['free_energy'].to_numpy()
        Cv = ct['heat_capacity'].to_numpy()
        I_vals = ct['I'].to_numpy()
        n = len(T)
        
        # Temperature spikes: |T| > mean + 2σ
        T_clean = T[~np.isnan(T)]
        if len(T_clean) > 3:
            T_mean = np.nanmean(T_clean)
            T_std = np.nanstd(T_clean)
            T_threshold = abs(T_mean) + 2 * T_std
            temp_spikes = np.where(np.abs(T) > T_threshold)[0] if T_std > 0 else []
        else:
            temp_spikes = []
        
        # Free energy zero-crossing or minimum
        F_clean = F[~np.isnan(F)]
        if len(F_clean) > 3:
            F_min_idx = np.nanargmin(np.abs(F_clean))
        else:
            F_min_idx = None
        
        # Heat capacity divergence: |Cᵥ| > mean + 3σ
        Cv_clean = Cv[~np.isnan(Cv)]
        if len(Cv_clean) > 3:
            Cv_mean = np.nanmean(np.abs(Cv_clean))
            Cv_std = np.nanstd(np.abs(Cv_clean))
            Cv_threshold = Cv_mean + 3 * Cv_std
            cv_spikes = np.where(np.abs(Cv) > Cv_threshold)[0] if Cv_std > 0 else []
        else:
            cv_spikes = []
        
        # Earliest detection: which signature fires first?
        detections = []
        
        if len(temp_spikes) > 0:
            first_T = temp_spikes[0]
            detections.append(('temperature_spike', int(I_vals[first_T]), first_T / n))
        
        if len(cv_spikes) > 0:
            first_Cv = cv_spikes[0]
            detections.append(('heat_capacity_divergence', int(I_vals[first_Cv]), first_Cv / n))
        
        detections.sort(key=lambda x: x[1])  # sort by I
        
        row = {
            'cohort': cohort,
            'n_windows': n,
            'n_temperature_spikes': len(temp_spikes),
            'n_heat_capacity_spikes': len(cv_spikes),
            'first_detection_type': detections[0][0] if detections else None,
            'first_detection_I': detections[0][1] if detections else None,
            'first_detection_pct_life': float(detections[0][2]) if detections else None,
            'mean_free_energy': float(np.nanmean(F_clean)) if len(F_clean) > 0 else None,
            'final_free_energy': float(F_clean[-1]) if len(F_clean) > 0 else None,
        }
        rows.append(row)
    
    return pl.DataFrame(rows)


def print_summary(thermo: pl.DataFrame, transitions: pl.DataFrame):
    """Print thermodynamic analysis summary."""
    
    print()
    print("=" * 70)
    print("THERMODYNAMIC ANALYSIS")
    print("=" * 70)
    
    n_cohorts = thermo['cohort'].n_unique()
    print(f"\n  Cohorts: {n_cohorts}")
    
    # Fleet-level statistics
    print()
    print("── Fleet Thermodynamic Profile ──")
    print()
    
    for qty in ['energy', 'entropy', 'temperature', 'free_energy']:
        if qty in thermo.columns:
            vals = thermo[qty].drop_nulls().to_numpy()
            if len(vals) > 0:
                print(f"  {qty:>15s}: mean={np.mean(vals):.3f}  std={np.std(vals):.3f}  "
                      f"range=[{np.min(vals):.3f}, {np.max(vals):.3f}]")
    
    # Phase transitions
    if len(transitions) > 0:
        detected = transitions.filter(pl.col('first_detection_type').is_not_null())
        
        print()
        print("── Phase Transition Detection ──")
        print()
        print(f"  Cohorts with detected transitions: {len(detected)}/{len(transitions)}")
        
        if len(detected) > 0:
            # How early are transitions detected?
            pct_life = detected['first_detection_pct_life'].drop_nulls().to_numpy()
            if len(pct_life) > 0:
                print(f"  Mean detection point: {np.mean(pct_life)*100:.1f}% of life")
                print(f"  Earliest detection:   {np.min(pct_life)*100:.1f}% of life")
                print(f"  Latest detection:     {np.max(pct_life)*100:.1f}% of life")
            
            # Detection type breakdown
            print()
            types = detected['first_detection_type'].value_counts()
            for row in types.iter_rows(named=True):
                print(f"  {row['first_detection_type']:>30s}: {row['count']} cohorts")
        
        # Temperature spike frequency
        spike_counts = transitions['n_temperature_spikes'].to_numpy()
        print()
        print(f"  Temperature spikes per cohort:  mean={np.mean(spike_counts):.1f}  "
              f"max={np.max(spike_counts)}")
        
        cv_counts = transitions['n_heat_capacity_spikes'].to_numpy()
        print(f"  Heat capacity spikes per cohort: mean={np.mean(cv_counts):.1f}  "
              f"max={np.max(cv_counts)}")
    
    # Free energy trajectory
    print()
    print("── Free Energy at End of Life ──")
    print()
    if 'final_free_energy' in transitions.columns:
        final_F = transitions['final_free_energy'].drop_nulls().to_numpy()
        if len(final_F) > 0:
            n_near_zero = np.sum(np.abs(final_F) < np.std(final_F))
            print(f"  Mean final free energy: {np.mean(final_F):.3f}")
            print(f"  Cohorts near F=0 at EOL: {n_near_zero}/{len(final_F)}")
            print(f"  (F→0 means all energy, no freedom = compressed spring)")
    
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Framework Thermodynamic Analysis')
    parser.add_argument('--geometry', required=True, help='Path to state_geometry.parquet')
    parser.add_argument('--output', default='thermodynamics.parquet', help='Output path')
    parser.add_argument('--engine', default=None, help='Filter to engine group')
    args = parser.parse_args()
    
    print(f"Loading geometry: {args.geometry}")
    geometry = pl.read_parquet(args.geometry)
    print(f"  {len(geometry)} rows, {geometry['cohort'].n_unique()} cohorts")
    
    # Compute thermodynamics
    print("\nComputing thermodynamic quantities...")
    thermo = compute_thermodynamics(geometry, args.engine)
    
    # Detect phase transitions
    print("Detecting phase transitions...")
    transitions = detect_phase_transitions(thermo)
    
    print_summary(thermo, transitions)
    
    # Save
    thermo.write_parquet(args.output)
    print(f"\nSaved: {args.output}")
    
    transitions_path = args.output.replace('.parquet', '_transitions.parquet')
    transitions.write_parquet(transitions_path)
    print(f"Saved: {transitions_path}")


if __name__ == '__main__':
    main()
