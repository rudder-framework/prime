#!/usr/bin/env python3
"""
Framework Analysis: Canary Signal Identification

Reads Manifold's signal_vector.parquet and state_geometry.parquet to identify
which sensor drives dimensional collapse per cohort.

"Canary" = the signal whose behavior change predicts system-level geometric
deformation. Not SHAP post-hoc. Structural causality.

Methods:
  1. Per-signal d1 velocity — which signal is changing fastest in early life?
  2. Per-signal correlation with eff_dim — which signal's features track collapse?
  3. Single-signal RUL predictive power — which signal alone predicts lifecycle?

Input:
    signal_vector.parquet   — per-signal windowed features (from Manifold)
    state_geometry.parquet  — per-cohort eigenvalue trajectories (from Manifold)
    observations.parquet    — for lifecycle / RUL ground truth

Output:
    canary_results.parquet  — per-signal importance ranking
    Console summary

Usage:
    python -m prime.analysis.canary \
        --signal-vector path/to/signal_vector.parquet \
        --geometry path/to/state_geometry.parquet \
        --observations path/to/observations.parquet \
        --output path/to/canary_results.parquet
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats
from collections import defaultdict


def get_lifecycle_per_cohort(observations: pl.DataFrame) -> dict:
    """Returns dict mapping cohort -> lifecycle_length."""
    first_signal = observations['signal_id'].unique().sort()[0]
    
    lifecycle = (
        observations
        .filter(pl.col('signal_id') == first_signal)
        .group_by('cohort')
        .agg([
            pl.col('signal_0').max().alias('max_signal_0'),
            pl.col('signal_0').min().alias('min_signal_0'),
        ])
        .with_columns(
            (pl.col('max_signal_0') - pl.col('min_signal_0') + 1).alias('lifecycle_length')
        )
    )
    
    return dict(zip(
        lifecycle['cohort'].to_list(),
        lifecycle['lifecycle_length'].to_list(),
    ))


def analyze_signal_velocity(signal_vector: pl.DataFrame, lifecycles: dict) -> pl.DataFrame:
    """
    Method 1: Per-signal rate of change (d1 velocity).
    
    For each signal in each cohort, compute mean absolute d1 in the first 20%
    and last 20% of life. The signal with the largest late-life acceleration
    is the canary.
    """
    signals = sorted(signal_vector['signal_id'].unique().to_list())
    cohorts = sorted(signal_vector['cohort'].unique().to_list())
    
    # We need a feature that captures rate of change
    # Look for d1-related columns, or compute from 'mean' if available
    value_col = None
    for candidate in ['d1', 'rate_of_change', 'mean', 'value']:
        if candidate in signal_vector.columns:
            value_col = candidate
            break
    
    if value_col is None:
        # Fall back to any numeric column that's not metadata
        meta = {'signal_id', 'signal_0', 'cohort', 'unit_id', 'n_samples', 'window_size'}
        numeric_cols = [c for c in signal_vector.columns if c not in meta]
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            raise ValueError("No numeric feature columns found in signal_vector")
    
    print(f"  Using '{value_col}' for velocity analysis")
    
    rows = []
    
    for signal in signals:
        early_velocities = []
        late_velocities = []
        full_velocities = []
        
        for cohort in cohorts:
            life = lifecycles.get(cohort)
            if life is None:
                continue
            
            sig_data = (
                signal_vector
                .filter(
                    (pl.col('signal_id') == signal) & 
                    (pl.col('cohort') == cohort)
                )
                .sort('signal_0')
            )
            
            if len(sig_data) < 5:
                continue
            
            values = sig_data[value_col].to_numpy()
            values = values[~np.isnan(values)]
            
            if len(values) < 5:
                continue
            
            # Compute velocity (first differences)
            velocity = np.abs(np.diff(values))
            n = len(velocity)
            n_early = max(1, int(n * 0.20))
            n_late = max(1, int(n * 0.20))
            
            early_v = float(np.mean(velocity[:n_early]))
            late_v = float(np.mean(velocity[-n_late:]))
            full_v = float(np.mean(velocity))
            
            early_velocities.append(early_v)
            late_velocities.append(late_v)
            full_velocities.append(full_v)
        
        if len(early_velocities) < 3:
            continue
        
        rows.append({
            'signal_id': signal,
            'mean_early_velocity': float(np.mean(early_velocities)),
            'mean_late_velocity': float(np.mean(late_velocities)),
            'mean_full_velocity': float(np.mean(full_velocities)),
            'velocity_acceleration': float(np.mean(late_velocities)) - float(np.mean(early_velocities)),
            'velocity_ratio': float(np.mean(late_velocities)) / max(float(np.mean(early_velocities)), 1e-10),
            'n_cohorts': len(early_velocities),
        })
    
    return pl.DataFrame(rows).sort('velocity_acceleration', descending=True)


def analyze_signal_collapse_correlation(
    signal_vector: pl.DataFrame,
    geometry: pl.DataFrame,
    engine_name: str = None,
) -> pl.DataFrame:
    """
    Method 2: Which signal's features correlate with eff_dim trajectory?
    
    For each signal, compute correlation between signal feature values
    and system-level effective_dim over time within each cohort.
    The signal most strongly anti-correlated with eff_dim is the one
    whose change drives dimensional collapse.
    """
    # Filter geometry to engine group if specified
    if engine_name and 'engine' in geometry.columns:
        geo = geometry.filter(pl.col('engine') == engine_name)
    else:
        geo = geometry
    
    signals = sorted(signal_vector['signal_id'].unique().to_list())
    cohorts = sorted(set(signal_vector['cohort'].unique().to_list()) & 
                     set(geo['cohort'].unique().to_list()))
    
    # Find a good feature column
    meta = {'signal_id', 'signal_0', 'cohort', 'unit_id', 'n_samples', 'window_size'}
    feature_cols = [c for c in signal_vector.columns if c not in meta
                    and signal_vector[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    
    if not feature_cols:
        raise ValueError("No numeric feature columns in signal_vector")
    
    # Use multiple features if available
    priority_features = ['hurst', 'spectral_entropy', 'kurtosis', 'permutation_entropy', 'sample_entropy', 'mean']
    use_features = [f for f in priority_features if f in feature_cols]
    if not use_features:
        use_features = feature_cols[:3]
    
    print(f"  Correlating signals with eff_dim using features: {use_features}")
    
    rows = []
    
    for signal in signals:
        correlations_by_feature = defaultdict(list)
        
        for cohort in cohorts:
            # Get eff_dim trajectory for this cohort
            cohort_geo = geo.filter(pl.col('cohort') == cohort).sort('signal_0')
            if len(cohort_geo) < 5:
                continue
            
            eff_dim_values = cohort_geo['effective_dim'].to_numpy()
            geo_signal_0 = cohort_geo['signal_0'].to_numpy()
            
            # Get signal features for this cohort
            sig_data = (
                signal_vector
                .filter(
                    (pl.col('signal_id') == signal) &
                    (pl.col('cohort') == cohort)
                )
                .sort('signal_0')
            )
            
            if len(sig_data) < 5:
                continue
            
            sig_signal_0 = sig_data['signal_0'].to_numpy()

            # Align on shared signal_0 values
            shared_signal_0 = np.intersect1d(geo_signal_0, sig_signal_0)
            if len(shared_signal_0) < 5:
                continue

            geo_mask = np.isin(geo_signal_0, shared_signal_0)
            sig_mask = np.isin(sig_signal_0, shared_signal_0)
            eff_dim_aligned = eff_dim_values[geo_mask]
            
            for feat in use_features:
                if feat not in sig_data.columns:
                    continue
                feat_values = sig_data[feat].to_numpy()[sig_mask]
                
                # Remove NaN pairs
                valid = ~(np.isnan(feat_values) | np.isnan(eff_dim_aligned))
                if valid.sum() < 5:
                    continue
                
                r, _ = stats.pearsonr(feat_values[valid], eff_dim_aligned[valid])
                correlations_by_feature[feat].append(r)
        
        if not correlations_by_feature:
            continue
        
        row = {'signal_id': signal}
        abs_r_values = []
        
        for feat, r_list in correlations_by_feature.items():
            mean_r = float(np.mean(r_list))
            row[f'r_{feat}_effdim'] = mean_r
            abs_r_values.append(abs(mean_r))
        
        row['mean_abs_r_effdim'] = float(np.mean(abs_r_values)) if abs_r_values else 0.0
        row['n_cohorts'] = max(len(v) for v in correlations_by_feature.values())
        rows.append(row)
    
    return pl.DataFrame(rows).sort('mean_abs_r_effdim', descending=True)


def analyze_single_signal_rul(
    signal_vector: pl.DataFrame,
    lifecycles: dict,
) -> pl.DataFrame:
    """
    Method 3: Single-signal RUL predictive power.
    
    For each signal, use ONLY that signal's early-life features
    to predict lifecycle length via simple linear regression.
    The signal with highest R² is the strongest single predictor.
    """
    signals = sorted(signal_vector['signal_id'].unique().to_list())
    cohorts = sorted(signal_vector['cohort'].unique().to_list())
    
    # Find numeric features
    meta = {'signal_id', 'signal_0', 'cohort', 'unit_id', 'n_samples', 'window_size'}
    feature_cols = [c for c in signal_vector.columns if c not in meta
                    and signal_vector[c].dtype in [pl.Float64, pl.Float32]]
    
    if not feature_cols:
        return pl.DataFrame()
    
    rows = []
    
    for signal in signals:
        # Collect early-life feature means per cohort
        cohort_features = []
        cohort_lifecycles = []
        
        for cohort in cohorts:
            life = lifecycles.get(cohort)
            if life is None:
                continue
            
            sig_data = (
                signal_vector
                .filter(
                    (pl.col('signal_id') == signal) &
                    (pl.col('cohort') == cohort)
                )
                .sort('signal_0')
            )
            
            if len(sig_data) < 3:
                continue
            
            # Take first 20% of windows
            n_early = max(1, int(len(sig_data) * 0.20))
            early = sig_data.head(n_early)
            
            # Mean of each feature in early life
            feat_means = []
            for f in feature_cols:
                vals = early[f].to_numpy()
                vals = vals[~np.isnan(vals)]
                feat_means.append(float(np.mean(vals)) if len(vals) > 0 else np.nan)
            
            if any(np.isnan(feat_means)):
                continue
            
            cohort_features.append(feat_means)
            cohort_lifecycles.append(life)
        
        if len(cohort_features) < 10:
            continue
        
        X = np.array(cohort_features)
        y = np.array(cohort_lifecycles)
        
        # Per-feature correlation with lifecycle
        best_r = 0.0
        best_feat = ''
        
        for j, feat in enumerate(feature_cols):
            r, p = stats.pearsonr(X[:, j], y)
            if abs(r) > abs(best_r):
                best_r = float(r)
                best_feat = feat
        
        # Overall R² using all features (simple: use best single feature)
        rows.append({
            'signal_id': signal,
            'best_feature': best_feat,
            'best_r': best_r,
            'best_r_squared': best_r ** 2,
            'n_cohorts': len(cohort_lifecycles),
        })
    
    return pl.DataFrame(rows).sort('best_r_squared', descending=True)


def print_summary(velocity: pl.DataFrame, correlation: pl.DataFrame, rul: pl.DataFrame):
    """Print canary analysis summary."""
    
    print()
    print("=" * 70)
    print("CANARY SIGNAL IDENTIFICATION")
    print("=" * 70)
    
    print()
    print("── Method 1: Velocity Acceleration (which signal changes fastest late in life?) ──")
    print()
    if len(velocity) > 0:
        for row in velocity.head(5).iter_rows(named=True):
            ratio = row['velocity_ratio']
            print(f"  {row['signal_id']:>12s}  acceleration={row['velocity_acceleration']:.4f}  "
                  f"ratio(late/early)={ratio:.2f}  "
                  f"(n={row['n_cohorts']})")
    
    print()
    print("── Method 2: Collapse Correlation (which signal tracks eff_dim?) ──")
    print()
    if len(correlation) > 0:
        for row in correlation.head(5).iter_rows(named=True):
            print(f"  {row['signal_id']:>12s}  |r|={row['mean_abs_r_effdim']:.3f}  "
                  f"(n={row['n_cohorts']})")
    
    print()
    print("── Method 3: Single-Signal RUL Prediction (which signal alone predicts lifecycle?) ──")
    print()
    if len(rul) > 0:
        for row in rul.head(5).iter_rows(named=True):
            print(f"  {row['signal_id']:>12s}  R²={row['best_r_squared']:.3f}  "
                  f"(best feature: {row['best_feature']}, r={row['best_r']:.3f}, "
                  f"n={row['n_cohorts']})")
    
    # Consensus canary
    print()
    print("── Consensus ──")
    print()
    
    # Score each signal across methods
    scores = defaultdict(float)
    
    if len(velocity) > 0:
        for rank, row in enumerate(velocity.head(5).iter_rows(named=True)):
            scores[row['signal_id']] += (5 - rank)
    
    if len(correlation) > 0:
        for rank, row in enumerate(correlation.head(5).iter_rows(named=True)):
            scores[row['signal_id']] += (5 - rank)
    
    if len(rul) > 0:
        for rank, row in enumerate(rul.head(5).iter_rows(named=True)):
            scores[row['signal_id']] += (5 - rank)
    
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    if ranked:
        canary = ranked[0][0]
        print(f"  🐤 CANARY SIGNAL: {canary} (consensus score: {ranked[0][1]:.0f})")
        print()
        for sig, score in ranked[:5]:
            bar = "█" * int(score)
            print(f"  {sig:>12s}  {bar} ({score:.0f})")
    
    print()
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Framework Canary Signal Identification')
    parser.add_argument('--signal-vector', required=True, help='Path to signal_vector.parquet')
    parser.add_argument('--geometry', required=True, help='Path to state_geometry.parquet')
    parser.add_argument('--observations', required=True, help='Path to observations.parquet')
    parser.add_argument('--output', default='canary_results.parquet', help='Output path')
    parser.add_argument('--engine', default=None, help='Filter geometry to engine group')
    args = parser.parse_args()
    
    print(f"Loading signal_vector: {args.signal_vector}")
    signal_vector = pl.read_parquet(args.signal_vector)
    
    print(f"Loading geometry: {args.geometry}")
    geometry = pl.read_parquet(args.geometry)
    
    print(f"Loading observations: {args.observations}")
    observations = pl.read_parquet(args.observations)
    
    lifecycles = get_lifecycle_per_cohort(observations)
    print(f"  {len(lifecycles)} cohorts")
    
    # Run all three methods
    print("\nMethod 1: Signal velocity...")
    velocity = analyze_signal_velocity(signal_vector, lifecycles)
    
    print("Method 2: Collapse correlation...")
    correlation = analyze_signal_collapse_correlation(signal_vector, geometry, args.engine)
    
    print("Method 3: Single-signal RUL prediction...")
    rul = analyze_single_signal_rul(signal_vector, lifecycles)
    
    print_summary(velocity, correlation, rul)
    
    # Save combined results
    # Join all three on signal_id
    if len(velocity) > 0 and len(rul) > 0:
        combined = velocity.join(rul, on='signal_id', how='outer', suffix='_rul')
        if len(correlation) > 0:
            combined = combined.join(correlation, on='signal_id', how='outer', suffix='_corr')
        combined.write_parquet(args.output)
        print(f"Saved: {args.output}")


if __name__ == '__main__':
    main()
