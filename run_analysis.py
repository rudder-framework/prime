#!/usr/bin/env python3
"""
Prime Analysis: Run All
=======================
Reads Manifold outputs (signal_vector, state_vector, state_geometry)
and observations.parquet. Produces analysis parquets.

Zero Manifold imports. Just polars, numpy, scipy.

Usage:
    python run_analysis.py --data ~/data/FD001

Expects in --data directory:
    observations.parquet
    signal_vector.parquet
    state_vector.parquet      (optional)
    state_geometry.parquet

Produces in --data/analysis/:
    twenty_twenty.parquet          — per-cohort early/late eff_dim + lifecycle
    feature_importance.parquet     — single-feature correlation with lifecycle
    detection_sensitivity.parquet  — optimal early_pct sweep
    canary_signals.parquet         — which signals drive collapse
    thermodynamics.parquet         — E, S, T, F, Cv per cohort per timestep
    analysis_summary.json          — key results for quick reference
"""

import argparse
import json
import numpy as np
import polars as pl
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# ============================================================
# HELPERS
# ============================================================

def get_lifecycles(observations: pl.DataFrame) -> dict:
    """Get lifecycle length per cohort."""
    first_sig = observations['signal_id'].unique().sort()[0]
    lc = (
        observations
        .filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('I').max() + 1)
    )
    return dict(lc.iter_rows())


def bootstrap_ci(x, y, n_boot=10000, seed=42):
    """Bootstrap 95% CI for Pearson r."""
    rng = np.random.RandomState(seed)
    x, y = np.array(x), np.array(y)
    rs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(x), len(x))
        try:
            r, _ = stats.pearsonr(x[idx], y[idx])
            rs.append(r)
        except:
            pass
    rs = np.array(rs)
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


# ============================================================
# 1. TWENTY-TWENTY
# ============================================================

def run_twenty_twenty(geo: pl.DataFrame, lifecycles: dict, early_pct=0.20):
    """Per-cohort 20/20 analysis across all engines and per-engine."""

    results = []

    # Engine groups + combined
    engines = geo['engine'].unique().sort().to_list() if 'engine' in geo.columns else []
    engine_groups = [None] + engines  # None = combined

    for engine_filter in engine_groups:
        label = engine_filter or 'combined'

        if engine_filter:
            g = geo.filter(pl.col('engine') == engine_filter)
        else:
            g = geo.group_by(['cohort', 'I']).agg([
                pl.col(c).mean() for c in geo.columns
                if c not in ['cohort', 'I', 'engine', 'n_signals', 'n_features', 'signal_ids']
                and geo[c].dtype in [pl.Float64, pl.Float32, pl.Int64]
            ])

        for cohort in sorted(lifecycles.keys()):
            cg = g.filter(pl.col('cohort') == cohort).sort('I')
            if len(cg) < 3 or 'effective_dim' not in cg.columns:
                continue

            eff = cg['effective_dim'].to_numpy()
            n = len(eff)
            n_early = max(1, int(n * early_pct))
            n_late = max(1, int(n * early_pct))

            early_mean = float(np.nanmean(eff[:n_early]))
            late_mean = float(np.nanmean(eff[-n_late:]))

            # Also grab total_variance if available
            tv_early = tv_late = np.nan
            if 'total_variance' in cg.columns:
                tv = cg['total_variance'].to_numpy()
                tv_early = float(np.nanmean(tv[:n_early]))
                tv_late = float(np.nanmean(tv[-n_late:]))

            # Eigenvector flip count
            flip_early = flip_total = np.nan
            if 'eigenvector_flip_count' in cg.columns:
                flips = cg['eigenvector_flip_count'].to_numpy()
                flip_early = float(np.nanmean(flips[:n_early]))
                flip_total = float(np.nansum(flips))

            # Eigenvalue entropy
            ee_early = np.nan
            if 'eigenvalue_entropy' in cg.columns:
                ee = cg['eigenvalue_entropy'].to_numpy()
                ee_early = float(np.nanmean(ee[:n_early]))

            results.append({
                'cohort': cohort,
                'engine': label,
                'lifecycle': lifecycles[cohort],
                'n_windows': n,
                'early_eff_dim': early_mean,
                'late_eff_dim': late_mean,
                'delta_eff_dim': early_mean - late_mean,
                'collapse': early_mean > late_mean,
                'early_total_variance': tv_early,
                'late_total_variance': tv_late,
                'delta_total_variance': tv_early - tv_late if not np.isnan(tv_early) else np.nan,
                'early_flip_count': flip_early,
                'total_flip_count': flip_total,
                'early_eigenvalue_entropy': ee_early,
            })

    return pl.DataFrame(results)


# ============================================================
# 2. FEATURE IMPORTANCE
# ============================================================

def run_feature_importance(geo: pl.DataFrame, lifecycles: dict, early_pct=0.20):
    """Which geometry feature best predicts lifecycle?"""

    skip_cols = {'I', 'engine', 'cohort', 'n_signals', 'n_features', 'signal_ids'}
    feature_cols = [c for c in geo.columns
                    if c not in skip_cols
                    and geo[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    rows = []

    for feat in sorted(feature_cols):
        # Average across engines
        g = geo.group_by(['cohort', 'I']).agg(pl.col(feat).mean())

        early_vals = []
        lives = []

        for cohort in sorted(lifecycles.keys()):
            cg = g.filter(pl.col('cohort') == cohort).sort('I')
            if len(cg) < 3:
                continue
            vals = cg[feat].to_numpy()
            n_early = max(1, int(len(vals) * early_pct))
            early = float(np.nanmean(vals[:n_early]))
            if not np.isnan(early) and np.isfinite(early):
                early_vals.append(early)
                lives.append(lifecycles[cohort])

        if len(early_vals) >= 20:
            try:
                r, p = stats.pearsonr(early_vals, lives)
                rho, p_rho = stats.spearmanr(early_vals, lives)

                if np.isnan(r):
                    continue

                ci_low, ci_high = bootstrap_ci(early_vals, lives, n_boot=5000)

                rows.append({
                    'feature': feat,
                    'r': float(r),
                    'r_squared': float(r ** 2),
                    'p_value': float(p),
                    'spearman_rho': float(rho),
                    'p_spearman': float(p_rho),
                    'ci_low': ci_low,
                    'ci_high': ci_high,
                    'n_cohorts': len(early_vals),
                    'direction': 'longer_life' if r < 0 else 'shorter_life',
                })
            except:
                pass

    df = pl.DataFrame(rows)
    if len(df) > 0:
        df = df.sort('r_squared', descending=True)
    return df


# ============================================================
# 3. DETECTION SENSITIVITY
# ============================================================

def run_detection_sensitivity(geo: pl.DataFrame, lifecycles: dict):
    """Sweep early_pct to find optimal detection window."""

    # Get top features to sweep
    top_features = ['effective_dim', 'total_variance', 'eigenvalue_entropy', 'eigenvector_flip_count']
    available_features = [f for f in top_features if f in geo.columns]

    rows = []

    for pct in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        for feat in available_features:
            g = geo.group_by(['cohort', 'I']).agg(pl.col(feat).mean())

            early_vals = []
            lives = []

            for cohort in sorted(lifecycles.keys()):
                cg = g.filter(pl.col('cohort') == cohort).sort('I')
                if len(cg) < 3:
                    continue
                vals = cg[feat].to_numpy()
                n_early = max(1, int(len(vals) * pct))
                early = float(np.nanmean(vals[:n_early]))
                if not np.isnan(early) and np.isfinite(early):
                    early_vals.append(early)
                    lives.append(lifecycles[cohort])

            if len(early_vals) >= 20:
                try:
                    r, p = stats.pearsonr(early_vals, lives)
                    if not np.isnan(r):
                        rows.append({
                            'feature': feat,
                            'early_pct': pct,
                            'r': float(r),
                            'r_squared': float(r ** 2),
                            'p_value': float(p),
                            'n_cohorts': len(early_vals),
                        })
                except:
                    pass

    return pl.DataFrame(rows)


# ============================================================
# 4. CANARY SIGNALS
# ============================================================

def run_canary(signal_vector: pl.DataFrame, geo: pl.DataFrame, lifecycles: dict):
    """Which signal drives collapse?"""

    # Get feature columns from signal_vector
    skip_cols = {'signal_id', 'I', 'cohort', 'unit_id'}
    feat_cols = [c for c in signal_vector.columns
                 if c not in skip_cols
                 and signal_vector[c].dtype in [pl.Float64, pl.Float32]]

    if not feat_cols:
        return pl.DataFrame()

    signals = sorted(signal_vector['signal_id'].unique().to_list())

    rows = []

    for signal_id in signals:
        sig_data = signal_vector.filter(pl.col('signal_id') == signal_id)

        for feat in feat_cols[:8]:  # Top 8 features for speed
            if feat not in sig_data.columns:
                continue

            early_vals = []
            lives = []

            for cohort in sorted(lifecycles.keys()):
                cg = sig_data.filter(pl.col('cohort') == cohort).sort('I')
                if len(cg) < 5:
                    continue

                vals = cg[feat].to_numpy()
                n = len(vals)
                n_early = max(1, int(n * 0.20))
                n_late = max(1, int(n * 0.20))

                early = float(np.nanmean(vals[:n_early]))
                late = float(np.nanmean(vals[-n_late:]))

                if np.isnan(early) or np.isnan(late):
                    continue

                # Use velocity (late - early) as predictor
                velocity = late - early
                if np.isfinite(velocity):
                    early_vals.append(velocity)
                    lives.append(lifecycles[cohort])

            if len(early_vals) >= 20:
                try:
                    r, p = stats.pearsonr(early_vals, lives)
                    if not np.isnan(r):
                        rows.append({
                            'signal_id': signal_id,
                            'feature': feat,
                            'r_velocity_lifecycle': float(r),
                            'r_squared': float(r ** 2),
                            'p_value': float(p),
                            'n_cohorts': len(early_vals),
                        })
                except:
                    pass

    df = pl.DataFrame(rows)
    if len(df) > 0:
        df = df.sort('r_squared', descending=True)
    return df


# ============================================================
# 5. THERMODYNAMICS
# ============================================================

def run_thermodynamics(geo: pl.DataFrame):
    """E, S, T, F, Cv from eigenvalue trajectories."""

    if 'total_variance' not in geo.columns or 'effective_dim' not in geo.columns:
        return pl.DataFrame()

    # Average across engines
    g = geo.group_by(['cohort', 'I']).agg([
        pl.col('total_variance').mean(),
        pl.col('effective_dim').mean(),
        pl.col('eigenvalue_entropy').mean() if 'eigenvalue_entropy' in geo.columns else pl.lit(None).alias('eigenvalue_entropy'),
    ]).sort(['cohort', 'I'])

    rows = []

    for cohort in sorted(g['cohort'].unique().to_list()):
        cg = g.filter(pl.col('cohort') == cohort).sort('I')

        E = cg['total_variance'].to_numpy()
        S = np.log(cg['effective_dim'].to_numpy())  # ln(eff_dim) as entropy
        I_vals = cg['I'].to_numpy()

        for i in range(len(cg)):
            T = np.nan  # Temperature = dS/dI
            F = np.nan  # Free energy = E - T*S
            Cv = np.nan  # Heat capacity = dE/dT

            if i > 0:
                dS = S[i] - S[i-1]
                dI = I_vals[i] - I_vals[i-1]
                T = dS / dI if dI != 0 else np.nan
                F = E[i] - T * S[i] if not np.isnan(T) else np.nan

            if i > 1:
                dE = E[i] - E[i-1]
                T_prev = (S[i-1] - S[i-2]) / (I_vals[i-1] - I_vals[i-2]) if (I_vals[i-1] - I_vals[i-2]) != 0 else np.nan
                dT = T - T_prev if not np.isnan(T_prev) else np.nan
                Cv = dE / dT if dT != 0 and not np.isnan(dT) else np.nan

            rows.append({
                'cohort': cohort,
                'I': int(I_vals[i]),
                'energy_E': float(E[i]),
                'entropy_S': float(S[i]),
                'temperature_T': float(T) if np.isfinite(T) else None,
                'free_energy_F': float(F) if np.isfinite(F) else None,
                'heat_capacity_Cv': float(Cv) if np.isfinite(Cv) else None,
            })

    return pl.DataFrame(rows)


# ============================================================
# CONSOLE REPORT
# ============================================================

def print_report(tt_df, fi_df, ds_df, canary_df):
    """Print summary to console."""

    print()
    print("=" * 70)
    print("  PRIME ANALYSIS REPORT")
    print("=" * 70)

    # 20/20 summary
    combined = tt_df.filter(pl.col('engine') == 'combined')
    if len(combined) > 0:
        early = combined['early_eff_dim'].to_numpy()
        lives = combined['lifecycle'].to_numpy()
        r, p = stats.pearsonr(early, lives)
        n_collapse = int(combined['collapse'].sum())

        print()
        print("── 20/20 Early Detection ──")
        print(f"  Pearson r:  {r:+.4f}  (p = {p:.2e})")
        print(f"  R²:         {r**2:.4f}")
        print(f"  Collapse:   {n_collapse}/{len(combined)} cohorts")

        # Also total_variance
        tv_early = combined['early_total_variance'].to_numpy()
        valid = ~np.isnan(tv_early)
        if valid.sum() >= 20:
            r_tv, p_tv = stats.pearsonr(tv_early[valid], lives[valid])
            print(f"  total_var r: {r_tv:+.4f}  (R² = {r_tv**2:.4f})")

    # Feature importance top 5
    if len(fi_df) > 0:
        print()
        print("── Feature Importance (top 5) ──")
        for row in fi_df.head(5).iter_rows(named=True):
            print(f"  {row['feature']:>30s}  r={row['r']:+.4f}  R²={row['r_squared']:.4f}  p={row['p_value']:.2e}")

    # Best detection window
    if len(ds_df) > 0:
        print()
        print("── Optimal Detection Window ──")
        best = ds_df.sort('r_squared', descending=True).head(5)
        for row in best.iter_rows(named=True):
            print(f"  {row['feature']:>25s} @ {row['early_pct']:.0%}:  R²={row['r_squared']:.4f}")

    # Top canary signals
    if len(canary_df) > 0:
        print()
        print("── Canary Signals (top 5 signal+feature combos) ──")
        for row in canary_df.head(5).iter_rows(named=True):
            print(f"  {row['signal_id']:>12s} × {row['feature']:>20s}  r={row['r_velocity_lifecycle']:+.4f}  R²={row['r_squared']:.4f}")

    print()
    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Prime Analysis — reads Manifold outputs, produces analysis parquets')
    parser.add_argument('--data', required=True, help='Directory with observations.parquet + Manifold outputs')
    parser.add_argument('--output', default=None, help='Output directory (default: data/analysis)')
    args = parser.parse_args()

    data = Path(args.data)
    out = Path(args.output) if args.output else data / 'analysis'
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  PRIME ANALYSIS")
    print("=" * 70)

    # Load
    print(f"\n  Loading from: {data}")

    obs = pl.read_parquet(str(data / 'observations.parquet'))
    geo = pl.read_parquet(str(data / 'state_geometry.parquet'))

    sv_path = data / 'signal_vector.parquet'
    signal_vector = pl.read_parquet(str(sv_path)) if sv_path.exists() else None

    print(f"  observations:    {obs.shape}")
    print(f"  state_geometry:  {geo.shape}")
    if signal_vector is not None:
        print(f"  signal_vector:   {signal_vector.shape}")

    lifecycles = get_lifecycles(obs)
    print(f"  cohorts: {len(lifecycles)}, lifecycle: {min(lifecycles.values())}–{max(lifecycles.values())}")

    # 1. Twenty-Twenty
    print("\n  [1/5] Twenty-Twenty analysis...")
    tt_df = run_twenty_twenty(geo, lifecycles)
    tt_df.write_parquet(str(out / 'twenty_twenty.parquet'))
    print(f"        → {out / 'twenty_twenty.parquet'} ({len(tt_df)} rows)")

    # 2. Feature importance
    print("  [2/5] Feature importance...")
    fi_df = run_feature_importance(geo, lifecycles)
    fi_df.write_parquet(str(out / 'feature_importance.parquet'))
    print(f"        → {out / 'feature_importance.parquet'} ({len(fi_df)} rows)")

    # 3. Detection sensitivity
    print("  [3/5] Detection sensitivity sweep...")
    ds_df = run_detection_sensitivity(geo, lifecycles)
    ds_df.write_parquet(str(out / 'detection_sensitivity.parquet'))
    print(f"        → {out / 'detection_sensitivity.parquet'} ({len(ds_df)} rows)")

    # 4. Canary signals
    if signal_vector is not None:
        print("  [4/5] Canary signal identification...")
        canary_df = run_canary(signal_vector, geo, lifecycles)
        canary_df.write_parquet(str(out / 'canary_signals.parquet'))
        print(f"        → {out / 'canary_signals.parquet'} ({len(canary_df)} rows)")
    else:
        print("  [4/5] Canary signals — skipped (no signal_vector.parquet)")
        canary_df = pl.DataFrame()

    # 5. Thermodynamics
    print("  [5/5] Thermodynamics...")
    thermo_df = run_thermodynamics(geo)
    thermo_df.write_parquet(str(out / 'thermodynamics.parquet'))
    print(f"        → {out / 'thermodynamics.parquet'} ({len(thermo_df)} rows)")

    # Summary JSON
    summary = {}

    # Best feature
    if len(fi_df) > 0:
        best = fi_df.row(0, named=True)
        summary['best_feature'] = best['feature']
        summary['best_r'] = best['r']
        summary['best_r_squared'] = best['r_squared']
        summary['best_p'] = best['p_value']

    # 20/20 combined
    combined = tt_df.filter(pl.col('engine') == 'combined')
    if len(combined) > 0:
        early = combined['early_eff_dim'].to_numpy()
        lives = combined['lifecycle'].to_numpy()
        r, p = stats.pearsonr(early, lives)
        ci_low, ci_high = bootstrap_ci(early, lives)
        summary['twenty_twenty'] = {
            'r': float(r),
            'r_squared': float(r ** 2),
            'p': float(p),
            'ci_low': ci_low,
            'ci_high': ci_high,
            'n_collapse': int(combined['collapse'].sum()),
            'n_cohorts': len(combined),
        }

    # Total variance (the real star)
    if len(combined) > 0:
        tv = combined['early_total_variance'].to_numpy()
        valid = ~np.isnan(tv)
        if valid.sum() >= 20:
            r_tv, p_tv = stats.pearsonr(tv[valid], lives[valid])
            ci_low, ci_high = bootstrap_ci(tv[valid], lives[valid])
            summary['total_variance'] = {
                'r': float(r_tv),
                'r_squared': float(r_tv ** 2),
                'p': float(p_tv),
                'ci_low': ci_low,
                'ci_high': ci_high,
            }

    # Detection optimal
    if len(ds_df) > 0:
        best_det = ds_df.sort('r_squared', descending=True).row(0, named=True)
        summary['optimal_detection'] = {
            'feature': best_det['feature'],
            'early_pct': best_det['early_pct'],
            'r_squared': best_det['r_squared'],
        }

    # Top canary
    if len(canary_df) > 0:
        top_canary = canary_df.row(0, named=True)
        summary['top_canary'] = {
            'signal_id': top_canary['signal_id'],
            'feature': top_canary['feature'],
            'r_squared': top_canary['r_squared'],
        }

    summary_path = out / 'analysis_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  → {summary_path}")

    # Console report
    print_report(tt_df, fi_df, ds_df, canary_df)

    print(f"\n  All outputs in: {out}")
    print()


if __name__ == '__main__':
    main()
