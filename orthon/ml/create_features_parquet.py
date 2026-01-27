"""
ORTHON → features.parquet
Creates ML-ready feature matrix from ORTHON outputs.

Input files:
  - primitives.parquet (per-signal metrics)
  - typology.parquet (signal classifications)
  - primitives_pairs.parquet (pairwise metrics)

Output:
  - features.parquet (one row per entity, all features as columns)

Usage:
  python create_features_parquet.py --input_dir ./data --output ./features.parquet
"""

import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def load_parquet_safe(path: Path) -> pd.DataFrame | None:
    """Load parquet file if it exists."""
    if path.exists():
        return pd.read_parquet(path)
    print(f"  Warning: {path.name} not found, skipping")
    return None


def pivot_signal_features(df: pd.DataFrame, value_cols: list, prefix: str = '') -> pd.DataFrame:
    """
    Pivot signal-level data to entity-level.

    From:
      entity_id | signal_id | hurst | entropy
      ent1      | sig_a     | 0.7   | 1.2
      ent1      | sig_b     | 0.5   | 0.9

    To:
      entity_id | sig_a_hurst | sig_a_entropy | sig_b_hurst | sig_b_entropy
      ent1      | 0.7         | 1.2           | 0.5         | 0.9
    """
    if df is None or len(df) == 0:
        return pd.DataFrame()

    # Only keep numeric columns for pivoting
    numeric_cols = [c for c in value_cols if c in df.columns and df[c].dtype in ['float64', 'int64', 'bool']]

    pivoted_dfs = []
    for col in numeric_cols:
        pivot = df.pivot_table(
            index='entity_id',
            columns='signal_id',
            values=col,
            aggfunc='first'
        )
        pivot.columns = [f"{prefix}{sig}_{col}" for sig in pivot.columns]
        pivoted_dfs.append(pivot)

    if pivoted_dfs:
        return pd.concat(pivoted_dfs, axis=1).reset_index()
    return pd.DataFrame()


def create_pairwise_features(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize pairwise metrics to entity-level features.

    Creates:
      - Top N causal drivers
      - Average correlation strength
      - Network density metrics
      - Tail dependence summaries
    """
    if pairs_df is None or len(pairs_df) == 0:
        return pd.DataFrame()

    features = []

    for entity_id, group in pairs_df.groupby('entity_id'):
        row = {'entity_id': entity_id}

        # === CORRELATION FEATURES ===
        if 'correlation' in group.columns:
            corr = group['correlation'].dropna()
            row['corr_mean'] = corr.mean()
            row['corr_std'] = corr.std()
            row['corr_max'] = corr.max()
            row['corr_min'] = corr.min()
            row['n_high_corr'] = (corr.abs() > 0.7).sum()
            row['n_anti_corr'] = (corr < -0.5).sum()

        # === CAUSALITY FEATURES ===
        if 'is_causal' in group.columns:
            causal = group[group['is_causal'] == True]
            row['n_causal_pairs'] = len(causal)
            row['causal_density'] = len(causal) / len(group) if len(group) > 0 else 0

            # Top drivers (signals that cause many others)
            if len(causal) > 0:
                drivers = causal.groupby('signal_a').size()
                row['top_driver'] = drivers.idxmax() if len(drivers) > 0 else None
                row['top_driver_n_caused'] = drivers.max() if len(drivers) > 0 else 0
                row['n_drivers'] = (drivers > 3).sum()  # signals causing >3 others

        # === TRANSFER ENTROPY FEATURES ===
        if 'transfer_entropy' in group.columns:
            te = group['transfer_entropy'].dropna()
            row['te_mean'] = te.mean()
            row['te_max'] = te.max()
            row['te_std'] = te.std()
            row['n_high_te'] = (te > 0.1).sum()

        # === COINTEGRATION FEATURES ===
        if 'is_cointegrated' in group.columns:
            row['n_cointegrated'] = group['is_cointegrated'].sum()
            row['coint_ratio'] = group['is_cointegrated'].mean()

        # === TAIL DEPENDENCE FEATURES ===
        if 'lower_tail' in group.columns:
            row['tail_lower_mean'] = group['lower_tail'].mean()
            row['tail_lower_max'] = group['lower_tail'].max()
        if 'upper_tail' in group.columns:
            row['tail_upper_mean'] = group['upper_tail'].mean()
            row['tail_upper_max'] = group['upper_tail'].max()

        # === MUTUAL INFORMATION FEATURES ===
        if 'mutual_info' in group.columns:
            mi = group['mutual_info'].dropna()
            row['mi_mean'] = mi.mean()
            row['mi_max'] = mi.max()

        # === DTW FEATURES ===
        if 'dtw_distance' in group.columns:
            dtw = group['dtw_distance'].dropna()
            row['dtw_mean'] = dtw.mean()
            row['dtw_min'] = dtw.min()  # most similar pair
            row['dtw_max'] = dtw.max()  # most different pair

        features.append(row)

    return pd.DataFrame(features)


def create_typology_summary(typology_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize typology to entity-level features.

    Creates:
      - Counts of each signal type
      - Dataset characteristics
    """
    if typology_df is None or len(typology_df) == 0:
        return pd.DataFrame()

    features = []

    for entity_id, group in typology_df.groupby('entity_id'):
        row = {'entity_id': entity_id}

        # Signal type counts
        if 'standard_form' in group.columns:
            for form in group['standard_form'].dropna().unique():
                row[f'n_{form}'] = (group['standard_form'] == form).sum()

        if 'predictability' in group.columns:
            for pred in group['predictability'].dropna().unique():
                row[f'n_{pred}'] = (group['predictability'] == pred).sum()

        if 'amplitude_continuity' in group.columns:
            for amp in group['amplitude_continuity'].dropna().unique():
                row[f'n_{amp}'] = (group['amplitude_continuity'] == amp).sum()

        # Aggregate numeric stats
        if 'hurst' in group.columns:
            row['hurst_mean'] = group['hurst'].mean()
            row['hurst_std'] = group['hurst'].std()
            row['n_trending'] = (group['hurst'] > 0.6).sum()
            row['n_mean_reverting'] = (group['hurst'] < 0.4).sum()

        features.append(row)

    return pd.DataFrame(features)


def create_signal_level_features(primitives_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot all primitives to wide format (one column per signal+metric).
    """
    if primitives_df is None:
        return pd.DataFrame()

    # Key metrics to pivot
    metrics = [
        'hurst', 'hurst_r2',
        'lyapunov', 'is_chaotic',
        'sample_entropy', 'permutation_entropy',
        'garch_persistence', 'garch_alpha', 'garch_beta',
        'acf_decay_rate', 'half_life',
        'determinism', 'laminarity', 'recurrence_rate',
        'lof_score', 'is_outlier',
        'spectral_centroid', 'dominant_frequency',
        'variance', 'skewness', 'kurtosis'
    ]

    return pivot_signal_features(primitives_df, metrics, prefix='')


def main(input_dir: str, output_path: str, include_signal_level: bool = True):
    """
    Main function to create features.parquet.
    """
    input_dir = Path(input_dir)

    print("=" * 60)
    print("ORTHON -> features.parquet")
    print("=" * 60)

    # Load all available files
    print("\n[1/5] Loading ORTHON outputs...")
    primitives = load_parquet_safe(input_dir / 'primitives.parquet')
    typology = load_parquet_safe(input_dir / 'typology.parquet')
    pairs = load_parquet_safe(input_dir / 'primitives_pairs.parquet')

    # Get entity list
    entities = set()
    if primitives is not None:
        entities.update(primitives['entity_id'].unique())
    if pairs is not None:
        entities.update(pairs['entity_id'].unique())

    print(f"  Found {len(entities)} entities")

    # Create base dataframe
    features = pd.DataFrame({'entity_id': list(entities)})

    # Add signal-level features (pivoted)
    if include_signal_level and primitives is not None:
        print("\n[2/5] Creating signal-level features...")
        signal_features = create_signal_level_features(primitives)
        if len(signal_features) > 0:
            features = features.merge(signal_features, on='entity_id', how='left')
            print(f"  Added {len(signal_features.columns) - 1} signal-level features")

    # Add pairwise summary features
    if pairs is not None:
        print("\n[3/5] Creating pairwise features...")
        pair_features = create_pairwise_features(pairs)
        if len(pair_features) > 0:
            features = features.merge(pair_features, on='entity_id', how='left')
            print(f"  Added {len(pair_features.columns) - 1} pairwise features")

    # Add typology summary features
    if typology is not None:
        print("\n[4/5] Creating typology summary features...")
        typ_features = create_typology_summary(typology)
        if len(typ_features) > 0:
            features = features.merge(typ_features, on='entity_id', how='left')
            print(f"  Added {len(typ_features.columns) - 1} typology features")

    # Save
    print(f"\n[5/5] Saving to {output_path}...")
    features.to_parquet(output_path, index=False)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"""
Output: {output_path}
  Entities: {len(features)}
  Features: {len(features.columns) - 1}

Columns:
  - Signal-level: [signal_id]_[metric] (e.g., xmv_22_hurst)
  - Pairwise: corr_mean, n_causal_pairs, te_max, etc.
  - Typology: n_noise_brown, n_trending, hurst_mean, etc.

Ready for ML training!
""")

    # Print column summary
    print("Feature groups:")
    cols = [c for c in features.columns if c != 'entity_id']

    signal_cols = [c for c in cols if '_hurst' in c or '_entropy' in c or '_lyapunov' in c]
    pair_cols = [c for c in cols if c.startswith('corr_') or c.startswith('te_') or c.startswith('n_causal')]
    typ_cols = [c for c in cols if c.startswith('n_') and c not in pair_cols]

    print(f"  Signal metrics: {len(signal_cols)}")
    print(f"  Pairwise stats: {len(pair_cols)}")
    print(f"  Typology counts: {len(typ_cols)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create ML features from ORTHON outputs')
    parser.add_argument('--input_dir', type=str, default='.', help='Directory with parquet files')
    parser.add_argument('--output', type=str, default='features.parquet', help='Output path')
    parser.add_argument('--no-signal-level', action='store_true', help='Skip per-signal features (smaller output)')

    args = parser.parse_args()
    main(args.input_dir, args.output, include_signal_level=not args.no_signal_level)
