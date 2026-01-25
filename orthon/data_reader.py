"""
ORTHON Data Reader
==================

Read and profile data for PRISM configuration.

Usage:
    # CLI
    orthon-config ./data.csv
    orthon-config ./data.parquet -o prism.yaml

    # Python
    from orthon.data_reader import DataReader
    reader = DataReader()
    reader.read('data.csv')
    profile = reader.profile_data()
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
import numpy as np

import polars as pl

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass
class DataProfile:
    """Profile of uploaded data."""
    n_rows: int
    n_entities: int
    n_signals: int
    n_timestamps: int

    # Per-entity stats
    min_lifecycle: int
    max_lifecycle: int
    mean_lifecycle: float
    median_lifecycle: float

    # Temporal characteristics
    sampling_interval: Optional[float]
    is_regular_sampling: bool

    # Signal characteristics
    signal_names: List[str]
    has_nulls: bool
    null_pct: float


class DataReader:
    """Read and profile data for PRISM."""

    SUPPORTED_FORMATS = ['.csv', '.parquet', '.tsv']

    def __init__(self):
        self.df: Optional[pl.DataFrame] = None
        self.profile: Optional[DataProfile] = None

    def read(self, path: Path) -> pl.DataFrame:
        """Read data from file."""
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        suffix = path.suffix.lower()

        if suffix == '.csv':
            self.df = pl.read_csv(path)
        elif suffix == '.tsv':
            self.df = pl.read_csv(path, separator='\t')
        elif suffix == '.parquet':
            self.df = pl.read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Use: {self.SUPPORTED_FORMATS}")

        return self.df

    def detect_columns(self) -> Dict[str, any]:
        """Auto-detect entity_id, timestamp, signal columns."""
        if self.df is None:
            raise ValueError("No data loaded. Call read() first.")

        columns = self.df.columns
        detected = {}

        # Detect entity_id
        entity_candidates = [
            'entity_id', 'unit_id', 'id', 'machine_id', 'asset_id',
            'battery_id', 'engine_id', 'device_id', 'sensor_id'
        ]
        for col in columns:
            if col.lower() in [c.lower() for c in entity_candidates]:
                detected['entity_id'] = col
                break

        # Detect timestamp
        time_candidates = [
            'timestamp', 'time', 'cycle', 'cycles', 't', 'datetime',
            'date', 'step', 'sample', 'index'
        ]
        for col in columns:
            if col.lower() in [c.lower() for c in time_candidates]:
                detected['timestamp'] = col
                break

        # Remaining columns are signals
        known_cols = set(detected.values())
        detected['signals'] = [c for c in columns if c not in known_cols]

        return detected

    def profile_data(
        self,
        entity_col: Optional[str] = None,
        timestamp_col: Optional[str] = None
    ) -> DataProfile:
        """Generate data profile."""
        if self.df is None:
            raise ValueError("No data loaded. Call read() first.")

        # Auto-detect columns if not specified
        detected = self.detect_columns()
        entity_col = entity_col or detected.get('entity_id')
        timestamp_col = timestamp_col or detected.get('timestamp')
        signal_cols = detected.get('signals', [])

        if not entity_col:
            # Assume single entity
            self.df = self.df.with_columns(pl.lit('entity_1').alias('entity_id'))
            entity_col = 'entity_id'

        if not timestamp_col:
            # Assume row index is timestamp
            self.df = self.df.with_row_index('timestamp')
            timestamp_col = 'timestamp'

        # Basic counts
        n_rows = len(self.df)
        n_entities = self.df[entity_col].n_unique()
        n_signals = len(signal_cols)
        n_timestamps = self.df[timestamp_col].n_unique()

        # Lifecycle per entity
        lifecycle = self.df.group_by(entity_col).agg(
            pl.col(timestamp_col).count().alias('lifecycle')
        )['lifecycle'].to_numpy()

        min_lifecycle = int(np.min(lifecycle))
        max_lifecycle = int(np.max(lifecycle))
        mean_lifecycle = float(np.mean(lifecycle))
        median_lifecycle = float(np.median(lifecycle))

        # Sampling interval (check if regular)
        first_entity = self.df[entity_col][0]
        timestamps = self.df.filter(
            pl.col(entity_col) == first_entity
        )[timestamp_col].to_numpy()

        if len(timestamps) > 1:
            diffs = np.diff(timestamps.astype(float))
            sampling_interval = float(np.median(diffs))
            is_regular = np.std(diffs) < 0.1 * abs(sampling_interval) if sampling_interval != 0 else True
        else:
            sampling_interval = None
            is_regular = True

        # Null check
        null_counts = self.df.null_count().to_numpy().flatten()
        has_nulls = np.any(null_counts > 0)
        total_cells = n_rows * len(self.df.columns)
        null_pct = 100 * float(np.sum(null_counts)) / total_cells if total_cells > 0 else 0

        self.profile = DataProfile(
            n_rows=n_rows,
            n_entities=n_entities,
            n_signals=n_signals,
            n_timestamps=n_timestamps,
            min_lifecycle=min_lifecycle,
            max_lifecycle=max_lifecycle,
            mean_lifecycle=mean_lifecycle,
            median_lifecycle=median_lifecycle,
            sampling_interval=sampling_interval,
            is_regular_sampling=is_regular,
            signal_names=signal_cols,
            has_nulls=has_nulls,
            null_pct=null_pct,
        )

        return self.profile


# =============================================================================
# CLI
# =============================================================================

def print_profile(profile: DataProfile):
    """Print data profile."""
    print("\n" + "=" * 60)
    print("DATA PROFILE")
    print("=" * 60)
    print(f"  Rows:        {profile.n_rows:,}")
    print(f"  Entities:    {profile.n_entities}")
    print(f"  Signals:     {profile.n_signals}")
    print(f"  Timestamps:  {profile.n_timestamps:,}")
    print()
    print(f"  Lifecycle (samples per entity):")
    print(f"    Min:    {profile.min_lifecycle}")
    print(f"    Max:    {profile.max_lifecycle}")
    print(f"    Mean:   {profile.mean_lifecycle:.1f}")
    print(f"    Median: {profile.median_lifecycle:.1f}")
    print()
    print(f"  Sampling: {'Regular' if profile.is_regular_sampling else 'Irregular'}")
    if profile.sampling_interval:
        print(f"    Interval: {profile.sampling_interval}")
    print()
    print(f"  Nulls: {'Yes' if profile.has_nulls else 'No'} ({profile.null_pct:.2f}%)")
    print()
    signals_preview = profile.signal_names[:5]
    more = f"... +{len(profile.signal_names) - 5}" if len(profile.signal_names) > 5 else ""
    print(f"  Signals: {signals_preview}{more}")


def print_recommendation(rec):
    """Print configuration recommendation."""
    print("\n" + "=" * 60)
    print("CONFIGURATION RECOMMENDATION")
    print("=" * 60)

    w = rec.window
    print(f"\n  WINDOWING ({w.confidence.upper()} confidence)")
    print(f"  ---------------------------------")
    print(f"    window_size:   {w.window_size}")
    print(f"    window_stride: {w.window_stride}")
    print(f"    overlap:       {w.overlap_pct:.0f}%")
    print(f"    windows/entity: ~{w.n_windows_approx}")
    print()
    print(f"  Rationale:")
    for line in w.rationale.split('\n'):
        print(f"    {line}")

    print(f"\n  ALTERNATIVES")
    print(f"  ---------------------------------")
    print(f"    Conservative: window={w.conservative['window_size']}, stride={w.conservative['window_stride']}")
    print(f"    Aggressive:   window={w.aggressive['window_size']}, stride={w.aggressive['window_stride']}")

    print(f"\n  OTHER PARAMETERS")
    print(f"  ---------------------------------")
    print(f"    n_clusters: {rec.n_clusters}")
    print(f"    n_regimes:  {rec.n_regimes}")

    print("\n" + "=" * 60)
    print("RECOMMENDED CONFIG")
    print("=" * 60)
    print()
    config = rec.to_dict()
    if HAS_YAML:
        print(yaml.dump(config, default_flow_style=False))
    else:
        print(json.dumps(config, indent=2))


def save_config(rec, path: Path):
    """Save configuration to file."""
    config = rec.to_dict()
    path = Path(path)

    if path.suffix in ['.yaml', '.yml'] and HAS_YAML:
        with open(path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
    else:
        if path.suffix in ['.yaml', '.yml']:
            print("WARNING: pyyaml not installed, saving as JSON")
            path = path.with_suffix('.json')
        with open(path, 'w') as f:
            json.dump(config, f, indent=2)

    print(f"\nConfig saved to: {path}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='ORTHON Data Reader & Config Recommender',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
    orthon-config ./data.csv
    orthon-config ./data.parquet -o prism.yaml

ZERO DEFAULTS POLICY:
    This tool RECOMMENDS configuration values based on your data.
    You MUST review and confirm before using with PRISM.
        """
    )

    parser.add_argument('data_file', nargs='?', help='Path to data file')
    parser.add_argument('--output', '-o', help='Save config to file')
    parser.add_argument('--entity-col', help='Column name for entity ID')
    parser.add_argument('--timestamp-col', help='Column name for timestamp')

    args = parser.parse_args()

    if not args.data_file:
        parser.print_help()
        print("\nTip: Run `streamlit run orthon/app.py` for interactive UI")
        sys.exit(1)

    # Read and profile
    reader = DataReader()
    try:
        reader.read(Path(args.data_file))
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    profile = reader.profile_data(args.entity_col, args.timestamp_col)
    print_profile(profile)

    # Get recommendation
    from orthon.config.recommender import ConfigRecommender
    recommender = ConfigRecommender(profile)
    rec = recommender.recommend()
    print_recommendation(rec)

    if args.output:
        save_config(rec, Path(args.output))
    else:
        print("\nTip: Use --output prism.yaml to save configuration")


if __name__ == '__main__':
    main()
