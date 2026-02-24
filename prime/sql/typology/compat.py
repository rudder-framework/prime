"""
Compatibility shim: SQL typology output → manifest generator expectations.

The SQL typology pipeline produces columns with slightly different names
than what prime/manifest/generator.py reads. This module renames/adds
columns so the manifest generator works without modification.
"""

import polars as pl
from pathlib import Path


def adapt_for_manifest(typology_path: str) -> None:
    """
    Read typology.parquet, rename/add columns for manifest generator
    compatibility, and overwrite the file in place.

    Renames:
        spectral_class      → spectral
        dominant_frequency   → dominant_freq
        n_obs                → n_samples

    Adds:
        temporal_primary     = temporal_pattern (scalar form)
    """
    path = Path(typology_path)
    df = pl.read_parquet(path)

    # spectral_class → spectral
    if "spectral_class" in df.columns and "spectral" not in df.columns:
        df = df.rename({"spectral_class": "spectral"})

    # dominant_frequency → dominant_freq
    if "dominant_frequency" in df.columns and "dominant_freq" not in df.columns:
        df = df.rename({"dominant_frequency": "dominant_freq"})

    # n_obs → n_samples (generator reads n_samples for window sizing)
    if "n_obs" in df.columns and "n_samples" not in df.columns:
        df = df.with_columns(pl.col("n_obs").alias("n_samples"))

    # temporal_primary = temporal_pattern (generator reads temporal_primary)
    if "temporal_pattern" in df.columns and "temporal_primary" not in df.columns:
        df = df.with_columns(pl.col("temporal_pattern").alias("temporal_primary"))

    df.write_parquet(path)
