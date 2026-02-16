"""
MATLAB .mat file converter for streaming pipeline.

Vectorized numpy → polars construction. Does NOT reuse
prime/ingest/upload.py:load_matlab_file() which is row-by-row pandas.

Designed for high-frequency SHM datasets (e.g. LUMO: 22 sensors, 1651 Hz,
~990k samples/file).
"""

import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import polars as pl

from .base_converter import BaseConverter, ConversionResult


class MatConverter(BaseConverter):
    """Convert .mat files to canonical schema using vectorized operations."""

    def __init__(
        self,
        signal_keys: Optional[List[str]] = None,
        exclude_keys: Optional[List[str]] = None,
        cohort_from: str = "config",
        cohort_value: Optional[str] = None,
        sampling_rate: Optional[float] = None,
    ):
        """
        Args:
            signal_keys: Explicit list of .mat keys to use as signals.
                         If None, auto-detect numeric arrays.
            exclude_keys: Keys to skip during auto-detect (e.g. metadata fields).
            cohort_from: How to derive cohort — 'config', 'filename', 'parent'.
            cohort_value: Fixed cohort name when cohort_from='config'.
            sampling_rate: Sampling rate in Hz (informational, stored but not used for I).
        """
        self.signal_keys = signal_keys
        self.exclude_keys = set(exclude_keys or [])
        self.cohort_from = cohort_from
        self.cohort_value = cohort_value
        self.sampling_rate = sampling_rate

    def convert_file(self, filepath: Path, cohort: Optional[str] = None) -> ConversionResult:
        """
        Convert a .mat file to canonical (cohort, signal_id, I, value) DataFrame.

        Vectorized: builds one polars DataFrame per signal, then concatenates.
        Memory: O(n_signals * n_samples) — one file at a time.
        """
        from scipy.io import loadmat

        data = loadmat(str(filepath), squeeze_me=True)

        # Resolve cohort
        if cohort is None:
            cohort = self._resolve_cohort(filepath)

        # Determine which keys are signals
        keys = self._get_signal_keys(data)

        # Build one DataFrame per signal, then vstack
        frames = []
        n_samples = 0
        for key in keys:
            arr = data[key]
            if not isinstance(arr, np.ndarray):
                continue
            arr = arr.flatten().astype(np.float64)
            n = len(arr)
            if n <= 1:
                continue
            n_samples = n

            frame = pl.DataFrame({
                "cohort": pl.Series([cohort] * n, dtype=pl.Utf8),
                "signal_id": pl.Series([key] * n, dtype=pl.Utf8),
                "I": pl.Series(np.arange(n, dtype=np.uint32)),
                "value": pl.Series(arr),
            })
            frames.append(frame)

        if not frames:
            return ConversionResult(
                df=pl.DataFrame(schema={"cohort": pl.Utf8, "signal_id": pl.Utf8, "I": pl.UInt32, "value": pl.Float64}),
                source_file=filepath,
                n_signals=0,
                n_samples_per_signal=0,
            )

        df = pl.concat(frames)
        return ConversionResult(
            df=df,
            source_file=filepath,
            n_signals=len(frames),
            n_samples_per_signal=n_samples,
        )

    def detect_signals(self, filepath: Path) -> List[str]:
        """Detect signal keys in a .mat file."""
        from scipy.io import loadmat

        data = loadmat(str(filepath), squeeze_me=True)
        return self._get_signal_keys(data)

    def file_pattern(self) -> str:
        return "**/*.mat"

    def _get_signal_keys(self, data: Dict) -> List[str]:
        """Return ordered list of signal keys from .mat data dict."""
        if self.signal_keys is not None:
            return [k for k in self.signal_keys if k in data]

        # Auto-detect: numeric arrays, skip metadata
        keys = []
        for key, value in sorted(data.items()):
            if key.startswith("__"):
                continue
            if key in self.exclude_keys:
                continue
            if not isinstance(value, np.ndarray):
                continue
            if value.size <= 1:
                continue
            # Must be numeric
            if not np.issubdtype(value.dtype, np.number):
                continue
            keys.append(key)
        return keys

    def _resolve_cohort(self, filepath: Path) -> str:
        """Derive cohort name from filepath based on cohort_from strategy."""
        if self.cohort_from == "config" and self.cohort_value:
            return self.cohort_value
        elif self.cohort_from == "filename":
            return filepath.stem
        elif self.cohort_from == "parent":
            return filepath.parent.name
        else:
            return filepath.stem
