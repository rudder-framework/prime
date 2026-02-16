"""
Base converter for streaming pipeline.

Each dataset format gets a concrete converter subclass.
Converters read raw files and return polars DataFrames
in Prime's canonical schema: (cohort, signal_id, I, value).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl


@dataclass
class ConversionResult:
    """Result of converting a single raw file."""
    df: pl.DataFrame  # (cohort, signal_id, I, value)
    source_file: Path
    n_signals: int
    n_samples_per_signal: int


class BaseConverter(ABC):
    """Abstract base for raw file → canonical schema converters."""

    @abstractmethod
    def convert_file(self, filepath: Path, cohort: Optional[str] = None) -> ConversionResult:
        """
        Convert a single raw file to canonical schema.

        Args:
            filepath: Path to raw file
            cohort: Override cohort name (else derived from config/filename)

        Returns:
            ConversionResult with polars DataFrame in (cohort, signal_id, I, value)
        """

    @abstractmethod
    def detect_signals(self, filepath: Path) -> List[str]:
        """
        Detect available signal names in a raw file.

        Args:
            filepath: Path to a sample raw file

        Returns:
            List of signal_id strings
        """

    @abstractmethod
    def file_pattern(self) -> str:
        """
        Glob pattern for raw files this converter handles.

        Returns:
            Glob pattern string, e.g. '**/*.mat'
        """
