"""
ORTHON Config Recommender
=========================

Recommend window/stride config based on data profile.
User reviews and confirms — PRISM won't run without explicit config.
"""

from dataclasses import dataclass
from typing import Dict, List

from ..core.data_reader import DataProfile


@dataclass
class WindowRecommendation:
    """Recommended window/stride configuration."""
    window_size: int
    window_stride: int
    overlap_pct: float
    n_windows_approx: int
    rationale: str
    confidence: str  # 'high', 'medium', 'low'

    # Alternatives
    conservative: Dict[str, int]
    aggressive: Dict[str, int]


@dataclass
class ConfigRecommendation:
    """Complete configuration recommendation."""
    window: WindowRecommendation
    n_clusters: int
    n_regimes: int

    def to_dict(self) -> dict:
        return {
            'window_size': self.window.window_size,
            'window_stride': self.window.window_stride,
            'n_clusters': self.n_clusters,
            'n_regimes': self.n_regimes,
        }


class ConfigRecommender:
    """Recommend config based on data profile."""

    TARGET_WINDOWS = 20
    DEFAULT_OVERLAP = 0.5

    def __init__(self, profile: DataProfile):
        self.profile = profile

    def recommend(self) -> ConfigRecommendation:
        """Generate configuration recommendation."""
        lifecycle = self.profile.median_lifecycle

        # Short data needs more overlap
        if lifecycle < 100:
            overlap = 0.75
            confidence = 'low'
        elif lifecycle < 500:
            overlap = 0.5
            confidence = 'medium'
        else:
            overlap = 0.5
            confidence = 'high'

        # Calculate window to get ~TARGET_WINDOWS windows
        # n_windows ≈ (lifecycle - window) / stride + 1
        # stride = window * (1 - overlap)
        window_size = int(lifecycle / (self.TARGET_WINDOWS * (1 - overlap) + overlap))
        window_size = max(10, min(window_size, int(lifecycle * 0.5)))

        window_stride = max(1, int(window_size * (1 - overlap)))

        n_windows = max(1, int((lifecycle - window_size) / window_stride) + 1)
        actual_overlap = 1 - (window_stride / window_size) if window_size > 0 else 0

        # Check shortest entity
        min_windows = max(1, (self.profile.min_lifecycle - window_size) // window_stride + 1)
        if min_windows < 5:
            confidence = 'low'

        rationale = (
            f"Based on median lifecycle of {lifecycle:.0f}:\n"
            f"- Window {window_size} gives ~{n_windows} windows per entity\n"
            f"- {actual_overlap*100:.0f}% overlap preserves temporal continuity\n"
            f"- Shortest entity ({self.profile.min_lifecycle}) gets ~{min_windows} windows"
        )

        if min_windows < 5:
            rationale += f"\n- WARNING: Shortest entity has only {min_windows} windows"

        window_rec = WindowRecommendation(
            window_size=window_size,
            window_stride=window_stride,
            overlap_pct=actual_overlap * 100,
            n_windows_approx=n_windows,
            rationale=rationale,
            confidence=confidence,
            conservative={
                'window_size': int(window_size * 1.5),
                'window_stride': int(window_stride * 1.5)
            },
            aggressive={
                'window_size': max(10, int(window_size * 0.67)),
                'window_stride': max(1, int(window_stride * 0.67))
            },
        )

        return ConfigRecommendation(
            window=window_rec,
            n_clusters=3,
            n_regimes=3,
        )
