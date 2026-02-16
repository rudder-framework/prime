"""
Overlap Zone Detection for Dual Classification
================================================

Signals on classification boundaries carry both labels. Only adjacent types
on natural continua have defined overlap regions. Each zone is defined by
feature ranges where a signal has meaningful evidence for both types.

Usage:
    from prime.typology.overlap_zones import check_overlap_zones, build_dual_result

    primary = classify_temporal_pattern(row, fft_size)
    secondary = check_overlap_zones(row, primary)
    dual = build_dual_result(primary, secondary)
"""

from typing import Any, Dict, Optional


# Defined overlap pairs — only these can produce dual labels.
# Keys are (type_a, type_b) tuples. Values are dicts mapping
# feature names to (lo, hi) ranges. ALL features must fall
# in range for the zone to fire.
#
# Special feature: 'variance_ratio_distance' = abs(variance_ratio - 1.0)
TEMPORAL_OVERLAP_ZONES = {
    ('PERIODIC', 'QUASI_PERIODIC'): {
        'spectral_flatness': (0.1, 0.3),
        'spectral_peak_snr': (10, 15),
        'perm_entropy': (0.7, 0.9),
    },
    ('QUASI_PERIODIC', 'RANDOM'): {
        'spectral_flatness': (0.2, 0.5),
        'perm_entropy': (0.85, 0.95),
    },
    ('STATIONARY', 'RANDOM'): {
        'perm_entropy': (0.9, 0.98),
        'spectral_flatness': (0.3, 0.6),
    },
    ('TRENDING', 'DRIFTING'): {
        'hurst': (0.75, 0.85),
        'variance_ratio_distance': (0.05, 0.15),
    },
    ('TRENDING', 'STATIONARY'): {
        'hurst': (0.7, 0.8),
    },
    ('DRIFTING', 'STATIONARY'): {
        'hurst': (0.7, 0.8),
        'variance_ratio_distance': (0.0, 0.1),
    },
}

# PR5 types are exclusive — no overlap possible
DISCRETE_SPARSE_TYPES = frozenset({
    'CONSTANT', 'BINARY', 'DISCRETE', 'IMPULSIVE',
    'EVENT', 'STEP', 'INTERMITTENT',
})


def _get_feature_value(row: Dict[str, Any], feature: str) -> Optional[float]:
    """Extract a feature value from a row, handling special computed features."""
    if feature == 'variance_ratio_distance':
        vr = row.get('variance_ratio')
        if vr is None:
            return None
        return abs(vr - 1.0)
    return row.get(feature)


def check_overlap_zones(
    row: Dict[str, Any],
    primary: str,
    zones: dict = None,
) -> Optional[str]:
    """
    Check if a signal falls in a defined overlap zone.

    Only checks pairs where primary matches one side. Returns the
    secondary label if ALL features for that zone fall within range.
    Returns None if no overlap zone matches.

    Args:
        row: Signal feature dict (from typology_raw)
        primary: Primary classification from tournament bracket
        zones: Overlap zone definitions (default: TEMPORAL_OVERLAP_ZONES)

    Returns:
        Secondary type label, or None
    """
    if zones is None:
        zones = TEMPORAL_OVERLAP_ZONES

    # Discrete/sparse types never get dual classification
    if primary in DISCRETE_SPARSE_TYPES:
        return None

    for (type_a, type_b), zone_ranges in zones.items():
        # Only check if primary matches one side of the pair
        if primary not in (type_a, type_b):
            continue

        secondary = type_b if primary == type_a else type_a

        # Check if ALL features fall within the zone ranges
        in_zone = True
        for feature, (lo, hi) in zone_ranges.items():
            val = _get_feature_value(row, feature)
            if val is None or not (lo <= val <= hi):
                in_zone = False
                break

        if in_zone:
            return secondary

    return None


def build_dual_result(
    primary: str,
    secondary: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the dual classification result dict.

    Args:
        primary: Primary classification (always present)
        secondary: Secondary classification (None if clear)

    Returns:
        Dict with temporal_pattern (list), temporal_primary (str),
        temporal_secondary (str|None), classification_confidence (str)
    """
    if secondary:
        return {
            'temporal_pattern': [primary, secondary],
            'temporal_primary': primary,
            'temporal_secondary': secondary,
            'classification_confidence': 'boundary',
        }
    else:
        return {
            'temporal_pattern': [primary],
            'temporal_primary': primary,
            'temporal_secondary': None,
            'classification_confidence': 'clear',
        }
