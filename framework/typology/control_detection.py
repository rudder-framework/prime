"""
Detect signals that look like control inputs vs system responses.

CONTROL signals (inputs):
- Binary/discrete levels (on/off, setpoints)
- Sharp step transitions
- Low unique value count
- High derivative sparsity (mostly flat, occasional jumps)

RESPONSE signals (outputs):
- Continuous variation
- Smooth transitions
- High unique value count
- Gradual changes
"""

import numpy as np
from typing import Dict, Any


def detect_control_signal(values: np.ndarray) -> Dict[str, Any]:
    """
    Classify signal as CONTROL (input) or RESPONSE (output).

    Args:
        values: Signal values as numpy array

    Returns:
        dict with:
            - is_control: bool
            - signal_role: 'CONTROL' | 'RESPONSE' | 'CONSTANT' | 'UNKNOWN'
            - n_levels: int (distinct values)
            - step_count: int (sharp transitions)
            - level_fraction: float (time at discrete levels)
    """
    n = len(values)

    if n < 2:
        return {
            'is_control': False,
            'signal_role': 'UNKNOWN',
            'n_levels': 1,
            'step_count': 0,
            'level_fraction': 1.0,
        }

    # Count unique values (discretized to 0.1% of range)
    value_range = np.ptp(values)
    if value_range < 1e-10:
        return {
            'is_control': True,
            'signal_role': 'CONSTANT',
            'n_levels': 1,
            'step_count': 0,
            'level_fraction': 1.0,
        }

    resolution = value_range / 1000
    discretized = np.round(values / resolution) * resolution
    unique_values = np.unique(discretized)
    n_levels = len(unique_values)

    # Detect step transitions (derivative > 10% of range in one sample)
    diff = np.abs(np.diff(values))
    step_threshold = value_range * 0.1
    step_count = int(np.sum(diff > step_threshold))

    # Time spent near discrete levels (within 1% of a level)
    level_tolerance = value_range * 0.01
    near_level = np.zeros(n, dtype=bool)
    for level in unique_values:
        near_level |= np.abs(values - level) < level_tolerance
    level_fraction = float(near_level.sum() / n)

    # Classification logic
    is_control = (
        (n_levels <= 10 and level_fraction > 0.9) or  # Few levels, mostly at them
        (n_levels <= 5 and step_count >= 1) or         # Very few levels with steps
        (n_levels == 2)                                 # Binary
    )

    if is_control:
        signal_role = 'CONTROL'
    elif n_levels > 100 and level_fraction < 0.5:
        signal_role = 'RESPONSE'
    else:
        signal_role = 'UNKNOWN'

    return {
        'is_control': is_control,
        'signal_role': signal_role,
        'n_levels': int(n_levels),
        'step_count': step_count,
        'level_fraction': level_fraction,
    }


def classify_signal_roles(
    observations_df,
    signal_column: str = 'signal_id',
    value_column: str = 'value',
) -> Dict[str, Dict[str, Any]]:
    """
    Classify all signals in observations as CONTROL or RESPONSE.

    Args:
        observations_df: Polars DataFrame with observations
        signal_column: Column with signal IDs
        value_column: Column with values

    Returns:
        Dict mapping signal_id to classification result
    """
    results = {}

    for signal_id in observations_df[signal_column].unique().to_list():
        signal_data = (
            observations_df
            .filter(observations_df[signal_column] == signal_id)
            .sort('I')
        )
        values = signal_data[value_column].to_numpy()
        results[signal_id] = detect_control_signal(values)

    return results


def find_control_response_pairs(
    signal_roles: Dict[str, Dict[str, Any]],
) -> list:
    """
    Find potential CONTROL -> RESPONSE pairs for transfer function analysis.

    Args:
        signal_roles: Dict from classify_signal_roles()

    Returns:
        List of (control_signal, response_signal) tuples
    """
    controls = [
        sig for sig, info in signal_roles.items()
        if info['signal_role'] == 'CONTROL'
    ]
    responses = [
        sig for sig, info in signal_roles.items()
        if info['signal_role'] == 'RESPONSE'
    ]

    # Return all combinations (RUDDER can filter by domain knowledge)
    pairs = []
    for ctrl in controls:
        for resp in responses:
            pairs.append((ctrl, resp))

    return pairs
