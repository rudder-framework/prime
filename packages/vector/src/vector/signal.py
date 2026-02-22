"""
Signal Vector
=============
Windowed feature extraction for a single signal.

Takes a 1D array of observations + window parameters.
Runs all engines on each window. Returns a list of dicts,
one per window, with all features namespaced by engine.

Usage:
    from vector.signal import compute_signal

    rows = compute_signal(
        signal_id='sensor_2',
        values=np.array([...]),
        window_size=256,
        stride=64,
        engines=None,  # None = all engines
    )
    # rows is a list of dicts, one per window
"""

import numpy as np
from typing import Dict, Any, List, Optional

from vector.registry import get_registry


def compute_signal(
    signal_id: str,
    values: np.ndarray,
    window_size: int,
    stride: int,
    engines: Optional[List[str]] = None,
    window_factor: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Compute windowed features for one signal.

    Args:
        signal_id: Signal identifier.
        values: 1D array of observations, sorted by canonical index.
        window_size: System-level window size (outer loop).
        stride: System-level stride.
        engines: List of engine names to run. None = all.
        window_factor: Typology-based multiplier for per-engine windows.

    Returns:
        List of dicts, one per window. Each dict has:
            signal_id, window_index, window_start, window_end,
            + all engine output keys (namespaced).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    n = len(values)

    if n == 0:
        return []

    registry = get_registry()

    if engines is None:
        engines = registry.engine_names

    # Group engines by their required window size
    engine_groups = registry.group_by_window(engines, window_factor)

    rows = []
    window_index = 0

    for window_end in range(window_size - 1, n, stride):
        window_start = max(0, window_end - window_size + 1)

        row = {
            'signal_id': signal_id,
            'window_index': window_index,
            'window_start': window_start,
            'window_end': window_end,
            'window_center': (window_start + window_end) / 2.0,
            'window_n_samples': window_end - window_start + 1,
        }

        # Run each engine group with its own window size
        for eng_window, eng_names in engine_groups.items():
            # Engine window is relative to window_end
            eng_start = max(0, window_end - eng_window + 1)
            actual_window = window_end - eng_start + 1

            if actual_window < 4:
                # Too small for any engine — fill NaN
                for name in eng_names:
                    for key in registry.get_outputs(name):
                        row[key] = float('nan')
                continue

            window_data = values[eng_start:window_end + 1]

            for name in eng_names:
                spec = registry.get_spec(name)

                # Check minimum samples
                if actual_window < spec.min_window:
                    for key in spec.outputs:
                        row[key] = float('nan')
                    continue

                try:
                    result = registry.get_compute(name)(window_data)
                    row.update(result)
                except Exception:
                    # Engine failed — fill NaN for its outputs
                    for key in spec.outputs:
                        row[key] = float('nan')

        rows.append(row)
        window_index += 1

    return rows


def compute_signal_batch(
    signals: Dict[str, np.ndarray],
    window_sizes: Dict[str, int],
    strides: Dict[str, int],
    engines: Optional[List[str]] = None,
    window_factors: Optional[Dict[str, float]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute windowed features for multiple signals.

    Args:
        signals: {signal_id: values_array}
        window_sizes: {signal_id: window_size}
        strides: {signal_id: stride}
        engines: Engine names to run. None = all.
        window_factors: {signal_id: factor}. None = 1.0 for all.

    Returns:
        Combined list of row dicts across all signals.
    """
    all_rows = []

    for signal_id, values in signals.items():
        ws = window_sizes.get(signal_id, 256)
        st = strides.get(signal_id, 64)
        wf = (window_factors or {}).get(signal_id, 1.0)

        rows = compute_signal(
            signal_id=signal_id,
            values=values,
            window_size=ws,
            stride=st,
            engines=engines,
            window_factor=wf,
        )
        all_rows.extend(rows)

    return all_rows
