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

# Rust-accelerated batch computation (optional)
try:
    import pmtvs_vector
    _HAS_CORE = pmtvs_vector.BACKEND == 'rust'
    _CORE_ENGINES = frozenset(pmtvs_vector.available_engines()) if _HAS_CORE else frozenset()
except ImportError:
    _HAS_CORE = False
    _CORE_ENGINES = frozenset()


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

    # Partition engines: Rust-accelerated vs Python fallback
    if _HAS_CORE:
        rust_engines = [e for e in engines if e in _CORE_ENGINES]
        python_engines = [e for e in engines if e not in _CORE_ENGINES]
    else:
        rust_engines = []
        python_engines = list(engines)

    # Rust batch path: one crossing per signal for all rust engines
    rust_rows = []
    if rust_engines:
        configs = []
        for name in rust_engines:
            spec = registry.get_spec(name)
            configs.append({
                'name': name,
                'base_window': spec.base_window,
                'min_window': spec.min_window,
                'outputs': spec.outputs,
            })
        rust_rows = pmtvs_vector.compute_signal_batch(
            values, window_size, stride, configs, window_factor,
        )

    # Python path for remaining engines
    python_groups = registry.group_by_window(python_engines, window_factor) if python_engines else {}

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

        # Merge Rust results for this window
        if window_index < len(rust_rows):
            row.update(rust_rows[window_index])

        # Run Python engines
        for eng_window, eng_names in python_groups.items():
            eng_start = max(0, window_end - eng_window + 1)
            actual_window = window_end - eng_start + 1

            if actual_window < 4:
                for name in eng_names:
                    for key in registry.get_outputs(name):
                        row[key] = float('nan')
                continue

            window_data = values[eng_start:window_end + 1]

            for name in eng_names:
                spec = registry.get_spec(name)

                if actual_window < spec.min_window:
                    for key in spec.outputs:
                        row[key] = float('nan')
                    continue

                try:
                    result = registry.get_compute(name)(window_data)
                    row.update(result)
                except Exception:
                    for key in spec.outputs:
                        row[key] = float('nan')

        rows.append(row)
        window_index += 1

    return rows
