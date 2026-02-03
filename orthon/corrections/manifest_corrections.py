"""
Manifest Generator Corrections

Fix: The top-level `params.stride` default was hardcoded to 1,
which means 99.6% overlap and ~500k engine calls for 5000 samples.

The per-signal stride IS correctly computed from recommend_stride(),
but the global default was still 1. Fix changes the global default
to match the most common per-signal stride, or uses 50% overlap
of the default window as a sane fallback.

Additionally: when all per-signal strides are identical, the global
default should match, not diverge.
"""


def compute_global_stride_default(
    signals_config: dict,
    default_window: int = 128,
) -> int:
    """
    Compute a sensible global stride default from per-signal strides.

    Logic:
        1. Collect all per-signal strides
        2. If they exist, use the median (most representative)
        3. If not, use 50% of default_window (sane default)
        4. Never return stride < 1

    Args:
        signals_config: Dict of signal_id -> signal config with 'stride' key
        default_window: Default window size (for fallback)

    Returns:
        Global stride default
    """
    strides = []
    for sig_id, config in signals_config.items():
        if isinstance(config, dict) and 'stride' in config:
            strides.append(config['stride'])

    if strides:
        # Use median to avoid outliers from constant signals
        import statistics
        return max(1, int(statistics.median(strides)))

    # Fallback: 50% overlap
    return max(1, default_window // 2)
