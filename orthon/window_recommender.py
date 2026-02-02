"""
ORTHON Window Recommender

Determines optimal analysis window size from typology raw measures.

This runs AFTER typology Level 1 (stationarity) and Level 2 (classification)
but BEFORE the manifest is generated. The recommended window is embedded in
the manifest so PRISM knows exactly how many samples per window to use.

The window must capture enough of the signal's natural structure to be meaningful:
    - For periodic signals: enough complete cycles
    - For correlated signals: enough of the correlation structure
    - For non-stationary signals: not so large that local behavior is lost

RULE HIERARCHY (first match wins):
    1. CONSTANT → None (no analysis needed)
    2. PERIODIC with known period → 4 × period (capture 4 complete cycles)
    3. ACF half-life available → 4 × acf_half_life (capture decorrelation)
    4. LONG_MEMORY (ACF never decayed) → 8 × acf_decay_lag, capped
    5. NON_STATIONARY → cap at signal_length / 10
    6. Default → 128

All windows are clamped to [MIN_WINDOW, MAX_WINDOW] and signal length.

WHY TYPOLOGY, NOT EIGENVALUES:
    Eigenvalues require a window to compute — chicken-and-egg problem.
    Typology already has the raw measures (period, ACF, stationarity) from
    Level 1/2 that define the signal's natural timescale.

    Eigenvalues become useful for REFINEMENT after the first PRISM pass:
    if eigenvalue spectrum is unstable across consecutive windows, the
    window was too small. That's a second-pass optimization (see notes).

Usage:
    from orthon.window_recommender import recommend_window, recommend_windows_batch

    window = recommend_window(typology_row)
    df = recommend_windows_batch(typology_df)
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional, List
import math

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ============================================================
# CONSTANTS
# ============================================================

MIN_WINDOW = 32       # Below this, statistics are unreliable
MAX_WINDOW = 2048     # Above this, computation becomes expensive
DEFAULT_WINDOW = 128  # When nothing else is known

# How many complete cycles to capture for periodic signals
CYCLES_TO_CAPTURE = 4

# Multiplier for ACF half-life → window
ACF_MULTIPLIER = 4

# For long-memory signals where ACF never decayed
LONG_MEMORY_MULTIPLIER = 8

# Non-stationary cap: window ≤ signal_length / this
NON_STATIONARY_DIVISOR = 10


# ============================================================
# RESULT
# ============================================================

@dataclass
class WindowRecommendation:
    """Result of window recommendation with reasoning."""
    window_size: int
    method: str          # Which rule determined the window
    reason: str          # Human-readable explanation
    confidence: str      # 'high', 'medium', 'low'
    raw_value: float     # Pre-clamping value (for debugging)

    # Inputs that drove the decision
    period: Optional[float] = None
    acf_half_life: Optional[float] = None
    acf_decayed: Optional[bool] = None
    n_samples: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Flat dict for parquet/manifest."""
        return {
            'recommended_window': self.window_size,
            'window_method': self.method,
            'window_reason': self.reason,
            'window_confidence': self.confidence,
        }


# ============================================================
# CORE LOGIC
# ============================================================

def _clamp(value: float, n_samples: Optional[int] = None) -> int:
    """Clamp window to valid range and round to int."""
    w = max(MIN_WINDOW, min(MAX_WINDOW, int(math.ceil(value))))

    # Never exceed half the signal length
    if n_samples is not None and n_samples > 0:
        w = min(w, n_samples // 2)

    # Final floor
    return max(MIN_WINDOW, w)


def recommend_window(row: Dict[str, Any]) -> WindowRecommendation:
    """
    Recommend analysis window size from a single signal's typology.

    Reads the following fields from the typology row:
        continuity:       str   — 'CONSTANT', 'CONTINUOUS', 'DISCRETE', 'EVENT'
        temporal_pattern: str   — 'PERIODIC', 'QUASI_PERIODIC', 'TRENDING', etc.
        stationarity:     str   — 'STATIONARY', 'NON_STATIONARY', etc.
        memory:           str   — 'LONG_MEMORY', 'SHORT_MEMORY', 'ANTI_PERSISTENT'
        seasonal_period:  float — Detected period in samples (from ACF peaks)
        dominant_freq:    float — Dominant frequency from FFT (cycles per sample)
        acf_half_life:    float — First lag where |ACF| < 0.5
        acf_decay_lag:    float — First lag where |ACF| < 1/e
        acf_decayed:      bool  — Whether ACF actually crossed threshold
        n_samples:        int   — Total samples in the signal

    Args:
        row: Dict with typology fields

    Returns:
        WindowRecommendation with window_size and reasoning
    """
    continuity = row.get('continuity', 'CONTINUOUS')
    temporal = row.get('temporal_pattern')
    stationarity = row.get('stationarity')
    memory = row.get('memory')

    # Raw measures
    seasonal_period = _safe_float(row.get('seasonal_period'))
    dominant_freq = _safe_float(row.get('dominant_freq'))
    acf_half_life = _safe_float(row.get('acf_half_life'))
    acf_decay_lag = _safe_float(row.get('acf_decay_lag'))
    acf_decayed = row.get('acf_decayed')
    n_samples = _safe_int(row.get('n_samples'))

    # ------------------------------------------------------------------
    # Rule 1: CONSTANT → no window
    # ------------------------------------------------------------------
    if continuity == 'CONSTANT':
        return WindowRecommendation(
            window_size=0,
            method='constant',
            reason='Signal is constant — no analysis window needed',
            confidence='high',
            raw_value=0,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Rule 2: PERIODIC with known period → 4 × period
    # ------------------------------------------------------------------
    period = _resolve_period(seasonal_period, dominant_freq, temporal)

    if period is not None and period > 1:
        raw = CYCLES_TO_CAPTURE * period
        w = _clamp(raw, n_samples)
        return WindowRecommendation(
            window_size=w,
            method='period',
            reason=f'Periodic signal: {CYCLES_TO_CAPTURE} × period({period:.1f}) = {raw:.0f}',
            confidence='high',
            raw_value=raw,
            period=period,
            acf_half_life=acf_half_life,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Rule 3: ACF half-life available and ACF decayed → 4 × half-life
    # ------------------------------------------------------------------
    if acf_half_life is not None and acf_half_life > 0:
        # Check if ACF actually decayed (not just hitting max lag)
        truly_decayed = acf_decayed if acf_decayed is not None else True

        if truly_decayed and acf_half_life < 500:  # Sane range
            raw = ACF_MULTIPLIER * acf_half_life
            w = _clamp(raw, n_samples)
            return WindowRecommendation(
                window_size=w,
                method='acf_half_life',
                reason=f'ACF half-life = {acf_half_life:.0f}: '
                       f'{ACF_MULTIPLIER} × {acf_half_life:.0f} = {raw:.0f}',
                confidence='high' if acf_half_life > 5 else 'medium',
                raw_value=raw,
                acf_half_life=acf_half_life,
                acf_decayed=truly_decayed,
                n_samples=n_samples,
            )

    # ------------------------------------------------------------------
    # Rule 4: LONG_MEMORY (ACF never decayed) → larger window
    # ------------------------------------------------------------------
    if memory == 'LONG_MEMORY' or (acf_decayed is not None and not acf_decayed):
        if acf_decay_lag is not None and acf_decay_lag > 0:
            raw = LONG_MEMORY_MULTIPLIER * acf_decay_lag
        else:
            raw = 512  # Conservative large window
        w = _clamp(raw, n_samples)
        return WindowRecommendation(
            window_size=w,
            method='long_memory',
            reason=f'Long-memory signal (ACF never decayed): window={w}',
            confidence='medium',
            raw_value=raw,
            acf_half_life=acf_half_life,
            acf_decayed=False,
            n_samples=n_samples,
        )

    # ------------------------------------------------------------------
    # Rule 5: NON_STATIONARY → cap to keep window local
    # ------------------------------------------------------------------
    if stationarity in ('NON_STATIONARY', 'DIFFERENCE_STATIONARY'):
        if n_samples is not None and n_samples > 0:
            raw = n_samples / NON_STATIONARY_DIVISOR
            w = _clamp(raw, n_samples)
            return WindowRecommendation(
                window_size=w,
                method='non_stationary_cap',
                reason=f'Non-stationary: n_samples({n_samples}) / {NON_STATIONARY_DIVISOR} = {raw:.0f}',
                confidence='medium',
                raw_value=raw,
                acf_half_life=acf_half_life,
                n_samples=n_samples,
            )

    # ------------------------------------------------------------------
    # Rule 6: Default
    # ------------------------------------------------------------------
    return WindowRecommendation(
        window_size=DEFAULT_WINDOW,
        method='default',
        reason=f'No strong signal timescale detected — using default {DEFAULT_WINDOW}',
        confidence='low',
        raw_value=float(DEFAULT_WINDOW),
        acf_half_life=acf_half_life,
        n_samples=n_samples,
    )


# ============================================================
# PERIOD RESOLUTION
# ============================================================

def _resolve_period(
    seasonal_period: Optional[float],
    dominant_freq: Optional[float],
    temporal_pattern: Optional[str],
) -> Optional[float]:
    """
    Resolve the signal's period from available sources.

    Priority:
        1. seasonal_period from ACF peaks (most reliable — direct measurement)
        2. 1/dominant_freq from FFT (reliable if periodic classification)
        3. None

    Only used when temporal_pattern indicates periodicity.
    """
    # Only use period if signal is actually periodic
    if temporal_pattern not in ('PERIODIC', 'QUASI_PERIODIC'):
        return None

    # ACF seasonal period: direct measurement from autocorrelation peaks
    if seasonal_period is not None and seasonal_period > 1:
        return seasonal_period

    # Dominant frequency: convert to period
    if dominant_freq is not None and dominant_freq > 0:
        period = 1.0 / dominant_freq
        if period > 1:  # At least 1 sample per cycle
            return period

    return None


# ============================================================
# BATCH PROCESSING
# ============================================================

def recommend_windows_batch(
    typology_path: str,
    output_column: str = 'recommended_window',
) -> 'pl.DataFrame':
    """
    Add window recommendations to a typology parquet file.

    Reads typology.parquet, computes window for each signal,
    adds recommended_window and window_method columns.

    Args:
        typology_path: Path to typology.parquet
        output_column: Name of the window column to add

    Returns:
        Polars DataFrame with window columns added
    """
    if not HAS_POLARS:
        raise ImportError("polars is required for batch processing")

    df = pl.read_parquet(typology_path)
    recs = []

    for row in df.iter_rows(named=True):
        rec = recommend_window(row)
        recs.append({
            output_column: rec.window_size,
            'window_method': rec.method,
            'window_confidence': rec.confidence,
        })

    rec_df = pl.DataFrame(recs)
    return pl.concat([df, rec_df], how='horizontal')


# ============================================================
# STRIDE COMPUTATION
# ============================================================

def compute_stride(window_size: int, overlap_pct: float = 50.0) -> int:
    """
    Compute stride from window and overlap percentage.

    Default 50% overlap balances temporal resolution vs compute cost.
    Higher overlap (75%) for non-stationary signals that need finer tracking.

    Args:
        window_size: Window size in samples
        overlap_pct: Overlap as percentage (0-95)

    Returns:
        Stride in samples (minimum 1)
    """
    if window_size <= 0:
        return 0
    overlap_pct = max(0.0, min(95.0, overlap_pct))
    stride = max(1, int(window_size * (1 - overlap_pct / 100.0)))
    return stride


def recommend_stride(row: Dict[str, Any], window_size: int) -> int:
    """
    Recommend stride based on signal characteristics.

    Non-stationary and trending signals get higher overlap (75%)
    to track changing behavior more precisely.
    Stationary signals use 50% overlap.
    """
    if window_size <= 0:
        return 0

    stationarity = row.get('stationarity', 'STATIONARY')
    temporal = row.get('temporal_pattern', 'RANDOM')

    # Higher overlap for signals where behavior is changing
    if stationarity in ('NON_STATIONARY', 'DIFFERENCE_STATIONARY'):
        return compute_stride(window_size, overlap_pct=75.0)
    if temporal == 'TRENDING':
        return compute_stride(window_size, overlap_pct=75.0)

    return compute_stride(window_size, overlap_pct=50.0)


# ============================================================
# EIGENVALUE REFINEMENT (SECOND PASS)
# ============================================================
# These functions are for AFTER the first PRISM pass.
# They read eigenvalue results and check if the window was right.
#
# NOT used in manifest generation. Used in a future refinement loop:
#   1. Typology → window → manifest → PRISM runs
#   2. ORTHON reads state_geometry.parquet
#   3. eigenvalue_window_check() → "window too small" / "window OK"
#   4. If bad → adjust manifest → PRISM re-runs
# ============================================================

def eigenvalue_window_check(
    eigenvalues_by_window: List[List[float]],
    threshold: float = 0.3,
) -> Dict[str, Any]:
    """
    Check if eigenvalue spectrum is stable across consecutive windows.

    If eigenvalue proportions jump significantly between windows,
    the window is too small — it's not capturing the full structure
    and each window sees a different slice of the dynamics.

    If the first eigenvalue completely dominates (>95% of variance),
    the window may be too large — it's averaging out local structure.

    Args:
        eigenvalues_by_window: List of eigenvalue arrays, one per window
        threshold: Max acceptable variation in eigenvalue proportions

    Returns:
        Dict with 'stable' (bool), 'recommendation' (str), 'variation' (float)
    """
    if len(eigenvalues_by_window) < 3:
        return {
            'stable': True,
            'recommendation': 'insufficient_windows',
            'variation': 0.0,
            'note': 'Need at least 3 windows to assess stability',
        }

    # Compute proportion of variance explained by each eigenvalue per window
    proportions = []
    for eigs in eigenvalues_by_window:
        eigs = [max(0, e) for e in eigs]  # clip negatives
        total = sum(eigs)
        if total > 0:
            proportions.append([e / total for e in eigs])

    if len(proportions) < 3:
        return {
            'stable': True,
            'recommendation': 'insufficient_data',
            'variation': 0.0,
        }

    # Check variation of first eigenvalue proportion across windows
    first_eig_props = [p[0] for p in proportions if len(p) > 0]
    if not first_eig_props:
        return {'stable': True, 'recommendation': 'no_eigenvalues', 'variation': 0.0}

    mean_prop = sum(first_eig_props) / len(first_eig_props)
    variation = max(abs(p - mean_prop) for p in first_eig_props)

    # Check for over-dominance (window too large)
    if mean_prop > 0.95:
        return {
            'stable': True,
            'recommendation': 'window_possibly_too_large',
            'variation': variation,
            'note': f'First eigenvalue explains {mean_prop:.0%} — consider smaller window',
        }

    # Check for instability (window too small)
    if variation > threshold:
        return {
            'stable': False,
            'recommendation': 'window_too_small',
            'variation': variation,
            'note': f'Eigenvalue proportions vary by {variation:.2f} across windows — '
                    f'increase window to capture full structure',
        }

    return {
        'stable': True,
        'recommendation': 'window_ok',
        'variation': variation,
    }


# ============================================================
# HELPERS
# ============================================================

def _safe_float(val: Any) -> Optional[float]:
    """Convert to float, returning None for None/NaN/invalid."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (ValueError, TypeError):
        return None


def _safe_int(val: Any) -> Optional[int]:
    """Convert to int, returning None for None/invalid."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


# ============================================================
# CLI
# ============================================================

def main():
    import sys
    import json

    usage = """
ORTHON Window Recommender
=========================
Determines optimal window size from typology.

Usage:
    python -m orthon.window_recommender <typology.parquet> [output.parquet]

    If output path given: writes augmented parquet with window columns.
    If no output: prints recommendations to stdout.
"""

    if len(sys.argv) < 2:
        print(usage)
        sys.exit(1)

    typology_path = sys.argv[1]

    if len(sys.argv) > 2:
        output_path = sys.argv[2]
        df = recommend_windows_batch(typology_path)
        df.write_parquet(output_path)
        print(f"Wrote {len(df)} signals with windows to {output_path}")
    else:
        if not HAS_POLARS:
            print("polars required for parquet reading")
            sys.exit(1)
        df = pl.read_parquet(typology_path)
        for row in df.iter_rows(named=True):
            rec = recommend_window(row)
            sig = row.get('signal_id', '?')
            print(f"  {sig}: w={rec.window_size} ({rec.method}) — {rec.reason}")


if __name__ == '__main__':
    main()
