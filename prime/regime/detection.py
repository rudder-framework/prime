"""
Regime Detection
================
Detect operating regimes from long-format observations.

Two modes:
  "settings" — cluster on known operational setting signals
               (e.g., op1, op2, op3 for C-MAPSS represent altitude/Mach/throttle)
  "variance"  — cluster on full signal state at each time point
  "auto"      — use settings if provided and present, else variance

Output schema: cohort, signal_0, regime_id (int), regime_confidence (float)

regime_id is a dense integer label (0, 1, 2, ...).
regime_confidence is 1 - normalized distance to nearest cluster center.

Regime detection is ordering-independent: the same regimes are found
regardless of which signal is used as signal_0.
"""

from __future__ import annotations

import warnings

import numpy as np
import polars as pl


def detect_regimes(
    observations: pl.DataFrame,
    operational_settings: list[str] | None = None,
    max_k: int = 10,
    min_k: int = 2,
    method: str = "auto",
) -> pl.DataFrame:
    """
    Detect operating regimes from long-format observations.

    Args:
        observations: Long-format DataFrame with (signal_0, signal_id, value, cohort).
        operational_settings: Signal IDs that define operating regimes.
            For C-MAPSS: ["op1", "op2", "op3"] (altitude, Mach, throttle).
            If None and method="settings", raises ValueError.
        max_k: Maximum number of regimes to consider.
        min_k: Minimum number of regimes to consider.
        method: "settings", "variance", or "auto".
            "auto" uses settings if provided and present, else variance.

    Returns:
        DataFrame with columns: cohort, signal_0, regime_id, regime_confidence
        One row per unique (cohort, signal_0) combination.
    """
    available_signals = set(observations["signal_id"].unique().to_list())

    if method == "auto":
        if operational_settings:
            present = available_signals & set(operational_settings)
            method = "settings" if len(present) == len(operational_settings) else "variance"
        else:
            method = "variance"

    if method == "settings":
        if not operational_settings:
            raise ValueError(
                "operational_settings must be provided when method='settings'."
            )
        missing = set(operational_settings) - available_signals
        if missing:
            raise ValueError(
                f"Regime signals not found in observations: {sorted(missing)}."
                f" Available: {sorted(available_signals)}"
            )
        return _detect_from_settings(observations, operational_settings, max_k, min_k)
    else:
        return _detect_from_variance(observations, max_k, min_k)


def _detect_from_settings(
    observations: pl.DataFrame,
    settings: list[str],
    max_k: int,
    min_k: int,
) -> pl.DataFrame:
    """
    Cluster on operational setting signals.

    Pivot setting signals wide → one row per (cohort, signal_0) → KMeans.
    """
    # Pivot setting signals wide to get feature matrix
    setting_wide = (
        observations
        .filter(pl.col("signal_id").is_in(settings))
        .pivot(on="signal_id", index=["cohort", "signal_0"], values="value")
    )

    setting_cols = [c for c in settings if c in setting_wide.columns]
    X = setting_wide.select(setting_cols).to_numpy().astype(float)

    # Valid rows (no NaN)
    valid_mask = np.isfinite(X).all(axis=1)
    X_valid = X[valid_mask]

    n = len(X_valid)
    if n < min_k * 2:
        print(f"  Regime detection: 1 regime (insufficient data: {n} points)")
        return setting_wide.select(["cohort", "signal_0"]).with_columns(
            pl.lit(0).cast(pl.Int32).alias("regime_id"),
            pl.lit(1.0).alias("regime_confidence"),
        )

    labels, confidence, best_k, best_score = _cluster_silhouette(X_valid, max_k, min_k)

    # Map back to all rows (including NaN rows → regime 0, confidence 0)
    all_labels = np.zeros(len(X), dtype=np.int32)
    all_confidence = np.zeros(len(X), dtype=np.float64)
    all_labels[valid_mask] = labels
    all_confidence[valid_mask] = confidence

    result = setting_wide.select(["cohort", "signal_0"]).with_columns(
        pl.Series("regime_id", all_labels, dtype=pl.Int32),
        pl.Series("regime_confidence", all_confidence, dtype=pl.Float64),
    )

    _print_regime_summary(best_k, best_score, all_labels)
    return result


def _detect_from_variance(
    observations: pl.DataFrame,
    max_k: int,
    min_k: int,
) -> pl.DataFrame:
    """
    Detect regimes from signal variance structure.

    Pivot all signals wide → one row per (cohort, signal_0) → KMeans.
    For datasets without explicit operating setting signals.
    """
    # Pivot all signals wide
    all_wide = observations.pivot(
        on="signal_id",
        index=["cohort", "signal_0"],
        values="value",
    )

    signal_cols = [c for c in all_wide.columns if c not in {"cohort", "signal_0"}]
    if not signal_cols:
        return all_wide.select(["cohort", "signal_0"]).with_columns(
            pl.lit(0).cast(pl.Int32).alias("regime_id"),
            pl.lit(1.0).alias("regime_confidence"),
        )

    X = all_wide.select(signal_cols).to_numpy().astype(float)

    # Drop columns with any NaN (can't cluster on partial feature vectors)
    col_valid = np.isfinite(X).all(axis=0)
    X = X[:, col_valid]

    if X.shape[1] == 0:
        return all_wide.select(["cohort", "signal_0"]).with_columns(
            pl.lit(0).cast(pl.Int32).alias("regime_id"),
            pl.lit(1.0).alias("regime_confidence"),
        )

    # Drop rows with any remaining NaN
    valid_mask = np.isfinite(X).all(axis=1)
    X_valid = X[valid_mask]

    n = len(X_valid)
    if n < min_k * 5:
        print(f"  Regime detection: 1 regime (insufficient data: {n} points)")
        return all_wide.select(["cohort", "signal_0"]).with_columns(
            pl.lit(0).cast(pl.Int32).alias("regime_id"),
            pl.lit(1.0).alias("regime_confidence"),
        )

    labels, confidence, best_k, best_score = _cluster_silhouette(X_valid, max_k, min_k)

    if best_score < 0.1:
        print(
            f"  Regime detection: 1 regime (no meaningful structure, "
            f"best silhouette={best_score:.3f})"
        )
        return all_wide.select(["cohort", "signal_0"]).with_columns(
            pl.lit(0).cast(pl.Int32).alias("regime_id"),
            pl.lit(1.0).alias("regime_confidence"),
        )

    all_labels = np.zeros(len(X), dtype=np.int32)
    all_confidence = np.zeros(len(X), dtype=np.float64)
    all_labels[valid_mask] = labels
    all_confidence[valid_mask] = confidence

    result = all_wide.select(["cohort", "signal_0"]).with_columns(
        pl.Series("regime_id", all_labels, dtype=pl.Int32),
        pl.Series("regime_confidence", all_confidence, dtype=pl.Float64),
    )

    _print_regime_summary(best_k, best_score, all_labels, method="variance-based")
    return result


def _cluster_silhouette(
    X: np.ndarray,
    max_k: int,
    min_k: int,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    """
    Sweep K via silhouette score. Fit final model with best K.

    Returns: (labels, confidence, best_k, best_score)
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        raise ImportError(
            "Regime detection requires scikit-learn. "
            "Install it with: pip install prime[ml]"
        ) from e

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    n = len(X_std)
    rng = np.random.RandomState(42)

    # Subsample for silhouette scoring on large datasets
    sample_size = min(n, 10_000)
    idx = rng.choice(n, sample_size, replace=False) if n > sample_size else np.arange(n)
    X_sample = X_std[idx]

    best_k = 1
    best_score = -1.0

    for k in range(min_k, min(max_k + 1, sample_size)):
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                labels_s = km.fit_predict(X_sample)
            if len(set(labels_s.tolist())) < 2:
                continue
            score = silhouette_score(X_sample, labels_s)
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    # Fit final model on all data with best_k
    km_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)
    labels = km_final.fit_predict(X_std)

    # Confidence = 1 - normalized distance to nearest center
    distances = km_final.transform(X_std)  # (n, best_k)
    nearest_dist = distances[np.arange(n), labels]
    max_dist = nearest_dist.max()
    confidence = 1.0 - (nearest_dist / max_dist) if max_dist > 0 else np.ones(n)

    return labels.astype(np.int32), confidence.astype(np.float64), best_k, float(best_score)


def _print_regime_summary(
    best_k: int,
    best_score: float,
    labels: np.ndarray,
    method: str = "settings-based",
) -> None:
    n_total = len(labels)
    print(f"  Regime detection ({method}): {best_k} regime(s) (silhouette={best_score:.3f})")
    for k in range(best_k):
        n = int((labels == k).sum())
        print(f"    Regime {k}: {n:,} observations ({n / n_total:.0%})")
