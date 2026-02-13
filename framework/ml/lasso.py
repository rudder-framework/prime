"""
LASSO Feature Selection Engine
===============================

Applies L1-regularized (LASSO) regression to a feature matrix and target,
driving unimportant feature coefficients to exactly zero. This is automated
feature selection disguised as regression.

Given N features from upstream engines, LASSO returns which features matter,
how much they matter (coefficients), and which are eliminated (zero'd out).

Method:
  1. Standardize features (zero mean, unit variance) so penalty is fair
  2. Solve LASSO objective: min ||y - X*beta||^2 + alpha * ||beta||_1
  3. Use coordinate descent (pure numpy, no sklearn dependency)
  4. Select optimal alpha via cross-validation on training data
  5. Report nonzero features, coefficients, and selection statistics

Layer: Feature Selection (bridges engines → prediction)
Used by: rudder-lens / rudder-ml feature evaluation pipeline

References:
    Tibshirani (1996) "Regression Shrinkage and Selection via the Lasso"
    Friedman, Hastie, Tibshirani (2010) "Regularization Paths for GLMs via Coordinate Descent"
"""

import numpy as np
from typing import Optional


def compute(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    n_alphas: int = 50,
    alpha: Optional[float] = None,
    cv_folds: int = 5,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict:
    """
    Run LASSO feature selection on feature matrix X against target y.

    Args:
        X: Feature matrix (n_samples, n_features) — output from RUDDER engines
        y: Target vector (n_samples,) — e.g. RUL values
        feature_names: Optional list of feature names (len = n_features)
        n_alphas: Number of alpha values to test on regularization path
        alpha: If provided, use this alpha directly (skip CV)
        cv_folds: Number of cross-validation folds for alpha selection
        max_iter: Maximum coordinate descent iterations
        tol: Convergence tolerance

    Returns:
        dict with:
            - n_features_in: int (total features received)
            - n_features_selected: int (features with nonzero coefficients)
            - n_features_eliminated: int (features driven to zero)
            - selection_ratio: float (selected / total)
            - alpha_optimal: float (regularization strength chosen by CV)
            - alpha_min: float (smallest alpha tested)
            - alpha_max: float (largest alpha tested)
            - coefficients: list of float (all coefficients, including zeros)
            - nonzero_indices: list of int (indices of selected features)
            - nonzero_names: list of str (names of selected features, if provided)
            - nonzero_coefficients: list of float (coefficients of selected features)
            - abs_coefficient_sum: float (sum of absolute coefficients)
            - r_squared_train: float (R² on training data with selected features)
            - mse_train: float (MSE on training data)
            - cv_mse_mean: float (mean CV MSE at optimal alpha)
            - cv_mse_std: float (std of CV MSE at optimal alpha)
            - feature_ranking: list of int (indices sorted by |coefficient| descending)
            - intercept: float
            - n_samples: int
    """
    # ── Validate inputs ──────────────────────────────────────────────
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    if n_samples != len(y):
        return _empty_result(n_features, n_samples, reason="shape_mismatch")

    if n_samples < 10:
        return _empty_result(n_features, n_samples, reason="insufficient_data")

    if n_features < 1:
        return _empty_result(n_features, n_samples, reason="no_features")

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # ── Handle NaN/Inf ───────────────────────────────────────────────
    # Remove rows with any NaN or Inf
    valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    n_samples = len(y_clean)

    if n_samples < 10:
        return _empty_result(n_features, n_samples, reason="insufficient_valid_data")

    # ── Standardize ──────────────────────────────────────────────────
    X_mean = np.mean(X_clean, axis=0)
    X_std = np.std(X_clean, axis=0)
    y_mean = np.mean(y_clean)

    # Drop constant features (std = 0)
    active_mask = X_std > 1e-10
    if np.sum(active_mask) == 0:
        return _empty_result(n_features, n_samples, reason="all_constant")

    X_scaled = np.zeros_like(X_clean)
    X_scaled[:, active_mask] = (X_clean[:, active_mask] - X_mean[active_mask]) / X_std[active_mask]
    y_centered = y_clean - y_mean

    # ── Compute alpha path ───────────────────────────────────────────
    # alpha_max: smallest alpha where all coefficients are zero
    # = max(|X^T y|) / n_samples
    alpha_max = np.max(np.abs(X_scaled.T @ y_centered)) / n_samples
    alpha_min = alpha_max * 0.001  # typical ratio

    if alpha is not None:
        # User provided alpha, skip CV
        best_alpha = alpha
        cv_mse_mean = float("nan")
        cv_mse_std = float("nan")
    else:
        # Cross-validation to find optimal alpha
        alphas = np.logspace(np.log10(alpha_max), np.log10(alpha_min), n_alphas)
        best_alpha, cv_mse_mean, cv_mse_std = _cross_validate(
            X_scaled, y_centered, alphas, cv_folds, max_iter, tol
        )

    # ── Fit final model with best alpha ──────────────────────────────
    beta = _coordinate_descent(X_scaled, y_centered, best_alpha, max_iter, tol)

    # ── Unscale coefficients to original feature space ───────────────
    coef = np.zeros(n_features)
    coef[active_mask] = beta[active_mask] / X_std[active_mask]
    intercept = y_mean - np.dot(X_mean, coef)

    # ── Compute statistics ───────────────────────────────────────────
    y_pred = X_clean @ coef + intercept
    residuals = y_clean - y_pred
    mse_train = float(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_clean - y_mean) ** 2)
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0

    # ── Selection results ────────────────────────────────────────────
    nonzero_mask = np.abs(coef) > 1e-10
    nonzero_indices = list(np.where(nonzero_mask)[0])
    n_selected = len(nonzero_indices)

    # Sort by absolute coefficient (most important first)
    abs_coefs = np.abs(coef)
    feature_ranking = list(np.argsort(-abs_coefs))

    return {
        "n_features_in": int(n_features),
        "n_features_selected": int(n_selected),
        "n_features_eliminated": int(n_features - n_selected),
        "selection_ratio": float(n_selected / n_features) if n_features > 0 else 0.0,
        "alpha_optimal": float(best_alpha),
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "coefficients": [float(c) for c in coef],
        "nonzero_indices": nonzero_indices,
        "nonzero_names": [feature_names[i] for i in nonzero_indices],
        "nonzero_coefficients": [float(coef[i]) for i in nonzero_indices],
        "abs_coefficient_sum": float(np.sum(abs_coefs)),
        "r_squared_train": float(r_squared),
        "mse_train": float(mse_train),
        "cv_mse_mean": float(cv_mse_mean) if not isinstance(cv_mse_mean, float) or not np.isnan(cv_mse_mean) else float("nan"),
        "cv_mse_std": float(cv_mse_std) if not isinstance(cv_mse_std, float) or not np.isnan(cv_mse_std) else float("nan"),
        "feature_ranking": feature_ranking,
        "intercept": float(intercept),
        "n_samples": int(n_samples),
    }


# ===========================================================================
# Coordinate Descent (pure numpy LASSO solver)
# ===========================================================================

def _coordinate_descent(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Solve LASSO via coordinate descent.

    min (1/2n) ||y - X*beta||^2 + alpha * ||beta||_1

    Assumes X is standardized (zero mean, unit variance columns).
    """
    n_samples, n_features = X.shape
    beta = np.zeros(n_features)
    residual = y.copy()  # r = y - X @ beta, starts as y since beta=0

    # Precompute X^T X diagonals (= n_samples for standardized X, but compute anyway)
    col_norms_sq = np.sum(X ** 2, axis=0)

    for iteration in range(max_iter):
        beta_old = beta.copy()

        for j in range(n_features):
            if col_norms_sq[j] < 1e-10:
                continue

            # Partial residual: add back j-th contribution
            residual += X[:, j] * beta[j]

            # Compute raw update
            rho = X[:, j] @ residual / n_samples

            # Soft thresholding
            beta[j] = _soft_threshold(rho, alpha) / (col_norms_sq[j] / n_samples)

            # Update residual
            residual -= X[:, j] * beta[j]

        # Check convergence
        if np.max(np.abs(beta - beta_old)) < tol:
            break

    return beta


def _soft_threshold(x: float, threshold: float) -> float:
    """Soft thresholding operator: sign(x) * max(|x| - threshold, 0)"""
    if x > threshold:
        return x - threshold
    elif x < -threshold:
        return x + threshold
    else:
        return 0.0


# ===========================================================================
# Cross-validation
# ===========================================================================

def _cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    alphas: np.ndarray,
    n_folds: int,
    max_iter: int,
    tol: float,
) -> tuple:
    """
    K-fold cross-validation to select optimal alpha.

    Returns (best_alpha, mean_mse, std_mse).
    """
    n_samples = len(y)
    n_folds = min(n_folds, n_samples)  # can't have more folds than samples

    # Create fold indices
    indices = np.arange(n_samples)
    np.random.RandomState(42).shuffle(indices)
    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[: n_samples % n_folds] += 1
    fold_starts = np.cumsum(np.concatenate([[0], fold_sizes[:-1]]))

    mse_per_alpha = np.zeros((len(alphas), n_folds))

    for fold_idx in range(n_folds):
        start = fold_starts[fold_idx]
        end = start + fold_sizes[fold_idx]

        val_indices = indices[start:end]
        train_indices = np.concatenate([indices[:start], indices[end:]])

        X_train, X_val = X[train_indices], X[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        # Warm start: use previous alpha's solution as initialization
        for alpha_idx, a in enumerate(alphas):
            beta = _coordinate_descent(X_train, y_train, a, max_iter, tol)
            y_pred = X_val @ beta
            mse_per_alpha[alpha_idx, fold_idx] = np.mean((y_val - y_pred) ** 2)

    mean_mse = np.mean(mse_per_alpha, axis=1)
    std_mse = np.std(mse_per_alpha, axis=1)

    # One standard error rule: pick simplest model within 1 SE of minimum
    min_idx = np.argmin(mean_mse)
    threshold = mean_mse[min_idx] + std_mse[min_idx]

    # Find largest alpha (most regularized) within threshold
    # Alphas are sorted descending (most regularized first)
    best_idx = min_idx
    for i in range(len(alphas)):
        if mean_mse[i] <= threshold:
            best_idx = i
            break

    return float(alphas[best_idx]), float(mean_mse[best_idx]), float(std_mse[best_idx])


# ===========================================================================
# Mutual Information (bonus: runs alongside LASSO for convergence check)
# ===========================================================================

def compute_mutual_info(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Optional[list] = None,
    n_bins: int = 20,
) -> dict:
    """
    Compute mutual information between each feature and target.

    Information-theoretic feature ranking — catches nonlinear
    relationships that LASSO (linear) misses.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target vector (n_samples,)
        feature_names: Optional feature name list
        n_bins: Number of bins for discretization

    Returns:
        dict with:
            - mi_scores: list of float (MI score per feature)
            - mi_ranking: list of int (feature indices sorted by MI descending)
            - mi_names: list of str (feature names sorted by MI descending)
            - mi_threshold: float (noise floor estimate)
            - n_above_threshold: int (features above noise floor)
            - n_features: int
            - n_samples: int
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).ravel()

    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_samples, n_features = X.shape

    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    # Remove NaN rows
    valid_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X_clean = X[valid_mask]
    y_clean = y[valid_mask]
    n_samples = len(y_clean)

    if n_samples < 10:
        return {
            "mi_scores": [0.0] * n_features,
            "mi_ranking": list(range(n_features)),
            "mi_names": feature_names,
            "mi_threshold": 0.0,
            "n_above_threshold": 0,
            "n_features": n_features,
            "n_samples": n_samples,
        }

    # Discretize target
    y_binned = _discretize(y_clean, n_bins)

    mi_scores = []
    for j in range(n_features):
        x_col = X_clean[:, j]
        if np.std(x_col) < 1e-10:
            mi_scores.append(0.0)
            continue
        x_binned = _discretize(x_col, n_bins)
        mi = _mutual_information(x_binned, y_binned, n_bins)
        mi_scores.append(float(max(mi, 0.0)))  # MI is non-negative

    mi_scores = np.array(mi_scores)

    # Noise floor: estimate MI you'd get from random data
    # Approximation: 1 / (2 * n_samples * ln(2))
    noise_floor = 1.0 / (2.0 * n_samples * np.log(2))

    mi_ranking = list(np.argsort(-mi_scores))
    n_above = int(np.sum(mi_scores > noise_floor))

    return {
        "mi_scores": [float(s) for s in mi_scores],
        "mi_ranking": mi_ranking,
        "mi_names": [feature_names[i] for i in mi_ranking],
        "mi_threshold": float(noise_floor),
        "n_above_threshold": n_above,
        "n_features": int(n_features),
        "n_samples": int(n_samples),
    }


def _discretize(x: np.ndarray, n_bins: int) -> np.ndarray:
    """Discretize continuous values into bins via equal-frequency binning."""
    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, percentiles)
    bin_edges[-1] += 1e-10  # ensure max value falls in last bin
    return np.digitize(x, bin_edges[1:-1])


def _mutual_information(x_bins: np.ndarray, y_bins: np.ndarray, n_bins: int) -> float:
    """
    Compute mutual information I(X;Y) from discretized variables.

    I(X;Y) = sum_xy p(x,y) * log(p(x,y) / (p(x)*p(y)))
    """
    n = len(x_bins)

    # Joint histogram
    joint = np.zeros((n_bins, n_bins))
    for i in range(n):
        xi = min(int(x_bins[i]), n_bins - 1)
        yi = min(int(y_bins[i]), n_bins - 1)
        joint[xi, yi] += 1

    joint /= n  # normalize to probabilities

    # Marginals
    px = np.sum(joint, axis=1)
    py = np.sum(joint, axis=0)

    # Mutual information
    mi = 0.0
    for i in range(n_bins):
        for j in range(n_bins):
            if joint[i, j] > 1e-15 and px[i] > 1e-15 and py[j] > 1e-15:
                mi += joint[i, j] * np.log(joint[i, j] / (px[i] * py[j]))

    return mi


# ===========================================================================
# Empty result
# ===========================================================================

def _empty_result(n_features: int, n_samples: int, reason: str = "unknown") -> dict:
    """Return empty result when computation cannot proceed."""
    nan = float("nan")
    return {
        "n_features_in": int(n_features),
        "n_features_selected": 0,
        "n_features_eliminated": int(n_features),
        "selection_ratio": 0.0,
        "alpha_optimal": nan,
        "alpha_min": nan,
        "alpha_max": nan,
        "coefficients": [0.0] * n_features,
        "nonzero_indices": [],
        "nonzero_names": [],
        "nonzero_coefficients": [],
        "abs_coefficient_sum": 0.0,
        "r_squared_train": nan,
        "mse_train": nan,
        "cv_mse_mean": nan,
        "cv_mse_std": nan,
        "feature_ranking": list(range(n_features)),
        "intercept": nan,
        "n_samples": int(n_samples),
    }
