"""
Eigenvalue trajectory dynamics.

Takes time series of eigendecomp outputs (effective_dim, eigenvalues,
total_variance across windows) and computes derivatives:
velocity, acceleration, jerk, curvature.

This is how we detect failure onset: effective_dim starts declining
(negative velocity), then accelerates (increasingly negative acceleration).
Curvature tells us when the trajectory is bending.

All derivatives use central differences with optional smoothing.
"""

import numpy as np
from typing import Dict, Any, List, Optional


def compute_derivatives(
    x: np.ndarray,
    dt: float = 1.0,
    smooth_window: int = 1,
) -> Dict[str, np.ndarray]:
    """
    Compute velocity, acceleration, jerk, and curvature of a 1D series.

    Parameters
    ----------
    x : np.ndarray
        Ordered series values (one per window).
    dt : float
        Index step between consecutive points.
    smooth_window : int
        Smoothing window size (1 = no smoothing).

    Returns
    -------
    dict with:
        velocity : np.ndarray — dx/dt (central differences)
        acceleration : np.ndarray — d²x/dt²
        jerk : np.ndarray — d³x/dt³
        curvature : np.ndarray — |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
        speed : np.ndarray — |dx/dt|
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    if n < 3:
        z = np.full(n, np.nan)
        return {
            'velocity': z.copy(),
            'acceleration': z.copy(),
            'jerk': z.copy(),
            'curvature': z.copy(),
            'speed': z.copy(),
        }

    # Optional smoothing
    if smooth_window > 1 and n > smooth_window:
        kernel = np.ones(smooth_window) / smooth_window
        xs = np.convolve(x, kernel, mode='same')
    else:
        xs = x

    # Central differences for velocity
    velocity = np.full(n, np.nan)
    velocity[1:-1] = (xs[2:] - xs[:-2]) / (2 * dt)
    velocity[0] = (xs[1] - xs[0]) / dt  # forward
    velocity[-1] = (xs[-1] - xs[-2]) / dt  # backward

    # Central differences for acceleration
    acceleration = np.full(n, np.nan)
    if n >= 3:
        acceleration[1:-1] = (xs[2:] - 2 * xs[1:-1] + xs[:-2]) / (dt ** 2)
        acceleration[0] = acceleration[1] if n > 1 else np.nan
        acceleration[-1] = acceleration[-2] if n > 1 else np.nan

    # Jerk (derivative of acceleration)
    jerk = np.full(n, np.nan)
    if n >= 4:
        jerk[1:-1] = (acceleration[2:] - acceleration[:-2]) / (2 * dt)
        jerk[0] = jerk[1] if n > 1 else np.nan
        jerk[-1] = jerk[-2] if n > 1 else np.nan

    # 1D curvature: κ = |d²x/dt²| / (1 + (dx/dt)²)^(3/2)
    curvature = np.full(n, np.nan)
    denom = (1 + velocity ** 2) ** 1.5
    valid = np.isfinite(acceleration) & np.isfinite(denom) & (denom > 1e-10)
    curvature[valid] = np.abs(acceleration[valid]) / denom[valid]

    speed = np.abs(velocity)

    return {
        'velocity': velocity,
        'acceleration': acceleration,
        'jerk': jerk,
        'curvature': curvature,
        'speed': speed,
    }


def compute_eigenvalue_dynamics(
    eigendecomp_results: List[Dict[str, Any]],
    dt: float = 1.0,
    smooth_window: int = 3,
    max_eigenvalues: int = 5,
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute dynamics from a sequence of eigendecomp results.

    Parameters
    ----------
    eigendecomp_results : list of dict
        Output from eigendecomp.compute_eigendecomp_batch.
        Each must have 'effective_dim', 'eigenvalues', 'total_variance',
        and optionally 'I', 'condition_number', 'ratio_2_1',
        'eigenvalue_entropy'.
    dt : float
        Step size between windows.
    smooth_window : int
        Smoothing window for derivatives.
    max_eigenvalues : int
        Number of eigenvalues to track dynamics for.

    Returns
    -------
    list of dict — one per window, with derivative metrics added.
    """
    n = len(eigendecomp_results)
    if n == 0:
        return []

    # Extract time series
    eff_dim = np.array([r.get('effective_dim', np.nan) for r in eigendecomp_results])
    total_var = np.array([r.get('total_variance', np.nan) for r in eigendecomp_results])
    cond_num = np.array([r.get('condition_number', np.nan) for r in eigendecomp_results])
    ratio_2_1 = np.array([r.get('ratio_2_1', np.nan) for r in eigendecomp_results])
    eig_entropy = np.array([r.get('eigenvalue_entropy', np.nan) for r in eigendecomp_results])

    # Per-eigenvalue series
    eigenvalue_series = {}
    for k in range(max_eigenvalues):
        vals = []
        for r in eigendecomp_results:
            eigs = r.get('eigenvalues', [])
            vals.append(float(eigs[k]) if k < len(eigs) and np.isfinite(eigs[k]) else np.nan)
        eigenvalue_series[k] = np.array(vals)

    # Spectral gap: eigenvalue_0 / total_variance
    eig0 = eigenvalue_series.get(0, np.full(n, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        spectral_gap = np.where(
            np.isfinite(eig0) & np.isfinite(total_var) & (total_var > 0),
            eig0 / total_var,
            np.nan,
        )

    # Compute derivatives — full (D1/D2/D3/curvature) for primary metrics
    eff_dim_deriv = compute_derivatives(eff_dim, dt=dt, smooth_window=smooth_window)
    total_var_deriv = compute_derivatives(total_var, dt=dt, smooth_window=smooth_window)
    cond_num_deriv = compute_derivatives(cond_num, dt=dt, smooth_window=smooth_window)
    ratio_2_1_deriv = compute_derivatives(ratio_2_1, dt=dt, smooth_window=smooth_window)
    eig_entropy_deriv = compute_derivatives(eig_entropy, dt=dt, smooth_window=smooth_window)
    spectral_gap_deriv = compute_derivatives(spectral_gap, dt=dt, smooth_window=smooth_window)

    # Per-eigenvalue derivatives (full D1/D2/D3/curvature)
    eigen_derivs = {}
    for k, series in eigenvalue_series.items():
        if np.isfinite(series).sum() >= 3:
            eigen_derivs[k] = compute_derivatives(series, dt=dt, smooth_window=smooth_window)

    # Build output rows
    rows = []
    for i in range(n):
        row = {}

        # Pass through window index
        if 'I' in eigendecomp_results[i]:
            row['I'] = eigendecomp_results[i]['I']

        # Effective dimension dynamics (D0/D1/D2/D3/curvature)
        row['effective_dim'] = eff_dim[i]
        row['effective_dim_velocity'] = eff_dim_deriv['velocity'][i]
        row['effective_dim_acceleration'] = eff_dim_deriv['acceleration'][i]
        row['effective_dim_jerk'] = eff_dim_deriv['jerk'][i]
        row['effective_dim_curvature'] = eff_dim_deriv['curvature'][i]

        # Total variance dynamics (D0/D1/D2)
        row['total_variance'] = total_var[i]
        row['variance_velocity'] = total_var_deriv['velocity'][i]
        row['total_variance_acceleration'] = total_var_deriv['acceleration'][i]

        # Condition number dynamics (D0/D1/D2/D3/curvature)
        row['condition_number'] = cond_num[i]
        row['condition_number_velocity'] = cond_num_deriv['velocity'][i]
        row['condition_number_acceleration'] = cond_num_deriv['acceleration'][i]
        row['condition_number_jerk'] = cond_num_deriv['jerk'][i]
        row['condition_number_curvature'] = cond_num_deriv['curvature'][i]

        # Ratio 2/1 dynamics (D0/D1/D2)
        row['ratio_2_1'] = ratio_2_1[i]
        row['ratio_2_1_velocity'] = ratio_2_1_deriv['velocity'][i]
        row['ratio_2_1_acceleration'] = ratio_2_1_deriv['acceleration'][i]

        # Eigenvalue entropy dynamics (D0/D1/D2)
        row['eigenvalue_entropy'] = eig_entropy[i]
        row['eigenvalue_entropy_velocity'] = eig_entropy_deriv['velocity'][i]
        row['eigenvalue_entropy_acceleration'] = eig_entropy_deriv['acceleration'][i]

        # Spectral gap dynamics (D0/D1/D2)
        row['spectral_gap'] = spectral_gap[i]
        row['spectral_gap_velocity'] = spectral_gap_deriv['velocity'][i]
        row['spectral_gap_acceleration'] = spectral_gap_deriv['acceleration'][i]

        # Per-eigenvalue dynamics (D0/D1/D2/D3/curvature)
        for k, deriv in eigen_derivs.items():
            row[f'eigenvalue_{k}'] = eigenvalue_series[k][i]
            row[f'eigenvalue_{k}_velocity'] = deriv['velocity'][i]
            row[f'eigenvalue_{k}_acceleration'] = deriv['acceleration'][i]
            row[f'eigenvalue_{k}_jerk'] = deriv['jerk'][i]
            row[f'eigenvalue_{k}_curvature'] = deriv['curvature'][i]

        rows.append(row)

    return rows
