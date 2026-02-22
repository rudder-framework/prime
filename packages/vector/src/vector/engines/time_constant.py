"""Engine: time_constant — exponential fit tau, equilibrium, R²."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'time_constant_tau': np.nan, 'time_constant_equilibrium': np.nan,
           'time_constant_fit_r2': np.nan, 'time_constant_is_decay': np.nan}
    n = len(y)
    if n < 10:
        return nan
    try:
        # Log-linear regression: if y = a*exp(b*t) + c, then log(|y-c|) ~ bt
        y_shifted = y - y[-1]  # assume last value ~ equilibrium
        mask = np.abs(y_shifted) > 1e-15
        if np.sum(mask) < 5:
            return nan
        t = np.arange(n, dtype=np.float64)
        log_y = np.log(np.abs(y_shifted[mask]))
        coeffs = np.polyfit(t[mask], log_y, 1)
        b = coeffs[0]
        tau = float(-1.0 / b) if abs(b) > 1e-15 else np.nan
        fitted = np.polyval(coeffs, t[mask])
        ss_res = np.sum((log_y - fitted) ** 2)
        ss_tot = np.sum((log_y - np.mean(log_y)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-15 else 0.0
        return {
            'time_constant_tau': tau,
            'time_constant_equilibrium': float(y[-1]),
            'time_constant_fit_r2': r2,
            'time_constant_is_decay': float(b < 0),
        }
    except Exception:
        return nan
