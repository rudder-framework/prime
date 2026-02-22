"""Engine: adf_stat — Augmented Dickey-Fuller test."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    n = len(y)
    nan = {'adf_stat_value': np.nan, 'adf_stat_pvalue': np.nan,
           'adf_stat_lags': np.nan, 'adf_stat_nobs': np.nan}
    if n < 20:
        return nan
    try:
        from pmtvs import adf_test
        result = adf_test(y)
        if isinstance(result, dict):
            return {
                'adf_stat_value': float(result.get('statistic', np.nan)),
                'adf_stat_pvalue': float(result.get('pvalue', np.nan)),
                'adf_stat_lags': float(result.get('usedlag', np.nan)),
                'adf_stat_nobs': float(result.get('nobs', np.nan)),
            }
        return nan
    except ImportError:
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(y, maxlag=min(int(np.sqrt(n)), 20), autolag='AIC')
            return {
                'adf_stat_value': float(result[0]),
                'adf_stat_pvalue': float(result[1]),
                'adf_stat_lags': float(result[2]),
                'adf_stat_nobs': float(result[3]),
            }
        except ImportError:
            return nan
