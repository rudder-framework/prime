"""
ORTHON Fallback Backend
=======================

Basic analysis when PRISM is not available.
Provides useful data profiling without any physics calculations.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List


def analyze(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Basic analysis without PRISM.

    Provides:
    - Column statistics
    - Data quality checks
    - Unit detection
    - Structure analysis

    Does NOT provide:
    - hd_slope
    - Transfer entropy
    - Physics calculations
    - Hamiltonian/Lagrangian
    """
    results = {
        'backend': 'fallback',
        'message': 'Install PRISM for full physics analysis',
        'rows': len(df),
        'columns': len(df.columns),
        'signals': {},
        'issues': [],
        'warnings': [],
    }

    # Analyze numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        series = df[col]

        stats = {
            'count': int(series.count()),
            'nulls': int(series.isna().sum()),
            'unique': int(series.nunique()),
        }

        if stats['count'] > 0:
            stats.update({
                'mean': float(series.mean()),
                'std': float(series.std()) if stats['count'] > 1 else 0.0,
                'min': float(series.min()),
                'max': float(series.max()),
                'median': float(series.median()),
            })

            # Quartiles
            if stats['count'] >= 4:
                stats['q25'] = float(series.quantile(0.25))
                stats['q75'] = float(series.quantile(0.75))

        results['signals'][col] = stats

        # Quality checks
        if stats['nulls'] == len(df):
            results['issues'].append(f"{col}: entirely null")
        elif stats['nulls'] > 0:
            pct = 100 * stats['nulls'] / len(df)
            results['warnings'].append(f"{col}: {stats['nulls']} nulls ({pct:.1f}%)")

        if stats['unique'] == 1:
            results['warnings'].append(f"{col}: constant value ({stats.get('mean', 'N/A')})")

    # Summary
    results['n_signals'] = len(results['signals'])
    results['n_issues'] = len(results['issues'])
    results['n_warnings'] = len(results['warnings'])

    return results


def get_capabilities() -> List[str]:
    """Return list of capabilities this backend provides."""
    return [
        'basic_stats',
        'null_detection',
        'constant_detection',
        'quartiles',
    ]


def get_missing_capabilities() -> List[str]:
    """Return list of capabilities that require PRISM."""
    return [
        'hd_slope',
        'transfer_entropy',
        'hamiltonian',
        'lagrangian',
        'gibbs_free_energy',
        'reynolds_number',
        'lyapunov_exponent',
        'recurrence_quantification',
        'spectral_analysis',
        'regime_detection',
    ]
