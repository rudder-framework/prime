"""Engine: phase_coherence — instantaneous phase stability."""
import numpy as np
from typing import Dict

def compute(y: np.ndarray) -> Dict[str, float]:
    nan = {'phase_coherence_value': np.nan, 'phase_coherence_std': np.nan,
           'phase_coherence_trend': np.nan}
    if len(y) < 16:
        return nan
    try:
        from scipy.signal import hilbert
        analytic = hilbert(y)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi)
        if len(inst_freq) < 2 or np.std(inst_freq) < 1e-15:
            return nan
        coherence = float(1.0 / (1.0 + np.std(inst_freq)))
        trend = float(np.polyfit(np.arange(len(inst_freq)), inst_freq, 1)[0])
        return {'phase_coherence_value': coherence,
                'phase_coherence_std': float(np.std(inst_freq)),
                'phase_coherence_trend': trend}
    except ImportError:
        return nan
