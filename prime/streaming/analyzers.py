"""
Prime Real-Time Analyzer

Progressive analysis with immediate and batch metric computation.
"""

import numpy as np
import collections
import time
from typing import Dict, Any, Optional, Tuple

# Import compute functions from the correct packages
from eigendecomp import compute_eigendecomp

try:
    from pmtvs import (
        lyapunov_rosenstein,
        dominant_frequency,
        spectral_entropy,
        permutation_entropy,
    )
    HAS_PMTVS = True
except ImportError:
    HAS_PMTVS = False


def _spectral_flatness(signal: np.ndarray) -> float:
    """Compute spectral flatness (Wiener entropy)."""
    from scipy.fft import rfft
    spectrum = np.abs(rfft(signal)) ** 2
    spectrum = spectrum[spectrum > 1e-30]  # Avoid log(0)
    if len(spectrum) == 0:
        return np.nan
    geo_mean = np.exp(np.mean(np.log(spectrum)))
    arith_mean = np.mean(spectrum)
    return float(geo_mean / arith_mean) if arith_mean > 1e-30 else np.nan


class RealTimeAnalyzer:
    """
    Real-time Manifold analysis with progressive metric computation.

    Handles immediate calculations (eff_dim, eigenvals) and batched
    calculations (Lyapunov, attractor analysis) as data accumulates.
    """

    def __init__(self,
                 window_size: int = 100,
                 batch_size: int = 500,
                 config: Optional[Dict] = None):

        self.window_size = window_size
        self.batch_size = batch_size
        self.config = config or self._default_config()

        # Data buffers
        self.window_buffer = collections.deque(maxlen=window_size)
        self.batch_buffer = collections.deque(maxlen=batch_size)
        self.timestamps = collections.deque(maxlen=batch_size)

        # Analysis state
        self.sample_count = 0
        self.analysis_stage = "initializing"
        self.last_batch_analysis = 0
        self.metrics_history = collections.deque(maxlen=1000)

        # Results cache
        self.latest_instant = {}
        self.latest_batch = {}
        self.alerts = []

        # Start time
        self.start_time = time.time()

    def process_data_point(self, data_point: Dict[str, float]) -> Tuple[Dict, Dict]:
        """
        Process new data point and return instant + batch results.

        Args:
            data_point: Dict with sensor readings {sensor_name: value}

        Returns:
            Tuple of (instant_results, batch_results)
        """
        timestamp = time.time()

        # Convert to array format for Manifold
        if isinstance(data_point, dict):
            values = np.array(list(data_point.values()))
        else:
            values = np.array(data_point)

        # Add to buffers
        self.window_buffer.append(values)
        self.batch_buffer.append(values)
        self.timestamps.append(timestamp)
        self.sample_count += 1

        # Update analysis stage
        self._update_analysis_stage()

        # Compute instant metrics
        instant_results = self._compute_instant_metrics()

        # Compute batch metrics (if enough data accumulated)
        batch_results = self._compute_batch_metrics()

        # Store results
        self.latest_instant = instant_results
        if batch_results:
            self.latest_batch.update(batch_results)

        # Check alerts
        self._check_alerts(instant_results, batch_results)

        # Store in history
        self.metrics_history.append({
            'timestamp': timestamp,
            'sample_count': self.sample_count,
            **instant_results,
            **batch_results
        })

        return instant_results, batch_results

    def _compute_instant_metrics(self) -> Dict[str, Any]:
        """Compute metrics available with current window."""
        if len(self.window_buffer) < 10:
            return {
                'status': 'insufficient_data',
                'sample_count': self.sample_count,
                'analysis_stage': self.analysis_stage
            }

        # Convert buffer to matrix
        signal_matrix = np.array(list(self.window_buffer))

        try:
            # Core eigenstructure analysis using eigendecomp package
            result = compute_eigendecomp(signal_matrix)
            eigenvals = result['eigenvalues']
            eff_dim = result['effective_dim']
            total_variance = result['total_variance']

            # Filter numerical noise
            eigenvals_clean = eigenvals[np.isfinite(eigenvals) & (eigenvals > np.nanmax(eigenvals) * 1e-6)]

            # Additional instant metrics
            eigenval_ratio = (eigenvals_clean[0] / eigenvals_clean[1]
                            if len(eigenvals_clean) > 1 else 1.0)

            # Signal characteristics
            signal_strength = np.mean(np.std(signal_matrix, axis=0))
            noise_level = np.mean(np.abs(np.diff(signal_matrix, axis=0)))

            return {
                'timestamp': time.time(),
                'sample_count': self.sample_count,
                'analysis_stage': self.analysis_stage,
                'eff_dim': float(eff_dim) if np.isfinite(eff_dim) else 0,
                'eigenval_1': float(eigenvals_clean[0]) if len(eigenvals_clean) > 0 else 0,
                'eigenval_2': float(eigenvals_clean[1]) if len(eigenvals_clean) > 1 else 0,
                'eigenval_3': float(eigenvals_clean[2]) if len(eigenvals_clean) > 2 else 0,
                'eigenval_ratio': float(eigenval_ratio),
                'total_variance': float(total_variance) if np.isfinite(total_variance) else 0,
                'signal_strength': float(signal_strength),
                'noise_level': float(noise_level),
                'num_eigenvals': len(eigenvals_clean),
                'status': 'active'
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'sample_count': self.sample_count,
                'analysis_stage': self.analysis_stage
            }

    def _compute_batch_metrics(self) -> Dict[str, Any]:
        """Compute metrics requiring larger data batches."""
        batch_results = {}

        if not HAS_PMTVS:
            return batch_results

        # Lyapunov analysis (requires 200+ samples)
        if (len(self.batch_buffer) >= 200 and
            self.sample_count - self.last_batch_analysis >= 50):

            try:
                # Convert to time series for Lyapunov
                if len(self.batch_buffer) > 0 and len(self.batch_buffer[0]) > 0:
                    # Use first signal for Lyapunov (can be extended to multivariate)
                    signal = np.array([point[0] for point in self.batch_buffer])

                    if len(signal) >= 200:
                        lyapunov = lyapunov_rosenstein(signal)
                        batch_results['lyapunov'] = float(lyapunov)
                        batch_results['lyapunov_status'] = (
                            'stable' if lyapunov < -0.01 else
                            'unstable' if lyapunov > 0.01 else 'neutral'
                        )
            except Exception as e:
                batch_results['lyapunov_error'] = str(e)

        # Spectral analysis (requires 100+ samples)
        if len(self.batch_buffer) >= 100:
            try:
                signal = np.array([point[0] for point in self.batch_buffer])

                if len(signal) >= 100:
                    # Spectral characteristics
                    dom_freq = dominant_frequency(signal)
                    spec_entropy = spectral_entropy(signal)
                    spec_flatness = _spectral_flatness(signal)

                    batch_results.update({
                        'dominant_freq': float(dom_freq),
                        'spectral_entropy': float(spec_entropy),
                        'spectral_flatness': float(spec_flatness)
                    })
            except Exception as e:
                batch_results['spectral_error'] = str(e)

        # Complexity analysis (requires 150+ samples)
        if len(self.batch_buffer) >= 150:
            try:
                signal = np.array([point[0] for point in self.batch_buffer])

                if len(signal) >= 150:
                    perm_entropy = permutation_entropy(signal[:100])  # Limit for speed

                    batch_results.update({
                        'perm_entropy': float(perm_entropy),
                    })
            except Exception as e:
                batch_results['complexity_error'] = str(e)

        if batch_results:
            self.last_batch_analysis = self.sample_count

        return batch_results

    def _update_analysis_stage(self):
        """Update analysis stage based on data availability."""
        if self.sample_count < 10:
            self.analysis_stage = "initializing"
        elif self.sample_count < 50:
            self.analysis_stage = "computing_eigenstructure"
        elif self.sample_count < 100:
            self.analysis_stage = "analyzing_dynamics"
        elif self.sample_count < 200:
            self.analysis_stage = "computing_complexity"
        else:
            self.analysis_stage = "full_analysis"

    def _check_alerts(self, instant_results: Dict, batch_results: Dict):
        """Check for alert conditions."""
        alerts = []

        if instant_results.get('status') == 'active':
            eff_dim = instant_results.get('eff_dim', 0)

            # Threshold checks from config
            critical_threshold = self.config['geometry']['thresholds']['critical_eff_dim']
            warning_threshold = self.config['geometry']['thresholds']['warning_eff_dim']

            if eff_dim < critical_threshold:
                alerts.append({
                    'level': 'CRITICAL',
                    'metric': 'effective_dimension',
                    'value': eff_dim,
                    'threshold': critical_threshold,
                    'message': f'Dimensional collapse detected: {eff_dim:.3f} < {critical_threshold:.3f}',
                    'timestamp': time.time()
                })
            elif eff_dim < warning_threshold:
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'effective_dimension',
                    'value': eff_dim,
                    'threshold': warning_threshold,
                    'message': f'System degradation detected: {eff_dim:.3f} < {warning_threshold:.3f}',
                    'timestamp': time.time()
                })

        # Lyapunov alerts
        if 'lyapunov' in batch_results:
            lyap = batch_results['lyapunov']

            if lyap > 0.1:  # Strongly positive = chaos
                alerts.append({
                    'level': 'WARNING',
                    'metric': 'lyapunov_exponent',
                    'value': lyap,
                    'threshold': 0.1,
                    'message': f'Chaotic behavior detected: lambda = {lyap:.6f}',
                    'timestamp': time.time()
                })

        self.alerts.extend(alerts)
        # Keep only recent alerts
        cutoff_time = time.time() - 300  # 5 minutes
        self.alerts = [a for a in self.alerts if a['timestamp'] > cutoff_time]

    def _default_config(self) -> Dict:
        """Default configuration for real-time analysis."""
        return {
            'geometry': {
                'thresholds': {
                    'critical_eff_dim': 1.5,
                    'warning_eff_dim': 2.0
                }
            },
            'dynamics': {
                'stability': {
                    'stable_threshold': -0.1
                }
            }
        }

    def get_status_summary(self) -> Dict[str, Any]:
        """Get current analysis status summary."""
        return {
            'sample_count': self.sample_count,
            'analysis_stage': self.analysis_stage,
            'window_fill': len(self.window_buffer) / self.window_size,
            'batch_fill': len(self.batch_buffer) / self.batch_size,
            'active_alerts': len([a for a in self.alerts
                                if time.time() - a['timestamp'] < 60]),
            'latest_eff_dim': self.latest_instant.get('eff_dim'),
            'latest_lyapunov': self.latest_batch.get('lyapunov'),
            'uptime_seconds': time.time() - self.start_time
        }

    def reset(self):
        """Reset analyzer state."""
        self.window_buffer.clear()
        self.batch_buffer.clear()
        self.timestamps.clear()
        self.sample_count = 0
        self.analysis_stage = "initializing"
        self.last_batch_analysis = 0
        self.metrics_history.clear()
        self.latest_instant = {}
        self.latest_batch = {}
        self.alerts = []
        self.start_time = time.time()
