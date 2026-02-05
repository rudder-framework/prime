"""
ORTHON Threshold Optimizer

Automatically optimizes alert thresholds based on system performance
and historical outcomes.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class ThresholdConfig:
    """Configuration for a single threshold."""
    metric: str
    current_value: float
    direction: str  # 'above' or 'below' - which direction triggers alert
    sensitivity: float = 0.5  # 0-1, how sensitive the threshold is
    last_optimized: float = field(default_factory=time.time)
    optimization_history: List[float] = field(default_factory=list)


@dataclass
class ThresholdPerformance:
    """Performance metrics for a threshold."""
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    total_alerts: int = 0
    avg_lead_time: float = 0.0  # Average time between alert and incident


class ThresholdOptimizer:
    """
    Optimizes alert thresholds to minimize false positives while maximizing
    early detection of actual problems.

    Uses historical outcomes to learn optimal threshold values for each metric.
    """

    def __init__(self, system_id: str):
        self.system_id = system_id

        # Threshold configurations
        self.thresholds: Dict[str, ThresholdConfig] = {}

        # Performance tracking
        self.performance: Dict[str, ThresholdPerformance] = {}

        # Event buffers for learning
        self.alert_buffer: deque = deque(maxlen=1000)
        self.incident_buffer: deque = deque(maxlen=100)

        # Optimization settings
        self.optimization_interval = 3600  # Optimize every hour
        self.min_samples_for_optimization = 50
        self.last_optimization_time = 0

        # Default thresholds
        self._initialize_default_thresholds()

    def _initialize_default_thresholds(self):
        """Initialize with default thresholds."""

        defaults = [
            ('effective_dimension', 2.0, 'below'),
            ('eigenvalue_ratio', 5.0, 'above'),
            ('energy_balance_error', 0.08, 'above'),
            ('mass_balance_error', 0.05, 'above'),
            ('system_efficiency', 0.7, 'below'),
            ('lyapunov_exponent', 0.1, 'above'),
        ]

        for metric, value, direction in defaults:
            self.thresholds[metric] = ThresholdConfig(
                metric=metric,
                current_value=value,
                direction=direction,
            )
            self.performance[metric] = ThresholdPerformance()

    def record_alert(
        self,
        metric: str,
        value: float,
        threshold: float,
        triggered: bool,
        timestamp: Optional[float] = None
    ):
        """Record an alert event for learning."""

        timestamp = timestamp or time.time()

        self.alert_buffer.append({
            'timestamp': timestamp,
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'triggered': triggered,
        })

        if triggered and metric in self.performance:
            self.performance[metric].total_alerts += 1

    def record_incident(
        self,
        incident_type: str,
        timestamp: Optional[float] = None,
        severity: str = 'warning',
        related_metrics: Optional[List[str]] = None
    ):
        """Record an actual incident for learning."""

        timestamp = timestamp or time.time()

        self.incident_buffer.append({
            'timestamp': timestamp,
            'incident_type': incident_type,
            'severity': severity,
            'related_metrics': related_metrics or [],
        })

        # Update performance metrics based on recent alerts
        self._update_performance_from_incident(timestamp, related_metrics or [])

    def _update_performance_from_incident(
        self,
        incident_time: float,
        related_metrics: List[str]
    ):
        """Update threshold performance based on incident."""

        lookahead_window = 1800  # 30 minutes before incident

        for alert in self.alert_buffer:
            if incident_time - lookahead_window <= alert['timestamp'] <= incident_time:
                metric = alert['metric']

                if metric not in self.performance:
                    continue

                perf = self.performance[metric]

                if alert['triggered']:
                    if metric in related_metrics or not related_metrics:
                        # Alert preceded incident - true positive
                        perf.true_positives += 1
                        lead_time = incident_time - alert['timestamp']
                        # Update average lead time
                        perf.avg_lead_time = (
                            perf.avg_lead_time * (perf.true_positives - 1) + lead_time
                        ) / perf.true_positives
                    else:
                        # Alert was unrelated to incident
                        pass
                else:
                    if metric in related_metrics:
                        # Missed alert - false negative
                        perf.false_negatives += 1

    def record_false_positive(self, metric: str, timestamp: Optional[float] = None):
        """Record a false positive alert."""

        if metric in self.performance:
            self.performance[metric].false_positives += 1

    def check_and_optimize(self, force: bool = False) -> Dict[str, Any]:
        """Check if optimization is needed and run if so."""

        current_time = time.time()

        should_optimize = (
            force or
            (current_time - self.last_optimization_time > self.optimization_interval and
             len(self.alert_buffer) >= self.min_samples_for_optimization)
        )

        if not should_optimize:
            return {'optimized': False, 'reason': 'Not time yet'}

        return self.optimize_all_thresholds()

    def optimize_all_thresholds(self) -> Dict[str, Any]:
        """Optimize all threshold values."""

        results = {'optimized': True, 'metrics': {}}

        for metric, config in self.thresholds.items():
            result = self.optimize_threshold(metric)
            results['metrics'][metric] = result

        self.last_optimization_time = time.time()

        return results

    def optimize_threshold(self, metric: str) -> Dict[str, Any]:
        """Optimize threshold for a specific metric."""

        if metric not in self.thresholds:
            return {'status': 'error', 'message': f'Unknown metric: {metric}'}

        config = self.thresholds[metric]
        perf = self.performance.get(metric, ThresholdPerformance())

        # Get relevant alerts
        metric_alerts = [a for a in self.alert_buffer if a['metric'] == metric]

        if len(metric_alerts) < 20:
            return {'status': 'insufficient_data', 'n_samples': len(metric_alerts)}

        # Extract values
        values = np.array([a['value'] for a in metric_alerts])

        # Calculate current performance metrics
        total = perf.true_positives + perf.false_positives + perf.false_negatives + perf.true_negatives

        if total > 0:
            current_precision = perf.true_positives / max(perf.true_positives + perf.false_positives, 1)
            current_recall = perf.true_positives / max(perf.true_positives + perf.false_negatives, 1)
        else:
            current_precision = 0.5
            current_recall = 0.5

        # Optimize threshold
        old_value = config.current_value
        new_value = self._find_optimal_threshold(
            values, config.direction, config.sensitivity
        )

        # Don't change too drastically
        max_change = 0.2  # Maximum 20% change per optimization
        if abs(new_value - old_value) / max(abs(old_value), 1e-6) > max_change:
            if new_value > old_value:
                new_value = old_value * (1 + max_change)
            else:
                new_value = old_value * (1 - max_change)

        # Update configuration
        config.current_value = new_value
        config.last_optimized = time.time()
        config.optimization_history.append(new_value)

        # Keep history bounded
        if len(config.optimization_history) > 100:
            config.optimization_history = config.optimization_history[-50:]

        return {
            'status': 'optimized',
            'old_value': old_value,
            'new_value': new_value,
            'change_pct': (new_value - old_value) / max(abs(old_value), 1e-6) * 100,
            'current_precision': current_precision,
            'current_recall': current_recall,
            'n_samples': len(metric_alerts),
        }

    def _find_optimal_threshold(
        self,
        values: np.ndarray,
        direction: str,
        sensitivity: float
    ) -> float:
        """Find optimal threshold value using percentile-based approach."""

        # sensitivity 0 = very few alerts (high threshold for 'above', low for 'below')
        # sensitivity 1 = many alerts (low threshold for 'above', high for 'below')

        if direction == 'above':
            # Lower percentile = more sensitive = more alerts
            percentile = (1 - sensitivity) * 100
            threshold = np.percentile(values, percentile)
        else:
            # Higher percentile = more sensitive = more alerts
            percentile = sensitivity * 100
            threshold = np.percentile(values, percentile)

        return float(threshold)

    def get_threshold(self, metric: str) -> Optional[float]:
        """Get current threshold value for a metric."""

        if metric in self.thresholds:
            return self.thresholds[metric].current_value
        return None

    def set_threshold(
        self,
        metric: str,
        value: float,
        direction: str = 'above',
        sensitivity: float = 0.5
    ):
        """Manually set a threshold."""

        self.thresholds[metric] = ThresholdConfig(
            metric=metric,
            current_value=value,
            direction=direction,
            sensitivity=sensitivity,
        )
        self.performance[metric] = ThresholdPerformance()

    def set_sensitivity(self, metric: str, sensitivity: float):
        """Set sensitivity for a threshold (0-1)."""

        if metric in self.thresholds:
            self.thresholds[metric].sensitivity = np.clip(sensitivity, 0, 1)

    def evaluate_value(self, metric: str, value: float) -> Dict[str, Any]:
        """Evaluate a value against threshold and return status."""

        if metric not in self.thresholds:
            return {'status': 'unknown', 'reason': 'No threshold configured'}

        config = self.thresholds[metric]
        threshold = config.current_value
        direction = config.direction

        if direction == 'above':
            triggered = value > threshold
            margin = (value - threshold) / max(abs(threshold), 1e-6)
        else:
            triggered = value < threshold
            margin = (threshold - value) / max(abs(threshold), 1e-6)

        if triggered:
            if margin > 0.5:
                severity = 'critical'
            elif margin > 0.2:
                severity = 'warning'
            else:
                severity = 'info'
        else:
            severity = 'ok'

        return {
            'metric': metric,
            'value': value,
            'threshold': threshold,
            'direction': direction,
            'triggered': triggered,
            'severity': severity,
            'margin': margin,
        }

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all thresholds."""

        summary = {}

        for metric, perf in self.performance.items():
            total = (perf.true_positives + perf.false_positives +
                     perf.true_negatives + perf.false_negatives)

            if total > 0:
                precision = perf.true_positives / max(perf.true_positives + perf.false_positives, 1)
                recall = perf.true_positives / max(perf.true_positives + perf.false_negatives, 1)
                f1 = 2 * precision * recall / max(precision + recall, 1e-6)
            else:
                precision = recall = f1 = 0

            config = self.thresholds.get(metric)

            summary[metric] = {
                'current_threshold': config.current_value if config else None,
                'direction': config.direction if config else None,
                'sensitivity': config.sensitivity if config else None,
                'total_alerts': perf.total_alerts,
                'true_positives': perf.true_positives,
                'false_positives': perf.false_positives,
                'false_negatives': perf.false_negatives,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_lead_time_seconds': perf.avg_lead_time,
            }

        return summary

    def export_config(self) -> Dict:
        """Export threshold configuration for persistence."""

        return {
            'system_id': self.system_id,
            'thresholds': {
                metric: {
                    'current_value': config.current_value,
                    'direction': config.direction,
                    'sensitivity': config.sensitivity,
                }
                for metric, config in self.thresholds.items()
            },
            'last_optimization': self.last_optimization_time,
        }

    def import_config(self, config: Dict):
        """Import threshold configuration from persistence."""

        if 'thresholds' in config:
            for metric, cfg in config['thresholds'].items():
                self.thresholds[metric] = ThresholdConfig(
                    metric=metric,
                    current_value=cfg['current_value'],
                    direction=cfg['direction'],
                    sensitivity=cfg.get('sensitivity', 0.5),
                )
                if metric not in self.performance:
                    self.performance[metric] = ThresholdPerformance()

        self.last_optimization_time = config.get('last_optimization', 0)
