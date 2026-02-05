"""
ORTHON Intelligent Pattern Learner

AI system that learns geometric-physical correlations specific to each industrial system.

Automatically discovers:
- Which geometric patterns predict which physical failures
- System-specific normal operating fingerprints
- Optimal warning thresholds for each system type
- User response patterns and preferences
"""

import numpy as np
import time
from collections import deque
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some learning features disabled.")


@dataclass
class LearningState:
    """Current state of the learning system."""
    learning_iterations: int = 0
    prediction_accuracy: float = 0.0
    false_positive_rate: float = 0.0
    last_training_time: Optional[float] = None
    samples_since_training: int = 0


@dataclass
class SystemFingerprint:
    """Learned baseline behavior for a specific system."""
    system_id: str
    system_type: str
    baseline_eff_dim: float = 0.0
    baseline_eff_dim_std: float = 0.0
    baseline_eigenval_ratios: List[float] = field(default_factory=list)
    normal_operating_range: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    failure_signatures: List[Dict] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class IntelligentPatternLearner:
    """
    Learns system-specific patterns from combined geometric and physical data.

    Provides intelligent assessment by combining learned patterns with
    real-time geometric and physical analysis.
    """

    def __init__(self, system_id: str, system_type: str = "generic"):
        self.system_id = system_id
        self.system_type = system_type

        # Learning models (initialized lazily when training)
        self.failure_predictor = None
        self.threshold_optimizer_model = None
        self.pattern_classifier = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None

        # Knowledge base
        self.geometric_patterns: Dict[str, Any] = {}
        self.physical_constraints: Dict[str, Any] = {}
        self.cross_correlations: Dict[str, Dict] = {}
        self.failure_signatures: List[Dict] = []
        self.optimal_thresholds: Dict[str, Dict] = {}

        # System fingerprint
        self.fingerprint: Optional[SystemFingerprint] = None

        # Learning state
        self.state = LearningState()

        # Data buffers for online learning
        self.feature_buffer = deque(maxlen=10000)
        self.label_buffer = deque(maxlen=10000)

        # Default thresholds (updated through learning)
        self.default_thresholds = {
            'effective_dimension': {
                'critical': 1.5,
                'warning': 2.0,
                'normal': 3.0,
            },
            'energy_balance_error': {
                'critical': 0.15,
                'warning': 0.08,
                'normal': 0.03,
            },
            'mass_balance_error': {
                'critical': 0.10,
                'warning': 0.05,
                'normal': 0.02,
            },
        }

    def learn_from_historical_data(
        self,
        historical_data: Any,  # pd.DataFrame or dict
        failure_events: Optional[List[Dict]] = None,
        maintenance_records: Optional[List[Dict]] = None
    ):
        """
        Learn system patterns from historical operation data.

        Args:
            historical_data: Time series with geometric + physical metrics
            failure_events: Known failure incidents with timestamps
            maintenance_records: Scheduled maintenance and repairs
        """
        print(f"Learning patterns for system {self.system_id}")

        failure_events = failure_events or []
        maintenance_records = maintenance_records or []

        # Convert to numpy if pandas DataFrame
        if hasattr(historical_data, 'values'):
            data_array = historical_data.values
            columns = list(historical_data.columns)
        elif isinstance(historical_data, dict):
            columns = list(historical_data.keys())
            data_array = np.column_stack([historical_data[c] for c in columns])
        else:
            data_array = np.array(historical_data)
            columns = [f'feature_{i}' for i in range(data_array.shape[1])]

        # Extract features for learning
        features = self._extract_learning_features(data_array, columns)

        # Create labels from failure events
        labels = self._create_failure_labels(data_array, failure_events, columns)

        # Train failure prediction model
        if SKLEARN_AVAILABLE and len(features) > 100:
            self._train_failure_predictor(features, labels)

        # Learn geometric-physical correlations
        self._learn_cross_domain_patterns(data_array, columns)

        # Optimize alert thresholds
        self._optimize_thresholds(features, labels, failure_events, columns)

        # Create system fingerprint
        self._create_system_fingerprint(data_array, columns)

        self.state.learning_iterations += 1
        self.state.last_training_time = time.time()

        print(f"Learning complete. Iteration: {self.state.learning_iterations}")

    def _extract_learning_features(
        self,
        data: np.ndarray,
        columns: List[str]
    ) -> np.ndarray:
        """Extract feature matrix for machine learning."""

        # Key feature columns (if available)
        feature_names = [
            # Geometric features (from PRISM)
            'effective_dimension', 'eff_dim', 'eigenvalue_ratio', 'participation_ratio',
            'dimensional_trend', 'eigenvalue_1', 'eigenvalue_2', 'eigenvalue_3',

            # Physical features
            'energy_balance_error', 'mass_balance_error', 'entropy_production',
            'temperature_gradient', 'pressure_differential', 'flow_rate_deviation',

            # Derived features
            'geometric_physical_coupling', 'constraint_violation_severity',
            'system_efficiency', 'stability_margin'
        ]

        # Find available features
        available_indices = []
        for name in feature_names:
            if name in columns:
                available_indices.append(columns.index(name))

        if available_indices:
            feature_matrix = data[:, available_indices]
        else:
            # Use all numeric columns
            feature_matrix = data

        # Handle missing values
        feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return feature_matrix

    def _create_failure_labels(
        self,
        data: np.ndarray,
        failure_events: List[Dict],
        columns: List[str]
    ) -> np.ndarray:
        """Create binary labels indicating pre-failure periods."""

        n_samples = len(data)
        labels = np.zeros(n_samples)

        # Find timestamp column if available
        timestamp_col = None
        for col in ['timestamp', 'time', 't']:
            if col in columns:
                timestamp_col = columns.index(col)
                break

        if timestamp_col is not None and failure_events:
            timestamps = data[:, timestamp_col]

            for event in failure_events:
                failure_time = event.get('timestamp', event.get('time', 0))
                lookahead = event.get('lookahead_minutes', 30) * 60  # Convert to seconds

                # Mark samples before failure
                pre_failure_mask = (timestamps >= failure_time - lookahead) & (timestamps < failure_time)
                labels[pre_failure_mask] = 1
        else:
            # No timestamp info - use simple heuristic
            # Assume last 10% of each failure sequence is pre-failure
            if failure_events:
                failure_fraction = min(len(failure_events) / 100, 0.1)
                n_failure_samples = int(n_samples * failure_fraction)
                if n_failure_samples > 0:
                    # Distribute failure labels
                    failure_indices = np.random.choice(
                        n_samples, size=n_failure_samples, replace=False
                    )
                    labels[failure_indices] = 1

        return labels

    def _train_failure_predictor(self, features: np.ndarray, labels: np.ndarray):
        """Train the failure prediction model."""

        if not SKLEARN_AVAILABLE:
            return

        print("Training failure prediction model...")

        # Scale features
        features_scaled = self.scaler.fit_transform(features)

        # Train random forest
        self.failure_predictor = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )

        try:
            self.failure_predictor.fit(features_scaled, labels)

            # Calculate training accuracy
            predictions = self.failure_predictor.predict(features_scaled)
            self.state.prediction_accuracy = np.mean(predictions == labels)

            # Calculate false positive rate
            false_positives = np.sum((predictions == 1) & (labels == 0))
            true_negatives = np.sum((predictions == 0) & (labels == 0))
            if false_positives + true_negatives > 0:
                self.state.false_positive_rate = false_positives / (false_positives + true_negatives)

            print(f"  Accuracy: {self.state.prediction_accuracy:.3f}")
            print(f"  False positive rate: {self.state.false_positive_rate:.3f}")

        except Exception as e:
            print(f"  Training failed: {e}")

    def _learn_cross_domain_patterns(self, data: np.ndarray, columns: List[str]):
        """Learn how geometric changes correlate with physical problems."""

        print("Learning geometric-physical correlations...")

        # Define correlations to discover
        correlations_to_learn = [
            ('effective_dimension', 'energy_balance_error'),
            ('eff_dim', 'energy_balance_error'),
            ('eigenvalue_ratio', 'mass_balance_error'),
            ('dimensional_trend', 'system_efficiency'),
            ('participation_ratio', 'entropy_production'),
            ('eigenvalue_1', 'temperature_gradient'),
        ]

        for geom_metric, phys_metric in correlations_to_learn:
            if geom_metric in columns and phys_metric in columns:
                geom_idx = columns.index(geom_metric)
                phys_idx = columns.index(phys_metric)

                geom_data = data[:, geom_idx]
                phys_data = data[:, phys_idx]

                # Remove NaN values
                valid_mask = ~(np.isnan(geom_data) | np.isnan(phys_data))
                if np.sum(valid_mask) < 10:
                    continue

                geom_valid = geom_data[valid_mask]
                phys_valid = phys_data[valid_mask]

                # Calculate correlation
                correlation = np.corrcoef(geom_valid, phys_valid)[0, 1]

                # Store significant correlations
                if not np.isnan(correlation) and abs(correlation) > 0.3:
                    self.cross_correlations[f"{geom_metric}->{phys_metric}"] = {
                        'correlation': float(correlation),
                        'strength': 'strong' if abs(correlation) > 0.7 else 'moderate',
                        'direction': 'positive' if correlation > 0 else 'negative',
                        'n_samples': int(np.sum(valid_mask)),
                    }

        print(f"  Learned {len(self.cross_correlations)} significant correlations")

    def _optimize_thresholds(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        failure_events: List[Dict],
        columns: List[str]
    ):
        """Optimize alert thresholds to minimize false positives while maximizing early detection."""

        print("Optimizing alert thresholds...")

        # Metrics to optimize
        metrics_to_optimize = {
            'effective_dimension': (0.5, 5.0, 50, 'below'),  # (min, max, n_candidates, direction)
            'eff_dim': (0.5, 5.0, 50, 'below'),
            'energy_balance_error': (0.0, 0.2, 50, 'above'),
            'mass_balance_error': (0.0, 0.1, 50, 'above'),
        }

        for metric, (min_val, max_val, n_candidates, direction) in metrics_to_optimize.items():
            if metric not in columns:
                continue

            metric_idx = columns.index(metric)
            metric_values = features[:, min(metric_idx, features.shape[1]-1)] if metric_idx < features.shape[1] else None

            if metric_values is None:
                continue

            candidates = np.linspace(min_val, max_val, n_candidates)
            best_threshold = None
            best_score = -1
            best_sensitivity = 0
            best_specificity = 0

            for threshold in candidates:
                # Determine predictions based on threshold direction
                if direction == 'below':
                    predictions = (metric_values < threshold).astype(int)
                else:
                    predictions = (metric_values > threshold).astype(int)

                # Calculate confusion matrix components
                tp = np.sum((predictions == 1) & (labels == 1))
                fp = np.sum((predictions == 1) & (labels == 0))
                fn = np.sum((predictions == 0) & (labels == 1))
                tn = np.sum((predictions == 0) & (labels == 0))

                # Calculate metrics
                if tp + fn > 0:
                    sensitivity = tp / (tp + fn)
                else:
                    sensitivity = 0

                if tn + fp > 0:
                    specificity = tn / (tn + fp)
                else:
                    specificity = 0

                # Weighted score (prefer early detection)
                score = 0.7 * sensitivity + 0.3 * specificity

                if score > best_score:
                    best_score = score
                    best_threshold = threshold
                    best_sensitivity = sensitivity
                    best_specificity = specificity

            if best_threshold is not None:
                self.optimal_thresholds[metric] = {
                    'threshold': float(best_threshold),
                    'direction': direction,
                    'expected_sensitivity': float(best_sensitivity),
                    'expected_specificity': float(best_specificity),
                    'optimization_score': float(best_score),
                }

        print(f"  Optimized {len(self.optimal_thresholds)} alert thresholds")

    def _create_system_fingerprint(self, data: np.ndarray, columns: List[str]):
        """Create baseline fingerprint for this system."""

        print("Creating system fingerprint...")

        fingerprint = SystemFingerprint(
            system_id=self.system_id,
            system_type=self.system_type,
        )

        # Calculate baseline effective dimension
        for eff_col in ['effective_dimension', 'eff_dim']:
            if eff_col in columns:
                idx = columns.index(eff_col)
                values = data[:, idx]
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    fingerprint.baseline_eff_dim = float(np.median(valid_values))
                    fingerprint.baseline_eff_dim_std = float(np.std(valid_values))
                break

        # Calculate normal operating ranges for key metrics
        range_metrics = [
            'effective_dimension', 'eff_dim', 'eigenvalue_1', 'eigenvalue_2',
            'energy_balance_error', 'mass_balance_error', 'system_efficiency'
        ]

        for metric in range_metrics:
            if metric in columns:
                idx = columns.index(metric)
                values = data[:, idx]
                valid_values = values[~np.isnan(values)]
                if len(valid_values) > 0:
                    p5 = float(np.percentile(valid_values, 5))
                    p95 = float(np.percentile(valid_values, 95))
                    fingerprint.normal_operating_range[metric] = (p5, p95)

        fingerprint.updated_at = time.time()
        self.fingerprint = fingerprint

        print(f"  Baseline eff_dim: {fingerprint.baseline_eff_dim:.3f} +/- {fingerprint.baseline_eff_dim_std:.3f}")

    def assess_current_state(
        self,
        geometric_state: Dict,
        physical_state: Dict
    ) -> Dict[str, Any]:
        """
        AI-powered assessment of current system health.

        Combines geometric + physical + learned patterns.
        """

        # Prepare feature vector
        current_features = self._prepare_current_features(geometric_state, physical_state)

        # AI prediction (if model trained)
        failure_probability = self._predict_failure_probability(current_features)

        # Cross-domain analysis
        interaction_risks = self._assess_interaction_risks(geometric_state, physical_state)

        # Learned threshold assessment
        threshold_violations = self._check_learned_thresholds(geometric_state, physical_state)

        # Integrated health score
        health_score = self._calculate_integrated_health_score(
            failure_probability, interaction_risks, threshold_violations
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(health_score, interaction_risks)

        return {
            'ai_health_assessment': {
                'overall_health_score': health_score,
                'failure_probability_30min': failure_probability,
                'critical_interactions': interaction_risks,
                'threshold_status': threshold_violations,
                'confidence_level': self._calculate_confidence(),
                'learning_maturity': self.state.learning_iterations,
            },
            'geometric_analysis': geometric_state,
            'physical_analysis': physical_state,
            'recommendations': recommendations,
            'system_fingerprint': {
                'baseline_eff_dim': self.fingerprint.baseline_eff_dim if self.fingerprint else None,
                'normal_ranges': self.fingerprint.normal_operating_range if self.fingerprint else {},
            },
        }

    def _prepare_current_features(
        self,
        geometric_state: Dict,
        physical_state: Dict
    ) -> np.ndarray:
        """Prepare feature vector from current state."""

        features = []

        # Geometric features
        features.append(geometric_state.get('effective_dimension', geometric_state.get('eff_dim', 0)))
        features.append(geometric_state.get('eigenvalue_ratio', 0))
        features.append(geometric_state.get('eigenvalue_1', 0))
        features.append(geometric_state.get('eigenvalue_2', 0))
        features.append(geometric_state.get('participation_ratio', 0))

        # Physical features
        features.append(physical_state.get('energy_balance_error', 0))
        features.append(physical_state.get('mass_balance_error', 0))
        features.append(physical_state.get('entropy_production', 0))
        features.append(physical_state.get('system_efficiency', 0))

        return np.array(features).reshape(1, -1)

    def _predict_failure_probability(self, features: np.ndarray) -> float:
        """Predict failure probability using trained model."""

        if self.failure_predictor is None or not SKLEARN_AVAILABLE:
            return 0.0

        try:
            # Scale features
            features_scaled = self.scaler.transform(features)

            # Get probability
            probabilities = self.failure_predictor.predict_proba(features_scaled)
            return float(probabilities[0, 1])  # Probability of failure class

        except Exception:
            return 0.0

    def _assess_interaction_risks(
        self,
        geometric_state: Dict,
        physical_state: Dict
    ) -> List[Dict]:
        """Assess risks from geometric-physical interactions."""

        risks = []

        for correlation_key, correlation_info in self.cross_correlations.items():
            geom_metric, phys_metric = correlation_key.split('->')

            geom_value = geometric_state.get(geom_metric)
            phys_value = physical_state.get(phys_metric)

            if geom_value is not None and phys_value is not None:
                # Check if values are in concerning ranges
                correlation = correlation_info['correlation']

                # If strongly correlated and one is abnormal, flag risk
                geom_abnormal = self._is_abnormal(geom_metric, geom_value)
                phys_abnormal = self._is_abnormal(phys_metric, phys_value)

                if geom_abnormal or phys_abnormal:
                    risk_level = 'high' if abs(correlation) > 0.7 else 'moderate'
                    risks.append({
                        'variables': f"{geom_metric}, {phys_metric}",
                        'correlation': correlation,
                        'risk_level': risk_level,
                        'geometric_abnormal': geom_abnormal,
                        'physical_abnormal': phys_abnormal,
                    })

        return risks

    def _is_abnormal(self, metric: str, value: float) -> bool:
        """Check if value is outside normal range."""

        if self.fingerprint and metric in self.fingerprint.normal_operating_range:
            low, high = self.fingerprint.normal_operating_range[metric]
            return value < low or value > high

        return False

    def _check_learned_thresholds(
        self,
        geometric_state: Dict,
        physical_state: Dict
    ) -> Dict[str, str]:
        """Check current state against learned thresholds."""

        combined_state = {**geometric_state, **physical_state}
        violations = {}

        for metric, threshold_info in self.optimal_thresholds.items():
            value = combined_state.get(metric)
            if value is None:
                continue

            threshold = threshold_info['threshold']
            direction = threshold_info['direction']

            if direction == 'below' and value < threshold:
                violations[metric] = 'violated'
            elif direction == 'above' and value > threshold:
                violations[metric] = 'violated'
            else:
                violations[metric] = 'ok'

        return violations

    def _calculate_integrated_health_score(
        self,
        failure_probability: float,
        interaction_risks: List[Dict],
        threshold_violations: Dict[str, str]
    ) -> float:
        """Calculate overall health score from all inputs."""

        # Start with base score
        score = 1.0

        # Factor in failure probability
        score -= failure_probability * 0.4

        # Factor in interaction risks
        high_risks = sum(1 for r in interaction_risks if r['risk_level'] == 'high')
        moderate_risks = sum(1 for r in interaction_risks if r['risk_level'] == 'moderate')
        score -= high_risks * 0.15
        score -= moderate_risks * 0.05

        # Factor in threshold violations
        n_violations = sum(1 for v in threshold_violations.values() if v == 'violated')
        score -= n_violations * 0.1

        return max(0.0, min(1.0, score))

    def _calculate_confidence(self) -> float:
        """Calculate confidence in AI assessment."""

        # Confidence increases with learning iterations
        learning_factor = min(self.state.learning_iterations / 10, 1.0) * 0.3

        # Confidence from prediction accuracy
        accuracy_factor = self.state.prediction_accuracy * 0.4

        # Confidence from having a fingerprint
        fingerprint_factor = 0.3 if self.fingerprint else 0.0

        return learning_factor + accuracy_factor + fingerprint_factor

    def _generate_recommendations(
        self,
        health_score: float,
        interactions: List[Dict]
    ) -> List[str]:
        """Generate AI-powered operational recommendations."""

        recommendations = []

        if health_score < 0.3:  # Critical
            recommendations.append("IMMEDIATE ACTION: System approaching failure state")
            recommendations.append("Recommend emergency shutdown sequence evaluation")

        elif health_score < 0.6:  # Warning
            recommendations.append("CAUTION: Degraded operation detected")

            # Specific recommendations based on learned patterns
            for interaction in interactions:
                if interaction['risk_level'] == 'high':
                    recommendations.append(f"Monitor {interaction['variables']} closely")

        else:  # Healthy
            recommendations.append("System operating within learned normal parameters")

        return recommendations

    def online_update(self, geometric_state: Dict, physical_state: Dict, outcome: Optional[str] = None):
        """Update learning from streaming data."""

        # Prepare features
        features = self._prepare_current_features(geometric_state, physical_state)

        # Add to buffer
        self.feature_buffer.append(features.flatten())

        # Add label if outcome provided
        if outcome is not None:
            label = 1 if outcome in ['failure', 'incident', 'alarm'] else 0
            self.label_buffer.append(label)

        self.state.samples_since_training += 1

        # Retrain periodically
        if (self.state.samples_since_training >= 1000 and
            len(self.label_buffer) >= 100):
            self._retrain_online()

    def _retrain_online(self):
        """Retrain model with accumulated online data."""

        if not SKLEARN_AVAILABLE or len(self.feature_buffer) < 100:
            return

        print(f"Retraining with {len(self.feature_buffer)} online samples...")

        features = np.array(list(self.feature_buffer))
        labels = np.array(list(self.label_buffer))

        # Ensure same length
        min_len = min(len(features), len(labels))
        features = features[:min_len]
        labels = labels[:min_len]

        self._train_failure_predictor(features, labels)
        self.state.samples_since_training = 0

    def get_learning_status(self) -> Dict[str, Any]:
        """Get current learning system status."""

        return {
            'system_id': self.system_id,
            'system_type': self.system_type,
            'learning_iterations': self.state.learning_iterations,
            'prediction_accuracy': self.state.prediction_accuracy,
            'false_positive_rate': self.state.false_positive_rate,
            'n_correlations_learned': len(self.cross_correlations),
            'n_thresholds_optimized': len(self.optimal_thresholds),
            'has_fingerprint': self.fingerprint is not None,
            'samples_in_buffer': len(self.feature_buffer),
            'confidence': self._calculate_confidence(),
        }
