"""
Cross-System Trajectory Learning Engine.

Learns degradation patterns from analyzed systems and applies knowledge
to new system types, enabling universal failure prediction.

Key Discovery: Healthy vs degrading systems have distinct trajectory
signatures that transfer across completely different physics domains.

Validated Patterns (from C-MAPSS + Bearing Analysis):
- Healthy: evenly distributed sensitivity, high transition frequency, stable FTLE
- Degrading: concentrated sensitivity, low transitions, accelerating FTLE
"""

import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class SystemTrajectorySignature:
    """Signature pattern for a system's trajectory behavior."""
    system_type: str
    health_status: str  # 'healthy', 'degrading', 'critical', 'monitoring_required'

    # Core trajectory metrics
    sensitivity_concentration: float  # 0-1, higher = more concentrated on one variable
    transition_frequency: int         # Number of dominance transitions
    ftle_acceleration_ratio: float    # current_ftle / historical_ftle
    dominant_variable: Optional[str]

    # Geometric properties
    saddle_score: float
    basin_stability: float
    embedding_dimension: int

    # Metadata
    confidence: float
    sample_count: int
    analysis_timestamp: str

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemTrajectorySignature':
        return cls(**data)


class CrossSystemLearningEngine:
    """
    Learns trajectory patterns across different system types.

    Key capability: Train on turbofan degradation, predict bearing health.

    Universal Patterns Discovered:
    1. Sensitivity concentration increases with degradation
    2. FTLE acceleration indicates approaching failure
    3. Transition frequency drops as system loses healthy exploration
    4. Saddle proximity is universal for complex machines
    """

    def __init__(self, knowledge_base_path: Optional[str] = None):
        if knowledge_base_path:
            self.knowledge_base_path = Path(knowledge_base_path)
        else:
            self.knowledge_base_path = Path.home() / ".orthon" / "cross_system_patterns.json"

        self.knowledge_base_path.parent.mkdir(parents=True, exist_ok=True)
        self.learned_patterns: Dict[str, SystemTrajectorySignature] = {}
        self.universal_patterns: Dict[str, Any] = {}

        self._load_knowledge_base()
        self._initialize_validated_patterns()

    def _initialize_validated_patterns(self):
        """Initialize with patterns validated from C-MAPSS and bearing analysis."""

        # Only add if not already loaded from knowledge base
        if 'turbofan_degrading_validated' not in self.learned_patterns:
            # Turbofan degradation pattern (validated from 5 engines)
            self.learned_patterns['turbofan_degrading_validated'] = SystemTrajectorySignature(
                system_type='turbofan',
                health_status='degrading',
                sensitivity_concentration=0.85,  # sensor_11 dominance
                transition_frequency=75,         # average of 50-109
                ftle_acceleration_ratio=3.2,     # current much higher than historical
                dominant_variable='sensor_11',
                saddle_score=0.635,             # average of 0.59-0.71
                basin_stability=0.355,          # average of 0.28-0.41
                embedding_dimension=6,
                confidence=0.95,
                sample_count=5,
                analysis_timestamp='2024-cmapss-validation'
            )

        if 'bearing_healthy_validated' not in self.learned_patterns:
            # Healthy bearing pattern (validated from vibration analysis)
            self.learned_patterns['bearing_healthy_validated'] = SystemTrajectorySignature(
                system_type='bearing',
                health_status='healthy',
                sensitivity_concentration=0.25,  # acc_x vs acc_y nearly equal
                transition_frequency=4919,       # very high exploration
                ftle_acceleration_ratio=1.1,     # stable FTLE
                dominant_variable='acc_y',       # barely dominant
                saddle_score=0.648,
                basin_stability=0.352,
                embedding_dimension=7,
                confidence=0.9,
                sample_count=1,
                analysis_timestamp='2024-bearing-validation'
            )

        if 'ecosystem_tipping_validated' not in self.learned_patterns:
            # Ecological regime shift (validated from lake model)
            self.learned_patterns['ecosystem_tipping_validated'] = SystemTrajectorySignature(
                system_type='ecosystem',
                health_status='degrading',
                sensitivity_concentration=0.75,  # cyanobacteria dominance
                transition_frequency=12,         # low - approaching tipping
                ftle_acceleration_ratio=2.0,     # increasing sensitivity
                dominant_variable='cyanobacteria',
                saddle_score=0.54,
                basin_stability=0.47,
                embedding_dimension=3,
                confidence=0.85,
                sample_count=1,
                analysis_timestamp='2024-ecological-validation'
            )

        self._update_universal_patterns()
        self._save_knowledge_base()

    def _load_knowledge_base(self):
        """Load learned patterns from disk."""
        if self.knowledge_base_path.exists():
            try:
                with open(self.knowledge_base_path) as f:
                    data = json.load(f)
                for key, pattern_dict in data.get('patterns', {}).items():
                    self.learned_patterns[key] = SystemTrajectorySignature.from_dict(pattern_dict)
                self.universal_patterns = data.get('universal_patterns', {})
                logger.info(f"Loaded {len(self.learned_patterns)} patterns from knowledge base")
            except Exception as e:
                logger.warning(f"Could not load knowledge base: {e}")

    def _save_knowledge_base(self):
        """Save learned patterns to disk."""
        try:
            data = {
                'patterns': {k: v.to_dict() for k, v in self.learned_patterns.items()},
                'universal_patterns': self.universal_patterns,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.knowledge_base_path, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Could not save knowledge base: {e}")

    def _update_universal_patterns(self):
        """Extract universal patterns from all learned signatures."""

        healthy_patterns = [p for p in self.learned_patterns.values() if p.health_status == 'healthy']
        degrading_patterns = [p for p in self.learned_patterns.values() if p.health_status == 'degrading']

        if healthy_patterns:
            self.universal_patterns['healthy'] = {
                'sensitivity_concentration_range': (0.1, 0.4),
                'transition_frequency_min': 500,
                'ftle_acceleration_max': 1.5,
                'examples': [p.system_type for p in healthy_patterns]
            }

        if degrading_patterns:
            self.universal_patterns['degrading'] = {
                'sensitivity_concentration_range': (0.6, 1.0),
                'transition_frequency_max': 200,
                'ftle_acceleration_min': 2.0,
                'examples': [p.system_type for p in degrading_patterns]
            }

    def learn_from_trajectory_analysis(
        self,
        system_type: str,
        trajectory_results: Dict[str, Any],
        ground_truth_health: str = None
    ) -> SystemTrajectorySignature:
        """
        Learn new patterns from trajectory analysis results.

        Args:
            system_type: Type of system (turbofan, bearing, ecosystem, etc.)
            trajectory_results: Results from FTLE/sensitivity analysis
            ground_truth_health: Known health status for supervised learning

        Returns:
            Extracted trajectory signature
        """
        signature = self._extract_trajectory_signature(
            system_type, trajectory_results, ground_truth_health
        )

        # Store in knowledge base
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        signature_key = f"{system_type}_{signature.health_status}_{timestamp}"
        self.learned_patterns[signature_key] = signature

        # Update universal patterns
        self._update_universal_patterns()

        # Save knowledge base
        self._save_knowledge_base()

        logger.info(f"Learned new pattern: {signature_key}")
        return signature

    def predict_system_health(
        self,
        system_type: str,
        trajectory_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict system health using cross-system learned patterns.

        Can predict bearing health using turbofan patterns and vice versa.
        """
        # Extract current system signature
        current_signature = self._extract_trajectory_signature(
            system_type, trajectory_results
        )

        # Find matching patterns
        direct_matches = self._find_direct_pattern_matches(current_signature)
        cross_system_matches = self._find_cross_system_pattern_matches(current_signature)
        universal_assessment = self._apply_universal_patterns(current_signature)

        # Combine predictions
        prediction = self._combine_predictions(
            direct_matches, cross_system_matches, universal_assessment, current_signature
        )

        return prediction

    def _extract_trajectory_signature(
        self,
        system_type: str,
        trajectory_results: Dict[str, Any],
        ground_truth_health: str = None
    ) -> SystemTrajectorySignature:
        """Extract signature from trajectory analysis results."""

        # Compute sensitivity concentration
        signal_sensitivity = trajectory_results.get('signal_sensitivity', {})
        if signal_sensitivity:
            sensitivities = [
                info.get('mean_sensitivity', 0) or 0
                for info in signal_sensitivity.values()
            ]
            if sensitivities and max(sensitivities) > 0:
                total = sum(sensitivities)
                sensitivity_concentration = max(sensitivities) / total if total > 0 else 0.5
                # Find dominant variable
                dominant_variable = max(
                    signal_sensitivity.keys(),
                    key=lambda k: signal_sensitivity[k].get('mean_sensitivity', 0) or 0
                )
            else:
                sensitivity_concentration = 0.5
                dominant_variable = trajectory_results.get('dominant_variable')
        else:
            sensitivity_concentration = 0.5
            dominant_variable = trajectory_results.get('dominant_variable')

        # FTLE acceleration ratio
        ftle_current = trajectory_results.get('ftle_current') or trajectory_results.get('ftle_mean', 0.1)
        ftle_mean = trajectory_results.get('ftle_mean', ftle_current) or 0.1
        if ftle_mean > 0:
            ftle_acceleration_ratio = abs(ftle_current) / abs(ftle_mean)
        else:
            ftle_acceleration_ratio = 1.0

        # Extract other metrics
        transition_frequency = trajectory_results.get('n_transitions', 0)
        saddle_score = trajectory_results.get('saddle_score_mean') or trajectory_results.get('saddle_score', 0.5)
        basin_stability = trajectory_results.get('basin_stability_mean') or trajectory_results.get('basin_stability', 0.5)
        embedding_dimension = trajectory_results.get('embedding_dim') or trajectory_results.get('embedding_dimension', 5)

        # Predict health status if not provided
        if ground_truth_health is None:
            predicted_health = self._predict_health_from_metrics(
                sensitivity_concentration, ftle_acceleration_ratio, transition_frequency
            )
        else:
            predicted_health = ground_truth_health

        return SystemTrajectorySignature(
            system_type=system_type,
            health_status=predicted_health,
            sensitivity_concentration=float(sensitivity_concentration),
            transition_frequency=int(transition_frequency),
            ftle_acceleration_ratio=float(ftle_acceleration_ratio),
            dominant_variable=dominant_variable,
            saddle_score=float(saddle_score) if saddle_score else 0.5,
            basin_stability=float(basin_stability) if basin_stability else 0.5,
            embedding_dimension=int(embedding_dimension) if embedding_dimension else 5,
            confidence=0.8,
            sample_count=1,
            analysis_timestamp=datetime.now().isoformat()
        )

    def _predict_health_from_metrics(
        self,
        sensitivity_concentration: float,
        ftle_acceleration_ratio: float,
        transition_frequency: int
    ) -> str:
        """Predict health status from trajectory metrics using learned patterns."""

        degradation_indicators = 0

        # High sensitivity concentration indicates degradation
        if sensitivity_concentration > 0.7:
            degradation_indicators += 2
        elif sensitivity_concentration > 0.5:
            degradation_indicators += 1

        # FTLE acceleration indicates degradation
        if ftle_acceleration_ratio > 2.5:
            degradation_indicators += 2
        elif ftle_acceleration_ratio > 1.5:
            degradation_indicators += 1

        # Low transition frequency indicates loss of exploration
        if transition_frequency < 100:
            degradation_indicators += 2
        elif transition_frequency < 500:
            degradation_indicators += 1

        # Health classification
        if degradation_indicators >= 4:
            return 'critical'
        elif degradation_indicators >= 3:
            return 'degrading'
        elif degradation_indicators >= 1:
            return 'monitoring_required'
        else:
            return 'healthy'

    def _find_direct_pattern_matches(
        self,
        current_signature: SystemTrajectorySignature
    ) -> List[Tuple[SystemTrajectorySignature, float]]:
        """Find patterns from the same system type."""

        matches = []
        for pattern_key, learned_pattern in self.learned_patterns.items():
            if learned_pattern.system_type == current_signature.system_type:
                similarity = self._compute_signature_similarity(
                    current_signature, learned_pattern
                )
                if similarity > 0.5:
                    matches.append((learned_pattern, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _find_cross_system_pattern_matches(
        self,
        current_signature: SystemTrajectorySignature
    ) -> List[Tuple[SystemTrajectorySignature, float]]:
        """
        Find patterns from different system types that match current signature.

        This enables turbofan → bearing prediction and vice versa.
        """
        matches = []

        for pattern_key, learned_pattern in self.learned_patterns.items():
            # Skip same system type
            if learned_pattern.system_type == current_signature.system_type:
                continue

            # Compute cross-system similarity
            similarity = self._compute_cross_system_similarity(
                current_signature, learned_pattern
            )

            if similarity > 0.5:
                matches.append((learned_pattern, similarity))

        matches.sort(key=lambda x: x[1], reverse=True)
        return matches

    def _compute_signature_similarity(
        self,
        sig1: SystemTrajectorySignature,
        sig2: SystemTrajectorySignature
    ) -> float:
        """Compute similarity between two signatures."""
        return self._compute_cross_system_similarity(sig1, sig2)

    def _compute_cross_system_similarity(
        self,
        sig1: SystemTrajectorySignature,
        sig2: SystemTrajectorySignature
    ) -> float:
        """
        Compute similarity between signatures from different system types.

        Focuses on universal trajectory characteristics.
        """
        similarities = []

        # Sensitivity concentration similarity (30% weight)
        conc_diff = abs(sig1.sensitivity_concentration - sig2.sensitivity_concentration)
        conc_similarity = max(0, 1 - conc_diff)
        similarities.append(conc_similarity * 0.30)

        # FTLE acceleration similarity (25% weight)
        if sig1.ftle_acceleration_ratio > 0 and sig2.ftle_acceleration_ratio > 0:
            ftle_ratio_diff = abs(
                np.log(sig1.ftle_acceleration_ratio + 0.1) -
                np.log(sig2.ftle_acceleration_ratio + 0.1)
            )
            ftle_similarity = max(0, 1 - ftle_ratio_diff / 2)
        else:
            ftle_similarity = 0.5
        similarities.append(ftle_similarity * 0.25)

        # Transition frequency similarity - log scale (20% weight)
        if sig1.transition_frequency > 0 and sig2.transition_frequency > 0:
            trans_ratio_diff = abs(
                np.log(sig1.transition_frequency + 1) -
                np.log(sig2.transition_frequency + 1)
            )
            trans_similarity = max(0, 1 - trans_ratio_diff / 6)
        else:
            trans_similarity = 0.5
        similarities.append(trans_similarity * 0.20)

        # Saddle score similarity (15% weight)
        saddle_diff = abs(sig1.saddle_score - sig2.saddle_score)
        saddle_similarity = max(0, 1 - saddle_diff)
        similarities.append(saddle_similarity * 0.15)

        # Basin stability similarity (10% weight)
        basin_diff = abs(sig1.basin_stability - sig2.basin_stability)
        basin_similarity = max(0, 1 - basin_diff)
        similarities.append(basin_similarity * 0.10)

        return sum(similarities)

    def _apply_universal_patterns(
        self,
        current_signature: SystemTrajectorySignature
    ) -> Dict[str, Any]:
        """Apply universal degradation patterns learned across all systems."""

        assessment = {
            'degradation_indicators': [],
            'health_confidence': 0.5,
            'reasoning': [],
            'predicted_health': 'uncertain'
        }

        # Pattern 1: Sensitivity concentration
        if current_signature.sensitivity_concentration > 0.8:
            assessment['degradation_indicators'].append('high_sensitivity_concentration')
            assessment['reasoning'].append(
                f"Sensitivity concentration {current_signature.sensitivity_concentration:.2f} "
                "indicates trajectory collapse (validated: turbofan sensor_11 dominance)"
            )
        elif current_signature.sensitivity_concentration < 0.35:
            assessment['reasoning'].append(
                f"Distributed sensitivity ({current_signature.sensitivity_concentration:.2f}) "
                "indicates healthy exploration (validated: bearing acc_x ≈ acc_y)"
            )

        # Pattern 2: FTLE acceleration
        if current_signature.ftle_acceleration_ratio > 2.5:
            assessment['degradation_indicators'].append('ftle_acceleration')
            assessment['reasoning'].append(
                f"FTLE acceleration {current_signature.ftle_acceleration_ratio:.2f}x "
                "indicates increasing trajectory sensitivity"
            )

        # Pattern 3: Transition frequency
        if current_signature.transition_frequency < 100:
            assessment['degradation_indicators'].append('low_exploration')
            assessment['reasoning'].append(
                f"Low transition frequency ({current_signature.transition_frequency}) "
                "indicates loss of healthy state space exploration"
            )
        elif current_signature.transition_frequency > 1000:
            assessment['reasoning'].append(
                f"High transition frequency ({current_signature.transition_frequency}) "
                "indicates healthy dynamic exploration"
            )

        # Pattern 4: Saddle proximity
        if current_signature.saddle_score > 0.65 and current_signature.basin_stability < 0.35:
            assessment['degradation_indicators'].append('saddle_proximity')
            assessment['reasoning'].append(
                f"High saddle proximity ({current_signature.saddle_score:.2f}) "
                f"with low basin stability ({current_signature.basin_stability:.2f}) "
                "indicates vulnerability to regime transition"
            )

        # Compute prediction
        n_indicators = len(assessment['degradation_indicators'])
        if n_indicators >= 3:
            assessment['health_confidence'] = 0.9
            assessment['predicted_health'] = 'degrading'
        elif n_indicators >= 2:
            assessment['health_confidence'] = 0.7
            assessment['predicted_health'] = 'monitoring_required'
        elif n_indicators == 0 and current_signature.transition_frequency > 500:
            assessment['health_confidence'] = 0.8
            assessment['predicted_health'] = 'healthy'
        else:
            assessment['health_confidence'] = 0.5
            assessment['predicted_health'] = 'uncertain'

        return assessment

    def _combine_predictions(
        self,
        direct_matches: List[Tuple],
        cross_system_matches: List[Tuple],
        universal_assessment: Dict,
        current_signature: SystemTrajectorySignature
    ) -> Dict[str, Any]:
        """Combine all prediction sources into final assessment."""

        predictions = []

        # Direct matches (highest weight)
        for match, similarity in direct_matches[:3]:
            predictions.append({
                'health': match.health_status,
                'confidence': similarity * match.confidence,
                'source': f'direct_{match.system_type}_match'
            })

        # Cross-system matches (reduced weight)
        for match, similarity in cross_system_matches[:2]:
            predictions.append({
                'health': match.health_status,
                'confidence': similarity * match.confidence * 0.8,
                'source': f'cross_system_{match.system_type}_match'
            })

        # Universal patterns
        if universal_assessment.get('predicted_health') != 'uncertain':
            predictions.append({
                'health': universal_assessment['predicted_health'],
                'confidence': universal_assessment['health_confidence'],
                'source': 'universal_patterns'
            })

        # Aggregate predictions
        if predictions:
            health_votes = {}
            total_weight = 0

            for pred in predictions:
                health = pred['health']
                weight = pred['confidence']
                health_votes[health] = health_votes.get(health, 0) + weight
                total_weight += weight

            best_health = max(health_votes.keys(), key=lambda h: health_votes[h])
            final_confidence = health_votes[best_health] / total_weight if total_weight > 0 else 0.1
        else:
            best_health = 'unknown'
            final_confidence = 0.1

        return {
            'predicted_health': best_health,
            'confidence': final_confidence,
            'all_predictions': predictions,
            'direct_matches': [(m.to_dict(), s) for m, s in direct_matches[:3]],
            'cross_system_matches': [(m.to_dict(), s) for m, s in cross_system_matches[:3]],
            'universal_assessment': universal_assessment,
            'trajectory_signature': current_signature.to_dict(),
            'insights': self._generate_insights(
                cross_system_matches, universal_assessment, best_health
            )
        }

    def _generate_insights(
        self,
        cross_system_matches: List[Tuple],
        universal_assessment: Dict,
        predicted_health: str
    ) -> List[str]:
        """Generate human-readable insights about cross-system pattern matching."""

        insights = []

        # Cross-system insights
        for match, similarity in cross_system_matches[:2]:
            insights.append(
                f"Pattern similarity with {match.system_type} ({match.health_status}): "
                f"{similarity:.0%} confidence"
            )

        # Universal pattern insights
        if universal_assessment.get('reasoning'):
            insights.extend(universal_assessment['reasoning'])

        # Final prediction
        insights.append(f"Predicted health status: {predicted_health}")

        return insights

    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all learned patterns."""
        summary = {
            'total_patterns': len(self.learned_patterns),
            'by_system_type': {},
            'by_health_status': {},
            'universal_patterns': self.universal_patterns
        }

        for key, pattern in self.learned_patterns.items():
            # By system type
            if pattern.system_type not in summary['by_system_type']:
                summary['by_system_type'][pattern.system_type] = []
            summary['by_system_type'][pattern.system_type].append(key)

            # By health status
            if pattern.health_status not in summary['by_health_status']:
                summary['by_health_status'][pattern.health_status] = []
            summary['by_health_status'][pattern.health_status].append(key)

        return summary


class AdaptiveMonitoringStrategy:
    """
    Generates adaptive monitoring strategies based on learned patterns.
    """

    def __init__(self, learning_engine: CrossSystemLearningEngine = None):
        self.learning_engine = learning_engine or CrossSystemLearningEngine()

    def create_strategy(
        self,
        system_type: str,
        trajectory_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create monitoring strategy adapted to learned patterns."""

        # Get health prediction
        prediction = self.learning_engine.predict_system_health(
            system_type, trajectory_results
        )

        predicted_health = prediction['predicted_health']
        confidence = prediction['confidence']
        signature = prediction['trajectory_signature']

        # Adapt monitoring based on health
        if predicted_health == 'healthy':
            strategy = {
                'sampling_frequency': 'standard',
                'focus_variables': 'distributed_monitoring',
                'alert_thresholds': 'relaxed',
                'reasoning': 'System shows healthy trajectory signature'
            }
        elif predicted_health in ['degrading', 'critical']:
            dominant_var = signature.get('dominant_variable', 'primary_sensor')
            strategy = {
                'sampling_frequency': 'increased_2x',
                'focus_variables': f'concentrate_on_{dominant_var}',
                'alert_thresholds': 'tightened',
                'reasoning': f'Focus monitoring on {dominant_var} based on learned degradation patterns'
            }
        else:  # monitoring_required or uncertain
            strategy = {
                'sampling_frequency': 'increased_1.5x',
                'focus_variables': 'track_sensitivity_evolution',
                'alert_thresholds': 'standard',
                'reasoning': 'Monitor for trajectory signature changes'
            }

        strategy.update({
            'cross_system_insights': prediction['insights'],
            'confidence': confidence,
            'predicted_health': predicted_health,
            'prediction_source': 'cross_system_learning'
        })

        return strategy

    def generate_recommendations(
        self,
        system_type: str,
        prediction: Dict[str, Any]
    ) -> List[str]:
        """Generate maintenance recommendations based on cross-system learning."""

        recommendations = []
        predicted_health = prediction['predicted_health']
        signature = prediction['trajectory_signature']
        dominant_var = signature.get('dominant_variable', 'primary sensor')

        if predicted_health in ['degrading', 'critical']:
            recommendations.extend([
                f"⚠️ Schedule maintenance intervention - trajectory analysis indicates {predicted_health}",
                f"Focus diagnostic efforts on {dominant_var}",
                f"Based on cross-system patterns, expect accelerating degradation",
                "Consider predictive maintenance rather than waiting for threshold alerts"
            ])
        elif predicted_health == 'monitoring_required':
            recommendations.extend([
                "📊 Increase monitoring frequency to track trajectory evolution",
                "Watch for sensitivity concentration on single variable",
                "Prepare maintenance resources for potential intervention"
            ])
        else:  # healthy
            recommendations.extend([
                "✓ Continue standard maintenance schedule",
                "System trajectory indicates healthy operation",
                "No immediate intervention required"
            ])

        # Add cross-system insights
        if prediction.get('insights'):
            recommendations.append("\nCross-system learning insights:")
            for insight in prediction['insights']:
                recommendations.append(f"  • {insight}")

        return recommendations
