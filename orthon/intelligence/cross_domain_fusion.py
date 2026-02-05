"""
ORTHON Cross-Domain Fusion

Intelligently combines geometric (PRISM) and physical constraint analysis
for superior system health assessment.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HealthCategory(Enum):
    """System health categories."""
    CRITICAL = "critical"
    WARNING = "warning"
    DEGRADED = "degraded"
    NORMAL = "normal"
    OPTIMAL = "optimal"


@dataclass
class FusedHealthAssessment:
    """Combined health assessment from geometric and physical domains."""
    overall_score: float  # 0-1, where 1 is perfect
    category: HealthCategory
    geometric_contribution: float
    physical_contribution: float
    confidence: float
    dominant_domain: str
    risk_factors: List[str]
    recommendations: List[str]


class CrossDomainFusion:
    """
    Fuses geometric stability analysis (PRISM) with physical constraint monitoring.

    The two domains provide complementary views:
    - Geometric: Captures system coherence, signal coupling, dimensional collapse
    - Physical: Validates conservation laws, thermodynamic efficiency, mass balance

    When both domains agree on health status, confidence is high.
    When domains disagree, deeper investigation is warranted.
    """

    def __init__(self, system_type: str = "generic"):
        self.system_type = system_type

        # Domain weights (can be learned or configured)
        self.geometric_weight = 0.5
        self.physical_weight = 0.5

        # Fusion history for trend analysis
        self.fusion_history: List[Dict] = []

        # Domain-specific thresholds
        self.thresholds = self._initialize_thresholds()

    def _initialize_thresholds(self) -> Dict:
        """Initialize domain-specific thresholds."""

        return {
            'geometric': {
                'effective_dimension': {
                    'critical': 1.5,
                    'warning': 2.0,
                    'normal': 3.0,
                },
                'eigenvalue_ratio': {
                    'critical': 10.0,
                    'warning': 5.0,
                    'normal': 3.0,
                },
            },
            'physical': {
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
                'efficiency': {
                    'critical': 0.5,
                    'warning': 0.7,
                    'normal': 0.85,
                },
            },
        }

    def fuse_domains(
        self,
        geometric_state: Dict,
        physical_state: Dict,
        context: Optional[Dict] = None
    ) -> FusedHealthAssessment:
        """
        Fuse geometric and physical analysis into unified assessment.

        Args:
            geometric_state: PRISM analysis results
            physical_state: Physics constraint analysis
            context: Additional context (operating mode, recent events, etc.)

        Returns:
            FusedHealthAssessment with combined health score and recommendations
        """

        context = context or {}

        # Analyze each domain
        geom_score, geom_issues = self._analyze_geometric_domain(geometric_state)
        phys_score, phys_issues = self._analyze_physical_domain(physical_state)

        # Check for domain agreement/disagreement
        agreement_level = self._calculate_domain_agreement(geom_score, phys_score)

        # Determine fusion strategy based on agreement
        if agreement_level > 0.8:
            # High agreement - simple weighted average
            overall_score = (
                self.geometric_weight * geom_score +
                self.physical_weight * phys_score
            )
            confidence = 0.9
        elif agreement_level > 0.5:
            # Moderate agreement - weighted average with reduced confidence
            overall_score = (
                self.geometric_weight * geom_score +
                self.physical_weight * phys_score
            )
            confidence = 0.7
        else:
            # Low agreement - investigate further
            # Use conservative (lower) score but flag the disagreement
            overall_score = min(geom_score, phys_score)
            confidence = 0.5

        # Determine health category
        category = self._score_to_category(overall_score)

        # Identify risk factors
        risk_factors = geom_issues + phys_issues
        if agreement_level < 0.5:
            risk_factors.append("Domain disagreement detected - manual review recommended")

        # Generate recommendations
        recommendations = self._generate_recommendations(
            overall_score, category, geom_issues, phys_issues, agreement_level
        )

        # Determine dominant domain
        if abs(geom_score - phys_score) < 0.1:
            dominant_domain = "balanced"
        elif geom_score < phys_score:
            dominant_domain = "geometric"
        else:
            dominant_domain = "physical"

        assessment = FusedHealthAssessment(
            overall_score=overall_score,
            category=category,
            geometric_contribution=geom_score,
            physical_contribution=phys_score,
            confidence=confidence,
            dominant_domain=dominant_domain,
            risk_factors=risk_factors,
            recommendations=recommendations,
        )

        # Store in history
        self._record_fusion(assessment, geometric_state, physical_state)

        return assessment

    def _analyze_geometric_domain(self, state: Dict) -> Tuple[float, List[str]]:
        """Analyze geometric domain and return score and issues."""

        score = 1.0
        issues = []

        thresholds = self.thresholds['geometric']

        # Effective dimension analysis
        eff_dim = state.get('effective_dimension', state.get('eff_dim'))
        if eff_dim is not None:
            eff_thresholds = thresholds['effective_dimension']

            if eff_dim < eff_thresholds['critical']:
                score -= 0.5
                issues.append(f"Critical dimensional collapse: {eff_dim:.2f}")
            elif eff_dim < eff_thresholds['warning']:
                score -= 0.25
                issues.append(f"Dimensional degradation: {eff_dim:.2f}")
            elif eff_dim < eff_thresholds['normal']:
                score -= 0.1
                issues.append(f"Below normal dimension: {eff_dim:.2f}")

        # Eigenvalue ratio analysis
        eigenval_ratio = state.get('eigenvalue_ratio')
        if eigenval_ratio is not None:
            ratio_thresholds = thresholds['eigenvalue_ratio']

            if eigenval_ratio > ratio_thresholds['critical']:
                score -= 0.3
                issues.append(f"Extreme eigenvalue concentration: ratio {eigenval_ratio:.2f}")
            elif eigenval_ratio > ratio_thresholds['warning']:
                score -= 0.15
                issues.append(f"High eigenvalue concentration: ratio {eigenval_ratio:.2f}")

        # Trend analysis
        dimensional_trend = state.get('dimensional_trend')
        if dimensional_trend is not None and dimensional_trend < -0.05:
            score -= 0.1
            issues.append(f"Negative dimensional trend: {dimensional_trend:.3f}/sample")

        return max(0, min(1, score)), issues

    def _analyze_physical_domain(self, state: Dict) -> Tuple[float, List[str]]:
        """Analyze physical domain and return score and issues."""

        score = 1.0
        issues = []

        thresholds = self.thresholds['physical']

        # Energy balance analysis
        energy_error = state.get('energy_balance_error')
        if energy_error is not None:
            energy_thresholds = thresholds['energy_balance_error']

            if energy_error > energy_thresholds['critical']:
                score -= 0.4
                issues.append(f"Critical energy imbalance: {energy_error:.1%}")
            elif energy_error > energy_thresholds['warning']:
                score -= 0.2
                issues.append(f"Energy imbalance: {energy_error:.1%}")
            elif energy_error > energy_thresholds['normal']:
                score -= 0.1

        # Mass balance analysis
        mass_error = state.get('mass_balance_error')
        if mass_error is not None:
            mass_thresholds = thresholds['mass_balance_error']

            if mass_error > mass_thresholds['critical']:
                score -= 0.4
                issues.append(f"Critical mass imbalance: {mass_error:.1%}")
            elif mass_error > mass_thresholds['warning']:
                score -= 0.2
                issues.append(f"Mass imbalance: {mass_error:.1%}")

        # Efficiency analysis
        efficiency = state.get('system_efficiency', state.get('efficiency'))
        if efficiency is not None:
            eff_thresholds = thresholds['efficiency']

            if efficiency < eff_thresholds['critical']:
                score -= 0.3
                issues.append(f"Critical efficiency loss: {efficiency:.1%}")
            elif efficiency < eff_thresholds['warning']:
                score -= 0.15
                issues.append(f"Reduced efficiency: {efficiency:.1%}")

        # Constraint violations
        n_violations = state.get('constraint_violations', 0)
        if n_violations > 0:
            score -= min(0.3, n_violations * 0.1)
            issues.append(f"{n_violations} physics constraint violation(s)")

        return max(0, min(1, score)), issues

    def _calculate_domain_agreement(self, geom_score: float, phys_score: float) -> float:
        """Calculate how much the two domains agree."""

        # Agreement is 1 when scores are identical, decreases with difference
        difference = abs(geom_score - phys_score)
        agreement = 1.0 - difference

        return agreement

    def _score_to_category(self, score: float) -> HealthCategory:
        """Convert health score to category."""

        if score < 0.3:
            return HealthCategory.CRITICAL
        elif score < 0.5:
            return HealthCategory.WARNING
        elif score < 0.7:
            return HealthCategory.DEGRADED
        elif score < 0.9:
            return HealthCategory.NORMAL
        else:
            return HealthCategory.OPTIMAL

    def _generate_recommendations(
        self,
        score: float,
        category: HealthCategory,
        geom_issues: List[str],
        phys_issues: List[str],
        agreement: float
    ) -> List[str]:
        """Generate actionable recommendations."""

        recommendations = []

        if category == HealthCategory.CRITICAL:
            recommendations.append("IMMEDIATE: Initiate emergency response protocol")
            recommendations.append("Prepare for controlled shutdown if condition persists")

        elif category == HealthCategory.WARNING:
            recommendations.append("ALERT: System requires attention")
            recommendations.append("Increase monitoring frequency")

            if geom_issues and not phys_issues:
                recommendations.append("Focus on signal coupling and sensor calibration")
            elif phys_issues and not geom_issues:
                recommendations.append("Check physical process parameters and equipment")

        elif category == HealthCategory.DEGRADED:
            recommendations.append("Monitor system closely")
            recommendations.append("Schedule maintenance review")

        if agreement < 0.5:
            recommendations.append("Domain disagreement: Manual inspection recommended")
            if len(geom_issues) > len(phys_issues):
                recommendations.append("Geometric issues dominate - check sensor array")
            else:
                recommendations.append("Physical issues dominate - check process equipment")

        if not recommendations:
            recommendations.append("System operating normally")

        return recommendations

    def _record_fusion(
        self,
        assessment: FusedHealthAssessment,
        geometric_state: Dict,
        physical_state: Dict
    ):
        """Record fusion result for trend analysis."""

        import time
        record = {
            'timestamp': time.time(),
            'overall_score': assessment.overall_score,
            'category': assessment.category.value,
            'geometric_score': assessment.geometric_contribution,
            'physical_score': assessment.physical_contribution,
            'confidence': assessment.confidence,
            'agreement': self._calculate_domain_agreement(
                assessment.geometric_contribution,
                assessment.physical_contribution
            ),
        }

        self.fusion_history.append(record)

        # Keep bounded
        if len(self.fusion_history) > 1000:
            self.fusion_history = self.fusion_history[-500:]

    def get_trend_analysis(self, window_minutes: int = 30) -> Dict:
        """Analyze trends in fused health over time window."""

        if not self.fusion_history:
            return {'status': 'insufficient_data'}

        import time
        cutoff = time.time() - window_minutes * 60

        recent = [r for r in self.fusion_history if r['timestamp'] > cutoff]

        if len(recent) < 5:
            return {'status': 'insufficient_data'}

        scores = [r['overall_score'] for r in recent]
        agreements = [r['agreement'] for r in recent]

        return {
            'status': 'ok',
            'window_minutes': window_minutes,
            'n_samples': len(recent),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'trend': float(np.polyfit(range(len(scores)), scores, 1)[0]),  # Slope
            'mean_agreement': float(np.mean(agreements)),
            'score_volatility': float(np.std(np.diff(scores))),
        }

    def set_domain_weights(self, geometric_weight: float, physical_weight: float):
        """Set domain weights for fusion."""

        total = geometric_weight + physical_weight
        self.geometric_weight = geometric_weight / total
        self.physical_weight = physical_weight / total

    def update_thresholds(self, domain: str, metric: str, thresholds: Dict[str, float]):
        """Update thresholds for a specific metric."""

        if domain in self.thresholds and metric in self.thresholds[domain]:
            self.thresholds[domain][metric].update(thresholds)
