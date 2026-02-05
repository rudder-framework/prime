"""
ORTHON AI UI Optimizer

Learns user behavior patterns and automatically optimizes dashboard layout,
alert sensitivity, and metric prioritization for each user and system combination.
"""

import numpy as np
import time
from collections import defaultdict, Counter
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class UserProfile:
    """User behavior profile for personalization."""
    user_id: str
    role: str
    expertise_level: float = 0.5  # 0=novice, 1=expert
    preferred_layout: str = "default"
    optimal_alert_sensitivity: float = 0.7
    information_density_preference: float = 0.5
    avg_response_time: float = 60.0  # seconds
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


class AIUIOptimizer:
    """
    Learns optimal UI layout and behavior for each user.

    Tracks:
    - Which metrics users actually look at during incidents
    - How quickly users respond to different alert types
    - What dashboard layouts lead to fastest problem resolution
    - User expertise level and preferred information density
    """

    def __init__(self, user_id: str, role: str = "operator"):
        self.user_id = user_id
        self.role = role

        # User profile
        self.profile = UserProfile(user_id=user_id, role=role)

        # Interaction tracking
        self.interaction_history: List[Dict] = []
        self.attention_heatmap: Dict[str, int] = defaultdict(int)
        self.alert_response_times: Dict[str, List[float]] = defaultdict(list)

        # Incident analysis
        self.incident_interactions: List[Dict] = []

        # Layout learning
        self.layout_effectiveness: Dict[str, float] = {}

    def track_user_interaction(self, interaction: Dict):
        """
        Track user interaction for learning.

        Args:
            interaction: Dict with keys:
                - type: 'click', 'hover', 'scroll', 'alert_response'
                - element: Which metric/chart interacted with
                - context: System state when interaction occurred
                - duration: How long user spent on element
        """

        record = {
            'timestamp': time.time(),
            'interaction_type': interaction.get('type', 'unknown'),
            'target_element': interaction.get('element', 'unknown'),
            'context': interaction.get('context', {}),
            'duration': interaction.get('duration', 0),
        }

        self.interaction_history.append(record)

        # Keep history bounded
        if len(self.interaction_history) > 10000:
            self.interaction_history = self.interaction_history[-5000:]

        # Update attention heatmap
        if interaction.get('type') in ['click', 'hover']:
            element = interaction.get('element', 'unknown')
            self.attention_heatmap[element] += 1

            # Weight by duration
            if interaction.get('duration', 0) > 5:
                self.attention_heatmap[element] += interaction['duration'] // 10

        # Track during incidents
        health_score = interaction.get('context', {}).get('system_health_score', 1.0)
        if health_score < 0.7:
            self.incident_interactions.append(record)

        # Track alert responses
        if interaction.get('type') == 'alert_response':
            alert_type = interaction.get('context', {}).get('alert_type', 'unknown')
            response_time = interaction.get('context', {}).get('response_time', 60)
            self.alert_response_times[alert_type].append(response_time)

        self.profile.updated_at = time.time()

    def generate_personalized_dashboard(
        self,
        current_system_state: Dict,
        available_metrics: List[str]
    ) -> Dict[str, Any]:
        """
        Generate optimally personalized dashboard layout for this user.
        """

        # Analyze user attention patterns
        priority_metrics = self._identify_priority_metrics(available_metrics)

        # Determine optimal layout based on user expertise
        layout_config = self._generate_layout_config(priority_metrics)

        # Optimize alert sensitivity for this user
        alert_config = self._optimize_alert_sensitivity()

        # Generate chart configurations
        chart_config = self._generate_chart_config(priority_metrics)

        dashboard_spec = {
            'user_id': self.user_id,
            'role': self.role,
            'generated_at': time.time(),
            'layout': layout_config,
            'alerts': alert_config,
            'charts': chart_config,
            'metrics_priority': priority_metrics,
            'personalization_confidence': self._calculate_personalization_confidence(),
            'expertise_level': self.profile.expertise_level,
        }

        return dashboard_spec

    def _identify_priority_metrics(self, available_metrics: List[str]) -> List[Dict]:
        """Identify which metrics this user pays attention to during incidents."""

        # Count attention during incidents
        incident_attention = Counter()
        for interaction in self.incident_interactions:
            if interaction['interaction_type'] in ['click', 'hover']:
                element = interaction['target_element']
                incident_attention[element] += 1

        # Also consider overall attention
        for element, count in self.attention_heatmap.items():
            if element in available_metrics:
                incident_attention[element] += count // 10  # Lower weight

        # Priority ranking
        priority_metrics = []
        for metric in available_metrics:
            attention_count = incident_attention.get(metric, 0) + self.attention_heatmap.get(metric, 0)

            priority_metrics.append({
                'metric': metric,
                'attention_score': attention_count,
                'priority_level': self._calculate_priority_level(attention_count),
                'display_prominence': self._calculate_display_prominence(attention_count),
            })

        # Sort by attention
        priority_metrics.sort(key=lambda x: x['attention_score'], reverse=True)

        return priority_metrics

    def _calculate_priority_level(self, attention_count: int) -> str:
        """Calculate priority level from attention count."""
        if attention_count > 50:
            return 'critical'
        elif attention_count > 20:
            return 'high'
        elif attention_count > 5:
            return 'medium'
        else:
            return 'low'

    def _calculate_display_prominence(self, attention_count: int) -> str:
        """Calculate display size from attention count."""
        if attention_count > 50:
            return 'large'
        elif attention_count > 20:
            return 'medium'
        else:
            return 'small'

    def _generate_layout_config(self, priority_metrics: List[Dict]) -> Dict:
        """Generate optimal layout configuration based on user expertise."""

        expertise = self.profile.expertise_level

        if expertise > 0.8:  # Expert user
            layout = {
                'style': 'high_density',
                'primary_panel_metrics': 8,
                'chart_complexity': 'advanced',
                'alert_verbosity': 'minimal',
                'update_frequency': 'high',
                'show_raw_values': True,
                'show_explanations': False,
            }
        elif expertise < 0.3:  # Novice user
            layout = {
                'style': 'guided',
                'primary_panel_metrics': 4,
                'chart_complexity': 'basic',
                'alert_verbosity': 'explanatory',
                'update_frequency': 'moderate',
                'show_raw_values': False,
                'show_explanations': True,
            }
        else:  # Intermediate user
            layout = {
                'style': 'balanced',
                'primary_panel_metrics': 6,
                'chart_complexity': 'intermediate',
                'alert_verbosity': 'concise',
                'update_frequency': 'high',
                'show_raw_values': True,
                'show_explanations': False,
            }

        # Customize metric placement based on attention patterns
        layout['metric_placement'] = {}
        n_primary = layout['primary_panel_metrics']

        for i, metric_info in enumerate(priority_metrics[:n_primary]):
            layout['metric_placement'][metric_info['metric']] = {
                'position': i,
                'size': metric_info['display_prominence'],
                'update_priority': 'high' if i < 3 else 'normal',
            }

        return layout

    def _optimize_alert_sensitivity(self) -> Dict:
        """Optimize alert thresholds based on user response patterns."""

        # Calculate average response times
        avg_response_times = {}
        for alert_type, response_times in self.alert_response_times.items():
            if response_times:
                avg_response_times[alert_type] = np.mean(response_times)

        # Base sensitivity by role
        role_sensitivity = {
            'operator': 0.8,
            'engineer': 0.6,
            'supervisor': 0.7,
            'analyst': 0.5,
        }
        base_sensitivity = role_sensitivity.get(self.role, 0.7)

        # Adjust based on demonstrated response speed
        if avg_response_times:
            overall_response_speed = np.mean(list(avg_response_times.values()))
            self.profile.avg_response_time = overall_response_speed

            if overall_response_speed < 30:  # Fast responder
                sensitivity_multiplier = 1.2
            elif overall_response_speed > 120:  # Slow responder
                sensitivity_multiplier = 0.8
            else:
                sensitivity_multiplier = 1.0
        else:
            sensitivity_multiplier = 1.0

        final_sensitivity = min(1.0, base_sensitivity * sensitivity_multiplier)
        self.profile.optimal_alert_sensitivity = final_sensitivity

        return {
            'base_sensitivity': base_sensitivity,
            'user_adjustment': sensitivity_multiplier,
            'final_sensitivity': final_sensitivity,
            'alert_types': {
                'critical': final_sensitivity * 1.0,
                'warning': final_sensitivity * 0.8,
                'info': final_sensitivity * 0.5,
            },
            'response_time_threshold': {
                'critical': 30,  # seconds
                'warning': 60,
                'info': 120,
            },
        }

    def _generate_chart_config(self, priority_metrics: List[Dict]) -> Dict:
        """Generate chart configurations based on user preferences."""

        expertise = self.profile.expertise_level

        if expertise > 0.7:
            chart_type_preference = 'advanced'
            default_timerange = 3600  # 1 hour
            show_confidence_bands = True
        elif expertise < 0.3:
            chart_type_preference = 'simple'
            default_timerange = 600  # 10 minutes
            show_confidence_bands = False
        else:
            chart_type_preference = 'standard'
            default_timerange = 1800  # 30 minutes
            show_confidence_bands = True

        charts = {}
        for metric_info in priority_metrics[:6]:
            metric = metric_info['metric']
            charts[metric] = {
                'type': 'line',
                'timerange': default_timerange,
                'show_trend': True,
                'show_threshold': True,
                'show_confidence': show_confidence_bands,
                'update_rate': 'realtime' if metric_info['priority_level'] in ['critical', 'high'] else 'batch',
            }

        return charts

    def update_expertise_level(self, interaction_outcomes: List[Dict]):
        """Update user expertise level based on interaction outcomes."""

        expertise_indicators = []

        for outcome in interaction_outcomes:
            outcome_type = outcome.get('outcome_type', '')

            if outcome_type == 'alert_response':
                response_time = outcome.get('response_time', 60)
                was_true_alert = outcome.get('was_true_alert', True)
                resolution_success = outcome.get('resolution_success', False)

                # Fast response to true alerts = expertise
                if was_true_alert and response_time < 30:
                    expertise_indicators.append(0.8)
                elif was_true_alert and response_time < 60:
                    expertise_indicators.append(0.6)
                elif was_true_alert and response_time > 120:
                    expertise_indicators.append(0.2)

                # False positive response = novice
                if not was_true_alert:
                    expertise_indicators.append(0.1)

                # Successful resolution = expertise
                if resolution_success:
                    expertise_indicators.append(0.9)

            elif outcome_type == 'config_change':
                # Users who change advanced settings are experts
                if outcome.get('setting_complexity', 'basic') == 'advanced':
                    expertise_indicators.append(0.8)

        # Update expertise with exponential smoothing
        if expertise_indicators:
            new_expertise_signal = np.mean(expertise_indicators)
            self.profile.expertise_level = (
                0.9 * self.profile.expertise_level +
                0.1 * new_expertise_signal
            )

        # Clamp to [0, 1]
        self.profile.expertise_level = np.clip(self.profile.expertise_level, 0, 1)

    def _calculate_personalization_confidence(self) -> float:
        """Calculate confidence in personalization."""

        # Confidence from interaction history
        history_factor = min(len(self.interaction_history) / 100, 1.0) * 0.4

        # Confidence from incident experience
        incident_factor = min(len(self.incident_interactions) / 20, 1.0) * 0.3

        # Confidence from alert response data
        response_factor = min(sum(len(v) for v in self.alert_response_times.values()) / 30, 1.0) * 0.3

        return history_factor + incident_factor + response_factor

    def get_user_profile_summary(self) -> Dict[str, Any]:
        """Get summary of user profile."""

        return {
            'user_id': self.user_id,
            'role': self.role,
            'expertise_level': self.profile.expertise_level,
            'expertise_category': (
                'expert' if self.profile.expertise_level > 0.7 else
                'intermediate' if self.profile.expertise_level > 0.3 else
                'novice'
            ),
            'preferred_layout': self.profile.preferred_layout,
            'avg_response_time': self.profile.avg_response_time,
            'total_interactions': len(self.interaction_history),
            'incident_interactions': len(self.incident_interactions),
            'top_focus_metrics': [
                k for k, v in sorted(
                    self.attention_heatmap.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
            ],
            'personalization_confidence': self._calculate_personalization_confidence(),
        }

    def export_profile(self) -> Dict:
        """Export user profile for persistence."""

        return {
            'profile': {
                'user_id': self.profile.user_id,
                'role': self.profile.role,
                'expertise_level': self.profile.expertise_level,
                'preferred_layout': self.profile.preferred_layout,
                'optimal_alert_sensitivity': self.profile.optimal_alert_sensitivity,
                'avg_response_time': self.profile.avg_response_time,
            },
            'attention_heatmap': dict(self.attention_heatmap),
            'alert_response_times': {k: list(v) for k, v in self.alert_response_times.items()},
        }

    def import_profile(self, data: Dict):
        """Import user profile from persistence."""

        if 'profile' in data:
            p = data['profile']
            self.profile.expertise_level = p.get('expertise_level', 0.5)
            self.profile.preferred_layout = p.get('preferred_layout', 'default')
            self.profile.optimal_alert_sensitivity = p.get('optimal_alert_sensitivity', 0.7)
            self.profile.avg_response_time = p.get('avg_response_time', 60.0)

        if 'attention_heatmap' in data:
            self.attention_heatmap = defaultdict(int, data['attention_heatmap'])

        if 'alert_response_times' in data:
            self.alert_response_times = defaultdict(list, {
                k: list(v) for k, v in data['alert_response_times'].items()
            })
