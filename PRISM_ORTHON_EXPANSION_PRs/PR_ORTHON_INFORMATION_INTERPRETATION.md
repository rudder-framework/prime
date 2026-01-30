# PR: Information Flow Interpretation for ORTHON

## Summary

Extend ORTHON's interpretation layer to include causal network insights. Translates transfer entropy, Granger causality, and network metrics into actionable maintenance narratives about **who drives whom** in the system.

---

## Motivation

PRISM Information Engine produces:
- Transfer entropy matrices
- Granger causality tests
- Network topology metrics (density, reciprocity, hierarchy)

ORTHON needs to translate these into:
- "Signal X is driving signal Y"
- "Feedback loop forming between A and B"
- "Causal hierarchy breaking down"
- "Control shifting from normal to abnormal path"

---

## New Interpretation Layer

### 1. Information Flow SQL Reports

```sql
-- orthon/sql/50_information_health.sql

-- Report: Causal Network Summary
SELECT
    entity_id,
    observation_idx,
    n_causal_edges,
    network_density,
    network_reciprocity,
    hierarchy_score,
    n_feedback_loops,
    CASE
        WHEN hierarchy_score > 0.8 THEN 'HIERARCHICAL'
        WHEN hierarchy_score > 0.5 THEN 'MIXED'
        WHEN hierarchy_score > 0.2 THEN 'COUPLED'
        ELSE 'CIRCULAR'
    END as network_type,
    CASE
        WHEN n_feedback_loops = 0 THEN 'HEALTHY'
        WHEN n_feedback_loops <= 2 THEN 'WATCH'
        WHEN n_feedback_loops <= 5 THEN 'WARNING'
        ELSE 'CRITICAL'
    END as feedback_risk
FROM information_flow;


-- Report: Causal Network Evolution
WITH lagged AS (
    SELECT
        entity_id,
        observation_idx,
        hierarchy_score,
        n_feedback_loops,
        network_density,
        LAG(hierarchy_score, 5) OVER (PARTITION BY entity_id ORDER BY observation_idx) as hierarchy_prev,
        LAG(n_feedback_loops, 5) OVER (PARTITION BY entity_id ORDER BY observation_idx) as loops_prev,
        LAG(network_density, 5) OVER (PARTITION BY entity_id ORDER BY observation_idx) as density_prev
    FROM information_flow
)
SELECT
    entity_id,
    observation_idx,
    hierarchy_score,
    hierarchy_prev,
    hierarchy_score - hierarchy_prev as hierarchy_change,
    CASE
        WHEN hierarchy_score < hierarchy_prev - 0.1 THEN 'HIERARCHY_BREAKING'
        WHEN hierarchy_score > hierarchy_prev + 0.1 THEN 'HIERARCHY_STRENGTHENING'
        ELSE 'STABLE'
    END as hierarchy_trend,
    n_feedback_loops - loops_prev as feedback_loop_change,
    CASE
        WHEN n_feedback_loops > loops_prev THEN 'FEEDBACK_FORMING'
        WHEN n_feedback_loops < loops_prev THEN 'FEEDBACK_RESOLVING'
        ELSE 'STABLE'
    END as feedback_trend
FROM lagged
WHERE hierarchy_prev IS NOT NULL;


-- Report: Control Shift Detection
SELECT
    entity_id,
    observation_idx,
    top_driver,
    top_sink,
    LAG(top_driver, 10) OVER (PARTITION BY entity_id ORDER BY observation_idx) as prev_driver,
    CASE
        WHEN top_driver != LAG(top_driver, 10) OVER (PARTITION BY entity_id ORDER BY observation_idx)
        THEN 'CONTROL_SHIFTED'
        ELSE 'STABLE_CONTROL'
    END as control_status,
    max_transfer_entropy,
    mean_transfer_entropy
FROM information_flow;


-- Report: Information Flow Anomalies
SELECT
    entity_id,
    observation_idx,
    network_density,
    n_feedback_loops,
    hierarchy_score,
    network_changed,
    CASE
        WHEN network_changed = TRUE THEN '🚨 NETWORK_RECONFIGURED'
        WHEN hierarchy_score < 0.2 THEN '🚨 CAUSAL_COLLAPSE'
        WHEN n_feedback_loops > 5 THEN '⚠️ EXCESSIVE_FEEDBACK'
        WHEN network_density > 0.7 THEN '⚠️ OVER_COUPLED'
        WHEN network_density < 0.1 THEN '⚠️ UNDER_COUPLED'
        ELSE '✓ NORMAL'
    END as causality_alert
FROM information_flow;
```

### 2. Information Flow Interpreter

```python
# orthon/interpreter/information_interpreter.py

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import polars as pl


@dataclass
class InformationInterpretation:
    """Interpretation of causal network state."""
    entity_id: str
    network_type: str
    feedback_risk: str
    hierarchy_trend: str
    control_status: str
    health_score: float
    alerts: List[str]
    narrative: str
    causal_summary: Dict[str, str]
    recommendations: List[str]


class InformationInterpreter:
    """
    Interpret information flow and causal network results.
    """
    
    # Network type descriptions
    NETWORK_TYPES = {
        'HIERARCHICAL': {
            'description': 'Clear causal hierarchy',
            'health': 1.0,
            'meaning': 'Information flows in a clear direction with minimal feedback. '
                      'This is typical of well-controlled systems with proper cause-effect relationships.',
        },
        'MIXED': {
            'description': 'Partial hierarchy with some coupling',
            'health': 0.7,
            'meaning': 'Mostly hierarchical information flow with some bidirectional coupling. '
                      'Normal for complex systems with regulated feedback mechanisms.',
        },
        'COUPLED': {
            'description': 'Significant bidirectional coupling',
            'health': 0.4,
            'meaning': 'Many signals influence each other bidirectionally. '
                      'May indicate developing feedback loops or loss of control hierarchy.',
        },
        'CIRCULAR': {
            'description': 'Circular causal structure',
            'health': 0.2,
            'meaning': 'Information flows in circles with no clear hierarchy. '
                      'Critical condition - system may be approaching runaway feedback.',
        },
    }
    
    def interpret(
        self,
        entity_id: str,
        info_data: pl.DataFrame,
        baseline: Optional[pl.DataFrame] = None
    ) -> InformationInterpretation:
        """
        Interpret information flow metrics for an entity.
        """
        # Get latest values
        latest = info_data.filter(pl.col('entity_id') == entity_id).sort('observation_idx').tail(1)
        
        if len(latest) == 0:
            return self._empty_interpretation(entity_id)
        
        row = latest.row(0, named=True)
        
        # Classify network
        network_type = self._classify_network(row)
        
        # Assess feedback risk
        feedback_risk = self._assess_feedback_risk(row)
        
        # Detect trends
        hierarchy_trend = self._detect_hierarchy_trend(entity_id, info_data)
        control_status = self._detect_control_shift(entity_id, info_data)
        
        # Generate alerts
        alerts = self._generate_alerts(row, hierarchy_trend, control_status)
        
        # Calculate health score
        health_score = self._calculate_health(network_type, feedback_risk, hierarchy_trend)
        
        # Build causal summary
        causal_summary = self._build_causal_summary(row)
        
        # Generate narrative
        narrative = self._generate_narrative(
            entity_id, network_type, feedback_risk, 
            hierarchy_trend, control_status, row
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            network_type, feedback_risk, hierarchy_trend, alerts
        )
        
        return InformationInterpretation(
            entity_id=entity_id,
            network_type=network_type,
            feedback_risk=feedback_risk,
            hierarchy_trend=hierarchy_trend,
            control_status=control_status,
            health_score=health_score,
            alerts=alerts,
            narrative=narrative,
            causal_summary=causal_summary,
            recommendations=recommendations,
        )
    
    def _classify_network(self, row: dict) -> str:
        """Classify network based on hierarchy score."""
        hierarchy = row.get('hierarchy_score', 0.5)
        
        if hierarchy > 0.8:
            return 'HIERARCHICAL'
        elif hierarchy > 0.5:
            return 'MIXED'
        elif hierarchy > 0.2:
            return 'COUPLED'
        else:
            return 'CIRCULAR'
    
    def _assess_feedback_risk(self, row: dict) -> str:
        """Assess feedback loop risk."""
        n_loops = row.get('n_feedback_loops', 0)
        reciprocity = row.get('network_reciprocity', 0)
        
        if n_loops == 0 and reciprocity < 0.1:
            return 'LOW'
        elif n_loops <= 2 and reciprocity < 0.3:
            return 'MODERATE'
        elif n_loops <= 5 and reciprocity < 0.5:
            return 'HIGH'
        else:
            return 'CRITICAL'
    
    def _detect_hierarchy_trend(self, entity_id: str, data: pl.DataFrame) -> str:
        """Detect hierarchy evolution trend."""
        entity_data = data.filter(pl.col('entity_id') == entity_id).sort('observation_idx')
        
        if len(entity_data) < 5:
            return 'INSUFFICIENT_DATA'
        
        hierarchy_values = entity_data['hierarchy_score'].to_numpy()
        recent = hierarchy_values[-5:].mean()
        earlier = hierarchy_values[-10:-5].mean() if len(hierarchy_values) >= 10 else hierarchy_values[:5].mean()
        
        if recent < earlier - 0.1:
            return 'BREAKING_DOWN'
        elif recent > earlier + 0.1:
            return 'STRENGTHENING'
        else:
            return 'STABLE'
    
    def _detect_control_shift(self, entity_id: str, data: pl.DataFrame) -> str:
        """Detect if control has shifted to unexpected signals."""
        entity_data = data.filter(pl.col('entity_id') == entity_id).sort('observation_idx')
        
        if len(entity_data) < 10:
            return 'INSUFFICIENT_DATA'
        
        recent_driver = entity_data.tail(1)['top_driver'][0]
        earlier_drivers = entity_data.head(len(entity_data) - 5)['top_driver'].to_list()
        
        # Check if recent driver was seen in earlier data
        if recent_driver in earlier_drivers:
            return 'STABLE'
        else:
            return 'SHIFTED'
    
    def _generate_alerts(
        self,
        row: dict,
        hierarchy_trend: str,
        control_status: str
    ) -> List[str]:
        """Generate information flow alerts."""
        alerts = []
        
        hierarchy = row.get('hierarchy_score', 0.5)
        n_loops = row.get('n_feedback_loops', 0)
        density = row.get('network_density', 0)
        changed = row.get('network_changed', False)
        
        if changed:
            alerts.append('🚨 CRITICAL: Causal network structure changed significantly')
        
        if hierarchy < 0.2:
            alerts.append('🚨 CRITICAL: Causal hierarchy collapsed - circular causation')
        
        if n_loops > 5:
            alerts.append('⚠️ WARNING: Excessive feedback loops ({} detected)'.format(n_loops))
        
        if density > 0.7:
            alerts.append('⚠️ WARNING: Network over-coupled - everything driving everything')
        
        if hierarchy_trend == 'BREAKING_DOWN':
            alerts.append('⚠️ WATCH: Causal hierarchy breaking down over time')
        
        if control_status == 'SHIFTED':
            alerts.append('⚠️ WATCH: Control has shifted to unexpected signal')
        
        return alerts
    
    def _calculate_health(
        self,
        network_type: str,
        feedback_risk: str,
        hierarchy_trend: str
    ) -> float:
        """Calculate overall information flow health score."""
        base = self.NETWORK_TYPES.get(network_type, {}).get('health', 0.5)
        
        # Adjust for feedback risk
        if feedback_risk == 'CRITICAL':
            base *= 0.5
        elif feedback_risk == 'HIGH':
            base *= 0.7
        elif feedback_risk == 'MODERATE':
            base *= 0.9
        
        # Adjust for trend
        if hierarchy_trend == 'BREAKING_DOWN':
            base *= 0.8
        
        return max(0.0, min(1.0, base))
    
    def _build_causal_summary(self, row: dict) -> Dict[str, str]:
        """Build summary of causal relationships."""
        return {
            'primary_driver': row.get('top_driver', 'Unknown'),
            'primary_sink': row.get('top_sink', 'Unknown'),
            'max_influence': f"{row.get('max_transfer_entropy', 0):.3f} bits",
            'mean_influence': f"{row.get('mean_transfer_entropy', 0):.3f} bits",
            'network_density': f"{row.get('network_density', 0):.1%}",
            'feedback_loops': str(row.get('n_feedback_loops', 0)),
        }
    
    def _generate_narrative(
        self,
        entity_id: str,
        network_type: str,
        feedback_risk: str,
        hierarchy_trend: str,
        control_status: str,
        row: dict
    ) -> str:
        """Generate human-readable narrative."""
        type_info = self.NETWORK_TYPES.get(network_type, {})
        
        narrative = f"""
## Information Flow Analysis: {entity_id}

**Network Classification:** {network_type}
{type_info.get('description', '')}

**Interpretation:**
{type_info.get('meaning', '')}

**Causal Structure:**
- Primary driver: {row.get('top_driver', 'Unknown')}
- Primary sink: {row.get('top_sink', 'Unknown')}
- Network density: {row.get('network_density', 0):.1%}
- Hierarchy score: {row.get('hierarchy_score', 0):.2f}
- Feedback loops: {row.get('n_feedback_loops', 0)}
- Reciprocity: {row.get('network_reciprocity', 0):.1%}

**Risk Assessment:**
- Feedback risk: {feedback_risk}
- Hierarchy trend: {hierarchy_trend}
- Control status: {control_status}

**What this means:**
"""
        
        if network_type == 'CIRCULAR' or feedback_risk == 'CRITICAL':
            narrative += """
The causal structure has broken down into circular dependencies. Information
flows in loops rather than through a clear hierarchy. This is a CRITICAL
condition that often precedes runaway behavior or cascade failures.

In healthy systems, signals follow clear cause-effect chains. When everything
starts driving everything else, the system loses predictability and control.
"""
        elif hierarchy_trend == 'BREAKING_DOWN':
            narrative += """
The causal hierarchy is degrading over time. While still partially intact,
the trend indicates developing feedback loops and loss of clear control
relationships. This is an early warning of potential instability.
"""
        elif control_status == 'SHIFTED':
            narrative += f"""
Control has shifted from normal drivers to unexpected signals. The current
primary driver ({row.get('top_driver', 'Unknown')}) was not dominant in
earlier operation. This may indicate:
- Sensor drift affecting measurements
- Actual change in system dynamics
- Developing fault changing causal relationships
"""
        elif network_type == 'HIERARCHICAL' and feedback_risk == 'LOW':
            narrative += """
The system shows healthy causal structure with clear information flow
hierarchy. Signals influence each other through proper cause-effect chains
with minimal feedback. This is characteristic of well-controlled operation.
"""
        
        return narrative.strip()
    
    def _generate_recommendations(
        self,
        network_type: str,
        feedback_risk: str,
        hierarchy_trend: str,
        alerts: List[str]
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if network_type == 'CIRCULAR' or feedback_risk == 'CRITICAL':
            recommendations.extend([
                'IMMEDIATE: Investigate source of circular causation',
                'CHECK: Control loops, feedback mechanisms, sensor validity',
                'CONSIDER: Reducing operating intensity to break feedback',
                'COMPARE: With baseline causal structure from healthy operation',
            ])
        
        elif hierarchy_trend == 'BREAKING_DOWN':
            recommendations.extend([
                'MONITOR: Track hierarchy score for continued degradation',
                'IDENTIFY: Which signals are developing new causal relationships',
                'CHECK: Control system parameters, feedback gains',
                'CORRELATE: With coherence and Lyapunov metrics',
            ])
        
        elif feedback_risk == 'HIGH':
            recommendations.extend([
                'WATCH: Feedback loops may indicate developing resonance',
                'IDENTIFY: Which signal pairs have bidirectional causation',
                'CHECK: Damping, gain margins, stability criteria',
            ])
        
        else:
            recommendations.append('CONTINUE: Normal monitoring - causal structure healthy')
        
        return recommendations
    
    def _empty_interpretation(self, entity_id: str) -> InformationInterpretation:
        """Return empty interpretation when no data available."""
        return InformationInterpretation(
            entity_id=entity_id,
            network_type='UNKNOWN',
            feedback_risk='UNKNOWN',
            hierarchy_trend='NO_DATA',
            control_status='NO_DATA',
            health_score=0.5,
            alerts=['No information flow data available'],
            narrative=f'No information flow data available for {entity_id}',
            causal_summary={},
            recommendations=['Ensure PRISM information engine has been run'],
        )
```

### 3. Story Templates

```python
# orthon/stories/information_stories.py

INFORMATION_STORIES = {
    'CAUSAL_COLLAPSE': """
🚨 CRITICAL CAUSAL ALERT: {entity_id}

The causal hierarchy has collapsed (hierarchy score: {hierarchy_score:.2f}).

WHAT THIS MEANS:
Information is flowing in circles rather than through clear cause-effect
chains. The system has lost directional causality - everything is driving
everything else.

MECHANICAL INTERPRETATION:
- Feedback loops have overwhelmed normal control
- Resonance conditions creating mutual reinforcement
- Control system no longer effective
- System approaching runaway or cascade failure

DETECTED FEEDBACK LOOPS: {n_feedback_loops}
Network reciprocity: {network_reciprocity:.1%}

IMMEDIATE ACTIONS:
1. Identify the strongest feedback loop
2. Consider reducing gain/intensity to break feedback
3. Check control system health and parameters
4. Prepare for potential emergency shutdown
""",

    'FEEDBACK_FORMING': """
⚠️ CAUSAL WARNING: {entity_id}

New feedback loops detected: {n_feedback_loops} (was {n_feedback_loops_prev})

CURRENT FEEDBACK PAIRS:
{feedback_pairs}

WHAT THIS MEANS:
Signals that previously had one-way causal relationships are now
influencing each other bidirectionally. This creates potential for:
- Resonance amplification
- Oscillatory instability
- Loss of control authority

POSSIBLE CAUSES:
- Developing mechanical coupling
- Wear changing dynamic response
- Control degradation
- Sensor cross-talk

RECOMMENDED ACTIONS:
1. Identify physical mechanism creating feedback
2. Check for developing resonance conditions
3. Monitor for oscillation growth
4. Consider damping or isolation interventions
""",

    'CONTROL_SHIFT': """
⚠️ CAUSAL WATCH: {entity_id}

Control has shifted to unexpected signal.

Previous primary driver: {prev_driver}
Current primary driver: {current_driver}

Transfer entropy: {max_transfer_entropy:.3f} bits

WHAT THIS MEANS:
The signal with the most causal influence has changed. This may indicate:
- Sensor drift making wrong signal appear dominant
- Actual physical change in system dynamics
- Developing fault taking over system behavior
- Control system degradation

RECOMMENDED ACTIONS:
1. Verify sensor health for {current_driver}
2. Check if {current_driver} has physical reason to dominate
3. Compare with coherence analysis for corroboration
4. Review control system logs for anomalies
""",

    'HEALTHY_CAUSAL': """
✓ CAUSAL STRUCTURE HEALTHY: {entity_id}

Network type: {network_type}
Hierarchy score: {hierarchy_score:.2f}
Feedback loops: {n_feedback_loops}
Primary driver: {top_driver}
Primary sink: {top_sink}

The causal structure shows:
- Clear information flow hierarchy
- Minimal feedback coupling
- Stable control relationships

This is characteristic of well-functioning systems where
cause-effect relationships are clear and controllable.
""",

    'CASCADE_RISK': """
🚨 CASCADE RISK ALERT: {entity_id}

Network density has reached {network_density:.1%} (threshold: 70%)

WHAT THIS MEANS:
Almost every signal is causally connected to almost every other signal.
This "fully coupled" state means:
- Small disturbances propagate everywhere
- No isolation between subsystems
- High risk of cascade failure

IN HEALTHY SYSTEMS:
Signals should be grouped into subsystems with limited cross-talk.
High density indicates loss of normal isolation.

MECHANICAL INTERPRETATION:
- Structural connections propagating vibration
- Fluid coupling between components
- Electrical cross-talk
- Loss of damping/isolation

IMMEDIATE ACTIONS:
1. Identify which connections are new/abnormal
2. Check isolation and damping systems
3. Consider reducing operating intensity
4. Monitor for cascade propagation
""",
}


def generate_information_story(
    entity_id: str,
    current: dict,
    previous: dict = None,
    story_type: str = 'auto'
) -> str:
    """
    Generate appropriate causal story based on current state.
    """
    if story_type == 'auto':
        # Determine story type from data
        if current.get('hierarchy_score', 1) < 0.2:
            story_type = 'CAUSAL_COLLAPSE'
        elif current.get('network_density', 0) > 0.7:
            story_type = 'CASCADE_RISK'
        elif previous and current.get('n_feedback_loops', 0) > previous.get('n_feedback_loops', 0):
            story_type = 'FEEDBACK_FORMING'
        elif previous and current.get('top_driver') != previous.get('top_driver'):
            story_type = 'CONTROL_SHIFT'
        else:
            story_type = 'HEALTHY_CAUSAL'
    
    template = INFORMATION_STORIES.get(story_type, INFORMATION_STORIES['HEALTHY_CAUSAL'])
    
    # Prepare template variables
    variables = {
        'entity_id': entity_id,
        **current
    }
    
    if previous:
        variables.update({
            f'{k}_prev': v for k, v in previous.items()
        })
        variables['prev_driver'] = previous.get('top_driver', 'Unknown')
        variables['current_driver'] = current.get('top_driver', 'Unknown')
    
    return template.format(**variables)
```

---

## Integration with Story Engine

### Unified Health Dashboard

```python
# orthon/stories/unified_story.py

def generate_unified_story(
    entity_id: str,
    geometry_interp: 'GeometryInterpretation',
    dynamics_interp: 'DynamicsInterpretation',
    topology_interp: 'TopologyInterpretation',
    information_interp: 'InformationInterpretation'
) -> str:
    """
    Generate unified health narrative from all analysis layers.
    """
    
    # Collect health scores
    scores = {
        'geometry': geometry_interp.health_score,
        'dynamics': dynamics_interp.health_score,
        'topology': topology_interp.health_score,
        'information': information_interp.health_score,
    }
    
    overall_health = sum(scores.values()) / len(scores)
    
    # Identify weakest layer
    weakest = min(scores, key=scores.get)
    
    # Collect all alerts
    all_alerts = (
        geometry_interp.alerts +
        dynamics_interp.alerts +
        topology_interp.alerts +
        information_interp.alerts
    )
    
    critical_alerts = [a for a in all_alerts if '🚨' in a]
    warning_alerts = [a for a in all_alerts if '⚠️' in a]
    
    story = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    UNIFIED HEALTH ASSESSMENT: {entity_id:^20}                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERALL HEALTH SCORE: {overall_health:.0%}

┌─────────────────┬──────────────┬─────────────────────────────────────────────┐
│ Analysis Layer  │ Health Score │ Status                                      │
├─────────────────┼──────────────┼─────────────────────────────────────────────┤
│ Geometry        │ {scores['geometry']:>10.0%}  │ {geometry_interp.structure_class:<43} │
│ Dynamics        │ {scores['dynamics']:>10.0%}  │ {dynamics_interp.stability_class:<43} │
│ Topology        │ {scores['topology']:>10.0%}  │ {topology_interp.topology_class:<43} │
│ Information     │ {scores['information']:>10.0%}  │ {information_interp.network_type:<43} │
└─────────────────┴──────────────┴─────────────────────────────────────────────┘

PRIMARY CONCERN: {weakest.upper()} layer showing weakest health

"""
    
    if critical_alerts:
        story += "🚨 CRITICAL ALERTS:\n"
        for alert in critical_alerts:
            story += f"   {alert}\n"
        story += "\n"
    
    if warning_alerts:
        story += "⚠️ WARNINGS:\n"
        for alert in warning_alerts:
            story += f"   {alert}\n"
        story += "\n"
    
    # Cross-layer correlation
    story += """
CROSS-LAYER ANALYSIS:
"""
    
    # Check for corroborating evidence
    if (geometry_interp.health_score < 0.5 and 
        dynamics_interp.health_score < 0.5):
        story += """
✗ Geometry AND Dynamics both degraded - STRONG evidence of real problem
  Coherence changes correlate with stability changes.
"""
    
    if (topology_interp.health_score < 0.5 and 
        information_interp.health_score < 0.5):
        story += """
✗ Topology AND Information flow both degraded - causal-structural breakdown
  Attractor shape changes correlate with causal network changes.
"""
    
    if (geometry_interp.health_score > 0.7 and 
        dynamics_interp.health_score > 0.7 and
        topology_interp.health_score > 0.7 and
        information_interp.health_score > 0.7):
        story += """
✓ All layers healthy - no cross-layer concerns
  System showing normal operation across all analysis dimensions.
"""
    
    # Consolidated recommendations
    all_recommendations = (
        geometry_interp.recommendations +
        dynamics_interp.recommendations +
        topology_interp.recommendations +
        information_interp.recommendations
    )
    
    # Deduplicate and prioritize
    immediate = [r for r in all_recommendations if 'IMMEDIATE' in r]
    investigate = [r for r in all_recommendations if 'INVESTIGATE' in r or 'CHECK' in r]
    monitor = [r for r in all_recommendations if 'MONITOR' in r or 'WATCH' in r]
    
    story += """
PRIORITIZED RECOMMENDATIONS:
"""
    
    if immediate:
        story += "\n  IMMEDIATE ACTIONS:\n"
        for r in immediate[:3]:
            story += f"    • {r}\n"
    
    if investigate:
        story += "\n  INVESTIGATE:\n"
        for r in investigate[:3]:
            story += f"    • {r}\n"
    
    if monitor:
        story += "\n  MONITOR:\n"
        for r in monitor[:3]:
            story += f"    • {r}\n"
    
    return story
```

---

## Output Files

### New SQL Reports
```
orthon/sql/50_information_health.sql
orthon/sql/51_causal_evolution.sql
orthon/sql/52_feedback_detection.sql
orthon/sql/53_control_shift.sql
```

### New Interpreter
```
orthon/interpreter/information_interpreter.py
```

### New Stories
```
orthon/stories/information_stories.py
orthon/stories/unified_story.py
```

---

## Success Criteria

1. **Alerts correlate with failures** (feedback loops → shorter RUL)
2. **Control shift detection** precedes failure in >50% of cases
3. **Unified health score** correlates with RUL (r > 0.3)
4. **Stories are actionable** (maintenance engineers can act on recommendations)

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 1 | SQL reports | 2 days |
| 2 | Interpreter | 3 days |
| 3 | Story templates | 2 days |
| 4 | Unified dashboard | 2 days |
| 5 | Validation | 3 days |

---

## The Complete Stack

With all PRs implemented:

```
Raw Signals
    │
    ├── PRISM Geometry → Coherence, effective dimension
    ├── PRISM Dynamics → Lyapunov, attractor dimension, RQA
    ├── PRISM Topology → Betti numbers, persistence
    └── PRISM Information → Transfer entropy, causal network
    │
    ▼
ORTHON Interpretation
    │
    ├── Geometry Interpreter → "Modes coupling/decoupling"
    ├── Dynamics Interpreter → "Stable/chaotic, basin depth"
    ├── Topology Interpreter → "Attractor shape, fragmentation"
    └── Information Interpreter → "Causal hierarchy, feedback loops"
    │
    ▼
Unified Story Engine
    │
    └── "Entity X shows degraded topology (collapsing loops) with
         causal hierarchy breakdown (feedback forming). Dynamics
         show approaching chaos (λ trending positive). Geometry
         confirms mode coupling. Cross-layer evidence strongly
         suggests developing failure. RECOMMEND: Immediate inspection
         of coupling mechanism between signals 7 and 12."
```

**The complete picture. Not just THAT failure is coming, but WHY.**
