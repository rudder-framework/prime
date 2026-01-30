# PR: Topology Interpretation for ORTHON

## Summary

Extend ORTHON's interpretation layer to include topological data analysis insights. Translates Betti numbers, persistence diagrams, and topological complexity into actionable maintenance narratives.

---

## Motivation

PRISM Topology Engine produces:
- Betti numbers (β₀, β₁, β₂)
- Persistence statistics
- Topological complexity

ORTHON needs to translate these into:
- Human-readable health assessments
- Failure mode classification
- Actionable recommendations

---

## Topology Classification

| Class | β₀ | β₁ | Description | Health |
|-------|----|----|-------------|--------|
| HEALTHY_CYCLE | 1 | 1 | Clean limit cycle | 1.0 |
| QUASI_PERIODIC | 1 | 2 | Multi-frequency (torus) | 0.8 |
| COMPLEX | 1 | >2 | Multiple interacting loops | 0.5 |
| COLLAPSED | 1 | 0 | No periodic structure | 0.3 |
| FRAGMENTED | >1 | any | Disconnected dynamics | 0.1 |

---

## SQL Reports

```sql
-- orthon/sql/40_topology_health.sql

SELECT
    entity_id,
    observation_idx,
    betti_0,
    betti_1,
    betti_2,
    CASE
        WHEN betti_0 > 1 THEN 'FRAGMENTED'
        WHEN betti_1 = 0 THEN 'COLLAPSED'
        WHEN betti_1 = 1 THEN 'HEALTHY_CYCLE'
        WHEN betti_1 = 2 THEN 'QUASI_PERIODIC'
        WHEN betti_1 > 2 THEN 'COMPLEX'
        ELSE 'UNKNOWN'
    END as topology_class,
    topological_complexity
FROM topology;

-- Topology Evolution
WITH lagged AS (
    SELECT
        entity_id,
        observation_idx,
        betti_1,
        LAG(betti_1, 5) OVER (PARTITION BY entity_id ORDER BY observation_idx) as betti_1_prev
    FROM topology
)
SELECT
    entity_id,
    observation_idx,
    betti_1,
    betti_1_prev,
    CASE
        WHEN betti_1 > betti_1_prev THEN 'LOOPS_FORMING'
        WHEN betti_1 < betti_1_prev THEN 'LOOPS_COLLAPSING'
        ELSE 'STABLE'
    END as loop_trend
FROM lagged
WHERE betti_1_prev IS NOT NULL;
```

---

## Interpreter

```python
# orthon/interpreter/topology_interpreter.py

from dataclasses import dataclass
from typing import List
import polars as pl


@dataclass
class TopologyInterpretation:
    entity_id: str
    topology_class: str
    complexity_level: str
    trend: str
    health_score: float
    alerts: List[str]
    narrative: str
    recommendations: List[str]


class TopologyInterpreter:
    
    TOPOLOGY_CLASSES = {
        'HEALTHY_CYCLE': {
            'description': 'Clean limit cycle dynamics',
            'health': 1.0,
            'meaning': 'System exhibits regular periodic behavior.',
        },
        'QUASI_PERIODIC': {
            'description': 'Multi-frequency dynamics',
            'health': 0.8,
            'meaning': 'Torus-like attractor with multiple frequencies.',
        },
        'COMPLEX': {
            'description': 'Complex attractor structure',
            'health': 0.5,
            'meaning': 'Multiple interacting loops - may indicate early degradation.',
        },
        'COLLAPSED': {
            'description': 'No loop structure',
            'health': 0.3,
            'meaning': 'Lost periodic structure - overdamping or failure.',
        },
        'FRAGMENTED': {
            'description': 'Disconnected dynamics',
            'health': 0.1,
            'meaning': 'Attractor broken into pieces - CRITICAL.',
        },
    }
    
    def interpret(self, entity_id: str, topology_data: pl.DataFrame) -> TopologyInterpretation:
        latest = topology_data.filter(pl.col('entity_id') == entity_id).sort('observation_idx').tail(1)
        
        if len(latest) == 0:
            return self._empty_interpretation(entity_id)
        
        row = latest.row(0, named=True)
        
        topology_class = self._classify_topology(row)
        trend = self._detect_trend(entity_id, topology_data)
        alerts = self._generate_alerts(row, trend)
        health_score = self._calculate_health(topology_class, trend)
        narrative = self._generate_narrative(entity_id, topology_class, trend, row)
        recommendations = self._generate_recommendations(topology_class, trend)
        
        return TopologyInterpretation(
            entity_id=entity_id,
            topology_class=topology_class,
            complexity_level=self._assess_complexity(row, topology_data),
            trend=trend,
            health_score=health_score,
            alerts=alerts,
            narrative=narrative,
            recommendations=recommendations,
        )
    
    def _classify_topology(self, row: dict) -> str:
        b0 = row.get('betti_0', 1)
        b1 = row.get('betti_1', 0)
        
        if b0 > 1:
            return 'FRAGMENTED'
        elif b1 == 0:
            return 'COLLAPSED'
        elif b1 == 1:
            return 'HEALTHY_CYCLE'
        elif b1 == 2:
            return 'QUASI_PERIODIC'
        else:
            return 'COMPLEX'
    
    def _detect_trend(self, entity_id: str, data: pl.DataFrame) -> str:
        entity_data = data.filter(pl.col('entity_id') == entity_id).sort('observation_idx')
        
        if len(entity_data) < 5:
            return 'INSUFFICIENT_DATA'
        
        b1_values = entity_data['betti_1'].to_numpy()
        recent = b1_values[-5:].mean()
        earlier = b1_values[-10:-5].mean() if len(b1_values) >= 10 else b1_values[:5].mean()
        
        if recent > earlier + 0.5:
            return 'LOOPS_INCREASING'
        elif recent < earlier - 0.5:
            return 'LOOPS_DECREASING'
        else:
            return 'STABLE'
    
    def _generate_alerts(self, row: dict, trend: str) -> List[str]:
        alerts = []
        
        if row.get('betti_0', 1) > 1:
            alerts.append('🚨 CRITICAL: Attractor fragmented')
        if row.get('betti_1', 0) == 0 and trend == 'LOOPS_DECREASING':
            alerts.append('🚨 CRITICAL: Loop structure collapsed')
        if row.get('betti_1', 0) > 3:
            alerts.append('⚠️ WARNING: Excessive loop structure')
        if trend == 'LOOPS_DECREASING':
            alerts.append('⚠️ WATCH: Loop structure simplifying')
        
        return alerts
    
    def _calculate_health(self, topology_class: str, trend: str) -> float:
        base = self.TOPOLOGY_CLASSES.get(topology_class, {}).get('health', 0.5)
        if trend == 'LOOPS_DECREASING':
            base *= 0.8
        return max(0.0, min(1.0, base))
    
    def _assess_complexity(self, row: dict, all_data: pl.DataFrame) -> str:
        complexity = row.get('topological_complexity', 0)
        mean_complexity = all_data['topological_complexity'].mean()
        
        if complexity > mean_complexity * 2:
            return 'HIGH'
        elif complexity < mean_complexity * 0.5:
            return 'LOW'
        return 'NORMAL'
    
    def _generate_narrative(self, entity_id: str, topology_class: str, trend: str, row: dict) -> str:
        class_info = self.TOPOLOGY_CLASSES.get(topology_class, {})
        
        return f"""
## Topology Analysis: {entity_id}

**Classification:** {topology_class}
{class_info.get('description', '')}

**Current State:**
- Connected components (β₀): {row.get('betti_0', 'N/A')}
- Loop structures (β₁): {row.get('betti_1', 'N/A')}
- Void structures (β₂): {row.get('betti_2', 'N/A')}
- Complexity: {row.get('topological_complexity', 0):.3f}

**Trend:** {trend}

**Interpretation:**
{class_info.get('meaning', '')}
""".strip()
    
    def _generate_recommendations(self, topology_class: str, trend: str) -> List[str]:
        recommendations = []
        
        if topology_class == 'FRAGMENTED':
            recommendations.extend([
                'IMMEDIATE: Schedule inspection - critical topology breakdown',
                'CHECK: Mechanical integrity, structural connections',
            ])
        elif topology_class == 'COLLAPSED':
            recommendations.extend([
                'INVESTIGATE: Loss of periodic structure',
                'CHECK: Damping, lubrication, mechanical freedom',
            ])
        elif trend == 'LOOPS_DECREASING':
            recommendations.extend([
                'WATCH: Decreasing complexity - early warning',
                'SCHEDULE: Preventive inspection',
            ])
        else:
            recommendations.append('CONTINUE: Normal monitoring')
        
        return recommendations
    
    def _empty_interpretation(self, entity_id: str) -> TopologyInterpretation:
        return TopologyInterpretation(
            entity_id=entity_id,
            topology_class='UNKNOWN',
            complexity_level='UNKNOWN',
            trend='NO_DATA',
            health_score=0.5,
            alerts=['No topology data available'],
            narrative=f'No topology data for {entity_id}',
            recommendations=['Run PRISM topology engine'],
        )
```

---

## Story Templates

```python
TOPOLOGY_STORIES = {
    'FRAGMENTED_CRITICAL': """
🚨 CRITICAL TOPOLOGY ALERT: {entity_id}

The attractor has fragmented into {betti_0} disconnected components.

MECHANICAL INTERPRETATION:
- Severe wear creating discontinuous behavior
- Structural damage affecting dynamic response
- Imminent catastrophic failure

IMMEDIATE ACTIONS:
1. Reduce load if possible
2. Schedule emergency inspection
3. Prepare for potential shutdown
""",

    'LOOP_COLLAPSE': """
⚠️ TOPOLOGY WARNING: {entity_id}

Loop structure collapsed: β₁ = {betti_1}

The system has lost periodic structure. Possible causes:
- Excessive damping
- Mechanical binding
- Loss of excitation

RECOMMENDED: Check damping, lubrication, mechanical freedom
""",

    'HEALTHY_STABLE': """
✓ TOPOLOGY HEALTHY: {entity_id}

Classification: {topology_class}
β₀ = {betti_0}, β₁ = {betti_1}
Trend: STABLE

System shows healthy periodic structure.
""",
}
```

---

## Integration

### ML Features

Add to `ml_features_dense.parquet`:
- `betti_0`, `betti_1`, `betti_2`
- `h1_max_persistence`
- `topological_complexity`
- `topology_class` (encoded)

### Unified Health

Topology health feeds into unified health score:
```python
unified_health = (
    0.25 * geometry_health +
    0.25 * dynamics_health +
    0.25 * topology_health +
    0.25 * information_health
)
```

---

## Validation

1. **Synthetic tests:** Circle (β₁=1), torus (β₁=2), noise (β₁=0)
2. **CWRU bearings:** Healthy vs faulty attractor shape
3. **C-MAPSS:** Topology changes correlate with RUL

---

## Files to Create

```
orthon/sql/40_topology_health.sql
orthon/sql/41_topology_evolution.sql
orthon/interpreter/topology_interpreter.py
orthon/stories/topology_stories.py
```

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| SQL reports | 2 days |
| Interpreter | 3 days |
| Stories | 2 days |
| Validation | 3 days |

---

## Summary

Topology interpretation answers: **"What is the shape of system behavior?"**

- Healthy systems: Clean loops (β₁ = 1 or 2)
- Failing systems: Fragmented (β₀ > 1) or collapsed (β₁ = 0)
- Trend matters: Decreasing loops = early warning
