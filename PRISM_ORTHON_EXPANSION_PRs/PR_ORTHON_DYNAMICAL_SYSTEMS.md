# PR: Dynamical Systems Extension for ORTHON

## Summary

Extend ORTHON's physics interpretation layer to include dynamical systems metrics that capture stability, chaos, and regime transitions. This complements existing geometric coherence analysis with phase space methods.

---

## Motivation

### Current State
ORTHON captures **static** geometric properties:
- Coherence (coupling strength at a point)
- Effective dimension (mode structure at a point)
- Velocity (first derivative of state)
- Dissipation (energy loss rate)

### Gap
We don't capture **dynamical** properties:
- How fast do trajectories diverge? (Lyapunov)
- How complex is the attractor? (dimension)
- Is the system near a regime change? (bifurcation)
- How stable is the current operating state? (basin stability)

### Why It Matters
The "birth certificate" finding (early-life fingerprints predict lifespan) suggests we're detecting **basin depth** - how stable the system's attractor is from the start. Dynamical systems metrics make this explicit.

---

## Proposed Architecture

```
physics.parquet (from PRISM)
        │
        ├── Current: eigenvalue metrics
        │   ├── coherence
        │   ├── effective_dim
        │   ├── dissipation
        │   └── state_distance
        │
        └── NEW: dynamics metrics (from PRISM dynamics engine)
            ├── lyapunov_max
            ├── lyapunov_spectrum
            ├── attractor_dim
            ├── recurrence_rate
            ├── determinism
            └── regime_flag
```

---

## New Interpretation Layer

### 1. Stability Classification

```sql
-- dynamics.sql (new ORTHON report)

SELECT 
    entity_id,
    lyapunov_max,
    CASE 
        WHEN lyapunov_max > 0.1 THEN 'CHAOTIC'
        WHEN lyapunov_max > 0 THEN 'WEAKLY_UNSTABLE'
        WHEN lyapunov_max > -0.1 THEN 'MARGINAL'
        ELSE 'STABLE'
    END as stability_class,
    attractor_dim,
    CASE
        WHEN attractor_dim > n_signals * 0.8 THEN 'HIGH_COMPLEXITY'
        WHEN attractor_dim > n_signals * 0.5 THEN 'MODERATE_COMPLEXITY'
        ELSE 'LOW_COMPLEXITY'
    END as complexity_class
FROM dynamics
```

### 2. Regime Change Detection

```sql
-- regime_transitions.sql

SELECT
    entity_id,
    observation_idx,
    lyapunov_max,
    lyapunov_max - LAG(lyapunov_max, 20) OVER w as lyapunov_trend,
    CASE
        WHEN lyapunov_max - LAG(lyapunov_max, 50) OVER w > 0.2 THEN 'DESTABILIZING'
        WHEN lyapunov_max - LAG(lyapunov_max, 50) OVER w < -0.2 THEN 'STABILIZING'
        ELSE 'STEADY'
    END as regime_trend
FROM dynamics
WINDOW w AS (PARTITION BY entity_id ORDER BY observation_idx)
```

### 3. Basin Stability Score

```sql
-- basin_stability.sql

SELECT
    entity_id,
    -- Lower Lyapunov = deeper basin = more stable
    -- Higher recurrence = more predictable = more stable
    (1 - GREATEST(0, lyapunov_max)) * 0.5 +
    recurrence_rate * 0.3 +
    determinism * 0.2 as basin_stability_score
FROM dynamics
```

---

## New ML Features

Add to `ml_features_dense.parquet`:

| Feature | Description | Leakage Risk |
|---------|-------------|--------------|
| `lyapunov_max` | Maximum Lyapunov exponent | LOW (local) |
| `lyapunov_trend` | Rate of Lyapunov change | LOW (local) |
| `attractor_dim` | Correlation dimension | LOW (local) |
| `recurrence_rate` | % of recurrent points | MEDIUM |
| `determinism` | RQA determinism | MEDIUM |
| `basin_stability` | Composite stability score | LOW |
| `regime_flag` | 0/1 regime transition detected | LOW |

Note: These can be computed from **truncated** trajectories - no leakage if windowed properly.

---

## Story Engine Integration

### New Story Templates

```python
STABILITY_STORIES = {
    'DESTABILIZING': """
        {entity_id} shows increasing dynamical instability.
        Lyapunov exponent trending positive ({lyapunov_trend:+.3f}/cycle).
        System approaching chaotic regime.
        Basin stability: {basin_stability:.0%}
        
        INTERPRETATION: The system's operating attractor is becoming 
        shallower. Small perturbations will have larger effects.
        
        WATCH FOR: Increased vibration variability, mode jumping,
        intermittent behavior.
    """,
    
    'REGIME_TRANSITION': """
        {entity_id} has undergone a regime transition at cycle {transition_cycle}.
        
        Before: {stability_before} (λ = {lyapunov_before:.3f})
        After:  {stability_after} (λ = {lyapunov_after:.3f})
        
        INTERPRETATION: The system has shifted to a new dynamical state.
        This may indicate wear progression, damage accumulation, or
        operational change.
    """,
    
    'BIRTH_CERTIFICATE': """
        Early-life stability analysis for {entity_id}:
        
        Basin stability score: {basin_stability:.0%}
        Lyapunov (first 20%): {early_lyapunov:.3f}
        Attractor complexity: {complexity_class}
        
        PROGNOSIS: {prognosis}
        
        Engines with similar early profiles had average lifespan: {similar_lifespan} cycles
    """
}
```

---

## Validation Plan

### 1. CWRU Bearings (Vibration Timescale)
- Compute Lyapunov from raw 12kHz data
- Compare healthy vs faulty bearings
- Expected: Faulty bearings show positive Lyapunov

### 2. C-MAPSS Turbofans (Operational Timescale)
- Compute Lyapunov from sensor trajectories
- Correlate early-life Lyapunov with RUL
- Expected: Higher early Lyapunov → shorter life

### 3. Cross-Domain Fingerprints
- Do Lyapunov patterns match coherence patterns?
- Bearings: Coherence↓ should correlate with Lyapunov↑
- Turbofans: Coherence↑ should correlate with... what?

---

## Dependencies

Requires PRISM to implement:
- `lyapunov_spectrum()` - compute Lyapunov exponents
- `correlation_dimension()` - estimate attractor dimension
- `recurrence_quantification()` - RQA metrics

See: `PR_PRISM_DYNAMICS_ENGINE.md`

---

## Files to Create/Modify

### New Files
```
orthon/sql/30_dynamics_stability.sql
orthon/sql/31_regime_transitions.sql
orthon/sql/32_basin_stability.sql
orthon/sql/33_birth_certificate.sql
orthon/interpreter/dynamics_interpreter.py
orthon/stories/stability_stories.py
```

### Modified Files
```
orthon/sql/26_ml_feature_export.sql  -- add dynamics features
orthon/interpreter/physics_interpreter.py  -- integrate dynamics
orthon/explorer/index.html  -- new Dynamics tab
```

---

## Success Criteria

1. **Lyapunov correlates with RUL** (r > 0.3, p < 0.01)
2. **Regime transitions precede failures** (>80% of failures have prior transition)
3. **Basin stability improves birth certificate** (early prediction accuracy +10%)
4. **Cross-domain consistency** (Lyapunov patterns match across CWRU, C-MAPSS)

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 1 | PRISM dynamics engine | 1-2 weeks |
| 2 | ORTHON interpretation layer | 1 week |
| 3 | Validation on CWRU + C-MAPSS | 1 week |
| 4 | Story engine integration | 3 days |
| 5 | Documentation + examples | 2 days |

---

## References

- Kantz & Schreiber, "Nonlinear Time Series Analysis"
- Rosenstein et al., "A practical method for calculating largest Lyapunov exponents"
- Marwan et al., "Recurrence plots for the analysis of complex systems"
- The "birth certificate" finding from C-MAPSS early-life analysis

---

## Notes

This extension positions ORTHON as a **stability monitoring** platform, not just a geometry analyzer. The story becomes:

> "Failure is loss of dynamical stability. We measure how stable your system is."

This is deeper than "coherence is dropping" - it's "the attractor basin is shallowing, you're approaching a tipping point."
