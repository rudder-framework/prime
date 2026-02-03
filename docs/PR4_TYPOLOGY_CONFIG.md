# PR #4: Config-Driven Typology Classification

**Branch:** `feature/typology-config`
**Target:** `orthon/` (ORTHON repo)
**Date:** 2026-02-02
**Status:** Ready for review

---

## Summary

Refactors all typology classification thresholds into a centralized configuration file. Eliminates magic numbers from classification logic, making threshold tuning easier and domain adaptation possible without code changes.

**Key insight from 100-engine C-MAPSS review:** The 5-engine sample had artificially clean trends (hurst=1.0). The full 100-engine dataset has noisier degradation (hurst=0.60-0.99), requiring more nuanced thresholds. A centralized config makes this tuning tractable.

---

## Files

```
orthon/
├── config/
│   ├── __init__.py              # Package exports
│   └── typology_config.py       # TYPOLOGY_CONFIG dict + helpers
├── typology/
│   ├── level2_corrections.py    # Refactored to use config
│   └── tests/
│       └── test_config_corrections.py  # 28 tests
```

**Test results:** 28/28 passing (0.12s)

---

## Configuration Structure

```python
TYPOLOGY_CONFIG = {
    'artifacts': {
        'first_bin_tolerance': 0.01,
        'first_bin_slope_threshold': -0.3,
        'default_fft_size': 256,
    },
    
    'periodic': {
        'spectral_flatness_max': 0.7,
        'snr_min': 6.0,
        'hurst_max': 0.95,
        'turning_point_ratio_max': 0.95,
    },
    
    'temporal': {
        'trending': {
            'hurst_strong': 0.99,
            'hurst_moderate': 0.85,
            'acf_ratio_min': 0.10,
            'sample_entropy_max': 0.15,
        },
        'random': {...},
        'chaotic': {...},
        'quasi_periodic': {...},
        'constant': {...},
    },
    
    'spectral': {...},
    'stationarity': {...},
    'windowing': {...},
    'engines': {...},
    'visualizations': {...},
}
```

---

## Key Changes from PR3

### 1. Gate Ordering: TRENDING Before PERIODIC

PR3 checked PERIODIC first, then TRENDING. This caused signals with spectral structure *and* high persistence to be misclassified as PERIODIC.

**New order:**
```
1. hurst >= 0.99 → TRENDING (immediate)
2. hurst > 0.85 + long ACF + low entropy → TRENDING
3. is_genuine_periodic() → PERIODIC
4. ... rest of tree
```

This ensures degradation trends with harmonic components are caught before the PERIODIC gate.

### 2. Relative ACF Threshold

PR3 required `acf_half_life = NaN` for TRENDING. This failed on C-MAPSS where ACF eventually decays (16% of series length).

**Config solution:**
```python
'acf_ratio_min': 0.10  # acf_half_life / n_samples
```

Now TRENDING catches signals where ACF spans >10% of the series, even if not infinite.

### 3. Sample Entropy Relaxation

PR3 used `sample_entropy < 0.02`. Real noisy degradation has entropy 0.04-0.15.

**Config solution:**
```python
'sample_entropy_max': 0.15
```

Tunable without code changes.

---

## Helper Functions

```python
from orthon.config import get_threshold, get_engine_adjustments

# Access any threshold by dot-path
hurst = get_threshold('temporal.trending.hurst_strong')  # 0.99

# Get engine adjustments for a pattern
adj = get_engine_adjustments('trending')
# {'add': ['hurst', 'trend_r2', ...], 'remove': ['harmonics_ratio', ...]}
```

---

## Domain Adaptation Example

To tune for a noisier industrial dataset:

```python
from orthon.config import TYPOLOGY_CONFIG

# Relax TRENDING thresholds
TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong'] = 0.95
TYPOLOGY_CONFIG['temporal']['trending']['sample_entropy_max'] = 0.30

# Tighten CHAOTIC to reduce false positives
TYPOLOGY_CONFIG['temporal']['chaotic']['min_samples'] = 1000
```

All classification functions automatically use the updated values.

---

## Integration with Existing Code

The config is designed to slot into the existing pipeline:

```python
# Option 1: Use apply_corrections() as before
from orthon.typology.level2_corrections import apply_corrections
corrected_row = apply_corrections(row)

# Option 2: Use individual functions with custom config
from orthon.config import TYPOLOGY_CONFIG
TYPOLOGY_CONFIG['temporal']['trending']['hurst_strong'] = 0.95

from orthon.typology.level2_corrections import classify_temporal_pattern
pattern = classify_temporal_pattern(row)  # Uses updated threshold
```

---

## Test Coverage

| Test Class | Tests | Coverage |
|------------|-------|----------|
| TestConfigAccess | 4 | Config retrieval, defaults, validation |
| TestArtifactDetection | 4 | First-bin detection, edge cases |
| TestTemporalPattern | 7 | All temporal gates |
| TestSpectralClassification | 5 | All spectral classes |
| TestEngineCorrections | 3 | Add/remove logic |
| TestIntegration | 4 | CSTR, C-MAPSS, Bearing cross-validation |
| TestConfigOverride | 1 | Runtime config changes |
| **Total** | **28** | **All passing** |

---

## Relationship to Other PRs

| PR | Relationship |
|----|--------------|
| PR #1 (Alphabet) | Config defines thresholds for the 10-dimension alphabet |
| PR #2 (Manifest) | Config provides engine/viz adjustments that manifest uses |
| PR #3 (Corrections) | **Superseded** — PR4 refactors PR3 into config-driven form |
| PR #5 (PRISM) | Independent — PRISM reads manifest, not config |

**Merge order:** PR #4 can replace PR #3 entirely. Merge PR #1 → PR #4 → PR #2 → PR #5.

---

## Remaining Work

1. **CONSTANT class:** Config has `constant` section but classification doesn't fully implement it yet. Need variance check in row.

2. **Per-domain profiles:** Could add preset configs for different domains:
   ```python
   PROFILES = {
       'industrial': {...},  # Relaxed thresholds
       'biomedical': {...},  # Strict CHAOTIC detection
       'financial': {...},   # Different volatility emphasis
   }
   ```

3. **Config validation:** `validate_config()` exists but could be expanded to catch more inconsistencies.
