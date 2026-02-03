# PR #3: Typology Classification Corrections

**Branch:** `fix/typology-classification-corrections`
**Target:** `orthon/typology/` (ORTHON repo)
**Date:** 2025-02-02
**Status:** Ready for review

---

## Summary

Fixes 5 critical misclassifications discovered during CSTR chemical reactor dataset review. Six of seven signals were classified as PERIODIC when they are actually monotonic trends (exponential decay, accumulation, coupled kinetics). The temperature signal (correctly RANDOM) was the only accurate classification.

**Root cause:** The FFT peak detector picks up the lowest FFT bin (1/N_fft = 1/256 = 0.00390625) as "dominant frequency" on any slow-evolving signal with a 1/f spectral slope. This false peak cascades through the entire classification chain: PERIODIC → HARMONIC → harmonic engines → spectral visualizations.

**Impact:** Before corrections, 4 unique typology cards out of 7 signals (poor discrimination). After corrections, TRENDING vs RANDOM clearly differentiated with appropriate engine selection and visualizations per class.

---

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `corrections/level2_corrections.py` | 472 | Spectral artifact detection, periodicity validation, temporal/spectral reclassification, engine/viz corrections |
| `corrections/level1_corrections.py` | 219 | Deterministic trend detection, stationarity override for bounded decay |
| `corrections/manifest_corrections.py` | 48 | Global stride default fix |
| `corrections/tests/test_corrections.py` | 447 | 35 tests covering all fixes |
| `corrections/__init__.py` | 0 | Package marker |
| `corrections/tests/__init__.py` | 0 | Package marker |

**Test results:** 35/35 passing (0.32s)

---

## Fix Details

### Fix 1: Spectral Peak Artifact Detection

**Function:** `is_first_bin_artifact()`

**Problem:** All 7 CSTR signals reported `dominant_frequency = 0.00390625 = 1/256` (first FFT bin).

**Why it happens:** Slow-evolving signals with 1/f spectral slope concentrate energy at the lowest frequencies. The FFT peak detector finds the first bin (frequency = 1/N_fft) as the "dominant" frequency — but this is just the lowest resolvable frequency, not a real spectral peak.

**Detection logic:**
1. Check if `dominant_freq ≈ 1/fft_size` (within 1% tolerance)
2. AND `spectral_slope < -0.3` (energy concentrated at low frequencies)

When both conditions are true, the frequency is an artifact and should be treated as `None`.

---

### Fix 2: Genuine Periodicity Validation

**Function:** `is_genuine_periodic()`

**Problem:** 6/7 signals classified as PERIODIC when they are monotonic trends.

**Six gates a signal must pass to be called PERIODIC:**

| Gate | Check | conc_A (fails) | sine wave (passes) |
|------|-------|-----------------|---------------------|
| 1 | Real spectral peak (not first-bin artifact) | ✗ freq = 1/256, slope = -0.57 | ✓ genuine peak |
| 2 | Not broadband (flatness < 0.7) | ✓ 0.30 | ✓ 0.01 |
| 3 | SNR > 6 dB | ✗ 4.2 dB | ✓ 30+ dB |
| 4 | ACF shows oscillation (half_life exists) | ✗ ACF never decays | ✓ oscillates |
| 5 | Hurst < 0.95 | ✗ hurst = 1.0 | ✓ 0.5 |
| 6 | turning_point_ratio < 0.95 | ✗ 0.959 | ✓ ~0.5 |

Chemical signals fail 4+ gates. A genuine periodic signal like a sine wave passes all 6.

---

### Fix 3: Corrected Temporal Pattern Classification

**Function:** `classify_temporal_pattern()`

**New decision tree:**

```
1. is_genuine_periodic()              → PERIODIC
2. hurst > 0.85 AND ACF never decays  → TRENDING
3. sample_entropy < 0.02 AND hurst > 0.9 → TRENDING
4. spectral_flatness > 0.9 AND perm_entropy > 0.99 → RANDOM
5. lyapunov_proxy > 0.5 AND perm_entropy > 0.95 → CHAOTIC
6. turning_point_ratio < 0.7          → QUASI_PERIODIC
7. default                            → STATIONARY
```

**CSTR results:**

| Signal | Before | After | Why |
|--------|--------|-------|-----|
| conc_A | PERIODIC | TRENDING | hurst=1.0, ACF never decays, exponential decay |
| conc_B | PERIODIC | TRENDING | hurst=0.998, coupled kinetics |
| conc_C | PERIODIC | TRENDING | hurst=0.992, accumulation |
| rate_A | PERIODIC | TRENDING | hurst=1.0, reaction rate |
| rate_B | PERIODIC | TRENDING | hurst=0.999, coupled rate |
| rate_C | PERIODIC | TRENDING | hurst=0.992, production rate |
| temperature | RANDOM | RANDOM | ✓ unchanged (was correct) |

---

### Fix 4: Corrected Spectral Classification

**Function:** `classify_spectral()`

**Problem:** HARMONIC label cascaded from false PERIODIC.

**New decision tree:**

```
1. temporal_pattern == PERIODIC AND HNR > 3.0  → HARMONIC
2. spectral_flatness > 0.8                     → BROADBAND
3. spectral_slope < -0.5                       → RED_NOISE
4. spectral_slope > 0.2                        → BLUE_NOISE
5. default                                     → NARROWBAND
```

**CSTR results:**

| Signal | Before | After | Why |
|--------|--------|-------|-----|
| conc_A..rate_C | HARMONIC | RED_NOISE | 1/f spectral slope (-0.57 to -0.74) |
| temperature | BROADBAND | BROADBAND | ✓ unchanged (flatness=0.985) |

---

### Fix 5: Engine & Visualization Corrections

**Functions:** `correct_engines()`, `correct_visualizations()`

**For TRENDING signals:**
- **Remove:** harmonics_ratio, band_ratios, thd, frequency_bands, waterfall, recurrence
- **Add:** hurst, rate_of_change_ratio, trend_r2, detrend_std, trend_overlay, segment_comparison

**For RED_NOISE spectral class:**
- **Add:** psd_slope (shows the 1/f characteristic)

---

### Fix 6: Level 1 Stationarity Override

**Function:** `detect_deterministic_trend()`, `correct_stationarity()`

**Problem:** conc_A classified STATIONARY despite monotonic decay from 1.0 → 0.01.

**Why this happens:**
- ADF p=0.00 (rejects unit root — technically correct: bounded exponential decay is not a random walk)
- KPSS p=0.01 (rejects stationarity — correct: mean is changing)
- ADF+KPSS disagree → classified as DIFFERENCE_STATIONARY, but the ADF "pass" is misleading

**Detection: Segment-mean divergence test**
1. Split signal into 4 equal segments
2. Compute mean of each segment
3. Check for monotonic progression (ascending or descending)
4. Compute strength = max_segment_diff / overall_std

**conc_A segments:** [0.72, 0.27, 0.07, 0.02] → monotonic descending, strength >> 2.0

**Override logic:**
When ADF passes but KPSS rejects AND any of:
- `mean_shift_ratio > 1.0`
- `is_deterministic_trend` with strength > 2.0
- `variance_ratio < 0.01` or `> 100`

→ Override to `TREND_STATIONARY`

---

### Fix 7: Manifest Global Stride Default

**Function:** `compute_global_stride_default()`

**Problem:** Global `params.stride` hardcoded to 1, creating 99.6% overlap and ~500k engine calls for 5000 samples.

**Solution:** Use median of per-signal recommended strides, or 50% of default_window as fallback. Per-signal strides are already correctly computed by `recommend_stride()`.

---

## Integration

These corrections are designed to slot in after the existing `compute_raw_measures()` call:

```python
# Existing pipeline
row = compute_raw_measures(signal)

# NEW: Apply corrections
from corrections.level2_corrections import apply_corrections
from corrections.level1_corrections import apply_level1_corrections

row = apply_level1_corrections(row, signal_values)  # stationarity override
row = apply_corrections(row)                         # Level 2 fixes
```

Or use `apply_corrections()` alone if Level 1 override isn't needed — the temporal/spectral fixes are independent.

**No changes to existing code** — these are additive correction functions that post-process the existing row dict.

---

## Test Coverage

| Test Class | Tests | What it validates |
|------------|-------|-------------------|
| TestFirstBinArtifact | 5 | CSTR artifact detection, real peaks preserved, None/NaN handling |
| TestCorrectedDominantFreq | 3 | Chemical→None, broadband→None, periodic preserved |
| TestGenuinePeriodic | 3 | conc_A fails, temperature fails, sine passes |
| TestTemporalPattern | 4 | conc_A→TRENDING, conc_B→TRENDING, temp→RANDOM, sine→PERIODIC |
| TestSpectralClassification | 3 | conc_A→RED_NOISE, temp→BROADBAND, periodic→HARMONIC |
| TestEngineCorrections | 2 | Remove harmonic engines, add trend engines |
| TestDeterministicTrend | 5 | Exponential decay, accumulation, random walk (no), stationary (no), sine (no) |
| TestStationarityCorrection | 3 | conc_A override, white noise unchanged, temperature unchanged |
| TestGlobalStride | 3 | Sensible default, empty fallback, never zero |
| TestApplyCorrections | 4 | Full pipeline for conc_A, conc_B, temperature, all 7 differentiated |
| **Total** | **35** | **All passing** |

---

## Theoretical Notes

**Why this matters beyond CSTR:** Any slow-evolving system (degradation, diffusion, thermal decay, chemical kinetics) will produce 1/f spectral slopes that hit the first FFT bin. Without this fix, EVERY such system gets misclassified as PERIODIC. This is particularly critical for industrial diagnostics (ORTHON's target domain) where trending signals indicating degradation are the most important class to identify correctly.

**The Hurst + ACF gate is the key insight:** A signal with hurst > 0.85 and ACF that never decays is, by definition, a persistent trend. No genuine periodic signal can have both properties simultaneously — periodic signals have ACF that oscillates (decays and recovers).

**RED_NOISE is the correct spectral label:** Chemical kinetics, thermal processes, and degradation all produce 1/f power spectra. This is well-established in the signal processing literature. The RED_NOISE label carries information about the underlying physics — these systems are governed by first-order dynamics with characteristic time constants that concentrate energy at low frequencies.
