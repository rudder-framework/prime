# Bearing Dataset Review

**Domain:** Rotating machinery vibration
**Dataset:** 3 bearings × 2 axes (acc_x, acc_y) = 6 signals, 10k samples each
**Date:** 2025-02-02

---

## Physical Structure

Each bearing has a distinct rotation frequency with clear harmonic series:

| Bearing | Fundamental (f0) | 2nd Harmonic | 3rd Harmonic | Cycles/1000 samples |
|---------|-------------------|--------------|--------------|---------------------|
| bearing_1 | 0.030 | 0.060 | 0.090 | 30 |
| bearing_2 | 0.035 | 0.070 | 0.105 | 35 |
| bearing_3 | 0.040 | 0.080 | 0.120 | 40 |

All harmonics have SNR > 30 dB — unmistakably periodic machinery vibration.

---

## Issue 1: CHAOTIC → PERIODIC (all 6 signals)

**Severity: HIGH**

All 6 signals classified as CHAOTIC. They are PERIODIC.

The current tree appears to trigger CHAOTIC based on `lyapunov_proxy > 0.5` without sufficient guards. For bearing vibration, a positive Lyapunov proxy reflects broadband noise superimposed on periodic rotation — not deterministic chaos. A truly chaotic signal cannot have SNR > 30 dB at discrete harmonic frequencies.

**PR3 corrected tree fixes this.** All 6 signals pass all 6 periodicity gates:

| Gate | Threshold | bearing_1/acc_x | Pass? |
|------|-----------|-----------------|-------|
| Not first-bin artifact | freq ≠ 1/256 | 0.03125 ≫ 0.00391 | ✓ |
| Not broadband | flatness < 0.7 | 0.068 | ✓ |
| Strong peak | SNR > 6 dB | 31.1 dB | ✓ |
| ACF oscillates | half_life exists | 5.0 | ✓ |
| Not extreme trend | Hurst < 0.95 | 0.610 | ✓ |
| Not monotonic | TPR < 0.95 | 0.633 | ✓ |

**Diagnostic impact:** CHAOTIC triggers attractor reconstruction (useless for bearings). PERIODIC triggers harmonic/spectral analysis (critical for defect detection in rotating machinery).

---

## Issue 2: Spectral Classification — PERIODIC Override Needed

**Severity: MEDIUM — Fixed in updated PR3**

The original PR3 spectral correction (`spectral_slope < -0.5 → RED_NOISE`) would miscategorize 4 of 6 bearing signals as RED_NOISE. The negative spectral slope comes from the harmonic series having decreasing amplitude (f0 > 2f0 > 3f0), which is normal harmonic rolloff, not continuous 1/f noise.

**Fix applied:** When `temporal_pattern == PERIODIC`, skip the slope-based RED_NOISE check entirely. Use HNR to distinguish:
- HNR > 3.0 → HARMONIC (strong harmonic dominance)
- HNR ≤ 3.0 → NARROWBAND (harmonics + noise floor)

This correctly gives all 6 bearing signals NARROWBAND while preserving RED_NOISE for CSTR trending signals.

---

## Issue 3: Manifest — 4 of 6 Signals Missing

**Severity: HIGH**

`manifest.yaml` reports `total_signals: 6` but only defines 2 signal entries (acc_x and acc_y).

**Root cause:** The manifest generator uses `signal_id` as the YAML dictionary key. When multiple unit_ids share the same signal_id, only the last one survives:

```
bearing_1/acc_x → signals.acc_x  (written)
bearing_2/acc_x → signals.acc_x  (overwritten!)
bearing_3/acc_x → signals.acc_x  (overwritten!)
```

Only bearing_3's data survives in the manifest. PRISM would only process 2 of 6 signals.

**Fix options:**
1. **Concatenated key:** `signals.bearing_1__acc_x` (simple, flat)
2. **Nested:** `signals.acc_x.bearing_1` (grouped by signal)
3. **List:** `signals` as array instead of dict (avoids key collision entirely)

---

## Issue 4: stride: 1 (Already Flagged)

Global `params.stride: 1` with `window: 128` and `n=10000` creates 9,873 windows per signal — 59k total windows across 6 signals. With stride=64 (50% of window), this drops to 930 total windows (64× reduction).

---

## Correct Classifications

| Dimension | Value | Assessment |
|-----------|-------|------------|
| stationarity | STATIONARY | ✓ ADF rejects, KPSS accepts |
| memory | SHORT_MEMORY | ✓ Hurst 0.45-0.61 |
| complexity | HIGH | ✓ sample_entropy 0.75-0.84 |
| continuity | CONTINUOUS | ✓ accelerometer data |
| determinism | STOCHASTIC | ✓ determinism_score < 0.27 |
| distribution | LIGHT_TAILED/GAUSSIAN | ✓ kurtosis 2.1-3.7 |
| amplitude | MIXED (x) / SMOOTH (y) | ✓ crest_factor split |
| volatility | CLUSTERING (x) / HOMOSCEDASTIC (y) | ✓ physically plausible |
| recommended_window | 128 | ✓ reasonable |

**Notable:** acc_x consistently shows VOLATILITY_CLUSTERING while acc_y is HOMOSCEDASTIC across all 3 bearings. This suggests different vibration coupling in the two axes, which is physically expected (radial vs. tangential loading).

---

## PR3 Update Summary

The spectral classification fix identified during this review has been applied to the PR3 correction code:

- **`classify_spectral()`** now has a PERIODIC early-return that bypasses slope checks
- **2 new spectral tests** added: `test_periodic_low_hnr_is_narrowband`, `test_periodic_bypasses_red_noise_slope`
- **5 new bearing integration tests** added: temporal, spectral, freq preservation, cross-domain differentiation
- **Test count: 35 → 42, all passing**
