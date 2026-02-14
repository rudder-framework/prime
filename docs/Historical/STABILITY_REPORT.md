# Global Market Stability Analysis

**Generated:** 2026-02-04 17:52
**Data Range:** 1996-2026
**Signals:** 30 (global indices, rates, commodities, macro)

## Executive Summary

| Metric | Current Value | Status |
|--------|---------------|--------|
| eff_dim | 3.57 | 🟠 STRESS |
| alignment | 0.482 | PC1 explains 48% |
| Failure mode | mass_accumulation | bifurcation_rate |
| CSD detected | 43% of signals | ⚠️ WARNING |

**Assessment:** ELEVATED - Coupling exceeds COVID-2020 levels, approaching GFC territory

---

## Effective Dimension by Year

Lower eff_dim = more coupled = higher fragility

| Year | Mean | Min | Std | Status | Event |
|------|------|-----|-----|--------|-------|
| 1996 | 3.57 | 3.50 | 0.09 | 🟠 STRESS |  |
| 2000 | 5.79 | 4.48 | 0.71 | 🟢 HEALTHY | Dot-com peak |
| 2001 | 4.80 | 3.60 | 1.24 | 🟢 HEALTHY | 9/11 + Recession |
| 2002 | 4.45 | 4.00 | 0.32 | 🟡 CAUTION | Dot-com bottom |
| 2003 | 4.81 | 4.10 | 0.47 | 🟢 HEALTHY |  |
| 2004 | 4.08 | 3.34 | 0.63 | 🟡 CAUTION |  |
| 2005 | 4.58 | 3.76 | 0.52 | 🟢 HEALTHY |  |
| 2006 | 4.15 | 3.35 | 0.48 | 🟡 CAUTION |  |
| 2007 | 4.18 | 3.76 | 0.38 | 🟡 CAUTION |  |
| 2008 | 3.67 | 2.73 | 0.51 | 🟠 STRESS | GFC CRASH |
| 2009 | 2.73 | 2.26 | 0.30 | 🔴 CRISIS | GFC bottom |
| 2010 | 3.51 | 2.33 | 0.95 | 🟠 STRESS | Flash Crash |
| 2011 | 4.20 | 3.73 | 0.29 | 🟡 CAUTION | Euro Crisis |
| 2012 | 4.07 | 3.43 | 0.43 | 🟡 CAUTION |  |
| 2013 | 4.90 | 4.26 | 0.33 | 🟢 HEALTHY |  |
| 2014 | 5.82 | 4.87 | 0.38 | 🟢 HEALTHY |  |
| 2015 | 4.83 | 4.04 | 0.63 | 🟢 HEALTHY | China deval |
| 2016 | 4.46 | 3.68 | 0.58 | 🟡 CAUTION |  |
| 2017 | 3.66 | 2.94 | 0.60 | 🟠 STRESS |  |
| 2018 | 4.64 | 3.30 | 0.77 | 🟢 HEALTHY | Vol spike |
| 2019 | 5.93 | 4.95 | 0.70 | 🟢 HEALTHY |  |
| 2020 | 4.41 | 3.94 | 0.47 | 🟡 CAUTION | COVID CRASH |
| 2021 | 3.52 | 2.78 | 0.56 | 🟠 STRESS |  |
| 2022 | 4.41 | 3.69 | 0.81 | 🟡 CAUTION | Rate hikes |
| 2023 | 4.41 | 3.99 | 0.32 | 🟡 CAUTION |  |
| 2024 | 4.60 | 4.02 | 0.22 | 🟢 HEALTHY | AI bubble? |
| 2025 | 5.14 | 4.04 | 0.59 | 🟢 HEALTHY |  |
| 2026 | 3.73 | 3.57 | 0.17 | 🟠 STRESS | NOW |

---

## Key Findings

### 1. Geometry Validates Against Known Events

- **GFC 2008-2009:** eff_dim collapsed to 2.26 (worst ever)
- **COVID 2020:** eff_dim = 4.41 (mild - V-shaped recovery prevented collapse)
- **Post-COVID 2021:** eff_dim = 3.52 (WORSE than crash - stimulus herding)
- **Current 2026:** eff_dim = 3.73 (stress zone, worse than COVID)

### 2. Mass Dynamics are Endogenous

Granger causality analysis shows:
- **M2 → Mass:** p=0.78 (NO causal effect)
- **Mass → M2:** p<0.0001 (STRONG effect)

The Fed FOLLOWS markets, not leads them. Investment flows are attracted by geometry.

### 3. Current Trajectory

| Period | eff_dim | Interpretation |
|--------|---------|----------------|
| 2019 | 5.93 | Peak health |
| 2024 | 4.60 | AI optimism |
| 2025 | 5.14 | Expansion |
| 2026 | 3.73 | Sharp collapse |

**2025 → 2026 drop of 1.4 points mirrors 2007 → 2008 pattern.**

---

## Lyapunov Exponents (Stability)

| Signal | λ | Stability | Predictability |
|--------|---|-----------|----------------|
| stoxx600_return_1yr | +0.0589 | weakly_unstable | 12 mo |
| unemployment | +0.0571 | weakly_unstable | 12 mo |
| nikkei_return_1yr | +0.0504 | weakly_unstable | 14 mo |
| shanghai | +0.0486 | weakly_unstable | 14 mo |
| msci_em_return_1yr | +0.0485 | weakly_unstable | 14 mo |
| india_nifty | +0.0474 | weakly_unstable | 15 mo |
| stoxx600 | +0.0455 | weakly_unstable | 15 mo |
| m2_velocity | +0.0446 | weakly_unstable | 16 mo |
| treasury_2y | +0.0446 | weakly_unstable | 16 mo |
| treasury_10y | +0.0420 | weakly_unstable | 17 mo |
| dax | +0.0414 | weakly_unstable | 17 mo |
| vix | +0.0411 | weakly_unstable | 17 mo |
| sp500_return_1yr | +0.0396 | weakly_unstable | 17 mo |
| nikkei | +0.0391 | weakly_unstable | 18 mo |

---

## Critical Slowing Down

**13/30 signals show CSD (43%)**

| Signal | CSD Score | AC(1) | Variance Trend |
|--------|-----------|-------|----------------|
| nasdaq | 0.730 | 0.934 | +6.22e+04 |
| gdp | 0.727 | 0.924 | +8.76e+04 |
| gold | 0.725 | 0.915 | +1.84e+03 |
| sp500 | 0.723 | 0.909 | +3.27e+03 |
| cpi | 0.721 | 0.902 | +7.39e-01 |
| dax | 0.712 | 0.875 | +3.50e+04 |
| m2 | 0.706 | 0.978 | +2.46e+04 |
| nikkei | 0.705 | 0.848 | +5.07e+04 |
| oil | 0.704 | 0.846 | +1.91e+00 |
| hang_seng | 0.701 | 0.835 | +2.54e+05 |

---

## Formal Assessment

```
======================================================================
DYNAMICAL SYSTEMS ASSESSMENT
======================================================================

SYSTEM CLASSIFICATION
  Attractor type: strange_attractor
  Stability: unstable
  System typology: accumulation
  Failure mode: mass_accumulation

GEOMETRY (Structure)
  Effective dimension: 3.57
  Alignment (PC1 dominance): 0.482
  Mean correlation: 0.359
  Dimensional collapse: Not detected

MASS (Slow Variable)
  Total variance: 4.20e+08
  Drift rate: +0.0023
  Mass dynamics: Accumulating

EARLY WARNING SIGNALS
  Critical slowing down: DETECTED in 43% of signals

TIPPING CLASSIFICATION
  Type: bifurcation_rate
  Mechanism: Combined bifurcation and rate effects

ASSESSMENT
  Bifurcation proximity: 64.3%
  Risk level: ELEVATED
======================================================================
```

---

## Interpretation

### Structure = Geometry × Mass

- **Geometry (eff_dim):** How signals move together
  - High = orthogonal, independent, healthy
  - Low = coupled, herding, fragile

- **Mass (total_variance):** Accumulated capital/valuations
  - Endogenously driven by geometry
  - Fed responds to mass, doesn't create it

### Tipping Types

- **B-tipping:** Geometry fails (dimensional collapse)
- **R-tipping:** Mass changes too fast for adaptation

Current state: **Combined B+R risk** - geometry collapsing while mass accumulated.

---

## Files

- `observations.parquet` - Raw data (30 signals, 1970-2026)
- `stability_output/signal_geometry.parquet` - 688 rolling windows
- `stability_output/lyapunov_analysis.parquet` - Per-signal stability
- `stability_output/csd_analysis.parquet` - Early warning indicators
- `stability_output/formal_assessment.json` - Classification

---

*Generated by PRISM Stability Pipeline*
