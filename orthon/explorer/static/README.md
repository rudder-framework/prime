# ORTHON

**Intelligent Orchestration Layer for PRISM Compute Engines**

*PRISM stays dumb, ORTHON does the thinking.*

---

## Architecture

```
User uploads file
       ↓
┌─────────────────────────────────────────────────────────┐
│  ORTHON GATEKEEPER                                      │
│                                                         │
│  • Scan for units (80+ patterns)                        │
│  • Detect constants (headers + columns)                 │
│  • Classify physical quantities                         │
│  • Route to appropriate engines                         │
│  • Pre-flight validation                                │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│  PRISM COMPUTE (dumb)                                   │
│                                                         │
│  • 200+ mathematical engines                            │
│  • Window slicing                                       │
│  • Raw metric computation                               │
│  • Parquet output                                       │
└─────────────────────────────────────────────────────────┘
       ↓
┌─────────────────────────────────────────────────────────┐
│  ORTHON RESULTS                                         │
│                                                         │
│  • DuckDB-WASM SQL queries                              │
│  • Validation checks                                    │
│  • Interactive exploration                              │
└─────────────────────────────────────────────────────────┘
```

---

## Quick Start

```bash
# Start the ORTHON server
uvicorn orthon.api:app --reload --port 8000

# Open browser
open http://localhost:8000
```

1. **Upload** a data file (CSV, Parquet, Excel)
2. **Review** detected structure, units, and constants
3. **Configure** window parameters
4. **Analyze** with PRISM
5. **Query** results with SQL

---

## Unit Detection

ORTHON automatically detects units from column name suffixes and converts to SI.

### Pressure
| Suffix | Unit | SI Conversion |
|--------|------|---------------|
| `_psi` | psi | × 6894.76 → Pa |
| `_psia` | psia | × 6894.76 → Pa |
| `_psig` | psig | × 6894.76 → Pa |
| `_bar` | bar | × 100000 → Pa |
| `_kpa` | kPa | × 1000 → Pa |
| `_atm` | atm | × 101325 → Pa |

### Temperature
| Suffix | Unit | SI Conversion |
|--------|------|---------------|
| `_F`, `_degF` | °F | (T × 5/9) + 255.37 → K |
| `_C`, `_degC` | °C | T + 273.15 → K |
| `_K` | K | (already SI) |
| `_degR`, `_R` | °R | × 5/9 → K |

### Velocity
| Suffix | Unit | SI Conversion |
|--------|------|---------------|
| `_m_s`, `_mps` | m/s | (already SI) |
| `_ft_s`, `_fps` | ft/s | × 0.3048 → m/s |
| `_km_h`, `_kph` | km/h | × 0.27778 → m/s |
| `_mph` | mph | × 0.44704 → m/s |

### Flow Rate
| Suffix | Unit | SI Conversion |
|--------|------|---------------|
| `_gpm` | gal/min | × 6.309e-5 → m³/s |
| `_lpm` | L/min | × 1.667e-5 → m³/s |
| `_cfm` | ft³/min | × 4.719e-4 → m³/s |
| `_kg_s` | kg/s | (already SI) |

### Electrical
| Suffix | Unit | SI Conversion |
|--------|------|---------------|
| `_V` | V | (already SI) |
| `_A` | A | (already SI) |
| `_W` | W | (already SI) |
| `_kW` | kW | × 1000 → W |
| `_hp` | hp | × 745.7 → W |

### Other
| Suffix | Unit | Quantity |
|--------|------|----------|
| `_rpm` | rpm | angular velocity |
| `_hz` | Hz | frequency |
| `_kg` | kg | mass |
| `_N` | N | force |
| `_Nm` | N·m | torque |
| `_pct` | % | ratio |

---

## Constant Detection

ORTHON extracts constants from:

### 1. CSV Header Comments
```csv
# density = 1020 kg_m3
# viscosity = 0.001 Pa_s
# diameter = 0.1 m
timestamp,flow_gpm,pressure_psi
...
```

### 2. Constant Columns
Columns with a single value (or constant per entity) are automatically detected:
```csv
entity_id,timestamp,flow_gpm,density_kg_m3
pump_1,0,100,1020
pump_1,1,102,1020
pump_1,2,98,1020
```

---

## Engine Reference

PRISM computes 200+ metrics across 4 stages. All formulas verified for academic research.

---

## Stage 1: Signal Typology

Per-signal behavioral metrics.

### 1.1 Hurst Exponent (Memory)

Measures long-range dependence via R/S analysis.

$$H = \frac{\log(R/S)}{\log(n)}$$

Where:
- $R$ = Range of cumulative deviations
- $S$ = Standard deviation
- $n$ = Window size

**Cumulative deviation:**
$$Y_t = \sum_{i=1}^{t}(x_i - \bar{x})$$

**Rescaled range:**
$$\frac{R}{S} = \frac{\max(Y) - \min(Y)}{\sigma}$$

| H Value | Behavior |
|---------|----------|
| H < 0.5 | Anti-persistent (mean-reverting) |
| H = 0.5 | Random walk |
| H > 0.5 | Persistent (trending) |

---

### 1.2 Sample Entropy (Regularity)

Measures signal predictability.

$$\text{SampEn}(m, r) = -\ln\frac{A}{B}$$

Where:
- $m$ = embedding dimension
- $r$ = tolerance (typically 0.2 × std)
- $A$ = matching templates of length m+1
- $B$ = matching templates of length m

| SampEn | Meaning |
|--------|---------|
| ~0 | Highly regular |
| 1-2 | Normal complexity |
| >2 | High irregularity |

---

### 1.3 Permutation Entropy (Complexity)

Measures ordinal pattern complexity.

$$H_p = -\sum_{i=1}^{m!} p_i \log_2(p_i)$$

Normalized: $PE = \frac{H_p}{\log_2(m!)}$

| PE | Meaning |
|----|---------|
| ~0 | Perfectly predictable |
| ~0.5 | Some structure |
| ~1 | Maximum complexity |

---

### 1.4 GARCH(1,1) Volatility

Models volatility clustering.

$$\sigma_t^2 = \omega + \alpha \epsilon_{t-1}^2 + \beta \sigma_{t-1}^2$$

**Persistence:** $\alpha + \beta$ (> 0.9 = high persistence)

**Half-life:** $\frac{\log(0.5)}{\log(\alpha+\beta)}$

---

### 1.5 Lyapunov Exponent (Chaos)

Measures sensitivity to initial conditions.

$$\lambda = \lim_{t \to \infty} \frac{1}{t} \ln \frac{|\delta(t)|}{|\delta_0|}$$

| λ Value | System Type |
|---------|-------------|
| λ < 0 | Stable |
| λ ≈ 0 | Marginally stable |
| λ > 0 | Chaotic |

---

### 1.6 Spectral Analysis

**Power Spectral Density:**
$$P(f) = |X(f)|^2$$

**Spectral Centroid:**
$$f_c = \frac{\sum_f f \cdot P(f)}{\sum_f P(f)}$$

**Spectral Entropy:**
$$SE = -\sum_f p(f) \log_2 p(f)$$

---

### 1.7 Recurrence Quantification (RQA)

**Recurrence Matrix:**
$$R_{ij} = \Theta(\epsilon - \|\mathbf{v}_i - \mathbf{v}_j\|)$$

**Recurrence Rate:**
$$RR = \frac{1}{N^2} \sum_{i,j} R_{ij}$$

**Determinism:**
$$DET = \frac{\sum_{l \geq l_{min}} l \cdot P(l)}{\sum_{i,j} R_{ij}}$$

| Metric | High Value Means |
|--------|------------------|
| RR | Many recurrences |
| DET | Deterministic dynamics |
| LAM | Laminar states |

---

### 1.8 Wavelet Decomposition

**Energy per scale:**
$$E_j = \sum_k |d_{j,k}|^2$$

**Wavelet Entropy:**
$$WE = -\sum_j e_j \log_2(e_j)$$

Where $e_j = E_j / \sum E$

---

### 1.9 ACF Decay (Memory)

**Autocorrelation:**
$$\rho(k) = \frac{\text{Cov}(x_t, x_{t+k})}{\text{Var}(x)}$$

**Exponential decay:**
$$\rho(k) \approx e^{-k/\tau}$$

---

## Stage 2: Behavioral Geometry

Cross-signal relationships.

### 2.1 Correlation Matrix

**Pearson correlation:**
$$r_{xy} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2 \sum(y_i - \bar{y})^2}}$$

---

### 2.2 Mutual Information

**Information shared between signals:**
$$I(X; Y) = H(X) + H(Y) - H(X, Y)$$

Gaussian approximation:
$$I(X; Y) \approx -\frac{1}{2} \log_2(1 - \rho^2)$$

---

### 2.3 Coherence

**Spectral coherence:**
$$C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f) P_{yy}(f)}$$

---

### 2.4 PCA (Dimensionality)

**Eigenvalue problem:**
$$\Sigma \mathbf{v} = \lambda \mathbf{v}$$

**Explained variance:**
$$\rho_k = \frac{\lambda_k}{\sum_i \lambda_i}$$

**Effective dimensionality:**
$$d_{eff} = \frac{(\sum_i \lambda_i)^2}{\sum_i \lambda_i^2}$$

---

### 2.5 Clustering (Silhouette)

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = mean distance within cluster
- $b(i)$ = mean distance to nearest cluster

| Score | Interpretation |
|-------|----------------|
| > 0.7 | Strong structure |
| 0.5-0.7 | Reasonable |
| < 0.25 | No structure |

---

## Stage 3: Dynamical Systems

System evolution over time.

### 3.1 Laplace Field

**Gradient (velocity):**
$$\nabla E(t) = \frac{E(t+1) - E(t-1)}{2}$$

**Laplacian (acceleration):**
$$\nabla^2 E(t) = E(t+1) - 2E(t) + E(t-1)$$

**Divergence:**
$$\text{div}(E) = \sum_i \frac{\partial^2 E_i}{\partial t^2}$$

| Divergence | Role |
|------------|------|
| > 0 | SOURCE (stress emanates) |
| < 0 | SINK (stress absorbs) |
| ≈ 0 | BRIDGE (stress transmits) |

---

### 3.2 Hamiltonian (Energy)

$$H = T + V = \frac{1}{2}\dot{x}^2 + \frac{1}{2}(x - \bar{x})^2$$

| dH/dt | Meaning |
|-------|---------|
| ~0 | Energy conserved |
| > 0 | Energy injected |
| < 0 | Energy dissipating |

---

### 3.3 HD-Slope (Degradation Rate)

Rate of drift from baseline in feature space.

$$hd\_slope = \frac{d(\|v - v_0\|)}{dt}$$

| hd_slope | Meaning |
|----------|---------|
| ~0 | Stable |
| Positive | Degrading |
| Large positive | Failure imminent |

**This is the key prognostic metric.**

---

## Stage 4: Causal Mechanics

Information flow and causality.

### 4.1 Granger Causality

Does X help predict Y beyond Y's own history?

**F-test:**
$$F = \frac{(RSS_r - RSS_u) / p}{RSS_u / (n - 2p - 1)}$$

| p-value | Interpretation |
|---------|----------------|
| < 0.05 | X Granger-causes Y |
| > 0.05 | No evidence |

---

### 4.2 Transfer Entropy

**Information flow in bits:**
$$TE_{X \to Y} = H(Y_{t+1} | Y_t) - H(Y_{t+1} | Y_t, X_t)$$

| TE | Meaning |
|----|---------|
| 0 | No information flow |
| Positive | X informs Y's future |
| TE(X→Y) > TE(Y→X) | Net flow X to Y |

---

## Physics Engines

Domain-specific calculations requiring constants.

### Reynolds Number
$$Re = \frac{\rho v D}{\mu}$$

**Requires:** density, velocity signal, diameter, viscosity

| Re | Flow Regime |
|----|-------------|
| < 2300 | Laminar |
| 2300-4000 | Transition |
| > 4000 | Turbulent |

---

### Pressure Drop (Darcy-Weisbach)
$$\Delta P = f \frac{L}{D} \frac{\rho v^2}{2}$$

**Requires:** density, velocity, diameter, length, friction factor

---

### Kinetic Energy
$$KE = \frac{1}{2}mv^2$$

**Requires:** mass, velocity signal

---

### Heat Transfer
$$Q = mc_p \Delta T$$

**Requires:** mass, specific heat, temperature signal

---

## Pre-flight Validation

ORTHON validates before sending to PRISM:

| Check | Pass | Fail |
|-------|------|------|
| Row count | ≥ window_size | Insufficient data |
| Window coverage | windows > 0 | Window too large |
| Signal columns | numeric found | No signals |
| Null rate | < 50% | Too many nulls |

---

## SQL Query Library

Built-in queries for common analysis patterns:

### Typology
- Signal Summary
- Signal Characterization
- Metric Evolution
- Anomalous Windows

### Geometry
- Correlation Matrix
- Coupling Strength
- Cluster Candidates
- Decoupling Events

### Dynamics
- System Stability
- Energy Evolution
- Bifurcation Detection
- Attractor Properties

### Mechanics
- Information Flow
- Granger Causality
- Causal Graph
- System Drivers

### Validation
- Unit Consistency
- Null Value Check
- Range Check
- Window Gap Check

---

## Engine Quick Reference

| Engine | Formula | Output |
|--------|---------|--------|
| Hurst | $\log(R/S) \propto H \log(n)$ | H ∈ [0,1] |
| SampEn | $-\ln(A/B)$ | regularity |
| PermEnt | $-\sum p_i \log p_i$ | complexity |
| GARCH | $\sigma^2 = \omega + \alpha\epsilon^2 + \beta\sigma^2$ | volatility |
| Lyapunov | $\lambda = \lim \frac{1}{t}\ln\frac{d(t)}{d(0)}$ | chaos |
| RQA | $RR = \frac{\sum R_{ij}}{N^2}$ | recurrence |
| PCA | $\Sigma v = \lambda v$ | dimensionality |
| MI | $H(X) + H(Y) - H(X,Y)$ | dependence |
| Granger | F-test on RSS | causality |
| TransEnt | $H(Y'|Y) - H(Y'|Y,X)$ | info flow |
| hd_slope | $\frac{d\|v-v_0\|}{dt}$ | degradation |
| Hamiltonian | $T + V$ | energy |
| Reynolds | $\rho vD/\mu$ | flow regime |

---

## Capability Levels

| Level | Name | Requirements |
|-------|------|--------------|
| 0 | Basic | Any numeric signals |
| 1 | Units | Unit suffixes detected |
| 2 | Geometry | 2+ signals |
| 3 | Constants | Constants available |
| 4 | Physics | Constants + physics quantities |

---

## Links

- [PRISM](https://github.com/prism) — The compute engine
- [GitHub](https://github.com/orthon) — Source code

---

*ORTHON interprets. PRISM computes. Geometry leads.*
