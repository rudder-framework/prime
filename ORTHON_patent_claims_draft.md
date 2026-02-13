# ORTHON — Draft Patent Claims
## Domain-Agnostic Signal Degradation Detection via Multi-Layer Geometric and Dynamical Analysis

**DISCLAIMER:** This is a technical draft, not legal advice. Requires review by a patent attorney before filing. All claims are preliminary and may need narrowing or restructuring to satisfy 35 U.S.C. §101 (patentable subject matter), §102 (novelty), §103 (non-obviousness), and §112 (enablement).

**Inventor:** Jason [Last Name]
**Date of Draft:** February 10, 2026

---

## INDEPENDENT CLAIM 1 — The Full Method

A computer-implemented method for detecting degradation in a physical or engineered system from multi-channel time-series signals without requiring domain-specific training data, supervised labels, or machine learning models, the method comprising:

**(a) Signal Typology Classification:**
receiving a plurality of time-series signals from a system under observation, and automatically classifying each signal into one or more signal typology categories based on intrinsic statistical and spectral properties of the signal, without reference to domain metadata, sensor labels, or prior knowledge of the system, wherein the typology classification determines subsequent analysis parameters including window sizes, embedding dimensions, and applicable geometric transforms;

**(b) Behavioral Geometry Extraction:**
for each of a plurality of temporally sliding windows over the classified signals:

- (b.i) constructing a multivariate state matrix from the windowed signal values,
- (b.ii) computing eigenvalue decomposition of said state matrix to obtain eigenvalue spectra and eigenvector loadings,
- (b.iii) deriving an effective dimensionality measure from the eigenvalue spectrum, wherein the effective dimensionality quantifies the number of independent degrees of freedom active in the system at each window,
- (b.iv) computing geometric signatures including eigenvalue ratios, eigenvalue entropy, condition number, and eigenvector stability across successive windows,
- (b.v) detecting dimensional collapse patterns wherein a sustained reduction in effective dimensionality over successive windows indicates progressive loss of system degrees of freedom characteristic of degradation;

**(c) Dynamical Systems Analysis:**
for each signal classified in step (a):

- (c.i) reconstructing a phase space embedding using time-delay embedding with embedding dimension and delay parameters selected by automated methods including Cao's method and mutual information minimization,
- (c.ii) computing finite-time Lyapunov exponents (FTLE) in both forward and backward temporal directions from said phase space embedding, wherein the FTLE quantifies the rate of divergence or convergence of nearby trajectories in phase space,
- (c.iii) computing velocity fields and acceleration components (parallel and perpendicular to the trajectory) in the reconstructed state space to characterize the system's motion through its state manifold,
- (c.iv) detecting FTLE ridges — surfaces of locally maximal trajectory separation — by computing ridge strength, ridge width, and corridor width across pairs of signals, wherein FTLE ridges indicate boundaries between qualitatively different dynamical regimes,
- (c.v) classifying dynamical stability of each signal as stable, marginally stable, or unstable based on the computed FTLE values and their temporal statistics;

**(d) Energy and Conservation Analysis:**
- (d.i) computing energy-like quantities from the eigenvalue spectrum, including total variance (analogous to total system energy) and its partition across principal components,
- (d.ii) tracking the temporal evolution of said energy partition to detect energy transfer between modes, wherein concentration of energy into fewer modes indicates approach to a degraded or failure state,
- (d.iii) computing curvature of the system trajectory in state space as a measure of how rapidly the system's dynamical behavior is changing;

**(e) Causal and Information-Theoretic Analysis:**
- (e.i) computing pairwise information flow measures between signals including Granger causality, transfer entropy, and KL divergence to detect directed causal relationships,
- (e.ii) comparing said information flow measures between temporal segments to detect emergence, disappearance, strengthening, or weakening of causal links as the system evolves,
- (e.iii) constructing network topology graphs from pairwise signal similarity measures and tracking changes in network connectivity, density, and degree distribution over time;

**(f) Degradation Scoring:**
producing one or more degradation scores for the system by combining outputs of steps (b) through (e) using deterministic rules, lookup tables, and threshold comparisons, without employing trained statistical models, neural networks, or machine learning classifiers, wherein the degradation scores are invariant to the specific domain, application, or physical nature of the monitored system.

---

## INDEPENDENT CLAIM 2 — The System Architecture

A system for domain-agnostic degradation detection comprising:

**(a)** one or more computational engines, each engine configured to receive windowed time-series data and produce numerical output arrays, wherein each engine performs a single mathematical transformation and contains no conditional logic, classification rules, or decision functions;

**(b)** a structured query processing layer configured to receive the numerical outputs of said computational engines and perform all analytical logic including layering, joining, windowing, temporal aggregation, classification, and threshold comparison, wherein said structured query processing layer implements the analytical intelligence of the system while the computational engines implement only mathematical transforms;

**(c)** a signal typology module that classifies input signals based on intrinsic properties and selects appropriate engine configurations without domain metadata;

**(d)** a geometric analysis module that computes eigenvalue decomposition over sliding windows and derives effective dimensionality, dimensional collapse trajectories, and geometric signatures;

**(e)** a dynamical systems module that reconstructs phase space embeddings, computes finite-time Lyapunov exponents in forward and backward directions, detects FTLE ridges between signal pairs, and classifies dynamical stability;

**(f)** a velocity field module that computes state-space velocities, accelerations, and trajectory curvature;

**(g)** an information flow module that computes directed causal measures between signal pairs and detects temporal changes in causal structure;

wherein the system produces degradation assessments without requiring training data, supervised labels, domain-specific configuration, or machine learning models.

---

## DEPENDENT CLAIMS — Narrowing Specifics

### On Signal Typology (from Claim 1a)

**Claim 3.** The method of Claim 1, wherein the signal typology classification in step (a) comprises determining whether each signal exhibits one or more of: monotonic drift, quasi-periodic oscillation, regime-switching behavior, stochastic wandering, or stationary fluctuation, and wherein signals of different typologies are analyzed with different window sizes, embedding parameters, and geometric transforms appropriate to their dynamical character.

**Claim 4.** The method of Claim 3, wherein the typology classification is performed using statistical tests including stationarity tests, autocorrelation structure analysis, spectral peak detection, and distribution shape measures, without requiring comparison to labeled exemplars or training sets.

### On Geometric Analysis (from Claim 1b)

**Claim 5.** The method of Claim 1, wherein the effective dimensionality in step (b.iii) is computed as the exponential of the normalized eigenvalue entropy: eff_dim = exp(−Σ pᵢ ln pᵢ), where pᵢ = λᵢ / Σλⱼ and λᵢ are the eigenvalues, providing a continuous measure of the number of active degrees of freedom that is robust to noise and does not require selection of a variance-explained threshold.

**Claim 6.** The method of Claim 1, wherein dimensional collapse detection in step (b.v) comprises computing the temporal derivative of effective dimensionality across successive windows and identifying sustained negative trends, and wherein the rate of dimensional collapse is used as a direct indicator of degradation severity without requiring calibration against known failure instances.

**Claim 7.** The method of Claim 1, wherein step (b.iv) further comprises tracking eigenvector rotation across successive windows by computing the alignment between eigenvector sets at adjacent time windows, wherein rapid eigenvector rotation indicates reorganization of the system's correlation structure.

### On Dynamical Systems (from Claim 1c)

**Claim 8.** The method of Claim 1, wherein the phase space reconstruction in step (c.i) uses Takens' embedding theorem to reconstruct an attractor from scalar time-series observations, with embedding dimension selected by Cao's method and time delay selected by first minimum of mutual information, and wherein the FTLE computation in step (c.ii) is performed by computing the maximum eigenvalue of the Cauchy-Green deformation tensor constructed from the linearized flow map of nearby trajectories in the reconstructed phase space.

**Claim 9.** The method of Claim 1, wherein the FTLE ridge detection in step (c.iv) identifies Lagrangian Coherent Structures (LCS) — material barriers to transport in the system's state space — by locating surfaces where the FTLE field achieves local maxima, and wherein the proximity of the system's current state to detected ridges is used as a predictive indicator of impending regime transitions or failure events.

**Claim 10.** The method of Claim 1, wherein the velocity field computation in step (c.iii) decomposes acceleration into components parallel and perpendicular to the instantaneous trajectory, wherein the parallel component measures whether the system is speeding up or slowing down along its current path and the perpendicular component measures how rapidly the system is changing direction, and wherein increasing perpendicular acceleration indicates departure from the system's historical behavioral manifold.

### On Energy Signatures (from Claim 1d)

**Claim 11.** The method of Claim 1, wherein the energy partition analysis in step (d.ii) tracks the fraction of total eigenvalue variance captured by the first k eigenvalues over time, and wherein progressive concentration of variance into fewer eigenvalues — analogous to energy condensation in a physical system — indicates loss of operational modes characteristic of degradation.

**Claim 12.** The method of Claim 1, wherein the trajectory curvature in step (d.iii) is computed from the velocity and acceleration fields of step (c.iii), and wherein sustained high curvature indicates the system is undergoing rapid behavioral change inconsistent with normal operation.

### On Causal Analysis (from Claim 1e)

**Claim 13.** The method of Claim 1, wherein the temporal comparison in step (e.ii) divides the observation period into early and late segments based on a percentage split, computes Granger causality F-statistics independently on each segment, and classifies each pairwise causal link as emerged, disappeared, strengthened, weakened, or persisted based on the comparison, providing a map of how the system's causal architecture evolves during degradation.

**Claim 14.** The method of Claim 1, wherein the network topology in step (e.iii) uses an adaptive similarity threshold derived from the statistical distribution of pairwise similarities within each temporal window, such that the number of network edges varies with the system's correlation structure, and wherein changes in network density, mean degree, and degree distribution over time serve as degradation indicators.

### On Domain Agnosticism (from Claim 1 overall)

**Claim 15.** The method of Claim 1, wherein the method is applied without modification to time-series data from any of: rotating machinery vibration, turbofan engine telemetry, battery charge-discharge cycling, electrocardiogram signals, seismic waveforms, hydraulic pressure systems, chemical process control, bearing degradation, climate sensor arrays, or physiological monitoring, and wherein the signal typology classification of step (a) automatically adapts the analysis parameters to the characteristics of each domain without requiring domain-specific configuration files, training data, or expert knowledge.

**Claim 16.** The method of Claim 1, wherein the entirety of steps (b) through (f) uses only: eigenvalue decomposition, sliding window statistics, lookup tables, deterministic threshold comparisons, and structured query operations, and explicitly excludes the use of gradient descent, backpropagation, neural network inference, support vector classification, random forest prediction, or any other trained statistical model.

### On Architecture (from Claim 2)

**Claim 17.** The system of Claim 2, wherein each computational engine in component (a) is a pure function that accepts a numerical array and returns a numerical array, performs no data persistence, maintains no internal state between invocations, and contains no conditional branching based on domain, sensor identity, or classification results.

**Claim 18.** The system of Claim 2, wherein the structured query processing layer of component (b) is implemented using SQL or an equivalent relational query language operating on columnar data stores, and wherein all classification logic, temporal windowing, cross-signal joining, aggregation, and degradation scoring is expressed as declarative queries rather than procedural code.

---

## ABSTRACT

A method and system for detecting degradation in physical and engineered systems from multi-channel time-series signals without requiring domain-specific knowledge, training data, or machine learning models. The method comprises four analytical layers applied in sequence: (1) automatic signal typology classification based on intrinsic signal properties; (2) behavioral geometry extraction via eigenvalue decomposition over sliding windows, producing effective dimensionality trajectories and dimensional collapse detection; (3) dynamical systems analysis including phase space reconstruction, finite-time Lyapunov exponent computation, FTLE ridge detection identifying Lagrangian Coherent Structures, velocity field characterization, and stability classification; and (4) causal mechanics analysis including directed information flow measurement, temporal causal structure comparison, and adaptive network topology construction. All analytical reasoning is performed through deterministic rules and structured queries operating on numerical outputs from stateless computational engines, producing domain-invariant degradation scores using only eigenvalue decomposition, lookup tables, and threshold logic. The system architecture enforces strict separation between mathematical computation (engines) and analytical logic (structured queries), enabling application across arbitrary signal domains including industrial machinery, medical devices, energy systems, and environmental monitoring without reconfiguration.

---

## NOTES FOR PATENT ATTORNEY

### Alice/§101 Strategy
The strongest defense against abstract idea rejection is that this is a **signal processing method** — a specific mathematical transformation pipeline producing measurable physical outputs, analogous to FFT, wavelet transforms, or Kalman filtering, all of which have extensive patent history. The explicit exclusion of AI/ML strengthens this: the core mechanism is mathematical transformation, not "applying an algorithm on a computer."

### Key Novelty Arguments
1. **No prior art combines** signal typology → eigenvalue geometry → FTLE/dynamical systems → causal mechanics in a single domain-agnostic pipeline
2. **No prior art achieves** domain-agnostic degradation detection using only eigenvalue decomposition and lookup tables (no training)
3. **The architecture claim** (engines compute, SQL reasons) is a novel system design that enables the domain-agnostic property — this is the inventive step that makes the math work across domains
4. **FTLE ridge detection** applied to multi-sensor industrial/medical degradation (not fluid dynamics) is a novel application of known mathematics to a new problem domain

### Prior Art to Distinguish From
- Nof US 12,050,511: AI-dependent, requires training history, focuses on repair sequencing not detection. Different layer of the stack.
- Standard PCA-based monitoring (Hotelling T², SPE): Uses eigenvalues but requires training on "normal" data. ORTHON doesn't.
- Deep learning prognostics (LSTM, CNN for RUL): Requires labeled failure data. ORTHON doesn't.
- Wavelet/EMD decomposition methods: Signal processing but single-layer, not multi-layer geometric + dynamical + causal.

### Recommended Filing Strategy
- File provisional first ($320 for small entity) to establish priority date
- This buys 12 months before full application required
- Publish paper after provisional is filed — publication then cannot be used as prior art against you
- Purdue patent office can convert provisional → full application if they partner

### What to Keep as Trade Secret (Do NOT Include in Patent)
- Specific engine minimum-sample thresholds
- Specific eigenvalue threshold values in lookup tables
- The exact SQL query structure
- Window size selection algorithms per typology
- The specific DuckDB/engine separation implementation
These implementation details make reproduction difficult even if the method is published. The patent protects the method; trade secret protects the implementation.
