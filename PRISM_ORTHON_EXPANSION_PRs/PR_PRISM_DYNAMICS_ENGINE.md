# PR: Dynamics Engine for PRISM

## Summary

Add a dynamics computation engine to PRISM that calculates Lyapunov exponents, attractor dimensions, and recurrence quantification metrics. This extends the geometric analysis with dynamical systems theory.

---

## Motivation

### The Physics Connection

Current PRISM computes **geometric** properties of the state space:
- Eigenvalue decomposition → coherence, effective dimension
- Hausdorff distance → state divergence
- Energy/momentum → conservation laws

But we're missing **dynamical** properties:
- How sensitive is the system to perturbations?
- How complex is the underlying attractor?
- How predictable is the trajectory?

### The "Birth Certificate" Finding

Early-life geometric fingerprints predict lifespan. Why?

**Hypothesis:** Early-life Lyapunov exponents reflect basin stability. Engines "born" with shallower basins (higher Lyapunov) fail faster because perturbations push them out of stability sooner.

### Multi-Scale Analysis

| Timescale | Data Rate | Current Method | Proposed Extension |
|-----------|-----------|----------------|-------------------|
| Operational | ~1 Hz | Eigenvalue coherence | Lyapunov from slow dynamics |
| Vibration | 12-48 kHz | (none) | Phase space reconstruction + Lyapunov |

---

## Proposed Implementation

### 1. Phase Space Reconstruction

```python
# prism/dynamics/reconstruction.py

def embed_time_series(x: np.ndarray, tau: int, dim: int) -> np.ndarray:
    """
    Takens' embedding theorem: reconstruct attractor from scalar time series.
    
    Parameters
    ----------
    x : array, shape (n_samples,)
        Scalar time series
    tau : int
        Time delay (in samples)
    dim : int
        Embedding dimension
        
    Returns
    -------
    embedded : array, shape (n_samples - (dim-1)*tau, dim)
        Embedded trajectory in reconstructed phase space
    """
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau : i * tau + n]
    return embedded


def optimal_delay(x: np.ndarray, max_tau: int = 100) -> int:
    """
    Estimate optimal time delay using first minimum of mutual information.
    """
    from scipy.stats import entropy
    
    mi = []
    for tau in range(1, max_tau):
        # Discretize into bins
        x1 = np.digitize(x[:-tau], bins=np.linspace(x.min(), x.max(), 20))
        x2 = np.digitize(x[tau:], bins=np.linspace(x.min(), x.max(), 20))
        
        # Joint and marginal distributions
        joint = np.histogram2d(x1, x2, bins=20)[0]
        joint = joint / joint.sum()
        
        px = joint.sum(axis=1)
        py = joint.sum(axis=0)
        
        # Mutual information
        mi_val = entropy(px) + entropy(py) - entropy(joint.flatten())
        mi.append(mi_val)
    
    # First local minimum
    for i in range(1, len(mi) - 1):
        if mi[i] < mi[i-1] and mi[i] < mi[i+1]:
            return i + 1
    
    return max_tau // 4  # fallback


def optimal_embedding_dim(x: np.ndarray, tau: int, max_dim: int = 10) -> int:
    """
    Estimate optimal embedding dimension using false nearest neighbors.
    """
    from scipy.spatial import KDTree
    
    fnn_ratios = []
    
    for dim in range(1, max_dim):
        embedded_d = embed_time_series(x, tau, dim)
        embedded_d1 = embed_time_series(x, tau, dim + 1)
        
        # Find nearest neighbors in d dimensions
        tree = KDTree(embedded_d)
        distances, indices = tree.query(embedded_d, k=2)
        
        # Check if they're still neighbors in d+1 dimensions
        n_false = 0
        for i in range(len(embedded_d1)):
            if indices[i, 1] < len(embedded_d1):
                j = indices[i, 1]
                dist_d = distances[i, 1]
                dist_d1 = np.linalg.norm(embedded_d1[i] - embedded_d1[j])
                
                if dist_d > 0 and (dist_d1 / dist_d) > 10:
                    n_false += 1
        
        fnn_ratios.append(n_false / len(embedded_d))
        
        if fnn_ratios[-1] < 0.01:
            return dim + 1
    
    return max_dim
```

### 2. Lyapunov Exponents

```python
# prism/dynamics/lyapunov.py

def largest_lyapunov_exponent(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    min_tsep: int = None,
    max_iter: int = 1000
) -> float:
    """
    Estimate largest Lyapunov exponent using Rosenstein's algorithm.
    
    Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
    "A practical method for calculating largest Lyapunov exponents 
    from small data sets."
    
    Parameters
    ----------
    x : array
        Time series
    tau : int, optional
        Time delay (auto-computed if None)
    dim : int, optional
        Embedding dimension (auto-computed if None)
    min_tsep : int, optional
        Minimum temporal separation for neighbors
    max_iter : int
        Maximum iterations for divergence tracking
        
    Returns
    -------
    lambda_max : float
        Largest Lyapunov exponent (bits per sample)
        > 0: chaotic
        ≈ 0: periodic/quasiperiodic
        < 0: stable fixed point
    """
    from scipy.spatial import KDTree
    
    # Auto-compute embedding parameters
    if tau is None:
        tau = optimal_delay(x)
    if dim is None:
        dim = optimal_embedding_dim(x, tau)
    if min_tsep is None:
        min_tsep = tau * dim
    
    # Embed
    embedded = embed_time_series(x, tau, dim)
    n_points = len(embedded)
    
    # Find nearest neighbors (excluding temporal neighbors)
    tree = KDTree(embedded)
    
    divergence = np.zeros(max_iter)
    counts = np.zeros(max_iter)
    
    for i in range(n_points - max_iter):
        # Find nearest neighbor outside temporal window
        dists, indices = tree.query(embedded[i], k=n_points)
        
        for j, idx in enumerate(indices):
            if abs(idx - i) > min_tsep:
                nn_idx = idx
                break
        else:
            continue
        
        # Track divergence over time
        for k in range(max_iter):
            if i + k >= n_points or nn_idx + k >= n_points:
                break
            
            dist = np.linalg.norm(embedded[i + k] - embedded[nn_idx + k])
            if dist > 0:
                divergence[k] += np.log(dist)
                counts[k] += 1
    
    # Average divergence curve
    valid = counts > 0
    divergence[valid] /= counts[valid]
    
    # Fit linear region to get Lyapunov exponent
    # Find linear region (typically early part of curve)
    linear_region = min(max_iter // 4, 50)
    valid_range = np.arange(linear_region)[valid[:linear_region]]
    
    if len(valid_range) < 10:
        return np.nan
    
    slope, _ = np.polyfit(valid_range, divergence[valid_range], 1)
    
    return slope / tau  # Convert to per-sample rate


def lyapunov_spectrum(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    n_exponents: int = None
) -> np.ndarray:
    """
    Estimate full Lyapunov spectrum using Wolf's algorithm.
    
    Returns array of exponents [λ₁, λ₂, ..., λₙ] in descending order.
    
    Note: Full spectrum estimation is less reliable than largest exponent.
    Use with caution for dim > 3.
    """
    # Implementation of Wolf et al. (1985) algorithm
    # More complex - can implement later if needed
    
    # For now, return just largest exponent
    lambda_max = largest_lyapunov_exponent(x, tau, dim)
    return np.array([lambda_max])
```

### 3. Attractor Dimension

```python
# prism/dynamics/dimension.py

def correlation_dimension(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    r_range: tuple = (0.01, 1.0),
    n_points: int = 20
) -> float:
    """
    Estimate correlation dimension using Grassberger-Procaccia algorithm.
    
    The correlation dimension D₂ characterizes the complexity of the attractor:
    - D₂ ≈ 1: limit cycle
    - D₂ ≈ 2: torus
    - D₂ non-integer: strange attractor (chaos)
    
    Parameters
    ----------
    x : array
        Time series
    tau, dim : int, optional
        Embedding parameters
    r_range : tuple
        Range of radii for correlation sum (fraction of std)
    n_points : int
        Number of radius values to compute
        
    Returns
    -------
    D2 : float
        Correlation dimension estimate
    """
    if tau is None:
        tau = optimal_delay(x)
    if dim is None:
        dim = optimal_embedding_dim(x, tau)
    
    embedded = embed_time_series(x, tau, dim)
    n = len(embedded)
    
    # Normalize
    std = np.std(embedded)
    radii = np.logspace(np.log10(r_range[0] * std), 
                        np.log10(r_range[1] * std), 
                        n_points)
    
    # Correlation sum C(r) = fraction of pairs within distance r
    from scipy.spatial.distance import pdist
    distances = pdist(embedded)
    
    C = np.zeros(len(radii))
    for i, r in enumerate(radii):
        C[i] = np.mean(distances < r)
    
    # Fit log(C) vs log(r) in scaling region
    valid = (C > 0.001) & (C < 0.5)  # Avoid edge effects
    if valid.sum() < 5:
        return np.nan
    
    log_r = np.log(radii[valid])
    log_C = np.log(C[valid])
    
    slope, _ = np.polyfit(log_r, log_C, 1)
    
    return slope


def kaplan_yorke_dimension(lyapunov_spectrum: np.ndarray) -> float:
    """
    Estimate attractor dimension from Lyapunov spectrum.
    
    D_KY = j + (λ₁ + ... + λⱼ) / |λⱼ₊₁|
    
    where j is largest index such that sum of first j exponents is positive.
    """
    cumsum = np.cumsum(lyapunov_spectrum)
    
    # Find j where cumsum becomes negative
    j = np.argmax(cumsum < 0)
    if j == 0:
        if cumsum[0] >= 0:
            return len(lyapunov_spectrum)  # All positive
        else:
            return 0  # All negative
    
    return j + cumsum[j-1] / abs(lyapunov_spectrum[j])
```

### 4. Recurrence Quantification Analysis

```python
# prism/dynamics/recurrence.py

def recurrence_matrix(
    x: np.ndarray,
    tau: int = None,
    dim: int = None,
    threshold: float = None
) -> np.ndarray:
    """
    Compute recurrence matrix R[i,j] = 1 if ||x(i) - x(j)|| < threshold.
    """
    if tau is None:
        tau = optimal_delay(x)
    if dim is None:
        dim = optimal_embedding_dim(x, tau)
    
    embedded = embed_time_series(x, tau, dim)
    
    if threshold is None:
        # Use 10% of maximum distance
        from scipy.spatial.distance import pdist
        threshold = 0.1 * np.max(pdist(embedded))
    
    from scipy.spatial.distance import cdist
    distances = cdist(embedded, embedded)
    
    return (distances < threshold).astype(int)


def rqa_metrics(R: np.ndarray) -> dict:
    """
    Compute Recurrence Quantification Analysis metrics.
    
    Returns
    -------
    dict with:
        recurrence_rate : float
            Density of recurrence points
        determinism : float
            Proportion of recurrent points forming diagonal lines
            High = predictable, Low = chaotic
        avg_diagonal_length : float
            Average length of diagonal structures
            Related to prediction horizon
        entropy : float
            Shannon entropy of diagonal length distribution
            Complexity of deterministic structure
        laminarity : float
            Proportion of recurrent points in vertical lines
            High = intermittency/laminar phases
        trapping_time : float
            Average length of vertical structures
            Duration of laminar states
    """
    n = len(R)
    
    # Recurrence rate
    rr = np.sum(R) / (n * n)
    
    # Find diagonal lines (exclude main diagonal)
    diag_lengths = []
    for k in range(-n+2, n-1):
        if k == 0:
            continue
        diag = np.diag(R, k)
        
        # Find runs of 1s
        in_run = False
        run_length = 0
        for val in diag:
            if val == 1:
                if in_run:
                    run_length += 1
                else:
                    in_run = True
                    run_length = 1
            else:
                if in_run and run_length >= 2:
                    diag_lengths.append(run_length)
                in_run = False
                run_length = 0
        if in_run and run_length >= 2:
            diag_lengths.append(run_length)
    
    # Determinism
    if len(diag_lengths) > 0:
        det = sum(diag_lengths) / max(1, np.sum(R) - n)  # exclude main diagonal
        avg_diag = np.mean(diag_lengths)
        
        # Entropy of diagonal length distribution
        hist, _ = np.histogram(diag_lengths, bins=range(2, max(diag_lengths)+2))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entr = -np.sum(hist * np.log(hist))
    else:
        det = 0
        avg_diag = 0
        entr = 0
    
    # Find vertical lines (for laminarity)
    vert_lengths = []
    for col in range(n):
        in_run = False
        run_length = 0
        for row in range(n):
            if R[row, col] == 1:
                if in_run:
                    run_length += 1
                else:
                    in_run = True
                    run_length = 1
            else:
                if in_run and run_length >= 2:
                    vert_lengths.append(run_length)
                in_run = False
                run_length = 0
        if in_run and run_length >= 2:
            vert_lengths.append(run_length)
    
    # Laminarity
    if len(vert_lengths) > 0:
        lam = sum(vert_lengths) / max(1, np.sum(R))
        trap = np.mean(vert_lengths)
    else:
        lam = 0
        trap = 0
    
    return {
        'recurrence_rate': rr,
        'determinism': det,
        'avg_diagonal_length': avg_diag,
        'entropy': entr,
        'laminarity': lam,
        'trapping_time': trap
    }
```

### 5. Main Dynamics Engine

```python
# prism/dynamics/engine.py

import polars as pl
import numpy as np
from typing import Optional
from pathlib import Path

from .reconstruction import optimal_delay, optimal_embedding_dim, embed_time_series
from .lyapunov import largest_lyapunov_exponent
from .dimension import correlation_dimension
from .recurrence import recurrence_matrix, rqa_metrics


class DynamicsEngine:
    """
    Compute dynamical systems metrics for time series data.
    
    Complements the geometric eigenvalue analysis with:
    - Lyapunov exponents (stability/chaos)
    - Attractor dimension (complexity)
    - Recurrence quantification (predictability)
    """
    
    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 10,
        embedding_dim: Optional[int] = None,
        time_delay: Optional[int] = None
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
    
    def compute_for_signal(
        self,
        x: np.ndarray,
        entity_id: str
    ) -> pl.DataFrame:
        """
        Compute dynamics metrics for a single signal using sliding windows.
        """
        n = len(x)
        results = []
        
        for i in range(0, n - self.window_size, self.step_size):
            window = x[i : i + self.window_size]
            
            # Get embedding parameters
            tau = self.time_delay or optimal_delay(window)
            dim = self.embedding_dim or optimal_embedding_dim(window, tau)
            
            # Lyapunov
            try:
                lyap = largest_lyapunov_exponent(window, tau, dim)
            except:
                lyap = np.nan
            
            # Correlation dimension
            try:
                corr_dim = correlation_dimension(window, tau, dim)
            except:
                corr_dim = np.nan
            
            # RQA metrics
            try:
                R = recurrence_matrix(window, tau, dim)
                rqa = rqa_metrics(R)
            except:
                rqa = {
                    'recurrence_rate': np.nan,
                    'determinism': np.nan,
                    'avg_diagonal_length': np.nan,
                    'entropy': np.nan,
                    'laminarity': np.nan,
                    'trapping_time': np.nan
                }
            
            results.append({
                'entity_id': entity_id,
                'observation_idx': i + self.window_size // 2,
                'window_start': i,
                'window_end': i + self.window_size,
                'lyapunov_max': lyap,
                'correlation_dim': corr_dim,
                'embedding_dim': dim,
                'time_delay': tau,
                **rqa
            })
        
        return pl.DataFrame(results)
    
    def compute_for_entity(
        self,
        signals: dict[str, np.ndarray],
        entity_id: str
    ) -> pl.DataFrame:
        """
        Compute dynamics metrics for all signals of an entity.
        
        Parameters
        ----------
        signals : dict
            {signal_name: time_series_array}
        entity_id : str
            Entity identifier
            
        Returns
        -------
        DataFrame with dynamics metrics per window, aggregated across signals
        """
        all_results = []
        
        for signal_name, x in signals.items():
            df = self.compute_for_signal(x, entity_id)
            df = df.with_columns(pl.lit(signal_name).alias('signal_name'))
            all_results.append(df)
        
        if not all_results:
            return pl.DataFrame()
        
        combined = pl.concat(all_results)
        
        # Aggregate across signals per window
        aggregated = combined.group_by(['entity_id', 'observation_idx']).agg([
            pl.col('lyapunov_max').mean().alias('lyapunov_max'),
            pl.col('lyapunov_max').max().alias('lyapunov_max_signal'),
            pl.col('lyapunov_max').std().alias('lyapunov_spread'),
            pl.col('correlation_dim').mean().alias('correlation_dim'),
            pl.col('recurrence_rate').mean().alias('recurrence_rate'),
            pl.col('determinism').mean().alias('determinism'),
            pl.col('avg_diagonal_length').mean().alias('avg_diagonal_length'),
            pl.col('entropy').mean().alias('rqa_entropy'),
            pl.col('laminarity').mean().alias('laminarity'),
            pl.col('trapping_time').mean().alias('trapping_time'),
        ])
        
        return aggregated.sort(['entity_id', 'observation_idx'])
    
    def to_parquet(self, df: pl.DataFrame, path: Path):
        """Save dynamics results to parquet."""
        df.write_parquet(path)


def compute_dynamics(
    data_path: Path,
    output_path: Path,
    window_size: int = 100,
    step_size: int = 10
) -> pl.DataFrame:
    """
    Main entry point: compute dynamics metrics for all entities.
    
    Parameters
    ----------
    data_path : Path
        Path to data.parquet (from PRISM fetch stage)
    output_path : Path
        Path to write dynamics.parquet
        
    Returns
    -------
    DataFrame with dynamics metrics
    """
    engine = DynamicsEngine(window_size=window_size, step_size=step_size)
    
    # Load data
    data = pl.read_parquet(data_path)
    
    # Get unique entities
    entities = data['entity_id'].unique().to_list()
    
    all_results = []
    
    for entity_id in entities:
        entity_data = data.filter(pl.col('entity_id') == entity_id)
        
        # Extract signals (exclude metadata columns)
        signal_cols = [c for c in entity_data.columns 
                      if c not in ['entity_id', 'observation_idx', 'timestamp']]
        
        signals = {
            col: entity_data[col].to_numpy() 
            for col in signal_cols
        }
        
        result = engine.compute_for_entity(signals, entity_id)
        all_results.append(result)
    
    combined = pl.concat(all_results)
    engine.to_parquet(combined, output_path)
    
    return combined
```

---

## Output Schema

### dynamics.parquet

| Column | Type | Description |
|--------|------|-------------|
| entity_id | str | Entity identifier |
| observation_idx | int | Center of analysis window |
| lyapunov_max | float | Largest Lyapunov exponent (mean across signals) |
| lyapunov_max_signal | float | Max Lyapunov across all signals |
| lyapunov_spread | float | Std of Lyapunov across signals |
| correlation_dim | float | Correlation dimension |
| recurrence_rate | float | RQA recurrence rate |
| determinism | float | RQA determinism |
| avg_diagonal_length | float | RQA average diagonal length |
| rqa_entropy | float | RQA entropy |
| laminarity | float | RQA laminarity |
| trapping_time | float | RQA trapping time |

---

## Integration with Existing PRISM

```python
# prism/pipeline.py (modified)

def run_pipeline(data_path: Path, output_dir: Path):
    """
    Full PRISM pipeline.
    """
    # Existing stages
    data = fetch_stage(data_path, output_dir / 'data.parquet')
    vector = characterize_stage(data, output_dir / 'vector.parquet')
    geometry = geometry_stage(vector, output_dir / 'geometry.parquet')
    dynamics_old = dynamics_stage(geometry, output_dir / 'dynamics.parquet')  # rename
    physics = physics_stage(dynamics_old, output_dir / 'physics.parquet')
    
    # NEW: Dynamical systems stage
    from prism.dynamics.engine import compute_dynamics
    dynamics_new = compute_dynamics(
        output_dir / 'data.parquet',
        output_dir / 'dynamics_systems.parquet',  # new file
        window_size=100,
        step_size=10
    )
    
    return {
        'data': data,
        'vector': vector,
        'geometry': geometry,
        'dynamics': dynamics_old,
        'physics': physics,
        'dynamics_systems': dynamics_new  # NEW
    }
```

---

## Validation Approach

### 1. Synthetic Test Cases

```python
def test_lorenz_attractor():
    """Lorenz system should show positive Lyapunov ~ 0.9"""
    x = generate_lorenz(n=5000)
    lyap = largest_lyapunov_exponent(x[:, 0])
    assert 0.8 < lyap < 1.0

def test_periodic_signal():
    """Periodic signal should show Lyapunov ~ 0"""
    x = np.sin(np.linspace(0, 100*np.pi, 5000))
    lyap = largest_lyapunov_exponent(x)
    assert abs(lyap) < 0.1

def test_white_noise():
    """White noise should show high Lyapunov"""
    x = np.random.randn(5000)
    lyap = largest_lyapunov_exponent(x)
    assert lyap > 1.0
```

### 2. CWRU Bearing Validation

- **Healthy bearings:** Expect low Lyapunov, high determinism
- **Faulty bearings:** Expect higher Lyapunov, lower determinism
- **Inner vs outer race:** Different recurrence patterns

### 3. C-MAPSS Correlation

- Compute Lyapunov trajectory for each engine
- Correlate early-life Lyapunov with RUL
- Expected: r < -0.3 (higher Lyapunov → shorter life)

---

## Performance Considerations

### Computational Cost

| Operation | Complexity | Mitigation |
|-----------|------------|------------|
| Embedding | O(n) | Fast |
| Lyapunov | O(n²) | Limit window size |
| RQA | O(n²) | Subsample for large n |
| Correlation dim | O(n²) | Subsample |

### Recommended Settings

| Data Type | Window Size | Step Size | Notes |
|-----------|-------------|-----------|-------|
| C-MAPSS (~200 pts) | 50-100 | 10 | Full computation |
| Vibration (10k+ pts) | 1000 | 100 | Subsample within window |
| Real-time | 500 | 50 | Balance speed/accuracy |

---

## Files to Create

```
prism/dynamics/
├── __init__.py
├── reconstruction.py      # Takens embedding
├── lyapunov.py           # Lyapunov exponents
├── dimension.py          # Attractor dimension
├── recurrence.py         # RQA metrics
└── engine.py             # Main orchestration

tests/dynamics/
├── test_reconstruction.py
├── test_lyapunov.py
├── test_dimension.py
├── test_recurrence.py
└── test_integration.py
```

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "polars>=0.20",
    "scipy>=1.10",  # For KDTree, pdist
]
```

No new external dependencies required.

---

## Success Criteria

1. **Synthetic tests pass** (Lorenz, periodic, noise)
2. **CWRU differentiation** (healthy vs faulty)
3. **C-MAPSS correlation** (Lyapunov vs RUL, r > 0.3)
4. **Performance** (<1 sec per entity for C-MAPSS scale)
5. **Integration** (dynamics.parquet feeds ORTHON)

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 1 | Core algorithms (embedding, Lyapunov) | 3-4 days |
| 2 | RQA implementation | 2 days |
| 3 | Engine integration | 2 days |
| 4 | Testing + validation | 3 days |
| 5 | Performance tuning | 1-2 days |

---

## References

1. Takens, F. (1981). "Detecting strange attractors in turbulence"
2. Rosenstein, M. T., et al. (1993). "A practical method for calculating largest Lyapunov exponents"
3. Grassberger, P. & Procaccia, I. (1983). "Characterization of strange attractors"
4. Marwan, N., et al. (2007). "Recurrence plots for the analysis of complex systems"
5. Kantz, H. & Schreiber, T. (2004). "Nonlinear Time Series Analysis"

---

## Notes

This implementation prioritizes:
1. **Correctness** over speed (can optimize later)
2. **Interpretability** over black-box metrics
3. **Integration** with existing PRISM architecture

The goal is to extend PRISM's "geometry of relationships" with "dynamics of stability" - same philosophy, deeper physics.
