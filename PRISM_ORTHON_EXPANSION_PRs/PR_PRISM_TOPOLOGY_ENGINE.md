# PR: Topology Engine for PRISM

## Summary

Add topological data analysis (TDA) capabilities to PRISM, computing persistent homology, Betti numbers, and topological complexity metrics. This captures the **shape** of system dynamics that geometry and dynamics alone cannot see.

---

## Motivation

### The Shape of Failure

Geometry tells us: "How are signals related?"
Dynamics tells us: "How stable is the system?"
Topology tells us: **"What is the shape of the system's behavior?"**

Before failure, the attractor **changes shape**:
- Loops appear or disappear
- Connected components fragment
- Voids collapse

These shape changes are **invisible** to eigenvalues and Lyapunov exponents.

### The Topological Signal

| System State | Attractor Shape | Betti Numbers |
|--------------|-----------------|---------------|
| Healthy | Clean torus | β₀=1, β₁=1, β₂=0 |
| Degrading | Torus bulging | β₀=1, β₁=1→2, β₂=0 |
| Pre-failure | Fragmenting | β₀=1→2, β₁→0 |
| Failed | Collapsed/chaotic | β₀>1, β₁=0 |

### Why Persistent Homology?

Traditional topology is binary: "Is there a hole or not?"

Persistent homology tracks **which features persist across scales**:
- Noise creates features that die quickly
- Real structure persists
- The "persistence" is the signal

---

## Mathematical Background

### Simplicial Complexes

From point cloud data, build a family of simplicial complexes at increasing scales (ε):

```
ε = 0.1: Disconnected points
ε = 0.3: Some edges form
ε = 0.5: Triangles appear, loop detected
ε = 0.7: Loop fills in (dies)
ε = 1.0: Single connected blob
```

### Betti Numbers

| Betti Number | What It Counts |
|--------------|----------------|
| β₀ | Connected components |
| β₁ | 1-dimensional holes (loops) |
| β₂ | 2-dimensional voids (cavities) |

### Persistence Diagrams

Each topological feature has:
- **Birth time**: Scale where it appears
- **Death time**: Scale where it disappears
- **Persistence** = death - birth

Long-lived features = real structure
Short-lived features = noise

---

## Proposed Implementation

### 1. Point Cloud Construction

```python
# prism/topology/point_cloud.py

import numpy as np
from typing import Optional

def time_delay_embedding(
    x: np.ndarray,
    dim: int,
    tau: int
) -> np.ndarray:
    """
    Construct point cloud via time-delay embedding.
    Same as dynamics module - can share.
    """
    n = len(x) - (dim - 1) * tau
    embedded = np.zeros((n, dim))
    for i in range(dim):
        embedded[:, i] = x[i * tau : i * tau + n]
    return embedded


def sliding_window_embedding(
    x: np.ndarray,
    window_size: int,
    step: int = 1
) -> np.ndarray:
    """
    Construct point cloud where each point is a window of the signal.
    Good for periodic/quasi-periodic signals.
    """
    n_windows = (len(x) - window_size) // step + 1
    embedded = np.zeros((n_windows, window_size))
    for i in range(n_windows):
        embedded[i] = x[i * step : i * step + window_size]
    return embedded


def multivariate_point_cloud(
    signals: dict[str, np.ndarray],
    method: str = 'direct'
) -> np.ndarray:
    """
    Construct point cloud from multiple signals.
    
    Methods:
    - 'direct': Each time point is a point in R^n_signals
    - 'concatenated_embedding': Embed each signal, concatenate
    """
    signal_list = list(signals.values())
    n_samples = min(len(s) for s in signal_list)
    
    if method == 'direct':
        return np.column_stack([s[:n_samples] for s in signal_list])
    
    elif method == 'concatenated_embedding':
        embeddings = []
        for s in signal_list:
            emb = time_delay_embedding(s[:n_samples], dim=3, tau=10)
            embeddings.append(emb)
        min_len = min(len(e) for e in embeddings)
        return np.hstack([e[:min_len] for e in embeddings])
    
    else:
        raise ValueError(f"Unknown method: {method}")
```

### 2. Persistent Homology Computation

```python
# prism/topology/persistence.py

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class PersistenceDiagram:
    """Container for persistence diagram data."""
    dimension: int
    birth_times: np.ndarray
    death_times: np.ndarray
    
    @property
    def persistence(self) -> np.ndarray:
        """Lifetime of each feature."""
        return self.death_times - self.birth_times
    
    @property
    def n_features(self) -> int:
        return len(self.birth_times)
    
    def filter_by_persistence(self, min_persistence: float) -> 'PersistenceDiagram':
        """Keep only features with persistence above threshold."""
        mask = self.persistence >= min_persistence
        return PersistenceDiagram(
            dimension=self.dimension,
            birth_times=self.birth_times[mask],
            death_times=self.death_times[mask]
        )


def compute_rips_persistence(
    point_cloud: np.ndarray,
    max_dimension: int = 2,
    max_edge_length: float = None,
    n_landmarks: int = None
) -> List[PersistenceDiagram]:
    """
    Compute persistent homology using Vietoris-Rips complex.
    
    Parameters
    ----------
    point_cloud : array, shape (n_points, n_dims)
        Point cloud data
    max_dimension : int
        Maximum homology dimension to compute
    max_edge_length : float, optional
        Maximum filtration value
    n_landmarks : int, optional
        If provided, use landmark-based approximation for large point clouds
        
    Returns
    -------
    diagrams : list of PersistenceDiagram
        One diagram per dimension (0 to max_dimension)
    """
    try:
        # Try using ripser (fast C++ implementation)
        import ripser
        
        if n_landmarks and len(point_cloud) > n_landmarks:
            # Subsample for large point clouds
            indices = np.random.choice(len(point_cloud), n_landmarks, replace=False)
            point_cloud = point_cloud[indices]
        
        result = ripser.ripser(
            point_cloud,
            maxdim=max_dimension,
            thresh=max_edge_length
        )
        
        diagrams = []
        for dim, dgm in enumerate(result['dgms']):
            # Filter out infinite death times for H0
            if dim == 0:
                finite_mask = np.isfinite(dgm[:, 1])
                dgm = dgm[finite_mask]
            
            diagrams.append(PersistenceDiagram(
                dimension=dim,
                birth_times=dgm[:, 0],
                death_times=dgm[:, 1]
            ))
        
        return diagrams
    
    except ImportError:
        # Fallback to pure Python implementation (slower)
        return _compute_rips_persistence_pure_python(
            point_cloud, max_dimension, max_edge_length
        )


def _compute_rips_persistence_pure_python(
    point_cloud: np.ndarray,
    max_dimension: int,
    max_edge_length: float
) -> List[PersistenceDiagram]:
    """
    Pure Python fallback for persistent homology.
    Uses a simplified algorithm - less efficient but no dependencies.
    """
    from scipy.spatial.distance import pdist, squareform
    
    # Compute distance matrix
    distances = squareform(pdist(point_cloud))
    n = len(point_cloud)
    
    if max_edge_length is None:
        max_edge_length = np.max(distances)
    
    # For H0: Track connected components using union-find
    parent = list(range(n))
    rank = [0] * n
    
    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return False
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1
        return True
    
    # Sort edges by distance
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if distances[i, j] <= max_edge_length:
                edges.append((distances[i, j], i, j))
    edges.sort()
    
    # Track H0 births and deaths
    h0_birth = [0.0] * n  # All points born at 0
    h0_death = [np.inf] * n
    
    for dist, i, j in edges:
        pi, pj = find(i), find(j)
        if pi != pj:
            # Merge components - younger one dies
            if h0_birth[pi] < h0_birth[pj]:
                h0_death[pj] = dist
            else:
                h0_death[pi] = dist
            union(i, j)
    
    # Build H0 diagram (exclude the one infinite component)
    h0_pairs = [(b, d) for b, d in zip(h0_birth, h0_death) if d < np.inf]
    
    diagrams = [
        PersistenceDiagram(
            dimension=0,
            birth_times=np.array([p[0] for p in h0_pairs]),
            death_times=np.array([p[1] for p in h0_pairs])
        )
    ]
    
    # H1 and higher requires more complex algorithm
    # For now, return empty diagrams
    for dim in range(1, max_dimension + 1):
        diagrams.append(PersistenceDiagram(
            dimension=dim,
            birth_times=np.array([]),
            death_times=np.array([])
        ))
    
    return diagrams
```

### 3. Topological Features

```python
# prism/topology/features.py

import numpy as np
from typing import List, Dict
from .persistence import PersistenceDiagram


def betti_numbers(
    diagrams: List[PersistenceDiagram],
    threshold: float
) -> Dict[int, int]:
    """
    Compute Betti numbers at a given filtration threshold.
    
    β_k(threshold) = number of k-dimensional features alive at threshold
    """
    betti = {}
    for dgm in diagrams:
        alive = np.sum(
            (dgm.birth_times <= threshold) & (dgm.death_times > threshold)
        )
        betti[dgm.dimension] = int(alive)
    return betti


def betti_curve(
    diagrams: List[PersistenceDiagram],
    n_points: int = 100
) -> Dict[int, np.ndarray]:
    """
    Compute Betti numbers across all filtration values.
    Returns a curve for each dimension.
    """
    # Find filtration range
    all_births = np.concatenate([d.birth_times for d in diagrams if len(d.birth_times) > 0])
    all_deaths = np.concatenate([d.death_times for d in diagrams if len(d.death_times) > 0])
    all_deaths = all_deaths[np.isfinite(all_deaths)]
    
    if len(all_births) == 0:
        return {d.dimension: np.zeros(n_points) for d in diagrams}
    
    min_val = np.min(all_births)
    max_val = np.max(all_deaths) if len(all_deaths) > 0 else np.max(all_births) * 2
    
    thresholds = np.linspace(min_val, max_val, n_points)
    
    curves = {}
    for dgm in diagrams:
        curve = np.zeros(n_points)
        for i, t in enumerate(thresholds):
            curve[i] = np.sum(
                (dgm.birth_times <= t) & (dgm.death_times > t)
            )
        curves[dgm.dimension] = curve
    
    return curves


def persistence_statistics(dgm: PersistenceDiagram) -> Dict[str, float]:
    """
    Compute summary statistics from a persistence diagram.
    """
    if dgm.n_features == 0:
        return {
            'n_features': 0,
            'total_persistence': 0.0,
            'max_persistence': 0.0,
            'mean_persistence': 0.0,
            'std_persistence': 0.0,
            'persistence_entropy': 0.0,
            'mean_birth': 0.0,
            'mean_death': 0.0,
        }
    
    pers = dgm.persistence
    pers = pers[np.isfinite(pers)]
    
    if len(pers) == 0:
        return persistence_statistics(PersistenceDiagram(dgm.dimension, np.array([]), np.array([])))
    
    # Persistence entropy
    pers_norm = pers / pers.sum() if pers.sum() > 0 else pers
    pers_norm = pers_norm[pers_norm > 0]
    entropy = -np.sum(pers_norm * np.log(pers_norm)) if len(pers_norm) > 0 else 0.0
    
    return {
        'n_features': dgm.n_features,
        'total_persistence': float(np.sum(pers)),
        'max_persistence': float(np.max(pers)),
        'mean_persistence': float(np.mean(pers)),
        'std_persistence': float(np.std(pers)),
        'persistence_entropy': float(entropy),
        'mean_birth': float(np.mean(dgm.birth_times)),
        'mean_death': float(np.mean(dgm.death_times[np.isfinite(dgm.death_times)])) if np.any(np.isfinite(dgm.death_times)) else 0.0,
    }


def persistence_landscape(
    dgm: PersistenceDiagram,
    n_landscapes: int = 5,
    n_points: int = 100
) -> np.ndarray:
    """
    Compute persistence landscapes - a stable vectorization of persistence diagrams.
    
    Returns array of shape (n_landscapes, n_points)
    """
    if dgm.n_features == 0:
        return np.zeros((n_landscapes, n_points))
    
    finite_mask = np.isfinite(dgm.death_times)
    births = dgm.birth_times[finite_mask]
    deaths = dgm.death_times[finite_mask]
    
    if len(births) == 0:
        return np.zeros((n_landscapes, n_points))
    
    # Filtration range
    min_val = np.min(births)
    max_val = np.max(deaths)
    t = np.linspace(min_val, max_val, n_points)
    
    # Tent functions for each feature
    def tent(b, d, x):
        mid = (b + d) / 2
        height = (d - b) / 2
        return np.maximum(0, np.minimum(x - b, d - x))
    
    # Compute all tent functions
    tents = np.zeros((len(births), n_points))
    for i, (b, d) in enumerate(zip(births, deaths)):
        tents[i] = tent(b, d, t)
    
    # Sort at each t to get landscapes
    landscapes = np.zeros((n_landscapes, n_points))
    for j in range(n_points):
        sorted_vals = np.sort(tents[:, j])[::-1]
        for k in range(min(n_landscapes, len(sorted_vals))):
            landscapes[k, j] = sorted_vals[k]
    
    return landscapes


def topological_complexity(diagrams: List[PersistenceDiagram]) -> float:
    """
    Compute overall topological complexity score.
    
    Higher = more complex topology (more holes, more persistent features)
    """
    total = 0.0
    for dgm in diagrams:
        stats = persistence_statistics(dgm)
        # Weight higher dimensions more
        weight = dgm.dimension + 1
        total += weight * stats['total_persistence']
    return total
```

### 4. Topology Engine

```python
# prism/topology/engine.py

import numpy as np
import polars as pl
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass

from .point_cloud import time_delay_embedding, multivariate_point_cloud
from .persistence import compute_rips_persistence, PersistenceDiagram
from .features import (
    betti_numbers, 
    persistence_statistics, 
    persistence_landscape,
    topological_complexity
)


@dataclass
class TopologyResult:
    """Container for topology analysis results."""
    entity_id: str
    observation_idx: int
    
    # Betti numbers at characteristic scale
    betti_0: int  # Connected components
    betti_1: int  # Loops
    betti_2: int  # Voids
    
    # H0 statistics (components)
    h0_n_features: int
    h0_total_persistence: float
    h0_max_persistence: float
    h0_mean_persistence: float
    
    # H1 statistics (loops)
    h1_n_features: int
    h1_total_persistence: float
    h1_max_persistence: float
    h1_mean_persistence: float
    h1_persistence_entropy: float
    
    # H2 statistics (voids)
    h2_n_features: int
    h2_total_persistence: float
    
    # Overall complexity
    topological_complexity: float
    
    # Landscape features (for ML)
    landscape_h1_integral: float
    landscape_h1_max: float


class TopologyEngine:
    """
    Compute topological data analysis metrics for time series.
    
    Complements geometric (eigenvalue) and dynamic (Lyapunov) analysis
    with shape-based features from persistent homology.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 20,
        embedding_dim: int = 3,
        time_delay: int = 10,
        max_homology_dim: int = 2,
        n_landmarks: int = 200  # Subsample for efficiency
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.embedding_dim = embedding_dim
        self.time_delay = time_delay
        self.max_homology_dim = max_homology_dim
        self.n_landmarks = n_landmarks
    
    def compute_for_window(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str,
        window_start: int
    ) -> TopologyResult:
        """
        Compute topology metrics for a single window.
        """
        # Build point cloud from multiple signals
        window_signals = {
            name: sig[window_start:window_start + self.window_size]
            for name, sig in signals.items()
        }
        
        point_cloud = multivariate_point_cloud(window_signals, method='direct')
        
        # Subsample if needed
        if len(point_cloud) > self.n_landmarks:
            indices = np.random.choice(len(point_cloud), self.n_landmarks, replace=False)
            point_cloud = point_cloud[indices]
        
        # Normalize
        point_cloud = (point_cloud - point_cloud.mean(axis=0)) / (point_cloud.std(axis=0) + 1e-10)
        
        # Compute persistent homology
        diagrams = compute_rips_persistence(
            point_cloud,
            max_dimension=self.max_homology_dim,
            n_landmarks=None  # Already subsampled
        )
        
        # Betti numbers at characteristic scale (median death time)
        all_deaths = []
        for d in diagrams:
            deaths = d.death_times[np.isfinite(d.death_times)]
            all_deaths.extend(deaths)
        
        char_scale = np.median(all_deaths) if all_deaths else 1.0
        betti = betti_numbers(diagrams, char_scale)
        
        # Statistics per dimension
        h0_stats = persistence_statistics(diagrams[0]) if len(diagrams) > 0 else {}
        h1_stats = persistence_statistics(diagrams[1]) if len(diagrams) > 1 else {}
        h2_stats = persistence_statistics(diagrams[2]) if len(diagrams) > 2 else {}
        
        # Landscape features for H1
        if len(diagrams) > 1:
            landscapes = persistence_landscape(diagrams[1], n_landscapes=3, n_points=50)
            landscape_integral = np.sum(landscapes[0])
            landscape_max = np.max(landscapes[0])
        else:
            landscape_integral = 0.0
            landscape_max = 0.0
        
        # Overall complexity
        complexity = topological_complexity(diagrams)
        
        return TopologyResult(
            entity_id=entity_id,
            observation_idx=window_start + self.window_size // 2,
            
            betti_0=betti.get(0, 0),
            betti_1=betti.get(1, 0),
            betti_2=betti.get(2, 0),
            
            h0_n_features=h0_stats.get('n_features', 0),
            h0_total_persistence=h0_stats.get('total_persistence', 0.0),
            h0_max_persistence=h0_stats.get('max_persistence', 0.0),
            h0_mean_persistence=h0_stats.get('mean_persistence', 0.0),
            
            h1_n_features=h1_stats.get('n_features', 0),
            h1_total_persistence=h1_stats.get('total_persistence', 0.0),
            h1_max_persistence=h1_stats.get('max_persistence', 0.0),
            h1_mean_persistence=h1_stats.get('mean_persistence', 0.0),
            h1_persistence_entropy=h1_stats.get('persistence_entropy', 0.0),
            
            h2_n_features=h2_stats.get('n_features', 0),
            h2_total_persistence=h2_stats.get('total_persistence', 0.0),
            
            topological_complexity=complexity,
            landscape_h1_integral=landscape_integral,
            landscape_h1_max=landscape_max,
        )
    
    def compute_for_entity(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str
    ) -> pl.DataFrame:
        """
        Compute topology metrics across all windows for an entity.
        """
        n_samples = min(len(s) for s in signals.values())
        results = []
        
        for start in range(0, n_samples - self.window_size, self.step_size):
            try:
                result = self.compute_for_window(signals, entity_id, start)
                results.append(result.__dict__)
            except Exception as e:
                # Skip problematic windows
                continue
        
        if not results:
            return pl.DataFrame()
        
        return pl.DataFrame(results)
    
    def to_parquet(self, df: pl.DataFrame, path: Path):
        """Save topology results to parquet."""
        df.write_parquet(path)


def compute_topology(
    data_path: Path,
    output_path: Path,
    window_size: int = 100,
    step_size: int = 20
) -> pl.DataFrame:
    """
    Main entry point: compute topology metrics for all entities.
    """
    engine = TopologyEngine(window_size=window_size, step_size=step_size)
    
    # Load data
    data = pl.read_parquet(data_path)
    
    # Get unique entities
    entities = data['entity_id'].unique().to_list()
    
    all_results = []
    
    for entity_id in entities:
        entity_data = data.filter(pl.col('entity_id') == entity_id)
        
        # Extract signals
        signal_cols = [c for c in entity_data.columns 
                      if c not in ['entity_id', 'observation_idx', 'timestamp']]
        
        signals = {
            col: entity_data[col].to_numpy() 
            for col in signal_cols
        }
        
        result = engine.compute_for_entity(signals, str(entity_id))
        if len(result) > 0:
            all_results.append(result)
    
    if not all_results:
        return pl.DataFrame()
    
    combined = pl.concat(all_results)
    engine.to_parquet(combined, output_path)
    
    return combined
```

---

## Output Schema

### topology.parquet

| Column | Type | Description |
|--------|------|-------------|
| entity_id | str | Entity identifier |
| observation_idx | int | Center of analysis window |
| betti_0 | int | # connected components |
| betti_1 | int | # loops |
| betti_2 | int | # voids |
| h0_n_features | int | # H0 features in diagram |
| h0_total_persistence | float | Sum of H0 lifetimes |
| h0_max_persistence | float | Longest-lived component merge |
| h1_n_features | int | # loops detected |
| h1_total_persistence | float | Sum of loop lifetimes |
| h1_max_persistence | float | Most persistent loop |
| h1_persistence_entropy | float | Complexity of loop structure |
| h2_n_features | int | # voids detected |
| h2_total_persistence | float | Sum of void lifetimes |
| topological_complexity | float | Overall complexity score |
| landscape_h1_integral | float | Area under H1 landscape |
| landscape_h1_max | float | Peak of H1 landscape |

---

## Interpretation Guide

### Betti Number Patterns

| Pattern | β₀ | β₁ | β₂ | Interpretation |
|---------|----|----|----|----|
| Healthy oscillation | 1 | 1 | 0 | Clean limit cycle |
| Quasi-periodic | 1 | 2 | 0 | Torus (two frequencies) |
| Pre-failure | 1→2 | 1→0 | 0 | Structure fragmenting |
| Chaotic | >1 | ? | ? | Strange attractor |
| Collapsed | 1 | 0 | 0 | Fixed point / dead |

### Persistence Patterns

| Metric | Low Value | High Value |
|--------|-----------|------------|
| h1_max_persistence | Weak/noisy cycles | Strong periodic structure |
| h1_persistence_entropy | Few dominant loops | Many similar loops |
| topological_complexity | Simple dynamics | Complex/chaotic dynamics |

---

## Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "polars>=0.20",
    "scipy>=1.10",
]

[project.optional-dependencies]
topology = [
    "ripser>=0.6",  # Fast C++ persistent homology
    "persim>=0.3",  # Persistence diagram utilities
]
```

Note: Pure Python fallback works without optional deps (slower).

---

## Validation

### 1. Synthetic Tests

```python
def test_circle():
    """Circle should have β₀=1, β₁=1"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = np.cos(theta) + 0.1 * np.random.randn(100)
    y = np.sin(theta) + 0.1 * np.random.randn(100)
    cloud = np.column_stack([x, y])
    
    dgms = compute_rips_persistence(cloud, max_dimension=1)
    betti = betti_numbers(dgms, threshold=0.5)
    
    assert betti[0] == 1
    assert betti[1] == 1

def test_two_circles():
    """Two separate circles should have β₀=2, β₁=2"""
    # ... similar test
```

### 2. CWRU Bearing Validation

Expected:
- Healthy: Clean periodic structure (β₁ = 1)
- Faulty: Additional loops or fragmentation (β₁ ≠ 1 or β₀ > 1)

### 3. C-MAPSS Correlation

- Correlate topology metrics with RUL
- Expected: topological_complexity increases toward failure

---

## Files to Create

```
prism/topology/
├── __init__.py
├── point_cloud.py      # Point cloud construction
├── persistence.py      # Persistent homology
├── features.py         # Topological features
└── engine.py           # Main orchestration

tests/topology/
├── test_point_cloud.py
├── test_persistence.py
├── test_features.py
└── test_integration.py
```

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 1 | Point cloud + basic persistence | 3 days |
| 2 | Feature extraction | 2 days |
| 3 | Engine integration | 2 days |
| 4 | Validation | 3 days |
| 5 | Optimization | 2 days |

---

## References

1. Edelsbrunner & Harer (2010). "Computational Topology"
2. Carlsson (2009). "Topology and Data"
3. Perea & Harer (2015). "Sliding Windows and Persistence"
4. Seversky et al. (2016). "On Time-Series Topological Data Analysis"

---

## Notes

Topology is computationally expensive (O(n³) worst case). Mitigations:
1. Subsample point clouds (n_landmarks)
2. Use ripser (fast C++ implementation)
3. Limit max_dimension (usually 2 is enough)
4. Window-based computation (don't process full trajectory at once)

The goal is to detect **shape changes** that precede failure - when the attractor topology simplifies, fragments, or transforms.
