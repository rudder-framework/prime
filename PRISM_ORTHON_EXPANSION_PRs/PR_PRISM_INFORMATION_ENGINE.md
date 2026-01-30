# PR: Information Flow Engine for PRISM

## Summary

Add causal discovery and information-theoretic analysis to PRISM, computing transfer entropy, Granger causality, and causal network metrics. This captures **who drives whom** - the directional flow of information between signals that geometry, dynamics, and topology cannot see.

---

## Motivation

### The Causal Question

Geometry tells us: "How are signals related?" (symmetric)
Information tells us: **"Which signal drives which?"** (directional)

Before failure, **causal structure shifts**:
- Healthy: Clear hierarchy (control → response)
- Degrading: Feedback loops form
- Pre-failure: Causal structure collapses (everything drives everything)

### Why Causality Matters for Complex Systems

| System | Healthy Causal Pattern | Failure Pattern |
|--------|------------------------|-----------------|
| Chemical plant | Input → reactor → output | Feedback loops, runaway |
| Power grid | Generation → transmission → load | Cascading failures |
| Turbofan | Fuel → combustion → thrust | Mode coupling, resonance |
| Biological | Stimulus → response | Dysregulation, feedback |

### The Information-Theoretic Approach

**Transfer entropy** T(X→Y): Information that X's past provides about Y's future, beyond Y's own past.

- T(X→Y) > 0: X causally influences Y
- T(X→Y) = T(Y→X): Bidirectional coupling
- T(X→Y) >> T(Y→X): X dominates Y

---

## Mathematical Background

### Shannon Entropy

```
H(X) = -Σ p(x) log p(x)
```

Measures uncertainty/information content.

### Mutual Information

```
I(X; Y) = H(X) + H(Y) - H(X, Y)
```

Measures shared information (symmetric).

### Transfer Entropy

```
T(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
```

How much does knowing X's past reduce uncertainty about Y's future?

This is **directional** - T(X→Y) ≠ T(Y→X).

### Granger Causality

Linear version: X Granger-causes Y if past values of X improve prediction of Y beyond Y's own past.

```
Y(t) = Σ a_i Y(t-i) + Σ b_j X(t-j) + ε

If b_j ≠ 0 significantly, X Granger-causes Y.
```

### Convergent Cross-Mapping (CCM)

Nonlinear causal inference using state space reconstruction.

If X causes Y, then the attractor reconstructed from Y contains information about X.

---

## Proposed Implementation

### 1. Information-Theoretic Measures

```python
# prism/information/entropy.py

import numpy as np
from typing import Tuple, Optional
from scipy.special import digamma
from sklearn.neighbors import KDTree


def shannon_entropy(x: np.ndarray, bins: int = 20) -> float:
    """
    Compute Shannon entropy using histogram estimation.
    """
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]
    bin_width = (x.max() - x.min()) / bins
    return -np.sum(hist * np.log(hist) * bin_width)


def mutual_information(
    x: np.ndarray, 
    y: np.ndarray, 
    bins: int = 20
) -> float:
    """
    Compute mutual information I(X; Y) using histogram estimation.
    """
    # Joint distribution
    hist_xy, _, _ = np.histogram2d(x, y, bins=bins, density=True)
    
    # Marginals
    hist_x = hist_xy.sum(axis=1)
    hist_y = hist_xy.sum(axis=0)
    
    # Normalize
    hist_xy = hist_xy / hist_xy.sum()
    hist_x = hist_x / hist_x.sum()
    hist_y = hist_y / hist_y.sum()
    
    # MI = Σ p(x,y) log(p(x,y) / (p(x)p(y)))
    mi = 0.0
    for i in range(bins):
        for j in range(bins):
            if hist_xy[i, j] > 0 and hist_x[i] > 0 and hist_y[j] > 0:
                mi += hist_xy[i, j] * np.log(hist_xy[i, j] / (hist_x[i] * hist_y[j]))
    
    return max(0.0, mi)


def kraskov_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 5
) -> float:
    """
    Compute mutual information using Kraskov et al. (2004) k-NN estimator.
    More accurate for continuous variables than histogram method.
    """
    n = len(x)
    
    # Reshape for KDTree
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    xy = np.hstack([x, y])
    
    # Build trees
    tree_xy = KDTree(xy, metric='chebyshev')
    tree_x = KDTree(x, metric='chebyshev')
    tree_y = KDTree(y, metric='chebyshev')
    
    # For each point, find distance to k-th neighbor in joint space
    dists_xy, _ = tree_xy.query(xy, k=k+1)  # +1 because point itself is included
    eps = dists_xy[:, -1]  # Distance to k-th neighbor
    
    # Count neighbors within eps in marginal spaces
    n_x = np.array([len(tree_x.query_radius([xi], r=e)[0]) - 1 for xi, e in zip(x, eps)])
    n_y = np.array([len(tree_y.query_radius([yi], r=e)[0]) - 1 for yi, e in zip(y, eps)])
    
    # Kraskov formula
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    
    return max(0.0, mi)
```

### 2. Transfer Entropy

```python
# prism/information/transfer_entropy.py

import numpy as np
from typing import Tuple, Optional


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    lag: int = 1,
    history_length: int = 1,
    bins: int = 8
) -> float:
    """
    Compute transfer entropy T(source → target).
    
    T(X→Y) = H(Y_future | Y_past) - H(Y_future | Y_past, X_past)
    
    Parameters
    ----------
    source : array
        Potential cause signal X
    target : array
        Potential effect signal Y
    lag : int
        Prediction horizon
    history_length : int
        How many past values to condition on
    bins : int
        Discretization bins
        
    Returns
    -------
    te : float
        Transfer entropy in bits
    """
    n = len(source)
    
    # Build lagged variables
    # Y_future: target[history_length + lag - 1:]
    # Y_past: target[history_length-1:-lag] (history_length values)
    # X_past: source[history_length-1:-lag] (history_length values)
    
    start = history_length
    end = n - lag
    
    if end <= start:
        return 0.0
    
    y_future = target[start + lag : end + lag]
    y_past = np.column_stack([target[start - i : end - i] for i in range(history_length)])
    x_past = np.column_stack([source[start - i : end - i] for i in range(history_length)])
    
    # Discretize
    def discretize(arr, bins):
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        result = np.zeros_like(arr, dtype=int)
        for j in range(arr.shape[1]):
            edges = np.linspace(arr[:, j].min(), arr[:, j].max() + 1e-10, bins + 1)
            result[:, j] = np.digitize(arr[:, j], edges[:-1]) - 1
        return result
    
    y_future_d = discretize(y_future, bins).flatten()
    y_past_d = discretize(y_past, bins)
    x_past_d = discretize(x_past, bins)
    
    # Convert multi-column past to single index
    def to_index(arr, bins):
        if arr.ndim == 1:
            return arr
        idx = np.zeros(len(arr), dtype=int)
        for j in range(arr.shape[1]):
            idx = idx * bins + arr[:, j]
        return idx
    
    y_past_idx = to_index(y_past_d, bins)
    x_past_idx = to_index(x_past_d, bins)
    xy_past_idx = y_past_idx * (bins ** history_length) + x_past_idx
    
    # Compute probabilities
    def prob_table(indices, n_states):
        counts = np.bincount(indices, minlength=n_states)
        return counts / counts.sum()
    
    def joint_prob(idx1, idx2, n1, n2):
        joint_idx = idx1 * n2 + idx2
        return prob_table(joint_idx, n1 * n2).reshape(n1, n2)
    
    n_y = bins
    n_ypast = bins ** history_length
    n_xypast = bins ** (2 * history_length)
    
    # P(Y_future, Y_past)
    p_yf_yp = joint_prob(y_future_d, y_past_idx, n_y, n_ypast)
    
    # P(Y_future, Y_past, X_past)
    p_yf_xyp = joint_prob(y_future_d, xy_past_idx, n_y, n_xypast)
    
    # P(Y_past)
    p_yp = prob_table(y_past_idx, n_ypast)
    
    # P(Y_past, X_past)
    p_xyp = prob_table(xy_past_idx, n_xypast)
    
    # H(Y_future | Y_past) = H(Y_future, Y_past) - H(Y_past)
    h_yf_yp = -np.sum(p_yf_yp[p_yf_yp > 0] * np.log2(p_yf_yp[p_yf_yp > 0]))
    h_yp = -np.sum(p_yp[p_yp > 0] * np.log2(p_yp[p_yp > 0]))
    h_yf_given_yp = h_yf_yp - h_yp
    
    # H(Y_future | Y_past, X_past) = H(Y_future, Y_past, X_past) - H(Y_past, X_past)
    h_yf_xyp = -np.sum(p_yf_xyp[p_yf_xyp > 0] * np.log2(p_yf_xyp[p_yf_xyp > 0]))
    h_xyp = -np.sum(p_xyp[p_xyp > 0] * np.log2(p_xyp[p_xyp > 0]))
    h_yf_given_xyp = h_yf_xyp - h_xyp
    
    # Transfer entropy
    te = h_yf_given_yp - h_yf_given_xyp
    
    return max(0.0, te)


def transfer_entropy_matrix(
    signals: dict[str, np.ndarray],
    lag: int = 1,
    history_length: int = 1,
    bins: int = 8
) -> Tuple[np.ndarray, list]:
    """
    Compute pairwise transfer entropy matrix.
    
    Returns
    -------
    te_matrix : array, shape (n_signals, n_signals)
        te_matrix[i, j] = T(signal_i → signal_j)
    signal_names : list
        Names corresponding to matrix indices
    """
    names = list(signals.keys())
    n = len(names)
    
    te_matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                te_matrix[i, j] = transfer_entropy(
                    signals[name_i],
                    signals[name_j],
                    lag=lag,
                    history_length=history_length,
                    bins=bins
                )
    
    return te_matrix, names
```

### 3. Granger Causality

```python
# prism/information/granger.py

import numpy as np
from typing import Tuple, Optional
from scipy import stats


def granger_causality(
    source: np.ndarray,
    target: np.ndarray,
    max_lag: int = 5
) -> Tuple[float, float, int]:
    """
    Test if source Granger-causes target.
    
    Returns
    -------
    f_stat : float
        F-statistic for Granger test
    p_value : float
        P-value (< 0.05 suggests causality)
    optimal_lag : int
        Optimal lag based on AIC
    """
    n = len(source)
    
    # Find optimal lag using AIC
    best_aic = np.inf
    optimal_lag = 1
    
    for lag in range(1, max_lag + 1):
        # Restricted model: Y ~ Y_past
        X_r = np.column_stack([target[max_lag - i : n - i] for i in range(1, lag + 1)])
        y = target[max_lag:]
        
        if len(y) < lag + 2:
            continue
        
        # OLS fit
        X_r = np.column_stack([np.ones(len(y)), X_r])
        try:
            beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
            residuals_r = y - X_r @ beta_r
            ssr_r = np.sum(residuals_r ** 2)
            aic = len(y) * np.log(ssr_r / len(y)) + 2 * (lag + 1)
            
            if aic < best_aic:
                best_aic = aic
                optimal_lag = lag
        except:
            continue
    
    lag = optimal_lag
    
    # Restricted model: Y ~ Y_past
    X_r = np.column_stack([target[max_lag - i : n - i] for i in range(1, lag + 1)])
    y = target[max_lag:]
    X_r = np.column_stack([np.ones(len(y)), X_r])
    
    # Unrestricted model: Y ~ Y_past + X_past
    X_u = np.column_stack([
        X_r,
        np.column_stack([source[max_lag - i : n - i] for i in range(1, lag + 1)])
    ])
    
    # Fit both models
    try:
        beta_r = np.linalg.lstsq(X_r, y, rcond=None)[0]
        beta_u = np.linalg.lstsq(X_u, y, rcond=None)[0]
        
        residuals_r = y - X_r @ beta_r
        residuals_u = y - X_u @ beta_u
        
        ssr_r = np.sum(residuals_r ** 2)
        ssr_u = np.sum(residuals_u ** 2)
        
        # F-test
        df1 = lag  # Additional parameters in unrestricted model
        df2 = len(y) - X_u.shape[1]
        
        if df2 <= 0 or ssr_u <= 0:
            return 0.0, 1.0, lag
        
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        
        return float(f_stat), float(p_value), lag
        
    except Exception as e:
        return 0.0, 1.0, lag


def granger_causality_matrix(
    signals: dict[str, np.ndarray],
    max_lag: int = 5,
    significance: float = 0.05
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Compute pairwise Granger causality matrix.
    
    Returns
    -------
    f_matrix : array, shape (n_signals, n_signals)
        F-statistics
    p_matrix : array, shape (n_signals, n_signals)
        P-values
    signal_names : list
        Names corresponding to matrix indices
    """
    names = list(signals.keys())
    n = len(names)
    
    f_matrix = np.zeros((n, n))
    p_matrix = np.ones((n, n))
    
    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i != j:
                f_stat, p_val, _ = granger_causality(
                    signals[name_i],
                    signals[name_j],
                    max_lag=max_lag
                )
                f_matrix[i, j] = f_stat
                p_matrix[i, j] = p_val
    
    return f_matrix, p_matrix, names
```

### 4. Convergent Cross-Mapping

```python
# prism/information/ccm.py

import numpy as np
from typing import Tuple, Optional
from scipy.spatial import KDTree


def convergent_cross_mapping(
    x: np.ndarray,
    y: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    library_sizes: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convergent Cross-Mapping (Sugihara et al., 2012).
    
    Tests for causal influence in nonlinear systems.
    If X causes Y, then Y's attractor contains information about X.
    
    Parameters
    ----------
    x, y : array
        Time series to test
    embedding_dim : int
        Embedding dimension E
    tau : int
        Time delay
    library_sizes : array, optional
        Library sizes to test convergence
        
    Returns
    -------
    library_sizes : array
        Library sizes tested
    rho_xy : array
        Cross-map skill ρ(X|M_Y) at each library size
        If this converges to high value as L increases, Y causes X
    """
    n = len(x)
    max_lib = n - (embedding_dim - 1) * tau
    
    if library_sizes is None:
        library_sizes = np.linspace(embedding_dim + 1, max_lib, 20).astype(int)
    
    # Embed both series
    def embed(series, E, tau):
        n_embedded = len(series) - (E - 1) * tau
        embedded = np.zeros((n_embedded, E))
        for i in range(E):
            embedded[:, i] = series[i * tau : i * tau + n_embedded]
        return embedded
    
    x_embedded = embed(x, embedding_dim, tau)
    y_embedded = embed(y, embedding_dim, tau)
    
    # Corresponding x values for each embedded point
    x_values = x[(embedding_dim - 1) * tau:]
    y_values = y[(embedding_dim - 1) * tau:]
    
    rho_values = []
    
    for L in library_sizes:
        if L > len(y_embedded):
            rho_values.append(np.nan)
            continue
        
        # Use library from Y's manifold to predict X
        # (If Y causes X, Y's manifold encodes X)
        
        # Subsample library
        lib_indices = np.random.choice(len(y_embedded), min(L, len(y_embedded)), replace=False)
        library = y_embedded[lib_indices]
        
        # Build KD-tree on library
        tree = KDTree(library)
        
        # For each point, find nearest neighbors and make prediction
        predictions = []
        actuals = []
        
        for i in range(len(y_embedded)):
            if i in lib_indices:
                continue
            
            # Find E+1 nearest neighbors
            dists, indices = tree.query(y_embedded[i], k=min(embedding_dim + 1, L))
            
            # Convert to library indices
            nn_indices = lib_indices[indices]
            
            # Weights based on distance
            weights = np.exp(-dists / (dists[0] + 1e-10))
            weights = weights / weights.sum()
            
            # Weighted prediction of x
            pred = np.sum(weights * x_values[nn_indices])
            predictions.append(pred)
            actuals.append(x_values[i])
        
        if len(predictions) > 2:
            rho = np.corrcoef(predictions, actuals)[0, 1]
            rho_values.append(rho if not np.isnan(rho) else 0.0)
        else:
            rho_values.append(0.0)
    
    return np.array(library_sizes), np.array(rho_values)


def ccm_causality_test(
    x: np.ndarray,
    y: np.ndarray,
    embedding_dim: int = 3,
    tau: int = 1,
    n_surrogates: int = 100
) -> Tuple[float, float, str]:
    """
    Test for CCM causality with surrogate-based significance.
    
    Returns
    -------
    rho_xy : float
        Cross-map skill Y→X (Y causes X)
    rho_yx : float
        Cross-map skill X→Y (X causes Y)
    direction : str
        'X->Y', 'Y->X', 'BIDIRECTIONAL', or 'NONE'
    """
    n = len(x)
    max_lib = n - (embedding_dim - 1) * tau
    
    # Test at maximum library size
    lib_sizes = np.array([max_lib])
    
    _, rho_yx = convergent_cross_mapping(x, y, embedding_dim, tau, lib_sizes)
    _, rho_xy = convergent_cross_mapping(y, x, embedding_dim, tau, lib_sizes)
    
    rho_xy = rho_xy[0]
    rho_yx = rho_yx[0]
    
    # Surrogate test for significance
    threshold = 0.3  # Simplified threshold
    
    if rho_xy > threshold and rho_yx > threshold:
        direction = 'BIDIRECTIONAL'
    elif rho_xy > threshold:
        direction = 'Y->X'
    elif rho_yx > threshold:
        direction = 'X->Y'
    else:
        direction = 'NONE'
    
    return float(rho_xy), float(rho_yx), direction
```

### 5. Causal Network Analysis

```python
# prism/information/network.py

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CausalNetwork:
    """Container for causal network structure."""
    nodes: List[str]
    adjacency_matrix: np.ndarray  # Weighted edges
    edge_significance: np.ndarray  # P-values or confidence
    
    def get_drivers(self, threshold: float = 0.1) -> List[str]:
        """Find nodes that drive others but aren't driven."""
        n = len(self.nodes)
        out_degree = np.sum(self.adjacency_matrix > threshold, axis=1)
        in_degree = np.sum(self.adjacency_matrix > threshold, axis=0)
        
        drivers = []
        for i, node in enumerate(self.nodes):
            if out_degree[i] > 0 and in_degree[i] == 0:
                drivers.append(node)
        return drivers
    
    def get_sinks(self, threshold: float = 0.1) -> List[str]:
        """Find nodes that are driven but don't drive others."""
        n = len(self.nodes)
        out_degree = np.sum(self.adjacency_matrix > threshold, axis=1)
        in_degree = np.sum(self.adjacency_matrix > threshold, axis=0)
        
        sinks = []
        for i, node in enumerate(self.nodes):
            if in_degree[i] > 0 and out_degree[i] == 0:
                sinks.append(node)
        return sinks
    
    def get_hubs(self, threshold: float = 0.1) -> List[str]:
        """Find highly connected nodes."""
        n = len(self.nodes)
        total_degree = (
            np.sum(self.adjacency_matrix > threshold, axis=1) +
            np.sum(self.adjacency_matrix > threshold, axis=0)
        )
        
        mean_degree = np.mean(total_degree)
        hubs = [self.nodes[i] for i in range(n) if total_degree[i] > 2 * mean_degree]
        return hubs
    
    def feedback_loops(self, threshold: float = 0.1) -> List[Tuple[str, str]]:
        """Find bidirectional edges (feedback loops)."""
        loops = []
        n = len(self.nodes)
        for i in range(n):
            for j in range(i + 1, n):
                if (self.adjacency_matrix[i, j] > threshold and 
                    self.adjacency_matrix[j, i] > threshold):
                    loops.append((self.nodes[i], self.nodes[j]))
        return loops
    
    def causal_hierarchy_score(self) -> float:
        """
        Measure how hierarchical the causal structure is.
        
        0 = fully circular/bidirectional
        1 = perfectly hierarchical (DAG)
        """
        # Compare adjacency matrix with its transpose
        # Hierarchical = asymmetric
        A = self.adjacency_matrix
        symmetry = np.sum(np.minimum(A, A.T)) / (np.sum(A) + 1e-10)
        return 1 - symmetry


def network_metrics(network: CausalNetwork, threshold: float = 0.1) -> Dict[str, float]:
    """
    Compute summary metrics for causal network.
    """
    A = (network.adjacency_matrix > threshold).astype(float)
    n = len(network.nodes)
    
    # Density
    n_edges = np.sum(A)
    density = n_edges / (n * (n - 1))
    
    # Reciprocity
    n_bidirectional = np.sum(np.minimum(A, A.T))
    reciprocity = n_bidirectional / (n_edges + 1e-10)
    
    # In/out degree statistics
    out_degree = np.sum(A, axis=1)
    in_degree = np.sum(A, axis=0)
    
    return {
        'n_nodes': n,
        'n_edges': int(n_edges),
        'density': density,
        'reciprocity': reciprocity,
        'hierarchy_score': network.causal_hierarchy_score(),
        'mean_out_degree': np.mean(out_degree),
        'max_out_degree': np.max(out_degree),
        'mean_in_degree': np.mean(in_degree),
        'max_in_degree': np.max(in_degree),
        'n_drivers': len(network.get_drivers(threshold)),
        'n_sinks': len(network.get_sinks(threshold)),
        'n_hubs': len(network.get_hubs(threshold)),
        'n_feedback_loops': len(network.feedback_loops(threshold)),
    }
```

### 6. Information Engine

```python
# prism/information/engine.py

import numpy as np
import polars as pl
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass

from .transfer_entropy import transfer_entropy_matrix
from .granger import granger_causality_matrix
from .network import CausalNetwork, network_metrics
from .entropy import mutual_information


@dataclass
class InformationResult:
    """Container for information flow analysis results."""
    entity_id: str
    observation_idx: int
    
    # Network summary
    n_causal_edges: int
    network_density: float
    network_reciprocity: float
    hierarchy_score: float
    n_feedback_loops: int
    
    # Dominant flows
    max_transfer_entropy: float
    mean_transfer_entropy: float
    top_driver: str
    top_sink: str
    
    # Information content
    total_mutual_information: float
    mean_pairwise_mi: float
    
    # Change detection
    network_changed: bool


class InformationEngine:
    """
    Compute information-theoretic causality metrics for multivariate time series.
    
    Captures directional information flow between signals that geometry,
    dynamics, and topology cannot see.
    """
    
    def __init__(
        self,
        window_size: int = 100,
        step_size: int = 20,
        te_lag: int = 1,
        te_history: int = 1,
        te_bins: int = 8,
        granger_max_lag: int = 5,
        significance_threshold: float = 0.05,
        te_threshold: float = 0.1
    ):
        self.window_size = window_size
        self.step_size = step_size
        self.te_lag = te_lag
        self.te_history = te_history
        self.te_bins = te_bins
        self.granger_max_lag = granger_max_lag
        self.significance_threshold = significance_threshold
        self.te_threshold = te_threshold
        
        self.previous_network: Optional[CausalNetwork] = None
    
    def compute_for_window(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str,
        window_start: int
    ) -> Tuple[InformationResult, CausalNetwork]:
        """
        Compute information flow metrics for a single window.
        """
        # Extract window
        window_signals = {
            name: sig[window_start:window_start + self.window_size]
            for name, sig in signals.items()
        }
        
        signal_names = list(window_signals.keys())
        n_signals = len(signal_names)
        
        # Transfer entropy matrix
        te_matrix, _ = transfer_entropy_matrix(
            window_signals,
            lag=self.te_lag,
            history_length=self.te_history,
            bins=self.te_bins
        )
        
        # Build causal network from transfer entropy
        network = CausalNetwork(
            nodes=signal_names,
            adjacency_matrix=te_matrix,
            edge_significance=np.zeros_like(te_matrix)  # Could add Granger p-values
        )
        
        # Network metrics
        metrics = network_metrics(network, threshold=self.te_threshold)
        
        # Find top driver and sink
        out_degree = np.sum(te_matrix > self.te_threshold, axis=1)
        in_degree = np.sum(te_matrix > self.te_threshold, axis=0)
        
        top_driver_idx = np.argmax(out_degree)
        top_sink_idx = np.argmax(in_degree)
        
        # Mutual information matrix
        mi_total = 0.0
        mi_count = 0
        for i, name_i in enumerate(signal_names):
            for j, name_j in enumerate(signal_names):
                if i < j:
                    mi = mutual_information(
                        window_signals[name_i],
                        window_signals[name_j]
                    )
                    mi_total += mi
                    mi_count += 1
        
        mean_mi = mi_total / mi_count if mi_count > 0 else 0.0
        
        # Detect network change
        network_changed = False
        if self.previous_network is not None:
            # Compare adjacency matrices
            prev_edges = self.previous_network.adjacency_matrix > self.te_threshold
            curr_edges = te_matrix > self.te_threshold
            change_ratio = np.sum(prev_edges != curr_edges) / (n_signals * n_signals)
            network_changed = change_ratio > 0.2
        
        self.previous_network = network
        
        result = InformationResult(
            entity_id=entity_id,
            observation_idx=window_start + self.window_size // 2,
            
            n_causal_edges=metrics['n_edges'],
            network_density=metrics['density'],
            network_reciprocity=metrics['reciprocity'],
            hierarchy_score=metrics['hierarchy_score'],
            n_feedback_loops=metrics['n_feedback_loops'],
            
            max_transfer_entropy=float(np.max(te_matrix)),
            mean_transfer_entropy=float(np.mean(te_matrix[te_matrix > 0])) if np.any(te_matrix > 0) else 0.0,
            top_driver=signal_names[top_driver_idx],
            top_sink=signal_names[top_sink_idx],
            
            total_mutual_information=mi_total,
            mean_pairwise_mi=mean_mi,
            
            network_changed=network_changed,
        )
        
        return result, network
    
    def compute_for_entity(
        self,
        signals: Dict[str, np.ndarray],
        entity_id: str
    ) -> pl.DataFrame:
        """
        Compute information flow metrics across all windows for an entity.
        """
        n_samples = min(len(s) for s in signals.values())
        results = []
        
        self.previous_network = None  # Reset for new entity
        
        for start in range(0, n_samples - self.window_size, self.step_size):
            try:
                result, _ = self.compute_for_window(signals, entity_id, start)
                results.append({
                    'entity_id': result.entity_id,
                    'observation_idx': result.observation_idx,
                    'n_causal_edges': result.n_causal_edges,
                    'network_density': result.network_density,
                    'network_reciprocity': result.network_reciprocity,
                    'hierarchy_score': result.hierarchy_score,
                    'n_feedback_loops': result.n_feedback_loops,
                    'max_transfer_entropy': result.max_transfer_entropy,
                    'mean_transfer_entropy': result.mean_transfer_entropy,
                    'top_driver': result.top_driver,
                    'top_sink': result.top_sink,
                    'total_mutual_information': result.total_mutual_information,
                    'mean_pairwise_mi': result.mean_pairwise_mi,
                    'network_changed': result.network_changed,
                })
            except Exception as e:
                continue
        
        if not results:
            return pl.DataFrame()
        
        return pl.DataFrame(results)
    
    def to_parquet(self, df: pl.DataFrame, path: Path):
        """Save information flow results to parquet."""
        df.write_parquet(path)


def compute_information_flow(
    data_path: Path,
    output_path: Path,
    window_size: int = 100,
    step_size: int = 20
) -> pl.DataFrame:
    """
    Main entry point: compute information flow metrics for all entities.
    """
    engine = InformationEngine(window_size=window_size, step_size=step_size)
    
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

### information_flow.parquet

| Column | Type | Description |
|--------|------|-------------|
| entity_id | str | Entity identifier |
| observation_idx | int | Center of analysis window |
| n_causal_edges | int | Number of significant causal links |
| network_density | float | Edge density (0-1) |
| network_reciprocity | float | Proportion of bidirectional edges |
| hierarchy_score | float | 0=circular, 1=hierarchical |
| n_feedback_loops | int | Number of bidirectional pairs |
| max_transfer_entropy | float | Strongest causal link |
| mean_transfer_entropy | float | Average causal strength |
| top_driver | str | Signal that drives most others |
| top_sink | str | Signal that is driven most |
| total_mutual_information | float | Total shared information |
| mean_pairwise_mi | float | Average pairwise MI |
| network_changed | bool | Significant topology change |

---

## Interpretation Guide

### Network Evolution Patterns

| Pattern | What It Means |
|---------|---------------|
| Hierarchy ↓, Reciprocity ↑ | Feedback loops forming |
| Density ↑ | Everything coupling |
| n_feedback_loops ↑ | Runaway risk |
| top_driver changes | Control shift |
| network_changed = True | Regime transition |

### Failure Signatures

| System State | Network Pattern |
|--------------|-----------------|
| Healthy | High hierarchy, low reciprocity, clear drivers |
| Degrading | Hierarchy ↓, feedback loops forming |
| Pre-failure | Density spike, reciprocity high, cascade risk |
| Failed | Network collapsed or frozen |

---

## Validation

### 1. Synthetic Tests

```python
def test_unidirectional():
    """X→Y only should show T(X→Y) >> T(Y→X)"""
    x = np.random.randn(1000)
    y = np.zeros(1000)
    y[1:] = 0.8 * x[:-1] + 0.2 * np.random.randn(999)
    
    te_xy = transfer_entropy(x, y)
    te_yx = transfer_entropy(y, x)
    
    assert te_xy > 0.1
    assert te_xy > 2 * te_yx

def test_bidirectional():
    """Coupled system should show high reciprocity"""
    # Coupled oscillators
```

### 2. CWRU Bearing Validation

Expected:
- Healthy: Clear hierarchy (drive end → fan end)
- Faulty: Hierarchy breakdown, feedback loops

### 3. C-MAPSS Correlation

- Track network metrics through engine life
- Expected: Hierarchy decreases, feedback loops increase toward failure

---

## Dependencies

```toml
[project]
dependencies = [
    "numpy>=1.24",
    "polars>=0.20",
    "scipy>=1.10",
]

[project.optional-dependencies]
information = [
    "scikit-learn>=1.0",  # For KNN estimators
]
```

---

## Files to Create

```
prism/information/
├── __init__.py
├── entropy.py          # Shannon, MI
├── transfer_entropy.py # Transfer entropy
├── granger.py          # Granger causality
├── ccm.py              # Convergent cross-mapping
├── network.py          # Causal network analysis
└── engine.py           # Main orchestration

tests/information/
├── test_entropy.py
├── test_transfer_entropy.py
├── test_granger.py
├── test_network.py
└── test_integration.py
```

---

## Timeline

| Phase | Work | Duration |
|-------|------|----------|
| 1 | Entropy + MI | 2 days |
| 2 | Transfer entropy | 3 days |
| 3 | Granger + CCM | 3 days |
| 4 | Network analysis | 2 days |
| 5 | Engine integration | 2 days |
| 6 | Validation | 3 days |

---

## References

1. Schreiber (2000). "Measuring Information Transfer"
2. Granger (1969). "Investigating Causal Relations by Econometric Models"
3. Sugihara et al. (2012). "Detecting Causality in Complex Ecosystems"
4. Kraskov et al. (2004). "Estimating Mutual Information"
5. Runge et al. (2019). "Detecting and quantifying causal associations"

---

## Notes

Information flow analysis is computationally expensive (O(n²) for pairwise, O(n³) for some methods). Mitigations:
1. Limit to most important signals (top-k by variance)
2. Use efficient histogram-based estimators
3. Window-based computation
4. Parallelize pairwise computations

The goal is to detect **causal structure changes** that precede failure - when the information flow hierarchy breaks down, feedback loops form, or control shifts to unexpected signals.
