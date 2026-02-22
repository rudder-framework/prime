"""
Per-signal geometry relative to the eigenstructure.

Each signal is a point in feature space. The eigendecomp gives us
the centroid (mean position) and principal components (dominant directions).
This module measures where each signal sits relative to that structure.

Metrics per signal:
- distance: Euclidean distance to centroid (how far from average?)
- coherence: cosine similarity to PC1 after centering (aligned with dominant mode?)
- contribution: projection magnitude onto centroid direction (how much does it contribute?)
- residual: orthogonal component (what's left after projecting out the centroid?)
- signal_magnitude: norm of the signal vector
- pc_projections: projection onto each PC (optional, for detailed analysis)
"""

import numpy as np
from typing import Dict, Any, List, Optional


MIN_NORM = 1e-10


def compute_signal_geometry(
    signal_matrix: np.ndarray,
    signal_ids: List[str],
    centroid: np.ndarray,
    principal_components: Optional[np.ndarray] = None,
    window_index: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Compute per-signal geometry at one window.

    Parameters
    ----------
    signal_matrix : np.ndarray
        (n_signals, n_features) — one row per signal.
    signal_ids : list of str
        Signal identifiers.
    centroid : np.ndarray
        (n_features,) cohort centroid from vector.cohort or eigendecomp.
    principal_components : np.ndarray, optional
        (n_components, n_features) — PC directions from eigendecomp.
        If provided, coherence is computed against PC1.
        If None, centroid direction is used as PC1 proxy.
    window_index : int, optional
        Window index (I) to include in output.

    Returns
    -------
    list of dict — one per signal.
    """
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    centroid = np.asarray(centroid, dtype=np.float64).flatten()
    N, D = signal_matrix.shape

    # Determine PC1 direction
    if principal_components is not None and len(principal_components) > 0:
        pc1 = np.asarray(principal_components[0], dtype=np.float64).flatten()
    else:
        pc1 = centroid.copy()

    centroid_norm = np.linalg.norm(centroid)
    pc1_norm = np.linalg.norm(pc1)

    results = []

    for i in range(N):
        signal = signal_matrix[i]
        row = {'signal_id': signal_ids[i]}

        if window_index is not None:
            row['I'] = window_index

        # Handle all-NaN signals
        if not np.isfinite(signal).any():
            row.update(_empty_signal_result())
            results.append(row)
            continue

        # Impute NaN features with centroid values
        signal_clean = np.where(np.isfinite(signal), signal, centroid)
        signal_norm = np.linalg.norm(signal_clean)

        # --- Distance to centroid ---
        row['distance'] = float(np.linalg.norm(signal_clean - centroid))

        # --- Coherence to PC1 ---
        # Center signal first, then cosine similarity with PC1
        centered = signal_clean - centroid
        centered_norm = np.linalg.norm(centered)

        if centered_norm > MIN_NORM and pc1_norm > MIN_NORM:
            row['coherence'] = float(np.dot(centered, pc1) / (centered_norm * pc1_norm))
        else:
            row['coherence'] = 0.0

        # --- Contribution (projection onto centroid direction) ---
        if centroid_norm > MIN_NORM:
            centroid_unit = centroid / centroid_norm
            row['contribution'] = float(np.dot(signal_clean, centroid_unit))
        else:
            row['contribution'] = 0.0

        # --- Residual (orthogonal to centroid direction) ---
        if centroid_norm > MIN_NORM:
            proj = np.dot(signal_clean, centroid_unit) * centroid_unit
            residual_vec = signal_clean - proj
            row['residual'] = float(np.linalg.norm(residual_vec))
        else:
            row['residual'] = float(signal_norm)

        # --- Signal magnitude ---
        row['signal_magnitude'] = float(signal_norm)

        # --- PC projections (optional, for top PCs) ---
        if principal_components is not None:
            n_pcs = min(3, len(principal_components))
            for pc_i in range(n_pcs):
                pc_vec = np.asarray(principal_components[pc_i], dtype=np.float64).flatten()
                pc_norm = np.linalg.norm(pc_vec)
                if pc_norm > MIN_NORM:
                    row[f'pc{pc_i}_projection'] = float(np.dot(centered, pc_vec / pc_norm))
                else:
                    row[f'pc{pc_i}_projection'] = 0.0

        results.append(row)

    return results


def compute_signal_geometry_batch(
    matrices: List[np.ndarray],
    signal_ids: List[str],
    centroids: List[np.ndarray],
    principal_components_list: Optional[List[np.ndarray]] = None,
    window_indices: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Compute signal geometry across multiple windows.

    Parameters
    ----------
    matrices : list of np.ndarray
        Each (n_signals, n_features), one per window.
    signal_ids : list of str
        Signal identifiers (same for all windows).
    centroids : list of np.ndarray
        Centroid per window.
    principal_components_list : list of np.ndarray, optional
        PCs per window.
    window_indices : list of int, optional
        Window index per entry.

    Returns
    -------
    list of dict — all signals across all windows.
    """
    all_results = []

    for idx, (matrix, centroid) in enumerate(zip(matrices, centroids)):
        pcs = principal_components_list[idx] if principal_components_list is not None else None
        win_idx = window_indices[idx] if window_indices is not None else idx

        rows = compute_signal_geometry(
            matrix, signal_ids, centroid,
            principal_components=pcs,
            window_index=win_idx,
        )
        all_results.extend(rows)

    return all_results


def _empty_signal_result() -> Dict[str, float]:
    """NaN result for all-NaN signals."""
    return {
        'distance': np.nan,
        'coherence': np.nan,
        'contribution': np.nan,
        'residual': np.nan,
        'signal_magnitude': np.nan,
    }
