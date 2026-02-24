"""Python preprocessor for 25_feature_relevance.sql.

Creates DuckDB tables from PCA on the signal_vector view:
  - v_pca_modes: top PCs with variance explained and top loadings
  - v_pca_loadings: per-feature loading for each PC
  - v_feature_categories: feature_name → category mapping
"""

from pathlib import Path

import duckdb
import numpy as np


# Category keywords — order matters (first match wins)
_CATEGORY_RULES = [
    ('GEOMETRY', [
        'effective_dim', 'condition_number', 'total_variance',
        'explained_ratio', 'eigenvalue', 'coherence', 'brittleness',
    ]),
    ('DYNAMICS', [
        'velocity', 'acceleration', 'jerk', 'curvature',
        'ftle', 'lyapunov', 'collapse',
    ]),
    ('COUPLING', [
        'correlation', 'mutual_info', 'transfer_entropy',
        'granger', 'cointegration', 'coupling',
    ]),
    ('TOPOLOGY', ['betti', 'homology', 'persistence', 'topolog']),
]

# Metadata columns to exclude from feature analysis
_META_COLUMNS = {'signal_id', 'cohort', 'window_index'}
_META_PREFIXES = ('signal_0_',)


def _categorize(feature_name: str) -> str:
    lower = feature_name.lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in lower for kw in keywords):
            return category
    return 'SIGNAL'


def preprocess(
    con: duckdb.DuckDBPyConnection,
    run_dir: Path,
    domain_dir: Path,
) -> None:
    """Create PCA and category tables for feature relevance report."""
    # Check signal_vector exists
    try:
        cols_info = con.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'signal_vector'"
        ).fetchall()
    except Exception:
        return
    if not cols_info:
        # Try as a view — DESCRIBE works on views
        try:
            cols_info = [(r[0],) for r in con.execute(
                "DESCRIBE SELECT * FROM signal_vector"
            ).fetchall()]
        except Exception:
            return

    all_cols = [r[0] for r in cols_info]
    feature_cols = [
        c for c in all_cols
        if c not in _META_COLUMNS
        and not any(c.startswith(p) for p in _META_PREFIXES)
    ]
    if len(feature_cols) < 2:
        return

    # --- Build category table (always, even if PCA fails) ---
    cat_rows = [(f, _categorize(f)) for f in feature_cols]
    con.execute("CREATE OR REPLACE TABLE v_feature_categories "
                "(feature_name VARCHAR, category VARCHAR)")
    con.executemany(
        "INSERT INTO v_feature_categories VALUES (?, ?)", cat_rows
    )

    # --- Aggregate to per-window means ---
    avg_exprs = ', '.join(
        f'AVG(CASE WHEN isnan("{c}") OR NOT isfinite("{c}") THEN NULL ELSE "{c}" END) AS "{c}"'
        for c in feature_cols
    )
    try:
        df = con.execute(
            f'SELECT {avg_exprs} FROM signal_vector '
            f'GROUP BY signal_0_center ORDER BY signal_0_center'
        ).fetchnumpy()
    except Exception:
        return

    # Build matrix (windows × features), drop all-NaN or constant columns
    arrays = []
    valid_features = []
    for c in feature_cols:
        arr = np.asarray(df[c], dtype=np.float64)
        # Replace inf with NaN for consistent handling
        arr[~np.isfinite(arr)] = np.nan
        if np.all(np.isnan(arr)):
            continue
        std = np.nanstd(arr)
        if std < 1e-15:
            continue
        arrays.append(arr)
        valid_features.append(c)

    if len(valid_features) < 2:
        return
    matrix = np.column_stack(arrays)  # (n_windows, n_features)

    # Drop rows with any NaN
    mask = ~np.any(np.isnan(matrix), axis=1)
    matrix = matrix[mask]
    n_windows, n_features = matrix.shape
    if n_windows < 3:
        return

    # --- Standardize ---
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds < 1e-15] = 1.0
    Z = (matrix - means) / stds

    # --- PCA via eigendecomposition of covariance ---
    cov = np.cov(Z, rowvar=False)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
    except np.linalg.LinAlgError:
        return

    # eigh returns ascending order — reverse for descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Clamp negatives (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    total_var = eigenvalues.sum()
    if total_var < 1e-15:
        return

    n_pcs = min(5, n_features)
    var_explained = eigenvalues[:n_pcs] / total_var

    # --- Create v_pca_loadings ---
    loading_rows = []
    for pc_idx in range(n_pcs):
        loadings = eigenvectors[:, pc_idx]
        for feat_idx, feat_name in enumerate(valid_features):
            loading_rows.append((
                pc_idx + 1,
                feat_name,
                float(loadings[feat_idx]),
                float(abs(loadings[feat_idx])),
            ))

    con.execute(
        "CREATE OR REPLACE TABLE v_pca_loadings "
        "(pc INT, feature_name VARCHAR, loading DOUBLE, abs_loading DOUBLE)"
    )
    con.executemany(
        "INSERT INTO v_pca_loadings VALUES (?, ?, ?, ?)", loading_rows
    )

    # --- Create v_pca_modes ---
    mode_rows = []
    for pc_idx in range(n_pcs):
        loadings = eigenvectors[:, pc_idx]
        abs_loadings = np.abs(loadings)
        top3_idx = np.argsort(abs_loadings)[::-1][:3]

        top_names = [valid_features[i] for i in top3_idx]
        top_vals = [float(loadings[i]) for i in top3_idx]

        # Auto-label from top loading categories
        top_cats = [_categorize(n) for n in top_names]
        # Most common category among top 3
        from collections import Counter
        dominant_cat = Counter(top_cats).most_common(1)[0][0]
        label = f'{dominant_cat.lower()}_mode'

        mode_rows.append((
            pc_idx + 1,
            float(var_explained[pc_idx]),
            f'{top_names[0]} ({top_vals[0]:+.3f})',
            f'{top_names[1]} ({top_vals[1]:+.3f})',
            f'{top_names[2]} ({top_vals[2]:+.3f})',
            label,
        ))

    con.execute(
        "CREATE OR REPLACE TABLE v_pca_modes "
        "(pc INT, variance_explained DOUBLE, "
        "top_loading_1 VARCHAR, top_loading_2 VARCHAR, top_loading_3 VARCHAR, "
        "interpretation VARCHAR)"
    )
    con.executemany(
        "INSERT INTO v_pca_modes VALUES (?, ?, ?, ?, ?, ?)", mode_rows
    )
