"""
SQL Typology Runner
===================
Orchestrates the SQL-first typology pipeline:

    1. SQL Layer 1: statistics, derivatives, temporal, windows, correlations
    2. Python/Rust: expensive primitives (hurst, entropy, spectral) 
    3. SQL Layer 2: classification + enrichment
    4. Manifest generation

The SQL scripts run against observations.parquet via DuckDB.
Python only handles the ~40% of features that require iterative algorithms.

Usage:
    from prime.sql.typology.runner import run_sql_typology
    typology_path = run_sql_typology(
        observations_path="observations.parquet",
        output_dir="output_time",
        window_size=84,
        stride=42
    )
"""

import duckdb
import time
from pathlib import Path
from typing import Optional


# SQL script execution order
SQL_LAYER_1 = [
    "01_signal_statistics.sql",
    "02_signal_derivatives.sql",
    "03_signal_temporal.sql",
    # These two require window_size/stride from manifest or defaults
    # "04_signal_windows.sql",
    # "05_signal_correlations.sql",
]

SQL_LAYER_1_WINDOWED = [
    "04_signal_windows.sql",
    "05_signal_correlations.sql",
]

SQL_LAYER_2 = [
    "06_typology_classify.sql",
]

SQL_LAYER_2B = [
    "06b_typology_enrich.sql",
]


def run_sql_typology(
    observations_path: str,
    output_dir: str,
    sql_dir: Optional[str] = None,
    window_size: int = 128,
    stride: int = 64,
    skip_expensive: bool = False,
    n_workers: int = 4,
    subsample_limit: int = 2000,
    verbose: bool = True
) -> str:
    """
    Run the full SQL typology pipeline.
    
    Parameters
    ----------
    observations_path : str
        Path to observations.parquet (canonical schema)
    output_dir : str
        Directory for output parquets
    sql_dir : str, optional
        Path to sql/typology/ directory. If None, looks relative to this file.
    window_size : int
        System window size (from manifest or default 128)
    stride : int
        System stride (from manifest or default 64)
    skip_expensive : bool
        If True, skip Python/Rust primitives (useful for testing SQL layer only)
    n_workers : int
        Number of workers for Python primitives
    subsample_limit : int
        Subsample signals longer than this for expensive primitives
    verbose : bool
        Print progress
        
    Returns
    -------
    str : Path to final typology.parquet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if sql_dir is None:
        sql_dir = Path(__file__).parent
    else:
        sql_dir = Path(sql_dir)
    
    con = duckdb.connect()
    
    # Configure DuckDB for performance
    con.execute(f"SET threads = {n_workers}")
    con.execute("SET memory_limit = '4GB'")
    
    # Load observations
    # SQL scripts reference ordering column as I; canonical schema uses signal_0
    if verbose:
        print(f"  Loading observations from {observations_path}")
    col_info = con.execute(
        f"SELECT * FROM read_parquet('{observations_path}') LIMIT 0"
    ).description
    col_names = [c[0] for c in col_info]
    if 'I' not in col_names and 'signal_0' in col_names:
        con.execute(f"""
            CREATE TABLE observations AS
            SELECT *, signal_0 AS I FROM read_parquet('{observations_path}')
        """)
    else:
        con.execute(f"""
            CREATE TABLE observations AS
            SELECT * FROM read_parquet('{observations_path}')
        """)
    
    row_count = con.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    signal_count = con.execute("SELECT COUNT(DISTINCT signal_id) FROM observations").fetchone()[0]
    cohort_count = con.execute("SELECT COUNT(DISTINCT cohort) FROM observations").fetchone()[0]
    
    if verbose:
        print(f"  {row_count:,} rows, {signal_count} signals, {cohort_count} cohorts")
    
    # ==========================================
    # SQL LAYER 1: Statistics, Derivatives, Temporal
    # ==========================================
    if verbose:
        print("\n  SQL Layer 1: Statistics + Derivatives + Temporal")
    
    for script_name in SQL_LAYER_1:
        t0 = time.time()
        sql_path = sql_dir / script_name
        sql = sql_path.read_text()
        sql = sql.replace("{output_dir}", str(output_dir))
        
        # Execute (may contain multiple statements)
        for statement in _split_sql(sql):
            if statement.strip():
                con.execute(statement)
        
        elapsed = time.time() - t0
        if verbose:
            print(f"    {script_name}: {elapsed:.1f}s")
    
    # Check how many constants were found
    constant_count = con.execute("""
        SELECT COUNT(*) FROM signal_statistics WHERE is_constant = TRUE
    """).fetchone()[0]
    active_count = signal_count - constant_count
    
    if verbose:
        print(f"    → {constant_count} constants detected, {active_count} active signals")
    
    # ==========================================
    # SQL LAYER 1 (windowed): Requires window_size/stride
    # ==========================================
    if verbose:
        print(f"\n  SQL Layer 1 (windowed): window={window_size}, stride={stride}")
    
    for script_name in SQL_LAYER_1_WINDOWED:
        t0 = time.time()
        sql_path = sql_dir / script_name
        sql = sql_path.read_text()
        sql = sql.replace("{output_dir}", str(output_dir))
        sql = sql.replace("{window_size}", str(window_size))
        sql = sql.replace("{stride}", str(stride))
        
        for statement in _split_sql(sql):
            if statement.strip():
                con.execute(statement)
        
        elapsed = time.time() - t0
        if verbose:
            print(f"    {script_name}: {elapsed:.1f}s")
    
    # ==========================================
    # SQL LAYER 2: Initial classification (pre-Python)
    # ==========================================
    if verbose:
        print("\n  SQL Layer 2: Classification")
    
    for script_name in SQL_LAYER_2:
        t0 = time.time()
        sql_path = sql_dir / script_name
        sql = sql_path.read_text()
        sql = sql.replace("{output_dir}", str(output_dir))
        
        for statement in _split_sql(sql):
            if statement.strip():
                con.execute(statement)
        
        elapsed = time.time() - t0
        if verbose:
            print(f"    {script_name}: {elapsed:.1f}s")
    
    # ==========================================
    # PYTHON/RUST: Expensive primitives
    # ==========================================
    if not skip_expensive:
        if verbose:
            print(f"\n  Python/Rust: Expensive primitives (subsample > {subsample_limit})")
        
        t0 = time.time()
        _compute_expensive_primitives(
            con=con,
            observations_path=observations_path,
            output_dir=output_dir,
            n_workers=n_workers,
            subsample_limit=subsample_limit,
            verbose=verbose
        )
        elapsed = time.time() - t0
        if verbose:
            print(f"    Primitives computed: {elapsed:.1f}s")
        
        # ==========================================
        # SQL LAYER 2B: Enrich with Python results
        # ==========================================
        if verbose:
            print("\n  SQL Layer 2B: Enrichment (post-Python)")
        
        for script_name in SQL_LAYER_2B:
            t0 = time.time()
            sql_path = sql_dir / script_name
            sql = sql_path.read_text()
            sql = sql.replace("{output_dir}", str(output_dir))
            
            for statement in _split_sql(sql):
                if statement.strip():
                    con.execute(statement)
            
            elapsed = time.time() - t0
            if verbose:
                print(f"    {script_name}: {elapsed:.1f}s")
    
    typology_path = output_dir / "typology.parquet"
    
    # Summary
    if verbose:
        col_count = len(con.execute("SELECT * FROM typology LIMIT 0").description)
        row_count = con.execute("SELECT COUNT(*) FROM typology").fetchone()[0]
        print(f"\n  ✓ typology.parquet: {row_count} rows × {col_count} columns")
        print(f"  ✓ Saved to {typology_path}")
    
    con.close()
    return str(typology_path)


def _compute_expensive_primitives(
    con: duckdb.DuckDBPyConnection,
    observations_path: str,
    output_dir: Path,
    n_workers: int = 4,
    subsample_limit: int = 2000,
    verbose: bool = True
):
    """
    Compute features that SQL can't: hurst, entropy, spectral.
    Only runs on non-constant signals. Subsamples long signals.
    
    Results are written to signal_primitives.parquet and loaded
    into the DuckDB connection as the signal_primitives table.
    """
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    # Get list of non-constant signals
    active_signals = con.execute("""
        SELECT DISTINCT cohort, signal_id
        FROM signal_statistics
        WHERE is_constant = FALSE
    """).fetchall()
    
    if verbose:
        print(f"    Computing primitives for {len(active_signals)} active signals")
    
    # Extract signal data for each active signal
    # DuckDB is fast at this — pull all at once
    signal_data = {}
    for cohort, signal_id in active_signals:
        values = con.execute(f"""
            SELECT value FROM observations
            WHERE cohort = '{cohort}' AND signal_id = '{signal_id}'
            ORDER BY I
        """).fetchnumpy()['value']

        # Keep full signal for ACF, subsample for expensive primitives
        full_values = values
        if len(values) > subsample_limit:
            step = len(values) // subsample_limit
            values = values[::step]

        signal_data[(cohort, signal_id)] = (values, full_values)

    # Compute primitives (parallel)
    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {}
        for (cohort, signal_id), (values, full_values) in signal_data.items():
            future = executor.submit(
                _compute_single_signal, cohort, signal_id, values, full_values
            )
            futures[future] = (cohort, signal_id)
        
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                cohort, signal_id = futures[future]
                if verbose:
                    print(f"    WARNING: Failed {cohort}/{signal_id}: {e}")
                results.append({
                    'cohort': cohort,
                    'signal_id': signal_id,
                    'hurst_exponent': None,
                    'sample_entropy': None,
                    'perm_entropy': None,
                    'spectral_slope': None,
                    'spectral_flatness': None,
                    'spectral_entropy': None,
                    'dominant_frequency': None,
                    'acf_half_life': None,
                })
    
    # Write signal_primitives.parquet
    import pandas as pd
    df = pd.DataFrame(results)
    primitives_path = output_dir / "signal_primitives.parquet"
    df.to_parquet(primitives_path)
    
    # Load into DuckDB for enrichment
    con.execute(f"""
        CREATE TABLE signal_primitives AS 
        SELECT * FROM read_parquet('{primitives_path}')
    """)
    
    if verbose:
        print(f"    → signal_primitives.parquet: {len(results)} signals")


def _compute_single_signal(cohort: str, signal_id: str, values, full_values=None) -> dict:
    """
    Compute expensive primitives for one signal.
    Called in parallel via ProcessPoolExecutor.

    Args:
        values: subsampled signal for hurst/entropy/spectral (O(n^2) algorithms)
        full_values: full signal for ACF half-life (O(n) algorithm, needs full resolution)
    """
    import numpy as np

    if full_values is None:
        full_values = values

    result = {
        'cohort': cohort,
        'signal_id': signal_id,
    }

    try:
        from pmtvs.hurst import hurst_exponent
        result['hurst_exponent'] = float(hurst_exponent(values))
    except ImportError:
        result['hurst_exponent'] = _fallback_hurst(values)
    except Exception:
        result['hurst_exponent'] = None

    try:
        from pmtvs.entropy import sample_entropy, permutation_entropy
        result['sample_entropy'] = float(sample_entropy(values, m=2, r=0.2))
        result['perm_entropy'] = float(permutation_entropy(values, order=3, normalize=True))
    except ImportError:
        result['sample_entropy'] = None
        result['perm_entropy'] = None
    except Exception:
        result['sample_entropy'] = None
        result['perm_entropy'] = None

    # Spectral features via FFT
    try:
        result.update(_compute_spectral(values))
    except Exception:
        result['spectral_slope'] = None
        result['spectral_flatness'] = None
        result['spectral_entropy'] = None
        result['dominant_frequency'] = None

    # ACF half-life — computed on FULL signal (not subsampled)
    # Subsampling destroys autocorrelation structure
    try:
        result['acf_half_life'] = _compute_acf_half_life(full_values)
    except Exception:
        result['acf_half_life'] = None

    return result


def _compute_spectral(values) -> dict:
    """Compute spectral features from FFT."""
    import numpy as np
    from scipy.fft import rfft, rfftfreq
    
    n = len(values)
    freqs = rfftfreq(n)
    fft_vals = np.abs(rfft(values - np.mean(values)))
    psd = fft_vals ** 2
    
    # Skip DC component
    freqs = freqs[1:]
    psd = psd[1:]
    
    if len(psd) == 0 or np.sum(psd) == 0:
        return {
            'spectral_slope': None,
            'spectral_flatness': None,
            'spectral_entropy': None,
            'dominant_frequency': None,
        }
    
    # Spectral slope (log-log regression)
    log_freqs = np.log10(freqs[freqs > 0])
    log_psd = np.log10(psd[freqs > 0] + 1e-20)
    if len(log_freqs) > 1:
        slope = np.polyfit(log_freqs, log_psd, 1)[0]
    else:
        slope = 0.0
    
    # Spectral flatness (geometric mean / arithmetic mean)
    psd_norm = psd / np.sum(psd)
    log_mean = np.exp(np.mean(np.log(psd_norm + 1e-20)))
    arith_mean = np.mean(psd_norm)
    flatness = log_mean / arith_mean if arith_mean > 0 else 0.0
    
    # Spectral entropy
    spectral_entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-20))
    
    # Dominant frequency
    dominant_freq = float(freqs[np.argmax(psd)])
    
    return {
        'spectral_slope': float(slope),
        'spectral_flatness': float(np.clip(flatness, 0, 1)),
        'spectral_entropy': float(spectral_entropy),
        'dominant_frequency': dominant_freq,
    }


def _compute_acf_half_life(values) -> float:
    """Compute ACF half-life (lag where ACF drops below 0.5)."""
    import numpy as np
    
    n = len(values)
    mean = np.mean(values)
    var = np.var(values)
    if var == 0:
        return None
    
    max_lag = min(n // 2, 500)
    centered = values - mean
    
    for lag in range(1, max_lag):
        acf = np.mean(centered[:n-lag] * centered[lag:]) / var
        if acf < 0.5:
            return float(lag)
    
    return float(max_lag)


def _fallback_hurst(values) -> float:
    """Simple R/S Hurst estimation without pmtvs."""
    import numpy as np
    
    n = len(values)
    if n < 20:
        return None
    
    max_k = min(n // 2, 512)
    sizes = []
    rs_values = []
    
    for size in [16, 32, 64, 128, 256, 512]:
        if size > max_k:
            break
        n_segments = n // size
        if n_segments < 1:
            continue
        
        rs_list = []
        for i in range(n_segments):
            segment = values[i*size:(i+1)*size]
            mean = np.mean(segment)
            deviations = np.cumsum(segment - mean)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(segment)
            if S > 0:
                rs_list.append(R / S)
        
        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))
    
    if len(sizes) < 2:
        return None
    
    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    hurst = np.polyfit(log_sizes, log_rs, 1)[0]
    
    return float(np.clip(hurst, 0, 1))


def _split_sql(sql_text: str) -> list:
    """
    Split SQL text into individual statements.
    Handles semicolons inside strings and comments.
    """
    statements = []
    current = []
    in_single_quote = False
    in_line_comment = False
    in_block_comment = False
    
    i = 0
    while i < len(sql_text):
        char = sql_text[i]
        
        # Track comment state
        if not in_single_quote and not in_block_comment and char == '-' and i + 1 < len(sql_text) and sql_text[i+1] == '-':
            in_line_comment = True
        if in_line_comment and char == '\n':
            in_line_comment = False
        if not in_single_quote and not in_line_comment and char == '/' and i + 1 < len(sql_text) and sql_text[i+1] == '*':
            in_block_comment = True
        if in_block_comment and char == '*' and i + 1 < len(sql_text) and sql_text[i+1] == '/':
            in_block_comment = False
            current.append('*/')
            i += 2
            continue
        
        # Track string state
        if not in_line_comment and not in_block_comment and char == "'":
            in_single_quote = not in_single_quote
        
        # Split on semicolons outside strings/comments
        if char == ';' and not in_single_quote and not in_line_comment and not in_block_comment:
            stmt = ''.join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
        else:
            current.append(char)
        
        i += 1
    
    # Handle last statement without trailing semicolon
    stmt = ''.join(current).strip()
    if stmt:
        statements.append(stmt)
    
    return statements


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python runner.py <observations_path> [output_dir] [window_size] [stride]")
        sys.exit(1)
    
    obs_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output_time"
    win_size = int(sys.argv[3]) if len(sys.argv) > 3 else 128
    stride_val = int(sys.argv[4]) if len(sys.argv) > 4 else 64
    
    run_sql_typology(
        observations_path=obs_path,
        output_dir=out_dir,
        window_size=win_size,
        stride=stride_val,
        verbose=True
    )
