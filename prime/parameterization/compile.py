"""
Cross-run parameterization compilation.

Reads geometry and signal loading data from each axis run directory,
produces cross-run summary and convergence parquets at the domain root.
"""

from pathlib import Path

import polars as pl
import yaml


def discover_runs(domain_path: Path) -> list[dict]:
    """
    Find subdirectories containing manifest.yaml with parameterization metadata.

    Returns a list of dicts sorted by run_id:
        [{"run_dir": Path, "run_id": int, "axis_signal": str}, ...]
    """
    domain_path = Path(domain_path)
    runs = []
    for child in sorted(domain_path.iterdir()):
        if not child.is_dir():
            continue
        manifest_path = child / "manifest.yaml"
        if not manifest_path.exists():
            continue
        with open(manifest_path) as f:
            manifest = yaml.safe_load(f)
        param = manifest.get("parameterization", {})
        run_id = param.get("run_id")
        axis_signal = param.get("axis_signal")
        if run_id is not None and axis_signal is not None:
            runs.append({
                "run_dir": child,
                "run_id": int(run_id),
                "axis_signal": str(axis_signal),
            })
    runs.sort(key=lambda r: r["run_id"])
    return runs


def _read_run(run: dict) -> dict | None:
    """Read geometry and signal position data for a single run."""
    run_dir = run["run_dir"]

    # Find cohort_geometry.parquet (under system/, cohort/, or <cohort_name>/)
    geometry_files = list(run_dir.glob("*/cohort_geometry.parquet"))
    positions_files = list(run_dir.glob("*/cohort_signal_positions.parquet"))

    if not geometry_files or not positions_files:
        return None

    # Read and concatenate across cohorts
    geometry = pl.concat([pl.read_parquet(f) for f in geometry_files])
    positions = pl.concat([pl.read_parquet(f) for f in positions_files])

    # Geometry aggregates
    effective_dim_mean = geometry["effective_dim"].mean()
    eigenvalue_entropy = geometry["eigenvalue_entropy"].mean()
    spectral_gap_mean = geometry["ratio_2_1"].mean()

    # Dominant signal: highest mean |pc1_loading|
    signal_loadings = (
        positions
        .with_columns(pl.col("pc1_loading").abs().alias("abs_loading"))
        .group_by("signal_id")
        .agg(pl.col("abs_loading").mean().alias("mean_abs_loading"))
        .sort("mean_abs_loading", descending=True)
    )

    dominant_signal = signal_loadings["signal_id"][0]
    dominant_loading = signal_loadings["mean_abs_loading"][0]

    return {
        "run_id": run["run_id"],
        "axis_signal": run["axis_signal"],
        "effective_dim_mean": effective_dim_mean,
        "eigenvalue_entropy": eigenvalue_entropy,
        "spectral_gap_mean": spectral_gap_mean,
        "dominant_signal": dominant_signal,
        "dominant_loading": dominant_loading,
    }


def compile_parameterization(domain_path: Path, verbose: bool = False) -> bool:
    """
    Compile cross-run parameterization summaries.

    Reads geometry + signal positions from each run directory,
    writes all_runs.parquet and convergence.parquet to
    domain_path/parameterization/.

    Returns True if compilation produced output, False if < 2 runs found.
    """
    domain_path = Path(domain_path)
    runs = discover_runs(domain_path)

    if len(runs) < 2:
        if verbose:
            print(f"  Parameterization: {len(runs)} run(s) found, need 2+ to compile.")
        return False

    if verbose:
        print(f"  Parameterization: compiling {len(runs)} runs...")

    # Read data from each run
    rows = []
    for run in runs:
        data = _read_run(run)
        if data is None:
            if verbose:
                print(f"    Skipping {run['axis_signal']} (missing geometry/positions)")
            continue
        rows.append(data)

    if len(rows) < 2:
        if verbose:
            print(f"  Only {len(rows)} run(s) have geometry data, need 2+ to compile.")
        return False

    # Build all_runs DataFrame
    all_runs = pl.DataFrame(rows)

    # Build convergence DataFrame (consecutive pairs)
    convergence_rows = []
    for i in range(len(rows) - 1):
        from_run = rows[i]
        to_run = rows[i + 1]
        converged = to_run["dominant_signal"] == to_run["axis_signal"]
        convergence_rows.append({
            "from_run": from_run["run_id"],
            "to_run": to_run["run_id"],
            "axis_change": f"{from_run['axis_signal']} -> {to_run['axis_signal']}",
            "delta_eff_dim": to_run["effective_dim_mean"] - from_run["effective_dim_mean"],
            "delta_entropy": to_run["eigenvalue_entropy"] - from_run["eigenvalue_entropy"],
            "converged": converged,
        })
    convergence = pl.DataFrame(convergence_rows)

    # Write output
    out_dir = domain_path / "parameterization"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs_path = out_dir / "all_runs.parquet"
    convergence_path = out_dir / "convergence.parquet"

    all_runs.write_parquet(all_runs_path)
    convergence.write_parquet(convergence_path)

    if verbose:
        print(f"    → {all_runs_path} ({len(all_runs)} runs)")
        print(f"    → {convergence_path} ({len(convergence)} pairs)")

    return True
