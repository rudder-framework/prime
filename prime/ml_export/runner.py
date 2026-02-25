"""
Orchestrate the ML export: passthrough -> derivatives -> manifest.
Reads analytical parquets. Writes to output_time/ml/.
Does NOT import from packages/. Does NOT call any engine function.
"""

from pathlib import Path
from .config import ALL_SPECS
from .passthrough import passthrough_copy
from .derivatives import compute_causal_derivatives
from .manifest import write_manifest


def run_ml_export(output_dir: Path, cohort_col: str = "cohort", window_col: str = "window") -> None:
    """
    Run the full ML export pipeline.

    Parameters
    ----------
    output_dir : The output_time/ directory where analytical parquets live.
    cohort_col : Column name for cohort grouping (default: "cohort")
    window_col : Column name for window ordering (default: "window")
    """
    ml_dir = output_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    specs_produced = []

    for spec in ALL_SPECS:
        source_path = output_dir / spec.source_path
        dest_path = ml_dir / f"{spec.ml_name}.parquet"

        # Try alternate source locations if primary doesn't exist
        if not source_path.exists():
            # Try without subdirectory prefix
            alt_name = Path(spec.source_path).name
            alt_path = output_dir / alt_name
            if alt_path.exists():
                source_path = alt_path
            else:
                print(f"  [ml_export] SKIP {spec.ml_name}: source not found ({spec.source_path})")
                continue

        if spec.action == "passthrough":
            success = passthrough_copy(source_path, dest_path)
        elif spec.action == "derive":
            signal_col = "signal_name" if "signal" in spec.grain else None
            success = compute_causal_derivatives(
                source_path=source_path,
                dest_path=dest_path,
                cohort_col=cohort_col,
                window_col=window_col,
                signal_col=signal_col,
            )
        else:
            success = False

        if success:
            specs_produced.append((spec, dest_path))
            print(f"  [ml_export] OK   {spec.ml_name} ({spec.action})")
        else:
            print(f"  [ml_export] FAIL {spec.ml_name}")

    # Write manifest
    write_manifest(ml_dir, specs_produced)
    print(f"  [ml_export] Manifest written: {len(specs_produced)}/{len(ALL_SPECS)} files")
