"""Write ml_manifest.yaml cataloging all files in ml/ directory."""

import yaml
from pathlib import Path
from datetime import datetime, timezone
import polars as pl


def write_manifest(ml_dir: Path, specs_produced: list) -> None:
    """
    Write ml_manifest.yaml with metadata about each file produced.

    specs_produced: list of (FileSpec, actual_path) tuples for files that were created.
    """
    manifest = {
        "ml_export": {
            "version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "causality_guarantee": "All derivative features at window t computed using only windows <= t",
            "files": {},
        }
    }

    for spec, path in specs_produced:
        entry = {
            "path": f"ml/{spec.ml_name}.parquet",
            "source": spec.source_path,
            "type": spec.action,
            "grain": spec.grain,
            "description": spec.description,
        }

        # Add column list if file exists
        if path.exists():
            try:
                df = pl.read_parquet(path)
                entry["columns"] = df.columns
                entry["rows"] = len(df)
            except Exception:
                pass

        manifest["ml_export"]["files"][spec.ml_name] = entry

    manifest_path = ml_dir / "ml_manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
