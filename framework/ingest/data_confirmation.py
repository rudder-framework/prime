"""
RUDDER Data Confirmation

Confirms observations.parquet is ready to send to PRISM.
Reads validation rules from canonical PRISM_SCHEMA.yaml.

Schema v2.0.0:
- REQUIRED: signal_id, I, value
- OPTIONAL: unit_id (just a label, blank is fine)

Usage:
    from framework.ingest.data_confirmation import confirm_data
    confirm_data(observations_path)
"""

import polars as pl
from pathlib import Path
import sys
from dataclasses import dataclass, field
from typing import Optional
import json

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


# =============================================================================
# SCHEMA
# =============================================================================

REQUIRED_COLUMNS = ["signal_id", "I", "value"]
OPTIONAL_COLUMNS = ["unit_id"]

SCHEMA_LOCATIONS = [
    Path(__file__).parent / "schema" / "PRISM_SCHEMA.yaml",
    Path(__file__).parent / "PRISM_SCHEMA.yaml",
]


def find_schema() -> Optional[Path]:
    for loc in SCHEMA_LOCATIONS:
        if loc.exists():
            return loc
    return None


def get_default_schema() -> dict:
    return {
        "required_columns": REQUIRED_COLUMNS,
        "optional_columns": OPTIONAL_COLUMNS,
        "requirements": {
            "min_signals": 2,
            "min_observations": 50,
        },
    }


def load_schema(path: Optional[Path] = None) -> dict:
    if path is None:
        path = find_schema()
    if path is None or not path.exists() or not HAS_YAML:
        return get_default_schema()
    with open(path) as f:
        return yaml.safe_load(f)


# =============================================================================
# RESULT CLASS
# =============================================================================

@dataclass
class ConfirmationResult:
    confirmed: bool = True
    ready_for_prism: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def error(self, rule_id: str, message: str, fix: Optional[str] = None):
        self.confirmed = False
        self.ready_for_prism = False
        self.errors.append({"rule": rule_id, "message": message, "fix": fix})

    def warn(self, rule_id: str, message: str):
        self.warnings.append({"rule": rule_id, "message": message})

    def note(self, message: str):
        self.notes.append(message)

    def to_dict(self) -> dict:
        return {
            "confirmed": self.confirmed,
            "ready_for_prism": self.ready_for_prism,
            "errors": self.errors,
            "warnings": self.warnings,
            "notes": self.notes,
            "stats": self.stats,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


# =============================================================================
# TYPE MAPPING
# =============================================================================

TYPE_MAP = {
    "string": [pl.String, pl.Utf8],
    "uint32": [pl.UInt32, pl.UInt64, pl.Int32, pl.Int64],
    "float64": [pl.Float64, pl.Float32],
}


# =============================================================================
# MAIN CONFIRMATION FUNCTION
# =============================================================================

def confirm_data(
    path: Path,
    schema_path: Optional[Path] = None,
    verbose: bool = True,
) -> ConfirmationResult:
    """
    Confirm observations.parquet is ready for PRISM.

    Required: signal_id, I, value
    Optional: unit_id (just a label)
    """

    result = ConfirmationResult()
    schema = load_schema(schema_path)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RUDDER DATA CONFIRMATION")
        print(f"{'='*60}")
        print(f"File: {path}")
        print()

    # =========================================================================
    # FILE EXISTS
    # =========================================================================

    if not path.exists():
        result.error("FILE_EXISTS", f"File not found: {path}",
                     "Run RUDDER transform to create observations.parquet")
        if verbose:
            print(f"[X] File not found: {path}")
        return result

    # =========================================================================
    # LOAD DATA
    # =========================================================================

    try:
        df = pl.read_parquet(path)
    except Exception as e:
        result.error("FILE_READABLE", f"Failed to read parquet: {e}")
        if verbose:
            print(f"[X] Failed to read: {e}")
        return result

    result.stats["file_size_mb"] = round(path.stat().st_size / 1024 / 1024, 2)
    result.stats["rows"] = df.shape[0]
    result.stats["columns"] = list(df.columns)

    if verbose:
        print(f"Loaded: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"Size: {result.stats['file_size_mb']} MB")
        print(f"Columns: {df.columns}\n")

    # =========================================================================
    # REQUIRED COLUMNS: signal_id, I, value
    # =========================================================================

    if verbose:
        print("-" * 40)
        print("REQUIRED COLUMNS")
        print("-" * 40)

    for col in REQUIRED_COLUMNS:
        if col in df.columns:
            if verbose:
                print(f"[OK] {col}: {df[col].dtype}")
        else:
            result.error("REQUIRED_COLUMN", f"Missing REQUIRED column: {col}")
            if verbose:
                print(f"[X] {col}: MISSING")

    if not result.confirmed:
        return result

    # =========================================================================
    # OPTIONAL COLUMN: unit_id
    # =========================================================================

    if verbose:
        print(f"\n{'-'*40}")
        print("OPTIONAL: unit_id")
        print("-" * 40)

    if "unit_id" not in df.columns:
        result.note("No unit_id column. Will use blank. This is fine.")
        if verbose:
            print("[i] No unit_id column found.")
            print("    This is fine - unit_id is optional.")
            print("    PRISM will use blank unit_id.")
        # Add blank unit_id for processing
        df = df.with_columns(pl.lit("").alias("unit_id"))
    else:
        units = df["unit_id"].unique().to_list()
        n_units = len(units)
        result.stats["n_units"] = n_units
        result.stats["units_sample"] = units[:10] if n_units > 10 else units
        if verbose:
            print(f"[OK] unit_id found: {n_units} unique values")
            if n_units <= 10:
                print(f"    Values: {units}")
            else:
                print(f"    Sample: {units[:5]}... and {n_units-5} more")
        # Note: We do NOT validate unit_id contents. It's just a label.

    # =========================================================================
    # SIGNAL REQUIREMENTS
    # =========================================================================

    if verbose:
        print(f"\n{'-'*40}")
        print("SIGNAL REQUIREMENTS")
        print("-" * 40)

    signals = df["signal_id"].unique().sort().to_list()
    n_signals = len(signals)
    result.stats["n_signals"] = n_signals
    result.stats["signals"] = signals

    min_signals = schema.get("requirements", {}).get("min_signals", 2)

    if n_signals < min_signals:
        result.error("MIN_SIGNALS",
                     f"Need >={min_signals} signals, found {n_signals}",
                     "Include more signals in transform")
        if verbose:
            print(f"[X] Only {n_signals} signal(s). Need >={min_signals}.")
    else:
        if verbose:
            print(f"[OK] {n_signals} signals: {signals}")

    # =========================================================================
    # INDEX INTEGRITY (I)
    # =========================================================================

    if verbose:
        print(f"\n{'-'*40}")
        print("INDEX INTEGRITY (I)")
        print("-" * 40)

    # Group by unit_id + signal_id for I checks
    group_cols = ["unit_id", "signal_id"] if "unit_id" in df.columns else ["signal_id"]

    seq_check = (
        df.group_by(group_cols)
        .agg([
            pl.col("I").min().alias("min_i"),
            pl.col("I").max().alias("max_i"),
            pl.len().alias("count")
        ])
        .with_columns(
            ((pl.col("max_i") - pl.col("min_i") + 1) == pl.col("count")).alias("is_sequential")
        )
    )

    non_sequential = seq_check.filter(~pl.col("is_sequential"))

    if len(non_sequential) > 0:
        result.error("I_SEQUENTIAL",
                     f"{len(non_sequential)} groups have non-sequential I",
                     "Use fix_sparse_index() in transform")
        if verbose:
            print(f"[X] Non-sequential I in {len(non_sequential)} groups")
    else:
        if verbose:
            print(f"[OK] I is sequential for all groups")

    # Check I starts at 0
    min_i_check = df.group_by(group_cols).agg(pl.col("I").min().alias("min_i"))
    non_zero = min_i_check.filter(pl.col("min_i") != 0)

    if len(non_zero) > 0:
        result.error("I_STARTS_ZERO",
                     f"{len(non_zero)} groups don't start at I=0")
        if verbose:
            print(f"[X] {len(non_zero)} groups don't start at I=0")
    else:
        if verbose:
            print(f"[OK] I starts at 0 for all groups")

    # =========================================================================
    # OBSERVATION COUNTS
    # =========================================================================

    if verbose:
        print(f"\n{'-'*40}")
        print("OBSERVATION COUNTS")
        print("-" * 40)

    min_obs = schema.get("requirements", {}).get("min_observations", 50)

    obs_per_group = df.group_by(group_cols).agg(pl.len().alias("n_obs"))
    result.stats["min_obs"] = obs_per_group["n_obs"].min()
    result.stats["max_obs"] = obs_per_group["n_obs"].max()
    result.stats["mean_obs"] = round(obs_per_group["n_obs"].mean(), 1)

    insufficient = obs_per_group.filter(pl.col("n_obs") < min_obs)

    if len(insufficient) > 0:
        result.warn("MIN_OBSERVATIONS",
                    f"{len(insufficient)} groups have <{min_obs} observations")
        if verbose:
            print(f"[!] {len(insufficient)} groups have <{min_obs} observations")
    else:
        if verbose:
            print(f"[OK] All groups have >={min_obs} observations")

    if verbose:
        print(f"    Min: {result.stats['min_obs']}")
        print(f"    Max: {result.stats['max_obs']}")
        print(f"    Mean: {result.stats['mean_obs']}")

    # =========================================================================
    # NULL CHECKS
    # =========================================================================

    if verbose:
        print(f"\n{'-'*40}")
        print("NULL VALUES")
        print("-" * 40)

    for col in ["signal_id", "I"]:
        null_count = df[col].null_count()
        if null_count > 0:
            result.error("NO_NULLS", f"{col} has {null_count} null values")
            if verbose:
                print(f"[X] {col}: {null_count} nulls")
        else:
            if verbose:
                print(f"[OK] {col}: 0 nulls")

    # value can have nulls (NaN)
    value_nulls = df["value"].null_count()
    if verbose:
        if value_nulls > 0:
            print(f"[i] value: {value_nulls} nulls (allowed)")
        else:
            print(f"[OK] value: 0 nulls")

    # =========================================================================
    # SUMMARY
    # =========================================================================

    if verbose:
        print(f"\n{'='*60}")
        if result.confirmed:
            print("[OK] DATA CONFIRMED - READY FOR PRISM")
            print()
            print("Next step:")
            print(f"   python -m prism {path}")
        else:
            print("[X] DATA NOT CONFIRMED")
            print()
            for err in result.errors:
                print(f"   [{err['rule']}] {err['message']}")
                if err.get('fix'):
                    print(f"      Fix: {err['fix']}")

        if result.warnings:
            print(f"\nWarnings (non-blocking):")
            for warn in result.warnings:
                print(f"   [{warn['rule']}] {warn['message']}")

        if result.notes:
            print(f"\nNotes:")
            for note in result.notes:
                print(f"   [i] {note}")

        print(f"{'='*60}\n")

    return result


# =============================================================================
# API / AI HELPERS
# =============================================================================

def confirm_for_api(path: Path, schema_path: Optional[Path] = None) -> dict:
    result = confirm_data(path, schema_path, verbose=False)
    return result.to_dict()


def confirm_for_ai(path: Path, schema_path: Optional[Path] = None) -> str:
    result = confirm_data(path, schema_path, verbose=False)

    if result.confirmed:
        return f"""[OK] Data confirmed and ready for PRISM.

Stats:
- {result.stats['rows']:,} total rows
- {result.stats['n_signals']} signals: {result.stats['signals']}
- {result.stats.get('n_units', 1)} unit(s)
- {result.stats['mean_obs']:.0f} observations per group (average)

Ready for compute."""
    else:
        errors = "\n".join([f"- {e['message']}" for e in result.errors])
        return f"""[X] Data NOT ready for PRISM.

Errors:
{errors}

Fix these issues and retry."""


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RUDDER Data Confirmation")
    parser.add_argument("path", type=Path, help="Path to observations.parquet")
    parser.add_argument("--schema", type=Path, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    if args.json:
        result = confirm_for_api(args.path, args.schema)
        print(json.dumps(result, indent=2))
    else:
        result = confirm_data(args.path, args.schema, verbose=not args.quiet)
        sys.exit(0 if result.confirmed else 1)
