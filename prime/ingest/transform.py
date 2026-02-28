"""
Universal Data Transformer

Raw data (any format) -> observations.parquet (canonical format)

Schema v3.0:
- REQUIRED: signal_id, signal_0, value
- OPTIONAL: cohort (grouping key, replaces legacy unit_id)
"""

import polars as pl
from pathlib import Path
from typing import Optional


# =============================================================================
# SCHEMA CONSTANTS
# =============================================================================

REQUIRED_COLUMNS = ["signal_id", "signal_0", "value"]
OPTIONAL_COLUMNS = ["cohort"]


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifold_schema(df: pl.DataFrame) -> tuple[bool, list[str]]:
    """
    Validate DataFrame meets Manifold requirements.

    Required: signal_id, signal_0, value
    Optional: cohort (grouping key, replaces legacy unit_id)
    """
    errors = []

    # Check required columns
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            errors.append(f"Missing REQUIRED column: {col}")

    if errors:
        return False, errors

    # Determine grouping column (cohort is new, unit_id is legacy)
    if "cohort" in df.columns:
        group_col = "cohort"
    elif "unit_id" in df.columns:
        group_col = "unit_id"
    else:
        group_col = None

    # Check types
    if df["signal_id"].dtype not in [pl.String, pl.Utf8]:
        errors.append(f"signal_id must be String, got {df['signal_id'].dtype}")

    if df["signal_0"].dtype != pl.Float64:
        errors.append(f"signal_0 must be Float64, got {df['signal_0'].dtype}")

    if df["value"].dtype not in [pl.Float64, pl.Float32]:
        errors.append(f"value must be Float, got {df['value'].dtype}")

    # Group columns for signal_0 checks
    group_cols = [group_col, "signal_id"] if group_col else ["signal_id"]

    # Check signal_0 is sorted ascending per group (no nulls)
    sort_check = (
        df.group_by(group_cols)
        .agg([
            (pl.col("signal_0").diff().drop_nulls() < 0).any().alias("has_unsorted"),
            pl.col("signal_0").null_count().alias("null_count"),
        ])
    )

    unsorted = sort_check.filter(pl.col("has_unsorted"))
    if len(unsorted) > 0:
        errors.append(f"signal_0 is not sorted ascending for {len(unsorted)} groups")

    has_nulls = sort_check.filter(pl.col("null_count") > 0)
    if len(has_nulls) > 0:
        errors.append(f"signal_0 has nulls in {len(has_nulls)} groups")

    # Check at least 2 signals
    n_signals = df["signal_id"].n_unique()
    if n_signals < 2:
        errors.append(f"Need >=2 signals, found {n_signals}")

    # Check for nulls in required columns
    for col in ["signal_id", "signal_0"]:
        null_count = df[col].null_count()
        if null_count > 0:
            errors.append(f"{col} has {null_count} null values")

    return len(errors) == 0, errors


# =============================================================================
# TRANSFORM FUNCTIONS
# =============================================================================

def transform_wide_to_long(
    df: pl.DataFrame,
    signal_columns: list[str],
    unit_column: Optional[str] = None,
    index_column: Optional[str] = None,
    default_cohort: str = "",
) -> pl.DataFrame:
    """
    Transform wide format (signals as columns) to canonical long format.

    Wide format:
        cohort    | timestamp | acc_x | acc_y | temp
        bearing_1 | 0         | 1.23  | 4.56  | 25.0

    Long format:
        cohort    | signal_0 | signal_id | value
        bearing_1 | 0.0      | acc_x     | 1.23
        bearing_1 | 0.0      | acc_y     | 4.56
        bearing_1 | 0.0      | temp      | 25.0

    Args:
        df: Input DataFrame (wide format)
        signal_columns: List of columns that are signals
        unit_column: Optional column to use as cohort (blank if None)
        index_column: Optional column to use as signal_0 (auto-generate if None)
    """

    # Handle cohort
    if unit_column and unit_column in df.columns:
        df = df.with_columns(pl.col(unit_column).cast(pl.String).alias("cohort"))
    else:
        # No unit column — use default_cohort (derived from directory name or empty)
        df = df.with_columns(pl.lit(default_cohort).alias("cohort"))

    # Handle signal_0 (index)
    if index_column and index_column in df.columns:
        df = df.with_columns(pl.col(index_column).cast(pl.Float64).alias("signal_0"))
    else:
        # Generate sequential signal_0 per unit
        df = df.with_row_index("_seq")
        df = df.with_columns(pl.col("_seq").cast(pl.Float64).alias("signal_0")).drop("_seq")

    # Melt wide to long
    id_vars = ["cohort", "signal_0"]
    df_subset = df.select(id_vars + signal_columns)

    df_long = df_subset.unpivot(
        index=id_vars,
        on=signal_columns,
        variable_name="signal_id",
        value_name="value"
    )

    # Ensure types
    df_long = df_long.with_columns([
        pl.col("cohort").cast(pl.String),
        pl.col("signal_0").cast(pl.Float64),
        pl.col("signal_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    ])

    # Sort
    df_long = df_long.sort(["cohort", "signal_0", "signal_id"])

    return df_long


def ensure_signal_0_sorted(df: pl.DataFrame) -> pl.DataFrame:
    """
    Ensure signal_0 is sorted ascending per group. Drop exact duplicates.
    """
    if "cohort" in df.columns:
        group_cols = ["cohort", "signal_id"]
    elif "unit_id" in df.columns:
        group_cols = ["unit_id", "signal_id"]
    else:
        group_cols = ["signal_id"]

    df = df.unique(subset=group_cols + ["signal_0"], keep="first")
    df = df.sort(group_cols + ["signal_0"])

    return df


def transform_to_manifold_format(
    input_path: Path,
    output_path: Path,
    unit_column: Optional[str] = None,
    index_column: Optional[str] = None,
    signal_columns: Optional[list[str]] = None,
    is_wide: bool = True,
    fix_sparse: bool = True,
) -> pl.DataFrame:
    """
    Main entry point: Transform any dataset to canonical format.

    Args:
        input_path: Path to raw data (parquet, csv)
        output_path: Path for observations.parquet
        unit_column: Column name for cohort (optional, blank if None)
        index_column: Column name for signal_0 (optional, auto-generate if None)
        signal_columns: List of signal column names (auto-detect if None)
        is_wide: True if signals are columns, False if already long
        fix_sparse: True to sort and deduplicate signal_0

    Returns:
        Validated DataFrame in canonical format
    """

    # Load data
    if input_path.suffix == ".parquet":
        df = pl.read_parquet(input_path)
    elif input_path.suffix == ".csv":
        df = pl.read_csv(input_path)
    elif input_path.suffix == ".txt":
        df = pl.read_csv(input_path, separator=" ", has_header=False, truncate_ragged_lines=True)
    else:
        raise ValueError(f"Unsupported format: {input_path.suffix}")

    print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
    print(f"Columns: {df.columns}")

    # Derive default cohort from output directory name
    default_cohort = output_path.parent.name if output_path else ""

    # Transform based on format
    if is_wide:
        if signal_columns is None:
            # Auto-detect: numeric columns that aren't unit/index
            exclude = {unit_column, index_column, "timestamp", "date", "time", "cohort", "unit_id", "signal_0"}
            signal_columns = [
                c for c in df.columns
                if c not in exclude and c is not None
                and df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]
            ]
            print(f"Auto-detected signal columns: {signal_columns}")

        df = transform_wide_to_long(df, signal_columns, unit_column, index_column, default_cohort)
    else:
        # Already long format, ensure column names
        if "signal_id" not in df.columns:
            raise ValueError("Long format requires 'signal_id' column")
        if "signal_0" not in df.columns:
            if "I" in df.columns:
                df = df.rename({"I": "signal_0"})
            elif index_column:
                df = df.rename({index_column: "signal_0"})
        if "cohort" not in df.columns:
            if unit_column and unit_column in df.columns:
                df = df.rename({unit_column: "cohort"})
            elif "unit_id" in df.columns:
                df = df.rename({"unit_id": "cohort"})
            else:
                df = df.with_columns(pl.lit(default_cohort).alias("cohort"))

    # Ensure signal_0 is Float64
    if "signal_0" in df.columns and df["signal_0"].dtype != pl.Float64:
        df = df.with_columns(pl.col("signal_0").cast(pl.Float64))

    # Sort and deduplicate signal_0 if needed
    if fix_sparse:
        df = ensure_signal_0_sorted(df)

    # Validate
    is_valid, errors = validate_manifold_schema(df)

    if not is_valid:
        print("\n[X] VALIDATION FAILED:")
        for error in errors:
            print(f"   - {error}")
        raise ValueError(f"Data does not meet schema requirements: {errors}")

    print("\n[OK] VALIDATION PASSED")

    # Summary stats
    group_col = "cohort" if "cohort" in df.columns else "unit_id" if "unit_id" in df.columns else None
    if group_col:
        n_cohorts = df[group_col].n_unique()
        n_signals = df["signal_id"].n_unique()
        n_obs = df.group_by([group_col, "signal_id"]).agg(pl.len()).select(pl.len()).mean().item()
        print(f"   Cohorts: {n_cohorts}" + (" (blank)" if n_cohorts == 1 and df[group_col][0] == "" else ""))
    else:
        n_signals = df["signal_id"].n_unique()
        n_obs = df.group_by(["signal_id"]).agg(pl.len()).select(pl.len()).mean().item()

    print(f"   Signals: {n_signals}")
    print(f"   Avg observations per group: {n_obs:.0f}")
    print(f"   Total rows: {df.shape[0]:,}")

    # Ensure unit column exists (default to "" for dimensionless/unknown)
    if "unit" not in df.columns:
        df = df.with_columns(pl.lit("").alias("unit"))

    # Select final columns in order
    final_cols = ["cohort", "signal_0", "signal_id", "value", "unit"] if "cohort" in df.columns else ["signal_0", "signal_id", "value", "unit"]
    df = df.select(final_cols)

    # Write output
    df.write_parquet(output_path)
    print(f"\n[OK] Written: {output_path}")

    # Write signals.parquet alongside observations.parquet
    from prime.ingest.signal_metadata import write_signal_metadata
    write_signal_metadata(df, output_path.parent)

    return df


# =============================================================================
# Dataset-Specific Transforms
# =============================================================================

def transform_femto(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform FEMTO bearing dataset to canonical format.

    FEMTO: cohort = bearing, signals = acc_x, acc_y
    """
    df = pl.read_parquet(raw_path)
    print(f"FEMTO raw columns: {df.columns}")

    signal_columns = ["acc_x", "acc_y"]

    return transform_to_manifold_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="entity_id",  # Original column name
        index_column="I",
        signal_columns=signal_columns,
        is_wide=True,
        fix_sparse=True,
    )


def transform_femto_bearing_dirs(
    source_dir: Path,
    output_dir: Path,
    fs: float = 25600.0,
    hf_cutoff_hz: float = 5000.0,
) -> pl.DataFrame:
    """
    Transform FEMTO bearing raw CSV directory tree to canonical observations.parquet.

    Each subdirectory of source_dir is one bearing (cohort). Each acc_NNNNN.csv
    is one 2560-sample recording. Columns 4 and 5 (0-indexed) are horizontal and
    vertical acceleration in g. Six features are computed per channel → 12 signal_ids.

    Signal IDs: rms_h/v, peak_h/v, kurtosis_h/v, crest_factor_h/v,
                spectral_centroid_h/v, hf_energy_ratio_h/v

    Args:
        source_dir: Directory containing bearing subdirs (e.g. Learning_set/)
        output_dir: Directory to write observations.parquet + signals.parquet
        fs:         Sampling frequency in Hz (FEMTO standard: 25600.0)
        hf_cutoff_hz: High-frequency cutoff for HF energy ratio feature

    Returns:
        Canonical long-format DataFrame (signal_0, signal_id, value, cohort)
    """
    import numpy as np
    import pandas as pd
    from scipy import stats as scipy_stats

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples = 2560
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    hf_mask = freqs >= hf_cutoff_hz

    feature_names = [
        "rms", "peak", "kurtosis", "crest_factor",
        "spectral_centroid", "hf_energy_ratio",
    ]
    signal_ids_ordered = [f"{f}_{c}" for f in feature_names for c in ("h", "v")]

    bearing_dirs = sorted([d for d in source_dir.iterdir() if d.is_dir()])
    if not bearing_dirs:
        raise ValueError(f"No bearing subdirectories found in {source_dir}")

    print(f"  Source:   {source_dir}")
    print(f"  Bearings: {[d.name for d in bearing_dirs]}")

    all_signal_0: list = []
    all_signal_id: list = []
    all_value: list = []
    all_cohort: list = []

    for bearing_dir in bearing_dirs:
        cohort = bearing_dir.name
        csv_files = sorted(bearing_dir.glob("acc_*.csv"))
        n = len(csv_files)
        print(f"  {cohort}: {n:,} recordings", flush=True)

        feat: dict = {sid: np.empty(n, dtype=np.float64) for sid in signal_ids_ordered}

        # Detect delimiter from the first file (some bearings use semicolons)
        sep = ","
        if csv_files:
            with open(csv_files[0]) as _f:
                sep = ";" if ";" in _f.readline() else ","

        for i, csv_path in enumerate(csv_files):
            try:
                raw = pd.read_csv(
                    csv_path, header=None, usecols=[4, 5],
                    dtype=np.float64, sep=sep,
                ).to_numpy()
            except Exception:
                raw = np.full((n_samples, 2), np.nan)

            if raw.shape[0] != n_samples:
                raw = np.full((n_samples, 2), np.nan)

            for col_idx, suffix in ((0, "_h"), (1, "_v")):
                ch = raw[:, col_idx]
                ch_clean = ch[~np.isnan(ch)]

                rms = float(np.sqrt(np.nanmean(ch ** 2)))
                peak = float(np.nanmax(np.abs(ch)))
                kurt = float(scipy_stats.kurtosis(ch_clean)) if len(ch_clean) > 3 else np.nan
                crest = float(peak / rms) if rms > 0.0 else 0.0

                spec = np.abs(np.fft.rfft(np.nan_to_num(ch)))
                total = float(np.sum(spec ** 2))
                sc = float(np.sum(freqs * spec ** 2) / total) if total > 0.0 else 0.0
                hf = float(np.sum(spec[hf_mask] ** 2) / total) if total > 0.0 else 0.0

                feat[f"rms{suffix}"][i] = rms
                feat[f"peak{suffix}"][i] = peak
                feat[f"kurtosis{suffix}"][i] = kurt
                feat[f"crest_factor{suffix}"][i] = crest
                feat[f"spectral_centroid{suffix}"][i] = sc
                feat[f"hf_energy_ratio{suffix}"][i] = hf

        signal_0_arr = np.arange(1, n + 1, dtype=np.float64).tolist()
        for sid in signal_ids_ordered:
            all_signal_0.extend(signal_0_arr)
            all_signal_id.extend([sid] * n)
            all_value.extend(feat[sid].tolist())
            all_cohort.extend([cohort] * n)

    df = pl.DataFrame(
        {
            "signal_0": all_signal_0,
            "signal_id": all_signal_id,
            "value": all_value,
            "cohort": all_cohort,
        },
        schema={
            "signal_0": pl.Float64,
            "signal_id": pl.String,
            "value": pl.Float64,
            "cohort": pl.String,
        },
    )
    df = df.sort(["cohort", "signal_id", "signal_0"])

    output_path = output_dir / "observations.parquet"
    df.write_parquet(output_path)

    print(f"\n  Written: {output_path}")
    print(f"    Rows:     {len(df):,}")
    print(f"    Cohorts:  {df['cohort'].n_unique()} bearings")
    print(f"    Signals:  {df['signal_id'].n_unique()}")
    print(f"    Max signal_0: {int(df['signal_0'].max())}")
    print(f"    Size:     {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Signal metadata
    from prime.ingest.signal_metadata import write_signal_metadata

    feat_labels = {
        "rms": "RMS acceleration",
        "peak": "Peak absolute acceleration",
        "kurtosis": "Kurtosis",
        "crest_factor": "Crest factor (peak/RMS)",
        "spectral_centroid": "Spectral centroid",
        "hf_energy_ratio": f"HF energy ratio (>{hf_cutoff_hz:.0f} Hz / total)",
    }
    feat_units = {
        "rms": "g", "peak": "g",
        "kurtosis": "", "crest_factor": "",
        "spectral_centroid": "Hz", "hf_energy_ratio": "",
    }
    units = {}
    descriptions = {}
    for sid in signal_ids_ordered:
        feat_name, ch = sid.rsplit("_", 1)
        channel_label = "horizontal" if ch == "h" else "vertical"
        units[sid] = feat_units[feat_name]
        descriptions[sid] = f"{feat_labels[feat_name]} — {channel_label} channel"

    write_signal_metadata(
        df, output_dir,
        units=units,
        descriptions=descriptions,
        signal_0_unit="recording",
        signal_0_description="Sequential recording index (acc_00001 = 1)",
    )

    return df


def transform_skab(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform SKAB dataset to canonical format.

    SKAB: cohort = experiment, signals = sensor columns
    """
    df = pl.read_parquet(raw_path)
    print(f"SKAB raw columns: {df.columns}")

    return transform_to_manifold_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="entity_id",
        index_column="I",
        signal_columns=None,  # auto-detect
        is_wide=True,
        fix_sparse=True,
    )


def transform_fama_french(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform Fama-French dataset to canonical format.

    Fama-French: cohort = blank (single cohort), signals = industries
    """
    return transform_to_manifold_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column=None,  # No unit column - blank is fine
        index_column=None,  # Auto-generate
        signal_columns=None,  # auto-detect
        is_wide=True,
        fix_sparse=True,
    )


def transform_cmapss(raw_path: Path, output_path: Path) -> pl.DataFrame:
    """
    Transform NASA C-MAPSS turbofan dataset to canonical format.

    C-MAPSS: cohort = engine, signals = sensors
    """
    signal_columns = [f"sensor_{i}" for i in range(1, 22)]

    return transform_to_manifold_format(
        input_path=raw_path,
        output_path=output_path,
        unit_column="unit_id",
        index_column="cycle",
        signal_columns=signal_columns,
        is_wide=True,
        fix_sparse=True,
    )


def transform_nasa_battery_dirs(
    source_dir: Path,
    output_dir: Path,
    eol_capacity_ahr: float = 1.4,
) -> pl.DataFrame:
    """
    Transform NASA Li-ion Battery Aging dataset to canonical observations.parquet
    plus impedance.parquet, charge.parquet, ground_truth.parquet, conditions.parquet.

    Scans all subdirectories of source_dir for .mat files. Supports both extracted
    directories and .zip archives. Deduplicates by battery ID (first occurrence wins).

    Architecture: extract 5 source tables → DuckDB JOIN → wide table → unpivot → long.

    Output files:
        observations.parquet  — per-discharge-cycle features (29 signals), long format
        impedance.parquet     — per-impedance-cycle EIS measurements (discharge-aligned)
        charge.parquet        — per-charge-cycle CC/CV analysis (discharge-aligned)
        ground_truth.parquet  — capacity, RUL, quality flags per discharge cycle
        conditions.parquet    — per-battery operating conditions
        signals.parquet       — signal metadata sidecar

    Prime schema mapping:
        signal_0 = discharge cycle number (1-indexed, sequential per battery)
        cohort   = battery ID (e.g. "B0005")
        signals  = 29 per-cycle features from discharge + charge + impedance + conditions

    Cycle alignment keys:
        charge.discharge_cycle  = discharge_num + 1 (this charge prepares NEXT discharge)
        impedance.discharge_cycle = discharge_num   (this impedance follows LAST discharge)
        Multiple impedances per discharge cycle → ARG_MAX keeps last (highest imp_seq)
    """
    import io, numpy as np
    import scipy.io
    import duckdb

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Signal columns by source table
    DISCHARGE_COLS = [
        "discharge_capacity_Ah", "discharge_duration_s",
        "voltage_start_V", "voltage_end_V", "voltage_mean_V", "voltage_std_V",
        "voltage_slope_V_per_s", "voltage_knee_pct",
        "current_mean_A", "current_std_A",
        "temp_start_C", "temp_end_C", "temp_max_C", "temp_rise_C",
        "energy_Wh", "avg_power_W",
    ]
    CHARGE_COLS = [
        "charge_duration_s", "cc_duration_s", "cv_duration_s",
        "cc_cv_ratio", "charge_capacity_Ah", "charge_temp_rise_C",
    ]
    IMP_COLS = ["Re_Ohm", "Rct_Ohm", "imp_min_Ohm", "imp_max_Ohm", "imp_mean_Ohm"]
    COND_COLS = ["ambient_temp_C", "discharge_current_A"]
    ALL_SIGNAL_COLS = DISCHARGE_COLS + CHARGE_COLS + IMP_COLS + COND_COLS

    # Per-battery accumulators
    dis_rows: list[dict] = []   # one row per discharge cycle (wide format)
    chg_rows: list[dict] = []   # one row per charge cycle, discharge-aligned
    imp_rows: list[dict] = []   # one row per impedance cycle, discharge-aligned + seq
    gt_rows: list[dict] = []
    cond_rows: list[dict] = []

    # Collect all .mat paths, deduplicating by battery ID
    seen: set[str] = set()
    mat_entries: list[tuple[str, Path, str | None]] = []

    for mat_path in sorted(source_dir.rglob("*.mat")):
        bid = mat_path.stem
        if bid not in seen and not bid.startswith("_"):
            seen.add(bid)
            mat_entries.append((bid, mat_path, None))

    import zipfile
    for zip_path in sorted(source_dir.glob("*.zip")):
        with zipfile.ZipFile(zip_path) as zf:
            for name in sorted(f for f in zf.namelist() if f.endswith(".mat")):
                bid = name.replace(".mat", "")
                if bid not in seen:
                    seen.add(bid)
                    mat_entries.append((bid, zip_path, name))

    if not mat_entries:
        raise ValueError(f"No .mat files found in {source_dir}")

    n_found = len(mat_entries)
    print(f"  Found {n_found} batteries in {source_dir}")

    def _load(bid: str, fpath: Path, name_in_zip: str | None) -> dict:
        if name_in_zip is None:
            return scipy.io.loadmat(str(fpath), simplify_cells=True)
        with zipfile.ZipFile(fpath) as zf:
            return scipy.io.loadmat(io.BytesIO(zf.read(name_in_zip)), simplify_cells=True)

    for battery_id, fpath, name_in_zip in mat_entries:
        try:
            mat = _load(battery_id, fpath, name_in_zip)
        except Exception as e:
            print(f"  [WARN] Could not load {battery_id}: {e}")
            continue

        battery_key = [k for k in mat if not k.startswith("__")][0]
        cycles = mat[battery_key]["cycle"]
        group_name = fpath.parent.name if name_in_zip is None else fpath.stem

        # Per-battery state
        discharge_num = charge_num = imp_num = 0
        all_caps: list[float] = []
        amb_temps: list[float] = []
        discharge_currents: list[float] = []
        cutoff_voltages: list[float] = []

        for c in cycles:
            ctype = c["type"]
            d = c["data"]
            amb_temp = float(c.get("ambient_temperature", np.nan))
            amb_temps.append(amb_temp)

            # ----------------------------------------------------------------
            # DISCHARGE — wide row, signal_0 = discharge_num
            # ----------------------------------------------------------------
            if ctype == "discharge":
                discharge_num += 1
                v = np.asarray(d.get("Voltage_measured", []), dtype=float)
                i = np.asarray(d.get("Current_measured", []), dtype=float)
                t = np.asarray(d.get("Temperature_measured", []), dtype=float)
                time = np.asarray(d.get("Time", []), dtype=float)

                cap_raw = d.get("Capacity", None)
                cap_arr = np.atleast_1d(cap_raw) if cap_raw is not None else np.array([])
                cap = float(cap_arr[-1]) if len(cap_arr) > 0 else np.nan
                all_caps.append(cap)

                n = len(v)
                if n < 2:
                    # Preserve cycle numbering with NaN row
                    row = {"cohort": battery_id, "signal_0": float(discharge_num)}
                    for col in DISCHARGE_COLS:
                        row[col] = np.nan
                    dis_rows.append(row)
                    continue

                dur = float(time[-1] - time[0]) if len(time) >= 2 else np.nan

                v_start = float(v[0])
                v_end = float(v[-1])
                v_mean = float(np.nanmean(v))
                v_std = float(np.nanstd(v))

                if dur and dur > 0 and len(time) == n:
                    try:
                        v_slope = float(np.polyfit(time - time[0], v, 1)[0])
                    except Exception:
                        v_slope = np.nan
                else:
                    v_slope = np.nan

                below_3v = np.where(v < 3.0)[0]
                if len(below_3v) > 0 and dur and dur > 0 and len(time) == n:
                    knee_pct = float((time[below_3v[0]] - time[0]) / dur * 100.0)
                else:
                    knee_pct = 100.0

                i_abs = np.abs(i)
                i_mean = float(np.nanmean(i_abs))
                i_std = float(np.nanstd(i_abs))
                discharge_currents.append(i_mean)
                cutoff_voltages.append(v_end)

                t_start = float(t[0]) if len(t) > 0 else np.nan
                t_end = float(t[-1]) if len(t) > 0 else np.nan
                t_max = float(np.nanmax(t)) if len(t) > 0 else np.nan
                t_rise = (
                    float(t_end - t_start)
                    if np.isfinite(t_start) and np.isfinite(t_end)
                    else np.nan
                )

                if len(time) >= 2 and len(v) >= 2 and len(i) >= 2:
                    dt = np.diff(time)
                    v_mid = (v[:-1] + v[1:]) / 2
                    i_mid = np.abs(i[:-1] + i[1:]) / 2
                    energy_j = float(np.nansum(v_mid * i_mid * dt))
                    energy_wh = energy_j / 3600.0
                    avg_power = float(energy_j / dur) if dur and dur > 0 else np.nan
                else:
                    energy_wh = avg_power = np.nan

                dis_rows.append({
                    "cohort": battery_id,
                    "signal_0": float(discharge_num),
                    "discharge_capacity_Ah": cap,
                    "discharge_duration_s": dur,
                    "voltage_start_V": v_start,
                    "voltage_end_V": v_end,
                    "voltage_mean_V": v_mean,
                    "voltage_std_V": v_std,
                    "voltage_slope_V_per_s": v_slope,
                    "voltage_knee_pct": knee_pct,
                    "current_mean_A": i_mean,
                    "current_std_A": i_std,
                    "temp_start_C": t_start,
                    "temp_end_C": t_end,
                    "temp_max_C": t_max,
                    "temp_rise_C": t_rise,
                    "energy_Wh": energy_wh,
                    "avg_power_W": avg_power,
                })

            # ----------------------------------------------------------------
            # CHARGE — discharge_cycle = discharge_num + 1 (prepares next discharge)
            # ----------------------------------------------------------------
            elif ctype == "charge":
                charge_num += 1
                v = np.asarray(d.get("Voltage_measured", []), dtype=float)
                i = np.asarray(d.get("Current_measured", []), dtype=float)
                t = np.asarray(d.get("Temperature_measured", []), dtype=float)
                time = np.asarray(d.get("Time", []), dtype=float)

                if len(v) < 2:
                    continue

                dur = float(time[-1] - time[0])

                cv_idx = np.where(v >= 4.15)[0]
                if len(cv_idx) > 0:
                    t_cc = float(time[cv_idx[0]] - time[0])
                    t_cv = float(time[-1] - time[cv_idx[0]])
                else:
                    t_cc, t_cv = dur, 0.0

                cc_ratio = float(t_cc / dur) if dur > 0 else np.nan

                if len(time) >= 2 and len(i) >= 2:
                    dt = np.diff(time)
                    i_mid = np.abs(i[:-1] + i[1:]) / 2
                    chg_cap = float(np.nansum(i_mid * dt) / 3600.0)
                else:
                    chg_cap = np.nan

                t_rise = float(t[-1] - t[0]) if len(t) >= 2 else np.nan

                chg_rows.append({
                    "cohort": battery_id,
                    "discharge_cycle": discharge_num + 1,  # prepares NEXT discharge
                    "charge_duration_s": dur,
                    "cc_duration_s": t_cc,
                    "cv_duration_s": t_cv,
                    "cc_cv_ratio": cc_ratio,
                    "charge_capacity_Ah": chg_cap,
                    "charge_temp_rise_C": t_rise,
                })

            # ----------------------------------------------------------------
            # IMPEDANCE — discharge_cycle = discharge_num (follows LAST discharge)
            #             imp_seq for ARG_MAX deduplication
            # ----------------------------------------------------------------
            elif ctype == "impedance":
                imp_num += 1
                re_arr = np.atleast_1d(d.get("Re", np.nan))
                rct_arr = np.atleast_1d(d.get("Rct", np.nan))
                imp_raw = d.get("Battery_impedance", None)

                re = float(np.real(re_arr[0])) if len(re_arr) > 0 else np.nan
                rct = float(np.real(rct_arr[0])) if len(rct_arr) > 0 else np.nan

                imp_min = imp_max = imp_mean = np.nan
                if imp_raw is not None:
                    imp_mag = np.abs(np.atleast_1d(imp_raw).astype(complex))
                    finite = imp_mag[np.isfinite(imp_mag)]
                    if len(finite) > 0:
                        imp_min = float(np.min(finite))
                        imp_max = float(np.max(finite))
                        imp_mean = float(np.mean(finite))

                imp_rows.append({
                    "cohort": battery_id,
                    "discharge_cycle": discharge_num,   # follows LAST discharge
                    "imp_seq": imp_num,                 # for ARG_MAX dedup
                    "Re_Ohm": re,
                    "Rct_Ohm": rct,
                    "imp_min_Ohm": imp_min,
                    "imp_max_Ohm": imp_max,
                    "imp_mean_Ohm": imp_mean,
                })

        # ----------------------------------------------------------------
        # GROUND TRUTH per battery
        # ----------------------------------------------------------------
        initial_cap = all_caps[0] if all_caps and np.isfinite(all_caps[0]) else np.nan

        eol_cycle: int | None = None
        for idx in range(len(all_caps) - 1, -1, -1):
            if np.isfinite(all_caps[idx]) and all_caps[idx] >= eol_capacity_ahr:
                eol_cycle = idx + 1
                break

        reached_eol = eol_cycle is not None and eol_cycle < len(all_caps)

        for idx, cap in enumerate(all_caps):
            cycle_num = idx + 1
            rul: int | None = None
            if eol_cycle is not None:
                rul = max(0, eol_cycle - cycle_num)

            cap_pct = (
                float(cap / initial_cap * 100.0)
                if np.isfinite(cap) and np.isfinite(initial_cap) and initial_cap > 0
                else None
            )
            is_eol_val = bool(cap < eol_capacity_ahr) if np.isfinite(cap) else None

            is_anomaly = (
                np.isfinite(cap) and np.isfinite(initial_cap) and initial_cap > 0
                and cap < 0.5 * initial_cap
                and idx + 1 < len(all_caps)
                and np.isfinite(all_caps[idx + 1])
                and all_caps[idx + 1] > 0.75 * initial_cap
            )

            gt_rows.append({
                "cohort": battery_id,
                "signal_0": float(cycle_num),
                "capacity_Ah": float(cap) if np.isfinite(cap) else None,
                "capacity_pct": cap_pct,
                "rul_cycles": rul,
                "eol_threshold_Ah": eol_capacity_ahr,
                "is_eol": is_eol_val,
                "quality_flag": "anomaly_transient" if is_anomaly else "ok",
            })

        # ----------------------------------------------------------------
        # CONDITIONS per battery
        # ----------------------------------------------------------------
        amb_temp_med = float(np.nanmedian(amb_temps)) if amb_temps else None
        discharge_current = float(np.nanmedian(discharge_currents)) if discharge_currents else None
        cutoff_v = float(np.nanmin(cutoff_voltages)) if cutoff_voltages else None
        final_cap = float(all_caps[-1]) if all_caps and np.isfinite(all_caps[-1]) else None

        cond_rows.append({
            "cohort": battery_id,
            "group": group_name,
            "ambient_temp_C": amb_temp_med,
            "discharge_current_A": discharge_current,
            "cutoff_voltage_V": cutoff_v,
            "eol_criteria_Ah": eol_capacity_ahr,
            "total_cycles": discharge_num,
            "initial_capacity_Ah": float(initial_cap) if np.isfinite(initial_cap) else None,
            "final_capacity_Ah": final_cap,
            "reached_eol": reached_eol,
        })

        eol_str = f"cycle {eol_cycle}" if eol_cycle is not None else "not reached"
        print(f"  {battery_id}: {discharge_num}d / {charge_num}c / {imp_num}i  EOL {eol_str}")

    # ----------------------------------------------------------------
    # BUILD SOURCE DATAFRAMES
    # ----------------------------------------------------------------
    dis_df = pl.DataFrame(
        dis_rows,
        schema={"cohort": pl.String, "signal_0": pl.Float64}
        | {c: pl.Float64 for c in DISCHARGE_COLS},
    )

    chg_df = pl.DataFrame(
        chg_rows,
        schema={"cohort": pl.String, "discharge_cycle": pl.Int64}
        | {c: pl.Float64 for c in CHARGE_COLS},
    ) if chg_rows else pl.DataFrame(
        schema={"cohort": pl.String, "discharge_cycle": pl.Int64}
        | {c: pl.Float64 for c in CHARGE_COLS}
    )

    imp_df = pl.DataFrame(
        imp_rows,
        schema={"cohort": pl.String, "discharge_cycle": pl.Int64, "imp_seq": pl.Int64}
        | {c: pl.Float64 for c in IMP_COLS},
    ) if imp_rows else pl.DataFrame(
        schema={"cohort": pl.String, "discharge_cycle": pl.Int64, "imp_seq": pl.Int64}
        | {c: pl.Float64 for c in IMP_COLS}
    )

    gt_df = pl.DataFrame(
        gt_rows,
        schema={
            "cohort": pl.String, "signal_0": pl.Float64,
            "capacity_Ah": pl.Float64, "capacity_pct": pl.Float64,
            "rul_cycles": pl.Int64, "eol_threshold_Ah": pl.Float64,
            "is_eol": pl.Boolean, "quality_flag": pl.String,
        },
    )

    cond_df = pl.DataFrame(
        cond_rows,
        schema={
            "cohort": pl.String, "group": pl.String,
            "ambient_temp_C": pl.Float64, "discharge_current_A": pl.Float64,
            "cutoff_voltage_V": pl.Float64, "eol_criteria_Ah": pl.Float64,
            "total_cycles": pl.Int64,
            "initial_capacity_Ah": pl.Float64, "final_capacity_Ah": pl.Float64,
            "reached_eol": pl.Boolean,
        },
    )

    # ----------------------------------------------------------------
    # WRITE SIDECAR FILES
    # ----------------------------------------------------------------
    # impedance.parquet — keep discharge_cycle alignment key, drop imp_seq
    (
        imp_df.drop("imp_seq")
        .rename({"discharge_cycle": "cycle_number"})
        .sort(["cohort", "cycle_number"])
        .write_parquet(output_dir / "impedance.parquet")
    )

    # charge.parquet
    (
        chg_df.rename({"discharge_cycle": "cycle_number"})
        .sort(["cohort", "cycle_number"])
        .write_parquet(output_dir / "charge.parquet")
    )

    gt_df.sort(["cohort", "signal_0"]).write_parquet(output_dir / "ground_truth.parquet")
    cond_df.sort("cohort").write_parquet(output_dir / "conditions.parquet")

    # ----------------------------------------------------------------
    # SQL JOIN: discharge + charge + impedance + conditions → wide table
    # ----------------------------------------------------------------
    con = duckdb.connect()
    con.register("discharge", dis_df.to_arrow())
    con.register("charge", chg_df.to_arrow())
    con.register("impedance", imp_df.to_arrow())
    con.register("conditions", cond_df.to_arrow())

    imp_dedup_exprs = ", ".join(
        f"ARG_MAX({col}, imp_seq) AS {col}" for col in IMP_COLS
    )
    chg_sel = ", ".join(f"c.{col}" for col in CHARGE_COLS)
    imp_sel = ", ".join(f"i.{col}" for col in IMP_COLS)
    dis_sel = ", ".join(f"d.{col}" for col in DISCHARGE_COLS)

    wide_arrow = con.execute(f"""
        WITH imp_deduped AS (
            SELECT cohort, discharge_cycle, {imp_dedup_exprs}
            FROM impedance
            GROUP BY cohort, discharge_cycle
        )
        SELECT
            d.cohort,
            d.signal_0,
            {dis_sel},
            {chg_sel},
            {imp_sel},
            cond.ambient_temp_C,
            cond.discharge_current_A
        FROM discharge d
        LEFT JOIN charge c
            ON d.cohort = c.cohort AND CAST(d.signal_0 AS BIGINT) = c.discharge_cycle
        LEFT JOIN imp_deduped i
            ON d.cohort = i.cohort AND CAST(d.signal_0 AS BIGINT) = i.discharge_cycle
        LEFT JOIN conditions cond
            ON d.cohort = cond.cohort
        ORDER BY d.cohort, d.signal_0
    """).arrow()

    wide_df = pl.from_arrow(wide_arrow)

    # ----------------------------------------------------------------
    # UNPIVOT wide → canonical long format
    # ----------------------------------------------------------------
    obs_df = (
        wide_df.unpivot(
            index=["cohort", "signal_0"],
            on=ALL_SIGNAL_COLS,
            variable_name="signal_id",
            value_name="value",
        )
        .sort(["cohort", "signal_id", "signal_0"])
    )

    obs_path = output_dir / "observations.parquet"
    obs_df.write_parquet(obs_path)

    # ----------------------------------------------------------------
    # SIGNALS METADATA
    # ----------------------------------------------------------------
    from prime.ingest.signal_metadata import write_signal_metadata

    units = {
        "discharge_capacity_Ah": "Ah",
        "discharge_duration_s": "s",
        "voltage_start_V": "V", "voltage_end_V": "V",
        "voltage_mean_V": "V", "voltage_std_V": "V",
        "voltage_slope_V_per_s": "V/s",
        "voltage_knee_pct": "%",
        "current_mean_A": "A", "current_std_A": "A",
        "temp_start_C": "°C", "temp_end_C": "°C",
        "temp_max_C": "°C", "temp_rise_C": "°C",
        "energy_Wh": "Wh", "avg_power_W": "W",
        "charge_duration_s": "s", "cc_duration_s": "s", "cv_duration_s": "s",
        "cc_cv_ratio": "", "charge_capacity_Ah": "Ah", "charge_temp_rise_C": "°C",
        "Re_Ohm": "Ω", "Rct_Ohm": "Ω",
        "imp_min_Ohm": "Ω", "imp_max_Ohm": "Ω", "imp_mean_Ohm": "Ω",
        "ambient_temp_C": "°C", "discharge_current_A": "A",
    }
    descriptions = {
        "discharge_capacity_Ah": "Discharge capacity — primary degradation indicator",
        "discharge_duration_s": "Total discharge time",
        "voltage_start_V": "Terminal voltage at start of discharge",
        "voltage_end_V": "Terminal voltage at end of discharge (cutoff)",
        "voltage_mean_V": "Mean terminal voltage during discharge",
        "voltage_std_V": "Std of terminal voltage during discharge",
        "voltage_slope_V_per_s": "Linear fit slope of V vs time (rate of voltage drop)",
        "voltage_knee_pct": "Discharge fraction (%) before V drops below 3.0V",
        "current_mean_A": "Mean discharge current magnitude",
        "current_std_A": "Std of discharge current (≈0 for CC, >0 for pulsed)",
        "temp_start_C": "Battery temperature at start of discharge",
        "temp_end_C": "Battery temperature at end of discharge",
        "temp_max_C": "Peak battery temperature during discharge",
        "temp_rise_C": "Temperature rise during discharge (temp_end − temp_start)",
        "energy_Wh": "Total energy delivered (∫V·I·dt / 3600)",
        "avg_power_W": "Average power (energy / duration)",
        "charge_duration_s": "Total charge time (CC + CV phases)",
        "cc_duration_s": "Constant-current phase duration",
        "cv_duration_s": "Constant-voltage phase duration",
        "cc_cv_ratio": "CC fraction of total charge time (CC / (CC+CV))",
        "charge_capacity_Ah": "Charge capacity delivered (∫I·dt / 3600)",
        "charge_temp_rise_C": "Temperature rise during charge cycle",
        "Re_Ohm": "Electrolyte resistance from EIS (real part of Z at high frequency)",
        "Rct_Ohm": "Charge transfer resistance from EIS",
        "imp_min_Ohm": "Minimum impedance magnitude across EIS frequency sweep",
        "imp_max_Ohm": "Maximum impedance magnitude across EIS frequency sweep",
        "imp_mean_Ohm": "Mean impedance magnitude across EIS frequency sweep",
        "ambient_temp_C": "Ambient temperature during battery operation",
        "discharge_current_A": "Nominal discharge current (median across all cycles)",
    }
    write_signal_metadata(
        obs_df, output_dir,
        units=units,
        descriptions=descriptions,
        signal_0_unit="cycle",
        signal_0_description="Discharge cycle number (1-indexed, sequential per battery)",
    )

    n_batteries = obs_df["cohort"].n_unique()
    print(f"\n  Written: {obs_path}")
    print(f"    Rows:          {len(obs_df):,}")
    print(f"    Batteries:     {n_batteries} / {n_found}")
    print(f"    Signals:       {obs_df['signal_id'].n_unique()}")
    print(f"    Max cycle:     {int(obs_df['signal_0'].max())}")
    print(f"    Size:          {obs_path.stat().st_size / 1024:.1f} KB")
    print(f"  impedance.parquet: {len(imp_rows):,} rows")
    print(f"  charge.parquet:    {len(chg_rows):,} rows")
    print(f"  ground_truth.parquet: {len(gt_rows):,} rows")
    print(f"  conditions.parquet:   {n_batteries} rows")

    return obs_df


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transform data to canonical format")
    parser.add_argument("input", type=Path, help="Input file (parquet or csv)")
    parser.add_argument("output", type=Path, help="Output observations.parquet")
    parser.add_argument("--unit", type=str, default=None, help="Unit column name (optional)")
    parser.add_argument("--index", type=str, default=None, help="Index column name")
    parser.add_argument("--signals", type=str, nargs="+", default=None, help="Signal column names")
    parser.add_argument("--long", action="store_true", help="Input is already long format")

    args = parser.parse_args()

    transform_to_manifold_format(
        input_path=args.input,
        output_path=args.output,
        unit_column=args.unit,
        index_column=args.index,
        signal_columns=args.signals,
        is_wide=not args.long,
    )
