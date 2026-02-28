"""
Full pipeline: domain path in, results out.
Every run is fresh. All intermediate files overwritten.
"""

import os
import sys
from pathlib import Path


def _check_dependencies():
    """Check required dependencies. Crash on missing."""
    # Fatal — typology cannot run without pmtvs
    try:
        import pmtvs
    except ImportError:
        print("FATAL: pmtvs not installed.")
        print("Run: pip install pmtvs")
        sys.exit(1)

    # Fatal — compute stage requires orchestration
    # This is part of the repo and cannot be missing.
    try:
        import orchestration
    except ImportError:
        print("FATAL: orchestration package not installed.")
        print("Run: pip install -e packages/orchestration")
        sys.exit(1)


def run_pipeline(domain_path: Path, axis: str = "time", force_ingest: bool = False):
    """
    Run the complete Prime pipeline.

    Args:
        domain_path: Path to domain directory containing raw data
                     (or at minimum, observations.parquet).
        axis: Signal to use as ordering axis. Default "time" uses
              row order (identical to current behavior). Any other
              value selects that signal's values as signal_0.
    """
    _check_dependencies()

    observations_path = domain_path / "observations.parquet"
    output_dir = domain_path / f"output_{axis}"
    typology_raw_path = output_dir / "typology_raw.parquet"
    typology_path = output_dir / "typology.parquet"
    manifest_path = output_dir / "manifest.yaml"

    print(f"=== PRIME: {domain_path.name} (order-by={axis}) ===\n")

    # Overwrite if output directory already exists
    if output_dir.exists():
        print(f"  Overwriting existing output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------------------------------------
    # Step 1: INGEST — raw files → observations.parquet
    # ----------------------------------------------------------
    if observations_path.exists() and not force_ingest:
        print("[1/7] observations.parquet exists — skipping ingest")
    else:
        if force_ingest and observations_path.exists():
            print("[1/7] Re-ingesting raw data (--force-ingest)...")
        else:
            print("[1/7] Ingesting raw data...")
        raw_file = _find_raw_file(domain_path)
        if raw_file is None:
            if not observations_path.exists():
                print(f"  No raw data files found and no observations.parquet. Cannot continue.")
                sys.exit(1)
            print(f"  No raw data files found — using existing observations.parquet")
        else:
            from prime.ingest.from_raw import detect_format, ingest_cmapss, write_observations

            fmt = detect_format(raw_file)
            if fmt == "cmapss":
                df = ingest_cmapss(raw_file)
                write_observations(df, domain_path)
            elif fmt == "matlab":
                from prime.ingest.upload import load_matlab_file
                df = load_matlab_file(raw_file)
                write_observations(df, domain_path)
            else:
                from prime.ingest.transform import transform_to_manifold_format
                transform_to_manifold_format(
                    input_path=raw_file,
                    output_path=observations_path,
                )
            print(f"  → {observations_path}")

    # Axis selection (post-ingest)
    # Each output_dir gets its own observations so typology/manifold are self-contained.
    axis_observations_path = output_dir / "observations.parquet"
    if axis == "time":
        import shutil
        shutil.copy2(observations_path, axis_observations_path)
        print(f"  Copied {observations_path} → {axis_observations_path}")
    else:
        print(f"  Applying axis selection (axis={axis})...")
        from prime.ingest.axis import reaxis_observations
        reaxis_observations(observations_path, axis, axis_observations_path)
    observations_path = axis_observations_path

    # ----------------------------------------------------------
    # Step 1b: REGIME DETECTION — cluster operating conditions
    # Runs on the axis observations. Output feeds both the Prime
    # analytical path (as context) and the ML normalization path.
    # ----------------------------------------------------------
    print("\n--- Step 1b: Regime Detection ---")
    regime_path = output_dir / "regime_detection.parquet"
    try:
        import polars as pl
        from prime.regime.detection import detect_regimes

        obs_for_regime = pl.read_parquet(observations_path)
        regimes_df = detect_regimes(obs_for_regime)
        regimes_df.write_parquet(regime_path)
        n_regimes = regimes_df["regime_id"].n_unique()
        print(f"  → {regime_path} ({n_regimes} regime(s))")
    except Exception as e:
        print(f"  [regime] WARNING: {e}")
        print("  [regime] Regime detection failed — skipping regime-normalized ML export.")
        regime_path = None

    # ----------------------------------------------------------
    # Steps 2-3: TYPOLOGY — observations → classified typology
    # ----------------------------------------------------------
    # PRIME_TYPOLOGY env var selects the backend:
    #   python  — original Python/pmtvs pipeline (default)
    #   sql     — SQL-first via DuckDB (fast)
    #   compare — run both, print diagnostics, use Python as canonical
    typology_backend = os.environ.get("PRIME_TYPOLOGY", "sql").lower()

    if typology_backend == "sql":
        typology, typology_raw = _run_typology_sql(
            observations_path, typology_path, output_dir,
        )
    elif typology_backend == "compare":
        typology, typology_raw = _run_typology_compare(
            observations_path, typology_raw_path, typology_path, output_dir,
        )
    else:
        typology, typology_raw = _run_typology_python(
            observations_path, typology_raw_path, typology_path,
        )

    # ----------------------------------------------------------
    # Step 4: MANIFEST — typology → engine selection per signal
    # ----------------------------------------------------------
    print("[4/7] Generating manifest...")
    from prime.entry_points.stage_04_manifest import run as generate_manifest

    manifest = generate_manifest(
        str(typology_path),
        str(manifest_path),
        observations_path=str(observations_path),
        output_dir=str(output_dir),
        verbose=False,
        axis=axis,
    )
    summary = manifest.get('summary', {})
    print(f"  → {manifest_path}")
    print(f"    Active signals: {summary.get('active_signals', 'N/A')}")
    print(f"    Engines: {len(summary.get('signal_engines', []))}")

    # ----------------------------------------------------------
    # Step 5: COMPUTE — orchestration.run()
    # ----------------------------------------------------------
    print("[5/7] Running compute engine...")

    # Orchestration writes into output_dir (which IS the axis output directory).
    compute_output_dir = Path(manifest['paths']['output_dir'])

    from prime.core.manifold_client import run_manifold

    run_manifold(
        observations_path=observations_path,
        manifest_path=manifest_path,
        output_dir=compute_output_dir,
        verbose=True,
    )
    compute_files = list(compute_output_dir.rglob("*.parquet"))
    print(f"  → {compute_output_dir}/ ({len(compute_files)} output files)")

    # ----------------------------------------------------------
    # Step 5b: ML Export (causal derivatives + pass-through)
    # ----------------------------------------------------------
    print("\n--- Step 5b: ML Export ---")
    try:
        from prime.ml_export import run_ml_export
        run_ml_export(
            output_dir=output_dir,
            cohort_col="cohort",
            window_col="window",
        )
    except Exception as e:
        print(f"  [ml_export] WARNING: {e}")
        print(f"  [ml_export] ML export failed but pipeline continues.")

    # ----------------------------------------------------------
    # Steps 5c/5d: REGIME NORMALIZATION + REGIME ML EXPORT
    # ML path only. Prime's analytical output (steps 1-6) uses raw data.
    # These steps normalize per operating regime and produce ml_normalized_*
    # parquets for downstream prediction models.
    # Skipped automatically if regime detection failed or found 0 regimes.
    # ----------------------------------------------------------
    normalized_obs = None
    if regime_path is not None and regime_path.exists():
        print("\n--- Steps 5c/5d: Regime-Normalized ML Export ---")
        try:
            import json
            import polars as pl
            from prime.regime.normalization import normalize_per_regime, apply_regime_stats, align_regime_ids
            from prime.ml_export.regime_export import run_regime_ml_export

            obs = pl.read_parquet(observations_path)
            regimes_loaded = pl.read_parquet(regime_path)

            # Check for sibling train-split regime stats to prevent test leakage.
            # If this is a test/Test split and a sibling Train run exists with
            # ml_regime_stats.json, use those training stats instead of refitting.
            split_name = domain_path.name  # e.g. "Test", "test", "Train", "train"
            is_test_split = split_name.lower() == "test"
            train_stats_path = None
            if is_test_split:
                for candidate in ("Train", "train"):
                    p = domain_path.parent / candidate / "output_time" / "ml" / "ml_regime_stats.json"
                    if p.exists():
                        train_stats_path = p
                        break

            if train_stats_path is not None:
                with open(train_stats_path) as _f:
                    train_regime_stats = json.load(_f)
                print(f"  [regime] Using training regime stats from {train_stats_path}")
                # CRITICAL: Align test regime IDs to train regime IDs before applying stats.
                # KMeans assigns arbitrary labels — test regime 0 may not equal train regime 0.
                print("  [regime] Aligning test regime IDs to training regime IDs...")
                regimes_aligned = align_regime_ids(regimes_loaded, obs, train_regime_stats)
                normalized_obs = apply_regime_stats(obs, regimes_aligned, train_regime_stats)
                regime_stats = train_regime_stats
                regimes_for_export = regimes_aligned
            else:
                normalized_obs, regime_stats = normalize_per_regime(obs, regimes_loaded)
                regimes_for_export = regimes_loaded

            run_regime_ml_export(
                output_dir=output_dir,
                regimes=regimes_for_export,
                regime_stats=regime_stats,
                normalized_obs=normalized_obs,
            )
        except Exception as e:
            print(f"  [regime_ml] WARNING: {e}")
            print("  [regime_ml] Regime ML export failed but pipeline continues.")
            normalized_obs = None

    # ----------------------------------------------------------
    # Step 5e: MODALITY ML EXPORT
    # Unit-based signal grouping → per-modality RT geometry + cross-modality coupling.
    # Uses regime-normalized observations when available; falls back to raw.
    # ----------------------------------------------------------
    print("\n--- Step 5e: Modality ML Export ---")
    signals_path = domain_path / "signals.parquet"
    if signals_path.exists():
        try:
            import polars as pl
            from prime.modality.export import run_modality_export
            _modality_obs = normalized_obs if normalized_obs is not None else pl.read_parquet(observations_path)
            run_modality_export(
                output_dir=output_dir,
                observations=_modality_obs,
                signals_path=signals_path,
            )
        except Exception as e:
            print(f"  [modality] WARNING: {e}")
            print("  [modality] Modality export failed but pipeline continues.")
    else:
        print("  [modality] No signals.parquet — skipping modality export")

    # ----------------------------------------------------------
    # Step 6: ANALYZE — SQL layers on parquets
    # ----------------------------------------------------------
    print("[6/7] Running SQL analysis...")
    from prime.sql.runner import run_sql_analysis

    try:
        run_sql_analysis(output_dir, domain_dir=domain_path)
    except Exception as e:
        print(f"  SQL analysis: {e}")

    # ----------------------------------------------------------
    # Step 6b: PARAMETERIZATION COMPILATION (if 2+ runs exist)
    # ----------------------------------------------------------
    from prime.parameterization.compile import discover_runs, compile_parameterization

    if len(discover_runs(domain_path)) >= 2:
        print("[6b/7] Compiling cross-run parameterization...")
        compile_parameterization(domain_path, verbose=True)

    # ----------------------------------------------------------
    # Step 7: SUMMARY
    # ----------------------------------------------------------
    print(f"\n[7/7] Done. Run 'prime query {output_dir}' to explore results.\n")
    _print_summary(domain_path, typology_raw, typology, output_dir, output_dir)


def _run_typology_python(observations_path, typology_raw_path, typology_path):
    """Original Python/pmtvs typology pipeline (steps 2 + 3)."""
    import polars as pl

    # Step 2: typology_raw
    if typology_raw_path.exists():
        print("[2/7] Typology raw exists — skipping recomputation")
        typology_raw = pl.read_parquet(typology_raw_path)
        n_signals = len(typology_raw)
        print(f"  → {typology_raw_path} ({n_signals} signals)")
    else:
        print("[2/7] Computing typology (pmtvs)...")
        from prime.ingest.typology_raw import compute_typology_raw

        typology_raw = compute_typology_raw(
            str(observations_path),
            str(typology_raw_path),
            verbose=True,
        )
        n_signals = len(typology_raw)
        print(f"  → {typology_raw_path} ({n_signals} signals)")

    # Step 3: classify
    if typology_path.exists():
        print("[3/7] Typology exists — skipping reclassification")
        typology = pl.read_parquet(typology_path)
        print(f"  → {typology_path}")
    else:
        print("[3/7] Classifying signals...")
        from prime.entry_points.stage_03_classify import run as classify

        typology = classify(
            str(typology_raw_path),
            str(typology_path),
            verbose=False,
        )
        print(f"  → {typology_path}")

    return typology, typology_raw


def _run_typology_sql(observations_path, typology_path, output_dir):
    """SQL-first typology pipeline (steps 2+3 combined)."""
    import polars as pl

    if typology_path.exists():
        print("[2-3/7] Typology exists — skipping")
        typology = pl.read_parquet(typology_path)
        print(f"  → {typology_path} ({len(typology)} signals)")
        return typology, typology

    print("[2-3/7] Computing typology (SQL + pmtvs)...")
    from prime.sql.typology import run_sql_typology
    from prime.sql.typology.compat import adapt_for_manifest

    typology_output = run_sql_typology(
        observations_path=str(observations_path),
        output_dir=str(output_dir),
        window_size=128,
        stride=64,
        verbose=True,
    )
    adapt_for_manifest(typology_output)

    typology = pl.read_parquet(typology_path)
    print(f"  → {typology_path} ({len(typology)} signals)")
    # SQL produces one combined output — use typology as typology_raw too
    return typology, typology


def _run_typology_compare(observations_path, typology_raw_path, typology_path, output_dir):
    """Run both backends, print diagnostics, use Python as canonical."""
    import polars as pl

    # Run Python (canonical)
    print("  [compare] Running Python backend...")
    typology_py, typology_raw = _run_typology_python(
        observations_path, typology_raw_path, typology_path,
    )

    # Run SQL to a temp path (don't overwrite Python's typology.parquet)
    print("  [compare] Running SQL backend...")
    sql_output_dir = output_dir / "_sql_compare"
    sql_output_dir.mkdir(parents=True, exist_ok=True)

    from prime.sql.typology import run_sql_typology
    from prime.sql.typology.compat import adapt_for_manifest

    sql_typology_output = run_sql_typology(
        observations_path=str(observations_path),
        output_dir=str(sql_output_dir),
        window_size=128,
        stride=64,
        verbose=True,
    )
    adapt_for_manifest(sql_typology_output)

    typology_sql = pl.read_parquet(sql_typology_output)

    # Compare common columns
    print("\n  [compare] Column-level diagnostics:")
    common_cols = sorted(set(typology_py.columns) & set(typology_sql.columns))
    py_only = sorted(set(typology_py.columns) - set(typology_sql.columns))
    sql_only = sorted(set(typology_sql.columns) - set(typology_py.columns))

    if py_only:
        print(f"    Python-only columns ({len(py_only)}): {py_only[:10]}")
    if sql_only:
        print(f"    SQL-only columns ({len(sql_only)}): {sql_only[:10]}")

    for col in common_cols:
        if col in ("cohort", "signal_id"):
            continue
        try:
            py_col = typology_py[col]
            sql_col = typology_sql[col]
            if py_col.dtype.is_numeric() and sql_col.dtype.is_numeric():
                diff = (py_col.cast(pl.Float64) - sql_col.cast(pl.Float64)).abs()
                max_diff = diff.drop_nulls().max()
                if max_diff is not None and max_diff > 0.01:
                    print(f"    {col}: max_diff={max_diff:.4f}")
            elif py_col.dtype == pl.Utf8 and sql_col.dtype == pl.Utf8:
                mismatches = (py_col != sql_col).sum()
                if mismatches > 0:
                    print(f"    {col}: {mismatches} mismatches out of {len(py_col)}")
        except Exception:
            pass

    print(f"  [compare] Using Python output as canonical\n")
    return typology_py, typology_raw


def _find_raw_file(domain_path: Path) -> Path | None:
    """Find a raw data file in the domain directory."""
    skip_stems = {
        'observations', 'typology', 'typology_raw', 'validated',
        'signals', 'ground_truth',
        # Battery-domain supplementary files (not raw sensor data)
        'charge', 'impedance', 'conditions',
    }
    for ext in ['*.csv', '*.parquet', '*.xlsx', '*.tsv', '*.txt', '*.mat']:
        candidates = [
            c for c in domain_path.glob(ext)
            if c.stem not in skip_stems
            and not c.stem.endswith('_observations')
        ]
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            return max(candidates, key=lambda f: f.stat().st_size)
    return None


def _print_summary(domain_path, typology_raw, typology, run_dir, output_dir: Path):
    """Print key results after pipeline completion."""
    n_signals = len(typology_raw)
    cohort_col = 'cohort' if 'cohort' in typology_raw.columns else None
    n_cohorts = typology_raw[cohort_col].n_unique() if cohort_col else 1

    print(f"=== RESULTS: {domain_path.name} ===")
    print(f"  Signals:  {n_signals}")
    print(f"  Cohorts:  {n_cohorts}")
    print()

    # Classification summary
    if 'temporal_primary' in typology.columns:
        print("  Temporal patterns:")
        patterns = typology.group_by('temporal_primary').len().sort('len', descending=True)
        for row in patterns.iter_rows(named=True):
            print(f"    {row['temporal_primary']}: {row['len']}")
        print()

    output_files = list(output_dir.rglob("*.parquet")) if output_dir.exists() else []
    if output_files:
        print(f"  Output files ({len(output_files)}):")
        for f in sorted(output_files):
            size_kb = f.stat().st_size / 1024
            rel = f.relative_to(run_dir)
            print(f"    {rel} ({size_kb:.1f} KB)")
    else:
        print("  No output files generated")

    # SQL reports
    sql_dir = run_dir / 'sql'
    if sql_dir.exists():
        md_files = list(sql_dir.glob('*.md'))
        if md_files:
            print(f"\n  SQL reports ({len(md_files)}): {sql_dir}")

    print()
    print(f"  Domain:     {domain_path}")
    print(f"  Output dir: {run_dir}")
