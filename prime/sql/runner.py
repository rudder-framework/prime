"""Run Prime SQL layers against Manifold output and write markdown results."""

import duckdb
from pathlib import Path


def load_manifold_output(
    con: duckdb.DuckDBPyConnection,
    output_dir: Path,
    default_cohort: str = "",
) -> list[str]:
    """Load all parquet files from Manifold output directory as DuckDB views.

    If default_cohort is non-empty, any parquet with an all-empty cohort column
    will have that column replaced with default_cohort at load time.
    """
    loaded = []
    if not output_dir.exists():
        return loaded
    for parquet_file in sorted(output_dir.rglob('*.parquet')):
        view_name = parquet_file.stem  # e.g. state_geometry
        try:
            if default_cohort:
                # Check if this parquet has a cohort column with empty values
                cols = [r[0] for r in con.execute(
                    f"DESCRIBE SELECT * FROM read_parquet('{parquet_file}')"
                ).fetchall()]
                if 'cohort' in cols:
                    other_cols = [c for c in cols if c != 'cohort']
                    col_list = ', '.join(other_cols)
                    con.execute(f"""
                        CREATE OR REPLACE VIEW {view_name} AS
                        SELECT {col_list},
                            CASE WHEN cohort = '' OR cohort IS NULL
                                THEN '{default_cohort}' ELSE cohort
                            END AS cohort
                        FROM read_parquet('{parquet_file}')
                    """)
                    loaded.append(view_name)
                    continue
            con.execute(f"""
                CREATE OR REPLACE VIEW {view_name} AS
                SELECT * FROM read_parquet('{parquet_file}')
            """)
            loaded.append(view_name)
        except Exception as e:
            print(f"  WARNING: Skipping {parquet_file.name}: {e}")
    return loaded


def execute_sql_layer(con: duckdb.DuckDBPyConnection, sql_path: Path) -> list[dict]:
    """Execute a SQL file and return results from each SELECT statement."""
    sql_text = sql_path.read_text()
    results = []

    # Split on semicolons, execute each statement
    for stmt in sql_text.split(';'):
        # Remove .print/.read directives (DuckDB CLI only)
        lines = [line for line in stmt.split('\n')
                 if not line.strip().startswith('.')]
        cleaned = '\n'.join(lines).strip()
        if not cleaned:
            continue

        # Skip blocks that are only comments
        has_sql = any(line.strip() and not line.strip().startswith('--')
                      for line in cleaned.split('\n'))
        if not has_sql:
            continue

        try:
            result = con.execute(cleaned)
            # Capture SELECT results (CREATE VIEW returns nothing useful)
            upper = cleaned.upper().lstrip()
            if upper.startswith('SELECT') or (
                not upper.startswith('CREATE') and 'FROM' in upper
            ):
                df = result.fetchdf()
                if len(df) > 0:
                    results.append({
                        'query': cleaned[:100] + '...' if len(cleaned) > 100 else cleaned,
                        'data': df,
                    })
        except Exception:
            pass  # CREATE VIEW, missing deps — skip silently

    return results


def write_markdown(results: list[dict], output_path: Path, title: str) -> None:
    """Write query results as a markdown file."""
    lines = [f"# {title}", ""]

    for r in results:
        df = r['data']
        cols = list(df.columns)
        lines.append(f"## {r['query'][:80]}")
        lines.append("")

        # Markdown table
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        # Limit to 50 rows in markdown, note if truncated
        display_df = df.head(50)
        for _, row in display_df.iterrows():
            values = [str(v) for v in row]
            lines.append("| " + " | ".join(values) + " |")

        if len(df) > 50:
            lines.append(f"\n*Showing 50 of {len(df):,} rows.*")

        lines.append("")
        lines.append(f"*{len(df):,} rows*")
        lines.append("")
        lines.append("---")
        lines.append("")

    output_path.write_text("\n".join(lines))


def run_sql_analysis(run_dir: Path, domain_dir: Path | None = None) -> None:
    """Main entry point: load Manifold output, run SQL, write markdown.

    Args:
        run_dir: The axis output directory (e.g. domain/output_time/).
                 Manifold parquets are in subdirectories (system/, cohort/).
        domain_dir: Domain root containing observations.parquet and typology.
                    Defaults to run_dir.parent.
    """
    domain_dir = domain_dir or run_dir.parent
    sql_output_dir = run_dir / 'sql'
    sql_output_dir.mkdir(parents=True, exist_ok=True)

    # Find SQL files in the repo
    sql_dir = Path(__file__).parent / 'layers'
    report_dir = Path(__file__).parent / 'reports'

    con = duckdb.connect()

    # Detect if observations have empty cohort — if so, backfill everywhere
    # Prefer per-axis observations from run_dir; fall back to domain root
    domain_name = domain_dir.name
    default_cohort = ""
    obs_path_run = run_dir / 'observations.parquet'
    obs_path = obs_path_run if obs_path_run.exists() else domain_dir / 'observations.parquet'
    if obs_path.exists():
        empty_check = con.execute(
            f"SELECT COUNT(DISTINCT cohort) AS n, MIN(cohort) AS first_val "
            f"FROM read_parquet('{obs_path}')"
        ).fetchone()
        if empty_check and empty_check[0] == 1 and (empty_check[1] == '' or empty_check[1] is None):
            default_cohort = domain_name

    # Load Manifold parquet files from run_dir (output_* IS the Manifold output)
    loaded = load_manifold_output(con, run_dir, default_cohort)
    print(f"  Loaded {len(loaded)} Manifold parquet files as DuckDB views")

    # Load observations — prefer per-axis version from run_dir (already loaded by rglob),
    # fall back to domain root for legacy layouts
    if 'observations' not in loaded:
        if obs_path.exists():
            if default_cohort:
                con.execute(f"""
                    CREATE OR REPLACE VIEW observations AS
                    SELECT signal_0, signal_id, value,
                        '{default_cohort}' AS cohort
                    FROM read_parquet('{obs_path}')
                """)
            else:
                con.execute(f"""
                    CREATE OR REPLACE VIEW observations AS
                    SELECT * FROM read_parquet('{obs_path}')
                """)
            loaded.append('observations')

    # Load typology intermediates — prefer per-axis versions from run_dir (already loaded
    # by rglob), fall back to domain root for legacy layouts.
    # signals.parquet is always domain-level metadata, always load from root.
    for name in ['typology', 'typology_raw',
                  'signal_statistics', 'signal_derivatives',
                  'signal_temporal', 'signal_primitives']:
        if name not in loaded:
            path = domain_dir / f'{name}.parquet'
            if path.exists():
                con.execute(f"""
                    CREATE OR REPLACE VIEW {name} AS
                    SELECT * FROM read_parquet('{path}')
                """)
                loaded.append(name)

    # signals.parquet is genuinely domain-level metadata — always load from domain root
    if 'signals' not in loaded:
        signals_path = domain_dir / 'signals.parquet'
        if signals_path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW signals AS
                SELECT * FROM read_parquet('{signals_path}')
            """)
            loaded.append('signals')

    # Execute alias and compatibility layers first (before any other SQL)
    alias_files = ['00_aliases.sql', '00_physics_compat.sql']
    for alias_name in alias_files:
        alias_path = sql_dir / alias_name
        if alias_path.exists():
            try:
                execute_sql_layer(con, alias_path)
                print(f"  Pre-loaded {alias_name}")
            except Exception as e:
                print(f"  WARNING: {alias_name} failed: {e}")

    # Execute each SQL layer and write markdown (skip alias files to avoid double-execution)
    alias_set = set(alias_files)
    sql_files = (
        sorted(f for f in sql_dir.glob('*.sql') if f.name not in alias_set)
        + sorted(report_dir.glob('*.sql'))
    )

    for sql_path in sql_files:
        # Check for companion Python preprocessor
        # e.g. feature_relevance.py for 25_feature_relevance.sql
        py_companion = sql_path.with_suffix('.py')
        if py_companion.exists():
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location(
                    sql_path.stem, py_companion
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                if hasattr(mod, 'preprocess'):
                    mod.preprocess(con, run_dir, domain_dir or run_dir.parent)
                    print(f"  Pre-processed {py_companion.name}")
            except Exception as e:
                print(f"  WARNING: {py_companion.name} failed: {e}")

        title = sql_path.stem.replace('_', ' ').title()
        print(f"  Running {sql_path.name}...")

        try:
            results = execute_sql_layer(con, sql_path)
            if results:
                md_path = sql_output_dir / f"{sql_path.stem}.md"
                write_markdown(results, md_path, title)
                print(f"    -> {md_path.name} ({len(results)} result sets)")
            else:
                print(f"    -> No SELECT results (views created only)")
        except Exception as e:
            print(f"    -> ERROR: {e}")

    con.close()
    md_count = len(list(sql_output_dir.glob('*.md')))
    print(f"  SQL analysis complete: {md_count} markdown files written")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m prime.sql.runner <run_dir>")
        sys.exit(1)
    run_sql_analysis(Path(sys.argv[1]))
