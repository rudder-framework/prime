"""
Prime Query CLI
===============

Query Manifold results via DuckDB + SQL views.

    prime query ~/domains/rossler/train/output_time
    prime query ~/domains/rossler/train/output_time --view typology
    prime query ~/domains/rossler/train --entity engine_1
    prime query ~/domains/rossler/train --alerts
    prime query ~/domains/rossler/train/output_time --schema
"""

from pathlib import Path

import duckdb

from .sql.runner import load_manifold_output, execute_sql_layer


# SQL for each view — maps CLI --view names to SQL queries
VIEW_QUERIES = {
    'typology': 'SELECT * FROM v_summary_typology',
    'geometry': 'SELECT * FROM v_summary_geometry',
    'dynamics': 'SELECT * FROM v_summary_dynamics',
    'causality': 'SELECT * FROM v_summary_causality',
    'all': 'SELECT * FROM v_summary_all_layers ORDER BY layer_order',
}

ALERTS_QUERY = """
SELECT * FROM v_signals_needing_attention
ORDER BY
    CASE health_status
        WHEN 'critical' THEN 1
        WHEN 'warning' THEN 2
        WHEN 'monitor' THEN 3
        ELSE 4
    END
"""

ENTITY_QUERY = """
SELECT signal_id, health_status, reasons
FROM v_signals_needing_attention
WHERE signal_id = ?
"""


def _load_views(con: duckdb.DuckDBPyConnection) -> int:
    """Load SQL view definitions from prime/sql/views/."""
    views_dir = Path(__file__).parent / 'sql' / 'views'
    loaded = 0
    for sql_file in sorted(views_dir.glob('*.sql')):
        try:
            execute_sql_layer(con, sql_file)
            loaded += 1
        except Exception:
            pass  # Views may fail if dependent parquets are missing
    return loaded


def _load_sql_layers(con: duckdb.DuckDBPyConnection) -> int:
    """Load SQL layer definitions (base views that summary views depend on)."""
    layers_dir = Path(__file__).parent / 'sql' / 'layers'
    loaded = 0
    for sql_file in sorted(layers_dir.glob('*.sql')):
        try:
            execute_sql_layer(con, sql_file)
            loaded += 1
        except Exception:
            pass  # Layers may fail if dependent parquets are missing
    return loaded


def _resolve_run_dir(path: Path) -> Path:
    """Resolve run directory from a path.

    Handles:
      domain/output_time/  → use as run_dir (new layout)
      domain/              → find first output_*/ child
      domain/time/         → legacy fallback → domain/output_time/
      domain/output/       → legacy fallback → domain/output_time/
    """
    path = path.expanduser().resolve()

    # If path is already an output_* directory, use it directly
    if path.name.startswith('output_'):
        return path

    # Legacy: domain/output/ → try domain/output_time/
    if path.name == 'output':
        return path.parent / 'output_time'

    # Legacy: domain/time/ → try domain/output_time/
    if (path.parent / f'output_{path.name}').exists():
        return path.parent / f'output_{path.name}'

    # If path has observations.parquet, it's a domain root — find first output_*/
    if (path / 'observations.parquet').exists():
        output_dirs = sorted(d for d in path.iterdir()
                             if d.is_dir() and d.name.startswith('output_'))
        if output_dirs:
            return output_dirs[0]

    # Default to output_time/
    candidate = path / 'output_time'
    if candidate.exists():
        return candidate

    # Fall back to the path itself (may not exist yet)
    return path / 'output_time'


def query(
    domain_path: str | Path,
    view: str = 'all',
    entity: str | None = None,
    alerts: bool = False,
    schema: bool = False,
    sql: bool = False,
) -> None:
    """
    Query Manifold results for a domain.

    Args:
        domain_path: Domain directory (contains output/*.parquet).
        view: Which summary view to display.
        entity: Filter to specific entity/signal.
        alerts: Show signals needing attention.
        schema: List loaded tables and exit.
        sql: Print SQL instead of executing.
    """
    run_dir = _resolve_run_dir(Path(domain_path))
    domain_dir = run_dir.parent

    if not run_dir.exists():
        print(f"Error: no run directory at {run_dir}")
        print(f"Run 'prime {domain_dir}' first to generate results.")
        return

    con = duckdb.connect()

    # Load Manifold parquets from run_dir (output_* IS the Manifold output)
    print(f"Reading Manifold outputs from: {run_dir}")
    loaded = load_manifold_output(con, run_dir)

    # Load observations — prefer per-axis version from run_dir (already loaded by rglob),
    # fall back to domain root for legacy layouts
    if 'observations' not in loaded:
        obs_path = domain_dir / 'observations.parquet'
        if obs_path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW observations AS
                SELECT * FROM read_parquet('{obs_path}')
            """)
            loaded.append('observations')

    # Load typology/typology_raw — prefer per-axis versions from run_dir (already loaded
    # by rglob), fall back to domain root for legacy layouts
    for name in ['typology', 'typology_raw']:
        if name not in loaded:
            path = domain_dir / f'{name}.parquet'
            if path.exists():
                con.execute(f"""
                    CREATE OR REPLACE VIEW {name} AS
                    SELECT * FROM read_parquet('{path}')
                """)
                loaded.append(name)

    print(f"  Loaded {len(loaded)} tables: {', '.join(loaded)}")

    # --schema: list tables and exit
    if schema:
        for table in loaded:
            try:
                cols = con.execute(f"SELECT column_name, column_type FROM (DESCRIBE {table})").fetchall()
                print(f"\n  {table}:")
                for col_name, col_type in cols:
                    print(f"    {col_name}: {col_type}")
            except Exception:
                print(f"\n  {table}: (cannot describe)")
        con.close()
        return

    # Load SQL layers (base views) then summary views
    _load_sql_layers(con)
    _load_views(con)

    # --alerts: show signals needing attention
    if alerts:
        if sql:
            print(ALERTS_QUERY.strip())
        else:
            try:
                df = con.execute(ALERTS_QUERY).fetchdf()
                if len(df) > 0:
                    print("\n=== SIGNALS NEEDING ATTENTION ===")
                    print(df.to_string(index=False))
                else:
                    print("No alerts — all signals healthy.")
            except Exception as e:
                print(f"Cannot show alerts (missing data): {e}")
        con.close()
        return

    # --entity: filter to specific entity
    if entity:
        if sql:
            print(ENTITY_QUERY.strip().replace('?', f"'{entity}'"))
        else:
            try:
                df = con.execute(ENTITY_QUERY, [entity]).fetchdf()
                if len(df) > 0:
                    print(f"\n=== ENTITY: {entity} ===")
                    print(df.to_string(index=False))
                else:
                    print(f"No data found for entity: {entity}")
            except Exception as e:
                print(f"Cannot query entity (missing data): {e}")
        con.close()
        return

    # --view: show summary view(s)
    q = VIEW_QUERIES[view]
    if sql:
        print(q)
    else:
        try:
            df = con.execute(q).fetchdf()
            view_label = view.upper()
            print(f"\n=== {view_label} ===")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Cannot show {view} (missing data): {e}")

    con.close()
