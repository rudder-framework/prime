"""
Prime Query CLI
===============

Query Manifold results via DuckDB + SQL views.

    prime query ~/domains/rossler
    prime query ~/domains/rossler --view typology
    prime query ~/domains/rossler --entity engine_1
    prime query ~/domains/rossler --alerts
    prime query ~/domains/rossler --schema
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


def _resolve_domain_dir(path: Path) -> Path:
    """Resolve domain directory from a path (could be domain dir or output dir)."""
    path = path.expanduser().resolve()
    if path.name == 'output':
        return path.parent
    return path


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
    domain_dir = _resolve_domain_dir(Path(domain_path))
    output_dir = domain_dir / 'output'

    if not output_dir.exists():
        print(f"Error: no output directory at {output_dir}")
        print(f"Run 'prime {domain_dir}' first to generate results.")
        return

    con = duckdb.connect()

    # Load parquets from output dir
    print(f"Reading Manifold outputs from: {output_dir}")
    loaded = load_manifold_output(con, output_dir)

    # Also load observations/typology from domain root
    for name in ['observations', 'typology', 'typology_raw']:
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
