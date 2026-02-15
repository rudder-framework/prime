"""
Prime CLI
=========

Interpret Manifold parquet files via DuckDB + SQL views.

Usage:
    prime --data output/
    prime --data output/ --view typology
    prime --data output/ --entity engine_1
    prime --data output/ --alerts
    prime --data output/ --schema
"""

import argparse
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
    """Load SQL layer definitions (these create the base views that summary views depend on)."""
    layers_dir = Path(__file__).parent / 'sql' / 'layers'
    loaded = 0
    for sql_file in sorted(layers_dir.glob('*.sql')):
        try:
            execute_sql_layer(con, sql_file)
            loaded += 1
        except Exception:
            pass  # Layers may fail if dependent parquets are missing
    return loaded


def main():
    parser = argparse.ArgumentParser(
        description='Prime - Interpret Manifold parquet outputs'
    )
    parser.add_argument(
        '--data', '-d',
        type=str,
        required=True,
        help='Directory containing Manifold parquet files'
    )
    parser.add_argument(
        '--entity', '-e',
        type=str,
        default=None,
        help='Filter to specific cohort (entity_id)'
    )
    parser.add_argument(
        '--view', '-v',
        type=str,
        choices=['typology', 'geometry', 'dynamics', 'causality', 'all'],
        default='all',
        help='Which view to display'
    )
    parser.add_argument(
        '--alerts',
        action='store_true',
        help='Show only signals needing attention'
    )
    parser.add_argument(
        '--sql',
        action='store_true',
        help='Print SQL instead of executing'
    )
    parser.add_argument(
        '--schema',
        action='store_true',
        help='List loaded tables and exit'
    )

    args = parser.parse_args()
    data_dir = Path(args.data)

    if not data_dir.exists():
        print(f"Error: directory not found: {data_dir}")
        return

    # Connect DuckDB in-memory
    con = duckdb.connect()

    # Load parquets from data dir
    print(f"Reading Manifold outputs from: {data_dir}")
    loaded = load_manifold_output(con, data_dir)

    # Also load observations/typology from parent dir (domain root)
    domain_dir = data_dir.parent if data_dir.name == 'output' else data_dir
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
    if args.schema:
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
    if args.alerts:
        if args.sql:
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
    if args.entity:
        if args.sql:
            print(ENTITY_QUERY.strip().replace('?', f"'{args.entity}'"))
        else:
            try:
                df = con.execute(ENTITY_QUERY, [args.entity]).fetchdf()
                if len(df) > 0:
                    print(f"\n=== ENTITY: {args.entity} ===")
                    print(df.to_string(index=False))
                else:
                    print(f"No data found for entity: {args.entity}")
            except Exception as e:
                print(f"Cannot query entity (missing data): {e}")
        con.close()
        return

    # --view: show summary view(s)
    query = VIEW_QUERIES[args.view]
    if args.sql:
        print(query)
    else:
        try:
            df = con.execute(query).fetchdf()
            view_label = args.view.upper()
            print(f"\n=== {view_label} ===")
            print(df.to_string(index=False))
        except Exception as e:
            print(f"Cannot show {args.view} (missing data): {e}")

    con.close()


if __name__ == '__main__':
    main()
