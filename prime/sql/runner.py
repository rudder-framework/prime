"""Run Prime SQL layers against Manifold output and write markdown results."""

import duckdb
from pathlib import Path


def load_manifold_output(con: duckdb.DuckDBPyConnection, output_dir: Path) -> list[str]:
    """Load all parquet files from Manifold output as DuckDB views."""
    loaded = []
    for parquet_file in sorted(output_dir.rglob('*.parquet')):
        view_name = parquet_file.stem  # e.g. state_geometry
        try:
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


def run_sql_analysis(domain_dir: Path) -> None:
    """Main entry point: load Manifold output, run SQL, write markdown."""
    output_dir = domain_dir / 'output'
    sql_output_dir = output_dir / 'sql'
    sql_output_dir.mkdir(parents=True, exist_ok=True)

    # Find SQL files in the repo
    sql_dir = Path(__file__).parent / 'layers'
    report_dir = Path(__file__).parent / 'reports'

    con = duckdb.connect()

    # Load all Manifold parquet files
    loaded = load_manifold_output(con, output_dir)
    print(f"  Loaded {len(loaded)} parquet files as DuckDB views")

    # Also load observations and typology from domain root
    for name in ['observations', 'typology', 'typology_raw']:
        path = domain_dir / f'{name}.parquet'
        if path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT * FROM read_parquet('{path}')
            """)
            loaded.append(name)

    # Execute each SQL layer and write markdown
    sql_files = sorted(sql_dir.glob('*.sql')) + sorted(report_dir.glob('*.sql'))

    for sql_path in sql_files:
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
        print("Usage: python -m prime.sql.runner <domain_dir>")
        sys.exit(1)
    run_sql_analysis(Path(sys.argv[1]))
