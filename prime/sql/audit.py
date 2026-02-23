"""SQL audit: test every SQL file against a Manifold run and report pass/fail."""

import duckdb
import sys
import traceback
from pathlib import Path
from datetime import datetime

from prime.sql.runner import load_manifold_output, execute_sql_layer


def audit_sql(run_dir: Path, domain_dir: Path | None = None) -> list[dict]:
    """Run every SQL file and capture pass/fail/error/row-count.

    Returns list of dicts with keys: file, category, status, rows, error.
    """
    domain_dir = domain_dir or run_dir.parent
    con = duckdb.connect()

    # Load Manifold parquets
    loaded = load_manifold_output(con, run_dir)
    print(f"Loaded {len(loaded)} Manifold views: {', '.join(sorted(loaded))}")

    # Load domain-root parquets
    for name in ['observations', 'typology', 'typology_raw', 'signals']:
        path = domain_dir / f'{name}.parquet'
        if path.exists():
            con.execute(f"""
                CREATE OR REPLACE VIEW {name} AS
                SELECT * FROM read_parquet('{path}')
            """)
            loaded.append(name)

    # Execute alias and physics compat layers
    sql_dir = Path(__file__).parent
    layers_dir = sql_dir / 'layers'
    for alias_name in ['00_aliases.sql', '00_physics_compat.sql']:
        alias_path = layers_dir / alias_name
        if alias_path.exists():
            try:
                execute_sql_layer(con, alias_path)
                print(f"Pre-loaded {alias_name}")
            except Exception as e:
                print(f"WARNING: {alias_name} failed: {e}")

    # Collect all SQL files
    categories = {
        'layers': sorted(layers_dir.glob('*.sql')),
        'reports': sorted((sql_dir / 'reports').glob('*.sql')),
        'stages': sorted((sql_dir / 'stages').glob('*.sql')),
        '_legacy': sorted((sql_dir / '_legacy').glob('*.sql')),
    }

    # Skip alias files from layers (already pre-loaded)
    alias_set = {'00_aliases.sql', '00_physics_compat.sql'}
    categories['layers'] = [f for f in categories['layers'] if f.name not in alias_set]

    results = []
    for category, files in categories.items():
        for sql_path in files:
            entry = {
                'file': sql_path.name,
                'category': category,
                'status': 'UNKNOWN',
                'rows': 0,
                'error': '',
            }
            try:
                query_results = execute_sql_layer(con, sql_path)
                total_rows = sum(len(r['data']) for r in query_results)
                entry['rows'] = total_rows
                entry['status'] = 'PASS' if total_rows > 0 else 'PASS (views only)'
            except Exception as e:
                entry['status'] = 'FAIL'
                entry['error'] = str(e)[:120]
            results.append(entry)
            status_icon = {
                'PASS': '+', 'PASS (views only)': '.', 'FAIL': 'X'
            }.get(entry['status'], '?')
            print(f"  [{status_icon}] {category}/{sql_path.name}"
                  f"  rows={entry['rows']}"
                  + (f"  err={entry['error'][:60]}" if entry['error'] else ""))

    con.close()
    return results


def write_audit_report(results: list[dict], output_path: Path, run_dir: Path) -> None:
    """Write SQL_AUDIT.md triage table."""
    lines = [
        "# SQL Audit Report",
        "",
        f"**Run directory**: `{run_dir}`",
        f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
    ]

    # Summary counts
    pass_count = sum(1 for r in results if r['status'].startswith('PASS'))
    fail_count = sum(1 for r in results if r['status'] == 'FAIL')
    total = len(results)
    lines.append(f"**Summary**: {pass_count}/{total} passed, {fail_count} failed")
    lines.append("")

    # Group by category
    for category in ['layers', 'reports', 'stages', '_legacy']:
        cat_results = [r for r in results if r['category'] == category]
        if not cat_results:
            continue

        lines.append(f"## {category}/")
        lines.append("")
        lines.append("| File | Status | Rows | Error |")
        lines.append("|------|--------|-----:|-------|")
        for r in cat_results:
            err = r['error'][:80] if r['error'] else ''
            lines.append(f"| {r['file']} | {r['status']} | {r['rows']:,} | {err} |")
        lines.append("")

    output_path.write_text("\n".join(lines))
    print(f"\nAudit report written to {output_path}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m prime.sql.audit <run_dir>")
        print("  e.g. python -m prime.sql.audit ~/domains/cmapss/FD_001/train/output_time")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    results = audit_sql(run_dir)
    # Write report to repo root
    repo_root = Path(__file__).parent.parent.parent
    write_audit_report(results, repo_root / 'SQL_AUDIT.md', run_dir)
