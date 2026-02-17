"""
Prime — one command, two modes.

    prime ~/domains/rossler                       Run full pipeline
    prime query ~/domains/rossler                 Query results via DuckDB
    prime query ~/domains/rossler --view typology
    prime query ~/domains/rossler --alerts
"""

import argparse
import sys
from pathlib import Path


def main():
    # Dispatch: `prime query ...` vs `prime <path>`
    if len(sys.argv) >= 2 and sys.argv[1] == 'query':
        _query_main(sys.argv[2:])
    else:
        _pipeline_main()


def _pipeline_main():
    parser = argparse.ArgumentParser(
        prog='prime',
        description='Run the full Prime pipeline on a domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime ~/domains/rossler         Run full pipeline
  prime ~/data/myfile.csv         Ingest + full pipeline
  prime query ~/domains/rossler   Query results after pipeline
""",
    )
    parser.add_argument('path', help='Domain directory or raw data file')
    parser.add_argument('--axis', default='time',
                        help='Signal to use as ordering axis (default: time)')

    args = parser.parse_args()
    domain_path = Path(args.path).expanduser().resolve()

    if not domain_path.exists():
        print(f"Error: {domain_path} does not exist")
        sys.exit(1)

    from prime.pipeline import run_pipeline
    run_pipeline(domain_path, axis=args.axis)


def _query_main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog='prime query',
        description='Query Manifold results via DuckDB.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime query ~/domains/rossler
  prime query ~/domains/rossler --view typology
  prime query ~/domains/rossler --entity engine_1
  prime query ~/domains/rossler --alerts
  prime query ~/domains/rossler --schema
""",
    )
    parser.add_argument('path', help='Domain directory')
    parser.add_argument(
        '--view', '-v',
        choices=['typology', 'geometry', 'dynamics', 'causality', 'all'],
        default='all',
        help='Which summary view to display (default: all)',
    )
    parser.add_argument('--entity', '-e', help='Filter to specific signal/entity')
    parser.add_argument('--alerts', action='store_true', help='Show signals needing attention')
    parser.add_argument('--schema', action='store_true', help='List loaded tables and exit')
    parser.add_argument('--sql', action='store_true', help='Print SQL instead of executing')

    args = parser.parse_args(argv)

    from prime.cli import query
    query(
        domain_path=args.path,
        view=args.view,
        entity=args.entity,
        alerts=args.alerts,
        schema=args.schema,
        sql=args.sql,
    )


if __name__ == "__main__":
    main()
