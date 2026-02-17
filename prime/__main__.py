"""
Prime — one command, three modes.

    prime ~/domains/rossler/train                       Run full pipeline (axis=time)
    prime ~/domains/rossler/train --axis x              Run with x as ordering axis
    prime query ~/domains/rossler/train/time            Query results via DuckDB
    prime query ~/domains/rossler/train/time --view typology
    prime query ~/domains/rossler/train --alerts        Defaults to time/ run
    prime generate rossler                              Generate synthetic dataset
    prime generate rossler --output ~/domains/rossler/train
"""

import argparse
import sys
from pathlib import Path


def main():
    # Dispatch: `prime query ...` vs `prime generate ...` vs `prime <path>`
    if len(sys.argv) >= 2 and sys.argv[1] == 'query':
        _query_main(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        _generate_main(sys.argv[2:])
    else:
        _pipeline_main()


def _pipeline_main():
    parser = argparse.ArgumentParser(
        prog='prime',
        description='Run the full Prime pipeline on a domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime ~/domains/rossler/train             Run full pipeline (creates time/ run dir)
  prime ~/domains/rossler/train --axis x    Run with x axis (creates x/ run dir)
  prime query ~/domains/rossler/train/time  Query results from time run
  prime query ~/domains/rossler/train       Defaults to time/ run
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
  prime query ~/domains/rossler/train/time
  prime query ~/domains/rossler/train/time --view typology
  prime query ~/domains/rossler/train --entity engine_1
  prime query ~/domains/rossler/train --alerts
  prime query ~/domains/rossler/train/time --schema
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


def _generate_main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog='prime generate',
        description='Generate synthetic datasets.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime generate rossler
  prime generate rossler --output ~/domains/rossler/train
  prime generate rossler --n_samples 10000 --dt 0.01
""",
    )
    subparsers = parser.add_subparsers(dest='dataset', required=True)

    # rossler subcommand
    rossler_parser = subparsers.add_parser('rossler', help='Generate Rössler attractor dataset')
    rossler_parser.add_argument('--output', type=str, default='~/domains/rossler/train',
                                help='Output directory (default: ~/domains/rossler/train)')
    rossler_parser.add_argument('--n_samples', type=int, default=24000,
                                help='Number of samples (default: 24000)')
    rossler_parser.add_argument('--dt', type=float, default=0.05,
                                help='Integration time step (default: 0.05)')
    rossler_parser.add_argument('--a', type=float, default=0.2, help='Rössler a (default: 0.2)')
    rossler_parser.add_argument('--b', type=float, default=0.2, help='Rössler b (default: 0.2)')
    rossler_parser.add_argument('--c', type=float, default=5.7, help='Rössler c (default: 5.7)')

    args = parser.parse_args(argv)

    if args.dataset == 'rossler':
        from prime.generators.rossler import generate_rossler
        output_dir = Path(args.output).expanduser().resolve()
        generate_rossler(
            output_dir=output_dir,
            n_samples=args.n_samples,
            dt=args.dt,
            a=args.a,
            b=args.b,
            c=args.c,
        )


if __name__ == "__main__":
    main()
