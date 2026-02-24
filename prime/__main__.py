"""
Prime — one command, four modes.

    prime ~/domains/rossler/train                       Run full pipeline (order-by=time)
    prime ~/domains/rossler/train --order-by x          Run with x as ordering axis
    prime query ~/domains/rossler/train/output_time     Query results via DuckDB
    prime query ~/domains/rossler/train/output_time --view typology
    prime query ~/domains/rossler/train --alerts        Defaults to first output_*/ run
    prime generate rossler                              Generate synthetic dataset
    prime generate rossler --output ~/domains/rossler/train
    prime parameterization compile ~/domains/rossler/train
"""

import argparse
import sys
from pathlib import Path


def main():
    # Dispatch: `prime query ...` vs `prime generate ...` vs `prime parameterization ...` vs `prime <path>`
    if len(sys.argv) >= 2 and sys.argv[1] == 'query':
        _query_main(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'generate':
        _generate_main(sys.argv[2:])
    elif len(sys.argv) >= 2 and sys.argv[1] == 'parameterization':
        _parameterization_main(sys.argv[2:])
    else:
        _pipeline_main()


def _pipeline_main():
    parser = argparse.ArgumentParser(
        prog='prime',
        description='Run the full Prime pipeline on a domain.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime ~/domains/rossler/train                  Run full pipeline (creates output_time/)
  prime ~/domains/rossler/train --order-by x     Run with x axis (creates output_x/)
  prime query ~/domains/rossler/train/output_time  Query results from time run
  prime query ~/domains/rossler/train              Defaults to first output_*/ run
""",
    )
    parser.add_argument('path', help='Domain directory or raw data file')
    parser.add_argument('--order-by', dest='axis', default='time',
                        help='Signal to use as ordering axis (default: time)')
    parser.add_argument('--force-ingest', action='store_true',
                        help='Force re-ingest even if observations.parquet exists')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers for typology (default: PRIME_WORKERS env or 4)')

    args = parser.parse_args()
    domain_path = Path(args.path).expanduser().resolve()

    if not domain_path.exists():
        print(f"Error: {domain_path} does not exist")
        sys.exit(1)

    from prime.pipeline import run_pipeline
    run_pipeline(domain_path, axis=args.axis, force_ingest=args.force_ingest, workers=args.workers)


def _query_main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog='prime query',
        description='Query Manifold results via DuckDB.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  prime query ~/domains/rossler/train/output_time
  prime query ~/domains/rossler/train/output_time --view typology
  prime query ~/domains/rossler/train --entity engine_1
  prime query ~/domains/rossler/train --alerts
  prime query ~/domains/rossler/train/output_time --schema
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


def _parameterization_main(argv: list[str]):
    parser = argparse.ArgumentParser(
        prog='prime parameterization',
        description='Cross-run parameterization tools.',
    )
    subparsers = parser.add_subparsers(dest='subcommand', required=True)

    compile_parser = subparsers.add_parser('compile',
        help='Compile cross-run summaries from multiple axis runs')
    compile_parser.add_argument('path', help='Domain directory (e.g. ~/domains/rossler/train)')

    args = parser.parse_args(argv)

    if args.subcommand == 'compile':
        domain_path = Path(args.path).expanduser().resolve()
        if not domain_path.exists():
            print(f"Error: {domain_path} does not exist")
            sys.exit(1)

        from prime.parameterization.compile import compile_parameterization
        ok = compile_parameterization(domain_path, verbose=True)
        if not ok:
            print("No compilation produced (need 2+ runs with geometry data).")
            sys.exit(1)


if __name__ == "__main__":
    main()
