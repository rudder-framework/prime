"""
Command-line interface for Prime Streaming Module.

Usage:
    python -m prime.streaming dashboard --source turbofan
    python -m prime.streaming analyze --source crypto --duration 60
    python -m prime.streaming demo --source turbofan --scenario degradation
"""

import argparse
import sys
import time
from pathlib import Path

from .data_sources import DATA_SOURCES, get_stream_connector


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Start live analysis dashboard."""
    try:
        # Import here to avoid loading FastAPI unless needed
        from .websocket_server import run_server

        print(f"Starting PRISM Live Dashboard...")
        print(f"  Data source: {args.source}")
        print(f"  Server: http://{args.host}:{args.port}")
        print(f"  Press Ctrl+C to stop\n")

        run_server(
            source_type=args.source,
            host=args.host,
            port=args.port,
            window_size=args.window_size,
            batch_size=args.batch_size,
        )
        return 0

    except ImportError as e:
        print(f"Error: Missing dependency - {e}", file=sys.stderr)
        print("Install with: pip install fastapi uvicorn websockets", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Run streaming analysis with console output."""
    from .analyzers import RealTimeAnalyzer

    # Get data source
    connector = get_stream_connector(args.source)
    analyzer = RealTimeAnalyzer(
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    print(f"PRISM Streaming Analysis")
    print(f"========================")
    print(f"Source: {args.source}")
    print(f"Duration: {args.duration}s")
    print(f"Window size: {args.window_size}")
    print()

    start_time = time.time()
    sample_count = 0
    stream = connector.stream()

    try:
        for data_point in stream:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break

            # Process the data point
            instant_results, batch_results = analyzer.process_data_point(data_point)
            sample_count += 1

            # Print status periodically
            if sample_count % 50 == 0:
                if instant_results.get('status') == 'active':
                    eff_dim = instant_results.get('eff_dim', 0)
                    stage = instant_results.get('analysis_stage', 'unknown')
                    print(f"[{elapsed:6.1f}s] Samples: {sample_count:5d} | "
                          f"Eff Dim: {eff_dim:.3f} | Stage: {stage}")
                else:
                    print(f"[{elapsed:6.1f}s] Samples: {sample_count:5d} | "
                          f"Status: {instant_results.get('status', 'initializing')}")

                # Print recent alerts
                for alert in analyzer.alerts[-3:]:
                    level = alert.get('level', 'INFO')
                    msg = alert.get('message', '')
                    print(f"  [{level}] {msg}")

    except KeyboardInterrupt:
        print("\nAnalysis stopped by user.")

    # Final summary
    elapsed = time.time() - start_time
    print()
    print("Analysis Complete")
    print("=================")
    print(f"Total samples: {sample_count}")
    print(f"Duration: {elapsed:.1f}s")

    final_status = analyzer.get_status_summary()
    if final_status.get('latest_eff_dim') is not None:
        print(f"Final effective dimension: {final_status['latest_eff_dim']:.3f}")
    if final_status.get('latest_lyapunov') is not None:
        print(f"Lyapunov exponent: {final_status['latest_lyapunov']:.6f}")

    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    """Run demonstration scenario."""
    from .analyzers import RealTimeAnalyzer
    from .data_sources import SyntheticDataSource

    print(f"PRISM Streaming Demo: {args.scenario}")
    print("=" * 40)

    # Configure synthetic source based on scenario
    if args.scenario == 'degradation':
        print("Simulating gradual system degradation...")
        print("Watch effective dimension decrease over time.\n")
        source = SyntheticDataSource(
            num_signals=6,
            pattern='degradation',
            noise_level=0.1,
        )
    elif args.scenario == 'collapse':
        print("Simulating dimensional collapse event...")
        print("Watch for sudden dimension drop and alerts.\n")
        source = SyntheticDataSource(
            num_signals=6,
            pattern='collapse',
            noise_level=0.05,
        )
    elif args.scenario == 'recovery':
        print("Simulating system recovery after disturbance...")
        print("Watch dimension recover to normal levels.\n")
        source = SyntheticDataSource(
            num_signals=6,
            pattern='recovery',
            noise_level=0.1,
        )
    else:
        print("Normal operation baseline...\n")
        source = SyntheticDataSource(
            num_signals=6,
            pattern='normal',
            noise_level=0.1,
        )

    analyzer = RealTimeAnalyzer(
        window_size=args.window_size,
        batch_size=args.batch_size,
    )

    sample_count = 0
    start_time = time.time()
    stream = source.stream()

    try:
        for data_point in stream:
            elapsed = time.time() - start_time
            if elapsed >= args.duration:
                break

            instant_results, batch_results = analyzer.process_data_point(data_point)
            sample_count += 1

            # Print status every 25 samples
            if sample_count % 25 == 0:
                if instant_results.get('status') == 'active':
                    eff_dim = instant_results.get('eff_dim', 0)

                    # Visual indicator
                    bar_len = min(50, max(0, int(eff_dim * 10)))
                    bar = '#' * bar_len + '-' * (50 - bar_len)

                    print(f"[{bar}] Eff Dim: {eff_dim:.3f}")
                else:
                    print(f"Initializing... ({sample_count} samples)")

                # Print alerts with emphasis
                for alert in analyzer.alerts[-3:]:
                    level = alert.get('level', 'INFO')
                    msg = alert.get('message', '')
                    if level == 'CRITICAL':
                        print(f"  !!! CRITICAL: {msg} !!!")
                    elif level == 'WARNING':
                        print(f"  ** WARNING: {msg} **")

    except KeyboardInterrupt:
        print("\nDemo stopped by user.")

    elapsed = time.time() - start_time
    print()
    print("Demo Complete")
    print(f"Processed {sample_count} samples in {elapsed:.1f}s")

    return 0


def cmd_sources(args: argparse.Namespace) -> int:
    """List available data sources."""
    print("Available Data Sources")
    print("======================")
    print()

    for name, info in DATA_SOURCES.items():
        print(f"  {name}")
        print(f"    Class: {info['class'].__name__}")
        print(f"    Description: {info.get('description', 'No description')}")
        print()

    return 0


def main() -> int:
    """Main entry point for streaming CLI."""
    parser = argparse.ArgumentParser(
        description="Prime Streaming Module - Real-time PRISM Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start live dashboard with turbofan simulator
  python -m prime.streaming dashboard --source turbofan

  # Run 60-second analysis on crypto data
  python -m prime.streaming analyze --source crypto --duration 60

  # Run degradation demo scenario
  python -m prime.streaming demo --scenario degradation

  # List available data sources
  python -m prime.streaming sources
        """,
    )

    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Dashboard subcommand
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Start live analysis dashboard"
    )
    dashboard_parser.add_argument(
        "--source", "-s",
        type=str,
        default="turbofan",
        choices=list(DATA_SOURCES.keys()),
        help="Data source type (default: turbofan)"
    )
    dashboard_parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    dashboard_parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Server port (default: 8080)"
    )
    dashboard_parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Analysis window size (default: 100)"
    )
    dashboard_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch buffer size (default: 500)"
    )
    dashboard_parser.set_defaults(func=cmd_dashboard)

    # Analyze subcommand
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Run streaming analysis with console output"
    )
    analyze_parser.add_argument(
        "--source", "-s",
        type=str,
        default="turbofan",
        choices=list(DATA_SOURCES.keys()),
        help="Data source type (default: turbofan)"
    )
    analyze_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=60.0,
        help="Analysis duration in seconds (default: 60)"
    )
    analyze_parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Analysis window size (default: 100)"
    )
    analyze_parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch buffer size (default: 500)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # Demo subcommand
    demo_parser = subparsers.add_parser(
        "demo",
        help="Run demonstration scenario"
    )
    demo_parser.add_argument(
        "--scenario",
        type=str,
        default="degradation",
        choices=["normal", "degradation", "collapse", "recovery"],
        help="Demo scenario (default: degradation)"
    )
    demo_parser.add_argument(
        "--duration", "-d",
        type=float,
        default=30.0,
        help="Demo duration in seconds (default: 30)"
    )
    demo_parser.add_argument(
        "--window-size",
        type=int,
        default=50,
        help="Analysis window size (default: 50)"
    )
    demo_parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Batch buffer size (default: 200)"
    )
    demo_parser.set_defaults(func=cmd_demo)

    # Sources subcommand
    sources_parser = subparsers.add_parser(
        "sources",
        help="List available data sources"
    )
    sources_parser.set_defaults(func=cmd_sources)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
