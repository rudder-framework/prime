"""
12: Stream Entry Point
=======================

Pure orchestration - calls streaming analysis modules.
Provides real-time analysis dashboard, streaming analysis, and demos.

Stages: data source → live analysis

Wraps prime.streaming CLI with dashboard, analyze, and demo modes.
"""

import sys
from typing import Optional


def run(
    mode: str = "dashboard",
    source: str = "turbofan",
    host: str = "0.0.0.0",
    port: int = 8080,
    duration: Optional[int] = None,
    window_size: int = 100,
    batch_size: int = 500,
    scenario: str = "degradation",
    verbose: bool = True,
) -> int:
    """
    Run streaming analysis.

    Args:
        mode: 'dashboard', 'analyze', or 'demo'
        source: Data source name
        host: Server host (dashboard mode)
        port: Server port (dashboard mode)
        duration: Analysis duration in seconds (analyze mode)
        window_size: Analysis window size
        batch_size: Batch size for processing
        scenario: Demo scenario name
        verbose: Print progress

    Returns:
        Exit code (0=success)
    """
    if verbose:
        print("=" * 70)
        print(f"12: STREAM - {mode.upper()}")
        print("=" * 70)

    if mode == "dashboard":
        from prime.streaming.websocket_server import run_server

        if verbose:
            print(f"  Source: {source}")
            print(f"  Server: http://{host}:{port}")
            print(f"  Press Ctrl+C to stop\n")

        run_server(
            source_type=source,
            host=host,
            port=port,
            window_size=window_size,
            batch_size=batch_size,
        )
        return 0

    elif mode == "analyze":
        from prime.streaming.analyzers import RealTimeAnalyzer
        from prime.streaming.data_sources import get_stream_connector

        connector = get_stream_connector(source)
        analyzer = RealTimeAnalyzer(
            window_size=window_size,
            batch_size=batch_size,
        )

        if verbose:
            print(f"  Source: {source}")
            print(f"  Duration: {duration or 'indefinite'}s")

        connector.start()
        try:
            import time
            start = time.time()
            while True:
                if duration and (time.time() - start) > duration:
                    break
                data = connector.read()
                if data:
                    result, alerts = analyzer.process_data_point(data)
                    if verbose and alerts:
                        for alert in alerts:
                            print(f"  ALERT: {alert}")
        except KeyboardInterrupt:
            pass
        finally:
            connector.stop()

        return 0

    elif mode == "demo":
        from prime.streaming.cli import cmd_demo
        import argparse
        ns = argparse.Namespace(
            source=source, scenario=scenario, duration=duration or 30,
            window_size=window_size, batch_size=batch_size, verbose=verbose,
        )
        return cmd_demo(ns)

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'dashboard', 'analyze', or 'demo'.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="12: Streaming Analysis")
    parser.add_argument('mode', choices=['dashboard', 'analyze', 'demo'],
                        help='Streaming mode')
    parser.add_argument('--source', default='turbofan', help='Data source')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--duration', type=int, help='Duration in seconds')
    parser.add_argument('--window-size', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=500)
    parser.add_argument('--scenario', default='degradation', help='Demo scenario')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    sys.exit(run(
        mode=args.mode,
        source=args.source,
        host=args.host,
        port=args.port,
        duration=args.duration,
        window_size=args.window_size,
        batch_size=args.batch_size,
        scenario=args.scenario,
        verbose=not args.quiet,
    ))


if __name__ == '__main__':
    main()
