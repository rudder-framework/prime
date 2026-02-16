#!/usr/bin/env python3
"""
Streaming Pipeline CLI

Processes large historical datasets in two phases:
  1. Ingest: raw files → partitioned observations.parquet
  2. Compute: per-partition Prime + Manifold

Usage:
    python streaming/run.py streaming/config/lumo.yaml
    python streaming/run.py streaming/config/lumo.yaml --ingest-only
    python streaming/run.py streaming/config/lumo.yaml --compute-only
    python streaming/run.py streaming/config/lumo.yaml --status
    python streaming/run.py streaming/config/lumo.yaml --reset
"""

import argparse
import sys
from pathlib import Path

# Ensure repo root is on sys.path so `streaming.*` and `prime.*` imports work
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def main():
    parser = argparse.ArgumentParser(
        description="Streaming pipeline for large historical datasets",
    )
    parser.add_argument(
        "config",
        help="Path to YAML config (e.g. streaming/config/lumo.yaml)",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only run ingest phase (raw → partitioned parquet)",
    )
    parser.add_argument(
        "--compute-only",
        action="store_true",
        help="Only run compute phase (partitions must already be ingested)",
    )
    parser.add_argument(
        "--skip-manifold",
        action="store_true",
        help="Skip Manifold compute (typology + classification only)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Print pipeline progress and exit",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete state file and exit (pipeline restarts from scratch)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output",
    )

    args = parser.parse_args()

    from streaming.workers.coordinator import Coordinator

    coordinator = Coordinator(args.config)

    if args.status:
        coordinator.print_status()
        return

    if args.reset:
        coordinator.reset()
        return

    coordinator.run(
        ingest_only=args.ingest_only,
        compute_only=args.compute_only,
        skip_manifold=args.skip_manifold,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
