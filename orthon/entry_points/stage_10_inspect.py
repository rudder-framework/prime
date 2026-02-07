"""
10: Inspect Entry Point
========================

Pure orchestration - calls inspection modules.
Inspects files, detects capabilities, validates results.

Stages: file path or output dir → inspection report

Use before pipeline to understand data, or after to validate outputs.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from orthon.inspection import (
    inspect_file,
    detect_capabilities,
    validate_results,
    FileInspection,
    Capabilities,
    ValidationResult,
)


def run(
    path: str,
    mode: str = "inspect",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run inspection on a file or directory.

    Args:
        path: Path to file or directory
        mode: 'inspect' (file), 'capabilities' (parquet), or 'validate' (output dir)
        verbose: Print progress

    Returns:
        Dict with inspection results
    """
    if verbose:
        print("=" * 70)
        print(f"10: INSPECT - {mode.upper()}")
        print("=" * 70)

    if mode == "inspect":
        result = inspect_file(path)

        if verbose:
            print(f"  File: {result.file_name}")
            print(f"  Format: {result.file_format}")
            print(f"  Rows: {result.n_rows:,}")
            print(f"  Columns: {result.n_columns}")
            for col in result.columns:
                print(f"    {col.name}: {col.dtype}")

        return {"inspection": result.__dict__}

    elif mode == "capabilities":
        result = detect_capabilities(path)

        if verbose:
            print(f"  Capabilities detected:")
            for cap, available in vars(result).items():
                status = "YES" if available else "no"
                print(f"    {cap}: {status}")

        return {"capabilities": vars(result)}

    elif mode == "validate":
        result = validate_results(path)

        if verbose:
            print(f"  Valid: {result.is_valid}")
            if result.errors:
                print(f"  Errors:")
                for err in result.errors:
                    print(f"    - {err}")
            if result.warnings:
                print(f"  Warnings:")
                for warn in result.warnings:
                    print(f"    - {warn}")

        return {"validation": vars(result)}

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'inspect', 'capabilities', or 'validate'.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="10: Inspect Files/Results")
    parser.add_argument('path', help='Path to file or output directory')
    parser.add_argument('--mode', choices=['inspect', 'capabilities', 'validate'],
                        default='inspect', help='Inspection mode')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run(args.path, mode=args.mode, verbose=not args.quiet)

    if not args.quiet:
        print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
