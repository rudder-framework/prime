#!/usr/bin/env python3
"""
RUDDER Analyze - One-command dynamical analysis

Usage:
    ./scripts/analyze.py data.csv
    ./scripts/analyze.py data.xlsx --output results/

This is the "stranger uploads a CSV" entry point.
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from framework.entry_points.csv_to_atlas import run, main

if __name__ == "__main__":
    main()
