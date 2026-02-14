#!/usr/bin/env python3
"""
Convenience wrapper — delegates to prime.ml.entry_points.cycle_features

Usage:
    python build_cycle_features.py build --obs ... --manifold ... --output ...
    python build_cycle_features.py train --train ... --test ... --rul ...
"""
import sys
import os

# Add repo root to path so prime package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from prime.ml.entry_points.cycle_features import main

if __name__ == '__main__':
    main()
