#!/usr/bin/env python3
"""
Convenience wrapper — run ML feature builder from repo root.

Usage:
    python build_ml_features.py --data ~/data/FD001/output --obs ~/data/FD001/observations.parquet
"""
from prime.ml.entry_points.features import main

if __name__ == '__main__':
    main()
