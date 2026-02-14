#!/usr/bin/env python3
"""
Convenience wrapper — run ML training from repo root.

Usage:
    python run_ml.py --data ~/data/FD001/output
"""
from prime.ml.entry_points.train import main

if __name__ == '__main__':
    main()
