#!/usr/bin/env python3
"""
Convenience wrapper — run official C-MAPSS test evaluation from repo root.

Usage:
    python evaluate_test.py --train ~/data/FD001/train/output \
                            --test  ~/data/FD001/test/output \
                            --rul   ~/data/FD001/RUL_FD001.txt
"""
from prime.ml.entry_points.evaluate_test import main

if __name__ == '__main__':
    main()
