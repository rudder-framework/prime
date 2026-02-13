"""
Entry point for running framework.streaming as a module.

Usage:
    python -m framework.streaming dashboard --source turbofan
    python -m framework.streaming analyze --source crypto --duration 60
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
