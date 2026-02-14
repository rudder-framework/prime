"""
Entry point for running prime.streaming as a module.

Usage:
    python -m prime.streaming dashboard --source turbofan
    python -m prime.streaming analyze --source crypto --duration 60
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
