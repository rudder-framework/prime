"""
Entry point for running orthon.streaming as a module.

Usage:
    python -m orthon.streaming dashboard --source turbofan
    python -m orthon.streaming analyze --source crypto --duration 60
"""

import sys
from .cli import main

if __name__ == "__main__":
    sys.exit(main())
