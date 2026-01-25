"""
ORTHON: The Interface Layer for PRISM

Drop data. Get physics.

ORTHON does zero calculations. All math happens in PRISM.
"""

__version__ = "0.1.0"

# Core functionality
from orthon.intake import load_file, validate, detect_columns, detect_units
from orthon.backend import get_backend, analyze, has_prism, get_backend_info
from orthon.display import generate_report, to_json, to_csv

__all__ = [
    # Version
    '__version__',
    # Intake
    'load_file',
    'validate',
    'detect_columns',
    'detect_units',
    # Backend
    'get_backend',
    'analyze',
    'has_prism',
    'get_backend_info',
    # Display
    'generate_report',
    'to_json',
    'to_csv',
]
