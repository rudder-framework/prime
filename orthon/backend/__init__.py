"""ORTHON Backend - HTTP to PRISM, fallback to basic analysis."""

from .bridge import get_backend, analyze, has_prism, get_backend_info, reset_backend
from .fallback import analyze as fallback_analyze

__all__ = [
    'get_backend',
    'analyze',
    'has_prism',
    'get_backend_info',
    'reset_backend',
    'fallback_analyze',
]
