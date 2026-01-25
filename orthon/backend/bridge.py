"""
ORTHON Backend Bridge
=====================

Connects to PRISM if available, falls back to basic stats otherwise.

Priority:
1. pip installed prism
2. PRISM_PATH environment variable
3. PRISM_URL for remote service
4. Fallback to basic analysis
"""

import os
import sys
from typing import Tuple, Any, Optional
import pandas as pd

_BACKEND: Optional[Tuple[str, Any]] = None


def get_backend() -> Tuple[str, Any]:
    """
    Get the best available backend.

    Returns:
        Tuple of (backend_name, backend_module)
        - ('prism', prism_module) if PRISM available
        - ('prism_remote', remote_client) if PRISM_URL set
        - ('fallback', fallback_module) otherwise
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # Try 1: pip installed prism
    try:
        import prism
        _BACKEND = ('prism', prism)
        return _BACKEND
    except ImportError:
        pass

    # Try 2: PRISM_PATH environment variable
    prism_path = os.environ.get('PRISM_PATH')
    if prism_path and os.path.isdir(prism_path):
        sys.path.insert(0, prism_path)
        try:
            import prism
            _BACKEND = ('prism', prism)
            return _BACKEND
        except ImportError:
            sys.path.remove(prism_path)

    # Try 3: PRISM_URL for remote service
    prism_url = os.environ.get('PRISM_URL')
    if prism_url:
        from . import remote
        _BACKEND = ('prism_remote', remote)
        return _BACKEND

    # Try 4: Fallback to basic analysis
    from . import fallback
    _BACKEND = ('fallback', fallback)
    return _BACKEND


def analyze(df: pd.DataFrame, **kwargs) -> Tuple[dict, str]:
    """
    Run analysis through whatever backend is available.

    Args:
        df: Input DataFrame
        **kwargs: Passed to backend.analyze()

    Returns:
        Tuple of (results_dict, backend_name)
    """
    name, backend = get_backend()
    results = backend.analyze(df, **kwargs)
    return results, name


def has_prism() -> bool:
    """Check if full PRISM is available."""
    name, _ = get_backend()
    return name in ('prism', 'prism_remote')


def get_backend_info() -> dict:
    """Get info about current backend for display."""
    name, backend = get_backend()

    info = {
        'name': name,
        'has_physics': name in ('prism', 'prism_remote'),
    }

    if name == 'prism':
        info['version'] = getattr(backend, '__version__', 'unknown')
        info['message'] = 'Full PRISM analysis available'
    elif name == 'prism_remote':
        info['url'] = os.environ.get('PRISM_URL')
        info['message'] = f"Connected to PRISM service"
    else:
        info['message'] = 'Using basic analysis (install prism for full physics)'

    return info
