"""
ORTHON Backend Bridge
=====================

HTTP only. No PRISM imports.

Priority:
1. HTTP to PRISM (localhost:8100 or PRISM_URL)
2. Fallback to basic analysis
"""

import os
import tempfile
from pathlib import Path
from typing import Tuple, Any, Optional, Dict
import pandas as pd

from ..prism_client import prism_available, prism_status, get_prism_client
from ..intake.transformer import transform_for_prism

_BACKEND: Optional[Tuple[str, Any]] = None


def get_backend() -> Tuple[str, Any]:
    """
    Get the best available backend.

    Returns:
        Tuple of (backend_name, backend_module)
        - ('prism', None) if PRISM HTTP available
        - ('fallback', fallback_module) otherwise
    """
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    # Try HTTP connection to PRISM
    if prism_available():
        _BACKEND = ('prism', None)  # No module, we use HTTP client
        return _BACKEND

    # Fallback to basic analysis
    from . import fallback
    _BACKEND = ('fallback', fallback)
    return _BACKEND


def analyze(df: pd.DataFrame, **kwargs) -> Tuple[Dict[str, Any], str]:
    """
    Run analysis through whatever backend is available.

    Args:
        df: Input DataFrame
        **kwargs: Options (discipline, etc.)

    Returns:
        Tuple of (results_dict, backend_name)
    """
    name, backend = get_backend()

    if name == 'prism':
        results = _analyze_via_http(df, **kwargs)
        return results, 'prism'
    else:
        results = backend.analyze(df, **kwargs)
        return results, 'fallback'


def _analyze_via_http(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Send data to PRISM via HTTP.

    1. Transform user data to observations.parquet + config
    2. POST to PRISM /compute
    3. Return results (including results_path for parquets)
    """
    discipline = kwargs.pop('discipline', None)

    # Transform to PRISM format
    observations_df, config_dict = transform_for_prism(df, discipline=discipline)

    # Merge any additional kwargs into config
    config_dict.update(kwargs)

    # Create temp directory for observations and results
    with tempfile.TemporaryDirectory(delete=False) as tmpdir:
        tmpdir = Path(tmpdir)

        # Write observations parquet
        obs_path = tmpdir / "observations.parquet"
        observations_df.to_parquet(obs_path)

        # Results directory
        results_dir = tmpdir / "results"
        results_dir.mkdir(exist_ok=True)

        # Send to PRISM
        client = get_prism_client()
        response = client.compute(
            config=config_dict,
            observations_path=str(obs_path),
            output_dir=str(results_dir),
        )

        if response.get("status") == "complete":
            return {
                "backend": "prism",
                "status": "complete",
                "results_path": response.get("results_path", str(results_dir)),
                "parquets": response.get("parquets", []),
                "config": config_dict,
            }
        else:
            return {
                "backend": "prism",
                "status": "error",
                "message": response.get("message", "Unknown error"),
                "hint": response.get("hint"),
            }


def has_prism() -> bool:
    """Check if PRISM HTTP is available."""
    return prism_available()


def get_backend_info() -> dict:
    """Get info about current backend for display."""
    status = prism_status()

    if status['available']:
        return {
            'name': 'prism',
            'has_physics': True,
            'url': status['url'],
            'version': status.get('version', 'unknown'),
            'message': f"Connected to PRISM ({status['url']})",
        }
    else:
        return {
            'name': 'fallback',
            'has_physics': False,
            'message': 'PRISM offline. Start: python -m prism.api',
        }


def reset_backend():
    """Reset cached backend (for testing)."""
    global _BACKEND
    _BACKEND = None
