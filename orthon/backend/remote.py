"""
ORTHON Remote Backend
=====================

Connect to PRISM as a remote service via HTTP API.

PRISM endpoints (port 8100):
- GET  /health            - Health check
- GET  /engines           - List available engines
- POST /compute           - Full analysis (synchronous)
- POST /analyze           - Full analysis → JSON summary (legacy)
- POST /intake            - Quick inspection (units, validation)

Request format for /compute:
{
  "config": {
    "discipline": "reaction",
    "entities": ["run_1", "run_2"],
    "global_constants": {...},
    "window": {"size": 50, "stride": 25}
  },
  "observations_path": "/path/to/observations.parquet"
}
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd

from ..intake.transformer import transform_for_prism

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    try:
        import requests as httpx
        HAS_HTTPX = True
    except ImportError:
        HAS_HTTPX = False


# Default PRISM port
PRISM_DEFAULT_PORT = 8100


def _get_client_kwargs() -> dict:
    """Get common kwargs for HTTP client."""
    prism_key = os.environ.get('PRISM_API_KEY')
    kwargs = {'timeout': 300.0}  # 5 min timeout
    if prism_key:
        kwargs['headers'] = {'Authorization': f'Bearer {prism_key}'}
    return kwargs


def _get_url() -> str:
    """Get PRISM URL from environment or default to localhost:8100."""
    url = os.environ.get('PRISM_URL')
    if not url:
        url = f"http://localhost:{PRISM_DEFAULT_PORT}"
    return url.rstrip('/')


def health_check() -> bool:
    """Check if remote PRISM service is available."""
    if not HAS_HTTPX:
        return False

    try:
        url = _get_url()
        response = httpx.get(f"{url}/health", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def get_engines() -> List[str]:
    """Get list of available PRISM engines."""
    if not HAS_HTTPX:
        raise ImportError("httpx or requests required. pip install httpx")

    url = _get_url()
    response = httpx.get(f"{url}/engines", **_get_client_kwargs())
    response.raise_for_status()
    return response.json()


def analyze(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Send data to remote PRISM service for full analysis.

    Transforms user data to PRISM format (observations.parquet + config.json)
    before sending.

    Args:
        df: Input DataFrame (wide format, raw user data)
        **kwargs: Additional config options (discipline, window, etc.)

    Returns:
        Analysis results as dict
    """
    if not HAS_HTTPX:
        raise ImportError("httpx or requests required. pip install httpx")

    url = _get_url()

    # Extract discipline if provided (for transformer)
    # Support both 'discipline' (new) and 'domain' (legacy)
    discipline = kwargs.pop('discipline', None) or kwargs.pop('domain', None)

    # Transform to PRISM format with discipline
    observations_df, config_dict = transform_for_prism(df, discipline=discipline)

    # Merge any remaining user-provided kwargs into config
    config_dict.update(kwargs)

    # Save observations to temp parquet
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        observations_df.to_parquet(f.name)
        obs_path = Path(f.name)

    try:
        # New PRISM /compute endpoint format
        request_data = {
            "config": config_dict,
            "observations_path": str(obs_path),
        }

        # Try new /compute endpoint first, fall back to /analyze
        try:
            response = httpx.post(
                f"{url}/compute",
                json=request_data,
                **_get_client_kwargs(),
            )
            response.raise_for_status()
        except Exception:
            # Fall back to legacy /analyze endpoint
            with open(obs_path, 'rb') as f:
                response = httpx.post(
                    f"{url}/analyze",
                    files={'observations': ('observations.parquet', f, 'application/octet-stream')},
                    data={'config': json.dumps(config_dict)},
                    **_get_client_kwargs(),
                )
            response.raise_for_status()

        result = response.json()
        result['backend'] = 'prism_remote'
        return result

    finally:
        obs_path.unlink(missing_ok=True)


def intake(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick inspection via PRISM's /intake endpoint.

    Returns units, validation, structure detection.
    """
    if not HAS_HTTPX:
        raise ImportError("httpx or requests required. pip install httpx")

    url = _get_url()

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)
        temp_path = Path(f.name)

    try:
        with open(temp_path, 'rb') as f:
            response = httpx.post(
                f"{url}/intake",
                files={'data': ('data.parquet', f, 'application/octet-stream')},
                **_get_client_kwargs(),
            )

        response.raise_for_status()
        return response.json()

    finally:
        temp_path.unlink(missing_ok=True)


def get_dataframe(df: pd.DataFrame, output: str, **kwargs) -> pd.DataFrame:
    """
    Get specific output as DataFrame via /analyze/dataframe.

    Transforms user data to PRISM format before sending.

    Args:
        df: Input DataFrame (wide format, raw user data)
        output: Which output to get ('vector', 'geometry', 'dynamics', 'physics')
        **kwargs: Config options

    Returns:
        Result as pandas DataFrame
    """
    if not HAS_HTTPX:
        raise ImportError("httpx or requests required. pip install httpx")

    url = _get_url()

    # Extract discipline if provided (for transformer)
    # Support both 'discipline' (new) and 'domain' (legacy)
    discipline = kwargs.pop('discipline', None) or kwargs.pop('domain', None)

    # Transform to PRISM format with discipline
    observations_df, config_dict = transform_for_prism(df, discipline=discipline)

    # Add output type and merge user kwargs
    config_dict['output'] = output
    config_dict.update(kwargs)

    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        observations_df.to_parquet(f.name)
        obs_path = Path(f.name)

    try:
        with open(obs_path, 'rb') as f:
            response = httpx.post(
                f"{url}/analyze/dataframe",
                files={'observations': ('observations.parquet', f, 'application/octet-stream')},
                data={'config': json.dumps(config_dict)},
                **_get_client_kwargs(),
            )

        response.raise_for_status()

        # Response is parquet bytes
        with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as out:
            out.write(response.content)
            result_path = Path(out.name)

        result_df = pd.read_parquet(result_path)
        result_path.unlink(missing_ok=True)
        return result_df

    finally:
        obs_path.unlink(missing_ok=True)


def get_info() -> Dict[str, Any]:
    """Get info about remote PRISM service."""
    if not HAS_HTTPX:
        return {'available': False, 'error': 'httpx not installed'}

    try:
        url = _get_url()

        # Health check
        health_resp = httpx.get(f"{url}/health", timeout=5.0)
        if health_resp.status_code != 200:
            return {'available': False, 'error': 'Health check failed'}

        # Get engines
        engines_resp = httpx.get(f"{url}/engines", timeout=5.0)
        engines = engines_resp.json() if engines_resp.status_code == 200 else []

        return {
            'available': True,
            'url': url,
            'engines': engines,
            'health': health_resp.json(),
        }

    except ValueError as e:
        return {'available': False, 'error': str(e)}
    except Exception as e:
        return {'available': False, 'error': f'Connection failed: {e}'}
