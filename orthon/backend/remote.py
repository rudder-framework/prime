"""
ORTHON Remote Backend
=====================

Connect to PRISM as a remote service via HTTP API.
"""

import os
import json
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
import pandas as pd

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


def analyze(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Send data to remote PRISM service for analysis.

    Environment variables:
        PRISM_URL: Base URL of PRISM service
        PRISM_API_KEY: Authentication key
    """
    if not HAS_REQUESTS:
        raise ImportError("requests package required for remote PRISM. pip install requests")

    prism_url = os.environ.get('PRISM_URL')
    prism_key = os.environ.get('PRISM_API_KEY')

    if not prism_url:
        raise ValueError("PRISM_URL environment variable not set")

    # Save DataFrame to temp parquet
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)
        temp_path = Path(f.name)

    try:
        # Build request
        headers = {}
        if prism_key:
            headers['Authorization'] = f'Bearer {prism_key}'

        # Upload and analyze
        with open(temp_path, 'rb') as f:
            response = requests.post(
                f"{prism_url.rstrip('/')}/analyze",
                files={'data': ('data.parquet', f, 'application/octet-stream')},
                data={'config': json.dumps(kwargs)},
                headers=headers,
                timeout=300,  # 5 min timeout
            )

        response.raise_for_status()
        return response.json()

    finally:
        temp_path.unlink(missing_ok=True)


def analyze_stream(df: pd.DataFrame, **kwargs) -> Generator[Dict[str, Any], None, None]:
    """
    Stream analysis from remote PRISM service with progress updates.
    """
    if not HAS_REQUESTS:
        raise ImportError("requests package required for remote PRISM")

    prism_url = os.environ.get('PRISM_URL')
    prism_key = os.environ.get('PRISM_API_KEY')

    if not prism_url:
        raise ValueError("PRISM_URL environment variable not set")

    # Save DataFrame to temp parquet
    with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
        df.to_parquet(f.name)
        temp_path = Path(f.name)

    try:
        headers = {}
        if prism_key:
            headers['Authorization'] = f'Bearer {prism_key}'

        with open(temp_path, 'rb') as f:
            response = requests.post(
                f"{prism_url.rstrip('/')}/analyze/stream",
                files={'data': ('data.parquet', f, 'application/octet-stream')},
                data={'config': json.dumps(kwargs)},
                headers=headers,
                stream=True,
                timeout=300,
            )

        response.raise_for_status()

        # Stream progress updates
        for line in response.iter_lines():
            if line:
                yield json.loads(line)

    finally:
        temp_path.unlink(missing_ok=True)


def health_check() -> bool:
    """Check if remote PRISM service is available."""
    if not HAS_REQUESTS:
        return False

    prism_url = os.environ.get('PRISM_URL')
    if not prism_url:
        return False

    try:
        response = requests.get(f"{prism_url.rstrip('/')}/health", timeout=5)
        return response.status_code == 200
    except Exception:
        return False
