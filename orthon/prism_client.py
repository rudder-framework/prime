"""
ORTHON → PRISM HTTP Client
==========================

HTTP only. No PRISM imports. No shared code.

ORTHON sends config + observations_path
PRISM computes, writes parquets, returns results_path
ORTHON reads parquets with DuckDB
"""

import os
from typing import Dict, Any, Optional

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# Default PRISM URL
PRISM_URL = os.environ.get("PRISM_URL", "http://localhost:8100")


class PRISMClient:
    """
    HTTP client for PRISM API.

    No imports from prism. HTTP requests only.
    """

    def __init__(self, base_url: str = PRISM_URL):
        if not HAS_HTTPX:
            raise ImportError("httpx required. pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=300.0)

    def health(self) -> Dict[str, Any]:
        """Check if PRISM is running"""
        try:
            r = self.client.get(f"{self.base_url}/health")
            return r.json()
        except httpx.ConnectError:
            return {"status": "offline", "message": "Cannot connect to PRISM"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def disciplines(self) -> list:
        """Get available disciplines from PRISM"""
        try:
            r = self.client.get(f"{self.base_url}/disciplines")
            return r.json()
        except:
            return []

    def compute(
        self,
        config: Dict[str, Any],
        observations_path: str,
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send compute request to PRISM.

        Args:
            config: Config dict (discipline, entities, signals, constants, window)
            observations_path: Path to observations.parquet (or raw CSV)
            output_dir: Where PRISM should write results (optional)

        Returns:
            {
                "status": "complete",
                "results_path": "/path/to/results/",
                "parquets": ["vector.parquet", "dynamics.parquet", ...]
            }
            or
            {
                "status": "error",
                "message": "...",
                "hint": "..."  # optional
            }
        """
        payload = {
            "config": config,
            "observations_path": observations_path,
        }
        if output_dir:
            payload["output_dir"] = output_dir

        try:
            r = self.client.post(
                f"{self.base_url}/compute",
                json=payload,
            )
            return r.json()

        except httpx.ConnectError:
            return {
                "status": "error",
                "message": "Cannot connect to PRISM. Is it running on port 8100?",
                "hint": "Start PRISM: cd prism && python -m prism.api",
            }
        except httpx.TimeoutException:
            return {
                "status": "error",
                "message": "PRISM request timed out (>5 min)",
                "hint": "Try smaller dataset or check PRISM logs",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    def close(self):
        """Close HTTP client"""
        self.client.close()


# =============================================================================
# SINGLETON
# =============================================================================

_client: Optional[PRISMClient] = None


def get_prism_client() -> PRISMClient:
    """Get singleton PRISM client"""
    global _client
    if _client is None:
        _client = PRISMClient()
    return _client


def prism_available() -> bool:
    """Quick check if PRISM is available"""
    try:
        client = get_prism_client()
        status = client.health()
        return status.get("status") == "ok"
    except:
        return False


def prism_status() -> Dict[str, Any]:
    """Get PRISM status with details"""
    try:
        client = get_prism_client()
        health = client.health()
        return {
            "available": health.get("status") == "ok",
            "url": client.base_url,
            "version": health.get("version"),
            "message": health.get("message", ""),
        }
    except Exception as e:
        return {
            "available": False,
            "url": PRISM_URL,
            "message": str(e),
        }
