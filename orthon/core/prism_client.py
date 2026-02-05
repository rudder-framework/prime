"""
ORTHON → PRISM HTTP Client
==========================

HTTP only. No PRISM imports. No shared code.

Orthon controls PRISM via:
1. Sending manifest (what to compute)
2. Sending observations.parquet (data to compute on)
3. Receiving callback when complete

ORTHON sends manifest + observations_path
PRISM computes, writes parquets, pings callback
ORTHON fetches parquets with DuckDB
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


# Default URLs
PRISM_URL = os.environ.get("PRISM_URL", "http://localhost:8100")
ORTHON_URL = os.environ.get("ORTHON_URL", "http://localhost:8000")


class PRISMClient:
    """
    HTTP client for PRISM API.

    No imports from prism. HTTP requests only.

    Orthon controls PRISM via:
    1. submit_manifest() - Send manifest + data to PRISM
    2. get_job_status() - Check job progress
    3. fetch_outputs() - Retrieve result parquets
    """

    def __init__(self, base_url: str = PRISM_URL, timeout: float = 300.0):
        if not HAS_HTTPX:
            raise ImportError("httpx required. pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout)

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

    # =========================================================================
    # LEGACY COMPUTE (for backwards compatibility)
    # =========================================================================

    def compute(
        self,
        observations_path: str,
        manifest: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send compute request to PRISM.

        Args:
            observations_path: Path to observations.parquet
            manifest: Manifest dict with engines, params, etc.

        Returns:
            {
                "status": "complete",
                "job_id": "...",
                "files": ["signal.parquet", ...],
                "file_urls": ["/results/job_id/signal.parquet", ...]
            }
            or
            {
                "status": "error",
                "message": "..."
            }
        """
        observations_path = Path(observations_path)
        if not observations_path.exists():
            return {
                "status": "error",
                "message": f"Observations file not found: {observations_path}",
            }

        try:
            # Send as multipart form
            with open(observations_path, 'rb') as obs_file:
                files = {
                    'observations': ('observations.parquet', obs_file, 'application/octet-stream'),
                    'manifest': ('manifest.json', json.dumps(manifest), 'application/json'),
                }
                r = self.client.post(
                    f"{self.base_url}/compute",
                    files=files,
                )

            return r.json()

        except httpx.ConnectError:
            return {
                "status": "error",
                "message": "Cannot connect to PRISM. Is it running on port 8100?",
                "hint": "Start PRISM: cd prism && uvicorn prism.server.routes:app --port 8100",
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

    # =========================================================================
    # MANIFEST-BASED COMPUTE (new architecture)
    # =========================================================================

    def submit_manifest(
        self,
        manifest: Union[Dict[str, Any], "PrismManifest"],
        observations_path: Union[str, Path],
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a compute job to PRISM using manifest.

        This is the new architecture where Orthon is the brain.
        Orthon builds the manifest, PRISM just executes it.

        Args:
            manifest: PrismManifest or dict with full job specification
            observations_path: Path to observations.parquet
            callback_url: URL for PRISM to ping when done (optional override)

        Returns:
            {
                "status": "queued",
                "job_id": "...",
                "message": "Job accepted"
            }
            or
            {
                "status": "error",
                "message": "...",
            }
        """
        # Convert manifest to dict if needed
        if hasattr(manifest, 'model_dump'):
            manifest_dict = manifest.model_dump()
        else:
            manifest_dict = manifest

        # Override callback URL if provided
        if callback_url:
            manifest_dict['callback_url'] = callback_url

        observations_path = Path(observations_path)
        if not observations_path.exists():
            return {
                "status": "error",
                "message": f"Observations file not found: {observations_path}",
            }

        try:
            # Send manifest + observations as multipart
            with open(observations_path, 'rb') as f:
                files = {
                    'manifest': ('manifest.json', json.dumps(manifest_dict), 'application/json'),
                    'observations': ('observations.parquet', f, 'application/octet-stream'),
                }
                r = self.client.post(
                    f"{self.base_url}/compute/manifest",
                    files=files,
                )

            if r.status_code == 404:
                # Fallback to legacy endpoint if manifest endpoint doesn't exist
                return self._submit_manifest_legacy(manifest_dict, observations_path)

            return r.json()

        except httpx.ConnectError:
            return {
                "status": "error",
                "message": "Cannot connect to PRISM. Is it running?",
                "hint": f"Start PRISM: cd prism && python -m prism.api (expected at {self.base_url})",
            }
        except httpx.TimeoutException:
            return {
                "status": "error",
                "message": "PRISM request timed out",
                "hint": "Try smaller dataset or check PRISM logs",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    def _submit_manifest_legacy(
        self,
        manifest_dict: Dict[str, Any],
        observations_path: Path,
    ) -> Dict[str, Any]:
        """
        Fallback: Convert manifest to legacy config format.

        Used when PRISM doesn't support manifest endpoint yet.
        """
        # Extract legacy config from manifest
        legacy_config = {
            "window": {
                "size": manifest_dict.get("window", {}).get("size", 100),
                "stride": manifest_dict.get("window", {}).get("stride", 50),
            },
            "global_constants": manifest_dict.get("constants", {}),
        }

        # Try to extract discipline from engines
        engine_names = [e.get("name") for e in manifest_dict.get("engines", [])]
        # Could infer discipline from engines, but for now just pass through

        return self.compute(
            config=legacy_config,
            observations_path=str(observations_path),
            output_dir=manifest_dict.get("output_dir"),
        )

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Check status of a submitted job.

        Args:
            job_id: Job ID returned from submit_manifest

        Returns:
            {
                "job_id": "...",
                "status": "queued" | "running" | "complete" | "failed",
                "progress": 0.5,  # optional
                "message": "..."  # optional
            }
        """
        try:
            r = self.client.get(f"{self.base_url}/jobs/{job_id}")
            return r.json()
        except httpx.ConnectError:
            return {"status": "error", "message": "Cannot connect to PRISM"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def fetch_output(self, job_id: str, filename: str) -> bytes:
        """
        Fetch a single result parquet from PRISM.

        Args:
            job_id: Job ID
            filename: Output filename (e.g., "hurst.parquet")

        Returns:
            Raw bytes of the parquet file
        """
        r = self.client.get(f"{self.base_url}/outputs/{job_id}/{filename}")
        r.raise_for_status()
        return r.content

    def fetch_all_outputs(
        self,
        job_id: str,
        output_filenames: List[str],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """
        Fetch all result parquets from a completed job.

        Args:
            job_id: Job ID
            output_filenames: List of output filenames to fetch
            output_dir: Local directory to save files

        Returns:
            Dict mapping filename to local path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fetched = {}
        for filename in output_filenames:
            try:
                content = self.fetch_output(job_id, filename)
                local_path = output_dir / filename
                local_path.write_bytes(content)
                fetched[filename] = local_path
            except Exception as e:
                # Log but continue with other files
                print(f"Warning: Failed to fetch {filename}: {e}")

        return fetched

    def close(self):
        """Close HTTP client"""
        self.client.close()


# =============================================================================
# ASYNC CLIENT (for FastAPI integration)
# =============================================================================

class AsyncPRISMClient:
    """
    Async HTTP client for PRISM API.

    Use this in FastAPI route handlers.
    """

    def __init__(self, base_url: str = PRISM_URL, timeout: float = 300.0):
        if not HAS_HTTPX:
            raise ImportError("httpx required. pip install httpx")
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout)

    async def health(self) -> Dict[str, Any]:
        """Check if PRISM is running"""
        try:
            r = await self.client.get(f"{self.base_url}/health")
            return r.json()
        except httpx.ConnectError:
            return {"status": "offline", "message": "Cannot connect to PRISM"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def submit_manifest(
        self,
        manifest: Union[Dict[str, Any], "PrismManifest"],
        observations_path: Union[str, Path],
        callback_url: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Submit a compute job to PRISM using manifest (async).
        """
        if hasattr(manifest, 'model_dump'):
            manifest_dict = manifest.model_dump()
        else:
            manifest_dict = manifest

        if callback_url:
            manifest_dict['callback_url'] = callback_url

        observations_path = Path(observations_path)
        if not observations_path.exists():
            return {
                "status": "error",
                "message": f"Observations file not found: {observations_path}",
            }

        try:
            with open(observations_path, 'rb') as f:
                files = {
                    'manifest': ('manifest.json', json.dumps(manifest_dict), 'application/json'),
                    'observations': ('observations.parquet', f, 'application/octet-stream'),
                }
                r = await self.client.post(
                    f"{self.base_url}/compute/manifest",
                    files=files,
                )

            if r.status_code == 404:
                # Fallback to legacy
                return await self._submit_legacy(manifest_dict, observations_path)

            return r.json()

        except httpx.ConnectError:
            return {
                "status": "error",
                "message": "Cannot connect to PRISM",
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
            }

    async def _submit_legacy(
        self,
        manifest_dict: Dict[str, Any],
        observations_path: Path,
    ) -> Dict[str, Any]:
        """Fallback to legacy compute endpoint."""
        legacy_config = {
            "window": {
                "size": manifest_dict.get("window", {}).get("size", 100),
                "stride": manifest_dict.get("window", {}).get("stride", 50),
            },
            "global_constants": manifest_dict.get("constants", {}),
        }

        payload = {
            "config": legacy_config,
            "observations_path": str(observations_path),
            "output_dir": manifest_dict.get("output_dir"),
        }

        try:
            r = await self.client.post(f"{self.base_url}/compute", json=payload)
            return r.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Check job status (async)."""
        try:
            r = await self.client.get(f"{self.base_url}/jobs/{job_id}")
            return r.json()
        except Exception as e:
            return {"status": "error", "message": str(e)}

    async def fetch_output(self, job_id: str, filename: str) -> bytes:
        """Fetch a single output file (async)."""
        r = await self.client.get(f"{self.base_url}/outputs/{job_id}/{filename}")
        r.raise_for_status()
        return r.content

    async def fetch_all_outputs(
        self,
        job_id: str,
        output_filenames: List[str],
        output_dir: Union[str, Path],
    ) -> Dict[str, Path]:
        """Fetch all outputs (async)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        fetched = {}
        for filename in output_filenames:
            try:
                content = await self.fetch_output(job_id, filename)
                local_path = output_dir / filename
                local_path.write_bytes(content)
                fetched[filename] = local_path
            except Exception as e:
                print(f"Warning: Failed to fetch {filename}: {e}")

        return fetched

    async def close(self):
        """Close async client"""
        await self.client.aclose()


# =============================================================================
# SINGLETONS
# =============================================================================

_client: Optional[PRISMClient] = None
_async_client: Optional[AsyncPRISMClient] = None


def get_prism_client() -> PRISMClient:
    """Get singleton PRISM client (sync)"""
    global _client
    if _client is None:
        _client = PRISMClient()
    return _client


def get_async_prism_client() -> AsyncPRISMClient:
    """Get singleton PRISM client (async)"""
    global _async_client
    if _async_client is None:
        _async_client = AsyncPRISMClient()
    return _async_client


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
