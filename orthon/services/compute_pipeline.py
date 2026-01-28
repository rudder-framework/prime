"""
ORTHON Compute Pipeline
=======================

Orchestrates the full compute workflow:

1. User uploads data
2. Orthon creates observations.parquet
3. Orthon analyzes data (units, signals, sampling)
4. Orthon builds manifest (what PRISM should compute)
5. Orthon sends manifest + data to PRISM
6. PRISM computes, pings callback
7. Orthon fetches results, processes, presents to user
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

from ..intake.config_generator import (
    DataAnalyzer,
    ManifestBuilder,
    DataAnalysis,
    generate_manifest,
)
from ..intake.manifest_schema import PrismManifest
from ..prism_client import get_prism_client, get_async_prism_client, ORTHON_URL
from .job_manager import JobManager, JobStatus, Job, get_job_manager


class ComputePipeline:
    """
    Orchestrates the full compute workflow.

    Orthon is the brain:
    1. Analyzes data
    2. Builds manifest
    3. Submits to PRISM
    4. Receives callback
    5. Fetches and processes results

    PRISM is the muscle:
    1. Receives manifest
    2. Executes engines
    3. Writes parquets
    4. Pings callback
    """

    def __init__(
        self,
        job_manager: Optional[JobManager] = None,
        output_base_dir: Optional[str] = None,
    ):
        """
        Initialize compute pipeline.

        Args:
            job_manager: JobManager instance (uses singleton if None)
            output_base_dir: Base directory for job outputs
        """
        self.job_manager = job_manager or get_job_manager()
        self.output_base_dir = Path(
            output_base_dir or os.environ.get("ORTHON_OUTPUT_DIR", "~/.orthon/outputs")
        ).expanduser()
        self.output_base_dir.mkdir(parents=True, exist_ok=True)

    def submit(
        self,
        observations_path: Union[str, Path],
        user_id: str = "default",
        window_size: int = 100,
        window_stride: int = 50,
        constants: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> Job:
        """
        Submit a compute job.

        This is the main entry point for the compute pipeline.
        It analyzes the data, builds a manifest, and submits to PRISM.

        Args:
            observations_path: Path to observations.parquet
            user_id: User submitting the job
            window_size: Window size for analysis
            window_stride: Stride between windows
            constants: Global constants for physics calculations
            callback_url: Override callback URL (uses default if None)

        Returns:
            Job object with job_id for tracking
        """
        observations_path = Path(observations_path)
        if not observations_path.exists():
            raise FileNotFoundError(f"Observations not found: {observations_path}")

        # Step 1: Analyze data
        analyzer = DataAnalyzer(observations_path)
        analysis = analyzer.analyze()

        # Step 2: Create output directory for this job
        # (job_id will be set by manifest)
        manifest = generate_manifest(
            observations_path,
            output_dir=self.output_base_dir,
            window_size=window_size,
            window_stride=window_stride,
            constants=constants,
        )

        job_id = manifest.job_id
        output_dir = self.output_base_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Update manifest with correct output_dir
        manifest.output_dir = str(output_dir)

        # Step 3: Create job record
        job = self.job_manager.create_job(
            user_id=user_id,
            manifest=manifest.model_dump(),
            analysis=self._analysis_to_dict(analysis),
            observations_path=str(observations_path),
            output_dir=str(output_dir),
        )
        job.job_id = job_id  # Use manifest's job_id

        # Step 4: Build callback URL
        if callback_url is None:
            orthon_url = os.environ.get("ORTHON_URL", ORTHON_URL)
            callback_url = f"{orthon_url}/api/callbacks/prism/{job_id}/complete"

        manifest.callback_url = callback_url

        # Step 5: Submit to PRISM
        self.job_manager.update_status(job_id, JobStatus.SUBMITTING)

        try:
            client = get_prism_client()
            result = client.submit_manifest(
                manifest=manifest,
                observations_path=observations_path,
                callback_url=callback_url,
            )

            if result.get("status") == "error":
                self.job_manager.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error=result.get("message", "PRISM submission failed"),
                    error_detail=result.get("hint"),
                )
            else:
                self.job_manager.update_status(job_id, JobStatus.QUEUED)

        except Exception as e:
            self.job_manager.update_status(
                job_id,
                JobStatus.FAILED,
                error="Failed to submit to PRISM",
                error_detail=str(e),
            )
            raise

        return self.job_manager.get_job(job_id)

    async def submit_async(
        self,
        observations_path: Union[str, Path],
        user_id: str = "default",
        window_size: int = 100,
        window_stride: int = 50,
        constants: Optional[Dict[str, Any]] = None,
        callback_url: Optional[str] = None,
    ) -> Job:
        """
        Submit a compute job (async version).

        Same as submit() but uses async PRISM client.
        """
        observations_path = Path(observations_path)
        if not observations_path.exists():
            raise FileNotFoundError(f"Observations not found: {observations_path}")

        # Step 1: Analyze data
        analyzer = DataAnalyzer(observations_path)
        analysis = analyzer.analyze()

        # Step 2: Generate manifest
        manifest = generate_manifest(
            observations_path,
            output_dir=self.output_base_dir,
            window_size=window_size,
            window_stride=window_stride,
            constants=constants,
        )

        job_id = manifest.job_id
        output_dir = self.output_base_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest.output_dir = str(output_dir)

        # Step 3: Create job record
        job = self.job_manager.create_job(
            user_id=user_id,
            manifest=manifest.model_dump(),
            analysis=self._analysis_to_dict(analysis),
            observations_path=str(observations_path),
            output_dir=str(output_dir),
        )
        job.job_id = job_id

        # Step 4: Build callback URL
        if callback_url is None:
            orthon_url = os.environ.get("ORTHON_URL", ORTHON_URL)
            callback_url = f"{orthon_url}/api/callbacks/prism/{job_id}/complete"

        manifest.callback_url = callback_url

        # Step 5: Submit to PRISM (async)
        self.job_manager.update_status(job_id, JobStatus.SUBMITTING)

        try:
            client = get_async_prism_client()
            result = await client.submit_manifest(
                manifest=manifest,
                observations_path=observations_path,
                callback_url=callback_url,
            )

            if result.get("status") == "error":
                self.job_manager.update_status(
                    job_id,
                    JobStatus.FAILED,
                    error=result.get("message", "PRISM submission failed"),
                    error_detail=result.get("hint"),
                )
            else:
                self.job_manager.update_status(job_id, JobStatus.QUEUED)

        except Exception as e:
            self.job_manager.update_status(
                job_id,
                JobStatus.FAILED,
                error="Failed to submit to PRISM",
                error_detail=str(e),
            )
            raise

        return self.job_manager.get_job(job_id)

    def handle_callback(
        self,
        job_id: str,
        status: str,
        outputs: list,
        error: Optional[str] = None,
    ) -> Job:
        """
        Handle callback from PRISM when job completes.

        This is called by the callback endpoint when PRISM pings back.

        Args:
            job_id: Job ID from PRISM
            status: "complete" or "failed"
            outputs: List of output filenames
            error: Error message if failed

        Returns:
            Updated Job object
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if status == "complete":
            # Update with outputs
            self.job_manager.set_outputs(job_id, outputs, job.output_dir)
            self.job_manager.update_status(job_id, JobStatus.COMPLETE)

        elif status == "failed":
            self.job_manager.update_status(
                job_id,
                JobStatus.FAILED,
                error=error or "PRISM execution failed",
            )

        return self.job_manager.get_job(job_id)

    async def fetch_results(self, job_id: str) -> Dict[str, Path]:
        """
        Fetch result parquets from PRISM.

        Called after callback if results need to be fetched.

        Args:
            job_id: Job ID

        Returns:
            Dict mapping filename to local path
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.COMPLETE:
            raise ValueError(f"Job not complete: {job.status}")

        self.job_manager.update_status(job_id, JobStatus.FETCHING)

        try:
            client = get_async_prism_client()
            fetched = await client.fetch_all_outputs(
                job_id=job_id,
                output_filenames=job.outputs,
                output_dir=job.output_dir,
            )

            self.job_manager.update_status(job_id, JobStatus.COMPLETE)
            return fetched

        except Exception as e:
            self.job_manager.update_status(
                job_id,
                JobStatus.FAILED,
                error="Failed to fetch results",
                error_detail=str(e),
            )
            raise

    def get_status(self, job_id: str) -> Optional[Job]:
        """
        Get current job status.

        Args:
            job_id: Job ID

        Returns:
            Job object or None
        """
        return self.job_manager.get_job(job_id)

    def get_results(self, job_id: str) -> Dict[str, Any]:
        """
        Get results of a completed job.

        Args:
            job_id: Job ID

        Returns:
            Dict with job info and result paths
        """
        job = self.job_manager.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        if job.status != JobStatus.COMPLETE:
            raise ValueError(f"Job not complete (status: {job.status})")

        return {
            "job_id": job_id,
            "status": job.status.value,
            "output_dir": job.output_dir,
            "outputs": job.outputs,
            "manifest": job.manifest,
            "analysis": job.analysis,
            "completed_at": job.completed_at,
        }

    def _analysis_to_dict(self, analysis: DataAnalysis) -> Dict[str, Any]:
        """Convert DataAnalysis to dict."""
        return {
            "entity_count": analysis.entity_count,
            "signal_count": analysis.signal_count,
            "observation_count": analysis.observation_count,
            "entities": analysis.entities,
            "signals": analysis.signals,
            "units": analysis.units,
            "categories": list(analysis.categories),
            "signal_units": analysis.signal_units,
            "signal_categories": analysis.signal_categories,
            "I_min": analysis.I_min,
            "I_max": analysis.I_max,
            "I_range": analysis.I_range,
            "sampling_rate": analysis.sampling_rate,
        }


# =============================================================================
# SINGLETON
# =============================================================================

_pipeline: Optional[ComputePipeline] = None


def get_compute_pipeline() -> ComputePipeline:
    """Get singleton compute pipeline."""
    global _pipeline
    if _pipeline is None:
        _pipeline = ComputePipeline()
    return _pipeline
