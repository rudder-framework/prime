"""
Prime Job Manager
=================

Manages job lifecycle for compute requests with queue support.

Job States:
- pending: Job created, waiting in queue
- submitting: Sending to Manifold
- queued: Manifold accepted the job (legacy)
- running: Manifold is computing
- fetching: Manifold done, Prime fetching parquets
- processing: Prime joining/analyzing results
- complete: Done
- failed: Error occurred

Queue Behavior:
- Only one job runs at a time
- New jobs are queued if a job is already running
- Queue is processed FIFO
"""

import os
import json
import uuid
from enum import Enum
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Callable
from dataclasses import dataclass, field, asdict
import threading
from queue import Queue
import asyncio


class JobStatus(str, Enum):
    """Job status states."""
    PENDING = "pending"
    SUBMITTING = "submitting"
    QUEUED = "queued"
    RUNNING = "running"
    FETCHING = "fetching"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class Job:
    """Job record."""

    job_id: str
    user_id: str = "default"
    status: JobStatus = JobStatus.PENDING

    # Manifest and analysis
    manifest: Dict[str, Any] = field(default_factory=dict)
    analysis: Dict[str, Any] = field(default_factory=dict)

    # Paths
    observations_path: str = ""
    output_dir: str = ""

    # Results
    outputs: List[str] = field(default_factory=list)
    results_path: str = ""

    # Error info
    error: Optional[str] = None
    error_detail: Optional[str] = None

    # Timestamps
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        d = asdict(self)
        d['status'] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Job":
        """Create from dictionary."""
        if 'status' in data and isinstance(data['status'], str):
            data['status'] = JobStatus(data['status'])
        return cls(**data)


class JobManager:
    """
    Manages job lifecycle with queue support.

    Uses file-based storage for simplicity.
    Can be swapped to database (PostgreSQL, SQLite) for production.

    Queue Behavior:
    - Only one job runs at a time (single Manifold worker)
    - New jobs are queued if a job is already running
    - Queue is processed FIFO
    """

    def __init__(self, jobs_dir: Optional[str] = None):
        """
        Initialize job manager.

        Args:
            jobs_dir: Directory to store job files. Defaults to ~/.prime/jobs/
        """
        if jobs_dir:
            self.jobs_dir = Path(jobs_dir)
        else:
            self.jobs_dir = Path.home() / ".prime" / "jobs"

        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

        # Queue state
        self._current_job_id: Optional[str] = None
        self._job_queue: List[str] = []  # List of job_ids waiting
        self._queue_lock = threading.Lock()

    def _job_path(self, job_id: str) -> Path:
        """Get path to job file."""
        return self.jobs_dir / f"{job_id}.json"

    def create_job(
        self,
        user_id: str = "default",
        manifest: Optional[Dict[str, Any]] = None,
        analysis: Optional[Dict[str, Any]] = None,
        observations_path: str = "",
        output_dir: str = "",
    ) -> Job:
        """
        Create a new job record.

        Args:
            user_id: User who submitted the job
            manifest: ManifoldManifest dict
            analysis: DataAnalysis dict
            observations_path: Path to observations.parquet
            output_dir: Output directory for results

        Returns:
            Created Job object
        """
        job_id = str(uuid.uuid4())

        job = Job(
            job_id=job_id,
            user_id=user_id,
            status=JobStatus.PENDING,
            manifest=manifest or {},
            analysis=analysis or {},
            observations_path=observations_path,
            output_dir=output_dir,
        )

        self._save_job(job)
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        """
        Get job by ID.

        Args:
            job_id: Job identifier

        Returns:
            Job object or None if not found
        """
        job_path = self._job_path(job_id)
        if not job_path.exists():
            return None

        with self._lock:
            with open(job_path) as f:
                data = json.load(f)
            return Job.from_dict(data)

    def update_status(
        self,
        job_id: str,
        status: JobStatus,
        error: Optional[str] = None,
        error_detail: Optional[str] = None,
    ) -> Optional[Job]:
        """
        Update job status.

        Args:
            job_id: Job identifier
            status: New status
            error: Error message (for failed status)
            error_detail: Detailed error info

        Returns:
            Updated Job or None if not found
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.status = status
        job.updated_at = datetime.utcnow().isoformat()

        if error:
            job.error = error
            job.error_detail = error_detail

        if status == JobStatus.COMPLETE:
            job.completed_at = datetime.utcnow().isoformat()

        self._save_job(job)
        return job

    def set_outputs(
        self,
        job_id: str,
        outputs: List[str],
        results_path: str = "",
    ) -> Optional[Job]:
        """
        Set job outputs after Manifold completes.

        Args:
            job_id: Job identifier
            outputs: List of output filenames
            results_path: Path to results directory

        Returns:
            Updated Job or None
        """
        job = self.get_job(job_id)
        if not job:
            return None

        job.outputs = outputs
        job.results_path = results_path
        job.updated_at = datetime.utcnow().isoformat()

        self._save_job(job)
        return job

    def list_jobs(
        self,
        user_id: Optional[str] = None,
        status: Optional[JobStatus] = None,
        limit: int = 50,
    ) -> List[Job]:
        """
        List jobs with optional filtering.

        Args:
            user_id: Filter by user
            status: Filter by status
            limit: Max jobs to return

        Returns:
            List of Job objects
        """
        jobs = []

        for job_file in sorted(self.jobs_dir.glob("*.json"), reverse=True):
            if len(jobs) >= limit:
                break

            try:
                with open(job_file) as f:
                    data = json.load(f)
                job = Job.from_dict(data)

                # Apply filters
                if user_id and job.user_id != user_id:
                    continue
                if status and job.status != status:
                    continue

                jobs.append(job)
            except Exception:
                continue  # Skip malformed files

        return jobs

    def delete_job(self, job_id: str) -> bool:
        """
        Delete a job record.

        Args:
            job_id: Job identifier

        Returns:
            True if deleted, False if not found
        """
        job_path = self._job_path(job_id)
        if job_path.exists():
            with self._lock:
                job_path.unlink()
            return True
        return False

    def cleanup_old_jobs(self, max_age_days: int = 7) -> int:
        """
        Remove jobs older than max_age_days.

        Args:
            max_age_days: Maximum age in days

        Returns:
            Number of jobs cleaned up
        """
        from datetime import timedelta

        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        cleaned = 0

        for job_file in self.jobs_dir.glob("*.json"):
            try:
                with open(job_file) as f:
                    data = json.load(f)

                created = datetime.fromisoformat(data.get('created_at', ''))
                if created < cutoff:
                    job_file.unlink()
                    cleaned += 1
            except Exception:
                continue

        return cleaned

    def _save_job(self, job: Job) -> None:
        """Save job to disk."""
        job_path = self._job_path(job.job_id)
        with self._lock:
            with open(job_path, 'w') as f:
                json.dump(job.to_dict(), f, indent=2, default=str)

    # =========================================================================
    # QUEUE MANAGEMENT
    # =========================================================================

    def is_job_running(self) -> bool:
        """Check if a job is currently running."""
        with self._queue_lock:
            return self._current_job_id is not None

    def get_current_job(self) -> Optional[Job]:
        """Get the currently running job."""
        with self._queue_lock:
            if self._current_job_id:
                return self.get_job(self._current_job_id)
            return None

    def get_queue_position(self, job_id: str) -> int:
        """
        Get position of a job in the queue.

        Returns:
            Position (0 = running, 1+ = waiting), -1 if not found
        """
        with self._queue_lock:
            if self._current_job_id == job_id:
                return 0
            try:
                return self._job_queue.index(job_id) + 1
            except ValueError:
                return -1

    def get_queue_length(self) -> int:
        """Get number of jobs waiting in queue (not including running job)."""
        with self._queue_lock:
            return len(self._job_queue)

    def get_queue_status(self) -> Dict[str, Any]:
        """
        Get full queue status.

        Returns:
            {
                "running": job_id or None,
                "queued": [job_ids...],
                "queue_length": int
            }
        """
        with self._queue_lock:
            return {
                "running": self._current_job_id,
                "queued": list(self._job_queue),
                "queue_length": len(self._job_queue),
            }

    def enqueue_job(self, job_id: str) -> Dict[str, Any]:
        """
        Add a job to the queue.

        If no job is running, the job starts immediately.
        Otherwise, it's added to the queue.

        Args:
            job_id: Job to enqueue

        Returns:
            {
                "status": "running" | "queued",
                "position": int (0 if running, 1+ if queued)
            }
        """
        with self._queue_lock:
            if self._current_job_id is None:
                # No job running, start immediately
                self._current_job_id = job_id
                self.update_status(job_id, JobStatus.RUNNING)
                return {"status": "running", "position": 0}
            else:
                # Add to queue
                self._job_queue.append(job_id)
                position = len(self._job_queue)
                return {"status": "queued", "position": position}

    def complete_current_job(self, status: JobStatus = JobStatus.COMPLETE) -> Optional[str]:
        """
        Mark current job as complete and start next in queue.

        Args:
            status: Final status (COMPLETE or FAILED)

        Returns:
            job_id of the next job to run, or None if queue empty
        """
        with self._queue_lock:
            if self._current_job_id:
                # Update the completed job
                self.update_status(self._current_job_id, status)
                self._current_job_id = None

            # Start next job in queue
            if self._job_queue:
                next_job_id = self._job_queue.pop(0)
                self._current_job_id = next_job_id
                self.update_status(next_job_id, JobStatus.RUNNING)
                return next_job_id

            return None

    def cancel_job(self, job_id: str) -> bool:
        """
        Cancel a queued job (cannot cancel running job).

        Args:
            job_id: Job to cancel

        Returns:
            True if cancelled, False if not found or running
        """
        with self._queue_lock:
            if job_id == self._current_job_id:
                return False  # Cannot cancel running job

            if job_id in self._job_queue:
                self._job_queue.remove(job_id)
                self.update_status(job_id, JobStatus.FAILED, error="Cancelled by user")
                return True

            return False


# =============================================================================
# SINGLETON
# =============================================================================

_job_manager: Optional[JobManager] = None


def get_job_manager() -> JobManager:
    """Get singleton job manager."""
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
