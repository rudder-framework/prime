"""
Coordinator for streaming pipeline.

Orchestrates ingest and compute across partitions with checkpoint/resume.
Runs workers in-process (sequential), persists state to JSON after each partition.
"""

import json
import os
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from streaming.workers.ingest_worker import create_converter, ingest_partition
from streaming.workers.compute_worker import compute_partition


@dataclass
class PartitionState:
    """State for a single partition."""
    partition_id: str
    files: List[str]           # Raw file paths
    ingested: bool = False
    computed: bool = False
    ingest_time_s: float = 0.0
    compute_time_s: float = 0.0
    error: Optional[str] = None


@dataclass
class PipelineState:
    """Persistent state for the entire pipeline run."""
    config_path: str
    output_dir: str
    partitions: Dict[str, dict] = field(default_factory=dict)  # partition_id → PartitionState as dict
    bootstrap_partition: Optional[str] = None
    started_at: Optional[str] = None
    updated_at: Optional[str] = None

    def save(self, path: Path) -> None:
        self.updated_at = datetime.now().isoformat()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: Path) -> "PipelineState":
        data = json.loads(path.read_text())
        return cls(**data)


class Coordinator:
    """Orchestrate streaming ingest and compute with checkpoint/resume."""

    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = yaml.safe_load(self.config_path.read_text())

        self.output_dir = Path(os.path.expanduser(
            self.config["paths"]["output_dir"]
        ))
        self.raw_dir = Path(os.path.expanduser(
            self.config["paths"]["raw_dir"]
        ))
        self.partitions_dir = self.output_dir / "partitions"
        self.state_path = self.output_dir / "pipeline_state.json"

        # Load or create state
        if self.state_path.exists():
            self.state = PipelineState.load(self.state_path)
        else:
            self.state = PipelineState(
                config_path=str(self.config_path),
                output_dir=str(self.output_dir),
                started_at=datetime.now().isoformat(),
            )

    def run(
        self,
        ingest_only: bool = False,
        compute_only: bool = False,
        skip_manifold: bool = False,
        verbose: bool = True,
    ) -> None:
        """
        Main pipeline loop.

        Args:
            ingest_only: Only run ingest phase
            compute_only: Only run compute phase (partitions must be ingested)
            skip_manifold: Skip Manifold compute (typology + classification only)
            verbose: Print progress
        """
        dataset_name = self.config.get("dataset", {}).get("name", "unknown")
        print(f"{'=' * 70}")
        print(f"Streaming Pipeline: {dataset_name}")
        print(f"{'=' * 70}")
        print(f"  Raw:    {self.raw_dir}")
        print(f"  Output: {self.output_dir}")
        print()

        # Plan partitions (discovers files, groups by week)
        if not compute_only:
            self._plan_partitions(verbose)

        partition_ids = sorted(self.state.partitions.keys())
        if not partition_ids:
            print("No partitions found. Check raw_dir and file_pattern.")
            return

        print(f"  Partitions: {len(partition_ids)}")
        print()

        # Phase 1: Ingest
        if not compute_only:
            self._run_ingest(partition_ids, verbose)

        if ingest_only:
            print("\nIngest complete (--ingest-only). Run without flag to continue with compute.")
            self.state.save(self.state_path)
            return

        # Phase 2: Bootstrap (typology on first partition)
        reuse_typology = self.config.get("bootstrap", {}).get("reuse_typology", True)
        if reuse_typology and not self.state.bootstrap_partition:
            self._run_bootstrap(partition_ids[0], skip_manifold, verbose)

        # Phase 3: Compute
        self._run_compute(partition_ids, skip_manifold, verbose)

        self.state.save(self.state_path)
        print(f"\nPipeline complete. State saved to {self.state_path}")

    def _plan_partitions(self, verbose: bool = True) -> None:
        """Discover raw files and group into weekly partitions."""
        file_pattern = self.config["paths"].get("file_pattern", "**/*.mat")
        files = sorted(self.raw_dir.glob(file_pattern))

        if verbose:
            print(f"  Discovered {len(files)} raw files")

        strategy = self.config.get("partitioning", {}).get("strategy", "weekly")

        if strategy == "weekly":
            partitions = self._group_weekly(files)
        else:
            raise ValueError(f"Unknown partitioning strategy: {strategy}")

        # Merge with existing state (preserve completed flags)
        for pid, file_list in partitions.items():
            if pid not in self.state.partitions:
                ps = PartitionState(partition_id=pid, files=[str(f) for f in file_list])
                self.state.partitions[pid] = asdict(ps)
            else:
                # Update file list but keep progress flags
                self.state.partitions[pid]["files"] = [str(f) for f in file_list]

        self.state.save(self.state_path)

    def _group_weekly(self, files: List[Path]) -> Dict[str, List[Path]]:
        """Group files by ISO week based on file modification time."""
        from collections import defaultdict
        partitions = defaultdict(list)

        for f in files:
            mtime = datetime.fromtimestamp(f.stat().st_mtime)
            iso_year, iso_week, _ = mtime.isocalendar()
            partition_id = f"{iso_year}-W{iso_week:02d}"
            partitions[partition_id].append(f)

        return dict(partitions)

    def _run_ingest(self, partition_ids: List[str], verbose: bool = True) -> None:
        """Sequential partition ingest with checkpoint after each."""
        converter = create_converter(self.config)
        files_per_flush = self.config.get("partitioning", {}).get("files_per_flush", 50)

        n_total = len(partition_ids)
        n_done = sum(1 for pid in partition_ids if self.state.partitions[pid].get("ingested"))
        if n_done == n_total:
            if verbose:
                print("  Ingest: all partitions already ingested (resuming)")
            return

        print(f"\n--- INGEST PHASE ({n_done}/{n_total} already done) ---\n")

        for i, pid in enumerate(partition_ids):
            ps = self.state.partitions[pid]
            if ps.get("ingested"):
                continue

            partition_dir = self.partitions_dir / pid
            files = [Path(f) for f in ps["files"]]

            t0 = time.time()
            try:
                ingest_partition(
                    partition_id=pid,
                    files=files,
                    output_dir=partition_dir,
                    converter=converter,
                    files_per_flush=files_per_flush,
                    verbose=verbose,
                )
                ps["ingested"] = True
                ps["ingest_time_s"] = round(time.time() - t0, 1)
            except Exception as e:
                ps["error"] = str(e)
                print(f"  ERROR ingesting {pid}: {e}")

            # Checkpoint after each partition
            self.state.save(self.state_path)

            if verbose:
                done = n_done + i + 1
                print(f"  Progress: {done}/{n_total} partitions ingested\n")

    def _run_bootstrap(self, first_partition_id: str, skip_manifold: bool, verbose: bool = True) -> None:
        """Run full compute on first partition to produce bootstrap typology."""
        print(f"\n--- BOOTSTRAP PHASE (partition {first_partition_id}) ---\n")

        partition_dir = self.partitions_dir / first_partition_id
        ps = self.state.partitions[first_partition_id]

        if not ps.get("ingested"):
            raise RuntimeError(f"Bootstrap partition {first_partition_id} not yet ingested")

        t0 = time.time()
        try:
            compute_partition(
                partition_dir=partition_dir,
                skip_manifold=skip_manifold,
                verbose=verbose,
            )
            ps["computed"] = True
            ps["compute_time_s"] = round(time.time() - t0, 1)
            self.state.bootstrap_partition = first_partition_id
        except Exception as e:
            ps["error"] = str(e)
            print(f"  ERROR in bootstrap: {e}")
            raise

        self.state.save(self.state_path)

    def _run_compute(self, partition_ids: List[str], skip_manifold: bool, verbose: bool = True) -> None:
        """Sequential partition compute with overlap handling."""
        reuse_typology = self.config.get("bootstrap", {}).get("reuse_typology", True)
        overlap_samples = self.config.get("compute", {}).get("overlap_samples", 0)

        # Bootstrap paths (if reusing)
        bootstrap_typ_path = None
        bootstrap_raw_path = None
        if reuse_typology and self.state.bootstrap_partition:
            bp_dir = self.partitions_dir / self.state.bootstrap_partition
            bootstrap_typ_path = bp_dir / "typology.parquet"
            bootstrap_raw_path = bp_dir / "typology_raw.parquet"

        n_total = len(partition_ids)
        n_done = sum(1 for pid in partition_ids if self.state.partitions[pid].get("computed"))
        if n_done == n_total:
            if verbose:
                print("  Compute: all partitions already computed (resuming)")
            return

        print(f"\n--- COMPUTE PHASE ({n_done}/{n_total} already done) ---\n")

        prev_partition_dir = None
        for i, pid in enumerate(partition_ids):
            ps = self.state.partitions[pid]
            partition_dir = self.partitions_dir / pid

            if ps.get("computed"):
                prev_partition_dir = partition_dir
                continue

            if not ps.get("ingested"):
                if verbose:
                    print(f"  Skipping {pid}: not yet ingested")
                continue

            if verbose:
                print(f"  Computing partition {pid} ({i + 1}/{n_total})...")

            t0 = time.time()
            try:
                compute_partition(
                    partition_dir=partition_dir,
                    bootstrap_typology_path=bootstrap_typ_path if reuse_typology else None,
                    bootstrap_typology_raw_path=bootstrap_raw_path if reuse_typology else None,
                    previous_partition_dir=prev_partition_dir,
                    overlap_samples=overlap_samples,
                    skip_manifold=skip_manifold,
                    verbose=verbose,
                )
                ps["computed"] = True
                ps["compute_time_s"] = round(time.time() - t0, 1)
            except Exception as e:
                ps["error"] = str(e)
                print(f"  ERROR computing {pid}: {e}")

            prev_partition_dir = partition_dir

            # Checkpoint after each partition
            self.state.save(self.state_path)

            if verbose:
                done = n_done + sum(1 for p in partition_ids[:i + 1]
                                    if self.state.partitions[p].get("computed"))
                print(f"  Progress: {done}/{n_total} partitions computed\n")

    def print_status(self) -> None:
        """Print pipeline progress summary."""
        if not self.state_path.exists():
            print("No state file found. Pipeline has not been run yet.")
            return

        state = PipelineState.load(self.state_path)
        partitions = state.partitions

        n_total = len(partitions)
        n_ingested = sum(1 for p in partitions.values() if p.get("ingested"))
        n_computed = sum(1 for p in partitions.values() if p.get("computed"))
        n_errors = sum(1 for p in partitions.values() if p.get("error"))

        total_files = sum(len(p.get("files", [])) for p in partitions.values())
        total_ingest_time = sum(p.get("ingest_time_s", 0) for p in partitions.values())
        total_compute_time = sum(p.get("compute_time_s", 0) for p in partitions.values())

        print(f"{'=' * 70}")
        print(f"Streaming Pipeline Status")
        print(f"{'=' * 70}")
        print(f"  Config:     {state.config_path}")
        print(f"  Output:     {state.output_dir}")
        print(f"  Started:    {state.started_at or 'N/A'}")
        print(f"  Updated:    {state.updated_at or 'N/A'}")
        print(f"  Bootstrap:  {state.bootstrap_partition or 'not yet'}")
        print()
        print(f"  Partitions: {n_total}")
        print(f"  Raw files:  {total_files}")
        print(f"  Ingested:   {n_ingested}/{n_total}")
        print(f"  Computed:   {n_computed}/{n_total}")
        if n_errors:
            print(f"  Errors:     {n_errors}")
        print()
        print(f"  Ingest time:  {total_ingest_time:.0f}s")
        print(f"  Compute time: {total_compute_time:.0f}s")

        # Show per-partition details
        if n_total <= 20:
            print()
            print(f"  {'Partition':<15} {'Files':>6} {'Ingested':>10} {'Computed':>10} {'Error'}")
            print(f"  {'-' * 60}")
            for pid in sorted(partitions.keys()):
                p = partitions[pid]
                ing = "yes" if p.get("ingested") else "no"
                comp = "yes" if p.get("computed") else "no"
                err = p.get("error", "")[:30] if p.get("error") else ""
                n_files = len(p.get("files", []))
                print(f"  {pid:<15} {n_files:>6} {ing:>10} {comp:>10} {err}")

    def reset(self) -> None:
        """Delete state file to start fresh."""
        if self.state_path.exists():
            self.state_path.unlink()
            print(f"State file deleted: {self.state_path}")
            print("Pipeline will start from scratch on next run.")
        else:
            print("No state file to delete.")
