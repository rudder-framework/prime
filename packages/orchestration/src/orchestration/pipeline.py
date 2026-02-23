"""
Pipeline runner: sequences all compute packages.

Defines the DAG of stages with inputs/outputs.
Runs stages in topological order, passing parquet paths.

No math lives here. Only wiring and file I/O.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any


@dataclass
class PipelineStage:
    """Definition of a compute stage."""
    name: str
    package: str
    function: str  # dotted path: 'vector.signal.compute_signal'
    inputs: List[str]  # parquet filenames consumed
    outputs: List[str]  # parquet filenames produced
    scale: str  # 'signal', 'cohort', 'fleet'
    optional: bool = False
    depends_on: List[str] = field(default_factory=list)


# Canonical stage ordering
STAGES: List[PipelineStage] = [
    PipelineStage(
        name='typology',
        package='typology',
        function='typology.classify.classify',
        inputs=['observations.parquet'],
        outputs=['signal_typology.parquet'],
        scale='signal',
    ),
    PipelineStage(
        name='vector',
        package='vector',
        function='vector.signal.compute_signal',
        inputs=['observations.parquet', 'signal_typology.parquet'],
        outputs=['signal_vector.parquet'],
        scale='signal',
        depends_on=['typology'],
    ),
    PipelineStage(
        name='eigendecomp',
        package='eigendecomp',
        function='eigendecomp.decompose.compute_eigendecomp',
        inputs=['signal_vector.parquet'],
        outputs=['cohort_eigendecomp.parquet'],
        scale='cohort',
        depends_on=['vector'],
    ),
    PipelineStage(
        name='geometry',
        package='geometry',
        function='geometry.signal.compute_signal_geometry',
        inputs=['signal_vector.parquet', 'cohort_eigendecomp.parquet'],
        outputs=['signal_geometry.parquet'],
        scale='signal',
        depends_on=['eigendecomp'],
    ),
    PipelineStage(
        name='geometry_dynamics',
        package='geometry',
        function='geometry.dynamics.compute_eigenvalue_dynamics',
        inputs=['cohort_eigendecomp.parquet'],
        outputs=['geometry_dynamics.parquet'],
        scale='cohort',
        depends_on=['eigendecomp'],
    ),
    PipelineStage(
        name='pairwise',
        package='pairwise',
        function='pairwise.signal.compute_signal_pairwise',
        inputs=['signal_vector.parquet'],
        outputs=['signal_pairwise.parquet'],
        scale='signal',
        depends_on=['vector'],
    ),
    PipelineStage(
        name='dynamics',
        package='dynamics',
        function='dynamics.ftle.compute_ftle',
        inputs=['observations.parquet'],
        outputs=['ftle.parquet'],
        scale='signal',
    ),
    PipelineStage(
        name='velocity',
        package='velocity',
        function='velocity.field.compute_velocity_field',
        inputs=['observations.parquet'],
        outputs=['velocity_field.parquet'],
        scale='signal',
    ),
    PipelineStage(
        name='ridge',
        package='ridge',
        function='ridge.proximity.compute_ridge_proximity',
        inputs=['ftle.parquet', 'velocity_field.parquet'],
        outputs=['ridge_proximity.parquet'],
        scale='signal',
        depends_on=['dynamics', 'velocity'],
    ),
    PipelineStage(
        name='divergence',
        package='divergence',
        function='divergence.causality.compute_granger',
        inputs=['observations.parquet', 'signal_pairwise.parquet'],
        outputs=['information_flow.parquet'],
        scale='signal',
        depends_on=['pairwise'],
        optional=True,
    ),
    PipelineStage(
        name='stability',
        package='stability',
        function='stability.rolling.compute_signal_stability',
        inputs=['observations.parquet'],
        outputs=['signal_stability.parquet'],
        scale='signal',
    ),
    PipelineStage(
        name='thermodynamics',
        package='thermodynamics',
        function='thermodynamics.thermo.compute_thermodynamics',
        inputs=['cohort_eigendecomp.parquet'],
        outputs=['cohort_thermodynamics.parquet'],
        scale='cohort',
        depends_on=['eigendecomp'],
    ),
    PipelineStage(
        name='topology',
        package='topology',
        function='topology.homology.compute_persistence',
        inputs=['signal_vector.parquet'],
        outputs=['persistent_homology.parquet'],
        scale='cohort',
        depends_on=['vector'],
        optional=True,
    ),
    PipelineStage(
        name='breaks',
        package='breaks',
        function='breaks.detection.detect_breaks_cusum',
        inputs=['observations.parquet'],
        outputs=['breaks.parquet'],
        scale='signal',
    ),
    PipelineStage(
        name='baseline',
        package='baseline',
        function='baseline.reference.compute_fleet_baseline',
        inputs=['signal_vector.parquet'],
        outputs=['cohort_baseline.parquet', 'segment_comparison.parquet'],
        scale='fleet',
        depends_on=['vector'],
    ),
    PipelineStage(
        name='fleet',
        package='fleet',
        function='fleet.analysis.compute_fleet_eigendecomp',
        inputs=['cohort_eigendecomp.parquet'],
        outputs=['system_geometry.parquet', 'cohort_pairwise.parquet'],
        scale='fleet',
        depends_on=['eigendecomp'],
    ),
]


class Pipeline:
    """
    Orchestrates execution of all compute stages.

    Usage:
        pipeline = Pipeline(data_dir='./output')
        pipeline.run()
        pipeline.run(stages=['typology', 'vector', 'eigendecomp'])
    """

    def __init__(
        self,
        data_dir: str = '.',
        stages: Optional[List[PipelineStage]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.stages = stages or STAGES
        self._stage_map = {s.name: s for s in self.stages}

    def get_execution_order(
        self,
        include: Optional[List[str]] = None,
        skip_optional: bool = False,
    ) -> List[PipelineStage]:
        """
        Get topologically sorted execution order.

        Parameters
        ----------
        include : list of str, optional
            Only include these stages (plus dependencies).
        skip_optional : bool
            Skip optional stages.

        Returns
        -------
        list of PipelineStage in execution order.
        """
        if include:
            # Resolve dependencies recursively
            needed = set()
            to_process = list(include)
            while to_process:
                name = to_process.pop()
                if name in needed:
                    continue
                needed.add(name)
                stage = self._stage_map.get(name)
                if stage:
                    to_process.extend(stage.depends_on)
            stages = [s for s in self.stages if s.name in needed]
        else:
            stages = list(self.stages)

        if skip_optional:
            stages = [s for s in stages if not s.optional]

        return stages

    def check_inputs(self, stage: PipelineStage) -> List[str]:
        """Check which inputs exist for a stage."""
        missing = []
        for inp in stage.inputs:
            path = self.data_dir / inp
            if not path.exists():
                missing.append(inp)
        return missing

    def run_stage(self, stage: PipelineStage, dry_run: bool = False) -> Dict[str, Any]:
        """
        Run a single pipeline stage.

        Returns dict with status, timing, and any errors.
        """
        missing = self.check_inputs(stage)

        result = {
            'stage': stage.name,
            'package': stage.package,
            'scale': stage.scale,
            'missing_inputs': missing,
        }

        if missing and not stage.optional:
            result['status'] = 'skipped'
            result['reason'] = f'missing inputs: {missing}'
            return result

        if dry_run:
            result['status'] = 'dry_run'
            return result

        t0 = time.time()
        try:
            # Dynamic import and execution
            parts = stage.function.rsplit('.', 1)
            module_path, func_name = parts[0], parts[1]
            import importlib
            mod = importlib.import_module(module_path)
            func = getattr(mod, func_name)
            result['status'] = 'available'
            result['function'] = stage.function
        except (ImportError, AttributeError) as e:
            result['status'] = 'not_installed'
            result['error'] = str(e)

        result['elapsed'] = time.time() - t0
        return result

    def plan(self, include: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Dry-run: show what would execute and what's missing."""
        stages = self.get_execution_order(include)
        return [self.run_stage(s, dry_run=True) for s in stages]

    def list_stages(self) -> List[Dict[str, str]]:
        """List all stages with their metadata."""
        return [
            {
                'name': s.name,
                'package': s.package,
                'scale': s.scale,
                'inputs': s.inputs,
                'outputs': s.outputs,
                'optional': s.optional,
            }
            for s in self.stages
        ]

    def run(
        self,
        include: Optional[List[str]] = None,
        skip_optional: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run the complete pipeline.

        Parameters
        ----------
        include : list of str, optional
            Only run these stages (plus dependencies).
        skip_optional : bool
            Skip optional stages.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with:
            stages: list of stage results
            elapsed: total time
            outputs: list of output files created
        """
        stages = self.get_execution_order(include, skip_optional=skip_optional)
        results = []
        outputs = []
        t0 = time.time()

        for stage in stages:
            if verbose:
                print(f"  [{stage.name}] Running...")
            result = self.run_stage(stage, dry_run=False)
            results.append(result)

            if result.get('status') == 'available':
                outputs.extend(stage.outputs)
                if verbose:
                    print(f"  [{stage.name}] OK ({result.get('elapsed', 0):.2f}s)")
            elif result.get('status') == 'skipped':
                if verbose:
                    print(f"  [{stage.name}] Skipped: {result.get('reason', 'unknown')}")
            elif result.get('status') == 'not_installed':
                if verbose:
                    print(f"  [{stage.name}] Not installed: {result.get('error', 'unknown')}")

        return {
            'stages': results,
            'elapsed': time.time() - t0,
            'outputs': outputs,
        }


def run(
    observations_path: str,
    manifest_path: str,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete orchestration pipeline.

    This is the main entry point, matching the old manifold.run() signature.

    Parameters
    ----------
    observations_path : str
        Path to observations.parquet
    manifest_path : str
        Path to manifest.yaml
    output_dir : str
        Directory for output parquets
    verbose : bool
        Print progress

    Returns
    -------
    dict with pipeline results
    """
    import yaml
    import polars as pl
    import numpy as np
    from pathlib import Path

    t0 = time.time()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if verbose:
        print(f"  Observations: {observations_path}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Output: {output_dir}")

    # Load observations
    obs = pl.read_parquet(observations_path)
    signals = obs['signal_id'].unique().to_list()
    cohorts = obs['cohort'].unique().to_list() if 'cohort' in obs.columns else ['']

    if verbose:
        print(f"  Signals: {len(signals)}, Cohorts: {len(cohorts)}")

    # Get window parameters from manifest
    system_config = manifest.get('system', {})
    window_size = system_config.get('window', 64)
    stride = system_config.get('stride', window_size // 2)

    outputs_created = []

    # =========================================================================
    # Stage 1: Signal Vector — windowed features per signal
    # =========================================================================
    if verbose:
        print(f"  [vector] Computing signal features (window={window_size}, stride={stride})...")

    try:
        from vector.signal import compute_signal
        from vector.registry import get_registry

        registry = get_registry()
        all_rows = []

        for cohort in cohorts:
            if cohort:
                cohort_obs = obs.filter(pl.col('cohort') == cohort)
            else:
                cohort_obs = obs

            for signal_id in signals:
                signal_data = cohort_obs.filter(pl.col('signal_id') == signal_id)
                if len(signal_data) == 0:
                    continue

                values = signal_data['value'].to_numpy()
                if len(values) < window_size:
                    continue

                try:
                    rows = compute_signal(
                        signal_id=signal_id,
                        values=values,
                        window_size=window_size,
                        stride=stride,
                    )
                    for row in rows:
                        row['cohort'] = cohort
                    all_rows.extend(rows)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {signal_id} failed: {e}")

        if all_rows:
            signal_vector = pl.DataFrame(all_rows)
            sv_path = output_path / 'signal_vector.parquet'
            signal_vector.write_parquet(sv_path)
            outputs_created.append('signal_vector.parquet')
            if verbose:
                print(f"    → signal_vector.parquet ({len(signal_vector)} rows)")
        else:
            if verbose:
                print(f"    → No signal vectors computed (insufficient data)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: vector package not installed ({e})")

    # =========================================================================
    # Stage 2: Eigendecomposition — cohort geometry
    # =========================================================================
    sv_path = output_path / 'signal_vector.parquet'
    if sv_path.exists():
        if verbose:
            print(f"  [eigendecomp] Computing eigenstructure...")

        try:
            from eigendecomp import compute_eigendecomp, flatten_result

            signal_vector = pl.read_parquet(sv_path)

            # Get numeric feature columns
            meta_cols = {'signal_id', 'cohort', 'window_index', 'window_start', 'window_end', 'I'}
            feature_cols = [c for c in signal_vector.columns
                          if c not in meta_cols and signal_vector[c].dtype in [pl.Float64, pl.Float32]]

            if not feature_cols:
                if verbose:
                    print(f"    Skipped: no numeric feature columns")
            else:
                all_eigen_rows = []
                window_indices = sorted(signal_vector['window_index'].unique().to_list())

                for cohort in cohorts:
                    if cohort:
                        cohort_sv = signal_vector.filter(pl.col('cohort') == cohort)
                    else:
                        cohort_sv = signal_vector

                    for win_idx in window_indices:
                        win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                        if len(win_data) < 2:
                            continue

                        matrix = win_data.select(feature_cols).to_numpy()
                        if np.all(np.isnan(matrix)):
                            continue

                        try:
                            result = compute_eigendecomp(matrix)
                            row = flatten_result(result)
                            row['cohort'] = cohort
                            row['window_index'] = win_idx
                            row['I'] = win_idx
                            all_eigen_rows.append(row)
                        except Exception as e:
                            pass

                if all_eigen_rows:
                    eigen_df = pl.DataFrame(all_eigen_rows)
                    eigen_path = output_path / 'cohort_eigendecomp.parquet'
                    eigen_df.write_parquet(eigen_path)
                    outputs_created.append('cohort_eigendecomp.parquet')
                    if verbose:
                        print(f"    → cohort_eigendecomp.parquet ({len(eigen_df)} rows)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: eigendecomp package not installed ({e})")

    # =========================================================================
    # Stage 3: Geometry Dynamics — eigenvalue trajectories
    # =========================================================================
    eigen_path = output_path / 'cohort_eigendecomp.parquet'
    if eigen_path.exists():
        if verbose:
            print(f"  [geometry] Computing dynamics...")

        try:
            from geometry.dynamics import compute_eigenvalue_dynamics

            eigen_df = pl.read_parquet(eigen_path)

            all_dynamics_rows = []
            for cohort in cohorts:
                if cohort:
                    cohort_eigen = eigen_df.filter(pl.col('cohort') == cohort)
                else:
                    cohort_eigen = eigen_df

                cohort_eigen = cohort_eigen.sort('window_index')
                if len(cohort_eigen) < 3:
                    continue

                # Build eigendecomp results list
                eigendecomp_results = []
                for row in cohort_eigen.iter_rows(named=True):
                    eigendecomp_results.append({
                        'effective_dim': row.get('effective_dim', np.nan),
                        'eigenvalues': np.array([row.get(f'eigenvalue_{i}', np.nan) for i in range(10)]),
                        'total_variance': row.get('total_variance', np.nan),
                    })

                try:
                    dynamics = compute_eigenvalue_dynamics(eigendecomp_results)
                    for i, d in enumerate(dynamics):
                        d['cohort'] = cohort
                        d['window_index'] = int(cohort_eigen['window_index'][i])
                        d['I'] = d['window_index']
                        all_dynamics_rows.append(d)
                except Exception:
                    pass

            if all_dynamics_rows:
                dynamics_df = pl.DataFrame(all_dynamics_rows)
                dyn_path = output_path / 'geometry_dynamics.parquet'
                dynamics_df.write_parquet(dyn_path)
                outputs_created.append('geometry_dynamics.parquet')
                if verbose:
                    print(f"    → geometry_dynamics.parquet ({len(dynamics_df)} rows)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: geometry package not installed ({e})")

    elapsed = time.time() - t0

    if verbose:
        print(f"  Completed: {len(outputs_created)} outputs in {elapsed:.2f}s")

    return {
        'outputs': outputs_created,
        'elapsed': elapsed,
        'observations_path': observations_path,
        'manifest_path': manifest_path,
        'output_dir': output_dir,
    }
