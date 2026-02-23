"""
Pipeline runner: sequences all compute packages.

Defines the DAG of stages with inputs/outputs.
Runs stages in topological order, passing parquet paths.

No math lives here. Only wiring and file I/O.
"""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Any


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
