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


def _build_trajectory_library(traj_sig_rows, cohorts):
    """Cluster cohorts by trajectory shape into a trajectory library."""
    import numpy as np

    # Build per-cohort trajectory vectors
    cohort_trajectories = {}
    cohort_n_windows = {}

    for cohort in set(r['cohort'] for r in traj_sig_rows):
        rows = sorted(
            [r for r in traj_sig_rows if r['cohort'] == cohort],
            key=lambda x: x['signal_0_end'],
        )
        if len(rows) < 2:
            continue

        features = []
        for r in rows:
            feat = [
                r.get('eigenvalue_1', 0.0),
                r.get('eigenvalue_2', 0.0),
                r.get('eigenvalue_3', 0.0),
                r.get('effective_dim', 0.0),
                r.get('total_variance', 0.0),
            ]
            features.append([f if np.isfinite(f) else 0.0 for f in feat])

        cohort_trajectories[cohort] = np.array(features)
        cohort_n_windows[cohort] = len(rows)

    if len(cohort_trajectories) < 2:
        return [], []

    cohort_ids = sorted(cohort_trajectories.keys())
    n = len(cohort_ids)

    # Pad to same length and flatten
    max_len = max(len(v) for v in cohort_trajectories.values())
    flat_trajs = {}
    for cid in cohort_ids:
        traj = cohort_trajectories[cid]
        padded = np.zeros((max_len, traj.shape[1]))
        padded[:len(traj)] = traj
        flat_trajs[cid] = padded.flatten()

    # Distance matrix
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(flat_trajs[cohort_ids[i]] - flat_trajs[cohort_ids[j]]))
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # k-medoids with k=2 (farthest pair as initial medoids)
    k = min(2, n)
    max_d = 0
    m1, m2 = 0, 1
    for i in range(n):
        for j in range(i + 1, n):
            if dist_matrix[i, j] > max_d:
                max_d = dist_matrix[i, j]
                m1, m2 = i, j

    medoids = [m1, m2][:k]

    # Assign clusters
    assignments = np.zeros(n, dtype=int)
    for i in range(n):
        dists = [dist_matrix[i, m] for m in medoids]
        assignments[i] = int(np.argmin(dists))

    # Build library rows
    lib_rows = []
    match_rows = []

    for cluster_id in range(k):
        members = [cohort_ids[i] for i in range(n) if assignments[i] == cluster_id]
        if not members:
            continue

        medoid = cohort_ids[medoids[cluster_id]]
        member_indices = [i for i in range(n) if assignments[i] == cluster_id]
        other_indices = [i for i in range(n) if assignments[i] != cluster_id]

        intra_dists = [dist_matrix[i, j] for i in member_indices for j in member_indices if i < j]
        inter_dists = [dist_matrix[i, j] for i in member_indices for j in other_indices]

        mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
        mean_inter = float(np.mean(inter_dists)) if inter_dists else 0.0
        silhouette = (mean_inter - mean_intra) / (max(mean_inter, mean_intra) + 1e-30)

        lib_rows.append({
            'trajectory_id': cluster_id,
            'n_members': len(members),
            'medoid_cohort': medoid,
            'member_cohorts': ','.join(members),
            'mean_intra_distance': mean_intra,
            'mean_inter_distance': mean_inter,
            'silhouette': silhouette,
            'compactness': mean_intra,
            'mean_n_windows': float(np.mean([cohort_n_windows.get(m, 0) for m in members])),
        })

    # Match rows
    for i, cid in enumerate(cohort_ids):
        cluster = assignments[i]
        dist_to_medoid = dist_matrix[i, medoids[cluster]]

        other_dists = [dist_matrix[i, medoids[c]] for c in range(k) if c != cluster]
        second_dist = float(min(other_dists)) if other_dists else np.nan

        confidence = 1.0 - dist_to_medoid / (second_dist + 1e-30) if np.isfinite(second_dist) and second_dist > 0 else 0.0

        match_rows.append({
            'cohort': cid,
            'trajectory_id': int(cluster),
            'match_distance': dist_to_medoid,
            'second_distance': second_dist,
            'match_confidence': max(0.0, confidence),
            'trajectory_position': 1.0,
            'n_windows': cohort_n_windows.get(cid, 0),
        })

    return lib_rows, match_rows


# ---------------------------------------------------------------------------
# Manifest config extraction
# ---------------------------------------------------------------------------

# Derivative depth → smooth_window mapping
_SMOOTH_MAP = {0: 1, 1: 1, 2: 3, 3: 5}

# Typology memory → Granger max_lag
_MEMORY_LAG_MAP = {'SHORT': 1, 'SHORT_MEMORY': 1, 'MODERATE': 5, 'LONG': 10, 'LONG_MEMORY': 10, 'ANTI_PERSISTENT': 3}

# Typology complexity → transfer entropy n_bins
_COMPLEXITY_BINS_MAP = {'LOW': 4, 'MODERATE': 8, 'HIGH': 16}


def build_package_configs(manifest: dict) -> dict:
    """
    Extract manifest fields and map to per-package config dicts.
    Called once at pipeline start. Each package gets only what it needs.
    """
    system = manifest.get('system', {})
    window = system.get('window', 64)
    stride = system.get('stride', window // 2)
    eigenvalue_budget = system.get('eigenvalue_budget', 10)

    return {
        'eigendecomp': {
            'max_eigenvalues': eigenvalue_budget,
        },
        'geometry': {
            'max_eigenvalues': eigenvalue_budget,
        },
        'stability': {
            'window_size': window,
            'stride': stride,
        },
        'divergence': {
            'max_lag': 5,
            'n_bins': 8,
        },
        'velocity': {},
        'ridge': {},
        'topology': {
            'max_dim': 1,
            'max_points': 500,
        },
        'baseline': {},
        'breaks': {},
        'dynamics': {},
        'thermodynamics': {},
        'fleet': {
            'system_mode': system.get('mode', 'auto'),
        },
        'engine_gates': manifest.get('engine_gates', {}),
        'skip_signals': manifest.get('skip_signals', []),
    }


def build_per_signal_configs(manifest: dict) -> dict:
    """
    Extract per-signal configs from manifest cohorts block.
    Returns dict keyed by (cohort_id, signal_id).
    """
    per_signal = {}
    for cohort_id, signals in manifest.get('cohorts', {}).items():
        if not isinstance(signals, dict):
            continue
        for signal_id, sig_config in signals.items():
            if not isinstance(sig_config, dict):
                continue
            per_signal[(cohort_id, signal_id)] = sig_config
    return per_signal


def get_signal_config(
    package_configs: dict,
    package_name: str,
    cohort_id: str,
    signal_id: str,
    per_signal: dict,
) -> dict:
    """
    Build final config for a specific package + signal combination.
    Merges package defaults with per-signal overrides from manifest.
    """
    base = package_configs.get(package_name, {}).copy()

    sig_key = (cohort_id, signal_id)
    sig_config = per_signal.get(sig_key, {})
    typology = sig_config.get('typology', {})

    # Derivative depth → smooth_window mapping
    deriv_depth = sig_config.get('derivative_depth', 1)
    if package_name in ('geometry', 'velocity', 'ridge'):
        base['smooth_window'] = _SMOOTH_MAP.get(deriv_depth, 3)

    # Typology-driven overrides for divergence
    if package_name == 'divergence':
        memory = typology.get('memory', 'MODERATE')
        complexity = typology.get('complexity', 'MODERATE')
        base['max_lag'] = _MEMORY_LAG_MAP.get(memory, 5)
        base['n_bins'] = _COMPLEXITY_BINS_MAP.get(complexity, 8)

    # Eigenvalue budget per-signal override
    if package_name in ('eigendecomp', 'geometry'):
        budget = sig_config.get('eigenvalue_budget')
        if budget is not None:
            base['max_eigenvalues'] = budget

    # D2 onset for dynamics/FTLE caching
    if package_name == 'dynamics':
        base['d2_onset_pct'] = typology.get('d2_onset_pct')

    # Topology: scale max_points with signal count
    if package_name == 'topology':
        n_signals = len([k for k in per_signal if k[0] == cohort_id])
        if n_signals < 10:
            base['max_points'] = 200
        elif n_signals < 20:
            base['max_points'] = 500
        else:
            base['max_points'] = 1000

    return base


def run(
    observations_path: str,
    manifest_path: str,
    output_dir: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run the complete orchestration pipeline.

    This is the main entry point, matching the old manifold.run() signature.
    Wires all 16 compute packages and writes outputs to the correct
    subdirectory structure under output_dir.

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

    # Create subdirectory structure
    for subdir in ['signal', 'cohort', 'cohort/cohort_dynamics',
                   'system', 'system/system_dynamics', 'parameterization']:
        (output_path / subdir).mkdir(parents=True, exist_ok=True)

    # Load manifest
    with open(manifest_path) as f:
        manifest = yaml.safe_load(f)

    if verbose:
        print(f"  Observations: {observations_path}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Output: {output_dir}")

    # Load observations
    obs = pl.read_parquet(observations_path)
    signals = sorted(obs['signal_id'].unique().to_list())
    cohorts = sorted(obs['cohort'].unique().to_list()) if 'cohort' in obs.columns else ['']
    n_cohorts = len(cohorts)

    if verbose:
        print(f"  Signals: {len(signals)}, Cohorts: {n_cohorts}")

    # Window parameters from manifest
    system_config = manifest.get('system', {})
    window_size = system_config.get('window', 64)
    stride = system_config.get('stride', window_size // 2)

    # Build per-package and per-signal config from manifest
    pkg_configs = build_package_configs(manifest)
    per_signal_configs = build_per_signal_configs(manifest)
    skip_signals = set(pkg_configs.get('skip_signals', []))

    # Build signal data lookup: (cohort, signal_id) → {values, signal_0}
    signal_lookup: Dict[tuple, dict] = {}
    for cohort in cohorts:
        cohort_obs = obs.filter(pl.col('cohort') == cohort) if cohort else obs
        for signal_id in signals:
            sig = cohort_obs.filter(pl.col('signal_id') == signal_id).sort('signal_0')
            if len(sig) == 0:
                continue
            signal_lookup[(cohort, signal_id)] = {
                'values': sig['value'].to_numpy(),
                'signal_0': sig['signal_0'].to_numpy(),
            }

    outputs_created: List[str] = []

    # In-memory stores for cross-stage dependencies
    eigen_store: Dict[tuple, dict] = {}   # (cohort, window_index) → raw eigendecomp result
    signal_vector_df = None
    feature_cols: List[str] = []

    # =====================================================================
    # STAGE 1: Signal Vector → signal/signal_vector.parquet
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [1/16 vector] Signal features (window={window_size}, stride={stride})...")

    try:
        from vector.signal import compute_signal

        all_rows = []
        for cohort in cohorts:
            for signal_id in signals:
                if signal_id in skip_signals:
                    continue
                key = (cohort, signal_id)
                if key not in signal_lookup:
                    continue
                sd = signal_lookup[key]
                values = sd['values']
                s0 = sd['signal_0']

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
                        # Map window indices to signal_0 space
                        ws = row.get('window_start', 0)
                        we = row.get('window_end', 0)
                        row['signal_0_start'] = float(s0[min(ws, len(s0) - 1)])
                        row['signal_0_end'] = float(s0[min(we, len(s0) - 1)])
                        row['signal_0_center'] = (row['signal_0_start'] + row['signal_0_end']) / 2.0
                        # Remove index-based columns (keep window_index for internal use)
                        row.pop('window_start', None)
                        row.pop('window_end', None)
                        row.pop('window_center', None)
                        row.pop('window_n_samples', None)
                    all_rows.extend(rows)
                except Exception as e:
                    if verbose:
                        print(f"    Warning: {signal_id}@{cohort}: {e}")

        if all_rows:
            signal_vector_df = pl.DataFrame(all_rows)
            sv_path = output_path / 'signal' / 'signal_vector.parquet'
            signal_vector_df.write_parquet(sv_path)
            outputs_created.append('signal/signal_vector.parquet')
            if verbose:
                print(f"    → signal/signal_vector.parquet ({len(signal_vector_df)} rows, {time.time() - _stage_t:.1f}s)")
        else:
            if verbose:
                print(f"    → No signal vectors (insufficient data)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # Identify feature columns from signal_vector
    if signal_vector_df is not None:
        meta_cols = {'signal_id', 'cohort', 'window_index', 'signal_0_start',
                     'signal_0_end', 'signal_0_center', 'engine_mask'}
        feature_cols = [c for c in signal_vector_df.columns
                       if c not in meta_cols
                       and signal_vector_df[c].dtype in [pl.Float64, pl.Float32]]

    # =====================================================================
    # STAGE 2: Eigendecomposition → cohort/cohort_geometry.parquet
    #                              + cohort/cohort_feature_loadings.parquet
    # =====================================================================
    if signal_vector_df is not None and feature_cols:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [2/16 eigendecomp] Eigenstructure...")

        try:
            from eigendecomp import compute_eigendecomp, flatten_result

            all_eigen_rows = []
            all_loadings_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)

                for win_idx in window_indices:
                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 2:
                        continue

                    full_matrix = win_data.select(feature_cols).to_numpy()
                    if np.all(np.isnan(full_matrix)):
                        continue

                    # Drop all-NaN columns before eigendecomp so rows aren't
                    # rejected for having NaN in irrelevant feature slots
                    col_has_data = ~np.all(np.isnan(full_matrix), axis=0)
                    matrix = full_matrix[:, col_has_data]
                    valid_feature_cols = [f for f, ok in zip(feature_cols, col_has_data) if ok]

                    if matrix.shape[1] < 2:
                        continue

                    s0_end = float(win_data['signal_0_end'][0])
                    s0_start = float(win_data['signal_0_start'][0])
                    s0_center = float(win_data['signal_0_center'][0])

                    try:
                        eigen_cfg = pkg_configs.get('eigendecomp', {})
                        eigen_max = eigen_cfg.get('max_eigenvalues', 10)
                        # Can't extract more eigenvalues than n_signals - 1
                        eigen_max = min(eigen_max, matrix.shape[0] - 1)
                        result = compute_eigendecomp(
                            matrix,
                            max_eigenvalues=eigen_max,
                            config=eigen_cfg,
                        )
                        # Store which columns of the full feature set were used
                        result['_col_has_data'] = col_has_data
                        eigen_store[(cohort, win_idx)] = result

                        row = flatten_result(result, max_eigenvalues=eigen_max)
                        row['cohort'] = cohort
                        row['window_index'] = win_idx
                        row['signal_0_end'] = s0_end
                        row['signal_0_start'] = s0_start
                        row['signal_0_center'] = s0_center
                        all_eigen_rows.append(row)

                        # Feature loadings: PC1 loading per feature
                        # pcs has shape (n_varying, n_varying) — only covers
                        # features where variance > 0, identified by varying_mask
                        pcs = result.get('principal_components')
                        varying_mask = result.get('varying_mask')
                        if pcs is not None and pcs.ndim == 2 and pcs.shape[0] > 0:
                            if varying_mask is not None:
                                varying_features = [f for f, v in zip(valid_feature_cols, varying_mask) if v]
                            else:
                                varying_features = valid_feature_cols
                            for v_idx, feat in enumerate(varying_features):
                                if v_idx < pcs.shape[1]:
                                    all_loadings_rows.append({
                                        'signal_0_end': s0_end,
                                        'cohort': cohort,
                                        'engine': 'aggregate',
                                        'feature': feat,
                                        'pc1_loading': float(pcs[0, v_idx]),
                                    })
                    except Exception:
                        pass

            if all_eigen_rows:
                eigen_df = pl.DataFrame(all_eigen_rows)
                eigen_df.write_parquet(output_path / 'cohort' / 'cohort_geometry.parquet')
                outputs_created.append('cohort/cohort_geometry.parquet')
                if verbose:
                    print(f"    → cohort/cohort_geometry.parquet ({len(eigen_df)} rows)")

            if all_loadings_rows:
                loadings_df = pl.DataFrame(all_loadings_rows)
                loadings_df.write_parquet(output_path / 'cohort' / 'cohort_feature_loadings.parquet')
                outputs_created.append('cohort/cohort_feature_loadings.parquet')
                if verbose:
                    print(f"    → cohort/cohort_feature_loadings.parquet ({len(loadings_df)} rows)")

            if verbose:
                print(f"    ({time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 3: Signal Geometry → signal/signal_geometry.parquet
    #                           + cohort/cohort_signal_positions.parquet
    # =====================================================================
    if signal_vector_df is not None and eigen_store:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [3/16 geometry] Signal geometry...")

        try:
            from geometry.signal import compute_signal_geometry

            all_geom_rows = []
            all_positions_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)

                for win_idx in window_indices:
                    ekey = (cohort, win_idx)
                    if ekey not in eigen_store:
                        continue

                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 2:
                        continue

                    full_matrix = win_data.select(feature_cols).to_numpy()
                    sig_ids = win_data['signal_id'].to_list()
                    s0_end = float(win_data['signal_0_end'][0])

                    er = eigen_store[ekey]
                    # Filter to same columns used by eigendecomp,
                    # then further to varying columns (what PCs are built from)
                    col_mask = er.get('_col_has_data')
                    varying_mask = er.get('varying_mask')
                    matrix = full_matrix[:, col_mask] if col_mask is not None else full_matrix
                    if varying_mask is not None and len(varying_mask) == matrix.shape[1]:
                        matrix = matrix[:, varying_mask]
                    centroid = np.nanmean(matrix, axis=0)
                    pcs = er.get('principal_components')

                    try:
                        rows = compute_signal_geometry(
                            signal_matrix=matrix,
                            signal_ids=sig_ids,
                            centroid=centroid,
                            principal_components=pcs,
                            window_index=win_idx,
                        )
                        for r in rows:
                            # Extract PC projections before adding to geom rows
                            pc0 = r.pop('pc0_projection', np.nan)
                            pc1 = r.pop('pc1_projection', np.nan)
                            pc2 = r.pop('pc2_projection', np.nan)
                            r.pop('I', None)

                            r['cohort'] = cohort
                            r['signal_0_end'] = s0_end
                            r['engine'] = 'aggregate'
                            if 'signal_magnitude' in r:
                                r['magnitude'] = r.pop('signal_magnitude')
                            all_geom_rows.append(r)

                            all_positions_rows.append({
                                'signal_0_end': s0_end,
                                'cohort': cohort,
                                'engine': 'aggregate',
                                'signal_id': r['signal_id'],
                                'pc1_loading': pc0,
                                'pc2_loading': pc1,
                                'pc3_loading': pc2,
                            })
                    except Exception:
                        pass

            if all_geom_rows:
                geom_df = pl.DataFrame(all_geom_rows)
                geom_df.write_parquet(output_path / 'signal' / 'signal_geometry.parquet')
                outputs_created.append('signal/signal_geometry.parquet')
                if verbose:
                    print(f"    → signal/signal_geometry.parquet ({len(geom_df)} rows)")

            if all_positions_rows:
                pos_df = pl.DataFrame(all_positions_rows)
                pos_df.write_parquet(output_path / 'cohort' / 'cohort_signal_positions.parquet')
                outputs_created.append('cohort/cohort_signal_positions.parquet')
                if verbose:
                    print(f"    → cohort/cohort_signal_positions.parquet ({len(pos_df)} rows)")

            if verbose:
                print(f"    ({time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 4: Geometry Dynamics → cohort/cohort_dynamics/geometry_dynamics.parquet
    # =====================================================================
    if eigen_store:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [4/16 geometry_dynamics] Eigenvalue dynamics...")

        try:
            from geometry.dynamics import compute_eigenvalue_dynamics

            all_dynamics_rows = []

            for cohort in cohorts:
                cohort_keys = sorted(
                    [k for k in eigen_store if k[0] == cohort], key=lambda x: x[1]
                )
                if len(cohort_keys) < 3:
                    continue

                eigendecomp_results = []
                s0_meta = []  # (s0_start, s0_end, s0_center) per window

                for ck in cohort_keys:
                    er = eigen_store[ck]
                    eigendecomp_results.append({
                        'effective_dim': er.get('effective_dim', np.nan),
                        'eigenvalues': er.get('eigenvalues', np.array([])),
                        'total_variance': er.get('total_variance', np.nan),
                    })

                    win_idx = ck[1]
                    if signal_vector_df is not None:
                        sv_row = signal_vector_df.filter(
                            (pl.col('cohort') == cohort) & (pl.col('window_index') == win_idx)
                        )
                        if len(sv_row) > 0:
                            s0_meta.append((
                                float(sv_row['signal_0_start'][0]),
                                float(sv_row['signal_0_end'][0]),
                                float(sv_row['signal_0_center'][0]),
                            ))
                        else:
                            s0_meta.append((np.nan, np.nan, np.nan))
                    else:
                        s0_meta.append((np.nan, np.nan, np.nan))

                try:
                    geom_cfg = pkg_configs.get('geometry', {})
                    dynamics = compute_eigenvalue_dynamics(
                        eigendecomp_results,
                        max_eigenvalues=geom_cfg.get('max_eigenvalues', 5),
                        config=geom_cfg,
                    )
                    for i, d in enumerate(dynamics):
                        d['cohort'] = cohort
                        d['signal_0_start'] = s0_meta[i][0]
                        d['signal_0_end'] = s0_meta[i][1]
                        d['signal_0_center'] = s0_meta[i][2]
                        all_dynamics_rows.append(d)
                except Exception:
                    pass

            if all_dynamics_rows:
                dynamics_df = pl.DataFrame(all_dynamics_rows)
                dyn_path = output_path / 'cohort' / 'cohort_dynamics' / 'geometry_dynamics.parquet'
                dynamics_df.write_parquet(dyn_path)
                outputs_created.append('cohort/cohort_dynamics/geometry_dynamics.parquet')
                if verbose:
                    print(f"    → cohort/cohort_dynamics/geometry_dynamics.parquet ({len(dynamics_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 5: Pairwise → cohort/cohort_pairwise.parquet
    # =====================================================================
    if signal_vector_df is not None and feature_cols:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [5/16 pairwise] Signal pairwise metrics...")

        try:
            from pairwise import compute_signal_pairwise

            all_pair_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)

                for win_idx in window_indices:
                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 2:
                        continue

                    full_matrix = win_data.select(feature_cols).to_numpy()
                    sig_ids = win_data['signal_id'].to_list()
                    s0_end = float(win_data['signal_0_end'][0])

                    # Filter out all-NaN columns for cleaner pairwise metrics
                    col_ok = ~np.all(np.isnan(full_matrix), axis=0)
                    matrix = full_matrix[:, col_ok]
                    centroid = np.nanmean(matrix, axis=0) if (cohort, win_idx) in eigen_store else None

                    try:
                        rows = compute_signal_pairwise(
                            signal_matrix=matrix,
                            signal_ids=sig_ids,
                            centroid=centroid,
                            window_index=win_idx,
                        )
                        for r in rows:
                            r['cohort'] = cohort
                            r['signal_0_end'] = s0_end
                            r.pop('I', None)
                        all_pair_rows.extend(rows)
                    except Exception:
                        pass

            if all_pair_rows:
                pair_df = pl.DataFrame(all_pair_rows)
                pair_df.write_parquet(output_path / 'cohort' / 'cohort_pairwise.parquet')
                outputs_created.append('cohort/cohort_pairwise.parquet')
                if verbose:
                    print(f"    → cohort/cohort_pairwise.parquet ({len(pair_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 6: FTLE → cohort/cohort_dynamics/ftle*.parquet + lyapunov.parquet
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [6/16 dynamics] FTLE + Lyapunov...")

    try:
        from dynamics import compute_ftle, compute_ftle_rolling

        ftle_fwd_rows = []
        ftle_bwd_rows = []
        ftle_rolling_rows = []
        lyapunov_rows = []

        for cohort in cohorts:
            for signal_id in signals:
                key = (cohort, signal_id)
                if key not in signal_lookup:
                    continue
                sd = signal_lookup[key]
                values = sd['values']
                s0 = sd['signal_0']

                # Forward FTLE
                try:
                    result = compute_ftle(values, direction='forward', min_samples=50)
                    result['signal_id'] = signal_id
                    result['cohort'] = cohort
                    ftle_fwd_rows.append(result)

                    lyapunov_rows.append({
                        'signal_id': signal_id,
                        'cohort': cohort,
                        'lyapunov': result.get('ftle', np.nan),
                        'embedding_dim': result.get('embedding_dim', 0),
                        'embedding_tau': result.get('embedding_tau', 0),
                        'confidence': result.get('confidence', 0.0),
                        'n_samples': result.get('n_samples', 0),
                    })
                except Exception:
                    pass

                # Backward FTLE
                try:
                    result = compute_ftle(values, direction='backward', min_samples=50)
                    result['signal_id'] = signal_id
                    result['cohort'] = cohort
                    ftle_bwd_rows.append(result)
                except Exception:
                    pass

                # Rolling FTLE
                try:
                    ftle_win = min(500, len(values) // 2)
                    ftle_min_samples = min(50, ftle_win)
                    if ftle_win >= 50:
                        dyn_cfg = get_signal_config(pkg_configs, 'dynamics', cohort or 'system', signal_id, per_signal_configs)
                        d2_onset = dyn_cfg.get('d2_onset_pct')
                        rolls = compute_ftle_rolling(
                            values, window_size=ftle_win, stride=stride,
                            min_samples=ftle_min_samples,
                            d2_onset_pct=d2_onset,
                        )
                        for r in rolls:
                            r['signal_id'] = signal_id
                            r['cohort'] = cohort
                            ws = r.pop('window_start', 0)
                            we = r.pop('window_end', 0)
                            center_idx = r.pop('I', 0)
                            r['signal_0_start'] = float(s0[min(ws, len(s0) - 1)])
                            r['signal_0_end'] = float(s0[min(we, len(s0) - 1)])
                            r['signal_0_center'] = float(s0[min(int(center_idx), len(s0) - 1)])
                        ftle_rolling_rows.extend(rolls)
                except Exception:
                    pass

        if ftle_fwd_rows:
            pl.DataFrame(ftle_fwd_rows).write_parquet(
                output_path / 'cohort' / 'cohort_dynamics' / 'ftle.parquet')
            outputs_created.append('cohort/cohort_dynamics/ftle.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/ftle.parquet ({len(ftle_fwd_rows)} rows)")

        if ftle_bwd_rows:
            pl.DataFrame(ftle_bwd_rows).write_parquet(
                output_path / 'cohort' / 'cohort_dynamics' / 'ftle_backward.parquet')
            outputs_created.append('cohort/cohort_dynamics/ftle_backward.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/ftle_backward.parquet ({len(ftle_bwd_rows)} rows)")

        if ftle_rolling_rows:
            pl.DataFrame(ftle_rolling_rows).write_parquet(
                output_path / 'cohort' / 'cohort_dynamics' / 'ftle_rolling.parquet')
            outputs_created.append('cohort/cohort_dynamics/ftle_rolling.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/ftle_rolling.parquet ({len(ftle_rolling_rows)} rows)")

        if lyapunov_rows:
            pl.DataFrame(lyapunov_rows).write_parquet(
                output_path / 'cohort' / 'cohort_dynamics' / 'lyapunov.parquet')
            outputs_created.append('cohort/cohort_dynamics/lyapunov.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/lyapunov.parquet ({len(lyapunov_rows)} rows)")

        # FTLE field: inter-cohort FTLE ridges from pairwise centroid interpolation
        if signal_vector_df is not None and feature_cols and len(cohorts) >= 2:
            ftle_field_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            # Collect per-cohort centroids at each window
            cohort_centroids_by_win: Dict[int, Dict[str, Any]] = {}
            for win_idx in window_indices:
                win_data = signal_vector_df.filter(pl.col('window_index') == win_idx)
                centroids: Dict[str, Any] = {}
                for cohort in cohorts:
                    cw = win_data.filter(pl.col('cohort') == cohort)
                    if len(cw) >= 2:
                        matrix = cw.select(feature_cols).to_numpy()
                        centroids[cohort] = np.nanmean(matrix, axis=0)
                if centroids:
                    cohort_centroids_by_win[win_idx] = centroids

            # Average centroids across windows per cohort
            avg_centroids: Dict[str, Any] = {}
            for cohort in cohorts:
                vecs = [cohort_centroids_by_win[w][cohort]
                        for w in cohort_centroids_by_win if cohort in cohort_centroids_by_win[w]]
                if vecs:
                    avg_centroids[cohort] = np.nanmean(vecs, axis=0)

            # For each pair of cohorts: interpolate path, compute FTLE along path
            cohort_list = sorted(avg_centroids.keys())
            n_interp = 25  # interpolation points between centroids
            for i_c in range(len(cohort_list)):
                for j_c in range(i_c + 1, len(cohort_list)):
                    ca_name, cb_name = cohort_list[i_c], cohort_list[j_c]
                    ca, cb = avg_centroids[ca_name], avg_centroids[cb_name]
                    dist = float(np.linalg.norm(ca - cb))
                    if dist < 1e-12:
                        continue

                    # Interpolate path and compute FTLE at each point
                    alphas = np.linspace(0, 1, n_interp)
                    path_ftles = []
                    for alpha in alphas:
                        pt = (1 - alpha) * ca + alpha * cb
                        if len(pt) >= 50:
                            try:
                                r = compute_ftle(pt, min_samples=min(50, len(pt)))
                                path_ftles.append(r.get('ftle', 0.0))
                            except Exception:
                                path_ftles.append(0.0)
                        else:
                            path_ftles.append(0.0)

                    path_ftles = np.array(path_ftles)
                    max_idx = int(np.argmax(path_ftles))
                    ridge_strength = float(path_ftles[max_idx])
                    ridge_width = float(np.std(path_ftles)) if len(path_ftles) > 1 else 0.0

                    # Corridor width: fraction of path with FTLE > median
                    median_ftle = float(np.median(path_ftles))
                    above = np.sum(path_ftles > median_ftle)
                    corridor_width = float(above / len(path_ftles)) if len(path_ftles) > 0 else 0.0

                    # Ridge location as string of interpolated centroid coordinates
                    ridge_pt = (1 - alphas[max_idx]) * ca + alphas[max_idx] * cb
                    ridge_loc_str = str(ridge_pt.tolist())

                    ftle_field_rows.append({
                        'engine': 'aggregate',
                        'centroid_a': ca_name,
                        'centroid_b': cb_name,
                        'ridge_location': ridge_loc_str,
                        'ridge_strength': ridge_strength,
                        'ridge_width': ridge_width,
                        'corridor_width': corridor_width,
                        'inter_centroid_distance': dist,
                    })

            if ftle_field_rows:
                pl.DataFrame(ftle_field_rows).write_parquet(
                    output_path / 'cohort' / 'cohort_dynamics' / 'ftle_field.parquet')
                outputs_created.append('cohort/cohort_dynamics/ftle_field.parquet')
                if verbose:
                    print(f"    → cohort/cohort_dynamics/ftle_field.parquet ({len(ftle_field_rows)} rows)")

        if verbose:
            print(f"    ({time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 7: Velocity → cohort/cohort_dynamics/velocity_field*.parquet
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [7/16 velocity] State-space velocity...")

    try:
        from velocity import compute_velocity_field

        all_velocity_rows = []
        all_component_rows = []

        for cohort in cohorts:
            # Find a reference signal for this cohort to get signal_0 values
            first_key = next(((c, s) for (c, s) in signal_lookup if c == cohort), None)
            if first_key is None:
                continue

            s0_vals = signal_lookup[first_key]['signal_0']
            n_timesteps = len(s0_vals)

            if n_timesteps < 3:
                continue

            # Build wide matrix: rows = timesteps, cols = signals
            valid_signals = []
            signal_arrays = []
            for signal_id in signals:
                key = (cohort, signal_id)
                if key in signal_lookup:
                    vals = signal_lookup[key]['values']
                    if len(vals) == n_timesteps:
                        valid_signals.append(signal_id)
                        signal_arrays.append(vals)

            if len(valid_signals) < 2:
                continue

            matrix = np.column_stack(signal_arrays)  # (n_timesteps, n_signals)

            try:
                # Use first signal's derivative_depth for cohort-level smooth_window
                vel_cfg = get_signal_config(pkg_configs, 'velocity', cohort, valid_signals[0], per_signal_configs)
                rows = compute_velocity_field(
                    matrix=matrix,
                    signal_ids=valid_signals,
                    indices=s0_vals,
                    smooth_window=vel_cfg.get('smooth_window', 1),
                    config=vel_cfg,
                )
                for r in rows:
                    idx = r.pop('I', 0)
                    i_pos = int(np.searchsorted(s0_vals, idx))
                    r['signal_0_end'] = float(idx)
                    r['signal_0_start'] = float(s0_vals[max(0, i_pos - 1)])
                    r['signal_0_center'] = (r['signal_0_start'] + r['signal_0_end']) / 2.0
                    r['cohort'] = cohort
                all_velocity_rows.extend(rows)
            except Exception:
                pass

            # Velocity components per signal
            try:
                v = np.diff(matrix, axis=0)  # (n-1, n_signals)
                for i in range(len(v)):
                    s0_val = float(s0_vals[i + 1])
                    s0_prev = float(s0_vals[i])
                    s0_c = (s0_prev + s0_val) / 2.0
                    for j, sid in enumerate(valid_signals):
                        all_component_rows.append({
                            'signal_0_end': s0_val,
                            'signal_0_start': s0_prev,
                            'signal_0_center': s0_c,
                            'cohort': cohort,
                            'signal_id': sid,
                            'velocity': float(v[i, j]),
                        })
            except Exception:
                pass

        if all_velocity_rows:
            vel_df = pl.DataFrame(all_velocity_rows)
            vel_df.write_parquet(output_path / 'cohort' / 'cohort_dynamics' / 'velocity_field.parquet')
            outputs_created.append('cohort/cohort_dynamics/velocity_field.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/velocity_field.parquet ({len(vel_df)} rows)")

        if all_component_rows:
            comp_df = pl.DataFrame(all_component_rows)
            comp_df.write_parquet(output_path / 'cohort' / 'cohort_dynamics' / 'velocity_field_components.parquet')
            outputs_created.append('cohort/cohort_dynamics/velocity_field_components.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/velocity_field_components.parquet ({len(comp_df)} rows)")

        if verbose:
            print(f"    ({time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 8: Ridge Proximity → cohort/cohort_dynamics/ridge_proximity.parquet
    # =====================================================================
    ftle_rolling_path = output_path / 'cohort' / 'cohort_dynamics' / 'ftle_rolling.parquet'
    velocity_path = output_path / 'cohort' / 'cohort_dynamics' / 'velocity_field.parquet'

    if ftle_rolling_path.exists() and velocity_path.exists():
        _stage_t = time.time()
        if verbose:
            print(f"\n  [8/16 ridge] Ridge proximity...")

        try:
            from ridge import compute_ridge_proximity

            ftle_roll_df = pl.read_parquet(ftle_rolling_path)
            vel_df = pl.read_parquet(velocity_path)

            all_ridge_rows = []

            for cohort in cohorts:
                cohort_ftle = ftle_roll_df.filter(pl.col('cohort') == cohort)
                cohort_vel = vel_df.filter(pl.col('cohort') == cohort)

                if len(cohort_ftle) == 0 or len(cohort_vel) == 0:
                    continue

                for signal_id in cohort_ftle['signal_id'].unique().to_list():
                    sig_ftle = cohort_ftle.filter(
                        pl.col('signal_id') == signal_id
                    ).sort('signal_0_center')

                    if len(sig_ftle) < 3:
                        continue

                    ftle_vals = sig_ftle['ftle'].to_numpy()
                    ftle_s0 = sig_ftle['signal_0_center'].to_numpy()

                    vel_sorted = cohort_vel.sort('signal_0_center')
                    vel_s0 = vel_sorted['signal_0_center'].to_numpy()
                    speed_arr = vel_sorted['speed'].to_numpy()

                    # Interpolate speed to FTLE window centers
                    speed_aligned = np.interp(ftle_s0, vel_s0, speed_arr)

                    try:
                        ridge_cfg = get_signal_config(pkg_configs, 'ridge', cohort, signal_id, per_signal_configs)
                        rows = compute_ridge_proximity(
                            ftle_series=ftle_vals,
                            speed_series=speed_aligned,
                            indices=ftle_s0,
                            ridge_threshold=ridge_cfg.get('ridge_threshold', 0.05),
                            smooth_window=ridge_cfg.get('smooth_window', 3),
                            config=ridge_cfg,
                        )
                        for r in rows:
                            r['cohort'] = cohort
                            r['signal_id'] = signal_id
                            idx = r.pop('I', 0)
                            r['signal_0_center'] = float(idx)
                            i_pos = int(np.searchsorted(ftle_s0, idx))
                            i_pos = min(i_pos, len(sig_ftle) - 1)
                            r['signal_0_end'] = float(sig_ftle['signal_0_end'][i_pos])
                            r['signal_0_start'] = float(sig_ftle['signal_0_start'][i_pos])
                        all_ridge_rows.extend(rows)
                    except Exception:
                        pass

            if all_ridge_rows:
                ridge_df = pl.DataFrame(all_ridge_rows)
                ridge_df.write_parquet(
                    output_path / 'cohort' / 'cohort_dynamics' / 'ridge_proximity.parquet')
                outputs_created.append('cohort/cohort_dynamics/ridge_proximity.parquet')
                if verbose:
                    print(f"    → cohort/cohort_dynamics/ridge_proximity.parquet ({len(ridge_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")
    elif verbose:
        print(f"\n  [8/16 ridge] Skipped: missing ftle_rolling or velocity inputs")

    # =====================================================================
    # STAGE 9: Stability → signal/signal_stability.parquet
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [9/16 stability] Signal stability...")

    try:
        from stability import compute_signal_stability

        all_stability_rows = []

        for cohort in cohorts:
            for signal_id in signals:
                key = (cohort, signal_id)
                if key not in signal_lookup:
                    continue
                sd = signal_lookup[key]
                values = sd['values']
                s0 = sd['signal_0']

                if len(values) < window_size:
                    continue

                try:
                    stab_cfg = pkg_configs.get('stability', {})
                    rows = compute_signal_stability(
                        values=values,
                        window_size=stab_cfg.get('window_size', window_size),
                        stride=stab_cfg.get('stride', stride),
                        config=stab_cfg,
                    )
                    for r in rows:
                        r['signal_id'] = signal_id
                        r['cohort'] = cohort
                        center_idx = int(r.pop('I', 0))
                        center_idx = min(center_idx, len(s0) - 1)
                        start_idx = max(0, center_idx - window_size // 2)
                        end_idx = min(len(s0) - 1, center_idx + window_size // 2)
                        r['signal_0_center'] = float(s0[center_idx])
                        r['signal_0_start'] = float(s0[start_idx])
                        r['signal_0_end'] = float(s0[end_idx])
                    all_stability_rows.extend(rows)
                except Exception:
                    pass

        if all_stability_rows:
            stab_df = pl.DataFrame(all_stability_rows)
            stab_df.write_parquet(output_path / 'signal' / 'signal_stability.parquet')
            outputs_created.append('signal/signal_stability.parquet')
            if verbose:
                print(f"    → signal/signal_stability.parquet ({len(stab_df)} rows, {time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 10: Thermodynamics → cohort/cohort_dynamics/thermodynamics.parquet
    # =====================================================================
    if eigen_store:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [10/16 thermo] Thermodynamic analogs...")

        try:
            from thermodynamics import compute_thermodynamics

            all_thermo_rows = []

            for cohort in cohorts:
                cohort_keys = sorted(
                    [k for k in eigen_store if k[0] == cohort], key=lambda x: x[1]
                )
                if len(cohort_keys) < 3:
                    continue

                eigenvalues_seq = []
                eff_dim_seq = []
                total_var_seq = []

                for ck in cohort_keys:
                    er = eigen_store[ck]
                    eigenvalues_seq.append(er.get('eigenvalues', np.array([])))
                    eff_dim_seq.append(er.get('effective_dim', np.nan))
                    total_var_seq.append(er.get('total_variance', np.nan))

                try:
                    result = compute_thermodynamics(
                        eigenvalues_sequence=eigenvalues_seq,
                        effective_dim_sequence=np.array(eff_dim_seq),
                        total_variance_sequence=np.array(total_var_seq),
                    )
                    result.pop('entropy_series', None)
                    result['cohort'] = cohort
                    result['energy_std'] = float(np.nanstd(total_var_seq))
                    result['n_samples'] = result.pop('n_windows', 0)
                    result.pop('heat_capacity', None)
                    all_thermo_rows.append(result)
                except Exception:
                    pass

            if all_thermo_rows:
                thermo_df = pl.DataFrame(all_thermo_rows)
                thermo_df.write_parquet(
                    output_path / 'cohort' / 'cohort_dynamics' / 'thermodynamics.parquet')
                outputs_created.append('cohort/cohort_dynamics/thermodynamics.parquet')
                if verbose:
                    print(f"    → cohort/cohort_dynamics/thermodynamics.parquet ({len(thermo_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 11: Breaks → cohort/cohort_dynamics/breaks.parquet
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [11/16 breaks] Change-point detection...")

    try:
        from breaks import detect_breaks_cusum

        all_break_rows = []

        for cohort in cohorts:
            for signal_id in signals:
                key = (cohort, signal_id)
                if key not in signal_lookup:
                    continue
                sd = signal_lookup[key]
                values = sd['values']
                s0 = sd['signal_0']

                try:
                    breaks_cfg = pkg_configs.get('breaks', {})
                    result = detect_breaks_cusum(
                        values,
                        threshold_sigma=breaks_cfg.get('threshold_sigma', 2.0),
                        min_segment=breaks_cfg.get('min_segment', 20),
                        config=breaks_cfg,
                    )
                    result.pop('cusum_series', None)

                    if result.get('break_detected', False) and result.get('break_index') is not None:
                        bi = result['break_index']
                        pre = values[:bi]
                        post = values[bi:]
                        pre_mean = float(np.nanmean(pre)) if len(pre) > 0 else np.nan
                        post_mean = float(np.nanmean(post)) if len(post) > 0 else np.nan

                        all_break_rows.append({
                            'signal_id': signal_id,
                            'cohort': cohort,
                            'signal_0': float(s0[min(bi, len(s0) - 1)]),
                            'magnitude': abs(post_mean - pre_mean),
                            'direction': 'up' if post_mean > pre_mean else 'down',
                            'sharpness': result.get('cusum_significance', 0.0),
                            'duration': len(post),
                            'pre_level': pre_mean,
                            'post_level': post_mean,
                            'snr': abs(post_mean - pre_mean) / (float(np.nanstd(values)) + 1e-30),
                        })
                except Exception:
                    pass

        if all_break_rows:
            break_df = pl.DataFrame(all_break_rows)
            break_df.write_parquet(output_path / 'cohort' / 'cohort_dynamics' / 'breaks.parquet')
            outputs_created.append('cohort/cohort_dynamics/breaks.parquet')
            if verbose:
                print(f"    → cohort/cohort_dynamics/breaks.parquet ({len(break_df)} rows, {time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 12: Divergence → cohort/cohort_information_flow.parquet (OPTIONAL)
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [12/16 divergence] Information flow (optional)...")

    try:
        from divergence import compute_granger, compute_transfer_entropy
        from divergence import kl_divergence, js_divergence

        all_div_rows = []

        for cohort in cohorts:
            cohort_signals = [s for s in signals if (cohort, s) in signal_lookup]

            for i in range(len(cohort_signals)):
                for j in range(i + 1, len(cohort_signals)):
                    sig_a = cohort_signals[i]
                    sig_b = cohort_signals[j]

                    x = signal_lookup[(cohort, sig_a)]['values']
                    y = signal_lookup[(cohort, sig_b)]['values']

                    n_samples = min(len(x), len(y))
                    if n_samples < 50:
                        continue

                    x_aligned, y_aligned = x[:n_samples], y[:n_samples]

                    # Use signal_a's typology for lag/bins config
                    div_cfg = get_signal_config(pkg_configs, 'divergence', cohort, sig_a, per_signal_configs)
                    div_max_lag = div_cfg.get('max_lag', 5)
                    div_n_bins = div_cfg.get('n_bins', 8)

                    row: Dict[str, Any] = {
                        'signal_a': sig_a,
                        'signal_b': sig_b,
                        'cohort': cohort,
                        'n_samples': n_samples,
                    }

                    try:
                        g_ab = compute_granger(x_aligned, y_aligned, max_lag=div_max_lag, config=div_cfg)
                        row['granger_f_a_to_b'] = g_ab.get('granger_f', np.nan)
                        row['granger_p_a_to_b'] = g_ab.get('granger_p', np.nan)
                    except Exception:
                        row['granger_f_a_to_b'] = np.nan
                        row['granger_p_a_to_b'] = np.nan

                    try:
                        g_ba = compute_granger(y_aligned, x_aligned, max_lag=div_max_lag, config=div_cfg)
                        row['granger_f_b_to_a'] = g_ba.get('granger_f', np.nan)
                        row['granger_p_b_to_a'] = g_ba.get('granger_p', np.nan)
                    except Exception:
                        row['granger_f_b_to_a'] = np.nan
                        row['granger_p_b_to_a'] = np.nan

                    try:
                        te_ab = compute_transfer_entropy(x_aligned, y_aligned, n_bins=div_n_bins, config=div_cfg)
                        row['transfer_entropy_a_to_b'] = te_ab.get('transfer_entropy', np.nan)
                    except Exception:
                        row['transfer_entropy_a_to_b'] = np.nan

                    try:
                        te_ba = compute_transfer_entropy(y_aligned, x_aligned, n_bins=div_n_bins, config=div_cfg)
                        row['transfer_entropy_b_to_a'] = te_ba.get('transfer_entropy', np.nan)
                    except Exception:
                        row['transfer_entropy_b_to_a'] = np.nan

                    try:
                        row['kl_divergence_a_to_b'] = float(kl_divergence(x_aligned, y_aligned))
                    except Exception:
                        row['kl_divergence_a_to_b'] = np.nan

                    try:
                        row['kl_divergence_b_to_a'] = float(kl_divergence(y_aligned, x_aligned))
                    except Exception:
                        row['kl_divergence_b_to_a'] = np.nan

                    try:
                        row['js_divergence'] = float(js_divergence(x_aligned, y_aligned))
                    except Exception:
                        row['js_divergence'] = np.nan

                    all_div_rows.append(row)

        if all_div_rows:
            div_df = pl.DataFrame(all_div_rows)
            div_df.write_parquet(output_path / 'cohort' / 'cohort_information_flow.parquet')
            outputs_created.append('cohort/cohort_information_flow.parquet')
            if verbose:
                print(f"    → cohort/cohort_information_flow.parquet ({len(div_df)} rows, {time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped (optional): {e}")

    # =====================================================================
    # STAGE 13: Topology (OPTIONAL — requires ripser)
    # =====================================================================
    if signal_vector_df is not None and feature_cols:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [13/16 topology] Persistent homology (optional)...")

        try:
            from topology import compute_persistence

            all_topo_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)
                # Get topology config scaled by n_signals for this cohort
                first_sig = signals[0] if signals else ''
                topo_cfg = get_signal_config(pkg_configs, 'topology', cohort or 'system', first_sig, per_signal_configs)
                topo_max_pts = topo_cfg.get('max_points', 500)

                for win_idx in window_indices:
                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 3:
                        continue

                    matrix = win_data.select(feature_cols).to_numpy()
                    s0_end = float(win_data['signal_0_end'][0])

                    try:
                        result = compute_persistence(matrix, max_points=topo_max_pts, config=topo_cfg)
                        result.pop('persistence_pairs', None)
                        result['cohort'] = cohort
                        result['signal_0_end'] = s0_end
                        result['window_index'] = win_idx
                        all_topo_rows.append(result)
                    except Exception:
                        pass

            if all_topo_rows:
                topo_df = pl.DataFrame(all_topo_rows)
                topo_df.write_parquet(output_path / 'cohort' / 'persistent_homology.parquet')
                outputs_created.append('cohort/persistent_homology.parquet')
                if verbose:
                    print(f"    → cohort/persistent_homology.parquet ({len(topo_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped (optional): {e}")
    elif verbose:
        print(f"\n  [13/16 topology] Skipped: no signal_vector data")

    # =====================================================================
    # STAGE 14: Baseline → parameterization/signal_dominance.parquet
    # =====================================================================
    if signal_vector_df is not None and feature_cols and eigen_store:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [14/16 baseline] Signal dominance ranking...")

        try:
            from baseline import compute_fleet_baseline

            # Build cohort matrices: {cohort_id: (n_windows, n_features)}
            cohort_matrices: Dict[str, Any] = {}
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)

                # Determine columns that are not ALL-NaN across every window
                all_matrix = cohort_sv.select(feature_cols).to_numpy()
                col_ok = ~np.all(np.isnan(all_matrix), axis=0)

                windows_data = []
                for win_idx in window_indices:
                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 2:
                        continue
                    matrix = win_data.select(feature_cols).to_numpy()[:, col_ok]
                    centroid = np.nanmean(matrix, axis=0)
                    windows_data.append(centroid)

                if len(windows_data) >= 2:
                    cohort_matrices[cohort] = np.array(windows_data)

            if cohort_matrices:
                base_cfg = pkg_configs.get('baseline', {})
                baseline = compute_fleet_baseline(
                    cohort_matrices,
                    min_windows=base_cfg.get('min_windows', 2),
                    config=base_cfg,
                )

                if baseline.get('n_cohorts', 0) > 0:
                    # Per-signal PC1 loading from eigendecomp store
                    signal_loadings: Dict[str, list] = {}

                    for (cohort, win_idx), er in eigen_store.items():
                        pcs = er.get('principal_components')
                        if pcs is None or pcs.ndim != 2:
                            continue

                        win_data = signal_vector_df.filter(
                            (pl.col('cohort') == cohort) & (pl.col('window_index') == win_idx)
                        )
                        if len(win_data) == 0:
                            continue

                        sig_ids = win_data['signal_id'].to_list()
                        # Filter to same columns used by eigendecomp, then
                        # further filter to varying columns (what PCs are built from)
                        col_mask = er.get('_col_has_data')
                        varying_mask = er.get('varying_mask')
                        full_matrix = win_data.select(feature_cols).to_numpy()
                        matrix = full_matrix[:, col_mask] if col_mask is not None else full_matrix
                        if varying_mask is not None and len(varying_mask) == matrix.shape[1]:
                            matrix = matrix[:, varying_mask]
                        centroid = np.nanmean(matrix, axis=0)
                        pc1_vec = pcs[0, :matrix.shape[1]]

                        for s_idx, sid in enumerate(sig_ids):
                            if s_idx < len(matrix):
                                centered = matrix[s_idx] - centroid
                                valid = np.isfinite(centered) & np.isfinite(pc1_vec[:len(centered)])
                                if valid.any():
                                    proj = float(np.dot(centered[valid], pc1_vec[valid]))
                                    if sid not in signal_loadings:
                                        signal_loadings[sid] = []
                                    signal_loadings[sid].append(proj)

                    dominance_rows = []
                    for sid, loadings in signal_loadings.items():
                        if loadings:
                            dominance_rows.append({
                                'engine': 'aggregate',
                                'signal_id': sid,
                                'mean_abs_pc1_loading': float(np.mean(np.abs(loadings))),
                                'final_abs_pc1_loading': float(np.abs(loadings[-1])),
                                'pc1_loading_sign': int(np.sign(np.mean(loadings))),
                                'n_windows': len(loadings),
                            })

                    if dominance_rows:
                        dominance_rows.sort(key=lambda x: x['mean_abs_pc1_loading'], reverse=True)
                        for rank, row in enumerate(dominance_rows, 1):
                            row['dominance_rank'] = rank

                        dom_df = pl.DataFrame(dominance_rows)
                        dom_df.write_parquet(
                            output_path / 'parameterization' / 'signal_dominance.parquet')
                        outputs_created.append('parameterization/signal_dominance.parquet')
                        if verbose:
                            print(f"    → parameterization/signal_dominance.parquet ({len(dom_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 15: Fleet → system/*.parquet
    # =====================================================================
    fleet_mode = pkg_configs.get('fleet', {}).get('system_mode', 'auto')
    fleet_skip = fleet_mode == 'skip'
    fleet_force = fleet_mode == 'force'
    fleet_eligible = (n_cohorts > 1 or fleet_force) and not fleet_skip

    if fleet_eligible and signal_vector_df is not None and feature_cols:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [15/16 fleet] System-level analysis (mode={fleet_mode})...")

        try:
            from fleet import compute_fleet_eigendecomp, compute_fleet_pairwise, compute_fleet_velocity

            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            # Per-window cohort centroids for fleet analysis
            system_vector_rows = []
            cohort_centroid_series: Dict[str, list] = {}

            for win_idx in window_indices:
                win_data = signal_vector_df.filter(pl.col('window_index') == win_idx)
                if len(win_data) == 0:
                    continue

                s0_end = float(win_data['signal_0_end'][0])
                s0_start = float(win_data['signal_0_start'][0])
                s0_center = float(win_data['signal_0_center'][0])

                cohort_centroids: Dict[str, Any] = {}
                for cohort in cohorts:
                    cohort_win = win_data.filter(pl.col('cohort') == cohort)
                    if len(cohort_win) < 2:
                        continue
                    full_m = cohort_win.select(feature_cols).to_numpy()
                    col_ok = ~np.all(np.isnan(full_m), axis=0)
                    matrix = full_m[:, col_ok]
                    centroid = np.nanmean(matrix, axis=0)
                    if np.any(np.isfinite(centroid)):
                        cohort_centroids[cohort] = centroid

                        if cohort not in cohort_centroid_series:
                            cohort_centroid_series[cohort] = []
                        cohort_centroid_series[cohort].append(centroid)

                if len(cohort_centroids) < 2:
                    continue

                try:
                    fleet_result = compute_fleet_eigendecomp(cohort_centroids)
                    fleet_centroid = fleet_result.get('fleet_centroid', np.array([]))

                    distances = []
                    for c, v in cohort_centroids.items():
                        if len(fleet_centroid) > 0:
                            distances.append(float(np.linalg.norm(v - fleet_centroid)))

                    system_vector_rows.append({
                        'signal_0_end': s0_end,
                        'signal_0_start': s0_start,
                        'signal_0_center': s0_center,
                        'n_cohorts': len(cohort_centroids),
                        'n_signals': len(cohort_centroids),
                        'mean_distance': float(np.mean(distances)) if distances else np.nan,
                        'max_distance': float(np.max(distances)) if distances else np.nan,
                        'std_distance': float(np.std(distances)) if distances else np.nan,
                    })
                except Exception:
                    pass

            # system_vector.parquet
            if system_vector_rows:
                sys_df = pl.DataFrame(system_vector_rows)
                sys_df.write_parquet(output_path / 'system' / 'system_vector.parquet')
                outputs_created.append('system/system_vector.parquet')
                if verbose:
                    print(f"    → system/system_vector.parquet ({len(sys_df)} rows)")

            # Trajectory signatures per cohort
            traj_sig_rows = []
            for cohort in cohorts:
                cohort_keys = sorted(
                    [k for k in eigen_store if k[0] == cohort], key=lambda x: x[1]
                )
                if len(cohort_keys) < 2:
                    continue

                for ck in cohort_keys:
                    er = eigen_store[ck]
                    win_idx = ck[1]

                    sv_row = signal_vector_df.filter(
                        (pl.col('cohort') == cohort) & (pl.col('window_index') == win_idx)
                    )
                    s0_end = float(sv_row['signal_0_end'][0]) if len(sv_row) > 0 else np.nan

                    eigenvalues = er.get('eigenvalues', np.array([]))

                    traj_sig_rows.append({
                        'cohort': cohort,
                        'signal_0_end': s0_end,
                        'eigenvalue_1': float(eigenvalues[0]) if len(eigenvalues) > 0 else np.nan,
                        'eigenvalue_2': float(eigenvalues[1]) if len(eigenvalues) > 1 else np.nan,
                        'eigenvalue_3': float(eigenvalues[2]) if len(eigenvalues) > 2 else np.nan,
                        'effective_dim': er.get('effective_dim', np.nan),
                        'total_variance': er.get('total_variance', np.nan),
                        'condition_number': er.get('condition_number', np.nan),
                        'effective_dim_velocity': np.nan,
                        'effective_dim_acceleration': np.nan,
                        'effective_dim_curvature': np.nan,
                        'speed': np.nan,
                        'curvature': np.nan,
                        'acceleration_magnitude': np.nan,
                        'torsion': np.nan,
                        'arc_length': np.nan,
                    })

            # Fill dynamics columns from geometry_dynamics
            gd_path = output_path / 'cohort' / 'cohort_dynamics' / 'geometry_dynamics.parquet'
            if gd_path.exists() and traj_sig_rows:
                gd_df = pl.read_parquet(gd_path)
                for row in traj_sig_rows:
                    match = gd_df.filter(
                        (pl.col('cohort') == row['cohort']) &
                        (pl.col('signal_0_end') == row['signal_0_end'])
                    )
                    if len(match) > 0:
                        for col in ['effective_dim_velocity', 'effective_dim_acceleration',
                                    'effective_dim_curvature']:
                            if col in match.columns:
                                row[col] = float(match[col][0])

            # Fill velocity columns from velocity_field
            vf_path = output_path / 'cohort' / 'cohort_dynamics' / 'velocity_field.parquet'
            if vf_path.exists() and traj_sig_rows:
                vf_df = pl.read_parquet(vf_path)
                for row in traj_sig_rows:
                    match = vf_df.filter(
                        (pl.col('cohort') == row['cohort']) &
                        (pl.col('signal_0_end') == row['signal_0_end'])
                    )
                    if len(match) > 0:
                        for col in ['speed', 'curvature', 'acceleration_magnitude']:
                            if col in match.columns:
                                row[col] = float(match[col][0])

            # Arc length: cumulative speed
            for cohort in cohorts:
                cohort_rows = sorted(
                    [r for r in traj_sig_rows if r['cohort'] == cohort],
                    key=lambda x: x['signal_0_end'],
                )
                arc = 0.0
                for r in cohort_rows:
                    r['arc_length'] = arc
                    spd = r.get('speed', np.nan)
                    if np.isfinite(spd):
                        arc += spd

            # trajectory_signatures.parquet
            if traj_sig_rows:
                traj_sig_df = pl.DataFrame(traj_sig_rows)
                traj_sig_df.write_parquet(output_path / 'system' / 'trajectory_signatures.parquet')
                outputs_created.append('system/trajectory_signatures.parquet')
                if verbose:
                    print(f"    → system/trajectory_signatures.parquet ({len(traj_sig_df)} rows)")

            # Trajectory library + match (clustering)
            if traj_sig_rows:
                traj_lib_rows, traj_match_rows = _build_trajectory_library(
                    traj_sig_rows, cohorts
                )

                if traj_lib_rows:
                    pl.DataFrame(traj_lib_rows).write_parquet(
                        output_path / 'system' / 'trajectory_library.parquet')
                    outputs_created.append('system/trajectory_library.parquet')
                    if verbose:
                        print(f"    → system/trajectory_library.parquet ({len(traj_lib_rows)} rows)")

                if traj_match_rows:
                    pl.DataFrame(traj_match_rows).write_parquet(
                        output_path / 'system' / 'trajectory_match.parquet')
                    outputs_created.append('system/trajectory_match.parquet')
                    if verbose:
                        print(f"    → system/trajectory_match.parquet ({len(traj_match_rows)} rows)")

            # System dynamics: velocity and FTLE at system level
            # Filter to cohorts with >= 3 windows (required by compute_fleet_velocity)
            filtered_centroid_series = {
                c: v for c, v in cohort_centroid_series.items() if len(v) >= 3
            }
            if filtered_centroid_series and len(filtered_centroid_series) >= 2:
                try:
                    fleet_vel = compute_fleet_velocity(filtered_centroid_series)
                    if fleet_vel:
                        for fv in fleet_vel:
                            idx = int(fv.get('I', 0))
                            if idx < len(window_indices):
                                sv_row = signal_vector_df.filter(
                                    pl.col('window_index') == window_indices[min(idx, len(window_indices) - 1)]
                                )
                                if len(sv_row) > 0:
                                    fv['signal_0_end'] = float(sv_row['signal_0_end'][0])
                                    fv['signal_0_start'] = float(sv_row['signal_0_start'][0])
                                    fv['signal_0_center'] = float(sv_row['signal_0_center'][0])

                        sys_vel_df = pl.DataFrame(fleet_vel)
                        sys_vel_df.write_parquet(
                            output_path / 'system' / 'system_dynamics' / 'velocity_field.parquet')
                        outputs_created.append('system/system_dynamics/velocity_field.parquet')
                        if verbose:
                            print(f"    → system/system_dynamics/velocity_field.parquet ({len(sys_vel_df)} rows)")
                except Exception:
                    pass

                # System-level FTLE from fleet centroid trajectory
                try:
                    from dynamics import compute_ftle as _sys_ftle

                    n_win = min(len(v) for v in filtered_centroid_series.values())
                    fleet_traj = []
                    for t in range(n_win):
                        vecs = [filtered_centroid_series[c][t] for c in filtered_centroid_series
                                if t < len(filtered_centroid_series[c])]
                        fleet_traj.append(np.mean(vecs, axis=0))
                    fleet_traj_arr = np.array(fleet_traj)

                    if fleet_traj_arr.shape[0] > 0 and fleet_traj_arr.shape[1] > 0:
                        fleet_scalar = np.linalg.norm(fleet_traj_arr, axis=1)

                        if len(fleet_scalar) >= 3:
                            sys_ftle_result = _sys_ftle(fleet_scalar, direction='forward',
                                                        min_samples=min(50, len(fleet_scalar)))
                            sys_ftle_rows = [{
                                'ftle': sys_ftle_result.get('ftle', np.nan),
                                'confidence': sys_ftle_result.get('confidence', 0.0),
                                'n_samples': sys_ftle_result.get('n_samples', 0),
                                'direction': 'forward',
                            }]
                            sys_ftle_df = pl.DataFrame(sys_ftle_rows)
                            sys_ftle_df.write_parquet(
                                output_path / 'system' / 'system_dynamics' / 'ftle.parquet')
                            outputs_created.append('system/system_dynamics/ftle.parquet')
                            if verbose:
                                print(f"    → system/system_dynamics/ftle.parquet")
                except (ImportError, Exception):
                    pass

            if verbose:
                print(f"    ({time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # STAGE 16: Cohort Vector → cohort/cohort_vector.parquet
    # =====================================================================
    if signal_vector_df is not None and feature_cols:
        _stage_t = time.time()
        if verbose:
            print(f"\n  [16/16 cohort_vector] Cohort centroid + dispersion...")

        try:
            from vector.cohort import compute_cohort

            all_cohort_rows = []
            window_indices = sorted(signal_vector_df['window_index'].unique().to_list())

            for cohort in cohorts:
                cohort_sv = signal_vector_df.filter(pl.col('cohort') == cohort)

                for win_idx in window_indices:
                    win_data = cohort_sv.filter(pl.col('window_index') == win_idx)
                    if len(win_data) < 2:
                        continue

                    full_m = win_data.select(feature_cols).to_numpy()
                    col_ok = ~np.all(np.isnan(full_m), axis=0)
                    matrix = full_m[:, col_ok]
                    valid_fnames = [f for f, ok in zip(feature_cols, col_ok) if ok]
                    s0_end = float(win_data['signal_0_end'][0])
                    s0_start = float(win_data['signal_0_start'][0])
                    s0_center = float(win_data['signal_0_center'][0])

                    try:
                        row = compute_cohort(
                            signal_matrix=matrix,
                            cohort_id=cohort,
                            window_index=win_idx,
                            feature_names=valid_fnames,
                        )
                        row['signal_0_end'] = s0_end
                        row['signal_0_start'] = s0_start
                        row['signal_0_center'] = s0_center
                        row['cohort'] = cohort
                        row.pop('cohort_id', None)
                        row.pop('window_index', None)
                        all_cohort_rows.append(row)
                    except Exception:
                        pass

            if all_cohort_rows:
                cohort_df = pl.DataFrame(all_cohort_rows)
                cohort_df.write_parquet(output_path / 'cohort' / 'cohort_vector.parquet')
                outputs_created.append('cohort/cohort_vector.parquet')
                if verbose:
                    print(f"    → cohort/cohort_vector.parquet ({len(cohort_df)} rows, {time.time() - _stage_t:.1f}s)")

        except ImportError as e:
            if verbose:
                print(f"    Skipped: {e}")

    # =====================================================================
    # Typology window-level outputs (bonus — signal/typology_vector.parquet
    # + signal/typology_windows.parquet)
    # =====================================================================
    _stage_t = time.time()
    if verbose:
        print(f"\n  [bonus] Typology window features...")

    try:
        from typology.observe import observe

        all_window_rows = []
        window_id = 0

        for cohort in cohorts:
            for signal_id in signals:
                key = (cohort, signal_id)
                if key not in signal_lookup:
                    continue
                sd = signal_lookup[key]
                values = sd['values']
                s0 = sd['signal_0']

                if len(values) < window_size:
                    continue

                for start in range(0, len(values) - window_size + 1, stride):
                    end = start + window_size
                    window_vals = values[start:end]
                    valid = window_vals[np.isfinite(window_vals)]

                    if len(valid) < window_size // 2:
                        continue

                    try:
                        measures = observe(valid)
                        measures['window_id'] = window_id
                        measures['signal_id'] = signal_id
                        measures['cohort'] = cohort
                        measures['signal_0_start'] = float(s0[start])
                        measures['signal_0_end'] = float(s0[min(end - 1, len(s0) - 1)])
                        measures['signal_0_center'] = (measures['signal_0_start'] + measures['signal_0_end']) / 2.0
                        measures['n_obs'] = len(valid)
                        all_window_rows.append(measures)
                        window_id += 1
                    except Exception:
                        pass

        if all_window_rows:
            # typology_windows.parquet: per-window measures
            windows_df = pl.DataFrame(all_window_rows)
            windows_df.write_parquet(output_path / 'signal' / 'typology_windows.parquet')
            outputs_created.append('signal/typology_windows.parquet')
            if verbose:
                print(f"    → signal/typology_windows.parquet ({len(windows_df)} rows)")

            # typology_vector.parquet: aggregated per signal
            measure_cols = [c for c in windows_df.columns
                          if c not in {'window_id', 'signal_id', 'cohort', 'signal_0_start',
                                       'signal_0_end', 'signal_0_center', 'n_obs'}
                          and windows_df[c].dtype in [pl.Float64, pl.Float32]]

            agg_exprs = [pl.count().alias('n_windows')]
            for mc in measure_cols:
                agg_exprs.append(pl.col(mc).mean().alias(f'{mc}_mean'))
                agg_exprs.append(pl.col(mc).std().alias(f'{mc}_std'))
                mean_val = pl.col(mc).mean()
                agg_exprs.append(
                    (pl.col(mc).std() / (mean_val.abs() + 1e-30)).alias(f'{mc}_cv')
                )
                agg_exprs.append(
                    (pl.col(mc).std() > 1e-10).alias(f'{mc}_varies')
                )

            tv_df = windows_df.group_by(['signal_id', 'cohort']).agg(agg_exprs)
            tv_df.write_parquet(output_path / 'signal' / 'typology_vector.parquet')
            outputs_created.append('signal/typology_vector.parquet')
            if verbose:
                print(f"    → signal/typology_vector.parquet ({len(tv_df)} rows)")

            if verbose:
                print(f"    ({time.time() - _stage_t:.1f}s)")

    except ImportError as e:
        if verbose:
            print(f"    Skipped: {e}")

    # =====================================================================
    # Checksums
    # =====================================================================
    from orchestration.checksums import generate_checksums

    checksums = generate_checksums(output_dir)
    if verbose:
        print(f"\n  Checksum: {checksums['pipeline_checksum'][:16]}...")
        print(f"  Hashed {checksums['n_files']} files ({checksums['total_bytes']:,} bytes)")

    # =====================================================================
    # Done
    # =====================================================================
    elapsed = time.time() - t0

    if verbose:
        print(f"\n  Completed: {len(outputs_created)} outputs in {elapsed:.1f}s")

    return {
        'outputs': outputs_created,
        'elapsed': elapsed,
        'observations_path': observations_path,
        'manifest_path': manifest_path,
        'output_dir': output_dir,
        'checksums': checksums,
    }
