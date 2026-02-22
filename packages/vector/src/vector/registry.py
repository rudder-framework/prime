"""
Engine Registry
===============
Auto-discovers engines from YAML configs in engine_configs/.
Each YAML declares: window requirements, output column names, metadata.
Each engine has a matching .py file in engines/ with a compute() function.

Usage:
    from vector.registry import Registry
    reg = Registry()
    func = reg.get_compute('statistics')
    result = func(window_data)
    # → {'statistics_kurtosis': 3.2, 'statistics_skewness': 0.1, ...}
"""

import importlib
import yaml
from pathlib import Path
from typing import Dict, Any, Callable, List, Optional
from dataclasses import dataclass


@dataclass
class EngineSpec:
    """Engine specification from YAML config."""
    name: str
    version: str
    base_window: int
    min_window: int
    max_window: int
    scaling: str
    outputs: List[str]
    dependencies: List[str]
    category: str
    description: str


class Registry:
    """
    Engine registry. Discovers engines from YAML configs.
    Lazily imports compute functions on first use.
    """

    def __init__(self):
        self._specs: Dict[str, EngineSpec] = {}
        self._compute_cache: Dict[str, Callable] = {}
        self._discover()

    def _discover(self):
        """Scan engine_configs/ for YAML files."""
        config_dir = Path(__file__).parent / 'engine_configs'
        if not config_dir.exists():
            return
        for path in sorted(config_dir.glob('*.yaml')):
            name = path.stem
            with open(path) as f:
                cfg = yaml.safe_load(f)
            req = cfg.get('requirements', {})
            meta = cfg.get('metadata', {})
            self._specs[name] = EngineSpec(
                name=name,
                version=cfg.get('version', '1.0'),
                base_window=req.get('base_window', 64),
                min_window=req.get('min_window', 16),
                max_window=req.get('max_window', 512),
                scaling=req.get('scaling', 'linear'),
                outputs=cfg.get('outputs', []),
                dependencies=cfg.get('dependencies', []),
                category=meta.get('category', 'unknown'),
                description=meta.get('description', ''),
            )

    @property
    def engine_names(self) -> List[str]:
        """All discovered engine names."""
        return list(self._specs.keys())

    def get_spec(self, name: str) -> EngineSpec:
        """Get engine specification."""
        if name not in self._specs:
            raise KeyError(f"Unknown engine: {name}. Available: {self.engine_names}")
        return self._specs[name]

    def get_compute(self, name: str) -> Callable:
        """
        Get compute function for an engine. Lazily imported.
        Returns a function: compute(np.ndarray) → Dict[str, float]
        """
        if name in self._compute_cache:
            return self._compute_cache[name]

        if name not in self._specs:
            raise KeyError(f"Unknown engine: {name}")

        module = importlib.import_module(f'vector.engines.{name}')
        func = getattr(module, 'compute')
        self._compute_cache[name] = func
        return func

    def get_window(self, name: str, window_factor: float = 1.0) -> int:
        """
        Get window size for an engine, scaled by typology factor.
        Clamped to [min_window, max_window].
        """
        spec = self.get_spec(name)
        window = int(spec.base_window * window_factor)
        return max(spec.min_window, min(window, spec.max_window))

    def get_outputs(self, name: str) -> List[str]:
        """Get declared output column names for an engine."""
        return self.get_spec(name).outputs

    def group_by_window(
        self,
        engine_names: Optional[List[str]] = None,
        window_factor: float = 1.0,
    ) -> Dict[int, List[str]]:
        """
        Group engines by their effective window size.
        Returns {window_size: [engine_names]}.
        """
        if engine_names is None:
            engine_names = self.engine_names

        groups: Dict[int, List[str]] = {}
        for name in engine_names:
            w = self.get_window(name, window_factor)
            groups.setdefault(w, []).append(name)
        return groups

    def validate_outputs(self, name: str, result: Dict[str, Any]) -> bool:
        """Check that engine returned its declared outputs."""
        expected = set(self.get_outputs(name))
        actual = set(result.keys())
        return expected.issubset(actual)


# Module-level singleton
_registry: Optional[Registry] = None


def get_registry() -> Registry:
    """Get or create the global engine registry."""
    global _registry
    if _registry is None:
        _registry = Registry()
    return _registry
