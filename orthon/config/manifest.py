"""
ORTHON Manifest Configuration

ONE file. ALL engines. NO exceptions.

This is the single source of truth for:
- ENGINES list (all engines, always enabled)
- Pydantic models for manifest structure
- Factory functions for creating manifests
- Validation to ensure full compute
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import json
import uuid


# =============================================================================
# ALL ENGINES - This is the source of truth
# =============================================================================

ENGINES = [
    # Tier 1: Basic Statistics
    "mean",
    "std",
    "rms",
    "peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "margin_factor",
    "skewness",
    "kurtosis",

    # Tier 2: Distribution
    "histogram",
    "percentiles",
    "iqr",
    "mad",
    "coefficient_of_variation",

    # Tier 3: Information Theory
    "entropy_shannon",
    "entropy_sample",
    "entropy_permutation",
    "entropy_spectral",
    "mutual_information",
    "transfer_entropy",

    # Tier 4: Spectral
    "fft",
    "psd",
    "spectral_centroid",
    "spectral_spread",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_slope",
    "spectral_entropy",
    "spectral_peaks",
    "harmonic_ratio",
    "bandwidth",

    # Tier 5: Dynamics
    "lyapunov",
    "correlation_dimension",
    "hurst_exponent",
    "dfa",
    "recurrence_rate",
    "determinism",
    "laminarity",
    "trapping_time",
    "divergence",
    "attractor_dimension",

    # Tier 6: Topology
    "betti_0",
    "betti_1",
    "persistence_entropy",
    "persistence_landscape",
    "wasserstein_distance",

    # Tier 7: Relationships
    "cross_correlation",
    "coherence",
    "phase_coupling",
    "granger_causality",
    "cointegration",
    "dtw_distance",
]


# =============================================================================
# DEFAULT CONFIG - ALL engines enabled, NO exceptions
# =============================================================================

DEFAULT_PRISM_CONFIG = {
    "engines": {engine: True for engine in ENGINES},
    "compute": {
        "skip_engines": [],       # EMPTY - never skip
        "skip_metrics": [],       # EMPTY - never skip
        "insufficient_data": "nan",   # Return NaN, don't skip
    },
    "ram": {
        "batch_size": "auto",     # Auto-tune based on available RAM
        "flush_interval": 100,    # Entities before flush
        "clear_cache": True,      # Clear after each batch
        "max_memory_pct": 0.8,    # Use max 80% of available RAM
    }
}


# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class RAMConfig(BaseModel):
    """RAM management configuration."""
    batch_size: int | str = "auto"
    flush_interval: int = 100
    clear_cache: bool = True
    max_memory_pct: float = 0.8


class ComputeConfig(BaseModel):
    """Compute settings - enforces FULL compute."""
    skip_engines: List[str] = Field(default_factory=list)  # Should always be empty
    skip_metrics: List[str] = Field(default_factory=list)  # Should always be empty
    insufficient_data: str = "nan"  # Return NaN, never skip


class WindowConfig(BaseModel):
    """Window/stride configuration for PRISM analysis."""
    size: int = Field(1024, description="Window size in sequence points")
    stride: Optional[int] = Field(None, description="Stride between windows (default: same as size)")
    min_samples: int = Field(50, description="Minimum samples required per window")


class EngineManifestEntry(BaseModel):
    """Specification for a single engine execution."""

    name: str = Field(..., description="Engine name (matches ENGINES)")
    output: str = Field(
        "vector",
        description="Output parquet file: vector, dynamics, geometry, pairs, physics"
    )
    granularity: str = Field(
        "signal",
        description="Output granularity: signal, observation, pair_directional, pair_symmetric"
    )
    groupby: List[str] = Field(default_factory=list)
    orderby: List[str] = Field(default_factory=lambda: ["I"])
    input_columns: List[str] = Field(default_factory=lambda: ["I", "y"])
    output_columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    min_rows: int = Field(10, description="Minimum rows required")
    enabled: bool = Field(True, description="Always True for full compute")


class ManifestMetadata(BaseModel):
    """Metadata about the input data."""

    entity_count: int = Field(0, description="Number of unique entities")
    signal_count: int = Field(0, description="Number of unique signals")
    observation_count: int = Field(0, description="Total observations")
    entities: List[str] = Field(default_factory=list)
    signals: List[str] = Field(default_factory=list)
    units_present: List[str] = Field(default_factory=list)
    sampling_rate: Optional[float] = None
    I_min: Optional[float] = None
    I_max: Optional[float] = None
    y_min: Optional[float] = None
    y_max: Optional[float] = None


class DataConfig(BaseModel):
    """Data path configuration."""
    observations_path: str
    entity_column: str = "entity_id"
    index_column: str = "I"
    signals: List[str] = Field(default_factory=list)


class DatasetConfig(BaseModel):
    """Dataset identification."""
    name: str
    domain: str = "universal"  # No domain-specific anything
    description: Optional[str] = None


class PRISMConfig(BaseModel):
    """PRISM execution configuration - FULL compute."""

    window: WindowConfig = Field(default_factory=WindowConfig)
    engines: Dict[str, bool] = Field(
        default_factory=lambda: {e: True for e in ENGINES}
    )
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    ram: RAMConfig = Field(default_factory=RAMConfig)
    constants: Dict[str, Any] = Field(default_factory=dict)


class Manifest(BaseModel):
    """
    Complete ORTHON/PRISM job manifest.

    Single source of truth. ALL engines. NO exceptions.
    """

    # Identifiers
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    orthon_version: str = "0.1.0"

    # Configuration
    dataset: DatasetConfig
    data: DataConfig
    prism: PRISMConfig = Field(default_factory=PRISMConfig)
    metadata: ManifestMetadata = Field(default_factory=ManifestMetadata)

    # Engine manifest (detailed execution plan)
    engines: List[EngineManifestEntry] = Field(default_factory=list)

    # Optional callback
    callback_url: Optional[str] = None

    def to_json(self, path: Union[str, Path]) -> None:
        """Write manifest to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "Manifest":
        """Load manifest from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Write manifest to YAML file."""
        import yaml
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Manifest":
        """Load manifest from YAML file."""
        import yaml
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)

    def get_enabled_engines(self) -> List[str]:
        """Get list of enabled engine names."""
        return [e for e, enabled in self.prism.engines.items() if enabled]

    def engine_count(self) -> int:
        """Count of enabled engines."""
        return len(self.get_enabled_engines())

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Manifest: {self.dataset.name}",
            f"  Job ID: {self.job_id[:8]}...",
            f"  Entities: {self.metadata.entity_count}",
            f"  Signals: {self.metadata.signal_count}",
            f"  Observations: {self.metadata.observation_count:,}",
            f"  Window: {self.prism.window.size}",
            f"  Engines: {self.engine_count()} (ALL enabled)",
        ]
        return "\n".join(lines)


# Backwards compatibility aliases
PrismManifest = Manifest
WindowManifest = WindowConfig


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

def create_manifest(
    name: str,
    observations_path: str,
    window_size: int = 1024,
    stride: Optional[int] = None,
    signals: Optional[List[str]] = None,
    **kwargs
) -> Manifest:
    """
    Create manifest with FULL compute enabled.

    ALL engines. No exceptions.
    """
    return Manifest(
        dataset=DatasetConfig(name=name),
        data=DataConfig(
            observations_path=observations_path,
            signals=signals or [],
        ),
        prism=PRISMConfig(
            window=WindowConfig(size=window_size, stride=stride),
        ),
    )


def generate_full_manifest(
    dataset_name: str,
    observations_path: str,
    window_size: int = 1024,
    stride: Optional[int] = None,
    signals: Optional[List[str]] = None,
) -> dict:
    """
    Generate a FULL manifest dict with ALL engines enabled.

    Returns dict for compatibility with existing code.
    """
    manifest = create_manifest(
        name=dataset_name,
        observations_path=observations_path,
        window_size=window_size,
        stride=stride,
        signals=signals,
    )
    return manifest.model_dump()


# =============================================================================
# VALIDATION
# =============================================================================

def validate_manifest(manifest: Union[dict, Manifest]) -> tuple[bool, List[str]]:
    """
    Validate manifest ensures FULL compute.

    Returns:
        (is_valid, list of warnings/errors)
    """
    if isinstance(manifest, Manifest):
        manifest = manifest.model_dump()

    issues = []
    prism = manifest.get("prism", {})

    # Check no engines are disabled
    engines = prism.get("engines", {})
    disabled = [e for e, enabled in engines.items() if not enabled]
    if disabled:
        issues.append(f"WARNING: Engines disabled (violates full compute): {disabled}")

    # Check no skip lists
    compute = prism.get("compute", {})
    if compute.get("skip_engines"):
        issues.append(f"ERROR: skip_engines is not empty: {compute['skip_engines']}")
    if compute.get("skip_metrics"):
        issues.append(f"ERROR: skip_metrics is not empty: {compute['skip_metrics']}")

    # Check insufficient_data handling
    if compute.get("insufficient_data") != "nan":
        issues.append(f"WARNING: insufficient_data should be 'nan', got '{compute.get('insufficient_data')}'")

    is_valid = not any(issue.startswith("ERROR") for issue in issues)
    return is_valid, issues


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Engine list
    "ENGINES",
    "DEFAULT_PRISM_CONFIG",

    # Pydantic models
    "Manifest",
    "PrismManifest",  # Alias
    "PRISMConfig",
    "DataConfig",
    "DatasetConfig",
    "WindowConfig",
    "WindowManifest",  # Alias
    "RAMConfig",
    "ComputeConfig",
    "EngineManifestEntry",
    "ManifestMetadata",

    # Factory functions
    "create_manifest",
    "generate_full_manifest",

    # Validation
    "validate_manifest",
]
