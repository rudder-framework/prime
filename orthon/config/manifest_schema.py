"""
ORTHON Manifest Schema

FULL compute. No exceptions. No domain-specific limits.

Rule: ALL engines. ALL metrics. ALL the time.
The only optimization allowed: RAM management.
"""

# ALL engines - always enabled
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

# ALL engines enabled by default - NO EXCEPTIONS
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


def generate_full_manifest(
    dataset_name: str,
    observations_path: str,
    window_size: int = 1024,
    stride: int = None,
    signals: list = None,
) -> dict:
    """
    Generate a FULL manifest with ALL engines enabled.

    No domain-specific configuration.
    No engine gating.
    No metric skipping.

    Args:
        dataset_name: Name of the dataset
        observations_path: Path to observations.parquet
        window_size: Window size for PRISM
        stride: Stride (default: same as window_size)
        signals: List of signal column names

    Returns:
        Full manifest dict with ALL engines enabled
    """

    stride = stride or window_size  # Default: non-overlapping

    manifest = {
        "dataset": {
            "name": dataset_name,
            "domain": "universal",  # No domain-specific anything
        },
        "data": {
            "observations_path": observations_path,
            "entity_column": "entity_id",
            "index_column": "I",
            "signals": signals or [],
        },
        "prism": {
            "window_size": window_size,
            "stride": stride,

            # ALL ENGINES - NO EXCEPTIONS
            "engines": {engine: True for engine in ENGINES},

            # Compute settings
            "compute": {
                "skip_engines": [],
                "skip_metrics": [],
                "insufficient_data": "nan",
            },

            # RAM optimization (the ONLY optimization allowed)
            "ram": {
                "batch_size": "auto",
                "flush_interval": 100,
                "clear_cache": True,
                "max_memory_pct": 0.8,
            }
        }
    }

    return manifest


def validate_manifest(manifest: dict) -> tuple[bool, list[str]]:
    """
    Validate a manifest ensures FULL compute.

    Returns:
        (is_valid, list of warnings/errors)
    """

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
        issues.append(f"WARNING: insufficient_data should be 'nan', not '{compute.get('insufficient_data')}'")

    is_valid = not any(issue.startswith("ERROR") for issue in issues)

    return is_valid, issues
