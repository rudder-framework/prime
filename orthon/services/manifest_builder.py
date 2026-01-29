"""
Manifest Builder
================

Converts Orthon config to PRISM manifest format.
Groups engines by execution type: signal, pair, symmetric_pair, windowed, sql
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class PrismManifest:
    """Simple PRISM manifest with grouped engines."""

    engines: Dict[str, List[str]] = field(default_factory=lambda: {
        "signal": [],
        "pair": [],
        "symmetric_pair": [],
        "windowed": [],
        "sql": []
    })
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Optional metadata
    job_id: Optional[str] = None
    input_file: Optional[str] = None
    output_dir: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "engines": self.engines,
            "params": self.params
        }
        if self.job_id:
            result["job_id"] = self.job_id
        if self.input_file:
            result["input_file"] = self.input_file
        if self.output_dir:
            result["output_dir"] = self.output_dir
        return result

    def to_json(self, path: str) -> None:
        """Write manifest to JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def dedupe(self) -> "PrismManifest":
        """Remove duplicate engines from each category."""
        for key in self.engines:
            self.engines[key] = list(set(self.engines[key]))
        return self


def config_to_manifest(config: dict) -> dict:
    """
    Convert Orthon config to PRISM manifest.

    Args:
        config: Orthon config with signals, layers, engines, windows

    Returns:
        PRISM manifest with grouped engines and params
    """

    manifest = PrismManifest()

    # ==========================================================================
    # Map core engines
    # ==========================================================================
    core_engines = config.get("engines", {}).get("core", [])
    for eng in core_engines:
        if eng in ["hurst", "entropy", "lyapunov"]:
            manifest.engines["signal"].append(eng)
        if eng == "trend":
            manifest.engines["signal"].append("rate_of_change")
        if eng == "stationarity":
            manifest.engines["signal"].append("garch")

    # ==========================================================================
    # Map physical_quantity to engines
    # ==========================================================================
    for signal in config.get("signals", []):
        qty = signal.get("physical_quantity")
        unit = signal.get("unit")

        if qty == "pressure":
            manifest.engines["signal"].extend(["pulsation_index", "peak"])
            manifest.engines["windowed"].extend(["rolling_range", "rolling_pulsation"])

        elif qty == "temperature":
            manifest.engines["signal"].extend(["rate_of_change", "time_constant"])

        elif qty == "current":
            manifest.engines["signal"].extend(["harmonics", "rms"])
            manifest.params["harmonics"] = {"sample_rate": 0.1}  # 10-sec intervals

        elif qty == "vibration":
            manifest.engines["signal"].extend(["fft", "envelope", "crest_factor"])
            manifest.engines["windowed"].extend(["rolling_rms", "rolling_peak"])

        elif qty == "flow":
            manifest.engines["signal"].extend(["pulsation_index"])
            manifest.engines["windowed"].extend(["rolling_mean", "rolling_range"])

        elif qty == "rotation":
            manifest.engines["signal"].extend(["harmonics", "order_analysis"])

    # ==========================================================================
    # Map layers to engine groups
    # ==========================================================================
    layers = config.get("layers", {})

    if layers.get("typology"):
        manifest.engines["signal"].extend(["kurtosis", "skewness", "spectral"])

    if layers.get("geometry"):
        manifest.engines["symmetric_pair"].extend(["correlation", "mutual_info", "cointegration"])

    if layers.get("dynamics"):
        manifest.engines["pair"].extend(["granger", "transfer_entropy"])
        manifest.engines["windowed"].extend(["derivatives", "stability"])

    if layers.get("mechanics"):
        manifest.engines["windowed"].extend(["manifold"])

    # ==========================================================================
    # Always add basic analytics
    # ==========================================================================
    manifest.engines["windowed"].extend(["rolling_mean", "rolling_std", "rolling_rms"])
    manifest.engines["sql"].extend(["zscore", "statistics"])

    # ==========================================================================
    # Window params
    # ==========================================================================
    windows = config.get("windows", {})
    window_size = windows.get("size", 100)
    stride = windows.get("stride", window_size // 2)

    for eng in manifest.engines["windowed"]:
        if eng.startswith("rolling_"):
            manifest.params[eng] = {"window": window_size}

    # Store window config
    manifest.params["_window"] = {"size": window_size, "stride": stride}

    # ==========================================================================
    # Dedupe and return
    # ==========================================================================
    manifest.dedupe()

    return manifest.to_dict()


def build_manifest_from_data(
    signals: List[dict],
    window_size: int = 100,
    stride: Optional[int] = None,
    layers: Optional[dict] = None,
    core_engines: Optional[List[str]] = None
) -> dict:
    """
    Build manifest directly from signal definitions.

    Args:
        signals: List of signal dicts with 'name', 'physical_quantity', 'unit'
        window_size: Window size for rolling calculations
        stride: Stride between windows (default: window_size // 2)
        layers: Dict of layer flags (typology, geometry, dynamics, mechanics)
        core_engines: List of core engines to enable

    Returns:
        PRISM manifest dict
    """
    config = {
        "signals": signals,
        "windows": {
            "size": window_size,
            "stride": stride or window_size // 2
        },
        "layers": layers or {
            "typology": True,
            "geometry": True,
            "dynamics": True,
            "mechanics": False
        },
        "engines": {
            "core": core_engines or ["hurst", "entropy", "lyapunov", "trend", "stationarity"]
        }
    }

    return config_to_manifest(config)


# =============================================================================
# Unit-based engine selection
# =============================================================================

# Maps physical quantities to recommended engines
QUANTITY_TO_ENGINES = {
    "pressure": {
        "signal": ["pulsation_index", "peak", "rate_of_change"],
        "windowed": ["rolling_range", "rolling_pulsation", "rolling_mean"]
    },
    "temperature": {
        "signal": ["rate_of_change", "time_constant", "trend"],
        "windowed": ["rolling_mean", "rolling_std"]
    },
    "current": {
        "signal": ["harmonics", "rms", "thd"],
        "windowed": ["rolling_rms"]
    },
    "voltage": {
        "signal": ["harmonics", "rms", "thd"],
        "windowed": ["rolling_rms"]
    },
    "vibration": {
        "signal": ["fft", "envelope", "crest_factor", "kurtosis"],
        "windowed": ["rolling_rms", "rolling_peak", "rolling_crest"]
    },
    "flow": {
        "signal": ["pulsation_index", "rate_of_change"],
        "windowed": ["rolling_mean", "rolling_range"]
    },
    "rotation": {
        "signal": ["harmonics", "order_analysis"],
        "windowed": ["rolling_mean", "rolling_std"]
    },
    "force": {
        "signal": ["peak", "rms"],
        "windowed": ["rolling_peak", "rolling_rms"]
    },
    "torque": {
        "signal": ["peak", "rms", "rate_of_change"],
        "windowed": ["rolling_peak", "rolling_rms"]
    }
}


def build_manifest_from_units(
    unit_categories: List[str],
    window_size: int = 100,
    stride: Optional[int] = None,
    include_universal: bool = True,
    include_causality: bool = True
) -> dict:
    """
    Build manifest directly from detected unit categories.

    Args:
        unit_categories: List of categories (pressure, temperature, vibration, etc.)
        window_size: Window size for rolling calculations
        stride: Stride between windows
        include_universal: Include universal engines (hurst, entropy, etc.)
        include_causality: Include causality engines (granger, transfer_entropy)

    Returns:
        PRISM manifest dict
    """
    manifest = PrismManifest()

    # Add engines based on detected categories
    for category in unit_categories:
        if category in QUANTITY_TO_ENGINES:
            engines = QUANTITY_TO_ENGINES[category]
            manifest.engines["signal"].extend(engines.get("signal", []))
            manifest.engines["windowed"].extend(engines.get("windowed", []))

    # Universal engines (always useful)
    if include_universal:
        manifest.engines["signal"].extend([
            "hurst", "entropy", "lyapunov", "garch",
            "kurtosis", "skewness", "spectral"
        ])

    # Causality engines (for multi-signal analysis)
    if include_causality:
        manifest.engines["pair"].extend(["granger", "transfer_entropy"])
        manifest.engines["symmetric_pair"].extend(["correlation", "mutual_info", "cointegration"])

    # Basic windowed analytics
    manifest.engines["windowed"].extend(["rolling_mean", "rolling_std", "rolling_rms"])

    # SQL-based analytics
    manifest.engines["sql"].extend(["zscore", "statistics"])

    # Set window params
    stride = stride or window_size // 2
    manifest.params["_window"] = {"size": window_size, "stride": stride}

    for eng in manifest.engines["windowed"]:
        if eng.startswith("rolling_"):
            manifest.params[eng] = {"window": window_size}

    # Dedupe and return
    manifest.dedupe()

    return manifest.to_dict()


__all__ = [
    'PrismManifest',
    'config_to_manifest',
    'build_manifest_from_data',
    'build_manifest_from_units',
    'QUANTITY_TO_ENGINES'
]
