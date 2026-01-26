"""
PRISM Config Schema — Shared between ORTHON and PRISM

ORTHON writes config.json, PRISM reads it.
This file should be identical in both repos.

Usage (ORTHON - writing):
    config = PrismConfig(
        sequence_column="timestamp",
        entities=["P-101", "P-102"],
        discipline="thermodynamics",
        ...
    )
    config.to_json("config.json")

Usage (PRISM - reading):
    config = PrismConfig.from_json("config.json")
    if config.discipline:
        # Run discipline-specific engines
    print(config.global_constants)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union, Literal
from pathlib import Path
import json


# =============================================================================
# DISCIPLINES (replaces domain for engine routing)
# =============================================================================

# Discipline registry - ORTHON uses for dropdown, PRISM uses for routing
# 12 disciplines total: 7 existing + 5 ChemE additions (~200 engines)
DISCIPLINES = {
    # =========================================================================
    # EXISTING DISCIPLINES (7)
    # =========================================================================
    "thermodynamics": {
        "name": "Thermodynamics",
        "icon": "🔥",
        "description": "Gibbs free energy, enthalpy, entropy, equations of state",
        "engine_count": 7,
        "required_signals": [],
        "required_signals_any": ["temperature", "pressure", "volume"],
        "required_constants": [],
        "optional_constants": [
            "temperature_K", "pressure_Pa", "molar_mass_kg_mol",
            "gas_constant_J_molK", "critical_temperature_K", "critical_pressure_Pa",
            "acentric_factor", "heat_capacity_J_molK",
        ],
        "engines": ["gibbs", "enthalpy", "entropy", "eos", "fugacity", "activity", "helmholtz"],
    },
    "transport": {
        "name": "Transport Phenomena",
        "icon": "🌊",
        "description": "Heat, mass, and momentum transfer",
        "engine_count": 12,
        "required_signals": [],
        "required_signals_any": ["velocity", "flow_rate", "temperature"],
        "required_constants": [],
        "optional_constants": [
            "density_kg_m3", "viscosity_Pa_s", "thermal_conductivity_W_mK",
            "diffusivity_m2_s", "heat_capacity_J_kgK", "pipe_diameter_m",
            "characteristic_length_m",
        ],
        "engines": [
            "reynolds", "nusselt", "prandtl", "schmidt", "peclet", "sherwood",
            "grashof", "rayleigh", "biot", "fourier", "lewis", "stanton"
        ],
    },
    "reaction": {
        "name": "Reaction Engineering",
        "icon": "⚗️",
        "description": "Arrhenius kinetics, reaction rates, yields, selectivity",
        "engine_count": 12,
        "required_signals": [],
        "required_signals_any": ["concentration", "conversion", "temperature"],
        "required_constants": [],
        "optional_constants": [
            # Experiment setup
            "reactor_volume_L", "feed_flow_rate_mL_min", "feed_concentration_mol_L",
            # Kinetics (if known - otherwise PRISM calculates)
            "activation_energy_J_mol", "pre_exponential_factor", "reaction_order",
            # Thermodynamic
            "heat_of_reaction_J_mol", "gas_constant_J_molK",
            # Catalyst
            "catalyst_loading", "catalyst_density_kg_m3",
        ],
        "engines": [
            # Core kinetics
            "conversion", "reaction_rate", "yield", "selectivity",
            # Arrhenius (forward and inverse)
            "arrhenius", "arrhenius_fit",
            # Reactor design
            "residence_time", "cstr_rate_constant", "pfr_rate_constant", "space_time",
            # Dimensionless
            "damkohler", "thiele_modulus", "effectiveness_factor",
        ],
    },
    "controls": {
        "name": "Process Control",
        "icon": "🎛️",
        "description": "Transfer functions, PID tuning, stability analysis",
        "engine_count": 7,
        "required_signals": [],
        "required_signals_any": ["setpoint", "process_variable", "controller_output", "manipulated_variable"],
        "required_constants": [],
        "optional_constants": [
            "Kp", "Ki", "Kd", "deadtime_s", "time_constant_s",
            "process_gain", "sample_time_s",
        ],
        "engines": [
            "transfer_function", "pid_response", "stability", "controllability",
            "observability", "bode", "nyquist"
        ],
    },
    "mechanics": {
        "name": "Mechanical Systems",
        "icon": "⚙️",
        "description": "Vibration analysis, fatigue, stress-strain",
        "engine_count": 5,
        "required_signals": [],
        "required_constants": [],
        "optional_constants": ["youngs_modulus_Pa", "poisson_ratio", "fatigue_limit_Pa"],
        "engines": ["vibration_spectrum", "fatigue_life", "stress_concentration", "modal_analysis", "bearing_fault"],
    },
    "electrical": {
        "name": "Electrical Systems",
        "icon": "⚡",
        "description": "Impedance, battery state, power quality",
        "engine_count": 5,
        "required_signals": [],
        "required_constants": [],
        "optional_constants": ["nominal_voltage_V", "capacity_Ah", "internal_resistance_ohm"],
        "engines": ["impedance", "soc", "soh", "power_factor", "harmonic_distortion"],
    },
    "fluid_dynamics": {
        "name": "Fluid Dynamics (CFD)",
        "icon": "💨",
        "description": "CFD validation, vorticity, Navier-Stokes",
        "engine_count": 6,
        "required_signals": [],
        "required_signals_any": ["velocity", "pressure", "velocity_x", "velocity_y", "velocity_z"],
        "required_constants": [],
        "optional_constants": [
            "density_kg_m3", "viscosity_Pa_s", "kinematic_viscosity_m2_s",
            "characteristic_length_m", "reference_velocity_m_s",
        ],
        "engines": ["vorticity", "divergence", "q_criterion", "lambda2", "tke", "energy_spectrum"],
    },

    # =========================================================================
    # NEW CHEME DISCIPLINES (5)
    # =========================================================================
    "separations": {
        "name": "Separations",
        "icon": "🧪",
        "description": "Distillation, absorption, extraction, membranes",
        "engine_count": 21,
        "required_signals": [],
        "required_signals_any": ["temperature", "composition", "vapor_fraction"],
        "required_constants": [],
        "optional_constants": [
            "relative_volatility", "feed_quality", "reflux_ratio",
            "tray_efficiency", "hetp_m", "membrane_area_m2"
        ],
        "engines": [
            # Distillation
            "mccabe_thiele", "fenske", "underwood", "gilliland", "kirkbride",
            # Absorption
            "ntu", "htu", "kremser", "colburn",
            # Extraction
            "extraction_stages", "partition_coefficient", "extract_raffinate",
            # Membranes
            "membrane_flux", "rejection", "concentration_polarization",
            # General
            "stage_efficiency", "flooding_velocity", "pressure_drop"
        ],
        "subdisciplines": ["distillation", "absorption", "extraction", "membranes"],
    },
    "phase_equilibria": {
        "name": "Phase Equilibria",
        "icon": "⚖️",
        "description": "VLE, LLE, flash calculations, activity models",
        "engine_count": 28,
        "required_signals": [],
        "required_signals_any": ["temperature", "pressure", "composition"],
        "required_constants": [],
        "optional_constants": [
            "antoine_A", "antoine_B", "antoine_C",
            "wilson_lambda", "nrtl_alpha", "nrtl_tau",
            "unifac_r", "unifac_q"
        ],
        "engines": [
            # VLE
            "antoine", "raoults_law", "modified_raoult", "bubble_point", "dew_point",
            # Flash
            "rachford_rice", "isothermal_flash", "adiabatic_flash", "tp_flash",
            # Activity models
            "wilson", "nrtl", "uniquac", "unifac", "margules", "van_laar",
            # EOS
            "peng_robinson", "srk", "van_der_waals", "virial",
            # LLE
            "lle_tie_line", "plait_point", "binodal",
            # General
            "fugacity_coefficient", "activity_coefficient", "excess_gibbs", "henry_law"
        ],
        "subdisciplines": ["vle", "lle", "flash", "activity_models", "eos"],
    },
    "balances": {
        "name": "Material & Energy Balances",
        "icon": "⚖️",
        "description": "Material balance, energy balance, recycle/purge",
        "engine_count": 22,
        "required_signals": [],
        "required_signals_any": ["flow_rate", "temperature", "concentration", "mass_flow", "composition"],
        "required_constants": [],
        "optional_constants": [
            # Energy balance
            "heat_capacity_J_kgK", "heat_capacity_J_molK", "heat_of_reaction_J_mol",
            "heat_transfer_coeff_W_m2K", "heat_transfer_area_m2",
            # Material balance
            "feed_flow_rate", "molecular_weight_kg_mol", "density_kg_m3",
            # Recycle/purge
            "purge_fraction", "recycle_ratio", "makeup_rate",
        ],
        "engines": [
            # Material balance
            "overall_mass_balance", "component_balance", "atom_balance",
            "extent_of_reaction", "limiting_reagent", "excess_reagent",
            # Energy balance
            "sensible_heat", "latent_heat", "heat_of_reaction",
            "adiabatic_temperature", "heat_duty",
            # Heat exchange
            "lmtd", "effectiveness_ntu", "heat_exchanger_design",
            # Recycle/purge
            "recycle_ratio", "purge_analysis", "makeup_rate",
            "steady_state_composition", "accumulation",
            # General
            "degree_of_freedom", "independence_check"
        ],
        "subdisciplines": ["material_balance", "energy_balance", "heat_exchange", "recycle_purge"],
    },
    "electrochemistry": {
        "name": "Electrochemistry",
        "icon": "🔋",
        "description": "Nernst, Butler-Volmer, Faraday, batteries, corrosion",
        "engine_count": 27,
        "required_signals": [],
        "required_signals_any": ["voltage", "current", "potential"],
        "required_constants": [],
        "optional_constants": [
            "exchange_current_density_A_m2", "transfer_coefficient",
            "faraday_constant", "standard_potential_V",
            "capacity_Ah", "internal_resistance_ohm"
        ],
        "engines": [
            # Fundamentals
            "nernst", "butler_volmer", "tafel", "faraday", "overpotential",
            # Kinetics
            "exchange_current", "mass_transfer_limited", "mixed_control",
            "diffusion_layer", "rotating_disk",
            # Batteries
            "soc_coulomb", "soh_capacity", "capacity_fade", "cycle_life",
            "internal_resistance", "open_circuit_voltage", "peukert",
            # Corrosion
            "corrosion_rate", "polarization_resistance", "icorr",
            "passivation", "pitting_potential",
            # Fuel cells
            "fuel_cell_polarization", "ohmic_loss", "activation_loss",
            "concentration_loss"
        ],
        "subdisciplines": ["fundamentals", "kinetics", "batteries", "corrosion", "fuel_cells"],
    },
}

DisciplineType = Optional[Literal[
    # Existing
    "thermodynamics",
    "transport",
    "reaction",
    "controls",
    "mechanics",
    "electrical",
    "fluid_dynamics",
    # New ChemE
    "separations",
    "phase_equilibria",
    "balances",
    "electrochemistry",
]]


# =============================================================================
# DOMAINS (legacy, maps to disciplines)
# =============================================================================

# Domain → discipline mapping for backwards compatibility
# Matches PRISM's legacy domain mapping
DOMAIN_TO_DISCIPLINE = {
    "turbomachinery": "mechanics",
    "battery": "electrochemistry",
    "bearing": "mechanics",
    "fluid": "fluid_dynamics",
    "chemical": "reaction",
}

DomainType = Optional[Literal[
    "turbomachinery",
    "fluid",
    "battery",
    "bearing",
    "chemical",
]]


# =============================================================================
# SUB-MODELS
# =============================================================================

class SignalInfo(BaseModel):
    """Metadata for a single signal"""
    column: str = Field(..., description="Original column name in source data")
    signal_id: str = Field(..., description="Normalized signal identifier")
    unit: Optional[str] = Field(None, description="Unit string (e.g., 'psi', 'gpm', '°F')")


class WindowConfig(BaseModel):
    """Window/stride configuration for PRISM analysis"""
    size: int = Field(50, description="Window size in sequence points")
    stride: int = Field(25, description="Stride between windows")
    min_samples: int = Field(50, description="Minimum samples required per window")


class BaselineConfig(BaseModel):
    """Baseline configuration for distance calculations"""
    fraction: float = Field(0.1, description="Fraction of data to use as baseline (0-1)")
    windows: Optional[int] = Field(None, description="Explicit number of baseline windows (overrides fraction)")


class RegimeConfig(BaseModel):
    """Regime detection configuration"""
    n_regimes: int = Field(3, description="Number of regimes for HMM")
    features: List[str] = Field(
        default=["baseline_distance", "hd_slope"],
        description="Features to use for regime detection"
    )


class StateConfig(BaseModel):
    """State space analysis configuration"""
    n_basins: int = Field(2, description="Number of basins for clustering")
    cycle_period: Optional[float] = Field(None, description="Known cycle period (None = discover via Takens)")
    embedding_delay: Optional[int] = Field(None, description="Embedding delay (None = auto via MI)")
    embedding_dim: Optional[int] = Field(None, description="Embedding dimension (None = auto via FNN)")


# =============================================================================
# MAIN CONFIG
# =============================================================================

class PrismConfig(BaseModel):
    """
    Configuration contract between ORTHON and PRISM.

    ORTHON produces this from user data.
    PRISM consumes this to run analysis.
    """

    # ==========================================================================
    # METADATA
    # ==========================================================================

    source_file: str = Field(
        default="",
        description="Original source file path"
    )
    created_at: str = Field(
        default="",
        description="ISO timestamp when config was created"
    )
    orthon_version: str = Field(
        default="0.1.0",
        description="ORTHON version that created this config"
    )

    # ==========================================================================
    # DISCIPLINE (primary routing)
    # ==========================================================================

    discipline: DisciplineType = Field(
        default=None,
        description="Discipline for physics engines. None = core engines only."
    )

    # ==========================================================================
    # DOMAIN (legacy, for backwards compatibility)
    # ==========================================================================

    domain: DomainType = Field(
        default=None,
        description="Legacy domain field. Use discipline instead."
    )

    # ==========================================================================
    # SEQUENCE (X-AXIS)
    # ==========================================================================

    sequence_column: Optional[str] = Field(
        default=None,
        description="Column used as x-axis (time, depth, cycle, etc.). None = row index."
    )
    sequence_unit: Optional[str] = Field(
        default=None,
        description="Unit of sequence column (e.g., 's', 'm', 'ft', 'cycle')"
    )
    sequence_name: str = Field(
        default="index",
        description="Semantic name: 'time', 'depth', 'cycle', 'distance', or 'index'"
    )

    # ==========================================================================
    # ENTITIES
    # ==========================================================================

    entity_column: Optional[str] = Field(
        default=None,
        description="Column used for entity grouping. None = single entity."
    )
    entities: List[str] = Field(
        default=["default"],
        description="List of unique entity identifiers"
    )

    # ==========================================================================
    # CONSTANTS
    # ==========================================================================

    global_constants: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constants that apply to all entities (e.g., fluid_density)"
    )
    per_entity_constants: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Constants that vary by entity (e.g., pipe diameter)"
    )

    # ==========================================================================
    # SIGNALS
    # ==========================================================================

    signals: List[SignalInfo] = Field(
        default_factory=list,
        description="List of signals detected in data"
    )

    # ==========================================================================
    # ANALYSIS CONFIG
    # ==========================================================================

    window: WindowConfig = Field(
        default_factory=WindowConfig,
        description="Window/stride configuration"
    )

    baseline: BaselineConfig = Field(
        default_factory=BaselineConfig,
        description="Baseline configuration for distance calculations"
    )

    regime: RegimeConfig = Field(
        default_factory=RegimeConfig,
        description="Regime detection configuration"
    )

    state: StateConfig = Field(
        default_factory=StateConfig,
        description="State space analysis configuration"
    )

    # ==========================================================================
    # STATS
    # ==========================================================================

    row_count: int = Field(
        default=0,
        description="Number of rows in source data"
    )
    observation_count: int = Field(
        default=0,
        description="Number of observations in observations.parquet"
    )

    # ==========================================================================
    # METHODS
    # ==========================================================================

    def to_json(self, path: Union[str, Path]) -> None:
        """Write config to JSON file"""
        path = Path(path)
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "PrismConfig":
        """Load config from JSON file"""
        path = Path(path)
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.model_validate(data)

    def get_constant(self, name: str, entity: Optional[str] = None) -> Optional[Any]:
        """
        Get a constant value, checking per-entity first, then global.

        Args:
            name: Constant name
            entity: Entity ID (optional, for per-entity lookup)

        Returns:
            Constant value or None
        """
        # Check per-entity first
        if entity and entity in self.per_entity_constants:
            if name in self.per_entity_constants[entity]:
                return self.per_entity_constants[entity][name]

        # Fall back to global
        return self.global_constants.get(name)

    def get_signal_unit(self, signal_id: str) -> Optional[str]:
        """Get unit for a signal by signal_id"""
        for sig in self.signals:
            if sig.signal_id == signal_id:
                return sig.unit
        return None

    def signal_ids(self) -> List[str]:
        """Get list of all signal IDs"""
        return [s.signal_id for s in self.signals]

    def get_discipline_info(self) -> Optional[Dict[str, Any]]:
        """Get discipline metadata if discipline is specified"""
        if self.discipline and self.discipline in DISCIPLINES:
            return DISCIPLINES[self.discipline]
        return None

    def get_discipline_engines(self) -> List[str]:
        """Get list of discipline-specific engines to run"""
        info = self.get_discipline_info()
        return info["engines"] if info else []

    def validate_discipline_requirements(self) -> Dict[str, List[str]]:
        """
        Check if discipline requirements are met.

        Returns:
            Dict with 'missing_signals' and 'missing_constants' lists
        """
        result = {"missing_signals": [], "missing_constants": []}

        info = self.get_discipline_info()
        if not info:
            return result

        # Check required signals
        current_signals = set(self.signal_ids())
        for req_signal in info.get("required_signals", []):
            if req_signal not in current_signals:
                result["missing_signals"].append(req_signal)

        # Check required constants
        current_constants = set(self.global_constants.keys())
        for req_const in info.get("required_constants", []):
            if req_const not in current_constants:
                result["missing_constants"].append(req_const)

        return result

    def can_run_discipline(self) -> bool:
        """Check if all discipline requirements are met"""
        reqs = self.validate_discipline_requirements()
        return len(reqs["missing_signals"]) == 0 and len(reqs["missing_constants"]) == 0

    def summary(self) -> str:
        """Human-readable summary"""
        lines = [
            "PrismConfig Summary",
            "=" * 40,
            f"Source: {self.source_file}",
            f"Discipline: {self.discipline or '(core only)'}",
            f"Sequence: {self.sequence_column or '(row index)'} [{self.sequence_unit or 'none'}]",
            f"Entities: {len(self.entities)} ({', '.join(self.entities[:3])}{'...' if len(self.entities) > 3 else ''})",
            f"Signals: {len(self.signals)}",
        ]

        for sig in self.signals[:5]:
            lines.append(f"  - {sig.signal_id} [{sig.unit or '?'}]")
        if len(self.signals) > 5:
            lines.append(f"  ... and {len(self.signals) - 5} more")

        if self.global_constants:
            lines.append(f"Global constants: {len(self.global_constants)}")
            for k, v in list(self.global_constants.items())[:3]:
                lines.append(f"  - {k}: {v}")

        lines.append(f"Window: size={self.window.size}, stride={self.window.stride}")
        lines.append(f"Baseline: {self.baseline.fraction*100:.0f}% of data")

        # Discipline requirements
        if self.discipline:
            reqs = self.validate_discipline_requirements()
            if reqs["missing_signals"] or reqs["missing_constants"]:
                lines.append("⚠️  Missing requirements:")
                for s in reqs["missing_signals"]:
                    lines.append(f"  - signal: {s}")
                for c in reqs["missing_constants"]:
                    lines.append(f"  - constant: {c}")
            else:
                lines.append("✓ All discipline requirements met")

        return "\n".join(lines)


# =============================================================================
# OBSERVATIONS SCHEMA (for reference)
# =============================================================================

"""
CANONICAL observations.parquet schema:

┌───────────┬─────────┬──────────┬───────────────────────────────────────────────┐
│  Column   │  Type   │ Required │                  Description                  │
├───────────┼─────────┼──────────┼───────────────────────────────────────────────┤
│ entity_id │ Utf8    │ Yes      │ Entity identifier                             │
│ signal_id │ Utf8    │ Yes      │ Signal identifier (matches config.signals)    │
│ I         │ Float64 │ Yes      │ Index (time, cycle, depth, distance, sample)  │
│ y         │ Float64 │ Yes      │ Value (the measurement)                       │
│ unit      │ Utf8    │ No       │ Unit string (denormalized for convenience)    │
└───────────┴─────────┴──────────┴───────────────────────────────────────────────┘

I means I. y means y. No aliases. No mapping after intake.
Column mapping happens at INTAKE, not downstream.

The (entity_id, signal_id, I) tuple should be unique.
"""


# =============================================================================
# PRISM OUTPUT SCHEMAS (for reference)
# =============================================================================

"""
7 Output Parquets from PRISM:

1. vector.parquet    - Index: entity + signal + window
2. geometry.parquet  - Index: entity + window
3. dynamics.parquet  - Index: entity + window
4. state.parquet     - Index: entity + window
5. physics.parquet   - Index: entity + window (discipline calculations)
6. fields.parquet    - Index: entity + coords + window (spatial)
7. systems.parquet   - Index: window (fleet metrics)

Key insight: geometry, dynamics, state, systems are NOT indexed by signal.
SQL joins at query time.
"""


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'PrismConfig',
    'SignalInfo',
    'WindowConfig',
    'BaselineConfig',
    'RegimeConfig',
    'StateConfig',
    'DISCIPLINES',
    'DisciplineType',
    'DOMAIN_TO_DISCIPLINE',
    'DomainType',
]
