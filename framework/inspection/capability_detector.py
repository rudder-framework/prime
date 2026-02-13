"""
Capability Detector
===================

Determine what engines/stages can be computed given the inspected file.

Uses the FileInspection from the gatekeeper to route to appropriate PRISM engines.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict, Set, Optional
from .file_inspector import FileInspection, ConstantInfo


@dataclass
class Capabilities:
    """What can be computed given this data"""

    # Capability level
    level: int  # 0-4
    level_name: str

    # Available stages
    can_vector: bool
    can_geometry: bool
    can_dynamics: bool
    can_mechanics: bool

    # Available engines
    available_engines: List[str]
    unavailable_engines: Dict[str, str]  # engine -> reason

    # Physical quantities detected
    quantities: List[str]

    # Constants available (with values for UI display)
    constants_available: Dict[str, float]

    # What's missing for more capability
    missing_for_next_level: List[str]

    # Recommended discipline
    recommended_discipline: str

    # Summary
    summary: str

    def to_dict(self):
        return asdict(self)


# =============================================================================
# ENGINE REQUIREMENTS
# =============================================================================

# Engine requirements by category
# Each requirement can be:
#   - 'quantity': need a signal of that physical quantity
#   - 'constant': need that constant
#   - 'min_signals': need at least N signals
#   - 'min_entities': need at least N entities
#   - 'min_rows': need at least N rows

ENGINE_REQUIREMENTS = {
    # =========================
    # CORE (always available)
    # =========================
    'statistics': {},
    'trend': {},
    'stationarity': {},
    'entropy': {},
    'hurst': {},
    'spectral': {'min_rows': 32},
    'lyapunov': {'min_rows': 100},
    'recurrence': {'min_rows': 50},

    # =========================
    # GEOMETRY (multi-signal)
    # =========================
    'correlation': {'min_signals': 2},
    'coherence': {'min_signals': 2, 'min_rows': 64},
    'mutual_information': {'min_signals': 2},
    'pca': {'min_signals': 3},
    'clustering': {'min_signals': 2, 'min_rows': 100},

    # =========================
    # DYNAMICS (phase space)
    # =========================
    'phase_space': {'min_rows': 100},
    'attractor': {'min_rows': 200},
    'bifurcation': {'min_rows': 500, 'min_entities': 3},

    # =========================
    # FLUID / TRANSPORT
    # =========================
    'reynolds': {
        'quantity_any': ['velocity', 'volumetric_flow'],
        'constant_any': ['density', 'rho'],
        'constant_any2': ['dynamic_viscosity', 'viscosity', 'mu'],
        'constant_any3': ['diameter', 'length'],
    },
    'pressure_drop': {
        'quantity_any': ['velocity', 'volumetric_flow'],
        'quantity_any2': ['pressure'],
        'constant_any': ['density', 'rho'],
        'constant_any2': ['dynamic_viscosity', 'viscosity'],
        'constant_any3': ['diameter'],
        'constant_any4': ['length', 'pipe_length'],
    },
    'friction_factor': {
        'quantity_any': ['velocity', 'volumetric_flow'],
        'constant_any': ['density'],
        'constant_any2': ['viscosity'],
        'constant_any3': ['roughness', 'epsilon'],
    },
    'head_loss': {
        'quantity_any': ['pressure'],
        'constant_any': ['density'],
    },
    'cavitation': {
        'quantity_any': ['pressure'],
        'constant_any': ['vapor_pressure', 'p_vapor'],
    },

    # =========================
    # THERMODYNAMICS
    # =========================
    'heat_transfer': {
        'quantity_any': ['temperature'],
        'quantity_any2': ['power', 'energy'],
    },
    'heat_capacity': {
        'quantity_any': ['temperature'],
        'constant_any': ['mass', 'cp', 'specific_heat'],
    },
    'thermal_efficiency': {
        'quantity_any': ['temperature'],
        'min_signals': 2,
    },
    'gibbs': {
        'quantity_any': ['temperature'],
        'quantity_any2': ['pressure'],
        'constant_any': ['enthalpy', 'entropy'],
    },

    # =========================
    # MECHANICAL
    # =========================
    'kinetic_energy': {
        'quantity_any': ['velocity', 'angular_velocity'],
        'constant_any': ['mass', 'inertia'],
    },
    'potential_energy': {
        'quantity_any': ['length'],  # position/height
        'constant_any': ['mass'],
    },
    'momentum': {
        'quantity_any': ['velocity'],
        'constant_any': ['mass'],
    },
    'torque_power': {
        'quantity_any': ['torque', 'angular_velocity'],
    },
    'vibration': {
        'quantity_any': ['acceleration', 'velocity'],
        'min_rows': 256,
    },

    # =========================
    # ELECTRICAL
    # =========================
    'power_factor': {
        'quantity_any': ['voltage'],
        'quantity_any2': ['current'],
    },
    'impedance': {
        'quantity_any': ['voltage'],
        'quantity_any2': ['current'],
        'quantity_any3': ['frequency'],
    },
    'efficiency_electrical': {
        'quantity_any': ['power'],
        'min_signals': 2,
    },

    # =========================
    # CHEMICAL / REACTION
    # =========================
    'conversion': {
        'min_signals': 2,  # inlet/outlet
        'quantity_any': ['concentration'],
    },
    'reaction_rate': {
        'quantity_any': ['concentration'],
        'quantity_any2': ['temperature'],
    },
    'arrhenius': {
        'quantity_any': ['temperature'],
        'min_entities': 3,  # need multiple runs at different temps
    },
    'residence_time': {
        'constant_any': ['volume', 'reactor_volume'],
        'quantity_any': ['volumetric_flow', 'mass_flow'],
    },

    # =========================
    # ELECTROCHEMISTRY
    # =========================
    'nernst': {
        'quantity_any': ['concentration'],
        'constant_any': ['standard_potential', 'e0'],
    },
    'faraday': {
        'quantity_any': ['current'],
        'constant_any': ['molecular_weight', 'mw'],
    },
}

# Discipline recommendations based on detected quantities
DISCIPLINE_SIGNALS = {
    'fluid_dynamics': ['velocity', 'volumetric_flow', 'pressure', 'mass_flow'],
    'thermodynamics': ['temperature', 'energy', 'power'],
    'mechanical': ['velocity', 'acceleration', 'torque', 'angular_velocity', 'force'],
    'electrical': ['voltage', 'current', 'power', 'resistance', 'frequency'],
    'chemical': ['concentration'],
    'core': [],  # fallback
}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def has_quantity(quantities: Set[str], needed: List[str]) -> bool:
    """Check if any of the needed quantities are present."""
    for q in needed:
        if q in quantities:
            return True
    return False


def has_constant(constants: Dict[str, ConstantInfo], names: List[str]) -> bool:
    """Check if any of the constant names are available."""
    const_keys = {k.lower() for k in constants.keys()}
    for name in names:
        if name.lower() in const_keys:
            return True
        # Also check quantity field
        for c in constants.values():
            if c.quantity and c.quantity.lower() == name.lower():
                return True
    return False


def check_engine_requirements(
    engine: str,
    reqs: dict,
    quantities: Set[str],
    constants: Dict[str, ConstantInfo],
    n_signals: int,
    n_entities: int,
    n_rows: int,
) -> tuple:
    """Check if an engine's requirements are met."""

    # No requirements = always available
    if not reqs:
        return True, ""

    # Check quantity requirements
    for key, names in reqs.items():
        if key.startswith('quantity_any'):
            if not has_quantity(quantities, names):
                return False, f"Need signal: {' or '.join(names)}"

    # Check constant requirements
    for key, names in reqs.items():
        if key.startswith('constant_any'):
            if not has_constant(constants, names):
                return False, f"Need constant: {' or '.join(names)}"

    # Check minimums
    if reqs.get('min_signals', 0) > n_signals:
        return False, f"Need {reqs['min_signals']} signals"

    if reqs.get('min_entities', 0) > n_entities:
        return False, f"Need {reqs['min_entities']} entities"

    if reqs.get('min_rows', 0) > n_rows:
        return False, f"Need {reqs['min_rows']} rows"

    return True, ""


def recommend_discipline(quantities: Set[str]) -> str:
    """Recommend best discipline based on detected quantities."""
    scores = {}

    for discipline, signals in DISCIPLINE_SIGNALS.items():
        if not signals:
            scores[discipline] = 0
            continue
        matches = sum(1 for s in signals if s in quantities)
        scores[discipline] = matches / len(signals)

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'core'


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_capabilities(
    inspection: FileInspection,
    discipline: Optional[str] = None,
) -> Capabilities:
    """
    Determine what can be computed given the inspected file.

    Uses the gatekeeper's FileInspection to route to appropriate engines.

    Args:
        inspection: Result from inspect_file()
        discipline: Selected discipline (optional, will recommend if not set)

    Returns:
        Capabilities object
    """
    # Get quantities from inspection
    quantities = set(inspection.quantities_detected)

    # Also infer quantities from signal names if not detected
    for sig in inspection.signals:
        if sig.quantity:
            quantities.add(sig.quantity)

    # Get constants
    constants = inspection.constants

    # Count signals (non-constant, non-structural)
    n_signals = sum(
        1 for s in inspection.signals
        if not s.is_constant and not s.is_entity_id and not s.is_sequence
    )

    n_entities = len(inspection.entities)
    n_rows = inspection.row_count

    # Check each engine
    available_engines = []
    unavailable_engines = {}

    for engine, reqs in ENGINE_REQUIREMENTS.items():
        can_run, reason = check_engine_requirements(
            engine, reqs, quantities, constants, n_signals, n_entities, n_rows
        )
        if can_run:
            available_engines.append(engine)
        else:
            unavailable_engines[engine] = reason

    # Determine capability level
    has_units = inspection.units_detected > 0
    has_constants = len(constants) > 0
    has_physics_engines = any(
        e in available_engines
        for e in ['reynolds', 'pressure_drop', 'heat_transfer', 'kinetic_energy']
    )

    if has_physics_engines and has_constants:
        level = 4
        level_name = "Physics"
    elif n_signals >= 3 and has_constants:
        level = 3
        level_name = "Constants"
    elif n_signals >= 2:
        level = 2
        level_name = "Geometry"
    elif has_units:
        level = 1
        level_name = "Units"
    else:
        level = 0
        level_name = "Basic"

    # What stages are available
    can_vector = n_signals >= 1
    can_geometry = n_signals >= 2
    can_dynamics = n_signals >= 2 and n_rows >= 50
    can_mechanics = has_constants and n_signals >= 2

    # What's missing for next level
    missing = []
    if level < 1 and not has_units:
        missing.append("Add unit suffixes to columns (e.g., pressure_psi, flow_gpm)")
    if level < 2 and n_signals < 2:
        missing.append("Add more signal columns for cross-signal analysis")
    if level < 3 and not has_constants:
        missing.append("Add constants (header comments or constant columns)")
    if level < 4:
        missing_physics = []
        if 'density' not in quantities and not has_constant(constants, ['density', 'rho']):
            missing_physics.append("density")
        if 'velocity' not in quantities and 'volumetric_flow' not in quantities:
            missing_physics.append("velocity or flow signal")
        if missing_physics:
            missing.append(f"For physics engines: {', '.join(missing_physics)}")

    # Recommend discipline
    recommended = discipline or recommend_discipline(quantities)

    # Extract constant values for UI
    constants_available = {
        k: v.value for k, v in constants.items()
    }

    # Summary
    core_count = sum(1 for e in available_engines if e in ['statistics', 'trend', 'entropy', 'hurst'])
    physics_count = len(available_engines) - core_count

    summary = (
        f"Level {level}: {level_name} | "
        f"{n_signals} signals, {len(quantities)} quantities, "
        f"{len(constants)} constants | "
        f"{len(available_engines)} engines ({core_count} core + {physics_count} domain)"
    )

    return Capabilities(
        level=level,
        level_name=level_name,
        can_vector=can_vector,
        can_geometry=can_geometry,
        can_dynamics=can_dynamics,
        can_mechanics=can_mechanics,
        available_engines=sorted(available_engines),
        unavailable_engines=unavailable_engines,
        quantities=sorted(quantities),
        constants_available=constants_available,
        missing_for_next_level=missing,
        recommended_discipline=recommended,
        summary=summary,
    )
