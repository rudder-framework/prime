"""
Physics Constants and Unit Conversions
======================================

When users provide units, Rudder computes real physics.
When they don't, proxy physics still works.

Proxy physics preserves DYNAMICS, not MAGNITUDE.
- You don't need units to detect energy CHANGE
- You need units for energy MAGNITUDE

The key insight: Symplectic structure is unit-agnostic.
Energy conservation (or loss) is detectable without knowing Joules.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, Any


@dataclass
class PhysicsConstants:
    """
    Domain-specific constants for real physics calculations.

    All optional - if not provided, proxy physics is used.
    Proxy physics detects the same dynamics, just without magnitudes.
    """
    # Mechanical
    mass: Optional[float] = None              # kg
    spring_constant: Optional[float] = None   # N/m
    damping_coefficient: Optional[float] = None  # N·s/m

    # Thermal
    specific_heat: Optional[float] = None     # J/(kg·K)
    thermal_mass: Optional[float] = None      # J/K (= mass * specific_heat)

    # Fluid
    volume: Optional[float] = None            # m³
    density: Optional[float] = None           # kg/m³

    # Electrical
    inductance: Optional[float] = None        # H
    capacitance: Optional[float] = None       # F
    resistance: Optional[float] = None        # Ω

    # Rotational
    moment_of_inertia: Optional[float] = None  # kg·m²

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# =============================================================================
# UNIT CATEGORY MAPPING
# =============================================================================

# Map actual units to physics categories
UNIT_TO_CATEGORY = {
    # Velocity → Kinetic energy (½mv²)
    'm/s': 'velocity', 'ft/s': 'velocity', 'mph': 'velocity', 'km/h': 'velocity',

    # Displacement → Potential energy (½kx²)
    'mm': 'displacement', 'm': 'displacement', 'cm': 'displacement',
    'in': 'displacement', 'ft': 'displacement',

    # Acceleration
    'g': 'acceleration', 'm/s2': 'acceleration', 'ft/s2': 'acceleration',

    # Temperature → Thermal energy (mcT)
    'degC': 'temperature', 'degF': 'temperature', 'K': 'temperature',
    '°C': 'temperature', '°F': 'temperature',

    # Pressure → PV work
    'bar': 'pressure', 'PSI': 'pressure', 'psi': 'pressure',
    'Pa': 'pressure', 'kPa': 'pressure', 'MPa': 'pressure',

    # Flow → Kinetic energy (½ρv²)
    'gpm': 'flow', 'lpm': 'flow', 'm3/s': 'flow', 'L/min': 'flow',

    # Current → Magnetic energy (½LI²)
    'A': 'current', 'mA': 'current', 'amp': 'current',

    # Voltage → Electric energy (½CV²)
    'V': 'voltage', 'mV': 'voltage', 'kV': 'voltage',

    # Angular velocity → Rotational energy (½Iω²)
    'rpm': 'angular_velocity', 'rad/s': 'angular_velocity', 'Hz': 'angular_velocity',

    # Vibration (treat as velocity for energy)
    'mm/s': 'velocity',

    # Force
    'N': 'force', 'kN': 'force', 'lbf': 'force',

    # Torque
    'Nm': 'torque', 'ft-lb': 'torque',
}

# Valid unit categories for energy calculations
VALID_UNIT_CATEGORIES = [
    'velocity',         # m/s, ft/s → kinetic energy
    'displacement',     # m, mm → potential energy (spring)
    'acceleration',     # m/s², g
    'temperature',      # K, °C → thermal energy
    'pressure',         # Pa, bar → PV work
    'flow',             # m³/s, gpm → kinetic energy
    'current',          # A → magnetic energy
    'voltage',          # V → electric energy
    'angular_velocity', # rad/s, rpm → rotational energy
]


# =============================================================================
# ENERGY FORMULAS
# =============================================================================

def _kinetic_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½mv²"""
    return 0.5 * c.mass * y**2 if c.mass else None

def _potential_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½kx²"""
    return 0.5 * c.spring_constant * y**2 if c.spring_constant else None

def _thermal_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = C_th × T (thermal mass × temperature)"""
    return c.thermal_mass * abs(y) if c.thermal_mass else None

def _pressure_work(y: float, c: PhysicsConstants) -> Optional[float]:
    """W = PV"""
    return y * c.volume if c.volume else None

def _flow_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½ρVv² (kinetic energy of fluid)"""
    if c.density and c.volume:
        return 0.5 * c.density * c.volume * y**2
    return None

def _magnetic_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½LI²"""
    return 0.5 * c.inductance * y**2 if c.inductance else None

def _electric_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½CV²"""
    return 0.5 * c.capacitance * y**2 if c.capacitance else None

def _rotational_energy(y: float, c: PhysicsConstants) -> Optional[float]:
    """E = ½Iω²"""
    return 0.5 * c.moment_of_inertia * y**2 if c.moment_of_inertia else None


# Map unit categories to energy formulas
ENERGY_FORMULAS: Dict[str, Callable[[float, PhysicsConstants], Optional[float]]] = {
    'velocity': _kinetic_energy,
    'displacement': _potential_energy,
    'acceleration': lambda y, c: None,  # Need integral to get energy
    'temperature': _thermal_energy,
    'pressure': _pressure_work,
    'flow': _flow_energy,
    'current': _magnetic_energy,
    'voltage': _electric_energy,
    'angular_velocity': _rotational_energy,
    'force': lambda y, c: None,  # Need displacement for work
    'torque': lambda y, c: None,  # Need angle for work
}


def can_compute_real_energy(unit_category: str, constants: PhysicsConstants) -> bool:
    """Check if we have enough info for real energy calculation."""
    if unit_category not in ENERGY_FORMULAS:
        return False
    result = ENERGY_FORMULAS[unit_category](1.0, constants)  # Test with dummy value
    return result is not None


def get_unit_category(unit: str) -> Optional[str]:
    """Get physics category for a unit string."""
    return UNIT_TO_CATEGORY.get(unit)


__all__ = [
    'PhysicsConstants',
    'UNIT_TO_CATEGORY',
    'VALID_UNIT_CATEGORIES',
    'ENERGY_FORMULAS',
    'can_compute_real_energy',
    'get_unit_category',
]
