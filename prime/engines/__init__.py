"""
Prime Engines

Process flow: Typology → Classification → Geometry → Mass → Structure → Stability → Tipping → Spin Glass

Level 0: System Typology     - What kind of system?
Level 1: Stationarity        - Is signal stable or evolving?
Level 2: Signal Classification - What behavior does each signal exhibit?
Geometry Engine              - How do signals couple?
Mass Engine                  - How much energy is in the system?
Structure Engine             - What is the system's resilience state?
Stability Engine             - Is the system stable?
Tipping Engine               - What kind of failure mode?
Spin Glass Engine            - Physical interpretation via statistical mechanics
Trajectory Monitor           - Real-time trajectory sensitivity and saddle detection
"""

from .typology_engine import classify_system_type, SystemType
from .stationarity_engine import test_stationarity, StationarityResult
from .classification_engine import classify_signal, SignalClass
from .signal_geometry import compute_geometry, GeometryResult
from .mass_engine import compute_mass, MassResult
from .structure_engine import compute_structure, StructureResult
from .stability_engine import compute_stability, StabilityResult
from .tipping_engine import classify_tipping, TippingType
from .spin_glass import compute_spin_glass, SpinGlassPhase
from .trajectory_monitor import (
    TrajectoryAlert,
    SensitivityChange,
    TrajectoryStatus,
    SensitivityReport,
    interpret_ftle,
    interpret_saddle_proximity,
    generate_sensitivity_report,
    generate_recommendations,
    assess_trajectory_status,
    monitor_trajectory,
)

__all__ = [
    # Level 0: Typology
    'classify_system_type',
    'SystemType',

    # Level 1: Stationarity
    'test_stationarity',
    'StationarityResult',

    # Level 2: Classification
    'classify_signal',
    'SignalClass',

    # Geometry
    'compute_geometry',
    'GeometryResult',

    # Mass
    'compute_mass',
    'MassResult',

    # Structure
    'compute_structure',
    'StructureResult',

    # Stability
    'compute_stability',
    'StabilityResult',

    # Tipping
    'classify_tipping',
    'TippingType',

    # Spin Glass
    'compute_spin_glass',
    'SpinGlassPhase',

    # Trajectory Monitor
    'TrajectoryAlert',
    'SensitivityChange',
    'TrajectoryStatus',
    'SensitivityReport',
    'interpret_ftle',
    'interpret_saddle_proximity',
    'generate_sensitivity_report',
    'generate_recommendations',
    'assess_trajectory_status',
    'monitor_trajectory',
]
