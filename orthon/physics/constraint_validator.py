"""
ORTHON Constraint Validator

Validates physical constraints and detects anomalies that violate
fundamental physics laws.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class ViolationType(Enum):
    """Types of constraint violations."""
    ENERGY_VIOLATION = "energy_violation"
    MASS_VIOLATION = "mass_violation"
    THERMODYNAMIC_VIOLATION = "thermodynamic_violation"
    RATE_VIOLATION = "rate_violation"
    BOUND_VIOLATION = "bound_violation"


@dataclass
class ConstraintDefinition:
    """Definition of a physical constraint."""
    name: str
    constraint_type: str
    expression: str  # Human-readable constraint
    tolerance: float
    severity_if_violated: str  # 'critical', 'warning', 'info'


class ConstraintValidator:
    """
    Validates that sensor data is physically consistent.

    Detects anomalies that would violate fundamental physics:
    - Temperatures below absolute zero
    - Negative masses or concentrations
    - Efficiency > 100%
    - Impossible rate changes
    """

    def __init__(self, system_type: str = "generic"):
        self.system_type = system_type

        # Physical bounds
        self.bounds = self._initialize_bounds()

        # Rate limits (maximum allowable rate of change)
        self.rate_limits = self._initialize_rate_limits()

        # Previous values for rate checking
        self.previous_values: Dict[str, Tuple[float, float]] = {}

        # Constraint definitions
        self.constraints: List[ConstraintDefinition] = self._define_constraints()

    def _initialize_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Initialize physical bounds for variables."""

        common_bounds = {
            # Temperatures (K)
            'temperature': (0, 5000),
            'T2': (200, 400),      # Turbofan inlet
            'T30': (800, 1800),    # Turbofan HPC outlet
            'T50': (500, 1500),    # Turbofan LPT outlet

            # Pressures (Pa or psia)
            'pressure': (0, 1e9),
            'P2': (5, 25),         # Turbofan inlet psia
            'P30': (100, 800),     # Turbofan HPC outlet psia

            # Concentrations (0-1 or mol/L)
            'concentration': (0, None),
            'conversion': (0, 1),
            'selectivity': (0, 1),

            # Flows (must be non-negative)
            'flow_rate': (0, None),
            'mass_flow': (0, None),

            # Efficiencies (0-1)
            'efficiency': (0, 1),
            'isentropic_efficiency': (0, 1),

            # Speeds (rpm)
            'Nf': (0, 5000),       # Fan speed
            'Nc': (0, 15000),      # Core speed

            # Ratios
            'epr': (0.5, 3),       # Engine pressure ratio

            # System metrics
            'cpu_percent': (0, 100),
            'memory_percent': (0, 100),
        }

        return common_bounds

    def _initialize_rate_limits(self) -> Dict[str, float]:
        """Initialize rate limits (max change per second)."""

        return {
            # Temperatures (K/s)
            'temperature': 50,
            'T2': 10,
            'T30': 100,
            'T50': 80,

            # Pressures (%/s)
            'pressure': 0.5,
            'P2': 0.3,
            'P30': 0.5,

            # Speeds (%/s)
            'Nf': 0.2,
            'Nc': 0.2,

            # Flow rates (%/s)
            'flow_rate': 0.3,

            # Concentrations
            'concentration': 0.1,
        }

    def _define_constraints(self) -> List[ConstraintDefinition]:
        """Define physical constraints for the system."""

        constraints = [
            ConstraintDefinition(
                name="temperature_positive",
                constraint_type="bound",
                expression="T > 0 K (above absolute zero)",
                tolerance=0,
                severity_if_violated="critical"
            ),
            ConstraintDefinition(
                name="mass_conservation",
                constraint_type="conservation",
                expression="mass_in = mass_out + accumulation",
                tolerance=0.05,
                severity_if_violated="warning"
            ),
            ConstraintDefinition(
                name="energy_conservation",
                constraint_type="conservation",
                expression="energy_in = energy_out + losses",
                tolerance=0.1,
                severity_if_violated="warning"
            ),
            ConstraintDefinition(
                name="efficiency_bound",
                constraint_type="bound",
                expression="0 <= efficiency <= 1",
                tolerance=0,
                severity_if_violated="critical"
            ),
            ConstraintDefinition(
                name="concentration_positive",
                constraint_type="bound",
                expression="concentration >= 0",
                tolerance=0,
                severity_if_violated="critical"
            ),
        ]

        return constraints

    def validate(self, sensor_data: Dict) -> Dict[str, Any]:
        """
        Validate sensor data against physical constraints.

        Returns validation results with any violations detected.
        """

        timestamp = sensor_data.get('timestamp', time.time())
        violations = []
        warnings = []

        # Check physical bounds
        bound_results = self._check_bounds(sensor_data)
        violations.extend(bound_results['violations'])
        warnings.extend(bound_results['warnings'])

        # Check rate limits
        rate_results = self._check_rate_limits(sensor_data, timestamp)
        violations.extend(rate_results['violations'])
        warnings.extend(rate_results['warnings'])

        # Check thermodynamic constraints
        thermo_results = self._check_thermodynamic_constraints(sensor_data)
        violations.extend(thermo_results['violations'])
        warnings.extend(thermo_results['warnings'])

        # Check consistency
        consistency_results = self._check_consistency(sensor_data)
        violations.extend(consistency_results['violations'])
        warnings.extend(consistency_results['warnings'])

        # Update previous values
        self._update_previous_values(sensor_data, timestamp)

        # Calculate validation score
        validation_score = self._calculate_validation_score(violations, warnings)

        return {
            'valid': len(violations) == 0,
            'validation_score': validation_score,
            'n_violations': len(violations),
            'n_warnings': len(warnings),
            'violations': violations,
            'warnings': warnings,
            'timestamp': timestamp,
        }

    def _check_bounds(self, data: Dict) -> Dict[str, List]:
        """Check that values are within physical bounds."""

        violations = []
        warnings = []

        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue

            if key in self.bounds:
                lower, upper = self.bounds[key]

                if lower is not None and value < lower:
                    violations.append({
                        'type': ViolationType.BOUND_VIOLATION.value,
                        'variable': key,
                        'value': value,
                        'bound': f'>= {lower}',
                        'message': f'{key} = {value} is below physical minimum {lower}',
                        'severity': 'critical',
                    })

                if upper is not None and value > upper:
                    violations.append({
                        'type': ViolationType.BOUND_VIOLATION.value,
                        'variable': key,
                        'value': value,
                        'bound': f'<= {upper}',
                        'message': f'{key} = {value} exceeds physical maximum {upper}',
                        'severity': 'critical',
                    })

            # Generic checks for common patterns
            if 'percent' in key.lower() or 'efficiency' in key.lower():
                if value < 0:
                    violations.append({
                        'type': ViolationType.BOUND_VIOLATION.value,
                        'variable': key,
                        'value': value,
                        'message': f'{key} cannot be negative',
                        'severity': 'critical',
                    })
                if value > 100 and 'percent' in key.lower():
                    violations.append({
                        'type': ViolationType.BOUND_VIOLATION.value,
                        'variable': key,
                        'value': value,
                        'message': f'{key} cannot exceed 100%',
                        'severity': 'critical',
                    })

        return {'violations': violations, 'warnings': warnings}

    def _check_rate_limits(self, data: Dict, timestamp: float) -> Dict[str, List]:
        """Check that rate of change is physically plausible."""

        violations = []
        warnings = []

        for key, value in data.items():
            if not isinstance(value, (int, float)):
                continue

            if key in self.previous_values:
                prev_value, prev_time = self.previous_values[key]
                dt = timestamp - prev_time

                if dt > 0:
                    rate = abs(value - prev_value) / dt

                    if key in self.rate_limits:
                        max_rate = self.rate_limits[key]

                        # For percentage-based rates
                        if prev_value != 0:
                            pct_rate = rate / abs(prev_value)
                        else:
                            pct_rate = 0

                        if pct_rate > max_rate:
                            severity = 'critical' if pct_rate > max_rate * 5 else 'warning'

                            if severity == 'critical':
                                violations.append({
                                    'type': ViolationType.RATE_VIOLATION.value,
                                    'variable': key,
                                    'rate': pct_rate,
                                    'max_rate': max_rate,
                                    'message': f'{key} changed too fast: {pct_rate:.1%}/s > {max_rate:.1%}/s limit',
                                    'severity': severity,
                                })
                            else:
                                warnings.append({
                                    'type': ViolationType.RATE_VIOLATION.value,
                                    'variable': key,
                                    'rate': pct_rate,
                                    'max_rate': max_rate,
                                    'message': f'{key} rate of change elevated: {pct_rate:.1%}/s',
                                    'severity': severity,
                                })

        return {'violations': violations, 'warnings': warnings}

    def _check_thermodynamic_constraints(self, data: Dict) -> Dict[str, List]:
        """Check thermodynamic constraints."""

        violations = []
        warnings = []

        # Efficiency cannot exceed Carnot limit
        if 'efficiency' in data:
            efficiency = data['efficiency']
            t_hot = data.get('hot_temperature', data.get('T30', 1000))
            t_cold = data.get('cold_temperature', data.get('T2', 300))

            if t_hot > t_cold and t_hot > 0:
                carnot = 1 - t_cold / t_hot
                if efficiency > carnot * 1.1:  # Allow 10% tolerance
                    violations.append({
                        'type': ViolationType.THERMODYNAMIC_VIOLATION.value,
                        'variable': 'efficiency',
                        'value': efficiency,
                        'limit': carnot,
                        'message': f'Efficiency {efficiency:.1%} exceeds Carnot limit {carnot:.1%}',
                        'severity': 'critical',
                    })

        # Temperature gradient direction
        if 'T30' in data and 'T2' in data:
            t_compressor_in = data['T2']
            t_compressor_out = data['T30']
            if t_compressor_out < t_compressor_in:
                warnings.append({
                    'type': ViolationType.THERMODYNAMIC_VIOLATION.value,
                    'message': 'Compressor outlet cooler than inlet (impossible without cooling)',
                    'severity': 'warning',
                })

        return {'violations': violations, 'warnings': warnings}

    def _check_consistency(self, data: Dict) -> Dict[str, List]:
        """Check for internal data consistency."""

        violations = []
        warnings = []

        # Pressure ratio consistency
        if 'P30' in data and 'P2' in data and 'epr' in data:
            calculated_epr = data['P30'] / data['P2']
            reported_epr = data['epr']

            if abs(calculated_epr - reported_epr) / reported_epr > 0.1:
                warnings.append({
                    'type': 'consistency',
                    'message': f'Pressure ratio inconsistency: calculated {calculated_epr:.2f}, reported {reported_epr:.2f}',
                    'severity': 'warning',
                })

        return {'violations': violations, 'warnings': warnings}

    def _update_previous_values(self, data: Dict, timestamp: float):
        """Update stored previous values for rate checking."""

        for key, value in data.items():
            if isinstance(value, (int, float)):
                self.previous_values[key] = (value, timestamp)

    def _calculate_validation_score(self, violations: List, warnings: List) -> float:
        """Calculate overall validation score (0-1)."""

        score = 1.0

        # Critical violations are severe
        critical_violations = sum(1 for v in violations if v.get('severity') == 'critical')
        score -= critical_violations * 0.3

        # Warnings are less severe
        score -= len(warnings) * 0.05

        return max(0, min(1, score))

    def add_bound(self, variable: str, lower: Optional[float], upper: Optional[float]):
        """Add or update a bound constraint."""

        self.bounds[variable] = (lower, upper)

    def add_rate_limit(self, variable: str, max_rate: float):
        """Add or update a rate limit."""

        self.rate_limits[variable] = max_rate

    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of defined constraints."""

        return {
            'n_bounds': len(self.bounds),
            'n_rate_limits': len(self.rate_limits),
            'n_constraints': len(self.constraints),
            'bounds': dict(self.bounds),
            'rate_limits': dict(self.rate_limits),
        }

    def reset_history(self):
        """Reset rate checking history."""

        self.previous_values.clear()
