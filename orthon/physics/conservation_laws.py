"""
ORTHON Conservation Law Monitor

Physical constraint monitoring for industrial systems.
Validates conservation of energy, mass, and momentum in real-time.
Detects physics violations that indicate equipment problems.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque


class ConstraintType(Enum):
    """Types of physics constraints."""
    ENERGY_CONSERVATION = "energy_conservation"
    MASS_CONSERVATION = "mass_conservation"
    MOMENTUM_CONSERVATION = "momentum_conservation"
    THERMODYNAMIC_EFFICIENCY = "thermodynamic_efficiency"
    PRESSURE_BALANCE = "pressure_balance"
    HEAT_BALANCE = "heat_balance"


@dataclass
class PhysicsViolation:
    """Represents a physics constraint violation."""
    constraint_type: ConstraintType
    severity: float  # 0-1, where 1 is complete violation
    expected_value: float
    actual_value: float
    confidence: float
    timestamp: float
    message: str = ""

    def to_dict(self) -> Dict:
        return {
            'constraint_type': self.constraint_type.value,
            'severity': self.severity,
            'expected_value': self.expected_value,
            'actual_value': self.actual_value,
            'confidence': self.confidence,
            'timestamp': self.timestamp,
            'message': self.message,
        }


class ConservationLawMonitor:
    """
    Real-time monitoring of fundamental physics constraints.

    Provides the "physics reality check" that complements geometric analysis.
    Detects violations of conservation laws that indicate equipment problems.
    """

    def __init__(self, system_type: str, system_parameters: Optional[Dict] = None):
        self.system_type = system_type
        self.system_params = system_parameters or self._default_parameters()

        # Violation tracking
        self.violation_history: deque = deque(maxlen=1000)
        self.baseline_efficiency: Optional[float] = None
        self.efficiency_history: deque = deque(maxlen=500)

        # Tolerances for constraint checking
        self.tolerances = {
            'energy': 0.05,  # 5% energy balance tolerance
            'mass': 0.02,    # 2% mass balance tolerance
            'momentum': 0.05,
            'efficiency': 0.1,
        }

    def _default_parameters(self) -> Dict:
        """Default system parameters."""

        defaults = {
            'generic': {
                'heat_capacity': 4.18,  # kJ/kg·K (water)
                'heat_loss_coefficient': 0.1,
                'reference_temperature': 298,  # K (25°C)
            },
            'chemical_reactor': {
                'heat_capacity': 4.18,
                'reaction_enthalpy': -50000,  # J/mol (exothermic)
                'heat_loss_coefficient': 0.05,
                'molecular_weight': 100,  # g/mol
                'reference_temperature': 298,
            },
            'turbofan': {
                'fuel_heating_value': 43000,  # kJ/kg
                'compressor_efficiency': 0.85,
                'turbine_efficiency': 0.90,
                'reference_temperature': 288,  # K
            },
            'heat_exchanger': {
                'heat_transfer_coefficient': 500,  # W/m²·K
                'heat_capacity': 4.18,
                'reference_temperature': 298,
            },
        }

        return defaults.get(self.system_type, defaults['generic'])

    def analyze_physics_constraints(self, sensor_data: Dict) -> Dict[str, Any]:
        """
        Comprehensive physics constraint analysis.

        Returns physics health metrics that complement PRISM geometric analysis.
        """

        constraints = {}
        violations = []
        timestamp = sensor_data.get('timestamp', time.time())

        # Energy conservation analysis
        energy_balance = self._check_energy_conservation(sensor_data)
        constraints['energy'] = energy_balance

        if energy_balance['violation_severity'] > self.tolerances['energy']:
            violations.append(PhysicsViolation(
                ConstraintType.ENERGY_CONSERVATION,
                energy_balance['violation_severity'],
                energy_balance['expected_balance'],
                energy_balance['actual_balance'],
                energy_balance['confidence'],
                timestamp,
                f"Energy imbalance: {energy_balance['balance_error']:.1%}"
            ))

        # Mass conservation analysis
        mass_balance = self._check_mass_conservation(sensor_data)
        constraints['mass'] = mass_balance

        if mass_balance['violation_severity'] > self.tolerances['mass']:
            violations.append(PhysicsViolation(
                ConstraintType.MASS_CONSERVATION,
                mass_balance['violation_severity'],
                mass_balance['expected_flow_balance'],
                mass_balance['actual_flow_balance'],
                mass_balance['confidence'],
                timestamp,
                f"Mass imbalance: {mass_balance['balance_error']:.1%}"
            ))

        # Thermodynamic efficiency
        thermo_analysis = self._check_thermodynamic_efficiency(sensor_data)
        constraints['thermodynamics'] = thermo_analysis

        if thermo_analysis.get('violation_detected', False):
            violations.append(PhysicsViolation(
                ConstraintType.THERMODYNAMIC_EFFICIENCY,
                thermo_analysis.get('violation_severity', 0),
                thermo_analysis.get('carnot_efficiency_limit', 0),
                thermo_analysis.get('actual_efficiency', 0),
                thermo_analysis.get('confidence', 0.5),
                timestamp,
                "Thermodynamic efficiency violation"
            ))

        # Record violations
        for v in violations:
            self.violation_history.append(v.to_dict())

        # Calculate overall physics health score
        physics_health_score = self._calculate_physics_health_score(constraints)

        return {
            'physics_health_score': physics_health_score,
            'constraint_analysis': constraints,
            'active_violations': [v.to_dict() for v in violations],
            'violation_count': len(violations),
            'violation_trend': self._calculate_violation_trend(),
            'system_efficiency': thermo_analysis.get('overall_efficiency', 0),
            'physics_stability': self._assess_physics_stability(constraints),
            'timestamp': timestamp,
        }

    def _check_energy_conservation(self, data: Dict) -> Dict:
        """Check energy balance for the system."""

        if self.system_type == "chemical_reactor":
            return self._reactor_energy_balance(data)
        elif self.system_type == "turbofan":
            return self._turbofan_energy_balance(data)
        elif self.system_type == "heat_exchanger":
            return self._heat_exchanger_energy_balance(data)
        else:
            return self._generic_energy_balance(data)

    def _reactor_energy_balance(self, data: Dict) -> Dict:
        """Energy balance for chemical reactor."""

        heat_capacity = self.system_params.get('heat_capacity', 4.18)
        reaction_enthalpy = self.system_params.get('reaction_enthalpy', -50000)
        heat_loss_coeff = self.system_params.get('heat_loss_coefficient', 0.05)
        ref_temp = self.system_params.get('reference_temperature', 298)

        # Energy inputs
        feed_flow = data.get('feed_flow', data.get('flow_rate', 0))
        feed_temp = data.get('feed_temperature', data.get('inlet_temp', ref_temp))
        feed_enthalpy = feed_flow * (feed_temp - ref_temp) * heat_capacity

        reaction_rate = data.get('reaction_rate', 0)
        reaction_heat = reaction_rate * abs(reaction_enthalpy) / 1000  # kW

        # Energy outputs
        product_flow = data.get('product_flow', feed_flow)
        product_temp = data.get('product_temperature', data.get('outlet_temp', data.get('temperature', ref_temp)))
        product_enthalpy = product_flow * (product_temp - ref_temp) * heat_capacity

        cooling_rate = data.get('cooling_rate', data.get('heat_removal', 0))
        reactor_temp = data.get('reactor_temperature', data.get('temperature', ref_temp))
        heat_loss = heat_loss_coeff * (reactor_temp - ref_temp)

        # Energy balance calculation
        energy_in = feed_enthalpy + reaction_heat
        energy_out = product_enthalpy + cooling_rate + heat_loss

        if energy_in > 0:
            balance_error = abs(energy_in - energy_out) / energy_in
        else:
            balance_error = 0

        return {
            'energy_in': float(energy_in),
            'energy_out': float(energy_out),
            'expected_balance': 0.0,
            'actual_balance': float(energy_in - energy_out),
            'balance_error': float(balance_error),
            'violation_severity': float(min(balance_error / 0.1, 1.0)),
            'confidence': 0.9 if all(k in data for k in ['feed_flow', 'temperature']) else 0.5,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _turbofan_energy_balance(self, data: Dict) -> Dict:
        """Energy balance for turbofan engine."""

        fuel_hv = self.system_params.get('fuel_heating_value', 43000)

        # Energy input (fuel)
        fuel_flow = data.get('fuel_flow', data.get('phi', 0) * data.get('Ps30', 0) / 1000)
        energy_in = fuel_flow * fuel_hv

        # Energy outputs
        thrust_power = data.get('thrust_power', 0)
        exhaust_temp = data.get('T50', data.get('exhaust_temp', 800))
        inlet_temp = data.get('T2', data.get('inlet_temp', 288))
        exhaust_loss = data.get('exhaust_flow', fuel_flow * 50) * 1.0 * (exhaust_temp - inlet_temp)

        energy_out = thrust_power + exhaust_loss

        if energy_in > 0:
            balance_error = abs(energy_in - energy_out) / energy_in
        else:
            balance_error = 0

        return {
            'energy_in': float(energy_in),
            'energy_out': float(energy_out),
            'expected_balance': 0.0,
            'actual_balance': float(energy_in - energy_out),
            'balance_error': float(balance_error),
            'violation_severity': float(min(balance_error / 0.15, 1.0)),
            'confidence': 0.7,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _heat_exchanger_energy_balance(self, data: Dict) -> Dict:
        """Energy balance for heat exchanger."""

        heat_capacity = self.system_params.get('heat_capacity', 4.18)

        # Hot side
        hot_flow = data.get('hot_flow', 0)
        hot_inlet_temp = data.get('hot_inlet_temp', 80)
        hot_outlet_temp = data.get('hot_outlet_temp', 50)
        hot_heat = hot_flow * heat_capacity * (hot_inlet_temp - hot_outlet_temp)

        # Cold side
        cold_flow = data.get('cold_flow', 0)
        cold_inlet_temp = data.get('cold_inlet_temp', 20)
        cold_outlet_temp = data.get('cold_outlet_temp', 45)
        cold_heat = cold_flow * heat_capacity * (cold_outlet_temp - cold_inlet_temp)

        # Heat loss
        heat_loss = data.get('heat_loss', 0)

        # Balance
        if hot_heat > 0:
            balance_error = abs(hot_heat - cold_heat - heat_loss) / hot_heat
        else:
            balance_error = 0

        return {
            'energy_in': float(hot_heat),
            'energy_out': float(cold_heat + heat_loss),
            'expected_balance': 0.0,
            'actual_balance': float(hot_heat - cold_heat - heat_loss),
            'balance_error': float(balance_error),
            'violation_severity': float(min(balance_error / 0.1, 1.0)),
            'confidence': 0.85,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _generic_energy_balance(self, data: Dict) -> Dict:
        """Generic energy balance when system type unknown."""

        energy_in = data.get('energy_in', data.get('power_in', 0))
        energy_out = data.get('energy_out', data.get('power_out', 0))
        losses = data.get('losses', data.get('heat_loss', 0))

        if energy_in > 0:
            balance_error = abs(energy_in - energy_out - losses) / energy_in
        else:
            balance_error = 0

        return {
            'energy_in': float(energy_in),
            'energy_out': float(energy_out + losses),
            'expected_balance': 0.0,
            'actual_balance': float(energy_in - energy_out - losses),
            'balance_error': float(balance_error),
            'violation_severity': float(min(balance_error / 0.1, 1.0)),
            'confidence': 0.5,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _check_mass_conservation(self, data: Dict) -> Dict:
        """Check mass balance - what goes in must come out."""

        if self.system_type == "chemical_reactor":
            return self._reactor_mass_balance(data)
        else:
            return self._generic_mass_balance(data)

    def _reactor_mass_balance(self, data: Dict) -> Dict:
        """Mass balance for chemical reactor."""

        # Mass flows
        feed_flow = data.get('feed_flow', data.get('flow_rate', 0))
        feed_density = data.get('feed_density', 1000)
        mass_in = feed_flow * feed_density / 1000  # kg/s

        product_flow = data.get('product_flow', feed_flow)
        product_density = data.get('product_density', feed_density)
        mass_out = product_flow * product_density / 1000

        # Account for accumulation
        accumulation = data.get('accumulation_rate', 0)

        # Expected balance
        expected_mass_out = mass_in - accumulation

        if mass_in > 0:
            mass_balance_error = abs(mass_out - expected_mass_out) / mass_in
        else:
            mass_balance_error = 0

        return {
            'mass_in': float(mass_in),
            'mass_out': float(mass_out),
            'expected_flow_balance': float(expected_mass_out),
            'actual_flow_balance': float(mass_out),
            'balance_error': float(mass_balance_error),
            'violation_severity': float(min(mass_balance_error / 0.05, 1.0)),
            'confidence': 0.8 if 'feed_flow' in data else 0.4,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _generic_mass_balance(self, data: Dict) -> Dict:
        """Generic mass balance."""

        mass_in = data.get('mass_in', data.get('inlet_flow', 0))
        mass_out = data.get('mass_out', data.get('outlet_flow', mass_in))
        accumulation = data.get('accumulation', 0)

        expected_out = mass_in - accumulation

        if mass_in > 0:
            balance_error = abs(mass_out - expected_out) / mass_in
        else:
            balance_error = 0

        return {
            'mass_in': float(mass_in),
            'mass_out': float(mass_out),
            'expected_flow_balance': float(expected_out),
            'actual_flow_balance': float(mass_out),
            'balance_error': float(balance_error),
            'violation_severity': float(min(balance_error / 0.05, 1.0)),
            'confidence': 0.5,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _check_thermodynamic_efficiency(self, data: Dict) -> Dict:
        """Thermodynamic efficiency analysis."""

        # Get temperatures
        t_hot = data.get('hot_temperature', data.get('T30', data.get('temperature', 400)))
        t_cold = data.get('cold_temperature', data.get('T2', data.get('ambient_temp', 300)))

        # Calculate Carnot efficiency limit
        if t_hot > t_cold:
            carnot_efficiency = 1 - (t_cold / t_hot)
        else:
            carnot_efficiency = 0

        # Actual efficiency calculation (system-specific)
        if self.system_type == "chemical_reactor":
            conversion = data.get('conversion_rate', data.get('conversion', 0.8))
            selectivity = data.get('selectivity', 1.0)
            actual_efficiency = conversion * selectivity
        elif self.system_type == "turbofan":
            thrust = data.get('thrust', 0)
            fuel_power = data.get('fuel_power', 1)
            actual_efficiency = thrust / max(fuel_power, 1) if fuel_power > 0 else 0
        else:
            work_out = data.get('work_output', data.get('power_out', 0))
            heat_in = data.get('heat_input', data.get('power_in', 1))
            actual_efficiency = work_out / heat_in if heat_in > 0 else 0

        # Check for efficiency violations
        violation_detected = actual_efficiency > carnot_efficiency and carnot_efficiency > 0
        efficiency_ratio = actual_efficiency / carnot_efficiency if carnot_efficiency > 0 else 0

        # Track efficiency
        self.efficiency_history.append(actual_efficiency)

        # Calculate efficiency degradation
        degradation = self._calculate_efficiency_degradation(actual_efficiency)

        return {
            'carnot_efficiency_limit': float(carnot_efficiency),
            'actual_efficiency': float(actual_efficiency),
            'efficiency_ratio': float(efficiency_ratio),
            'overall_efficiency': float(actual_efficiency),
            'violation_detected': violation_detected,
            'violation_severity': float(max(0, efficiency_ratio - 1)) if violation_detected else 0,
            'degradation_indicator': float(degradation),
            'confidence': 0.7,
            'timestamp': data.get('timestamp', time.time()),
        }

    def _calculate_efficiency_degradation(self, current_efficiency: float) -> float:
        """Calculate efficiency degradation from baseline."""

        if self.baseline_efficiency is None:
            if len(self.efficiency_history) >= 10:
                self.baseline_efficiency = np.mean(list(self.efficiency_history)[:10])
            else:
                return 0.0

        if self.baseline_efficiency > 0:
            degradation = (self.baseline_efficiency - current_efficiency) / self.baseline_efficiency
            return max(0, degradation)

        return 0.0

    def _calculate_physics_health_score(self, constraints: Dict) -> float:
        """Overall physics health score (0-1, where 1 is perfect)."""

        scores = []
        weights = []

        # Energy conservation score
        if 'energy' in constraints:
            energy_violation = constraints['energy'].get('violation_severity', 0)
            energy_score = 1 - energy_violation
            scores.append(energy_score)
            weights.append(0.4)

        # Mass conservation score
        if 'mass' in constraints:
            mass_violation = constraints['mass'].get('violation_severity', 0)
            mass_score = 1 - mass_violation
            scores.append(mass_score)
            weights.append(0.4)

        # Efficiency score
        if 'thermodynamics' in constraints:
            efficiency = constraints['thermodynamics'].get('overall_efficiency', 0)
            efficiency_score = min(efficiency * 1.5, 1.0)  # Normalize assuming 66% is good
            scores.append(efficiency_score)
            weights.append(0.2)

        if not scores:
            return 1.0

        # Weighted average
        total_weight = sum(weights)
        health_score = sum(w * s for w, s in zip(weights, scores)) / total_weight

        return float(health_score)

    def _calculate_violation_trend(self) -> str:
        """Calculate trend in violations over time."""

        if len(self.violation_history) < 10:
            return 'stable'

        recent = list(self.violation_history)[-10:]
        older = list(self.violation_history)[-20:-10] if len(self.violation_history) >= 20 else []

        recent_count = len(recent)

        if older:
            older_count = len(older)
            if recent_count > older_count * 1.5:
                return 'increasing'
            elif recent_count < older_count * 0.5:
                return 'decreasing'

        return 'stable'

    def _assess_physics_stability(self, constraints: Dict) -> str:
        """Assess overall physics stability."""

        violations = sum(
            1 for c in constraints.values()
            if isinstance(c, dict) and c.get('violation_severity', 0) > 0.1
        )

        if violations >= 2:
            return 'unstable'
        elif violations == 1:
            return 'marginal'
        else:
            return 'stable'

    def get_violation_summary(self, window_minutes: int = 30) -> Dict:
        """Get summary of violations in time window."""

        cutoff = time.time() - window_minutes * 60
        recent = [v for v in self.violation_history if v.get('timestamp', 0) > cutoff]

        if not recent:
            return {
                'window_minutes': window_minutes,
                'total_violations': 0,
                'by_type': {},
                'avg_severity': 0,
            }

        by_type = {}
        for v in recent:
            constraint_type = v.get('constraint_type', 'unknown')
            by_type[constraint_type] = by_type.get(constraint_type, 0) + 1

        severities = [v.get('severity', 0) for v in recent]

        return {
            'window_minutes': window_minutes,
            'total_violations': len(recent),
            'by_type': by_type,
            'avg_severity': float(np.mean(severities)),
            'max_severity': float(np.max(severities)),
        }

    def set_tolerance(self, constraint_type: str, tolerance: float):
        """Set tolerance for a constraint type."""

        if constraint_type in self.tolerances:
            self.tolerances[constraint_type] = tolerance

    def reset_baseline(self):
        """Reset efficiency baseline for fresh learning."""

        self.baseline_efficiency = None
        self.efficiency_history.clear()
