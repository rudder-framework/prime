"""
ORTHON Thermodynamics Analyzer

Thermodynamic analysis for industrial systems including entropy production,
efficiency calculations, and heat transfer analysis.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class ThermodynamicState:
    """Thermodynamic state of a system."""
    temperature: float  # K
    pressure: float     # Pa
    enthalpy: float     # J/kg
    entropy: float      # J/kg·K
    quality: float = 1.0  # For two-phase systems


class ThermodynamicsAnalyzer:
    """
    Analyzes thermodynamic properties and efficiency of industrial systems.

    Provides:
    - Entropy production analysis
    - Exergy analysis
    - Heat transfer calculations
    - Efficiency assessment
    """

    # Physical constants
    R_UNIVERSAL = 8.314  # J/mol·K

    def __init__(self, system_type: str = "generic"):
        self.system_type = system_type

        # Material properties
        self.properties = self._default_properties()

    def _default_properties(self) -> Dict:
        """Default material properties."""

        return {
            'water': {
                'cp': 4186,      # J/kg·K
                'cv': 4186,      # J/kg·K (approx for liquid)
                'rho': 1000,     # kg/m³
                'mu': 0.001,     # Pa·s
                'k': 0.6,        # W/m·K
            },
            'air': {
                'cp': 1005,      # J/kg·K
                'cv': 718,       # J/kg·K
                'rho': 1.2,      # kg/m³
                'mu': 0.000018,  # Pa·s
                'k': 0.026,      # W/m·K
                'gamma': 1.4,
            },
            'combustion_gas': {
                'cp': 1100,
                'cv': 800,
                'gamma': 1.35,
            },
        }

    def analyze_entropy_production(self, data: Dict) -> Dict[str, Any]:
        """
        Analyze entropy production in the system.

        Entropy production indicates irreversibility and efficiency losses.
        """

        # Get temperatures
        t_hot = data.get('hot_temperature', data.get('T_hot', 500))  # K
        t_cold = data.get('cold_temperature', data.get('T_cold', 300))  # K
        t_ambient = data.get('ambient_temperature', 298)  # K

        # Get heat transfer rate
        q_transfer = data.get('heat_transfer_rate', data.get('Q', 0))  # W

        # Entropy production from heat transfer
        if t_hot > 0 and t_cold > 0 and t_hot != t_cold:
            s_production_heat = q_transfer * (1/t_cold - 1/t_hot)
        else:
            s_production_heat = 0

        # Entropy production from flow (friction, mixing)
        pressure_drop = data.get('pressure_drop', 0)  # Pa
        mass_flow = data.get('mass_flow', 0)  # kg/s
        density = data.get('density', 1000)  # kg/m³
        t_avg = (t_hot + t_cold) / 2

        if t_avg > 0 and density > 0:
            s_production_flow = mass_flow * pressure_drop / (density * t_avg)
        else:
            s_production_flow = 0

        # Total entropy production
        s_production_total = s_production_heat + s_production_flow

        # Entropy production rate (normalized)
        s_production_rate = s_production_total / max(abs(q_transfer), 1)

        return {
            'entropy_production_total': float(s_production_total),  # W/K
            'entropy_production_heat': float(s_production_heat),
            'entropy_production_flow': float(s_production_flow),
            'entropy_production_rate': float(s_production_rate),
            'irreversibility_indicator': float(min(s_production_rate * 1000, 1.0)),
            'second_law_efficiency': float(1 - s_production_rate * 100) if s_production_rate < 0.01 else 0,
        }

    def analyze_exergy(self, data: Dict) -> Dict[str, Any]:
        """
        Exergy (available work) analysis.

        Exergy destruction indicates wasted potential for useful work.
        """

        t_ambient = data.get('ambient_temperature', 298)  # K
        t_system = data.get('system_temperature', data.get('temperature', 400))  # K

        # Thermal exergy
        q_thermal = data.get('heat_rate', data.get('Q', 0))  # W
        if t_system > 0:
            exergy_thermal = q_thermal * (1 - t_ambient / t_system)
        else:
            exergy_thermal = 0

        # Mechanical exergy (pressure difference)
        p_system = data.get('pressure', 101325)  # Pa
        p_ambient = data.get('ambient_pressure', 101325)
        v_specific = 1 / data.get('density', 1000)  # m³/kg
        mass_flow = data.get('mass_flow', 0)

        exergy_mechanical = mass_flow * (p_system - p_ambient) * v_specific

        # Chemical exergy (simplified)
        exergy_chemical = data.get('chemical_exergy', 0)

        # Total exergy
        exergy_total = exergy_thermal + exergy_mechanical + exergy_chemical

        # Exergy destruction
        exergy_in = data.get('exergy_in', exergy_total)
        exergy_out = data.get('exergy_out', 0)
        exergy_destruction = exergy_in - exergy_out - data.get('work_output', 0)

        # Exergy efficiency
        if exergy_in > 0:
            exergy_efficiency = 1 - exergy_destruction / exergy_in
        else:
            exergy_efficiency = 0

        return {
            'exergy_thermal': float(exergy_thermal),
            'exergy_mechanical': float(exergy_mechanical),
            'exergy_chemical': float(exergy_chemical),
            'exergy_total': float(exergy_total),
            'exergy_destruction': float(max(0, exergy_destruction)),
            'exergy_efficiency': float(max(0, min(1, exergy_efficiency))),
        }

    def analyze_heat_transfer(self, data: Dict) -> Dict[str, Any]:
        """
        Heat transfer analysis.
        """

        # Get parameters
        t_hot = data.get('hot_temperature', 100)  # °C or K
        t_cold = data.get('cold_temperature', 20)
        area = data.get('heat_transfer_area', 1)  # m²
        h_coeff = data.get('heat_transfer_coefficient', 500)  # W/m²·K

        # Temperature difference
        delta_t = t_hot - t_cold

        # Heat transfer rate (Newton's law of cooling)
        q_convection = h_coeff * area * delta_t

        # Log mean temperature difference (for heat exchangers)
        t_hot_in = data.get('hot_inlet_temp', t_hot)
        t_hot_out = data.get('hot_outlet_temp', t_cold + 10)
        t_cold_in = data.get('cold_inlet_temp', t_cold)
        t_cold_out = data.get('cold_outlet_temp', t_hot - 10)

        delta_t1 = t_hot_in - t_cold_out
        delta_t2 = t_hot_out - t_cold_in

        if delta_t1 > 0 and delta_t2 > 0 and delta_t1 != delta_t2:
            lmtd = (delta_t1 - delta_t2) / np.log(delta_t1 / delta_t2)
        else:
            lmtd = delta_t

        # Effectiveness (for heat exchangers)
        c_min = data.get('c_min', 1000)  # W/K
        q_max = c_min * (t_hot_in - t_cold_in)
        q_actual = data.get('heat_transfer_rate', q_convection)

        if q_max > 0:
            effectiveness = q_actual / q_max
        else:
            effectiveness = 0

        return {
            'temperature_difference': float(delta_t),
            'lmtd': float(lmtd),
            'heat_transfer_rate': float(q_convection),
            'effectiveness': float(min(1, effectiveness)),
            'ntu': float(-np.log(1 - effectiveness)) if 0 < effectiveness < 1 else 0,
        }

    def calculate_isentropic_efficiency(
        self,
        device_type: str,
        data: Dict
    ) -> Dict[str, Any]:
        """
        Calculate isentropic efficiency for compressors, turbines, etc.
        """

        if device_type == 'compressor':
            return self._compressor_efficiency(data)
        elif device_type == 'turbine':
            return self._turbine_efficiency(data)
        elif device_type == 'pump':
            return self._pump_efficiency(data)
        else:
            return {'efficiency': 0, 'status': 'unknown_device'}

    def _compressor_efficiency(self, data: Dict) -> Dict[str, Any]:
        """Isentropic efficiency for compressor."""

        t_inlet = data.get('inlet_temperature', 300)  # K
        t_outlet = data.get('outlet_temperature', 400)  # K
        p_inlet = data.get('inlet_pressure', 101325)  # Pa
        p_outlet = data.get('outlet_pressure', 500000)  # Pa
        gamma = data.get('gamma', 1.4)

        # Isentropic outlet temperature
        pressure_ratio = p_outlet / p_inlet
        t_outlet_isentropic = t_inlet * (pressure_ratio ** ((gamma - 1) / gamma))

        # Isentropic efficiency
        if t_outlet > t_inlet:
            eta_isentropic = (t_outlet_isentropic - t_inlet) / (t_outlet - t_inlet)
        else:
            eta_isentropic = 0

        return {
            'isentropic_efficiency': float(min(1, max(0, eta_isentropic))),
            'pressure_ratio': float(pressure_ratio),
            't_outlet_ideal': float(t_outlet_isentropic),
            't_outlet_actual': float(t_outlet),
            'irreversibility': float(t_outlet - t_outlet_isentropic),
        }

    def _turbine_efficiency(self, data: Dict) -> Dict[str, Any]:
        """Isentropic efficiency for turbine."""

        t_inlet = data.get('inlet_temperature', 1000)  # K
        t_outlet = data.get('outlet_temperature', 600)  # K
        p_inlet = data.get('inlet_pressure', 1000000)  # Pa
        p_outlet = data.get('outlet_pressure', 101325)  # Pa
        gamma = data.get('gamma', 1.35)

        # Isentropic outlet temperature
        pressure_ratio = p_outlet / p_inlet
        t_outlet_isentropic = t_inlet * (pressure_ratio ** ((gamma - 1) / gamma))

        # Isentropic efficiency
        if t_inlet > t_outlet_isentropic:
            eta_isentropic = (t_inlet - t_outlet) / (t_inlet - t_outlet_isentropic)
        else:
            eta_isentropic = 0

        return {
            'isentropic_efficiency': float(min(1, max(0, eta_isentropic))),
            'pressure_ratio': float(pressure_ratio),
            't_outlet_ideal': float(t_outlet_isentropic),
            't_outlet_actual': float(t_outlet),
            'irreversibility': float(t_outlet - t_outlet_isentropic),
        }

    def _pump_efficiency(self, data: Dict) -> Dict[str, Any]:
        """Isentropic efficiency for pump."""

        p_inlet = data.get('inlet_pressure', 101325)  # Pa
        p_outlet = data.get('outlet_pressure', 500000)  # Pa
        flow_rate = data.get('flow_rate', 0.01)  # m³/s
        power_input = data.get('power_input', 1000)  # W
        density = data.get('density', 1000)  # kg/m³

        # Ideal (isentropic) power
        delta_p = p_outlet - p_inlet
        power_ideal = flow_rate * delta_p

        # Efficiency
        if power_input > 0:
            eta = power_ideal / power_input
        else:
            eta = 0

        return {
            'isentropic_efficiency': float(min(1, max(0, eta))),
            'pressure_rise': float(delta_p),
            'power_ideal': float(power_ideal),
            'power_actual': float(power_input),
            'power_loss': float(power_input - power_ideal),
        }

    def comprehensive_analysis(self, data: Dict) -> Dict[str, Any]:
        """Run comprehensive thermodynamic analysis."""

        entropy = self.analyze_entropy_production(data)
        exergy = self.analyze_exergy(data)
        heat_transfer = self.analyze_heat_transfer(data)

        # Overall thermodynamic health
        thermo_health = (
            0.3 * (1 - entropy['irreversibility_indicator']) +
            0.4 * exergy['exergy_efficiency'] +
            0.3 * heat_transfer['effectiveness']
        )

        return {
            'entropy_analysis': entropy,
            'exergy_analysis': exergy,
            'heat_transfer_analysis': heat_transfer,
            'thermodynamic_health_score': float(thermo_health),
            'status': 'healthy' if thermo_health > 0.7 else 'degraded' if thermo_health > 0.4 else 'critical',
        }
