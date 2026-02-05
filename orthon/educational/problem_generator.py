"""
ORTHON Problem Generator

Generates educational problems from live streaming data and historical analyses.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import hashlib

from .calculation_engine import CalculationProblem, EducationalCalculationEngine


@dataclass
class ProblemSet:
    """A collection of related problems."""
    set_id: str
    title: str
    description: str
    problems: List[CalculationProblem]
    total_points: int
    time_estimate_minutes: int
    difficulty_distribution: Dict[int, int]
    topics_covered: List[str]
    created_at: float


class ProblemGenerator:
    """
    Generates educational problems from live ORTHON data.

    Can create:
    - Individual problems from snapshots
    - Problem sets from time series data
    - Comparative problems (before/after events)
    - Scenario-based problems
    """

    def __init__(self):
        self.engine = EducationalCalculationEngine()
        self.generated_count = 0

    def generate_from_live_data(
        self,
        geometric_state: Dict,
        physical_state: Dict,
        difficulty: int = 2
    ) -> CalculationProblem:
        """Generate a problem from current live system state."""

        # Combine states into problem data
        combined_state = {**geometric_state, **physical_state}

        # Determine problem type based on available data
        if 'eigenvalues' in geometric_state or 'eff_dim' in geometric_state:
            # Create eigenstructure problem
            if 'eigenvalues' in geometric_state:
                # Reconstruct a simple sensor matrix from eigenvalues
                eigenvals = np.array(geometric_state['eigenvalues'])
                n = len(eigenvals)
                # Create synthetic correlated data
                sensor_matrix = self._generate_synthetic_data(eigenvals, n_samples=10 + difficulty * 5)
            else:
                # Use simple synthetic data
                sensor_matrix = np.random.randn(15, 4)

            return self.engine.generate_eigenstructure_problem(sensor_matrix, difficulty)

        elif any(k in physical_state for k in ['temperature', 'flow_rate', 'energy_balance_error']):
            # Create energy balance problem
            process_data = {
                'inlet_temp': physical_state.get('inlet_temp', physical_state.get('temperature', 25) - 60),
                'outlet_temp': physical_state.get('outlet_temp', physical_state.get('temperature', 85)),
                'flow_rate': physical_state.get('flow_rate', physical_state.get('mass_flow', 10)),
                'heat_capacity': 4.18,
            }
            return self.engine.generate_energy_balance_problem(process_data, difficulty)

        else:
            # Default to eigenstructure with synthetic data
            sensor_matrix = np.random.randn(15, 4)
            return self.engine.generate_eigenstructure_problem(sensor_matrix, difficulty)

    def generate_problem_set(
        self,
        data_history: List[Dict],
        title: str = "Industrial Analysis Problem Set",
        num_problems: int = 5,
        difficulty_range: Tuple[int, int] = (2, 4)
    ) -> ProblemSet:
        """Generate a complete problem set from historical data."""

        problems = []
        topics = set()

        for i in range(min(num_problems, len(data_history))):
            data = data_history[i % len(data_history)]
            difficulty = np.random.randint(difficulty_range[0], difficulty_range[1] + 1)

            geometric = data.get('geometric', data.get('instant_results', {}))
            physical = data.get('physical', data.get('batch_results', {}))

            problem = self.generate_from_live_data(geometric, physical, difficulty)
            problems.append(problem)
            topics.add(problem.topic)

        # Calculate totals
        total_points = sum(
            sum(s.points for s in p.solution_steps)
            for p in problems
        )
        total_time = sum(p.time_estimate_minutes for p in problems)

        # Difficulty distribution
        difficulty_dist = {}
        for p in problems:
            difficulty_dist[p.difficulty_level] = difficulty_dist.get(p.difficulty_level, 0) + 1

        set_id = hashlib.md5(f"{title}_{time.time()}".encode()).hexdigest()[:12]

        return ProblemSet(
            set_id=set_id,
            title=title,
            description=f"Problem set with {num_problems} problems covering {', '.join(topics)}",
            problems=problems,
            total_points=total_points,
            time_estimate_minutes=total_time,
            difficulty_distribution=difficulty_dist,
            topics_covered=list(topics),
            created_at=time.time(),
        )

    def generate_comparative_problem(
        self,
        before_state: Dict,
        after_state: Dict,
        event_description: str,
        difficulty: int = 3
    ) -> CalculationProblem:
        """
        Generate a problem comparing system states before and after an event.

        Great for understanding how geometric/physical metrics change.
        """

        # Extract key metrics
        before_eff_dim = before_state.get('eff_dim', before_state.get('effective_dimension', 3.0))
        after_eff_dim = after_state.get('eff_dim', after_state.get('effective_dimension', 2.0))

        before_efficiency = before_state.get('system_efficiency', 0.85)
        after_efficiency = after_state.get('system_efficiency', 0.70)

        problem_id = f"compare_{hashlib.md5(event_description.encode()).hexdigest()[:8]}"

        from .calculation_engine import SolutionStep

        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate the change in effective dimension',
                formula='Δd_eff = d_eff_after - d_eff_before',
                expected_result=float(after_eff_dim - before_eff_dim),
                points=15,
            ),
            SolutionStep(
                step_number=2,
                description='Calculate percentage change in effective dimension',
                formula='%Δd_eff = (Δd_eff / d_eff_before) × 100%',
                expected_result=float((after_eff_dim - before_eff_dim) / before_eff_dim * 100),
                points=15,
            ),
            SolutionStep(
                step_number=3,
                description='Calculate efficiency change',
                formula='Δη = η_after - η_before',
                expected_result=float(after_efficiency - before_efficiency),
                points=15,
            ),
            SolutionStep(
                step_number=4,
                description='Assess correlation between geometric and physical changes',
                formula='Both metrics declined → coupled degradation',
                expected_result='Geometric collapse correlated with efficiency loss',
                points=25,
            ),
            SolutionStep(
                step_number=5,
                description='Recommend operational response',
                formula='N/A - Engineering judgment',
                expected_result=self._generate_recommendation(after_eff_dim, after_efficiency),
                points=30,
            ),
        ]

        return CalculationProblem(
            problem_id=problem_id,
            title='System State Comparison Analysis',
            description=f'''
An industrial system experienced the following event:
"{event_description}"

Analyze the change in system state:

BEFORE Event:
- Effective dimension: {before_eff_dim:.3f}
- System efficiency: {before_efficiency:.1%}

AFTER Event:
- Effective dimension: {after_eff_dim:.3f}
- System efficiency: {after_efficiency:.1%}

Calculate the changes and assess the system health impact.
            '''.strip(),
            given_data={
                'before': {
                    'eff_dim': before_eff_dim,
                    'efficiency': before_efficiency,
                },
                'after': {
                    'eff_dim': after_eff_dim,
                    'efficiency': after_efficiency,
                },
                'event': event_description,
            },
            expected_calculations=[
                'Calculate effective dimension change',
                'Calculate efficiency change',
                'Assess correlation between metrics',
                'Recommend operational response',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Understand relationships between geometric and physical metrics',
                'Analyze system state transitions',
                'Connect metric changes to operational impacts',
                'Develop engineering judgment for responses',
            ],
            topic='comparative_analysis',
            time_estimate_minutes=20,
        )

    def generate_scenario_problem(
        self,
        scenario_type: str,
        parameters: Optional[Dict] = None,
        difficulty: int = 3
    ) -> CalculationProblem:
        """
        Generate a scenario-based problem.

        Scenarios:
        - 'turbofan_degradation': Engine performance decline
        - 'reactor_runaway': Chemical reactor temperature excursion
        - 'dimensional_collapse': Sudden loss of signal independence
        """

        parameters = parameters or {}

        if scenario_type == 'turbofan_degradation':
            return self._turbofan_scenario(parameters, difficulty)
        elif scenario_type == 'reactor_runaway':
            return self._reactor_scenario(parameters, difficulty)
        elif scenario_type == 'dimensional_collapse':
            return self._collapse_scenario(parameters, difficulty)
        else:
            # Default scenario
            return self._generic_scenario(parameters, difficulty)

    def _turbofan_scenario(self, params: Dict, difficulty: int) -> CalculationProblem:
        """Generate turbofan degradation scenario."""

        from .calculation_engine import SolutionStep

        # Scenario parameters
        epr_initial = params.get('epr_initial', 1.30)
        epr_current = params.get('epr_current', 1.22)
        thrust_loss = params.get('thrust_loss', 8)  # percent
        eff_dim = params.get('eff_dim', 2.1)

        problem_id = f"turbofan_{hashlib.md5(str(params).encode()).hexdigest()[:8]}"

        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate EPR degradation',
                formula='ΔEPR = EPR_current - EPR_initial',
                expected_result=float(epr_current - epr_initial),
                points=15,
            ),
            SolutionStep(
                step_number=2,
                description='Calculate EPR degradation percentage',
                formula='%ΔEPR = (ΔEPR / EPR_initial) × 100%',
                expected_result=float((epr_current - epr_initial) / epr_initial * 100),
                points=15,
            ),
            SolutionStep(
                step_number=3,
                description='Estimate remaining useful life using degradation rate',
                formula='RUL = (EPR_limit - EPR_current) / degradation_rate',
                expected_result='Estimate based on trend analysis',
                points=25,
            ),
            SolutionStep(
                step_number=4,
                description='Assess dimensional collapse risk',
                formula='Risk = HIGH if d_eff < 2.0',
                expected_result='HIGH' if eff_dim < 2.0 else 'MODERATE' if eff_dim < 2.5 else 'LOW',
                points=20,
            ),
            SolutionStep(
                step_number=5,
                description='Recommend maintenance action',
                formula='N/A - Engineering decision',
                expected_result='Schedule inspection' if eff_dim > 2.0 else 'Immediate grounding recommended',
                points=25,
            ),
        ]

        return CalculationProblem(
            problem_id=problem_id,
            title='Turbofan Engine Degradation Analysis',
            description=f'''
A commercial aircraft turbofan engine is showing signs of degradation.

Current measurements:
- Engine Pressure Ratio (EPR): {epr_current} (initial: {epr_initial})
- Thrust loss: {thrust_loss}%
- PRISM Effective Dimension: {eff_dim:.2f}

The effective dimension indicates the geometric health of the sensor array.
Values below 2.0 suggest dimensional collapse and imminent failure.

Analyze the degradation and recommend action.
            '''.strip(),
            given_data={
                'epr_initial': epr_initial,
                'epr_current': epr_current,
                'thrust_loss': thrust_loss,
                'eff_dim': eff_dim,
            },
            expected_calculations=[
                'Calculate EPR degradation',
                'Estimate remaining useful life',
                'Assess collapse risk',
                'Recommend maintenance action',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Apply degradation analysis to real aircraft engines',
                'Connect geometric metrics to physical performance',
                'Make engineering decisions under uncertainty',
            ],
            topic='turbofan_analysis',
            time_estimate_minutes=25,
        )

    def _reactor_scenario(self, params: Dict, difficulty: int) -> CalculationProblem:
        """Generate chemical reactor scenario."""

        from .calculation_engine import SolutionStep

        t_current = params.get('temperature', 380)
        t_setpoint = params.get('setpoint', 350)
        cooling_rate = params.get('cooling_rate', 50)  # kW
        reaction_rate = params.get('reaction_rate', 0.5)  # mol/s

        problem_id = f"reactor_{hashlib.md5(str(params).encode()).hexdigest()[:8]}"

        heat_generation = reaction_rate * 50  # kW (assuming 50 kJ/mol reaction heat)
        net_heat = heat_generation - cooling_rate

        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate heat generation from reaction',
                formula='Q_gen = r × ΔH_rxn',
                expected_result=float(heat_generation),
                points=20,
            ),
            SolutionStep(
                step_number=2,
                description='Calculate net heat accumulation',
                formula='Q_net = Q_gen - Q_cooling',
                expected_result=float(net_heat),
                points=20,
            ),
            SolutionStep(
                step_number=3,
                description='Assess runaway risk',
                formula='Risk = CRITICAL if Q_net > 0 and T > T_setpoint',
                expected_result='CRITICAL' if net_heat > 0 and t_current > t_setpoint else 'CONTROLLED',
                points=25,
            ),
            SolutionStep(
                step_number=4,
                description='Calculate required cooling increase',
                formula='ΔQ_cool = Q_net + safety_margin',
                expected_result=float(max(0, net_heat + 10)),
                points=20,
            ),
            SolutionStep(
                step_number=5,
                description='Recommend emergency action',
                formula='N/A - Safety decision',
                expected_result='Increase cooling' if net_heat > 0 else 'Monitor',
                points=15,
            ),
        ]

        return CalculationProblem(
            problem_id=problem_id,
            title='Chemical Reactor Thermal Runaway Analysis',
            description=f'''
A chemical reactor is experiencing elevated temperatures.

Current state:
- Reactor temperature: {t_current} K (setpoint: {t_setpoint} K)
- Cooling rate: {cooling_rate} kW
- Reaction rate: {reaction_rate} mol/s
- Reaction heat: 50 kJ/mol (exothermic)

Analyze the thermal balance and assess runaway risk.
            '''.strip(),
            given_data={
                'temperature': t_current,
                'setpoint': t_setpoint,
                'cooling_rate': cooling_rate,
                'reaction_rate': reaction_rate,
                'reaction_heat': 50,
            },
            expected_calculations=[
                'Calculate heat generation',
                'Calculate net heat accumulation',
                'Assess runaway risk',
                'Recommend action',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Apply energy balance to reactive systems',
                'Assess thermal runaway risk',
                'Make safety-critical decisions',
            ],
            topic='reactor_analysis',
            time_estimate_minutes=20,
        )

    def _collapse_scenario(self, params: Dict, difficulty: int) -> CalculationProblem:
        """Generate dimensional collapse scenario."""

        from .calculation_engine import SolutionStep

        eigenvals = params.get('eigenvalues', [10.0, 2.0, 0.5, 0.1])
        eigenvals = np.array(eigenvals)

        eff_dim = (np.sum(eigenvals) ** 2) / np.sum(eigenvals ** 2)
        total_var = np.sum(eigenvals)
        var_explained_1 = eigenvals[0] / total_var * 100

        problem_id = f"collapse_{hashlib.md5(str(params).encode()).hexdigest()[:8]}"

        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate effective dimension',
                formula='d_eff = (Σλ)² / Σλ²',
                expected_result=float(eff_dim),
                points=25,
            ),
            SolutionStep(
                step_number=2,
                description='Calculate variance explained by first component',
                formula='%var₁ = λ₁ / Σλ × 100%',
                expected_result=float(var_explained_1),
                points=20,
            ),
            SolutionStep(
                step_number=3,
                description='Assess collapse severity',
                formula='CRITICAL if d_eff < 1.5',
                expected_result='CRITICAL' if eff_dim < 1.5 else 'WARNING' if eff_dim < 2.0 else 'NORMAL',
                points=25,
            ),
            SolutionStep(
                step_number=4,
                description='Interpret physical meaning',
                formula='N/A - Interpretation',
                expected_result='Sensors becoming highly correlated - system losing degrees of freedom',
                points=30,
            ),
        ]

        return CalculationProblem(
            problem_id=problem_id,
            title='Dimensional Collapse Analysis',
            description=f'''
A monitoring system has detected potential dimensional collapse.

Eigenvalues from the sensor covariance matrix:
{eigenvals.tolist()}

Analyze the eigenstructure to assess collapse severity.
            '''.strip(),
            given_data={'eigenvalues': eigenvals.tolist()},
            expected_calculations=[
                'Calculate effective dimension',
                'Calculate variance distribution',
                'Assess collapse severity',
                'Interpret physical meaning',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Understand dimensional collapse',
                'Connect eigenstructure to system health',
                'Interpret geometric metrics physically',
            ],
            topic='collapse_analysis',
            time_estimate_minutes=15,
        )

    def _generic_scenario(self, params: Dict, difficulty: int) -> CalculationProblem:
        """Generate generic analysis scenario."""

        sensor_data = np.random.randn(15, 4)
        return self.engine.generate_eigenstructure_problem(sensor_data, difficulty)

    def _generate_synthetic_data(self, eigenvals: np.ndarray, n_samples: int) -> np.ndarray:
        """Generate synthetic sensor data with specified eigenvalue structure."""

        n_features = len(eigenvals)

        # Create orthogonal basis
        Q, _ = np.linalg.qr(np.random.randn(n_features, n_features))

        # Create covariance with desired eigenvalues
        cov = Q @ np.diag(eigenvals) @ Q.T

        # Generate samples
        mean = np.zeros(n_features)
        data = np.random.multivariate_normal(mean, cov, size=n_samples)

        return data

    def _generate_recommendation(self, eff_dim: float, efficiency: float) -> str:
        """Generate operational recommendation."""

        if eff_dim < 1.5 or efficiency < 0.5:
            return "IMMEDIATE: Initiate emergency shutdown protocol"
        elif eff_dim < 2.0 or efficiency < 0.7:
            return "URGENT: Schedule maintenance within 24 hours"
        elif eff_dim < 2.5 or efficiency < 0.8:
            return "MONITOR: Increase monitoring frequency"
        else:
            return "NORMAL: Continue standard operation"
