"""
ORTHON Educational Calculation Engine

Generates educational problems from real ORTHON analyses, allowing students
to manually calculate what the system computes automatically.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
import hashlib
import time


@dataclass
class SolutionStep:
    """A step in the solution process."""
    step_number: int
    description: str
    formula: str
    expected_result: Any
    points: int
    hints: List[str] = field(default_factory=list)
    tolerance: float = 0.05  # 5% tolerance for numerical answers


@dataclass
class CalculationProblem:
    """A manually solvable problem based on real ORTHON analysis."""
    problem_id: str
    title: str
    description: str
    given_data: Dict[str, Any]
    expected_calculations: List[str]
    solution_steps: List[SolutionStep]
    difficulty_level: int  # 1-5
    learning_objectives: List[str]
    topic: str
    time_estimate_minutes: int = 15
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            'problem_id': self.problem_id,
            'title': self.title,
            'description': self.description,
            'given_data': self.given_data,
            'expected_calculations': self.expected_calculations,
            'solution_steps': [
                {
                    'step': s.step_number,
                    'description': s.description,
                    'formula': s.formula,
                    'points': s.points,
                    'hints': s.hints,
                }
                for s in self.solution_steps
            ],
            'difficulty_level': self.difficulty_level,
            'learning_objectives': self.learning_objectives,
            'topic': self.topic,
            'time_estimate_minutes': self.time_estimate_minutes,
        }


class EducationalCalculationEngine:
    """
    Generates educational problems from real ORTHON analyses.

    Students can manually verify every calculation ORTHON performs,
    connecting mathematical theory to industrial practice.
    """

    def __init__(self):
        self.problem_templates = self._initialize_templates()

    def _initialize_templates(self) -> Dict[str, Dict]:
        """Initialize problem templates for different topics."""

        return {
            'eigenstructure': {
                'title_template': 'Industrial System Eigenstructure Analysis',
                'learning_objectives': [
                    'Understand covariance matrix calculation',
                    'Perform eigenvalue decomposition',
                    'Connect eigenstructure to system health',
                    'Interpret effective dimension in industrial context',
                ],
            },
            'energy_balance': {
                'title_template': 'Industrial Process Energy Balance',
                'learning_objectives': [
                    'Apply conservation of energy to real systems',
                    'Calculate heat transfer requirements',
                    'Recognize physics constraint violations',
                    'Connect theory to industrial practice',
                ],
            },
            'mass_balance': {
                'title_template': 'Process Mass Balance Analysis',
                'learning_objectives': [
                    'Apply conservation of mass to flow systems',
                    'Account for chemical reactions in mass balances',
                    'Identify mass balance violations',
                ],
            },
            'entropy': {
                'title_template': 'Entropy and Efficiency Analysis',
                'learning_objectives': [
                    'Calculate entropy production in real processes',
                    'Understand irreversibility in industrial systems',
                    'Apply second law analysis',
                ],
            },
        }

    def generate_eigenstructure_problem(
        self,
        sensor_data: np.ndarray,
        difficulty: int = 2
    ) -> CalculationProblem:
        """
        Generate eigenstructure analysis problem from real sensor data.

        Students manually compute what PRISM calculates automatically.
        """

        # Select subset of data appropriate for hand calculation
        if difficulty <= 2:
            # Simple 3x3 covariance matrix
            n_samples = min(10, len(sensor_data))
            n_signals = min(3, sensor_data.shape[1] if len(sensor_data.shape) > 1 else 1)
        elif difficulty == 3:
            # 4x4 covariance matrix
            n_samples = min(15, len(sensor_data))
            n_signals = min(4, sensor_data.shape[1] if len(sensor_data.shape) > 1 else 1)
        else:
            # Larger matrix
            n_samples = min(20, len(sensor_data))
            n_signals = min(5, sensor_data.shape[1] if len(sensor_data.shape) > 1 else 1)

        if len(sensor_data.shape) == 1:
            # Convert 1D to 2D
            sensor_data = sensor_data.reshape(-1, 1)

        data_subset = sensor_data[:n_samples, :n_signals]

        # Calculate expected solutions
        means = np.mean(data_subset, axis=0)
        centered = data_subset - means
        cov_matrix = np.cov(data_subset.T)

        if cov_matrix.ndim == 0:
            cov_matrix = np.array([[cov_matrix]])

        eigenvals, eigenvecs = np.linalg.eigh(cov_matrix)
        eigenvals = eigenvals[::-1]  # Descending order
        eigenvecs = eigenvecs[:, ::-1]

        # Calculate PRISM metrics
        eigenvals_positive = eigenvals[eigenvals > 0]
        if len(eigenvals_positive) > 0:
            eff_dim = (np.sum(eigenvals_positive) ** 2) / np.sum(eigenvals_positive ** 2)
            participation_ratio = eff_dim / len(eigenvals_positive)
        else:
            eff_dim = 0
            participation_ratio = 0

        total_variance = np.sum(eigenvals)

        # Create problem ID
        data_hash = hashlib.md5(data_subset.tobytes()).hexdigest()[:8]
        problem_id = f"eigen_{difficulty}_{data_hash}"

        # Create solution steps
        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate the mean of each sensor column',
                formula='μⱼ = (1/n) Σᵢ xᵢⱼ',
                expected_result=means.tolist(),
                points=10,
                hints=['Sum all values in each column', 'Divide by number of samples'],
            ),
            SolutionStep(
                step_number=2,
                description='Center the data by subtracting means',
                formula='X̃ᵢⱼ = Xᵢⱼ - μⱼ',
                expected_result='Centered matrix',
                points=10,
                hints=['Subtract column mean from each value'],
            ),
            SolutionStep(
                step_number=3,
                description='Calculate the sample covariance matrix',
                formula='C = (1/(n-1)) X̃ᵀX̃',
                expected_result=cov_matrix.tolist(),
                points=20,
                hints=['Multiply centered data transpose by centered data',
                       'Divide by (n-1) for sample covariance'],
            ),
            SolutionStep(
                step_number=4,
                description='Find eigenvalues using characteristic equation',
                formula='det(C - λI) = 0',
                expected_result=eigenvals.tolist(),
                points=25,
                hints=['Set up the characteristic polynomial',
                       'Solve for roots (eigenvalues)'],
            ),
            SolutionStep(
                step_number=5,
                description='Calculate effective dimension (participation ratio)',
                formula='d_eff = (Σλᵢ)² / Σλᵢ²',
                expected_result=float(eff_dim),
                points=15,
                hints=['Sum all eigenvalues and square the result',
                       'Sum squares of all eigenvalues',
                       'Divide first by second'],
            ),
            SolutionStep(
                step_number=6,
                description='Interpret the results for system health',
                formula='N/A - Interpretation',
                expected_result=self._generate_eigenstructure_interpretation(eff_dim, eigenvals),
                points=20,
                hints=['Consider what low effective dimension means',
                       'Think about signal correlation'],
            ),
        ]

        return CalculationProblem(
            problem_id=problem_id,
            title='Industrial System Eigenstructure Analysis',
            description=f'''
You are analyzing sensor data from an industrial monitoring system.
The following {n_samples} measurements were taken from {n_signals} sensors.

This is the same analysis that ORTHON/PRISM performs automatically to assess
system health through geometric stability.

Data Matrix (rows = samples, columns = sensors):
{self._format_matrix(data_subset)}

Perform the complete eigenstructure analysis.
            '''.strip(),
            given_data={
                'sensor_data': data_subset.tolist(),
                'num_samples': n_samples,
                'num_sensors': n_signals,
            },
            expected_calculations=[
                'Calculate column means',
                'Center the data matrix',
                'Calculate covariance matrix',
                'Find eigenvalues and eigenvectors',
                'Calculate effective dimension',
                'Interpret system health status',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Understand covariance matrix calculation',
                'Perform eigenvalue decomposition by hand',
                'Connect eigenstructure to system health',
                'Interpret effective dimension in industrial context',
            ],
            topic='eigenstructure',
            time_estimate_minutes=20 + difficulty * 5,
        )

    def generate_energy_balance_problem(
        self,
        process_data: Dict,
        difficulty: int = 2
    ) -> CalculationProblem:
        """Generate energy balance calculation problem."""

        if difficulty <= 2:
            # Simple steady-state energy balance
            t_in = process_data.get('inlet_temp', 25.0)
            t_out = process_data.get('outlet_temp', 85.0)
            flow_rate = process_data.get('flow_rate', 10.0)
            heat_capacity = 4.18  # kJ/kg·K for water

            heat_required = flow_rate * heat_capacity * (t_out - t_in)
            heat_loss = 0
        else:
            # More complex with heat loss
            t_in = process_data.get('inlet_temp', 25.0)
            t_out = process_data.get('outlet_temp', 85.0)
            flow_rate = process_data.get('flow_rate', 10.0)
            heat_capacity = process_data.get('heat_capacity', 4.18)
            heat_loss_rate = process_data.get('heat_loss', 5.0)  # kW

            heat_required = flow_rate * heat_capacity * (t_out - t_in) + heat_loss_rate
            heat_loss = heat_loss_rate

        data_hash = hashlib.md5(str(process_data).encode()).hexdigest()[:8]
        problem_id = f"energy_{difficulty}_{data_hash}"

        solution_steps = [
            SolutionStep(
                step_number=1,
                description='Calculate temperature change',
                formula='ΔT = T_out - T_in',
                expected_result=float(t_out - t_in),
                points=10,
            ),
            SolutionStep(
                step_number=2,
                description='Calculate sensible heat requirement',
                formula='Q = ṁ × cp × ΔT',
                expected_result=float(flow_rate * heat_capacity * (t_out - t_in)),
                points=30,
                hints=['ṁ = mass flow rate', 'cp = specific heat capacity'],
            ),
            SolutionStep(
                step_number=3,
                description='Account for heat losses',
                formula='Q_total = Q_sensible + Q_loss',
                expected_result=float(heat_required),
                points=20,
            ),
            SolutionStep(
                step_number=4,
                description='Verify energy conservation',
                formula='Energy_in = Energy_out',
                expected_result='Balance verified' if heat_loss == 0 else f'Heat loss of {heat_loss} kW accounted',
                points=20,
            ),
            SolutionStep(
                step_number=5,
                description='Assess physics constraint status',
                formula='N/A - Assessment',
                expected_result=self._generate_energy_interpretation(heat_required),
                points=20,
            ),
        ]

        description = f'''
You are designing the heating system for a chemical process.

Given:
- Inlet temperature: {t_in}°C
- Outlet temperature: {t_out}°C
- Mass flow rate: {flow_rate} kg/s
- Heat capacity: {heat_capacity} kJ/kg·K'''

        if difficulty > 2:
            description += f'\n- Heat loss to surroundings: {heat_loss} kW'

        description += '\n\nCalculate the energy balance and verify conservation laws.'

        return CalculationProblem(
            problem_id=problem_id,
            title='Industrial Process Energy Balance',
            description=description.strip(),
            given_data={
                'T_inlet': t_in,
                'T_outlet': t_out,
                'mass_flow': flow_rate,
                'heat_capacity': heat_capacity,
                'heat_loss': heat_loss if difficulty > 2 else None,
            },
            expected_calculations=[
                'Calculate temperature change',
                'Calculate sensible heat requirement',
                'Account for heat losses',
                'Verify energy conservation',
            ],
            solution_steps=solution_steps,
            difficulty_level=difficulty,
            learning_objectives=[
                'Apply conservation of energy to real systems',
                'Calculate heat transfer requirements',
                'Recognize physics constraint violations',
                'Connect theory to industrial practice',
            ],
            topic='energy_balance',
            time_estimate_minutes=15 + difficulty * 3,
        )

    def verify_student_solution(
        self,
        problem: CalculationProblem,
        student_solution: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Verify student's manual calculations against expected solutions."""

        results = {
            'total_score': 0,
            'max_possible_score': sum(step.points for step in problem.solution_steps),
            'step_results': [],
            'overall_feedback': '',
            'conceptual_understanding': 0.0,
        }

        for step in problem.solution_steps:
            step_key = f'step_{step.step_number}'
            student_answer = student_solution.get(step_key)

            step_result = self._verify_step(step, student_answer)
            results['step_results'].append(step_result)
            results['total_score'] += step_result['points_earned']

        # Calculate score percentage
        score_pct = results['total_score'] / results['max_possible_score']

        # Overall feedback
        if score_pct > 0.9:
            results['overall_feedback'] = "Excellent! You demonstrate deep understanding of the calculations."
        elif score_pct > 0.7:
            results['overall_feedback'] = "Good work. Review the steps where you lost points."
        elif score_pct > 0.5:
            results['overall_feedback'] = "Fair attempt. Focus on understanding the underlying concepts."
        else:
            results['overall_feedback'] = "Needs improvement. Review the mathematical fundamentals."

        results['conceptual_understanding'] = self._assess_conceptual_understanding(
            problem, student_solution, results['step_results']
        )

        return results

    def _verify_step(self, step: SolutionStep, student_answer: Any) -> Dict:
        """Verify a single solution step."""

        result = {
            'step_number': step.step_number,
            'description': step.description,
            'max_points': step.points,
            'points_earned': 0,
            'feedback': '',
            'correct': False,
        }

        if student_answer is None:
            result['feedback'] = 'No answer provided'
            return result

        expected = step.expected_result

        # Handle different types of expected results
        if isinstance(expected, (int, float)):
            if isinstance(student_answer, (int, float)):
                error = abs(student_answer - expected) / max(abs(expected), 1e-6)
                if error <= step.tolerance:
                    result['correct'] = True
                    result['points_earned'] = step.points
                    result['feedback'] = 'Correct!'
                elif error <= step.tolerance * 2:
                    result['points_earned'] = int(step.points * 0.7)
                    result['feedback'] = f'Close. Expected {expected:.4g}, got {student_answer:.4g}'
                else:
                    result['feedback'] = f'Incorrect. Expected {expected:.4g}'

        elif isinstance(expected, list):
            if isinstance(student_answer, list):
                try:
                    expected_arr = np.array(expected)
                    student_arr = np.array(student_answer)
                    if expected_arr.shape == student_arr.shape:
                        error = np.max(np.abs(expected_arr - student_arr) / (np.abs(expected_arr) + 1e-6))
                        if error <= step.tolerance:
                            result['correct'] = True
                            result['points_earned'] = step.points
                            result['feedback'] = 'Correct!'
                        else:
                            result['feedback'] = 'Values differ from expected'
                    else:
                        result['feedback'] = f'Wrong dimensions. Expected {expected_arr.shape}'
                except:
                    result['feedback'] = 'Could not compare arrays'

        elif isinstance(expected, str):
            # For interpretation questions, check for key concepts
            if isinstance(student_answer, str):
                key_concepts = self._extract_key_concepts(expected)
                matched = sum(1 for c in key_concepts if c.lower() in student_answer.lower())
                if matched >= len(key_concepts) * 0.6:
                    result['correct'] = True
                    result['points_earned'] = step.points
                    result['feedback'] = 'Good interpretation!'
                elif matched > 0:
                    result['points_earned'] = int(step.points * matched / len(key_concepts))
                    result['feedback'] = 'Partial credit - missing some key concepts'
                else:
                    result['feedback'] = 'Review the interpretation guidelines'

        return result

    def _assess_conceptual_understanding(
        self,
        problem: CalculationProblem,
        student_solution: Dict,
        step_results: List[Dict]
    ) -> float:
        """Assess conceptual understanding beyond just numerical accuracy."""

        # Weight later steps more heavily (they build on earlier understanding)
        weighted_scores = []
        for i, result in enumerate(step_results):
            weight = 1.0 + 0.2 * i  # Later steps have higher weight
            score = result['points_earned'] / max(result['max_points'], 1)
            weighted_scores.append(weight * score)

        if weighted_scores:
            return sum(weighted_scores) / sum(1.0 + 0.2 * i for i in range(len(weighted_scores)))
        return 0.0

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from expected answer."""

        # Simple keyword extraction
        keywords = ['dimensional', 'collapse', 'degradation', 'coupling', 'correlation',
                    'healthy', 'warning', 'critical', 'stable', 'unstable', 'balance',
                    'conservation', 'efficiency', 'loss']
        return [k for k in keywords if k in text.lower()]

    def _generate_eigenstructure_interpretation(
        self,
        eff_dim: float,
        eigenvals: np.ndarray
    ) -> str:
        """Generate interpretation text for eigenstructure results."""

        if eff_dim < 1.5:
            health = "CRITICAL - Dimensional collapse detected"
            detail = "Signals are highly correlated, indicating potential system failure"
        elif eff_dim < 2.0:
            health = "WARNING - Low dimensional complexity"
            detail = "Signal coupling is elevated, monitor closely"
        elif eff_dim < 3.0:
            health = "MARGINAL - Below optimal complexity"
            detail = "Some signal correlation present"
        else:
            health = "HEALTHY - Normal dimensional complexity"
            detail = "Signals show expected independence"

        eigenval_ratio = eigenvals[0] / eigenvals[1] if len(eigenvals) > 1 and eigenvals[1] > 0 else 0

        return f"{health}. {detail}. Eigenvalue ratio: {eigenval_ratio:.2f}."

    def _generate_energy_interpretation(self, heat_required: float) -> str:
        """Generate interpretation for energy balance results."""

        if heat_required < 0:
            return "System is cooling - heat removal required"
        elif heat_required < 100:
            return "Moderate heating load - standard equipment sufficient"
        else:
            return "High heating load - consider heat recovery options"

    def _format_matrix(self, matrix: np.ndarray) -> str:
        """Format matrix for readable display."""

        if len(matrix.shape) == 1:
            return '  '.join(f'{val:8.3f}' for val in matrix)

        return '\n'.join([
            '  '.join(f'{val:8.3f}' for val in row)
            for row in matrix
        ])

    def generate_homework_set(
        self,
        live_data: Dict,
        num_problems: int = 5,
        difficulty_range: Tuple[int, int] = (2, 4)
    ) -> List[CalculationProblem]:
        """Generate a complete homework set from live system data."""

        problems = []

        # Available problem types
        problem_types = ['eigenstructure', 'energy_balance']

        for i in range(num_problems):
            problem_type = problem_types[i % len(problem_types)]
            difficulty = np.random.randint(difficulty_range[0], difficulty_range[1] + 1)

            if problem_type == 'eigenstructure':
                sensor_data = live_data.get('sensor_matrix')
                if sensor_data is not None:
                    problem = self.generate_eigenstructure_problem(
                        np.array(sensor_data), difficulty
                    )
                    problems.append(problem)

            elif problem_type == 'energy_balance':
                process_data = live_data.get('process_data', {})
                problem = self.generate_energy_balance_problem(
                    process_data, difficulty
                )
                problems.append(problem)

        return problems
