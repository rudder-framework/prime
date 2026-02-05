"""
ORTHON Educational Module

Framework for generating educational problems from live industrial data,
enabling students to manually verify calculations that ORTHON performs automatically.
"""

from .calculation_engine import EducationalCalculationEngine, CalculationProblem
from .problem_generator import ProblemGenerator
from .solution_checker import SolutionChecker

__all__ = [
    'EducationalCalculationEngine',
    'CalculationProblem',
    'ProblemGenerator',
    'SolutionChecker',
]
