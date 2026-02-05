"""
ORTHON Solution Checker

Verifies student solutions and provides detailed feedback.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import re

from .calculation_engine import CalculationProblem, SolutionStep


@dataclass
class GradingResult:
    """Result of grading a solution."""
    total_score: int
    max_score: int
    percentage: float
    grade_letter: str
    step_results: List[Dict]
    feedback: str
    strengths: List[str]
    areas_for_improvement: List[str]
    conceptual_score: float


class SolutionChecker:
    """
    Verifies student solutions against expected answers.

    Provides:
    - Numerical answer checking with tolerance
    - Partial credit for close answers
    - Conceptual understanding assessment
    - Detailed feedback for learning
    """

    def __init__(self, tolerance: float = 0.05):
        self.default_tolerance = tolerance

        # Grade boundaries
        self.grade_boundaries = {
            'A': 90,
            'B': 80,
            'C': 70,
            'D': 60,
            'F': 0,
        }

    def check_solution(
        self,
        problem: CalculationProblem,
        student_answers: Dict[str, Any]
    ) -> GradingResult:
        """
        Check a complete solution against expected answers.

        Args:
            problem: The problem being solved
            student_answers: Dict mapping step numbers to student answers
                e.g., {'step_1': 42.5, 'step_2': [1, 2, 3], ...}

        Returns:
            GradingResult with detailed feedback
        """

        step_results = []
        total_score = 0
        max_score = 0
        strengths = []
        improvements = []

        for step in problem.solution_steps:
            step_key = f'step_{step.step_number}'
            student_answer = student_answers.get(step_key)

            result = self._check_step(step, student_answer)
            step_results.append(result)

            total_score += result['points_earned']
            max_score += step.points

            # Track strengths and improvements
            if result['correct']:
                strengths.append(f"Step {step.step_number}: {step.description}")
            elif result['points_earned'] > 0:
                improvements.append(f"Step {step.step_number}: Partially correct - {result['feedback']}")
            else:
                improvements.append(f"Step {step.step_number}: {result['feedback']}")

        # Calculate percentage and grade
        percentage = (total_score / max_score * 100) if max_score > 0 else 0
        grade_letter = self._calculate_grade(percentage)

        # Calculate conceptual score
        conceptual_score = self._assess_conceptual_understanding(
            problem, student_answers, step_results
        )

        # Generate overall feedback
        feedback = self._generate_feedback(percentage, conceptual_score, step_results)

        return GradingResult(
            total_score=total_score,
            max_score=max_score,
            percentage=percentage,
            grade_letter=grade_letter,
            step_results=step_results,
            feedback=feedback,
            strengths=strengths[:3],  # Top 3 strengths
            areas_for_improvement=improvements[:3],  # Top 3 areas
            conceptual_score=conceptual_score,
        )

    def _check_step(self, step: SolutionStep, student_answer: Any) -> Dict:
        """Check a single solution step."""

        result = {
            'step_number': step.step_number,
            'description': step.description,
            'max_points': step.points,
            'points_earned': 0,
            'correct': False,
            'feedback': '',
            'expected': step.expected_result,
            'received': student_answer,
        }

        if student_answer is None:
            result['feedback'] = 'No answer provided'
            return result

        expected = step.expected_result
        tolerance = step.tolerance

        # Check based on type
        if isinstance(expected, (int, float)):
            result.update(self._check_numeric(expected, student_answer, step.points, tolerance))

        elif isinstance(expected, list):
            result.update(self._check_array(expected, student_answer, step.points, tolerance))

        elif isinstance(expected, str):
            result.update(self._check_text(expected, student_answer, step.points))

        elif isinstance(expected, dict):
            result.update(self._check_dict(expected, student_answer, step.points, tolerance))

        else:
            result['feedback'] = 'Unknown answer type'

        return result

    def _check_numeric(
        self,
        expected: float,
        received: Any,
        max_points: int,
        tolerance: float
    ) -> Dict:
        """Check a numeric answer."""

        if not isinstance(received, (int, float)):
            try:
                received = float(received)
            except (ValueError, TypeError):
                return {
                    'points_earned': 0,
                    'correct': False,
                    'feedback': 'Answer must be a number',
                }

        # Calculate relative error
        if abs(expected) > 1e-10:
            error = abs(received - expected) / abs(expected)
        else:
            error = abs(received - expected)

        if error <= tolerance:
            return {
                'points_earned': max_points,
                'correct': True,
                'feedback': 'Correct!',
            }
        elif error <= tolerance * 2:
            points = int(max_points * 0.7)
            return {
                'points_earned': points,
                'correct': False,
                'feedback': f'Close! Expected {expected:.4g}, got {received:.4g}. Error: {error:.1%}',
            }
        elif error <= tolerance * 5:
            points = int(max_points * 0.3)
            return {
                'points_earned': points,
                'correct': False,
                'feedback': f'Partially correct. Expected {expected:.4g}, got {received:.4g}',
            }
        else:
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': f'Incorrect. Expected {expected:.4g}, got {received:.4g}',
            }

    def _check_array(
        self,
        expected: List,
        received: Any,
        max_points: int,
        tolerance: float
    ) -> Dict:
        """Check an array/list answer."""

        if not isinstance(received, (list, np.ndarray)):
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': 'Answer must be a list or array',
            }

        try:
            expected_arr = np.array(expected)
            received_arr = np.array(received)

            if expected_arr.shape != received_arr.shape:
                return {
                    'points_earned': int(max_points * 0.1),
                    'correct': False,
                    'feedback': f'Wrong shape. Expected {expected_arr.shape}, got {received_arr.shape}',
                }

            # Element-wise comparison
            errors = np.abs(expected_arr - received_arr) / (np.abs(expected_arr) + 1e-10)
            max_error = np.max(errors)
            mean_error = np.mean(errors)

            if max_error <= tolerance:
                return {
                    'points_earned': max_points,
                    'correct': True,
                    'feedback': 'Correct!',
                }
            elif mean_error <= tolerance:
                points = int(max_points * 0.8)
                return {
                    'points_earned': points,
                    'correct': False,
                    'feedback': f'Mostly correct. Some elements have errors up to {max_error:.1%}',
                }
            elif mean_error <= tolerance * 3:
                points = int(max_points * 0.5)
                return {
                    'points_earned': points,
                    'correct': False,
                    'feedback': f'Partially correct. Mean error: {mean_error:.1%}',
                }
            else:
                # Check if signs are correct
                sign_match = np.mean(np.sign(expected_arr) == np.sign(received_arr))
                if sign_match > 0.8:
                    points = int(max_points * 0.2)
                    return {
                        'points_earned': points,
                        'correct': False,
                        'feedback': 'Signs correct but magnitudes differ',
                    }
                return {
                    'points_earned': 0,
                    'correct': False,
                    'feedback': 'Incorrect array values',
                }

        except Exception as e:
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': f'Could not compare arrays: {str(e)}',
            }

    def _check_text(
        self,
        expected: str,
        received: Any,
        max_points: int
    ) -> Dict:
        """Check a text/interpretation answer."""

        if not isinstance(received, str):
            received = str(received)

        # Extract key concepts from expected answer
        key_concepts = self._extract_concepts(expected)

        # Count concept matches
        received_lower = received.lower()
        matches = sum(1 for c in key_concepts if c.lower() in received_lower)

        if not key_concepts:
            # No specific concepts to check
            if len(received) > 10:  # Some effort made
                return {
                    'points_earned': int(max_points * 0.5),
                    'correct': False,
                    'feedback': 'Answer provided but cannot auto-grade text responses fully',
                }
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': 'Insufficient response',
            }

        match_ratio = matches / len(key_concepts)

        if match_ratio >= 0.8:
            return {
                'points_earned': max_points,
                'correct': True,
                'feedback': 'Excellent interpretation!',
            }
        elif match_ratio >= 0.6:
            points = int(max_points * 0.8)
            return {
                'points_earned': points,
                'correct': False,
                'feedback': 'Good interpretation, but missing some key concepts',
            }
        elif match_ratio >= 0.4:
            points = int(max_points * 0.5)
            return {
                'points_earned': points,
                'correct': False,
                'feedback': 'Partial understanding shown',
            }
        elif match_ratio > 0:
            points = int(max_points * 0.2)
            return {
                'points_earned': points,
                'correct': False,
                'feedback': 'Some relevant concepts mentioned',
            }
        else:
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': 'Review the expected concepts for this question',
            }

    def _check_dict(
        self,
        expected: Dict,
        received: Any,
        max_points: int,
        tolerance: float
    ) -> Dict:
        """Check a dictionary answer."""

        if not isinstance(received, dict):
            return {
                'points_earned': 0,
                'correct': False,
                'feedback': 'Answer must be a dictionary',
            }

        # Check each key
        total_keys = len(expected)
        correct_keys = 0

        for key, exp_value in expected.items():
            if key in received:
                recv_value = received[key]

                if isinstance(exp_value, (int, float)):
                    if isinstance(recv_value, (int, float)):
                        error = abs(recv_value - exp_value) / (abs(exp_value) + 1e-10)
                        if error <= tolerance:
                            correct_keys += 1
                elif exp_value == recv_value:
                    correct_keys += 1

        ratio = correct_keys / total_keys if total_keys > 0 else 0

        if ratio >= 0.9:
            return {
                'points_earned': max_points,
                'correct': True,
                'feedback': 'Correct!',
            }
        elif ratio >= 0.7:
            return {
                'points_earned': int(max_points * ratio),
                'correct': False,
                'feedback': f'{correct_keys}/{total_keys} values correct',
            }
        else:
            return {
                'points_earned': int(max_points * ratio),
                'correct': False,
                'feedback': f'Only {correct_keys}/{total_keys} values correct',
            }

    def _extract_concepts(self, text: str) -> List[str]:
        """Extract key concepts from expected answer text."""

        # Keywords that indicate important concepts
        concept_keywords = [
            # Health/status terms
            'critical', 'warning', 'healthy', 'degraded', 'normal', 'abnormal',
            'stable', 'unstable', 'marginal',

            # Technical terms
            'collapse', 'dimensional', 'correlation', 'coupling', 'eigenvalue',
            'efficiency', 'conservation', 'balance', 'violation',

            # Action terms
            'shutdown', 'monitor', 'inspection', 'maintenance', 'immediate',

            # Trend terms
            'increasing', 'decreasing', 'stable', 'volatile',
        ]

        found = []
        text_lower = text.lower()

        for keyword in concept_keywords:
            if keyword in text_lower:
                found.append(keyword)

        return found

    def _calculate_grade(self, percentage: float) -> str:
        """Calculate letter grade from percentage."""

        for grade, boundary in self.grade_boundaries.items():
            if percentage >= boundary:
                return grade
        return 'F'

    def _assess_conceptual_understanding(
        self,
        problem: CalculationProblem,
        answers: Dict,
        step_results: List[Dict]
    ) -> float:
        """
        Assess conceptual understanding beyond numerical accuracy.

        Gives higher weight to later steps (which build on earlier understanding)
        and interpretation questions.
        """

        if not step_results:
            return 0.0

        weighted_scores = []

        for i, result in enumerate(step_results):
            step = problem.solution_steps[i]

            # Base weight increases with step number
            weight = 1.0 + 0.2 * i

            # Extra weight for interpretation steps
            if 'interpret' in step.description.lower():
                weight *= 1.5

            score = result['points_earned'] / max(result['max_points'], 1)
            weighted_scores.append((weight, score))

        total_weight = sum(w for w, s in weighted_scores)
        if total_weight > 0:
            return sum(w * s for w, s in weighted_scores) / total_weight

        return 0.0

    def _generate_feedback(
        self,
        percentage: float,
        conceptual_score: float,
        step_results: List[Dict]
    ) -> str:
        """Generate overall feedback message."""

        if percentage >= 90:
            opening = "Excellent work!"
        elif percentage >= 80:
            opening = "Good job!"
        elif percentage >= 70:
            opening = "Satisfactory performance."
        elif percentage >= 60:
            opening = "Needs improvement."
        else:
            opening = "Significant review needed."

        # Add conceptual assessment
        if conceptual_score >= 0.8:
            conceptual = "You demonstrate strong conceptual understanding."
        elif conceptual_score >= 0.6:
            conceptual = "Your conceptual understanding is developing well."
        elif conceptual_score >= 0.4:
            conceptual = "Focus on understanding the underlying concepts better."
        else:
            conceptual = "Review the fundamental concepts before attempting similar problems."

        # Identify specific issues
        issues = []
        for result in step_results:
            if not result['correct'] and result['points_earned'] == 0:
                issues.append(f"Step {result['step_number']}")

        if issues:
            specific = f"Review: {', '.join(issues[:3])}."
        else:
            specific = ""

        return f"{opening} {conceptual} {specific}".strip()

    def get_hints(
        self,
        problem: CalculationProblem,
        step_number: int,
        student_answer: Any
    ) -> List[str]:
        """Get hints for a specific step based on student's answer."""

        if step_number > len(problem.solution_steps):
            return []

        step = problem.solution_steps[step_number - 1]
        hints = list(step.hints)  # Copy default hints

        # Add contextual hints based on answer
        if student_answer is not None:
            expected = step.expected_result

            if isinstance(expected, (int, float)) and isinstance(student_answer, (int, float)):
                if abs(student_answer) > abs(expected) * 10:
                    hints.append("Check your units - answer seems too large")
                elif abs(student_answer) < abs(expected) / 10:
                    hints.append("Check your units - answer seems too small")
                elif student_answer * expected < 0:
                    hints.append("Check the sign of your answer")

        return hints[:3]  # Return up to 3 hints
