"""
Core Prime pipeline components.
"""

from .pipeline import ObservationPipeline, PipelineResult
from .data_reader import DataReader
from .validation import (
    SignalValidator,
    ValidationConfig,
    ValidationReport,
    validate_observations,
)
from .manifold_client import run_manifold, manifold_available, manifold_status

# Aliases for backward compatibility
Pipeline = ObservationPipeline

__all__ = [
    'Pipeline',
    'ObservationPipeline',
    'PipelineResult',
    'DataReader',
    'SignalValidator',
    'ValidationConfig',
    'ValidationReport',
    'validate_observations',
    'run_manifold',
    'manifold_available',
    'manifold_status',
]
