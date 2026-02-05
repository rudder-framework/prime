"""
Core ORTHON pipeline components.
"""

from .pipeline import ObservationPipeline, PipelineResult
from .data_reader import DataReader
from .validation import (
    SignalValidator,
    ValidationConfig,
    ValidationReport,
    validate_observations,
)
from .prism_client import PRISMClient, AsyncPRISMClient

# Aliases for backward compatibility
Pipeline = ObservationPipeline
PrismClient = PRISMClient

__all__ = [
    'Pipeline',
    'ObservationPipeline',
    'PipelineResult',
    'DataReader',
    'SignalValidator',
    'ValidationConfig',
    'ValidationReport',
    'validate_observations',
    'PRISMClient',
    'AsyncPRISMClient',
    'PrismClient',
]
