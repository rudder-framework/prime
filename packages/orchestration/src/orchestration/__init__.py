"""
Orchestration package for the Rudder Framework.

Runs the compute pipeline in correct order:
typology → vector → eigendecomp → geometry → dynamics → ...

This package imports and sequences all other packages.
It reads parquet, calls package functions, writes parquet.
No math lives here — only wiring.
"""

from orchestration.pipeline import Pipeline, PipelineStage

__all__ = ['Pipeline', 'PipelineStage']
