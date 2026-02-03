"""
ORTHON Observation Processing Pipeline

Integrates validation and cohort discovery into a single pipeline.
This is the main entry point for processing observations before PRISM.

PIPELINE:
    1. VALIDATE: Remove constants, duplicates, flag orphans
    2. DISCOVER: Identify cohort structure (system vs component signals)
    3. OUTPUT: Validated data + ML-ready signal lists + report

PRINCIPLE: "Garbage in, REJECTED"

Usage:
    from orthon.pipeline import process_observations
    
    result = process_observations(
        observations_path='data/observations.parquet',
        output_dir='data/processed/',
    )
    
    # Use validated data for PRISM
    validated_df = result.validated_df
    ml_signals = result.cohort_result.get_ml_signals()
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yaml

import polars as pl

from .validation import (
    SignalValidator,
    ValidationConfig,
    ValidationReport,
    validate_observations,
)
from .cohort_discovery import (
    CohortDiscovery,
    CohortResult,
    process_observations as discover_cohorts,
)


@dataclass
class PipelineResult:
    """Complete result from observation processing pipeline."""
    
    # Validated data
    validated_df: pl.DataFrame = None
    validation_report: ValidationReport = None
    
    # Cohort discovery
    cohort_result: CohortResult = None
    
    # Combined outputs
    ml_signals: List[str] = field(default_factory=list)
    exclude_signals: List[str] = field(default_factory=list)
    
    # Metadata
    input_signals: int = 0
    output_signals: int = 0
    input_rows: int = 0
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 70,
            "ORTHON OBSERVATION PROCESSING PIPELINE",
            "=" * 70,
            "",
            f"INPUT:  {self.input_signals} signals, {self.input_rows:,} rows",
            f"OUTPUT: {self.output_signals} signals (validated)",
            "",
            "─" * 70,
            "STAGE 1: VALIDATION",
            "─" * 70,
        ]
        
        if self.validation_report:
            if self.validation_report.excluded:
                lines.append(f"Excluded {len(self.validation_report.excluded)} signals:")
                for sv in self.validation_report.excluded:
                    lines.append(f"  ✗ {sv.signal_id}: {sv.issue.value}")
            else:
                lines.append("No signals excluded.")
            
            if self.validation_report.warnings:
                lines.append(f"Warnings ({len(self.validation_report.warnings)}):")
                for sv in self.validation_report.warnings:
                    lines.append(f"  ⚠ {sv.signal_id}: {sv.issue.value}")
        
        lines.extend([
            "",
            "─" * 70,
            "STAGE 2: COHORT DISCOVERY",
            "─" * 70,
        ])
        
        if self.cohort_result:
            if self.cohort_result.constants:
                lines.append(f"Constants ({len(self.cohort_result.constants)}): {self.cohort_result.constants}")
            if self.cohort_result.system_signals:
                lines.append(f"System signals ({len(self.cohort_result.system_signals)}): {self.cohort_result.system_signals[:5]}{'...' if len(self.cohort_result.system_signals) > 5 else ''}")
            if self.cohort_result.component_signals:
                lines.append(f"Component signals ({len(self.cohort_result.component_signals)}): {self.cohort_result.component_signals[:5]}{'...' if len(self.cohort_result.component_signals) > 5 else ''}")
            
            if self.cohort_result.coupled_units:
                lines.append(f"Coupled units ({len(self.cohort_result.coupled_units)}): {self.cohort_result.coupled_units[:5]}")
            if self.cohort_result.decoupled_units:
                lines.append(f"Decoupled units ({len(self.cohort_result.decoupled_units)}): {self.cohort_result.decoupled_units[:5]}")
        
        lines.extend([
            "",
            "─" * 70,
            "ML RECOMMENDATIONS",
            "─" * 70,
            f"USE ({len(self.ml_signals)} signals): {self.ml_signals[:10]}{'...' if len(self.ml_signals) > 10 else ''}",
            f"EXCLUDE ({len(self.exclude_signals)} signals): {self.exclude_signals}",
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'input_signals': self.input_signals,
            'output_signals': self.output_signals,
            'input_rows': self.input_rows,
            'ml_signals': self.ml_signals,
            'exclude_signals': self.exclude_signals,
            'validation': self.validation_report.to_dict() if self.validation_report else None,
            'cohort_discovery': self.cohort_result.to_dict() if self.cohort_result else None,
        }


class ObservationPipeline:
    """
    Complete observation processing pipeline.
    
    Stages:
        1. Validation - Remove garbage signals
        2. Cohort Discovery - Identify signal structure
        3. Output - Validated data + recommendations
    """
    
    def __init__(
        self,
        validation_config: Optional[ValidationConfig] = None,
        run_cohort_discovery: bool = True,
        cohort_thresholds: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize pipeline.
        
        Args:
            validation_config: Validation settings (default: strict)
            run_cohort_discovery: Whether to run cohort discovery (default: True)
            cohort_thresholds: Optional thresholds for cohort discovery
        """
        self.validation_config = validation_config or ValidationConfig.strict_mode()
        self.run_cohort_discovery = run_cohort_discovery
        self.cohort_thresholds = cohort_thresholds or {}
    
    def process(
        self,
        df: pl.DataFrame,
    ) -> PipelineResult:
        """
        Process observations through the pipeline.
        
        Args:
            df: Input observations DataFrame
            
        Returns:
            PipelineResult with validated data and recommendations
        """
        result = PipelineResult(
            input_rows=len(df),
        )
        
        # Count input signals
        meta_cols = {'timestamp', 'time', 'cycle', 'I', 'unit_id', 'cohort_id', 
                     'window', 'observation_id', 'signal_id', 'signal', 'value',
                     'RUL', 'rul', 'label', 'target'}
        input_signals = [c for c in df.columns if c.lower() not in {m.lower() for m in meta_cols}]
        result.input_signals = len(input_signals)
        
        # Stage 1: Validation
        validator = SignalValidator(self.validation_config)
        validated_df, validation_report = validator.validate(df)
        
        result.validated_df = validated_df
        result.validation_report = validation_report
        result.output_signals = validation_report.valid_signals
        
        # Collect excluded signals from validation
        validation_excluded = [sv.signal_id for sv in validation_report.excluded]
        
        # Stage 2: Cohort Discovery (optional, on validated data)
        if self.run_cohort_discovery and validation_report.validation_passed:
            cd = CohortDiscovery(
                constant_threshold=self.cohort_thresholds.get('constant', 0.99),
                system_threshold=self.cohort_thresholds.get('system', 0.70),
                component_threshold=self.cohort_thresholds.get('component', 0.50),
            )
            cd._observations = validated_df
            
            cohort_result = cd.discover()
            result.cohort_result = cohort_result
            
            # Combine ML signals and exclusions
            result.ml_signals = cohort_result.get_ml_signals()
            result.exclude_signals = list(set(
                validation_excluded + 
                cohort_result.get_exclude_list()
            ))
        else:
            # No cohort discovery - use validation results directly
            result.ml_signals = validation_report.valid
            result.exclude_signals = validation_excluded
        
        return result
    
    def process_file(
        self,
        observations_path: str,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        Process observations from file.
        
        Args:
            observations_path: Path to observations.parquet
            output_dir: Optional output directory
            
        Returns:
            PipelineResult
        """
        df = pl.read_parquet(observations_path)
        result = self.process(df)
        
        print(result.summary())
        
        if output_dir:
            self._save_results(result, Path(output_dir))
        
        return result
    
    def _save_results(self, result: PipelineResult, output_dir: Path):
        """Save results to output directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save validated data
        if result.validated_df is not None:
            result.validated_df.write_parquet(output_dir / 'observations_validated.parquet')
        
        # Save complete report
        with open(output_dir / 'pipeline_report.yaml', 'w') as f:
            yaml.dump(result.to_dict(), f, default_flow_style=False)
        
        with open(output_dir / 'pipeline_report.txt', 'w') as f:
            f.write(result.summary())
        
        # Save ML-ready signal list
        with open(output_dir / 'ml_signals.txt', 'w') as f:
            for sig in result.ml_signals:
                f.write(f"{sig}\n")
        
        # Save exclusion list
        with open(output_dir / 'exclude_signals.txt', 'w') as f:
            for sig in result.exclude_signals:
                f.write(f"{sig}\n")
        
        # Save manifest update
        manifest_update = {
            'preprocessing': {
                'pipeline_version': '1.0',
                'validation_passed': result.validation_report.validation_passed if result.validation_report else True,
                'signals': {
                    'input': result.input_signals,
                    'output': result.output_signals,
                    'ml_ready': result.ml_signals,
                    'excluded': result.exclude_signals,
                },
            }
        }
        
        with open(output_dir / 'manifest_update.yaml', 'w') as f:
            yaml.dump(manifest_update, f, default_flow_style=False)
        
        print(f"\nResults saved to {output_dir}")


def process_observations(
    observations_path: str,
    output_dir: Optional[str] = None,
    strict: bool = True,
    run_cohort_discovery: bool = True,
) -> PipelineResult:
    """
    Main entry point: Process observations through validation and cohort discovery.
    
    Args:
        observations_path: Path to observations.parquet
        output_dir: Optional output directory
        strict: If True, exclude bad signals. If False, only warn.
        run_cohort_discovery: Whether to run cohort discovery
        
    Returns:
        PipelineResult with validated data and recommendations
        
    Example:
        >>> result = process_observations('data/observations.parquet', 'data/processed/')
        >>> print(f"Use {len(result.ml_signals)} signals for ML")
        >>> print(f"Exclude: {result.exclude_signals}")
    """
    config = ValidationConfig.strict_mode() if strict else ValidationConfig.permissive()
    
    pipeline = ObservationPipeline(
        validation_config=config,
        run_cohort_discovery=run_cohort_discovery,
    )
    
    return pipeline.process_file(observations_path, output_dir)


# CLI
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Process observations through validation and cohort discovery'
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--permissive', action='store_true', help='Warn only, do not exclude')
    parser.add_argument('--no-cohort-discovery', action='store_true', help='Skip cohort discovery')
    
    args = parser.parse_args()
    
    result = process_observations(
        args.observations,
        output_dir=args.output,
        strict=not args.permissive,
        run_cohort_discovery=not args.no_cohort_discovery,
    )
    
    if result.validation_report and not result.validation_report.validation_passed:
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
