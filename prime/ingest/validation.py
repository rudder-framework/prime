"""
Signal Validation Module

Strict input validation before Manifold runs.
Ensures only valid, information-carrying signals enter the analysis.

PRINCIPLE: "Garbage in, REJECTED" (not "garbage in, garbage out")

VALIDATION RULES:
    1. Constants (std = 0) → EXCLUDE
    2. Near-constants (std < ε) → EXCLUDE
    3. Duplicates (ρ > 0.999) → EXCLUDE (keep first)
    4. Orphans (max ρ < 0.1) → WARN
    5. Insufficient data → FAIL

Usage:
    from prime.ingest.validation import validate_observations

    validated_df, report = validate_observations(df)
    # Only validated_df goes to Manifold
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import warnings

import polars as pl


class ValidationAction(Enum):
    """Action to take on validation failure."""
    EXCLUDE = "exclude"  # Remove signal, continue
    WARN = "warn"        # Keep signal, log warning
    FAIL = "fail"        # Raise exception, stop


class ValidationIssue(Enum):
    """Types of validation issues."""
    CONSTANT = "constant"
    NEAR_CONSTANT = "near_constant"
    DUPLICATE = "duplicate"
    ORPHAN = "orphan"
    INSUFFICIENT_DATA = "insufficient_data"
    INVALID_TYPE = "invalid_type"


@dataclass
class SignalValidation:
    """Validation result for a single signal."""
    signal_id: str
    valid: bool
    issue: Optional[ValidationIssue] = None
    action_taken: Optional[ValidationAction] = None
    details: str = ""
    related_signal: Optional[str] = None  # For duplicates


@dataclass
class ValidationReport:
    """Complete validation report."""
    
    # Counts
    total_signals: int = 0
    valid_signals: int = 0
    excluded_signals: int = 0
    warned_signals: int = 0
    
    # Lists
    valid: List[str] = field(default_factory=list)
    excluded: List[SignalValidation] = field(default_factory=list)
    warnings: List[SignalValidation] = field(default_factory=list)
    
    # Metadata
    total_rows: int = 0
    validation_passed: bool = True
    failure_reason: Optional[str] = None
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            "=" * 60,
            "MANIFOLD VALIDATION REPORT",
            "=" * 60,
            "",
            f"INPUT: {self.total_signals} signals, {self.total_rows:,} rows",
            "",
        ]
        
        if self.excluded:
            lines.append(f"EXCLUDED ({len(self.excluded)}):")
            for sv in self.excluded:
                detail = f" → {sv.related_signal}" if sv.related_signal else ""
                lines.append(f"  ├── {sv.signal_id}: {sv.issue.value}{detail}")
                if sv.details:
                    lines.append(f"  │   {sv.details}")
            lines.append("")
        
        if self.warnings:
            lines.append(f"WARNINGS ({len(self.warnings)}):")
            for sv in self.warnings:
                lines.append(f"  ├── {sv.signal_id}: {sv.issue.value}")
                if sv.details:
                    lines.append(f"  │   {sv.details}")
            lines.append("")
        
        if self.validation_passed:
            lines.append(f"VALIDATED: {self.valid_signals} signals → proceeding to analysis")
        else:
            lines.append(f"VALIDATION FAILED: {self.failure_reason}")
        
        lines.extend(["", "=" * 60])
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dict."""
        return {
            'total_signals': self.total_signals,
            'valid_signals': self.valid_signals,
            'excluded_signals': self.excluded_signals,
            'warned_signals': self.warned_signals,
            'valid': self.valid,
            'excluded': [
                {
                    'signal': sv.signal_id,
                    'issue': sv.issue.value if sv.issue else None,
                    'details': sv.details,
                    'related_signal': sv.related_signal,
                }
                for sv in self.excluded
            ],
            'warnings': [
                {
                    'signal': sv.signal_id,
                    'issue': sv.issue.value if sv.issue else None,
                    'details': sv.details,
                }
                for sv in self.warnings
            ],
            'validation_passed': self.validation_passed,
            'failure_reason': self.failure_reason,
        }


@dataclass
class ValidationConfig:
    """Configuration for validation thresholds and actions."""
    
    # Thresholds
    constant_std: float = 1e-10          # Below this = constant
    near_constant_std: float = 1e-6      # Below this = near-constant
    duplicate_corr: float = 0.999        # Above this = duplicate
    orphan_max_corr: float = 0.1         # Below this = orphan
    min_signals: int = 2                 # Minimum valid signals required
    min_rows: int = 10                   # Minimum observations required
    
    # Actions
    on_constant: ValidationAction = ValidationAction.EXCLUDE
    on_near_constant: ValidationAction = ValidationAction.EXCLUDE
    on_duplicate: ValidationAction = ValidationAction.EXCLUDE
    on_orphan: ValidationAction = ValidationAction.WARN
    on_insufficient_signals: ValidationAction = ValidationAction.FAIL
    on_insufficient_rows: ValidationAction = ValidationAction.FAIL
    
    # Behavior
    strict: bool = True                  # If False, only warn (never exclude)
    
    @classmethod
    def permissive(cls) -> 'ValidationConfig':
        """Permissive config - warn but don't exclude."""
        return cls(
            on_constant=ValidationAction.WARN,
            on_near_constant=ValidationAction.WARN,
            on_duplicate=ValidationAction.WARN,
            on_orphan=ValidationAction.WARN,
            strict=False,
        )
    
    @classmethod
    def strict_mode(cls) -> 'ValidationConfig':
        """Strict config - exclude bad signals, fail on critical issues."""
        return cls(strict=True)


class SignalValidator:
    """
    Validates signals before Manifold analysis.

    Ensures only valid, information-carrying signals enter the pipeline.
    Handles both wide format (one column per signal) and long format
    (signal_id + value columns).
    """

    def __init__(self, config: Optional[ValidationConfig] = None):
        """
        Initialize validator.

        Args:
            config: Validation configuration (defaults to strict mode)
        """
        self.config = config or ValidationConfig.strict_mode()

    @staticmethod
    def _is_long_format(df: pl.DataFrame) -> bool:
        """Check if DataFrame is in long format (signal_id + value columns)."""
        return 'signal_id' in df.columns and 'value' in df.columns

    def _long_to_wide(self, df: pl.DataFrame) -> pl.DataFrame:
        """Pivot long format to wide for validation checks."""
        # Group by cohort+I or just I, pivot signal_id into columns
        index_cols = ['signal_0']
        if 'cohort' in df.columns:
            index_cols = ['cohort'] + index_cols
        return df.pivot(on='signal_id', index=index_cols, values='value')

    def _detect_signal_columns(self, df: pl.DataFrame) -> List[str]:
        """Detect which columns are signals (numeric, non-metadata)."""
        meta_cols = {
            'timestamp', 'time', 'cycle', 'signal_0',
            'unit_id', 'cohort_id', 'cohort', 'window', 'observation_id',
            'signal_id', 'signal', 'value',  # Long format columns
            'RUL', 'rul', 'label', 'target',  # Target columns
        }

        signal_cols = []
        for col in df.columns:
            if col.lower() in {m.lower() for m in meta_cols}:
                continue
            if df[col].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32, pl.Int16, pl.Int8]:
                signal_cols.append(col)

        return signal_cols
    
    def _check_constant(self, df: pl.DataFrame, signal: str) -> Optional[SignalValidation]:
        """Check if signal is constant or near-constant."""
        std = df[signal].std()
        
        if std is None or std < self.config.constant_std:
            return SignalValidation(
                signal_id=signal,
                valid=False,
                issue=ValidationIssue.CONSTANT,
                action_taken=self.config.on_constant,
                details=f"std={std:.2e} < {self.config.constant_std:.2e}",
            )
        
        if std < self.config.near_constant_std:
            return SignalValidation(
                signal_id=signal,
                valid=False,
                issue=ValidationIssue.NEAR_CONSTANT,
                action_taken=self.config.on_near_constant,
                details=f"std={std:.2e} < {self.config.near_constant_std:.2e}",
            )
        
        return None
    
    def _check_duplicates(
        self, 
        df: pl.DataFrame, 
        signals: List[str],
        already_validated: Set[str],
    ) -> Dict[str, SignalValidation]:
        """Check for duplicate signals (highly correlated pairs)."""
        duplicates = {}
        
        validated_list = list(already_validated)
        
        for signal in signals:
            if signal in duplicates:
                continue
            
            for other in validated_list:
                if other == signal or other in duplicates:
                    continue
                
                # Compute correlation
                try:
                    valid_data = df.select([signal, other]).drop_nulls()
                    if len(valid_data) < 10:
                        continue
                    
                    corr = abs(valid_data[signal].pearson_corr(valid_data[other]))
                    
                    if corr is not None and corr > self.config.duplicate_corr:
                        duplicates[signal] = SignalValidation(
                            signal_id=signal,
                            valid=False,
                            issue=ValidationIssue.DUPLICATE,
                            action_taken=self.config.on_duplicate,
                            details=f"ρ={corr:.4f} > {self.config.duplicate_corr}",
                            related_signal=other,
                        )
                        break
                except:
                    continue
            
            # If not a duplicate, add to validated for future comparisons
            if signal not in duplicates:
                validated_list.append(signal)
        
        return duplicates
    
    def _check_orphans(
        self,
        df: pl.DataFrame,
        signals: List[str],
    ) -> Dict[str, SignalValidation]:
        """Check for orphan signals (uncorrelated with everything)."""
        orphans = {}
        
        for signal in signals:
            max_corr = 0.0
            
            for other in signals:
                if other == signal:
                    continue
                
                try:
                    valid_data = df.select([signal, other]).drop_nulls()
                    if len(valid_data) < 10:
                        continue
                    
                    corr = abs(valid_data[signal].pearson_corr(valid_data[other]))
                    if corr is not None and corr > max_corr:
                        max_corr = corr
                except:
                    continue
            
            if max_corr < self.config.orphan_max_corr:
                orphans[signal] = SignalValidation(
                    signal_id=signal,
                    valid=False,
                    issue=ValidationIssue.ORPHAN,
                    action_taken=self.config.on_orphan,
                    details=f"max_ρ={max_corr:.4f} < {self.config.orphan_max_corr}",
                )
        
        return orphans
    
    def validate(self, df: pl.DataFrame) -> Tuple[pl.DataFrame, ValidationReport]:
        """
        Validate observations DataFrame.

        Handles both wide format (one column per signal) and long format
        (signal_id + value columns). Long format is pivoted to wide for
        validation, then filtered results are returned in the original format.

        Args:
            df: Input observations DataFrame

        Returns:
            Tuple of (validated_df, report)

        Raises:
            ValueError: If validation fails with FAIL action
        """
        is_long = self._is_long_format(df)

        # For long format, pivot to wide so all validation checks work
        if is_long:
            wide_df = self._long_to_wide(df)
        else:
            wide_df = df

        report = ValidationReport(total_rows=len(df))

        # Check minimum rows
        if len(df) < self.config.min_rows:
            report.validation_passed = False
            report.failure_reason = f"Insufficient rows: {len(df)} < {self.config.min_rows}"
            if self.config.on_insufficient_rows == ValidationAction.FAIL:
                raise ValueError(report.failure_reason)
            return df, report

        # Detect signal columns (from the wide representation)
        signal_cols = self._detect_signal_columns(wide_df)
        report.total_signals = len(signal_cols)

        if len(signal_cols) < self.config.min_signals:
            report.validation_passed = False
            report.failure_reason = f"Insufficient signals: {len(signal_cols)} < {self.config.min_signals}"
            if self.config.on_insufficient_signals == ValidationAction.FAIL:
                raise ValueError(report.failure_reason)
            return df, report

        # Track which signals to keep
        valid_signals = set(signal_cols)
        validated_set = set()  # For duplicate checking

        # Check constants
        for signal in signal_cols:
            result = self._check_constant(wide_df, signal)
            if result:
                if result.action_taken == ValidationAction.EXCLUDE:
                    valid_signals.discard(signal)
                    report.excluded.append(result)
                elif result.action_taken == ValidationAction.WARN:
                    report.warnings.append(result)
                    validated_set.add(signal)
                elif result.action_taken == ValidationAction.FAIL:
                    report.validation_passed = False
                    report.failure_reason = f"Constant signal: {signal}"
                    raise ValueError(report.failure_reason)
            else:
                validated_set.add(signal)

        # Check duplicates (only among non-constant signals)
        remaining = [s for s in signal_cols if s in valid_signals]
        duplicates = self._check_duplicates(wide_df, remaining, validated_set)

        for signal, result in duplicates.items():
            if result.action_taken == ValidationAction.EXCLUDE:
                valid_signals.discard(signal)
                report.excluded.append(result)
            elif result.action_taken == ValidationAction.WARN:
                report.warnings.append(result)

        # Check orphans (only among remaining signals)
        remaining = [s for s in signal_cols if s in valid_signals]
        orphans = self._check_orphans(wide_df, remaining)

        for signal, result in orphans.items():
            if result.action_taken == ValidationAction.EXCLUDE:
                valid_signals.discard(signal)
                report.excluded.append(result)
            elif result.action_taken == ValidationAction.WARN:
                report.warnings.append(result)

        # Final check: enough signals remaining?
        if len(valid_signals) < self.config.min_signals:
            report.validation_passed = False
            report.failure_reason = f"Too few signals after validation: {len(valid_signals)} < {self.config.min_signals}"
            if self.config.on_insufficient_signals == ValidationAction.FAIL:
                raise ValueError(report.failure_reason)

        # Build report
        report.valid = sorted(valid_signals)
        report.valid_signals = len(valid_signals)
        report.excluded_signals = len(report.excluded)
        report.warned_signals = len(report.warnings)

        # Build validated DataFrame in original format
        if is_long:
            # Filter long format to only keep valid signal_ids
            validated_df = df.filter(pl.col('signal_id').is_in(sorted(valid_signals)))
        else:
            meta_cols = [c for c in df.columns if c not in signal_cols]
            keep_cols = meta_cols + sorted(valid_signals)
            validated_df = df.select([c for c in keep_cols if c in df.columns])

        return validated_df, report


def validate_observations(
    df: pl.DataFrame,
    strict: bool = True,
    config: Optional[ValidationConfig] = None,
) -> Tuple[pl.DataFrame, ValidationReport]:
    """
    Validate observations before Manifold analysis.

    Main entry point for validation.

    Args:
        df: Input observations DataFrame
        strict: If True, use strict validation (default). If False, only warn.
        config: Optional custom configuration

    Returns:
        Tuple of (validated_df, report)

    Example:
        >>> validated_df, report = validate_observations(df)
        >>> print(report.summary())
        >>> # Only validated_df goes to Manifold
    """
    if config is None:
        config = ValidationConfig.strict_mode() if strict else ValidationConfig.permissive()
    
    validator = SignalValidator(config)
    return validator.validate(df)


def validate_parquet(
    path: str,
    output_dir: Optional[str] = None,
    strict: bool = True,
) -> Tuple[pl.DataFrame, ValidationReport]:
    """
    Validate observations from parquet file.
    
    Args:
        path: Path to observations.parquet
        output_dir: Optional directory to save validated data and report
        strict: If True, use strict validation
        
    Returns:
        Tuple of (validated_df, report)
    """
    import yaml
    
    df = pl.read_parquet(path)
    validated_df, report = validate_observations(df, strict=strict)
    
    print(report.summary())
    
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save validated data
        validated_df.write_parquet(output_path / 'observations_validated.parquet')
        
        # Save report
        with open(output_path / 'validation_report.yaml', 'w') as f:
            yaml.dump(report.to_dict(), f, default_flow_style=False)
        
        with open(output_path / 'validation_report.txt', 'w') as f:
            f.write(report.summary())
        
        # Save signal lists
        with open(output_path / 'valid_signals.txt', 'w') as f:
            for sig in report.valid:
                f.write(f"{sig}\n")
        
        if report.excluded:
            with open(output_path / 'excluded_signals.txt', 'w') as f:
                for sv in report.excluded:
                    f.write(f"{sv.signal_id}\t{sv.issue.value}\t{sv.details}\n")
        
        print(f"\nResults saved to {output_path}")
    
    return validated_df, report


# CLI
def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Validate observations before Manifold analysis'
    )
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', help='Output directory')
    parser.add_argument('--permissive', action='store_true', help='Warn only, do not exclude')
    parser.add_argument('--constant-std', type=float, default=1e-10)
    parser.add_argument('--duplicate-corr', type=float, default=0.999)
    parser.add_argument('--orphan-corr', type=float, default=0.1)
    
    args = parser.parse_args()
    
    config = ValidationConfig(
        constant_std=args.constant_std,
        duplicate_corr=args.duplicate_corr,
        orphan_max_corr=args.orphan_corr,
    )
    
    if args.permissive:
        config = ValidationConfig.permissive()
    
    validated_df, report = validate_parquet(
        args.observations,
        output_dir=args.output,
        strict=not args.permissive,
    )
    
    if not report.validation_passed:
        return 1
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
