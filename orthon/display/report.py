"""
ORTHON Report Generation
========================

Generate analysis reports from validation and analysis results.
"""

from datetime import datetime
from typing import Dict, Any, Optional
import pandas as pd


def generate_report(
    validation: Dict[str, Any],
    analysis: Optional[Dict[str, Any]] = None,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a complete report from validation and analysis results.

    Args:
        validation: Results from intake.validate()
        analysis: Results from backend.analyze() (optional)
        filename: Original filename

    Returns:
        Complete report dictionary
    """
    report = {
        'meta': {
            'generated_at': datetime.now().isoformat(),
            'orthon_version': _get_version(),
            'filename': filename,
        },
        'summary': {
            'rows': validation['rows'],
            'columns': validation['columns'],
            'signals': len(validation['structure']['signals']),
            'entities': 1,  # Updated below if entity column exists
            'valid': validation['valid'],
        },
        'structure': validation['structure'],
        'columns': validation['column_info'],
        'issues': validation['issues'],
        'warnings': validation['warnings'],
    }

    # Count entities if we have entity column
    if validation['structure']['entity_col']:
        # Need the dataframe to count - use column info
        report['summary']['has_entity_column'] = True

    # Add analysis results if available
    if analysis:
        report['analysis'] = analysis
        report['summary']['backend'] = analysis.get('backend', 'unknown')

        # Extract key metrics for summary
        if 'signals' in analysis:
            report['summary']['signals_analyzed'] = len(analysis['signals'])

    return report


def format_report_text(report: Dict[str, Any]) -> str:
    """
    Format report as human-readable text.
    """
    lines = []

    # Header
    lines.append("=" * 60)
    lines.append("ORTHON DATA REPORT")
    lines.append("=" * 60)
    lines.append("")

    # Summary
    s = report['summary']
    lines.append(f"File:     {report['meta'].get('filename', 'unknown')}")
    lines.append(f"Rows:     {s['rows']:,}")
    lines.append(f"Columns:  {s['columns']}")
    lines.append(f"Signals:  {s['signals']}")
    lines.append(f"Valid:    {'Yes' if s['valid'] else 'No'}")
    lines.append("")

    # Structure
    struct = report['structure']
    lines.append("STRUCTURE")
    lines.append("-" * 40)
    lines.append(f"Time column:   {struct['time_col'] or 'not detected'}")
    lines.append(f"Entity column: {struct['entity_col'] or 'single entity'}")
    lines.append(f"Signals:       {', '.join(struct['signals'][:5])}")
    if len(struct['signals']) > 5:
        lines.append(f"               (+{len(struct['signals']) - 5} more)")
    lines.append("")

    # Issues
    if report['issues']:
        lines.append("ISSUES")
        lines.append("-" * 40)
        for issue in report['issues']:
            lines.append(f"  ❌ {issue}")
        lines.append("")

    # Warnings
    if report['warnings']:
        lines.append("WARNINGS")
        lines.append("-" * 40)
        for warning in report['warnings']:
            lines.append(f"  ⚠️  {warning}")
        lines.append("")

    # Signals table
    signals = [c for c in report['columns'] if c['name'] in struct['signals']]
    if signals:
        lines.append("SIGNALS")
        lines.append("-" * 40)
        lines.append(f"{'Column':<20} {'Unit':<8} {'Min':>10} {'Max':>10} {'Mean':>10}")
        lines.append("-" * 60)
        for sig in signals[:10]:
            unit = sig.get('unit') or '?'
            min_val = f"{sig.get('min', 0):.4g}" if 'min' in sig else '-'
            max_val = f"{sig.get('max', 0):.4g}" if 'max' in sig else '-'
            mean_val = f"{sig.get('mean', 0):.4g}" if 'mean' in sig else '-'
            lines.append(f"{sig['name']:<20} {unit:<8} {min_val:>10} {max_val:>10} {mean_val:>10}")
        if len(signals) > 10:
            lines.append(f"... +{len(signals) - 10} more signals")
        lines.append("")

    # Analysis results (if present)
    if 'analysis' in report:
        analysis = report['analysis']
        lines.append("ANALYSIS")
        lines.append("-" * 40)
        lines.append(f"Backend: {analysis.get('backend', 'unknown')}")
        if analysis.get('message'):
            lines.append(f"Note:    {analysis['message']}")
        lines.append("")

    # Footer
    lines.append("=" * 60)
    lines.append(f"Generated: {report['meta']['generated_at']}")

    return "\n".join(lines)


def format_signals_table(report: Dict[str, Any]) -> pd.DataFrame:
    """
    Extract signals as a DataFrame for display.
    """
    signals = report['structure']['signals']
    columns = report['columns']

    rows = []
    for col_info in columns:
        if col_info['name'] in signals:
            rows.append({
                'Column': col_info['name'],
                'Unit': col_info.get('unit') or '?',
                'Min': col_info.get('min'),
                'Max': col_info.get('max'),
                'Mean': col_info.get('mean'),
                'Nulls': col_info.get('nulls', 0),
            })

    return pd.DataFrame(rows)


def _get_version() -> str:
    """Get ORTHON version."""
    try:
        from orthon import __version__
        return __version__
    except Exception:
        return 'unknown'
