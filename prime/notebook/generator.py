"""Generate analysis notebooks from Manifold run output.

Builds Jupyter notebook JSON directly (no nbformat dependency).
Each known-good SQL report becomes a markdown header + code cell.
Four key reports get matplotlib chart cells.

Usage:
    python -m prime.notebook.generator <run_dir>
    python -m prime.notebook.generator ~/domains/cmapss/FD_001/train/output_time
"""

import json
import sys
from pathlib import Path


# Reports to include (order matters)
REPORTS = [
    ('03_drift_detection', 'Drift Detection'),
    ('04_signal_ranking', 'Signal Ranking'),
    ('05_periodicity', 'Periodicity'),
    ('06_regime_detection', 'Regime Detection'),
    ('07_correlation_changes', 'Correlation Changes'),
    ('08_lead_lag', 'Lead-Lag Analysis'),
    ('09_causality_influence', 'Causality & Influence'),
    ('10_system_departure', 'System Departure'),
    ('11_validation_thresholds', 'Validation Thresholds'),
    ('23_baseline_deviation', 'Baseline Deviation'),
    ('24_incident_summary', 'Incident Summary'),
]

# Chart definitions: (report_stem, chart_title, chart_code_template)
# These generate matplotlib cells after the corresponding report cell.
CHARTS = {
    '03_drift_detection': (
        'Drift: Slope Progression by Signal',
        """\
import matplotlib.pyplot as plt

df = _results['{stem}']
if df is not None and len(df) > 0:
    # Find slope ratio column
    slope_col = next((c for c in df.columns if 'slope_ratio' in c.lower()), None)
    sig_col = next((c for c in df.columns if 'signal' in c.lower()), None)
    if slope_col and sig_col:
        pivot = df.groupby(sig_col)[slope_col].mean().sort_values()
        fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.3)))
        pivot.plot.barh(ax=ax)
        ax.set_xlabel('Mean Slope Ratio')
        ax.set_title('Drift: Slope Progression by Signal')
        ax.axvline(x=1.0, color='red', linestyle='--', alpha=0.5, label='no drift')
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print(f"Columns available: {{list(df.columns)}}")
else:
    print("No drift data available")
""",
    ),
    '07_correlation_changes': (
        'Coupling Heatmap: Correlation Change',
        """\
import matplotlib.pyplot as plt
import numpy as np

df = _results['{stem}']
if df is not None and len(df) > 0:
    # Find delta/change column and signal columns
    delta_col = next((c for c in df.columns if 'delta' in c.lower() or 'change' in c.lower()), None)
    sig_a = next((c for c in df.columns if c.lower() in ('signal_a', 'signal_id_a', 'signal_1')), None)
    sig_b = next((c for c in df.columns if c.lower() in ('signal_b', 'signal_id_b', 'signal_2')), None)
    if delta_col and sig_a and sig_b:
        pivot = df.pivot_table(index=sig_a, columns=sig_b, values=delta_col, aggfunc='mean')
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(pivot.values, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=90, fontsize=7)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index, fontsize=7)
        ax.set_title('Correlation Change: Early vs Late')
        plt.colorbar(im, ax=ax, label=delta_col)
        plt.tight_layout()
        plt.show()
    else:
        print(f"Columns available: {{list(df.columns)}}")
else:
    print("No correlation data available")
""",
    ),
    '10_system_departure': (
        'Fleet Instability: System Departure Scatter',
        """\
import matplotlib.pyplot as plt

df = _results['{stem}']
if df is not None and len(df) > 0:
    # Find numeric columns for scatter
    num_cols = [c for c in df.columns if df[c].dtype in ('float64', 'int64')]
    cohort_col = next((c for c in df.columns if 'cohort' in c.lower()), None)
    if len(num_cols) >= 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        x_col, y_col = num_cols[0], num_cols[1]
        if cohort_col:
            for name, group in df.groupby(cohort_col):
                ax.scatter(group[x_col], group[y_col], label=str(name), alpha=0.6, s=30)
            ax.legend(fontsize=7, ncol=2)
        else:
            ax.scatter(df[x_col], df[y_col], alpha=0.6, s=30)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title('Fleet Instability: System Departure')
        plt.tight_layout()
        plt.show()
    else:
        print(f"Columns available: {{list(df.columns)}}")
else:
    print("No departure data available")
""",
    ),
}


def _make_cell(cell_type: str, source: str | list[str], execution_count=None) -> dict:
    """Build a single notebook cell dict."""
    if isinstance(source, str):
        source = source.split('\n')
    # Ensure each line ends with newline except the last
    lines = [line + '\n' for line in source[:-1]] + [source[-1]] if source else ['']
    cell = {
        'cell_type': cell_type,
        'metadata': {},
        'source': lines,
    }
    if cell_type == 'code':
        cell['execution_count'] = execution_count
        cell['outputs'] = []
    return cell


def _discover_parquets(run_dir: Path, domain_dir: Path) -> list[tuple[str, Path]]:
    """Find all parquet files to load as DuckDB views."""
    views = []
    # Manifold output parquets
    for pq in sorted(run_dir.rglob('*.parquet')):
        views.append((pq.stem, pq))
    # Domain-root parquets
    for name in ['observations', 'typology', 'typology_raw', 'signals']:
        path = domain_dir / f'{name}.parquet'
        if path.exists():
            views.append((name, path))
    return views


def _setup_cell(parquets: list[tuple[str, Path]]) -> str:
    """Generate the setup code cell that loads parquets and creates alias views."""
    lines = [
        'import duckdb',
        'import pandas as pd',
        '',
        'con = duckdb.connect()',
        '',
        '# Load parquet files as DuckDB views',
    ]
    for name, path in parquets:
        lines.append(f"con.execute(\"CREATE OR REPLACE VIEW {name} AS "
                     f"SELECT * FROM read_parquet('{path}')\")")

    lines.extend([
        '',
        '# Alias old view names to current parquet stems',
        "try: con.execute('CREATE OR REPLACE VIEW state_geometry AS SELECT * FROM cohort_geometry')",
        "except: pass",
        "try: con.execute('CREATE OR REPLACE VIEW state_vector AS SELECT * FROM cohort_vector')",
        "except: pass",
        "try: con.execute('CREATE OR REPLACE VIEW signal_pairwise AS SELECT * FROM cohort_pairwise')",
        "except: pass",
        "try: con.execute('CREATE OR REPLACE VIEW information_flow AS SELECT * FROM cohort_information_flow')",
        "except: pass",
        "try: con.execute('CREATE OR REPLACE VIEW cohort_thermodynamics AS SELECT * FROM thermodynamics')",
        "except: pass",
        '',
        '# Results dict for chart cells',
        '_results = {}',
        '',
        "print(f'Loaded {len(con.execute(\"SELECT table_name FROM information_schema.tables\").fetchdf())} views')",
    ])
    return '\n'.join(lines)


def _report_cell(stem: str, title: str, sql_dir: Path) -> tuple[str | None, str | None]:
    """Read a SQL report file and return (sql_text, description).

    Returns (None, None) if file not found.
    """
    sql_path = sql_dir / f'{stem}.sql'
    if not sql_path.exists():
        return None, None
    sql_text = sql_path.read_text()

    # Extract description from header comments
    desc_lines = []
    for line in sql_text.split('\n'):
        stripped = line.strip()
        if stripped.startswith('--') and not stripped.startswith('-- ==='):
            desc_lines.append(stripped.lstrip('- ').strip())
        elif desc_lines and not stripped.startswith('--'):
            break
    description = ' '.join(desc_lines[:3]) if desc_lines else ''

    return sql_text, description


def _make_report_code(stem: str, sql_text: str) -> str:
    """Generate code cell that runs a SQL report and stores the result."""
    # Escape the SQL for embedding in a Python string
    escaped = sql_text.replace('\\', '\\\\').replace("'''", "\\'\\'\\'")
    lines = [
        f"# Execute {stem}",
        f"_sql = '''{escaped}'''",
        '',
        '_df = None',
        'for stmt in _sql.split(";"):',
        '    lines = [l for l in stmt.split("\\n") if not l.strip().startswith(".")]',
        '    cleaned = "\\n".join(lines).strip()',
        '    if not cleaned:',
        '        continue',
        '    has_sql = any(l.strip() and not l.strip().startswith("--") for l in cleaned.split("\\n"))',
        '    if not has_sql:',
        '        continue',
        '    try:',
        '        result = con.execute(cleaned)',
        '        upper = cleaned.upper().lstrip()',
        '        if upper.startswith("SELECT") or (not upper.startswith("CREATE") and "FROM" in upper):',
        '            _df = result.fetchdf()',
        '    except Exception as e:',
        '        pass  # View deps may be missing',
        '',
        f"_results['{stem}'] = _df",
        'if _df is not None and len(_df) > 0:',
        '    display(_df.head(50))',
        '    print(f"\\n{len(_df):,} rows")',
        'else:',
        '    print("No results (views only or no data)")',
    ]
    return '\n'.join(lines)


def generate_notebook(run_dir: Path, domain_dir: Path | None = None) -> Path:
    """Generate an analysis notebook for a Manifold run.

    Args:
        run_dir: Output directory (e.g. domain/output_time/).
        domain_dir: Domain root. Defaults to run_dir.parent.

    Returns:
        Path to the generated .ipynb file.
    """
    domain_dir = domain_dir or run_dir.parent
    sql_dir = Path(__file__).parent.parent / 'sql' / 'reports'

    parquets = _discover_parquets(run_dir, domain_dir)
    cells = []

    # Title cell
    cells.append(_make_cell('markdown', [
        f'# Analysis Notebook',
        f'',
        f'**Run directory**: `{run_dir}`  ',
        f'**Domain**: `{domain_dir}`  ',
        f'**Parquets loaded**: {len(parquets)}',
    ]))

    # Setup cell
    cells.append(_make_cell('code', _setup_cell(parquets)))

    # Physics compat cell
    physics_sql_path = Path(__file__).parent.parent / 'sql' / 'layers' / '00_physics_compat.sql'
    if physics_sql_path.exists():
        physics_sql = physics_sql_path.read_text()
        cells.append(_make_cell('markdown', ['## Physics Compatibility View']))
        cells.append(_make_cell('code', [
            '# Create physics compatibility view from cohort_geometry + cohort_vector',
            'try:',
            f"    con.execute('''{physics_sql}''')",
            "    print('Physics view created')",
            'except Exception as e:',
            f"    print(f'Physics view skipped: {{e}}')",
        ]))

    # Report cells
    for stem, title in REPORTS:
        sql_text, description = _report_cell(stem, title, sql_dir)
        if sql_text is None:
            continue

        # Markdown header
        header_lines = [f'## {title}']
        if description:
            header_lines.extend(['', description])
        cells.append(_make_cell('markdown', header_lines))

        # Code cell running the SQL
        cells.append(_make_cell('code', _make_report_code(stem, sql_text)))

        # Chart cell (if this report has one)
        if stem in CHARTS:
            chart_title, chart_code = CHARTS[stem]
            cells.append(_make_cell('markdown', [f'### {chart_title}']))
            cells.append(_make_cell('code',
                                    chart_code.format(stem=stem)))

    # Build notebook JSON
    notebook = {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.11.0',
            },
        },
        'cells': cells,
    }

    output_path = run_dir / 'analysis_notebook.ipynb'
    output_path.write_text(json.dumps(notebook, indent=1))
    return output_path


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python -m prime.notebook.generator <run_dir>")
        print("  e.g. python -m prime.notebook.generator ~/domains/cmapss/FD_001/train/output_time")
        sys.exit(1)

    run_dir = Path(sys.argv[1]).resolve()
    if not run_dir.exists():
        print(f"ERROR: {run_dir} does not exist")
        sys.exit(1)

    output = generate_notebook(run_dir)
    print(f"Notebook written to {output}")
