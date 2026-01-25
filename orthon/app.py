"""
Orthon — Drop Data, Get Physics

MVP Streamlit app:
1. Instructions to prepare data
2. Upload file
3. Report on what was uploaded
4. Download results back

Works with or without PRISM.
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from orthon.intake import load_file, validate, detect_columns
from orthon.backend import get_backend, analyze, get_backend_info
from orthon.display import generate_report, to_json, to_csv, render_results_page
from orthon.display.report import format_signals_table
from orthon.shared import DISCIPLINES

st.set_page_config(page_title="Orthon", page_icon="⚡", layout="wide")


# =============================================================================
# UI
# =============================================================================

st.title("⚡ Orthon")
st.caption("Drop data. Get physics.")

# Show backend status in sidebar
with st.sidebar:
    st.subheader("Backend")
    backend_info = get_backend_info()
    if backend_info['has_physics']:
        st.success(f"✅ {backend_info['name']}")
    else:
        st.warning("📊 Basic mode")
    st.caption(backend_info['message'])

    st.divider()
    st.caption("Orthon — *Systems lose coherence before they fail*")

tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📤 Upload", "📊 Results"])

# -----------------------------------------------------------------------------
# INSTRUCTIONS
# -----------------------------------------------------------------------------

with tab1:
    st.header("How to Prepare Your Data")

    st.markdown("""
### Quick Start

**Name your columns with units. We figure out the rest.**

```csv
timestamp, flow_gpm, pressure_psi, temp_F
2024-01-01 08:00, 50, 120, 150
2024-01-01 08:01, 51, 121, 151
```

---

### Unit Suffixes

| Measurement | Suffixes |
|-------------|----------|
| Pressure | `_psi`, `_bar`, `_kpa` |
| Temperature | `_F`, `_C`, `_K`, `_degF`, `_degR` |
| Flow | `_gpm`, `_lpm` |
| Length | `_in`, `_ft`, `_m`, `_mm` |
| Speed | `_rpm`, `_hz` |
| Electrical | `_V`, `_A`, `_W`, `_kW` |

---

### Multiple Equipment

Add an `entity_id` column:

```csv
entity_id, diameter_in, flow_gpm, pressure_psi
P-101, 4, 50, 120
P-101, 4, 51, 121
P-102, 6, 100, 115
```

We detect:
- `entity_id` → grouping column
- `diameter_in` → constant per entity
- `flow_gpm`, `pressure_psi` → signals

---

### Supported Formats

- CSV ✅
- Parquet ✅
- TSV ✅
""")

    # Show what's available based on backend
    st.divider()
    if backend_info['has_physics']:
        st.success("**Full Analysis Available**")
        st.markdown("""
With PRISM, you get:
- hd_slope (degradation rate)
- Transfer entropy
- Hamiltonian / Lagrangian
- Reynolds number
- Spectral analysis
- And 60+ more metrics
""")
    else:
        st.info("**Basic Analysis Mode**")
        st.markdown("""
Currently available:
- Data validation
- Unit detection
- Basic statistics
- Quality warnings

*Start PRISM on port 8100 for full physics analysis*
""")

# -----------------------------------------------------------------------------
# UPLOAD
# -----------------------------------------------------------------------------

with tab2:
    st.header("Upload Your Data")

    uploaded = st.file_uploader("CSV, Excel, Parquet, or TSV", type=['csv', 'xlsx', 'xls', 'parquet', 'tsv', 'txt'])

    if uploaded:
        try:
            # Load file
            df = load_file(uploaded, filename=uploaded.name)

            st.session_state['df'] = df
            st.session_state['filename'] = uploaded.name

            st.success(f"✅ `{uploaded.name}` — {len(df):,} rows × {len(df.columns)} columns")
            st.dataframe(df.head(10), use_container_width=True)

            # Discipline selection (optional)
            st.subheader("Analysis Options")
            discipline_options = {"(Core Engines Only)": None}
            for key, info in DISCIPLINES.items():
                discipline_options[f"{info['icon']} {info['name']} — {info['description']}"] = key

            selected_discipline_label = st.selectbox(
                "Discipline (optional)",
                options=list(discipline_options.keys()),
                help="Select a discipline for specialized physics engines. Leave as Core for universal diagnostics."
            )
            selected_discipline = discipline_options[selected_discipline_label]

            if selected_discipline:
                discipline_info = DISCIPLINES[selected_discipline]
                engine_count = discipline_info.get('engine_count', len(discipline_info['engines']))
                st.caption(f"{engine_count} engines: {', '.join(discipline_info['engines'][:8])}{'...' if len(discipline_info['engines']) > 8 else ''}")

                # Show requirements
                has_requirements = False
                if discipline_info.get('required_constants'):
                    st.info(f"**Required constants:** {', '.join(discipline_info['required_constants'])}")
                    has_requirements = True
                if discipline_info.get('required_signals'):
                    st.info(f"**Required signals:** {', '.join(discipline_info['required_signals'])}")
                    has_requirements = True
                if discipline_info.get('required_signals_any'):
                    st.info(f"**Needs at least one of:** {', '.join(discipline_info['required_signals_any'])}")
                    has_requirements = True
                if discipline_info.get('optional_constants'):
                    with st.expander("Optional constants (enable more engines)"):
                        st.write(', '.join(discipline_info['optional_constants']))
                if not has_requirements:
                    st.success("No special requirements - works with any numeric data")

            if st.button("🔍 Analyze", type="primary"):
                # Validate
                validation = validate(df)

                # Run backend analysis with discipline
                analysis_results, backend_name = analyze(df, discipline=selected_discipline)

                # Generate report
                report = generate_report(
                    validation=validation,
                    analysis=analysis_results,
                    filename=uploaded.name,
                )

                st.session_state['report'] = report
                st.session_state['validation'] = validation
                st.session_state['analysis'] = analysis_results
                st.session_state['discipline'] = selected_discipline

                # Store results path if PRISM provided one (for SQL results page)
                if analysis_results.get('results_path'):
                    st.session_state['results_path'] = analysis_results['results_path']

                st.success(f"Done! (using {backend_name} backend) See Results tab →")

        except Exception as e:
            st.error(f"Error: {e}")

# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------

with tab3:
    if 'report' not in st.session_state:
        st.header("Results")
        st.info("Upload a file and click Analyze first.")
    else:
        report = st.session_state['report']
        validation = st.session_state['validation']
        analysis = st.session_state.get('analysis', {})
        results_path = st.session_state.get('results_path')

        # Check if we have PRISM parquet results
        has_parquets = False
        if results_path:
            from pathlib import Path
            parquet_path = Path(results_path)
            if parquet_path.exists() and list(parquet_path.glob("*.parquet")):
                has_parquets = True

        # Show SQL-based results page if parquets available
        if has_parquets:
            render_results_page(results_path)
        else:
            # Basic results view (fallback or no parquets)
            st.header("Results")

            # Metrics
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rows", f"{report['summary']['rows']:,}")
            c2.metric("Columns", report['summary']['columns'])
            c3.metric("Signals", report['summary']['signals'])
            c4.metric("Backend", analysis.get('backend', 'unknown'))

            # Issues
            if report['issues']:
                st.error("**Issues**")
                for i in report['issues']:
                    st.write(f"❌ {i}")

            if report['warnings']:
                st.warning("**Warnings**")
                for w in report['warnings']:
                    st.write(f"⚠️ {w}")

            if not report['issues'] and not report['warnings']:
                st.success("Data looks good!")

            # Structure
            st.subheader("Structure")
            struct = report['structure']
            st.write(f"**Time:** `{struct['time_col'] or 'not detected'}`")
            st.write(f"**Entity:** `{struct['entity_col'] or 'single entity'}`")

            # Signals table
            st.subheader("Signals")
            signals_df = format_signals_table(report)
            if not signals_df.empty:
                # Format numeric columns
                for col in ['Min', 'Max', 'Mean']:
                    if col in signals_df.columns:
                        signals_df[col] = signals_df[col].apply(
                            lambda x: f"{x:.4g}" if pd.notna(x) else "-"
                        )
                st.dataframe(signals_df, hide_index=True, use_container_width=True)

            # Analysis details (if PRISM was used)
            if analysis.get('backend') not in ('fallback', None):
                st.subheader("Analysis Details")
                with st.expander("Full Analysis Results"):
                    st.json(analysis)

            # PRISM prompt (if using fallback)
            if analysis.get('backend') == 'fallback':
                st.divider()
                st.info("""
**Want full physics?** Start PRISM on port 8100:
- `hd_slope` — degradation rate detection
- `transfer_entropy` — causal relationships
- `hamiltonian` — energy analysis
- And 60+ physics-based metrics

```bash
cd prism && python -m prism.api
```
""")

        # Downloads (always show)
        st.divider()
        st.subheader("Download")

        c1, c2 = st.columns(2)
        c1.download_button(
            "Report (JSON)",
            data=to_json(report),
            file_name=f"orthon_report_{datetime.now():%Y%m%d_%H%M%S}.json",
            mime="application/json"
        )
        c2.download_button(
            "Signals (CSV)",
            data=to_csv(report),
            file_name=f"orthon_signals_{datetime.now():%Y%m%d_%H%M%S}.csv",
            mime="text/csv"
        )


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point."""
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', __file__])


if __name__ == '__main__':
    pass  # Streamlit runs this directly
