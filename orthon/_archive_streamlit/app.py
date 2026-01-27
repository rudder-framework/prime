"""
Orthon — Drop Data, Get Physics

Upload data → PRISM computes → View results via SQL

PRISM must be running on port 8100 for analysis.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from datetime import datetime

from orthon.intake import load_file, validate, detect_columns
from orthon.backend import get_backend_info, has_prism
from orthon.prism_client import get_prism_client, prism_available
from orthon.intake.transformer import transform_for_prism
from orthon.display import render_results_page
from orthon.shared import DISCIPLINES

st.set_page_config(page_title="Orthon", page_icon="⚡", layout="wide")


# =============================================================================
# SIDEBAR - PRISM STATUS
# =============================================================================

with st.sidebar:
    st.subheader("PRISM Status")

    if prism_available():
        st.success("✅ Connected (port 8100)")
    else:
        st.error("❌ Offline")
        st.code("cd prism && python -m prism.api", language="bash")
        st.caption("Start PRISM to enable analysis")

    st.divider()
    st.caption("*Systems lose coherence before they fail*")


# =============================================================================
# MAIN UI
# =============================================================================

st.title("⚡ Orthon")
st.caption("Drop data. Get physics.")

tab1, tab2, tab3 = st.tabs(["📖 Instructions", "📤 Upload & Analyze", "📊 Results"])


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

---

### Supported Formats

- CSV ✅
- Excel (.xlsx) ✅
- Parquet ✅
- TSV ✅
""")


# -----------------------------------------------------------------------------
# UPLOAD & ANALYZE
# -----------------------------------------------------------------------------

with tab2:
    st.header("Upload & Analyze")

    uploaded = st.file_uploader(
        "Drop your data file",
        type=['csv', 'xlsx', 'xls', 'parquet', 'tsv', 'txt']
    )

    if uploaded:
        try:
            # Load and preview
            df = load_file(uploaded, filename=uploaded.name)
            st.session_state['df'] = df
            st.session_state['filename'] = uploaded.name

            st.success(f"✅ `{uploaded.name}` — {len(df):,} rows × {len(df.columns)} columns")

            with st.expander("Preview data", expanded=True):
                st.dataframe(df.head(10), use_container_width=True)

            # Discipline selection
            st.subheader("Discipline")

            discipline_options = {"(Auto-detect / Core engines)": None}
            for key, info in DISCIPLINES.items():
                discipline_options[f"{info['icon']} {info['name']}"] = key

            selected_label = st.selectbox(
                "Select discipline",
                options=list(discipline_options.keys()),
            )
            selected_discipline = discipline_options[selected_label]

            if selected_discipline:
                info = DISCIPLINES[selected_discipline]
                st.caption(f"{info['description']}")
                st.caption(f"{info.get('engine_count', len(info['engines']))} engines available")

            # Analyze button
            st.divider()

            if not prism_available():
                st.warning("⚠️ PRISM is offline. Start it to analyze.")
                st.code("cd prism && python -m prism.api", language="bash")
                analyze_disabled = True
            else:
                analyze_disabled = False

            if st.button("🔬 Run Analysis", type="primary", disabled=analyze_disabled):

                with st.spinner("PRISM is computing... this may take a moment"):

                    # Transform data to PRISM format
                    observations_df, config = transform_for_prism(df, discipline=selected_discipline)

                    # Create temp directory for data exchange
                    import tempfile
                    tmpdir = tempfile.mkdtemp(prefix="orthon_")
                    tmpdir = Path(tmpdir)

                    # Write observations
                    obs_path = tmpdir / "observations.parquet"
                    observations_df.to_parquet(obs_path)

                    # Results directory
                    results_dir = tmpdir / "results"
                    results_dir.mkdir(exist_ok=True)

                    # Call PRISM
                    client = get_prism_client()
                    response = client.compute(
                        config=config,
                        observations_path=str(obs_path),
                        output_dir=str(results_dir),
                    )

                # Handle response
                if response.get("status") == "complete":
                    results_path = response.get("results_path", str(results_dir))

                    # Verify parquets exist
                    parquets = list(Path(results_path).glob("*.parquet"))

                    if parquets:
                        st.session_state['results_path'] = results_path
                        st.session_state['results_ready'] = True
                        st.session_state['compute_status'] = 'complete'
                        st.success(f"✅ Analysis complete! {len(parquets)} output tables. See Results tab →")
                    else:
                        st.warning("PRISM completed but no parquets found.")
                        st.session_state['compute_status'] = 'no_parquets'

                elif response.get("status") == "error":
                    st.error(f"❌ PRISM error: {response.get('message', 'Unknown')}")
                    if response.get("hint"):
                        st.info(f"💡 {response['hint']}")
                    st.session_state['compute_status'] = 'error'
                    st.session_state['compute_error'] = response.get('message')
                else:
                    st.error(f"Unexpected response from PRISM")
                    st.json(response)

        except Exception as e:
            st.error(f"Error loading file: {e}")


# -----------------------------------------------------------------------------
# RESULTS
# -----------------------------------------------------------------------------

with tab3:
    st.header("Results")

    # Check state
    results_ready = st.session_state.get('results_ready', False)
    results_path = st.session_state.get('results_path')
    compute_status = st.session_state.get('compute_status')

    if results_ready and results_path:
        # Verify parquets still exist
        parquet_path = Path(results_path)
        if parquet_path.exists() and list(parquet_path.glob("*.parquet")):
            render_results_page(results_path)
        else:
            st.warning("Results directory no longer exists. Re-run analysis.")
            st.session_state['results_ready'] = False

    elif compute_status == 'error':
        st.error(f"Last analysis failed: {st.session_state.get('compute_error', 'Unknown error')}")
        st.info("Fix the issue and run analysis again from the Upload tab.")

    elif compute_status == 'no_parquets':
        st.warning("PRISM completed but produced no output parquets.")
        st.info("Check PRISM logs for details.")

    else:
        # No results yet
        st.info("No results yet.")

        if not prism_available():
            st.warning("PRISM is offline. Start it first:")
            st.code("cd prism && python -m prism.api", language="bash")
        else:
            st.success("PRISM is ready. Upload data and click Analyze.")


# =============================================================================
# CLI ENTRY POINT
# =============================================================================

def main():
    """CLI entry point."""
    import subprocess
    import sys
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', __file__])


if __name__ == '__main__':
    pass
