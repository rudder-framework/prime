"""
ORTHON Streamlit Application
============================

User-facing interface for PRISM analysis.

1. Upload data
2. Profile & recommend config
3. User confirms/edits
4. Run PRISM (with progress)
5. Display results + ML discovery

PRISM computes. ORTHON interprets.
"""

import sys
from pathlib import Path
import tempfile

try:
    import streamlit as st
except ImportError:
    print("ERROR: streamlit required. pip install streamlit")
    print("       Or: pip install orthon[ui]")
    sys.exit(1)

import polars as pl

from orthon.data_reader import DataReader
from orthon.config.recommender import ConfigRecommender
from orthon.compute.prism_runner import run_prism
from orthon.ml.discovery import DiscoveryEngine


# Page config
st.set_page_config(page_title="ORTHON", page_icon="🔬", layout="wide")


# =============================================================================
# SESSION STATE
# =============================================================================
if 'step' not in st.session_state:
    st.session_state.step = 'upload'
if 'data_path' not in st.session_state:
    st.session_state.data_path = None
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'config' not in st.session_state:
    st.session_state.config = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'output_dir' not in st.session_state:
    st.session_state.output_dir = None


# =============================================================================
# HEADER
# =============================================================================
st.title("ORTHON")
st.caption("Diagnostic interpreter for PRISM outputs")


# =============================================================================
# STEP 1: UPLOAD
# =============================================================================
if st.session_state.step == 'upload':
    st.header("Upload Data")

    uploaded = st.file_uploader(
        "Time series data (CSV, Parquet, TSV)",
        type=['csv', 'parquet', 'tsv']
    )

    if uploaded:
        # Save temp file
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(uploaded.getvalue())
            st.session_state.data_path = Path(f.name)

        # Profile
        reader = DataReader()
        reader.read(st.session_state.data_path)
        st.session_state.profile = reader.profile_data()

        p = st.session_state.profile
        st.success(f"Loaded {p.n_rows:,} rows from {uploaded.name}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Entities", p.n_entities)
        col2.metric("Signals", p.n_signals)
        col3.metric("Median Lifecycle", f"{p.median_lifecycle:.0f}")
        col4.metric("Nulls", f"{p.null_pct:.1f}%")

        with st.expander("Signal names"):
            st.write(p.signal_names)

        if st.button("Configure", type="primary"):
            st.session_state.step = 'config'
            st.rerun()


# =============================================================================
# STEP 2: CONFIGURE
# =============================================================================
elif st.session_state.step == 'config':
    st.header("Configuration")

    profile = st.session_state.profile
    recommender = ConfigRecommender(profile)
    rec = recommender.recommend()

    # Show recommendation rationale
    confidence_colors = {'high': 'green', 'medium': 'orange', 'low': 'red'}
    st.markdown(f"**Confidence:** :{confidence_colors[rec.window.confidence]}[{rec.window.confidence.upper()}]")

    with st.expander("Recommendation rationale", expanded=True):
        st.markdown(rec.window.rationale)

    # Editable config
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Windowing")
        window_size = st.number_input(
            "Window Size",
            min_value=5,
            max_value=int(profile.max_lifecycle * 0.9),
            value=rec.window.window_size,
            help="Number of samples per window"
        )

        window_stride = st.number_input(
            "Window Stride",
            min_value=1,
            max_value=window_size,
            value=rec.window.window_stride,
            help="Samples between window starts"
        )

        overlap = (1 - window_stride / window_size) * 100 if window_size > 0 else 0
        st.caption(f"Overlap: {overlap:.0f}%")

    with col2:
        st.subheader("Clustering")
        n_clusters = st.number_input(
            "Clusters",
            min_value=2,
            max_value=10,
            value=rec.n_clusters,
            help="For geometry layer"
        )

        n_regimes = st.number_input(
            "Regimes",
            min_value=2,
            max_value=10,
            value=rec.n_regimes,
            help="For dynamics layer"
        )

    # Preview
    n_windows = max(1, int((profile.median_lifecycle - window_size) / window_stride) + 1)
    min_win = max(1, int((profile.min_lifecycle - window_size) / window_stride) + 1)

    st.divider()
    st.markdown(f"**Preview:** ~{n_windows} windows per entity (median), shortest entity gets ~{min_win}")

    if min_win < 5:
        st.warning("Shortest entity has few windows. Consider smaller window size.")

    # Navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back"):
            st.session_state.step = 'upload'
            st.rerun()

    with col2:
        if st.button("Run PRISM", type="primary"):
            st.session_state.config = {
                'window_size': window_size,
                'window_stride': window_stride,
                'n_clusters': n_clusters,
                'n_regimes': n_regimes,
            }
            st.session_state.step = 'compute'
            st.rerun()


# =============================================================================
# STEP 3: COMPUTE (with progress)
# =============================================================================
elif st.session_state.step == 'compute':
    st.header("Computing")

    # Stage weights for progress calculation
    STAGE_WEIGHTS = {
        'init': 2,
        'load': 3,
        'characterize': 10,
        'vector': 40,
        'geometry': 15,
        'dynamics': 20,
        'physics': 10,
    }

    status_box = st.status("Starting PRISM...", expanded=True)
    progress_bar = st.progress(0)
    stage_text = st.empty()

    stage_progress = {k: 0 for k in STAGE_WEIGHTS}
    output_dir = Path(tempfile.mkdtemp(prefix='orthon_results_'))
    st.session_state.output_dir = output_dir

    try:
        for update in run_prism(
            st.session_state.data_path,
            st.session_state.config,
            output_dir
        ):
            stage = update.get('stage', '')
            message = update.get('message', '')
            progress = update.get('progress', 0)

            if stage in stage_progress:
                stage_progress[stage] = progress

            # Calculate global progress
            total_weight = sum(STAGE_WEIGHTS.values())
            weighted_progress = sum(
                STAGE_WEIGHTS.get(s, 0) * stage_progress.get(s, 0) / 100
                for s in STAGE_WEIGHTS
            )
            global_progress = int(100 * weighted_progress / total_weight)

            progress_bar.progress(min(global_progress, 99))
            stage_text.markdown(f"**{stage}:** {message}")

            if stage == 'complete':
                status_box.update(label="Complete!", state="complete")
                progress_bar.progress(100)

                # Load results
                results = {}
                for name in ['observations', 'vector', 'geometry', 'dynamics', 'physics']:
                    parquet_path = output_dir / f'{name}.parquet'
                    if parquet_path.exists():
                        results[name] = pl.read_parquet(parquet_path)

                st.session_state.results = results

                if st.button("View Results", type="primary"):
                    st.session_state.step = 'results'
                    st.rerun()

    except ImportError as e:
        st.error(str(e))
        if st.button("Back to Config"):
            st.session_state.step = 'config'
            st.rerun()

    except Exception as e:
        st.error(f"PRISM failed: {e}")
        if st.button("Back to Config"):
            st.session_state.step = 'config'
            st.rerun()


# =============================================================================
# STEP 4: RESULTS
# =============================================================================
elif st.session_state.step == 'results':
    st.header("Results")

    results = st.session_state.results

    if not results:
        st.warning("No results available")
        if st.button("Start Over"):
            st.session_state.step = 'upload'
            st.rerun()
    else:
        # Tabs for different views
        tabs = st.tabs(["Summary", "Dynamics", "Physics", "Geometry", "ML Discovery"])

        # SUMMARY TAB
        with tabs[0]:
            col1, col2, col3 = st.columns(3)

            if 'dynamics' in results and not results['dynamics'].is_empty():
                dyn = results['dynamics']
                if 'hd_slope' in dyn.columns:
                    mean_slope = dyn['hd_slope'].mean()
                    col1.metric("Mean hd_slope", f"{mean_slope:.4f}")

                    degrading = (dyn['hd_slope'] < -0.01).sum()
                    col2.metric("Degrading", f"{degrading}/{len(dyn)}")

                    critical = (dyn['hd_slope'] < -0.05).sum()
                    col3.metric("Critical", f"{critical}/{len(dyn)}")

            if 'geometry' in results and not results['geometry'].is_empty():
                geo = results['geometry']
                if 'effective_dimensionality' in geo.columns:
                    eff_dim = geo['effective_dimensionality'].mean()
                    st.metric("Effective Dimensionality", f"{eff_dim:.2f}")

        # DYNAMICS TAB
        with tabs[1]:
            st.subheader("Dynamics")
            if 'dynamics' in results and not results['dynamics'].is_empty():
                st.dataframe(results['dynamics'].to_pandas(), use_container_width=True)
            else:
                st.info("No dynamics data available")

        # PHYSICS TAB
        with tabs[2]:
            st.subheader("Physics")
            if 'physics' in results and not results['physics'].is_empty():
                st.dataframe(results['physics'].to_pandas(), use_container_width=True)
            else:
                st.info("No physics data available")

        # GEOMETRY TAB
        with tabs[3]:
            st.subheader("Geometry")
            if 'geometry' in results and not results['geometry'].is_empty():
                st.dataframe(results['geometry'].to_pandas(), use_container_width=True)
            else:
                st.info("No geometry data available")

        # ML DISCOVERY TAB
        with tabs[4]:
            st.subheader("ML Discovery")
            st.markdown("*Finding interesting patterns in dynamics + physics...*")

            dynamics = results.get('dynamics', pl.DataFrame())
            physics = results.get('physics', pl.DataFrame())

            engine = DiscoveryEngine(dynamics, physics)
            findings = engine.discover()

            if findings:
                for i, finding in enumerate(findings, 1):
                    severity_icons = {
                        'critical': '🔴',
                        'warning': '🟡',
                        'info': '🔵'
                    }
                    icon = severity_icons.get(finding.get('severity', 'info'), '🔵')

                    with st.expander(f"{icon} {finding['title']}", expanded=(i <= 3)):
                        st.markdown(finding['description'])
                        if finding.get('recommendation'):
                            st.info(f"Recommendation: {finding['recommendation']}")
            else:
                st.info("No significant patterns detected.")

        # Export / Navigation
        st.divider()
        col1, col2 = st.columns(2)

        with col1:
            if st.button("New Analysis"):
                st.session_state.step = 'upload'
                st.session_state.results = None
                st.session_state.data_path = None
                st.session_state.profile = None
                st.session_state.config = None
                st.rerun()

        with col2:
            if st.session_state.output_dir:
                st.caption(f"Results saved to: {st.session_state.output_dir}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Entry point for streamlit run."""
    # Streamlit runs this file directly, so this is just for CLI
    import subprocess
    subprocess.run(['streamlit', 'run', __file__])


if __name__ == '__main__':
    # When run directly, just let Streamlit handle it
    pass
