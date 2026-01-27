"""
ORTHON Results Page
===================

Display PRISM results via DuckDB SQL queries on parquets.
"""

import streamlit as st
import pandas as pd
import io
import zipfile
from pathlib import Path
from typing import Optional

from .db import ResultsDB


# =============================================================================
# PRESET QUERIES
# =============================================================================

PRESET_QUERIES = {
    "Select preset...": "",

    "Degrading entities (hd_slope > 0)": """
SELECT entity_id,
       AVG(hd_slope) as avg_hd_slope,
       COUNT(*) as windows
FROM dynamics
GROUP BY entity_id
HAVING AVG(hd_slope) > 0
ORDER BY avg_hd_slope DESC""",

    "Signal statistics": """
SELECT signal_id,
       AVG(mean) as avg_mean,
       AVG(std) as avg_std,
       AVG(entropy) as avg_entropy
FROM vector
GROUP BY signal_id
ORDER BY signal_id""",

    "Regime transitions": """
SELECT entity_id, window, regime, regime_prob
FROM dynamics
WHERE is_transition = true
ORDER BY entity_id, window""",

    "Full picture at window 0": """
SELECT v.entity_id, v.signal_id, v.slope, v.entropy,
       g.baseline_distance, g.effective_dim,
       d.hd_slope, d.regime
FROM vector v
LEFT JOIN geometry g ON v.entity_id = g.entity_id AND v.window = g.window
LEFT JOIN dynamics d ON v.entity_id = d.entity_id AND v.window = d.window
WHERE v.window = 0""",

    "Correlation structure": """
SELECT entity_id, window,
       corr_largest_eigenvalue,
       corr_condition_number,
       effective_dim
FROM geometry
ORDER BY entity_id, window""",

    "High entropy signals": """
SELECT entity_id, signal_id, window, entropy
FROM vector
WHERE entropy > 3.0
ORDER BY entropy DESC
LIMIT 100""",

    "Negative slopes (decreasing)": """
SELECT entity_id, signal_id, window, slope
FROM vector
WHERE slope < 0
ORDER BY slope ASC
LIMIT 100""",
}


# =============================================================================
# MAIN RENDER FUNCTION
# =============================================================================

def render_results_page(results_path: str):
    """Main results page with SQL tables"""

    st.header("Results")

    # Initialize DB
    db = ResultsDB(results_path)
    tables = db.tables()

    if not tables:
        st.error("No results found. Run analysis first.")
        return

    # Show available tables
    st.caption(f"Tables: {', '.join(tables)}")

    # Tabs for each table
    tab_names = ["Summary"] + [t.title() for t in tables]
    tabs = st.tabs(tab_names)

    # Summary tab
    with tabs[0]:
        render_summary(db)

    # Table tabs
    for i, table in enumerate(tables):
        with tabs[i + 1]:
            render_table_view(db, table)

    # Custom SQL section
    st.divider()
    render_sql_editor(db)

    # Export section
    st.divider()
    render_export(db, results_path)


# =============================================================================
# SUMMARY TAB
# =============================================================================

def render_summary(db: ResultsDB):
    """Summary statistics and health checks"""

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    entities = db.entities()
    signals = db.signals()
    windows = db.windows()

    col1.metric("Entities", len(entities))
    col2.metric("Signals", len(signals))
    col3.metric("Windows", len(windows))
    col4.metric("Tables", len(db.tables()))

    # Degradation check
    st.subheader("Degradation Check")

    if 'dynamics' in db.tables():
        degrading = db.query("""
            SELECT entity_id,
                   AVG(hd_slope) as avg_hd_slope,
                   COUNT(*) as windows
            FROM dynamics
            GROUP BY entity_id
            HAVING AVG(hd_slope) > 0
            ORDER BY avg_hd_slope DESC
        """)

        if isinstance(degrading, pd.DataFrame) and len(degrading) > 0:
            st.warning(f"{len(degrading)} entities showing positive hd_slope (degradation)")
            st.dataframe(degrading, use_container_width=True, hide_index=True)
        else:
            st.success("No entities showing degradation (all hd_slope <= 0)")
    else:
        st.info("Dynamics table not available")

    # Table row counts
    st.subheader("Data Overview")

    table_info = []
    for table in db.tables():
        schema = db.schema(table)
        table_info.append({
            "Table": table,
            "Rows": db.row_count(table),
            "Columns": len(schema),
        })

    if table_info:
        st.dataframe(pd.DataFrame(table_info), use_container_width=True, hide_index=True)


# =============================================================================
# TABLE VIEW
# =============================================================================

def render_table_view(db: ResultsDB, table: str):
    """View a single table with filters"""

    st.subheader(f"{table}")

    # Schema expander
    with st.expander("Schema"):
        schema = db.schema(table)
        schema_df = pd.DataFrame(schema, columns=["Column", "Type"])
        st.dataframe(schema_df, use_container_width=True, hide_index=True)

    # Get column names for this table
    columns = db.column_names(table)

    # Filters
    col1, col2, col3 = st.columns(3)

    # Entity filter (if entity_id exists)
    selected_entity = "All"
    if "entity_id" in columns:
        entities = db.entities()
        selected_entity = col1.selectbox(
            "Entity",
            ["All"] + entities,
            key=f"{table}_entity"
        )

    # Signal filter (if signal_id exists)
    selected_signal = "All"
    if "signal_id" in columns:
        signals = db.signals()
        selected_signal = col2.selectbox(
            "Signal",
            ["All"] + signals,
            key=f"{table}_signal"
        )

    # Limit
    limit = col3.number_input("Limit", 10, 10000, 100, key=f"{table}_limit")

    # Build query
    where_clauses = []
    if selected_entity != "All":
        where_clauses.append(f"entity_id = '{selected_entity}'")
    if selected_signal != "All":
        where_clauses.append(f"signal_id = '{selected_signal}'")

    where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

    query = f"SELECT * FROM {table} {where} LIMIT {limit}"

    # Show query
    with st.expander("Query"):
        st.code(query, language="sql")

    # Execute and display
    result = db.query(query)
    if isinstance(result, pd.DataFrame):
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.caption(f"{len(result)} rows shown")
    else:
        st.error(f"Query failed: {result[1] if isinstance(result, tuple) else 'Unknown error'}")


# =============================================================================
# SQL EDITOR
# =============================================================================

def render_sql_editor(db: ResultsDB):
    """Custom SQL editor with presets"""

    st.subheader("Custom SQL")

    # Preset selector
    preset = st.selectbox("Preset queries", list(PRESET_QUERIES.keys()))

    # Default query
    if preset != "Select preset...":
        default_query = PRESET_QUERIES[preset]
    else:
        default_query = "SELECT * FROM vector LIMIT 10"

    # Query input
    query = st.text_area(
        "SQL Query",
        value=default_query,
        height=150,
        key="sql_query"
    )

    # Run button
    if st.button("Run Query", type="primary"):
        result = db.query(query)

        if isinstance(result, pd.DataFrame):
            st.dataframe(result, use_container_width=True, hide_index=True)
            st.caption(f"{len(result)} rows returned")

            # Store for export
            st.session_state['last_query_result'] = result
            st.session_state['last_query'] = query
        else:
            error_msg = result[1] if isinstance(result, tuple) else "Unknown error"
            st.error(f"Query error: {error_msg}")


# =============================================================================
# EXPORT
# =============================================================================

def render_export(db: ResultsDB, results_path: str):
    """Export options"""

    st.subheader("Export")

    col1, col2 = st.columns(2)

    # ZIP download
    with col1:
        zip_data = create_zip(results_path)
        if zip_data:
            st.download_button(
                "Download All Parquets (ZIP)",
                data=zip_data,
                file_name="orthon_results.zip",
                mime="application/zip"
            )
        else:
            st.info("No parquet files to export")

    # CSV export of last query
    with col2:
        if 'last_query_result' in st.session_state:
            csv_data = st.session_state['last_query_result'].to_csv(index=False)
            st.download_button(
                "Download Last Query (CSV)",
                data=csv_data,
                file_name="query_result.csv",
                mime="text/csv"
            )
        else:
            st.info("Run a query to enable CSV export")


def create_zip(results_path: str) -> Optional[bytes]:
    """Create ZIP of all parquets"""
    path = Path(results_path)
    parquets = list(path.glob("*.parquet"))

    if not parquets:
        return None

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for parquet in parquets:
            zf.write(parquet, parquet.name)

    buffer.seek(0)
    return buffer.getvalue()
