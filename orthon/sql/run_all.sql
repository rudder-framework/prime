-- ============================================================================
-- ORTHON SQL: run_all.sql
-- ============================================================================
-- Main entry point for ORTHON SQL execution
--
-- ORTHON's role:
--   1. Load observations → observations.parquet (ONLY file ORTHON creates)
--   2. Classify by units → work orders for PRISM
--   3. Load PRISM results → visualization views
--
-- ORTHON does NOT compute. PRISM computes everything.
-- ============================================================================

-- =========================================================================
-- PHASE 1: OBSERVATIONS (run on upload)
-- =========================================================================
-- Creates observations.parquet from uploaded data
-- Parameters: {input_path}, {output_path}

.read sql/00_observations.sql

-- =========================================================================
-- PHASE 2: CLASSIFICATION (run after observations)
-- =========================================================================
-- Classifies signals by UNIT only (no compute)
-- Generates PRISM work orders

.read sql/01_classification_units.sql
.read sql/02_work_orders.sql

SELECT '=== ORTHON: SIGNAL CLASSIFICATION (by unit) ===' AS section;
SELECT * FROM v_signal_class_summary;

SELECT '=== ORTHON: PRISM WORK ORDERS ===' AS section;
SELECT * FROM v_work_order_summary;

-- At this point, ORTHON sends work orders to PRISM and waits.
-- PRISM creates: signal_typology.parquet, behavioral_geometry.parquet,
--                dynamical_systems.parquet, causal_mechanics.parquet

-- =========================================================================
-- PHASE 3: VISUALIZATION (run after PRISM completes)
-- =========================================================================
-- Load PRISM results and create visualization views
-- Parameters: {prism_output}

.read sql/03_load_prism_results.sql
.read sql/04_visualization.sql
.read sql/05_summaries.sql
.read sql/06_general_views.sql

SELECT '=== ORTHON: PRISM RESULTS LOADED ===' AS section;
SELECT * FROM v_summary_all_layers;

SELECT '=== ORTHON: SYSTEM HEALTH ===' AS section;
SELECT * FROM v_dashboard_system_health;

SELECT '=== ORTHON: ALERTS ===' AS section;
SELECT * FROM v_dashboard_alerts LIMIT 10;

SELECT '=== ORTHON: DATASET OVERVIEW ===' AS section;
SELECT * FROM v_dataset_overview;

SELECT '=== ORTHON: INSIGHT CARDS ===' AS section;
SELECT * FROM v_insight_cards;

-- =========================================================================
-- VERIFY
-- =========================================================================
SELECT '=== ORTHON SQL Complete ===' AS status;

SELECT
    'observations' AS table_name,
    (SELECT COUNT(*) FROM observations) AS row_count
UNION ALL
SELECT
    'signal_typology',
    (SELECT COUNT(*) FROM signal_typology)
UNION ALL
SELECT
    'behavioral_geometry',
    (SELECT COUNT(*) FROM behavioral_geometry)
UNION ALL
SELECT
    'dynamical_systems',
    (SELECT COUNT(*) FROM dynamical_systems)
UNION ALL
SELECT
    'causal_mechanics',
    (SELECT COUNT(*) FROM causal_mechanics);
