-- ============================================================================
-- VIEW ALIASES: Old Names → Current Parquet Stems
-- ============================================================================
-- The Manifold output directory changed from flat system/ layout to nested
-- signal/, cohort/, system/ subdirectories with renamed parquets.
--
-- This layer maps legacy SQL view names to their current equivalents so
-- downstream SQL files continue to work without modification.
--
-- Alias                    → Source parquet (loaded by runner.py)
-- ─────────────────────────────────────────────────────────────────
-- state_geometry            → cohort_geometry
-- state_vector              → cohort_vector
-- signal_pairwise           → cohort_pairwise
-- information_flow          → cohort_information_flow
-- cohort_thermodynamics     → thermodynamics
-- ============================================================================

CREATE OR REPLACE VIEW state_geometry AS
SELECT * FROM cohort_geometry;

CREATE OR REPLACE VIEW state_vector AS
SELECT * FROM cohort_vector;

CREATE OR REPLACE VIEW signal_pairwise AS
SELECT * FROM cohort_pairwise;

CREATE OR REPLACE VIEW information_flow AS
SELECT * FROM cohort_information_flow;

CREATE OR REPLACE VIEW cohort_thermodynamics AS
SELECT * FROM thermodynamics;
