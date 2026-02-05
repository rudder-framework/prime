-- ============================================================
-- ORTHON/PRISM Data Loading
-- Run first to load all parquet files into tables
-- ============================================================

-- Observations (raw signal data)
CREATE OR REPLACE TABLE observations AS
SELECT * FROM read_parquet('observations.parquet');

-- Typology (signal classifications)
CREATE OR REPLACE TABLE typology AS
SELECT * FROM read_parquet('typology.parquet');

-- Signal Vector (computed features per window)
CREATE OR REPLACE TABLE signal_vector AS
SELECT * FROM read_parquet('signal_vector.parquet');

-- Typology Raw (pre-classification metrics) - optional
CREATE OR REPLACE TABLE typology_raw AS
SELECT * FROM read_parquet('typology_raw.parquet');

-- Summary
SELECT 'observations' as table_name, COUNT(*) as rows FROM observations
UNION ALL
SELECT 'typology', COUNT(*) FROM typology
UNION ALL
SELECT 'signal_vector', COUNT(*) FROM signal_vector;
