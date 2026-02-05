-- ============================================================================
-- ORTHON SQL: 02_work_orders.sql
-- ============================================================================
-- Generate PRISM work orders based on signal classification
--
-- ORTHON tells PRISM what to compute for each signal.
-- PRISM executes and returns results in parquet files.
--
-- Work order structure:
--   - signal_id: which signal
--   - compute_*: boolean flags for what PRISM should compute
--   - priority: execution order
-- ============================================================================

-- Per-signal work orders
CREATE OR REPLACE VIEW v_prism_work_orders AS
SELECT
    s.signal_id,
    s.signal_class,
    s.quantity,
    s.calculus_valid,
    s.interpolation_valid,

    -- =========================================================================
    -- LAYER 1: SIGNAL TYPOLOGY (WHAT)
    -- =========================================================================
    -- Hurst exponent: persistence/anti-persistence (analog/periodic only)
    s.calculus_valid AS compute_hurst,

    -- Entropy: information content (all signals)
    TRUE AS compute_entropy,

    -- Stationarity: mean/variance stability (analog/periodic)
    s.calculus_valid AS compute_stationarity,

    -- =========================================================================
    -- LAYER 2: BEHAVIORAL GEOMETRY (HOW)
    -- =========================================================================
    -- Correlation: pairwise relationships (all signals)
    TRUE AS compute_correlation,

    -- DTW distance: shape similarity (analog/periodic)
    s.calculus_valid AS compute_dtw,

    -- Curvature: local geometry (analog/periodic)
    s.calculus_valid AS compute_curvature,

    -- =========================================================================
    -- LAYER 3: DYNAMICAL SYSTEMS (WHEN/HOW)
    -- =========================================================================
    -- Regime detection: behavioral changes (all signals)
    TRUE AS compute_regime,

    -- Lyapunov: chaos detection (analog/periodic, high curvature variance)
    s.calculus_valid AS compute_lyapunov,

    -- Phase space: attractor reconstruction (analog/periodic)
    s.calculus_valid AS compute_phase_space,

    -- =========================================================================
    -- LAYER 4: CAUSAL MECHANICS (WHY)
    -- =========================================================================
    -- Granger causality: predictive relationships (all signals)
    TRUE AS compute_granger,

    -- Transfer entropy: information flow (analog/periodic)
    s.calculus_valid AS compute_transfer_entropy,

    -- Lead-lag: temporal ordering (all signals)
    TRUE AS compute_lead_lag,

    -- =========================================================================
    -- GLOBAL ANALYSIS (system-level)
    -- =========================================================================
    -- These are computed at system level, not per-signal
    TRUE AS include_in_umap,
    TRUE AS include_in_pca,

    -- =========================================================================
    -- PRIORITY
    -- =========================================================================
    CASE s.signal_class
        WHEN 'analog' THEN 1    -- Highest priority
        WHEN 'periodic' THEN 2
        WHEN 'digital' THEN 3
        WHEN 'event' THEN 4     -- Lowest priority
    END AS priority

FROM v_signal_class_unit s;

-- Summary of work orders
CREATE OR REPLACE VIEW v_work_order_summary AS
SELECT
    signal_class,
    COUNT(*) AS n_signals,
    SUM(CASE WHEN compute_hurst THEN 1 ELSE 0 END) AS needs_hurst,
    SUM(CASE WHEN compute_entropy THEN 1 ELSE 0 END) AS needs_entropy,
    SUM(CASE WHEN compute_correlation THEN 1 ELSE 0 END) AS needs_correlation,
    SUM(CASE WHEN compute_dtw THEN 1 ELSE 0 END) AS needs_dtw,
    SUM(CASE WHEN compute_regime THEN 1 ELSE 0 END) AS needs_regime,
    SUM(CASE WHEN compute_lyapunov THEN 1 ELSE 0 END) AS needs_lyapunov,
    SUM(CASE WHEN compute_granger THEN 1 ELSE 0 END) AS needs_granger
FROM v_prism_work_orders
GROUP BY signal_class
ORDER BY MIN(priority);

-- Export work orders as JSON for PRISM
CREATE OR REPLACE VIEW v_work_orders_json AS
SELECT
    signal_id,
    TO_JSON({
        'signal_class': signal_class,
        'compute': {
            'hurst': compute_hurst,
            'entropy': compute_entropy,
            'stationarity': compute_stationarity,
            'correlation': compute_correlation,
            'dtw': compute_dtw,
            'curvature': compute_curvature,
            'regime': compute_regime,
            'lyapunov': compute_lyapunov,
            'phase_space': compute_phase_space,
            'granger': compute_granger,
            'transfer_entropy': compute_transfer_entropy,
            'lead_lag': compute_lead_lag
        },
        'global': {
            'include_umap': include_in_umap,
            'include_pca': include_in_pca
        },
        'priority': priority
    }) AS work_order
FROM v_prism_work_orders;

-- Show summary
SELECT * FROM v_work_order_summary;
