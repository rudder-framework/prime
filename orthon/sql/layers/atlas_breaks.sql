-- ============================================================
-- Atlas Break Cascade Classification
-- Classifies fault propagation from break_sequence.parquet
--
-- Engines detects breaks. ORTHON classifies cascades.
-- ============================================================

-- ------------------------------------------------------------
-- BREAK CASCADE
-- Classifies each signal's role in a structural break cascade
-- based on propagation order, timing, and magnitude.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_break_cascade AS
SELECT
    cohort,
    signal_id,
    first_break_I,
    detection_latency,
    propagation_rank,
    cascade_delay,
    magnitude,
    direction,
    snr,
    reference_index,

    -- Cascade role from propagation rank
    CASE
        WHEN propagation_rank = 1 THEN 'initiator'
        WHEN propagation_rank <= 3 THEN 'early_follower'
        WHEN cascade_delay IS NOT NULL AND cascade_delay < 50 THEN 'follower'
        ELSE 'late'
    END AS cascade_role,

    -- Propagation speed from cascade delay
    CASE
        WHEN cascade_delay IS NULL OR cascade_delay = 0 THEN 'origin'
        WHEN cascade_delay < 10 THEN 'fast'
        WHEN cascade_delay < 50 THEN 'medium'
        ELSE 'slow'
    END AS propagation_speed,

    -- Break significance from magnitude and SNR
    CASE
        WHEN snr IS NULL THEN 'unknown'
        WHEN snr > 10 AND ABS(magnitude) > 1.0 THEN 'strong'
        WHEN snr > 3 AND ABS(magnitude) > 0.5 THEN 'moderate'
        ELSE 'weak'
    END AS break_significance,

    -- Break direction
    CASE
        WHEN direction > 0 THEN 'upward'
        WHEN direction < 0 THEN 'downward'
        ELSE 'neutral'
    END AS break_direction

FROM break_sequence;
