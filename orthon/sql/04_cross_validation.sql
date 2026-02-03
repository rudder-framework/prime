-- ============================================================
-- Cross-File Validation
-- ============================================================

-- Signals in observations but not in typology
SELECT DISTINCT signal_id, 'in observations, not in typology' as issue
FROM observations
WHERE signal_id NOT IN (SELECT signal_id FROM typology);

-- Signals in typology but not in signal_vector
SELECT signal_id, 'in typology, not in signal_vector' as issue
FROM typology
WHERE signal_id NOT IN (SELECT DISTINCT signal_id FROM signal_vector);

-- Window alignment check
SELECT
    o.signal_id,
    COUNT(DISTINCT o.I) as obs_indices,
    COUNT(DISTINCT sv.I) as sv_windows,
    MIN(sv.I) as first_window,
    MAX(sv.I) as last_window
FROM observations o
LEFT JOIN signal_vector sv USING (signal_id)
GROUP BY o.signal_id;

-- Typology vs Signal Vector metric comparison
SELECT
    t.signal_id,
    t.hurst as typo_hurst,
    ROUND(AVG(sv.hurst), 4) as sv_hurst_mean,
    ROUND(ABS(t.hurst - AVG(sv.hurst)), 4) as hurst_diff,
    t.perm_entropy as typo_perm_ent,
    ROUND(AVG(sv.permutation_entropy), 4) as sv_perm_ent_mean,
    ROUND(ABS(t.perm_entropy - AVG(sv.permutation_entropy)), 4) as perm_ent_diff
FROM typology t
JOIN signal_vector sv USING (signal_id)
GROUP BY t.signal_id, t.hurst, t.perm_entropy
ORDER BY hurst_diff DESC;
