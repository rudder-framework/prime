-- ============================================================================
-- 00_config.sql
-- ============================================================================
-- Configuration summary report
-- Depends on: config_engine_requirements, config_thresholds (layer 00)
-- ============================================================================

.print ''
.print '=============================================='
.print 'Prime CONFIGURATION'
.print '=============================================='
.print ''
.print 'Engine Minimum Data Requirements:'
.print '──────────────────────────────────────────────'

SELECT
    printf('  %-20s %6d pts   %s', engine, min_observations, rationale) AS requirement
FROM config_engine_requirements
WHERE min_observations IS NOT NULL
ORDER BY min_observations DESC;

.print ''
.print 'Interpretation Thresholds:'
.print '──────────────────────────────────────────────'

SELECT '  Lyapunov chaotic:     λ > ' || lyapunov_chaotic_threshold FROM config_thresholds
UNION ALL SELECT '  Lyapunov stable:      λ < ' || lyapunov_stable_threshold FROM config_thresholds
UNION ALL SELECT '  Coherence coupled:    > ' || coherence_strongly_coupled FROM config_thresholds
UNION ALL SELECT '  State significant:    > ' || state_distance_significant || 'σ' FROM config_thresholds
UNION ALL SELECT '  Hurst persistent:     H > ' || hurst_persistent_threshold FROM config_thresholds;

.print ''
.print 'Adjust values in config_thresholds table as needed.'
.print '=============================================='
.print ''
