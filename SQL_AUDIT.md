# SQL Audit Report

**Run directory**: `/Users/jasonrudder/domains/cmapss/FD_001/train/output_time`
**Generated**: 2026-02-23 08:22

**Summary**: 64/64 passed, 0 failed

## layers/

| File | Status | Rows | Error |
|------|--------|-----:|-------|
| 00_config.sql | PASS | 16 |  |
| 00_index_detection.sql | PASS (views only) | 0 |  |
| 00_load.sql | PASS (views only) | 0 |  |
| 00_observations.sql | PASS (views only) | 0 |  |
| 01_calculus.sql | PASS (views only) | 0 |  |
| 01_typology.sql | PASS (views only) | 0 |  |
| 02_geometry.sql | PASS (views only) | 0 |  |
| 02_statistics.sql | PASS (views only) | 0 |  |
| 03_dynamics.sql | PASS (views only) | 0 |  |
| 03_signal_class.sql | PASS (views only) | 0 |  |
| 04_causality.sql | PASS (views only) | 0 |  |
| 05_manifold_derived.sql | PASS (views only) | 0 |  |
| 08_entropy.sql | PASS (views only) | 0 |  |
| 12_brittleness.sql | PASS | 43 |  |
| 12_ranked_derived.sql | PASS (views only) | 0 |  |
| 13_canary_sequence.sql | PASS | 618 |  |
| 14_curvature_ranking.sql | PASS (views only) | 0 |  |
| 15_geometry_ranked.sql | PASS (views only) | 0 |  |
| 16_coupling_ranked.sql | PASS (views only) | 0 |  |
| 17_dimension_trajectory.sql | PASS | 241 |  |
| 18_ci_breach.sql | PASS | 307 |  |
| atlas_analytics.sql | PASS (views only) | 0 |  |
| atlas_breaks.sql | PASS (views only) | 0 |  |
| atlas_ftle.sql | PASS (views only) | 0 |  |
| atlas_ridge_proximity.sql | PASS (views only) | 0 |  |
| atlas_topology.sql | PASS (views only) | 0 |  |
| atlas_velocity_field.sql | PASS (views only) | 0 |  |
| break_classification.sql | PASS (views only) | 0 |  |
| classification.sql | PASS (views only) | 0 |  |
| constants_units.sql | PASS (views only) | 0 |  |
| typology_v2.sql | PASS | 2,400 |  |

## reports/

| File | Status | Rows | Error |
|------|--------|-----:|-------|
| 01_baseline_geometry.sql | PASS (views only) | 0 |  |
| 02_stable_baseline.sql | PASS (views only) | 0 |  |
| 03_drift_detection.sql | PASS | 18,463 |  |
| 04_signal_ranking.sql | PASS | 5,233 |  |
| 05_periodicity.sql | PASS | 9,600 |  |
| 06_regime_detection.sql | PASS | 8,582 |  |
| 07_correlation_changes.sql | PASS | 23,267 |  |
| 08_lead_lag.sql | PASS | 15,708 |  |
| 09_causality_influence.sql | PASS | 33,582 |  |
| 10_system_departure.sql | PASS | 2,518 |  |
| 11_validation_thresholds.sql | PASS | 2,620 |  |
| 12_ff_stable_periods.sql | PASS (views only) | 0 |  |
| 23_baseline_deviation.sql | PASS | 22 |  |
| 24_incident_summary.sql | PASS | 10 |  |
| 60_ground_truth.sql | PASS (views only) | 0 |  |
| 61_lead_time_analysis.sql | PASS (views only) | 0 |  |
| 62_fault_signatures.sql | PASS (views only) | 0 |  |
| 63_threshold_optimization.sql | PASS (views only) | 0 |  |

## stages/

| File | Status | Rows | Error |
|------|--------|-----:|-------|
| 01_typology.sql | PASS | 36 |  |
| 02_signal_vector.sql | PASS | 307 |  |
| 03_state_vector.sql | PASS | 1,365 |  |
| 04_geometry.sql | PASS | 3,911 |  |
| 05_dynamics.sql | PASS (views only) | 0 |  |

## _legacy/

| File | Status | Rows | Error |
|------|--------|-----:|-------|
| 00_configuration_audit.sql | PASS (views only) | 0 |  |
| 00_run_all.sql | PASS (views only) | 0 |  |
| 06_physics.sql | PASS | 690 |  |
| 25_sensitivity_analysis.sql | PASS (views only) | 0 |  |
| 30_dynamics_stability.sql | PASS (views only) | 0 |  |
| 31_regime_transitions.sql | PASS (views only) | 0 |  |
| 32_basin_stability.sql | PASS (views only) | 0 |  |
| 33_birth_certificate.sql | PASS (views only) | 0 |  |
| 40_topology_departure.sql | PASS (views only) | 0 |  |
| 50_information_departure.sql | PASS (views only) | 0 |  |
