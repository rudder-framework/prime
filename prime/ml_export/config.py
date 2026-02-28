from dataclasses import dataclass


@dataclass
class FileSpec:
    ml_name: str           # output filename in ml/ directory
    source_path: str       # path relative to output_time/
    action: str            # "passthrough" or "derive"
    grain: str             # "signal-cohort-window" or "cohort-window"
    description: str


# All files the ml/ directory will contain
ALL_SPECS = [
    # === PASS-THROUGH (already per-window, no cross-window dependency) ===
    FileSpec("ml_typology", "typology.parquet", "passthrough", "signal-cohort-window",
             "Signal classifications per window"),
    FileSpec("ml_signal_primitives", "signal_primitives.parquet", "passthrough", "signal-cohort-window",
             "Hurst, entropy, spectral per window"),
    FileSpec("ml_signal_statistics", "signal_statistics.parquet", "passthrough", "signal-cohort-window",
             "Mean, std, skew, kurtosis per window"),
    FileSpec("ml_eigendecomp", "cohort/cohort_geometry.parquet", "passthrough", "cohort-window",
             "Eigenvalues, effective_dim, condition_number per window"),
    FileSpec("ml_centroid", "cohort/cohort_vector.parquet", "passthrough", "cohort-window",
             "Centroid position and dispersion per window"),
    FileSpec("ml_coupling", "coupling_progression.parquet", "passthrough", "cohort-window",
             "Per-window coupling metrics"),
    FileSpec("ml_information_flow", "cohort/cohort_information_flow.parquet", "passthrough", "cohort-window",
             "Transfer entropy per window"),
    FileSpec("ml_persistent_homology", "cohort/persistent_homology.parquet", "passthrough", "cohort-window",
             "Topological features per window"),
    FileSpec("ml_trajectory_match", "system/trajectory_match.parquet", "passthrough", "cohort-window",
             "Template match scores per window"),
    FileSpec("ml_pairwise", "pairwise_windowed.parquet", "passthrough", "cohort-window",
             "Per-window pairwise signal metrics"),

    # === DERIVED (need backward-only derivatives computed by ml_export) ===
    FileSpec("ml_eigendecomp_derivatives", "cohort/cohort_geometry.parquet", "derive", "cohort-window",
             "Backward D1/D2 of eigenvalues, effective_dim, condition_number"),
    FileSpec("ml_centroid_derivatives", "cohort/cohort_vector.parquet", "derive", "cohort-window",
             "Backward D1/D2 of centroid drift and dispersion"),
    FileSpec("ml_signal_derivatives", "signal_primitives.parquet", "derive", "signal-cohort-window",
             "Backward D1/D2 of Hurst, entropy, spectral metrics"),
]


# Feature configs for model training — controls which ml/ parquets the assembler includes.
# B = regime-normalized cohort RT (current best, Config 4 z-score)
# E = B + canary cohort RT (TBD — canary not yet implemented)
# F = B + modality RT (thermal, pressure, speed, ratio, flow)
# G = B + canary + modality (full stack)
FEATURE_CONFIGS = {
    "B": ["ml_normalized_rt", "ml_normalized_csv"],
    "F": ["ml_normalized_rt", "ml_normalized_csv", "ml_modality_features"],
    "G": ["ml_normalized_rt", "ml_normalized_csv", "ml_modality_features"],  # canary TBD
}
