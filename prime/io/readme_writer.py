"""Generate README.md for each Manifold output directory."""

from pathlib import Path

try:
    import polars as pl
except ImportError:
    pl = None


DIRECTORY_DESCRIPTIONS = {
    "1_signal_features": {
        "title": "Signal Features",
        "description": "Per-signal statistical, spectral, and stability measures computed from individual time series.",
        "files": {
            "signal_vector.parquet": "Feature vector per (signal_id, cohort, window): kurtosis, spectral entropy, Hurst, ACF decay, permutation entropy, and more.",
            "signal_geometry.parquet": "Per-signal distance and coherence relative to the system state geometry.",
            "signal_stability.parquet": "Rolling Hilbert amplitude, instantaneous frequency, and wavelet energy per signal.",
        }
    },
    "2_system_state": {
        "title": "System State",
        "description": "System-level geometry computed from the full signal ensemble. Eigenvalues, eigenvectors, and their dynamics over time.",
        "files": {
            "state_vector.parquet": "System centroid (mean position in feature space) per cohort per window.",
            "state_geometry.parquet": "Eigenvalues, effective dimension, condition number, eigenvector loadings, bootstrap confidence intervals.",
            "state_geometry_loadings.parquet": "Full eigenvector loading matrix showing which signals contribute to each principal component.",
            "state_geometry_feature_loadings.parquet": "Feature-level loadings decomposing eigenvector contributions by feature type.",
            "geometry_dynamics.parquet": "Velocity, acceleration, jerk of the eigenvalue trajectory. Collapse onset detection.",
            "sensor_eigendecomp.parquet": "Rolling eigendecomposition at the observation level for real-time geometry tracking.",
        }
    },
    "3_departure_scoring": {
        "title": "Departure Scoring",
        "description": "Baseline comparison and break detection for departure assessment.",
        "files": {
            "breaks.parquet": "Regime changes (steps and impulses) detected per signal via CUSUM/Pettitt.",
            "observation_geometry.parquet": "Per-cycle centroid distance and PC1 projection relative to fleet baseline. Real-time departure indicator.",
            "cohort_baseline.parquet": "Fleet baseline eigenstructure computed from early-life pooled data.",
        }
    },
    "4_signal_relationships": {
        "title": "Signal Relationships",
        "description": "Pairwise coupling, causality, and information flow between signals.",
        "files": {
            "signal_pairwise.parquet": "Pairwise correlation, DTW distance, cosine similarity between all signal pairs per window.",
            "information_flow.parquet": "Granger causality F-statistics and p-values for every directed signal pair.",
            "segment_comparison.parquet": "Early-life vs late-life geometry delta per cohort. How much did the system change?",
            "info_flow_delta.parquet": "Change in Granger causality between early and late segments. Which causal links strengthened or broke?",
        }
    },
    "5_evolution": {
        "title": "Evolution",
        "description": "Dynamical systems analysis: Lyapunov exponents, FTLE fields, velocity, topology.",
        "files": {
            "ftle.parquet": "Forward Finite-Time Lyapunov Exponents per signal.",
            "lyapunov.parquet": "Maximal Lyapunov exponent per signal (Rosenstein method).",
            "ftle_field.parquet": "Spatiotemporal FTLE field across the state space.",
            "ftle_backward.parquet": "Backward FTLE (attracting structures).",
            "ftle_rolling.parquet": "Rolling FTLE stability evolution over time.",
            "ridge_proximity.parquet": "Urgency metric: velocity toward nearest FTLE ridge.",
            "velocity_field.parquet": "State-space velocity (speed, curvature, direction) per timestep.",
            "velocity_field_components.parquet": "Component-wise velocity decomposition.",
            "cohort_thermodynamics.parquet": "System entropy, energy, temperature per cohort.",
            "persistent_homology.parquet": "Betti numbers and persistence diagrams from topological data analysis.",
        }
    },
    "6_fleet": {
        "title": "Fleet",
        "description": "Cross-cohort analysis. Requires multiple cohorts and cohort_vector from Prime.",
        "files": {
            "system_geometry.parquet": "Eigendecomposition of the fleet (cohorts as rows, centroids as columns).",
            "cohort_pairwise.parquet": "DTW, correlation, similarity between every pair of cohorts.",
            "cohort_information_flow.parquet": "Granger causality between cohort trajectories.",
            "cohort_ftle.parquet": "FTLE computed on the fleet-level trajectory.",
            "cohort_velocity_field.parquet": "Speed and curvature of the fleet trajectory through system space.",
        }
    },
}


def generate_manifold_readmes(output_dir: Path) -> None:
    """Generate README.md for each Manifold output subdirectory."""

    if pl is None:
        print("  → Skipping Manifold READMEs (polars not installed)")
        return

    output_dir = Path(output_dir)
    n_generated = 0

    for dir_name, info in DIRECTORY_DESCRIPTIONS.items():
        dir_path = output_dir / dir_name
        if not dir_path.exists():
            continue

        lines = [
            f"# {info['title']}",
            "",
            info['description'],
            "",
            "---",
            "",
            "## Files",
            "",
        ]

        for filename, description in info['files'].items():
            filepath = dir_path / filename
            if filepath.exists():
                try:
                    df = pl.read_parquet(filepath)
                    size_kb = filepath.stat().st_size / 1024
                    lines.append(f"### {filename}")
                    lines.append("")
                    lines.append(description)
                    lines.append("")
                    lines.append(f"- Rows: {df.height:,}")
                    lines.append(f"- Columns: {df.width}")
                    lines.append(f"- Size: {size_kb:.1f} KB")
                    lines.append("")
                    lines.append("| Column | Type |")
                    lines.append("|--------|------|")
                    for col in df.columns:
                        lines.append(f"| {col} | {df[col].dtype} |")
                    lines.append("")
                except Exception as e:
                    lines.append(f"### {filename}")
                    lines.append(f"*Error reading: {e}*")
                    lines.append("")
            else:
                lines.append(f"### {filename}")
                lines.append("*Not generated in this run.*")
                lines.append("")

        readme_path = dir_path / "README.md"
        readme_path.write_text("\n".join(lines))
        n_generated += 1

    print(f"  → READMEs generated for {n_generated} directories")
