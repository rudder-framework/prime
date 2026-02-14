"""
Physics Interpreter
====================

Detecting Symplectic Structure Loss in High-Dimensional Dynamical Systems
via Information-Geometric Coherence Functionals.

The coherence functional measures deviation from the symplectic leaf.
When it diverges, Liouville's theorem fails locally - energy is no longer
conserved, phase space volume contracts, and the system is dissipating
into a lower-dimensional attractor.

PRISM computes. Rudder interprets.

Philosophy:
-----------
Start with thermodynamics (the constraint), work backwards to mechanism:

    L4: Thermodynamics  → Is energy conserved or dissipating?
            ↓
    L3: Mechanics       → Where is it flowing? What forces?
            ↓
    L2: Coherence       → Through what couplings? (symplectic structure)
            ↓
    L1: State           → Resulting phase space position?

The Rudder Signal:
------------------
    dissipating + decoupling + diverging = symplectic structure loss

This is the generalized degradation signal. It detects when the system's
geometric structure is breaking down, regardless of domain.
"""

import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
from collections import Counter

from prime.shared.physics_constants import (
    PhysicsConstants,
    ENERGY_FORMULAS,
    UNIT_TO_CATEGORY,
    can_compute_real_energy,
    get_unit_category,
)


@dataclass
class SystemDiagnosis:
    """Diagnosis of system state from physics analysis."""
    severity: str  # 'normal', 'watch', 'warning', 'critical'
    rudder_signal: bool  # The key degradation indicator
    issues: List[str]
    summary: str


class PhysicsInterpreter:
    """
    Interprets physics.parquet from PRISM.

    Detects symplectic structure loss via information-geometric coherence.

    When the coherence functional diverges:
    - Energy conservation fails (Liouville's theorem violation)
    - Phase space volume contracts (dissipation)
    - System collapses to lower-dimensional attractor

    This is what "degradation" means mathematically.
    """

    def __init__(
        self,
        physics_path: Path = None,
        physics_df: pl.DataFrame = None,
        obs_enriched_path: Optional[Path] = None,
        obs_enriched_df: pl.DataFrame = None,
        signal_units: Optional[Dict[str, str]] = None,
        constants: Optional[PhysicsConstants] = None
    ):
        """
        Initialize interpreter.

        Args:
            physics_path: Path to physics.parquet from PRISM
            physics_df: Or provide DataFrame directly
            obs_enriched_path: Path to observations_enriched.parquet (for per-signal analysis)
            obs_enriched_df: Or provide DataFrame directly
            signal_units: Map signal_id -> unit string (e.g., {'Motor_current': 'A'})
            constants: Physics constants for real energy calculations
        """
        # Load physics data
        if physics_df is not None:
            self.physics = physics_df
        elif physics_path:
            self.physics = pl.read_parquet(physics_path)
        else:
            raise ValueError("Must provide physics_path or physics_df")

        # Load observations enriched (optional, for per-signal analysis)
        if obs_enriched_df is not None:
            self.obs_enriched = obs_enriched_df
        elif obs_enriched_path and Path(obs_enriched_path).exists():
            self.obs_enriched = pl.read_parquet(obs_enriched_path)
        else:
            self.obs_enriched = None

        # Physics configuration
        self.signal_units = signal_units or {}
        self.constants = constants or PhysicsConstants()

        # Determine if we can use real physics
        self.use_real_physics = self._check_real_physics_available()

    def _check_real_physics_available(self) -> bool:
        """Check if we have units + constants for real physics."""
        if not self.signal_units:
            return False

        # Check if at least one signal has computable energy
        for signal, unit in self.signal_units.items():
            unit_category = get_unit_category(unit)
            if unit_category and can_compute_real_energy(unit_category, self.constants):
                return True
        return False

    def check_configuration_homogeneity(self) -> Tuple[bool, Dict[str, int], List[str]]:
        """
        Check if all entities have the same signal count.

        This MUST be called before cross-entity comparisons. Different signal
        counts confound fingerprint metrics like effective_dim and entropy.

        Returns:
            Tuple of:
            - is_homogeneous: bool - True if all entities have same n_signals
            - config_map: Dict[entity_id, n_signals]
            - warnings: List of warning messages

        Example:
            is_homo, configs, warnings = interpreter.check_configuration_homogeneity()
            if not is_homo:
                print("⚠️ HETEROGENEOUS CONFIGURATION DETECTED")
                for w in warnings:
                    print(f"   {w}")
        """
        # Get n_signals per entity
        config_df = (
            self.physics
            .group_by("entity_id")
            .agg([
                pl.col("n_signals").mode().first().alias("typical_signals"),
                pl.col("n_signals").min().alias("min_signals"),
                pl.col("n_signals").max().alias("max_signals"),
            ])
        )

        config_map = dict(zip(
            config_df["entity_id"].to_list(),
            config_df["typical_signals"].to_list()
        ))

        unique_configs = set(config_map.values())
        is_homogeneous = len(unique_configs) == 1

        warnings = []

        if not is_homogeneous:
            warnings.append(
                f"HETEROGENEOUS CONFIGURATION: Signal counts vary ({sorted(unique_configs)})"
            )
            warnings.append(
                "effective_dim and entropy comparisons are confounded"
            )
            warnings.append(
                "Use normalized metrics: norm_dim = eff_dim / n_signals"
            )

            # Group entities by config
            config_groups = {}
            for entity, n_sig in config_map.items():
                if n_sig not in config_groups:
                    config_groups[n_sig] = []
                config_groups[n_sig].append(entity)

            for n_sig, entities in sorted(config_groups.items()):
                if len(entities) == 1:
                    warnings.append(
                        f"SINGLETON: Entity {entities[0]} has unique config ({n_sig} signals)"
                    )

        # Check for within-entity variation
        variable_entities = config_df.filter(
            pl.col("min_signals") != pl.col("max_signals")
        )["entity_id"].to_list()

        if variable_entities:
            warnings.append(
                f"VARIABLE WITHIN ENTITY: {variable_entities} have changing signal counts"
            )

        return is_homogeneous, config_map, warnings

    def get_normalized_metrics(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Get metrics normalized by signal count for fair cross-entity comparison.

        Returns normalized versions of metrics that depend on n_signals:
        - norm_effective_dim = effective_dim / n_signals
        - norm_entropy = eigenvalue_entropy / log2(n_signals)

        Metrics that are already normalized (unaffected by n_signals):
        - coherence (λ₁/Σλ - already a ratio)
        - state_distance (normalized by covariance)
        - dissipation_rate
        """
        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)

        if entity_data.is_empty():
            return None

        n_signals = entity_data["n_signals"].to_numpy()[0]

        # Get mean values
        coherence = float(entity_data["coherence"].mean())
        effective_dim = float(entity_data["effective_dim"].mean())
        entropy = float(entity_data["eigenvalue_entropy"].mean())

        # Normalize
        norm_effective_dim = effective_dim / n_signals
        max_entropy = np.log2(n_signals) if n_signals > 1 else 1.0
        norm_entropy = entropy / max_entropy

        return {
            "entity_id": entity_id,
            "n_signals": int(n_signals),
            "coherence": round(coherence, 4),
            "effective_dim": round(effective_dim, 3),
            "norm_effective_dim": round(norm_effective_dim, 4),
            "eigenvalue_entropy": round(entropy, 4),
            "norm_entropy": round(norm_entropy, 4),
            "max_entropy": round(max_entropy, 4),
        }

    def get_entities(self) -> List[str]:
        """Get list of all entities in physics data."""
        return self.physics.select("entity_id").unique().to_series().to_list()

    # =========================================================================
    # L4: THERMODYNAMICS - Is energy conserved?
    # =========================================================================

    def analyze_energy_budget(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        First question: Is energy conserved?

        This is the starting point. If energy is conserved, the system
        maintains its symplectic structure. If not, we investigate why.

        Liouville's theorem: Hamiltonian flow preserves phase space volume.
        Dissipation violates this - the system is no longer Hamiltonian.
        """
        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)

        if entity_data.is_empty():
            return None

        # Get energy time series
        if self.use_real_physics and self.obs_enriched is not None:
            energy = self._compute_real_energy(entity_id)
        else:
            energy = entity_data["energy_proxy"].to_numpy()

        I = entity_data["I"].to_numpy()
        n = len(I)

        if n < 2:
            return None

        # Analyze energy budget
        energy_mean = np.nanmean(energy)
        energy_std = np.nanstd(energy)

        # Fit linear trend
        valid_mask = ~np.isnan(energy)
        if np.sum(valid_mask) > 1:
            energy_trend = np.polyfit(I[valid_mask], energy[valid_mask], 1)[0]
        else:
            energy_trend = 0

        # Conservation check: is variance low and trend near zero?
        coefficient_of_variation = energy_std / (abs(energy_mean) + 1e-10)

        is_conserved = (
            coefficient_of_variation < 0.1 and  # Low variance
            abs(energy_trend) < 0.01 * abs(energy_mean)  # Low trend
        )

        # Classify trend
        if abs(energy_trend) < 0.001 * abs(energy_mean + 1e-10):
            trend = 'stable'
        elif energy_trend > 0:
            trend = 'accumulating'
        else:
            trend = 'dissipating'

        # Dissipation rate from PRISM
        if "dissipation_rate" in entity_data.columns:
            dissipation = entity_data["dissipation_rate"].to_numpy()
            mean_dissipation = float(np.nanmean(dissipation))
        else:
            mean_dissipation = max(0, -energy_trend)

        # Entropy production
        if "entropy_production" in entity_data.columns:
            entropy_prod = entity_data["entropy_production"].to_numpy()
            mean_entropy_prod = float(np.nanmean(entropy_prod))
        else:
            mean_entropy_prod = 0

        if mean_entropy_prod < 0.001:
            entropy_trend = 'stable'
        elif mean_entropy_prod > 0:
            entropy_trend = 'increasing'
        else:
            entropy_trend = 'decreasing'

        return {
            'energy_conserved': is_conserved,
            'energy_trend': trend,
            'energy_mean': float(energy_mean),
            'energy_std': float(energy_std),
            'energy_trend_slope': float(energy_trend),
            'dissipation_rate_mean': mean_dissipation,
            'entropy_trend': entropy_trend,
            'entropy_production_mean': mean_entropy_prod,
            'using_real_physics': self.use_real_physics,
        }

    def _compute_real_energy(self, entity_id: str) -> np.ndarray:
        """Compute real energy using units and constants."""
        if self.obs_enriched is None:
            return self.physics.filter(
                pl.col("entity_id") == entity_id
            )["energy_proxy"].to_numpy()

        entity_obs = self.obs_enriched.filter(pl.col("entity_id") == entity_id)
        I_values = entity_obs.select("I").unique().sort("I")["I"].to_numpy()

        total_energy = []
        for I_val in I_values:
            I_data = entity_obs.filter(pl.col("I") == I_val)

            E_total = 0
            for row in I_data.iter_rows(named=True):
                signal = row.get("signal_id", "")
                y = row.get("value", 0) or 0

                unit = self.signal_units.get(signal)
                unit_cat = get_unit_category(unit) if unit else None

                if unit_cat and unit_cat in ENERGY_FORMULAS:
                    E_real = ENERGY_FORMULAS[unit_cat](y, self.constants)
                    if E_real is not None:
                        E_total += E_real
                        continue

                # Fallback to proxy
                dy = row.get("dy", 0) or 0
                E_total += y**2 + dy**2

            total_energy.append(E_total)

        return np.array(total_energy)

    # =========================================================================
    # L3: MECHANICS - Where is energy flowing?
    # =========================================================================

    def analyze_energy_flow(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Second question: Where is energy going?

        Only meaningful if energy is NOT conserved.
        Identifies energy flow patterns and asymmetries.
        """
        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)

        if entity_data.is_empty():
            return None

        # Energy flow asymmetry over time
        if "energy_flow_asymmetry" in entity_data.columns:
            asymmetry = entity_data["energy_flow_asymmetry"].to_numpy()
            mean_asymmetry = float(np.nanmean(asymmetry))
        else:
            mean_asymmetry = 0

        # High asymmetry = energy concentrated in few signals (Gini-like)
        if mean_asymmetry > 0.6:
            distribution = 'concentrated'
        elif mean_asymmetry > 0.3:
            distribution = 'uneven'
        else:
            distribution = 'distributed'

        # Energy velocity (rate of change)
        if "energy_velocity" in entity_data.columns:
            energy_velocity = entity_data["energy_velocity"].to_numpy()
            velocity_std = float(np.nanstd(energy_velocity))

            # Identify periods of energy transfer (high velocity)
            transfer_threshold = velocity_std * 2
            transfer_periods = np.where(np.abs(energy_velocity) > transfer_threshold)[0]
            n_transfer_events = len(transfer_periods)
        else:
            velocity_std = 0
            n_transfer_events = 0

        return {
            'energy_distribution': distribution,
            'asymmetry_mean': mean_asymmetry,
            'n_transfer_events': n_transfer_events,
            'energy_velocity_std': velocity_std,
        }

    def identify_energy_sources_sinks(self, entity_id: str) -> Dict[str, Any]:
        """
        Identify which signals are energy sources vs sinks.

        Sources: energy increasing (work being done ON system)
        Sinks: energy decreasing (work being done BY system, or dissipation)
        """
        if self.obs_enriched is None:
            return {'error': 'Need observations_enriched for per-signal analysis'}

        entity_obs = self.obs_enriched.filter(pl.col("entity_id") == entity_id)
        signals = entity_obs.select("signal_id").unique().to_series().to_list()

        if not signals:
            return {'error': 'No signals found'}

        signal_energy_trend = {}

        for signal in signals:
            signal_data = entity_obs.filter(
                pl.col("signal_id") == signal
            ).sort("I")

            if signal_data.is_empty():
                continue

            y = signal_data["value"].to_numpy()
            I = signal_data["I"].to_numpy()

            if "dy" in signal_data.columns:
                dy = signal_data["dy"].fill_null(0).to_numpy()
            else:
                dy = np.gradient(y, I) if len(I) > 1 else np.zeros_like(y)

            # Per-signal energy proxy
            E_signal = y**2 + dy**2

            # Trend
            if len(I) > 1:
                valid_mask = ~np.isnan(E_signal)
                if np.sum(valid_mask) > 1:
                    trend = np.polyfit(I[valid_mask], E_signal[valid_mask], 1)[0]
                else:
                    trend = 0
            else:
                trend = 0

            # Classify
            if trend > 0.001:
                role = 'source'
            elif trend < -0.001:
                role = 'sink'
            else:
                role = 'neutral'

            signal_energy_trend[signal] = {
                'trend': float(trend),
                'mean_energy': float(np.nanmean(E_signal)),
                'role': role
            }

        # Rank by absolute trend
        sources = [s for s, v in signal_energy_trend.items() if v['role'] == 'source']
        sinks = [s for s, v in signal_energy_trend.items() if v['role'] == 'sink']

        sources = sorted(sources, key=lambda s: signal_energy_trend[s]['trend'], reverse=True)
        sinks = sorted(sinks, key=lambda s: signal_energy_trend[s]['trend'])

        return {
            'signal_trends': signal_energy_trend,
            'sources': sources,
            'sinks': sinks,
        }

    # =========================================================================
    # L2: COHERENCE - Symplectic structure integrity (Eigenvalue-Based)
    # =========================================================================

    def analyze_coherence(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Third question: Is the symplectic structure intact?

        Now uses eigenvalue-based coherence metrics:
        - coherence: λ₁/Σλ (spectral coherence) - fraction of variance in dominant mode
        - effective_dim: participation ratio - how many independent modes
        - eigenvalue_entropy: spectral disorder - 0 (ordered) to 1 (disordered)

        Eigenvalue coherence captures STRUCTURE, not just average correlation.
        When the first eigenvalue dominates, signals move as one.
        When eigenvalues spread out, system is fragmenting into independent modes.

        Decoupling = energy transfer pathways breaking
        Fragmentation = system splitting into independent subsystems
        """
        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)

        if entity_data.is_empty():
            return None

        I = entity_data["I"].to_numpy()
        n = len(I)

        # Get coherence data (now eigenvalue-based)
        if "coherence" in entity_data.columns:
            coherence = entity_data["coherence"].to_numpy()
        else:
            return {'error': 'No coherence data available'}

        if "coherence_velocity" in entity_data.columns:
            coherence_velocity = entity_data["coherence_velocity"].to_numpy()
        else:
            coherence_velocity = np.gradient(coherence, I) if n > 1 else np.zeros(n)

        # New eigenvalue-based metrics
        if "effective_dim" in entity_data.columns:
            effective_dim = entity_data["effective_dim"].to_numpy()
        else:
            effective_dim = np.ones(n)  # Fallback: assume unified

        if "eigenvalue_entropy" in entity_data.columns:
            eigenvalue_entropy = entity_data["eigenvalue_entropy"].to_numpy()
        else:
            eigenvalue_entropy = np.zeros(n)  # Fallback: assume ordered

        # Get signal count
        n_signals = 1
        for col in ["n_signals", "n_pairs"]:
            if col in entity_data.columns:
                val = entity_data[col][0]
                if col == "n_pairs":
                    # n_pairs = n*(n-1)/2, solve for n
                    n_signals = int((1 + np.sqrt(1 + 8 * val)) / 2)
                else:
                    n_signals = int(val)
                break

        if "n_pairs" in entity_data.columns:
            n_pairs = int(entity_data["n_pairs"][0])
        else:
            n_pairs = n_signals * (n_signals - 1) // 2

        # Current values
        current_coherence = float(coherence[-1]) if n > 0 else np.nan
        current_effective_dim = float(effective_dim[-1]) if n > 0 else np.nan
        current_entropy = float(eigenvalue_entropy[-1]) if n > 0 else np.nan

        # Baseline (first 10%)
        baseline_n = max(1, n // 10)
        baseline_coherence = float(np.nanmean(coherence[:baseline_n]))
        baseline_effective_dim = float(np.nanmean(effective_dim[:baseline_n]))

        # Coupling state assessment (eigenvalue-based thresholds)
        if current_coherence > 0.7:
            coupling_state = 'strongly_coupled'
        elif current_coherence > 0.4:
            coupling_state = 'weakly_coupled'
        else:
            coupling_state = 'decoupled'

        # Structure state assessment
        if current_effective_dim < 1.5:
            structure_state = 'unified'
        elif current_effective_dim < n_signals / 2:
            structure_state = 'clustered'
        else:
            structure_state = 'fragmented'

        # Trend analysis
        if n > 1:
            valid_mask = ~np.isnan(coherence)
            if np.sum(valid_mask) > 1:
                coherence_trend = np.polyfit(I[valid_mask], coherence[valid_mask], 1)[0]
            else:
                coherence_trend = 0

            valid_mask_dim = ~np.isnan(effective_dim)
            if np.sum(valid_mask_dim) > 1:
                effective_dim_trend = np.polyfit(I[valid_mask_dim], effective_dim[valid_mask_dim], 1)[0]
            else:
                effective_dim_trend = 0
        else:
            coherence_trend = 0
            effective_dim_trend = 0

        # Decoupling detection (more nuanced with eigenvalues)
        is_decoupling = (
            coherence_trend < -0.001 or  # Coherence dropping
            (current_coherence < baseline_coherence * 0.8 and baseline_coherence > 0.3) or  # Below 80% of baseline
            (current_effective_dim > baseline_effective_dim * 1.5 and baseline_effective_dim > 1)  # Dimensions expanding
        )

        # Mode fragmentation detection (new)
        is_fragmenting = (
            effective_dim_trend > 0.01 and  # Dimensions increasing
            current_entropy > 0.5  # Disorder high
        )

        # Backward compatibility
        is_coupled = coupling_state == 'strongly_coupled'

        return {
            # Coupling assessment
            'coupling_state': coupling_state,
            'structure_state': structure_state,
            'is_coupled': is_coupled,
            'is_decoupling': is_decoupling,
            'is_fragmenting': is_fragmenting,

            # Current values
            'current_coherence': current_coherence,
            'current_effective_dim': current_effective_dim,
            'current_eigenvalue_entropy': current_entropy,

            # Baseline comparison
            'baseline_coherence': baseline_coherence,
            'baseline_effective_dim': baseline_effective_dim,
            'coherence_vs_baseline': float(current_coherence / baseline_coherence) if baseline_coherence > 0 else np.nan,

            # Trends
            'coherence_trend': float(coherence_trend),
            'effective_dim_trend': float(effective_dim_trend),
            'coherence_velocity_mean': float(np.nanmean(coherence_velocity)),

            # Counts
            'n_signals': n_signals,
            'n_pairs': n_pairs,
        }

    def interpret_coherence_change(self, entity_id: str) -> str:
        """
        Generate human-readable interpretation of coherence state.

        Uses eigenvalue-based metrics to explain what's happening
        with the system's coupling structure.
        """
        analysis = self.analyze_coherence(entity_id)

        if analysis is None:
            return "No coherence data available."

        if 'error' in analysis:
            return analysis['error']

        parts = []

        # Coupling state
        coupling = analysis['coupling_state']
        coh = analysis['current_coherence']
        if coupling == 'strongly_coupled':
            parts.append(f"Signals are strongly coupled (coherence {coh:.2f})")
        elif coupling == 'weakly_coupled':
            parts.append(f"Signals are weakly coupled (coherence {coh:.2f})")
        else:
            parts.append(f"Signals are decoupled (coherence {coh:.2f})")

        # Structure
        structure = analysis['structure_state']
        eff_dim = analysis['current_effective_dim']
        n_sig = analysis['n_signals']

        if structure == 'unified':
            parts.append(f"System moving as one mode (effective dim {eff_dim:.1f} of {n_sig})")
        elif structure == 'clustered':
            parts.append(f"System has ~{int(round(eff_dim))} independent clusters")
        else:
            parts.append(f"System fragmented into ~{int(round(eff_dim))} independent modes")

        # Warnings
        if analysis['is_decoupling']:
            parts.append("WARNING: Decoupling detected")
        if analysis['is_fragmenting']:
            parts.append("WARNING: Mode fragmentation in progress")

        # Comparison to baseline
        ratio = analysis.get('coherence_vs_baseline', 1.0)
        if ratio and not np.isnan(ratio) and ratio < 0.8:
            parts.append(f"Coherence is {(1-ratio)*100:.0f}% below baseline")

        return ". ".join(parts) + "."

    # =========================================================================
    # L1: STATE - Phase space position
    # =========================================================================

    def analyze_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """
        Fourth question: Where is the system in phase space?

        State is the CONSEQUENCE of energy dynamics.
        state_distance measures deviation from baseline in metric-space.
        state_velocity is the generalized hd_slope - using ALL metrics.
        """
        entity_data = self.physics.filter(pl.col("entity_id") == entity_id)

        if entity_data.is_empty():
            return None

        I = entity_data["I"].to_numpy()
        n = len(I)

        # Get state metrics
        if "state_distance" in entity_data.columns:
            state_distance = entity_data["state_distance"].to_numpy()
        else:
            return {'error': 'No state_distance data available'}

        if "state_velocity" in entity_data.columns:
            state_velocity = entity_data["state_velocity"].to_numpy()
        else:
            state_velocity = np.gradient(state_distance, I) if n > 1 else np.zeros(n)

        if "state_acceleration" in entity_data.columns:
            state_acceleration = entity_data["state_acceleration"].to_numpy()
        else:
            state_acceleration = np.gradient(state_velocity, I) if n > 1 else np.zeros(n)

        # Handle n_metrics_used (might be n_metrics in some versions)
        n_metrics = 0
        for col in ["n_metrics_used", "n_metrics"]:
            if col in entity_data.columns:
                n_metrics = int(entity_data[col][0])
                break

        current_distance = float(state_distance[-1]) if n > 0 else np.nan
        current_velocity = float(state_velocity[-1]) if n > 0 else np.nan
        current_acceleration = float(state_acceleration[-1]) if n > 0 else 0

        # Stability check
        is_stable = (
            abs(current_velocity) < 0.01 and
            current_distance < 2.0  # Within 2σ of baseline
        )

        # Trend classification
        if current_velocity > 0.01:
            trend = 'diverging'
        elif current_velocity < -0.01:
            trend = 'converging'
        else:
            trend = 'stable'

        return {
            'is_stable': is_stable,
            'trend': trend,
            'current_distance': current_distance,
            'current_velocity': current_velocity,
            'current_acceleration': current_acceleration,
            'max_distance': float(np.nanmax(state_distance)),
            'n_metrics_used': n_metrics,
        }

    # =========================================================================
    # FULL SYSTEM ANALYSIS
    # =========================================================================

    def analyze_system(self, entity_id: str) -> Dict[str, Any]:
        """
        Full system analysis, starting with thermodynamics.

        The order matters - we follow the physics hierarchy:
        1. Is energy conserved? (thermodynamics - the constraint)
        2. Where is it going? (mechanics - the flow)
        3. Through what couplings? (coherence - the structure)
        4. Resulting in what state? (position - the consequence)

        The Rudder signal is detected when all three symptoms appear:
        - Energy dissipating
        - Signals decoupling
        - State diverging

        This indicates symplectic structure loss.
        """
        # L4: Start with energy (the constraint)
        thermo = self.analyze_energy_budget(entity_id)
        if thermo is None:
            return {'error': f'No data for entity {entity_id}'}

        # L3: If not conserved, where is it going?
        if not thermo['energy_conserved']:
            mechanics = self.analyze_energy_flow(entity_id)
            sources_sinks = self.identify_energy_sources_sinks(entity_id)
        else:
            mechanics = {'note': 'Energy conserved, flow analysis skipped'}
            sources_sinks = {'note': 'Energy conserved, source/sink analysis skipped'}

        # L2: Coherence (coupling pathways / symplectic structure)
        coherence = self.analyze_coherence(entity_id)

        # L1: State (consequence)
        state = self.analyze_state(entity_id)

        # Synthesize diagnosis
        diagnosis = self._diagnose(thermo, mechanics, coherence, state)

        return {
            'entity_id': entity_id,
            'using_real_physics': self.use_real_physics,

            # Top-down layers
            'L4_thermodynamics': thermo,
            'L3_mechanics': mechanics,
            'L3_sources_sinks': sources_sinks,
            'L2_coherence': coherence,
            'L1_state': state,

            # Synthesis
            'diagnosis': diagnosis,
        }

    def _diagnose(
        self,
        thermo: Dict,
        mechanics: Dict,
        coherence: Optional[Dict],
        state: Optional[Dict]
    ) -> Dict[str, Any]:
        """
        Synthesize findings into diagnosis.

        The Rudder signal: dissipating + decoupling + diverging
        This is the mathematical signature of degradation.

        Updated for eigenvalue-based coherence metrics.
        """
        issues = []
        severity = 'normal'

        # Check thermodynamics (L4)
        if thermo['energy_trend'] == 'dissipating':
            issues.append('Energy dissipating from system')
            severity = 'warning'
        elif thermo['energy_trend'] == 'accumulating':
            issues.append('Energy accumulating in system')
            severity = 'watch'

        if thermo['entropy_trend'] == 'increasing':
            issues.append('Entropy increasing (disorder)')
            if severity == 'normal':
                severity = 'watch'

        # Check coherence (L2) - Updated for eigenvalue-based metrics
        if coherence and not coherence.get('error'):
            # Decoupling detection
            if coherence.get('is_decoupling'):
                coh_val = coherence.get('current_coherence', 0)
                issues.append(f"Signals decoupling (coherence {coh_val:.2f})")
                if severity == 'normal':
                    severity = 'watch'
                elif severity == 'watch':
                    severity = 'warning'

            # Mode fragmentation detection (new)
            if coherence.get('is_fragmenting'):
                eff_dim = coherence.get('current_effective_dim', 0)
                issues.append(f"Mode fragmentation ({eff_dim:.1f} independent modes)")
                if severity == 'normal':
                    severity = 'watch'
                elif severity == 'watch':
                    severity = 'warning'

            # Severe: coupling state is fully decoupled
            if coherence.get('coupling_state') == 'decoupled':
                if thermo['energy_trend'] == 'dissipating':
                    severity = 'critical'
                elif severity in ('normal', 'watch'):
                    severity = 'warning'

        # Check state (L1)
        if state and state.get('trend') == 'diverging':
            issues.append('State diverging from baseline')
            if severity in ('normal', 'watch'):
                severity = 'warning'

        # The Rudder Signal: all three symptoms together
        # Now uses eigenvalue-based coherence for more sensitive detection
        coherence_failing = False
        if coherence and not coherence.get('error'):
            coherence_failing = (
                coherence.get('is_decoupling', False) or
                coherence.get('is_fragmenting', False) or
                coherence.get('coupling_state') == 'decoupled'
            )

        rudder_signal = (
            thermo['energy_trend'] == 'dissipating' and
            coherence_failing and
            (state.get('trend') == 'diverging' if state else False)
        )

        if rudder_signal:
            severity = 'critical'
            issues.insert(0, 'RUDDER SIGNAL: Symplectic structure loss detected')

        # Severity scoring for gradation
        severity_score = sum([
            1 if thermo['energy_trend'] == 'dissipating' else 0,
            1 if coherence and coherence.get('is_decoupling') else 0,
            1 if coherence and coherence.get('is_fragmenting') else 0,
            2 if coherence and coherence.get('coupling_state') == 'decoupled' else 0,
            1 if state and state.get('trend') == 'diverging' else 0,
            1 if state and state.get('current_distance', 0) > 3 else 0,
        ])

        return {
            'severity': severity,
            'severity_score': severity_score,
            'issues': issues,
            'rudder_signal': rudder_signal,
            'summary': self._generate_summary(thermo, coherence, state, issues),
        }

    def _generate_summary(
        self,
        thermo: Dict,
        coherence: Optional[Dict],
        state: Optional[Dict],
        issues: List[str]
    ) -> str:
        """Generate human-readable summary using eigenvalue-based coherence."""
        if not issues:
            return "System operating normally. Energy conserved, signals coupled, state stable."

        parts = []

        # Energy status
        if thermo['energy_trend'] == 'dissipating':
            parts.append(f"Energy dissipating at {thermo['dissipation_rate_mean']:.4f}/unit")

        # Coherence status (eigenvalue-based)
        if coherence and not coherence.get('error'):
            if coherence.get('is_decoupling') or coherence.get('is_fragmenting'):
                coh_current = coherence.get('current_coherence', 0)
                coh_baseline = coherence.get('baseline_coherence', 0)
                eff_dim = coherence.get('current_effective_dim', 1)
                n_signals = coherence.get('n_signals', 1)

                if coherence.get('is_fragmenting'):
                    parts.append(
                        f"System fragmenting into {eff_dim:.1f} modes (of {n_signals} signals)"
                    )
                elif coherence.get('is_decoupling'):
                    parts.append(
                        f"Coherence dropped from {coh_baseline:.2f} to {coh_current:.2f}"
                    )

                # Add structure state if fragmented
                structure = coherence.get('structure_state', 'unknown')
                if structure == 'fragmented':
                    parts.append(f"Structure: fragmented")
                elif structure == 'clustered':
                    parts.append(f"Structure: {int(round(eff_dim))} clusters")

        # State status
        if state and state.get('trend') == 'diverging':
            parts.append(
                f"State distance {state['current_distance']:.1f}σ from baseline, "
                f"velocity {state['current_velocity']:.4f}"
            )

        return ". ".join(parts) + "." if parts else "Issues detected."

    # =========================================================================
    # FLEET ANALYSIS
    # =========================================================================

    def analyze_fleet(self) -> Dict[str, Any]:
        """
        Analyze all entities, return summary.

        Identifies which entities have the Rudder signal.
        """
        entities = self.get_entities()

        results = []
        rudder_signals = []
        severities = []

        for entity in entities:
            try:
                analysis = self.analyze_system(entity)
                if 'error' in analysis:
                    continue

                severity = analysis['diagnosis']['severity']
                has_signal = analysis['diagnosis']['rudder_signal']

                results.append({
                    'entity_id': entity,
                    'severity': severity,
                    'rudder_signal': has_signal,
                    'state_distance': analysis['L1_state'].get('current_distance') if analysis.get('L1_state') else None,
                })

                severities.append(severity)
                if has_signal:
                    rudder_signals.append(entity)

            except Exception as e:
                continue

        severity_counts = dict(Counter(severities))
        n_entities = len(results)

        return {
            'n_entities': n_entities,
            'severity_counts': severity_counts,
            'rudder_signals': rudder_signals,
            'n_critical': severity_counts.get('critical', 0),
            'n_warning': severity_counts.get('warning', 0),
            'pct_healthy': severity_counts.get('normal', 0) / n_entities * 100 if n_entities > 0 else 0,
            'entities': sorted(results, key=lambda x: (
                0 if x['severity'] == 'critical' else
                1 if x['severity'] == 'warning' else
                2 if x['severity'] == 'watch' else 3
            )),
        }

    # =========================================================================
    # ML FEATURE EXPORT
    # =========================================================================

    def get_ml_features(self, entity_id: str, window_size: int = 20) -> Optional[Dict[str, Any]]:
        """
        Extract ML-ready features for predictive maintenance models.

        Based on turbofan failure trajectory analysis:
        - Divergence begins at ~cycle 110
        - state_velocity > 0.1 is primary early warning
        - FAILS_EARLY engines hit trigger at 72.9% of life (43 cycles remaining)
        - SURVIVES_LONGER engines hit trigger at 60.9% (87 cycles remaining)

        Args:
            entity_id: Entity to extract features for
            window_size: Rolling window size for smoothed features (default 20)

        Returns:
            Dict with ML features or None if entity not found
        """
        entity_data = self.physics.filter(
            pl.col("entity_id") == entity_id
        ).sort("I")

        if entity_data.is_empty():
            return None

        n = len(entity_data)
        if n < window_size:
            return {'error': f'Need at least {window_size} observations'}

        # Extract arrays
        coherence = entity_data["coherence"].to_numpy()
        state_velocity = entity_data["state_velocity"].to_numpy() if "state_velocity" in entity_data.columns else np.zeros(n)
        dissipation = entity_data["dissipation_rate"].to_numpy() if "dissipation_rate" in entity_data.columns else np.zeros(n)
        n_signals = entity_data["n_signals"].to_numpy()[0] if "n_signals" in entity_data.columns else 1

        # Normalized effective dim
        if "effective_dim" in entity_data.columns:
            norm_dim = entity_data["effective_dim"].to_numpy() / n_signals
        else:
            norm_dim = np.ones(n)

        # Current values (latest observation)
        current = {
            'current_coherence': float(coherence[-1]),
            'current_velocity': float(state_velocity[-1]),
            'current_dissipation': float(dissipation[-1]),
            'current_norm_dim': float(norm_dim[-1]),
        }

        # Rolling averages (last window_size)
        smoothed = {
            'coherence_ma': float(np.mean(coherence[-window_size:])),
            'velocity_ma': float(np.mean(state_velocity[-window_size:])),
            'dissipation_ma': float(np.mean(dissipation[-window_size:])),
        }

        # Volatility (last window_size)
        volatility = {
            'coherence_volatility': float(np.std(coherence[-window_size:])),
            'velocity_volatility': float(np.std(state_velocity[-window_size:])),
        }

        # Drift from initial state
        drift = {
            'coherence_drift': float(coherence[-1] - coherence[0]),
            'velocity_drift': float(state_velocity[-1] - state_velocity[0]),
        }

        # Warning state accumulators
        high_velocity_mask = state_velocity > 0.1
        low_coherence_mask = coherence < 0.5
        rudder_mask = high_velocity_mask & low_coherence_mask

        accumulators = {
            'pct_time_high_velocity': float(np.sum(high_velocity_mask) / n * 100),
            'pct_time_low_coherence': float(np.sum(low_coherence_mask) / n * 100),
            'pct_time_rudder_signal': float(np.sum(rudder_mask) / n * 100),
        }

        # First warning cycle
        first_high_velocity = np.where(high_velocity_mask)[0]
        first_rudder = np.where(rudder_mask)[0]

        triggers = {
            'first_high_velocity_cycle': int(first_high_velocity[0]) if len(first_high_velocity) > 0 else None,
            'first_rudder_signal_cycle': int(first_rudder[0]) if len(first_rudder) > 0 else None,
            'first_high_velocity_pct_life': float(first_high_velocity[0] / n * 100) if len(first_high_velocity) > 0 else None,
        }

        # Velocity spikes (>50% increase in 10 cycles)
        if n >= 10:
            velocity_change = (state_velocity[10:] - state_velocity[:-10]) / np.maximum(np.abs(state_velocity[:-10]), 1e-6)
            n_velocity_spikes = int(np.sum(velocity_change > 0.5))
        else:
            n_velocity_spikes = 0

        # Composite risk score (0-100)
        current_high_velocity = 1 if state_velocity[-1] > 0.1 else 0
        current_rudder_signal = 1 if (state_velocity[-1] > 0.1 and coherence[-1] < 0.5) else 0

        risk_score = (
            (current_high_velocity * 30) +
            (current_rudder_signal * 20) +
            (min(accumulators['pct_time_rudder_signal'], 50) * 0.5) +
            (min(n_velocity_spikes, 20) * 1.0) +
            (10 if drift['velocity_drift'] > 0.1 else 0) +
            (10 if drift['coherence_drift'] < -0.2 else 0)
        )

        risk_level = (
            'CRITICAL' if risk_score >= 70 else
            'WARNING' if risk_score >= 50 else
            'WATCH' if risk_score >= 30 else
            'NORMAL'
        )

        return {
            'entity_id': entity_id,
            'n_observations': n,
            **current,
            **smoothed,
            **volatility,
            **drift,
            **accumulators,
            **triggers,
            'n_velocity_spikes': n_velocity_spikes,
            'current_high_velocity': current_high_velocity,
            'current_rudder_signal': current_rudder_signal,
            'rudder_risk_score': round(risk_score, 1),
            'risk_level': risk_level,
        }

    def get_all_ml_features(self, window_size: int = 20) -> List[Dict[str, Any]]:
        """
        Extract ML features for all entities.

        Returns:
            List of feature dicts, one per entity
        """
        entities = self.get_entities()
        features = []

        for entity in entities:
            feat = self.get_ml_features(entity, window_size=window_size)
            if feat and 'error' not in feat:
                features.append(feat)

        return sorted(features, key=lambda x: x['rudder_risk_score'], reverse=True)


# =============================================================================
# FACTORY AND CACHING
# =============================================================================

_interpreters: Dict[str, PhysicsInterpreter] = {}
_physics_configs: Dict[str, Tuple[Optional[Dict], Optional[PhysicsConstants]]] = {}


def get_physics_interpreter(
    job_id: str = None,
    physics_path: Path = None,
    physics_df: pl.DataFrame = None,
) -> PhysicsInterpreter:
    """
    Get or create PhysicsInterpreter for a job.

    Args:
        job_id: Job ID to look up physics.parquet
        physics_path: Or provide path directly
        physics_df: Or provide DataFrame directly
    """
    if physics_df is not None:
        return PhysicsInterpreter(physics_df=physics_df)

    if physics_path:
        return PhysicsInterpreter(physics_path=physics_path)

    if job_id:
        # Check cache
        if job_id in _interpreters:
            return _interpreters[job_id]

        # Find physics.parquet for this job
        prism_output = Path(f"/Users/jasonrudder/prism/data/output/{job_id}")
        physics_path = prism_output / "physics.parquet"
        obs_path = prism_output / "observations_enriched.parquet"

        if not physics_path.exists():
            raise FileNotFoundError(f"physics.parquet not found for job {job_id}")

        # Get config if set
        signal_units, constants = _physics_configs.get(job_id, (None, None))

        interpreter = PhysicsInterpreter(
            physics_path=physics_path,
            obs_enriched_path=obs_path if obs_path.exists() else None,
            signal_units=signal_units,
            constants=constants,
        )
        _interpreters[job_id] = interpreter
        return interpreter

    raise ValueError("Must provide job_id, physics_path, or physics_df")


def set_physics_config(
    job_id: str,
    signal_units: Optional[Dict[str, str]],
    constants: Optional[PhysicsConstants]
):
    """
    Configure real physics calculations for a job.

    When units and constants are provided, PhysicsInterpreter computes
    real energy (Joules) instead of proxy.
    """
    _physics_configs[job_id] = (signal_units, constants)

    # Invalidate cached interpreter so it gets recreated with new config
    if job_id in _interpreters:
        del _interpreters[job_id]


def clear_physics_cache():
    """Clear all cached interpreters."""
    _interpreters.clear()


__all__ = [
    'PhysicsInterpreter',
    'SystemDiagnosis',
    'get_physics_interpreter',
    'set_physics_config',
    'clear_physics_cache',
]
