"""
ORTHON ML Discovery Engine
==========================

Find interesting patterns in PRISM dynamics + physics outputs.
Zero calculations — just pattern recognition on what PRISM computed.
"""

import polars as pl
import numpy as np
from typing import List, Dict, Optional


class DiscoveryEngine:
    """Find interesting patterns in dynamics + physics."""

    def __init__(
        self,
        dynamics: pl.DataFrame,
        physics: Optional[pl.DataFrame] = None,
    ):
        self.dyn = dynamics
        self.phys = physics

    def discover(self) -> List[Dict]:
        """
        Run all discovery routines.

        Returns:
            List of findings, each with:
            - title: str
            - description: str
            - recommendation: Optional[str]
            - severity: str ('info', 'warning', 'critical')
        """
        findings = []
        findings.extend(self._dynamics_findings())
        if self.phys is not None and not self.phys.is_empty():
            findings.extend(self._physics_findings())
            findings.extend(self._cross_findings())
        return findings

    def _dynamics_findings(self) -> List[Dict]:
        """Analyze dynamics.parquet for insights."""
        findings = []

        if self.dyn.is_empty():
            return findings

        # Degradation rate (hd_slope)
        if 'hd_slope' in self.dyn.columns:
            slopes = self.dyn['hd_slope'].drop_nulls().to_numpy()
            if len(slopes) > 0:
                degrading = (slopes < -0.01).sum()
                degrading_pct = 100 * degrading / len(slopes)
                critical = (slopes < -0.05).sum()

                if critical > 0:
                    findings.append({
                        'title': f'{critical} entities in critical state',
                        'description': (
                            f'{critical}/{len(slopes)} entities have hd_slope < -0.05. '
                            f'These are experiencing rapid coherence loss.'
                        ),
                        'recommendation': 'Prioritize inspection of critical entities.',
                        'severity': 'critical',
                    })

                if degrading_pct > 80:
                    findings.append({
                        'title': f'{degrading_pct:.0f}% of entities degrading',
                        'description': (
                            f'Fleet-wide degradation detected. '
                            f'{degrading}/{len(slopes)} have hd_slope < -0.01.'
                        ),
                        'recommendation': 'Investigate common cause affecting entire fleet.',
                        'severity': 'warning',
                    })

                # Mean degradation rate
                mean_slope = np.mean(slopes)
                if mean_slope < -0.02:
                    findings.append({
                        'title': f'Fleet mean hd_slope: {mean_slope:.4f}',
                        'description': 'Average degradation rate is elevated across the fleet.',
                        'recommendation': None,
                        'severity': 'info',
                    })

        # Acceleration
        if 'hd_acceleration_mean' in self.dyn.columns:
            accel = self.dyn['hd_acceleration_mean'].drop_nulls().to_numpy()
            if len(accel) > 0:
                accelerating = (accel < -0.001).sum()
                accel_pct = 100 * accelerating / len(accel)

                if accel_pct > 50:
                    findings.append({
                        'title': f'{accel_pct:.0f}% accelerating degradation',
                        'description': (
                            f'Degradation is speeding up for {accelerating}/{len(accel)} entities. '
                            f'Negative acceleration means increasing loss rate.'
                        ),
                        'recommendation': 'Expect worsening conditions without intervention.',
                        'severity': 'warning',
                    })

        # Regime transitions
        if 'n_transitions' in self.dyn.columns:
            trans = self.dyn['n_transitions'].drop_nulls().to_numpy()
            if len(trans) > 0:
                p90 = np.percentile(trans, 90)
                outliers = (trans > p90).sum()

                if outliers > 0 and p90 > 5:
                    findings.append({
                        'title': f'{outliers} entities with high regime instability',
                        'description': (
                            f'These entities have >{p90:.0f} transitions (90th percentile). '
                            f'High transition counts indicate behavioral instability.'
                        ),
                        'recommendation': 'Check for measurement noise or genuine instability.',
                        'severity': 'info',
                    })

        return findings

    def _physics_findings(self) -> List[Dict]:
        """Analyze physics.parquet for insights."""
        findings = []

        if self.phys is None or self.phys.is_empty():
            return findings

        # Energy profile
        if 'hamiltonian_T' in self.phys.columns and 'hamiltonian_V' in self.phys.columns:
            ke = self.phys['hamiltonian_T'].drop_nulls().to_numpy()
            pe = self.phys['hamiltonian_V'].drop_nulls().to_numpy()

            if len(ke) > 0 and len(pe) > 0:
                total = ke + pe
                ke_frac = np.mean(ke / np.where(total > 0, total, 1))

                if ke_frac > 0.95:
                    findings.append({
                        'title': 'System dominated by kinetic energy',
                        'description': (
                            f'{ke_frac*100:.0f}% kinetic, minimal potential wells. '
                            f'No restoring force exists in the dynamics.'
                        ),
                        'recommendation': 'Degradation is irreversible without external intervention.',
                        'severity': 'warning',
                    })
                elif ke_frac < 0.3:
                    findings.append({
                        'title': 'System in potential-dominated state',
                        'description': (
                            f'Only {ke_frac*100:.0f}% kinetic energy. '
                            f'System is relatively stable in potential wells.'
                        ),
                        'recommendation': None,
                        'severity': 'info',
                    })

        # Gibbs free energy / spontaneity
        if 'gibbs_free_energy' in self.phys.columns:
            gibbs = self.phys['gibbs_free_energy'].drop_nulls().to_numpy()
            if len(gibbs) > 0:
                spontaneous = (gibbs < 0).sum()
                spont_pct = 100 * spontaneous / len(gibbs)

                if spont_pct > 90:
                    findings.append({
                        'title': f'{spont_pct:.0f}% thermodynamically spontaneous',
                        'description': (
                            f'Gibbs free energy < 0 for {spontaneous}/{len(gibbs)} entities. '
                            f'Degradation is energetically favored.'
                        ),
                        'recommendation': None,
                        'severity': 'info',
                    })

        # Hamiltonian variance (conservation)
        if 'hamiltonian_variance' in self.phys.columns:
            h_var = self.phys['hamiltonian_variance'].drop_nulls().to_numpy()
            if len(h_var) > 0:
                dissipative = (h_var > 0.1).sum()
                diss_pct = 100 * dissipative / len(h_var)

                if diss_pct > 50:
                    findings.append({
                        'title': f'{diss_pct:.0f}% of entities are dissipative',
                        'description': (
                            f'High Hamiltonian variance indicates energy is not conserved. '
                            f'System is losing energy to friction/damping.'
                        ),
                        'recommendation': None,
                        'severity': 'info',
                    })

        return findings

    def _cross_findings(self) -> List[Dict]:
        """Cross-correlate dynamics and physics."""
        findings = []

        if self.phys is None or self.phys.is_empty():
            return findings

        # Correlation: hd_slope vs gibbs_free_energy
        if 'hd_slope' in self.dyn.columns and 'gibbs_free_energy' in self.phys.columns:
            try:
                merged = self.dyn.select(['entity_id', 'hd_slope']).join(
                    self.phys.select(['entity_id', 'gibbs_free_energy']),
                    on='entity_id',
                    how='inner'
                )

                if len(merged) > 5:
                    x = merged['hd_slope'].drop_nulls().to_numpy()
                    y = merged['gibbs_free_energy'].drop_nulls().to_numpy()

                    if len(x) > 5 and len(y) > 5 and len(x) == len(y):
                        corr = np.corrcoef(x, y)[0, 1]

                        if abs(corr) > 0.5:
                            findings.append({
                                'title': f'hd_slope correlates with Gibbs energy (r={corr:.2f})',
                                'description': (
                                    f'Degradation rate is linked to thermodynamic favorability. '
                                    f'Correlation coefficient: {corr:.3f}'
                                ),
                                'recommendation': None,
                                'severity': 'info',
                            })
            except Exception:
                pass  # Skip if join fails

        # Correlation: hd_slope vs kinetic energy
        if 'hd_slope' in self.dyn.columns and 'hamiltonian_T' in self.phys.columns:
            try:
                merged = self.dyn.select(['entity_id', 'hd_slope']).join(
                    self.phys.select(['entity_id', 'hamiltonian_T']),
                    on='entity_id',
                    how='inner'
                )

                if len(merged) > 5:
                    x = merged['hd_slope'].drop_nulls().to_numpy()
                    y = merged['hamiltonian_T'].drop_nulls().to_numpy()

                    if len(x) > 5 and len(y) > 5 and len(x) == len(y):
                        corr = np.corrcoef(x, y)[0, 1]

                        if abs(corr) > 0.5:
                            findings.append({
                                'title': f'hd_slope correlates with kinetic energy (r={corr:.2f})',
                                'description': (
                                    f'Faster-degrading entities have different energy profiles. '
                                    f'Correlation coefficient: {corr:.3f}'
                                ),
                                'recommendation': 'Investigate energy-degradation relationship.',
                                'severity': 'info',
                            })
            except Exception:
                pass

        return findings
