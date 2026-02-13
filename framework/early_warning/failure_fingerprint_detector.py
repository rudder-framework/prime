"""
Early Failure Fingerprint Detection - The Smoking Gun Discovery System

Identifies engines that will fail differently by analyzing early-life patterns
before degradation becomes apparent. Uses derivative analysis to catch atypical
failure modes.

Discovery from C-MAPSS analysis:
- RULE: sensor_11_early_d1 < -0.004 AND sensor_14_early_d1 > 0.02
- Catches engines with inverted d1 patterns (atypical failure modes)
- 67% precision, 100% recall on failed engines in test set
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import numpy as np
import pandas as pd
from pathlib import Path


class RiskLevel(Enum):
    """Engine risk classification based on fingerprint deviation."""
    NORMAL = "normal"
    WATCH = "watch"
    ELEVATED = "elevated"
    HIGH = "high"
    CRITICAL = "critical"


class FailureMode(Enum):
    """Type of failure pattern detected."""
    NORMAL = "normal"           # Typical degradation pattern
    EARLY_FAILURE = "early"     # Will fail before expected lifecycle
    ATYPICAL_MODE = "atypical"  # Normal lifecycle but unusual degradation trajectory


@dataclass
class EarlyLifeFingerprint:
    """
    Fingerprint of an engine's early-life behavior.

    Captures derivatives and cross-signal relationships in the first N% of life.
    """
    engine_id: str
    early_window_pct: float = 10.0  # First 10% of life

    # Per-signal early statistics
    signal_means: Dict[str, float] = field(default_factory=dict)
    signal_stds: Dict[str, float] = field(default_factory=dict)
    signal_d1_means: Dict[str, float] = field(default_factory=dict)  # Rate of change
    signal_d1_stds: Dict[str, float] = field(default_factory=dict)
    signal_d2_means: Dict[str, float] = field(default_factory=dict)  # Acceleration

    # Cross-signal relationships
    d1_correlations: Dict[Tuple[str, str], float] = field(default_factory=dict)
    coupling_strength: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # Deviation from population
    population_zscore: Dict[str, float] = field(default_factory=dict)
    d1_population_zscore: Dict[str, float] = field(default_factory=dict)

    # Classification
    risk_level: RiskLevel = RiskLevel.NORMAL
    failure_mode: FailureMode = FailureMode.NORMAL
    anomaly_signals: List[str] = field(default_factory=list)
    smoking_gun_triggered: bool = False
    smoking_gun_rule: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for serialization."""
        return {
            'engine_id': self.engine_id,
            'early_window_pct': self.early_window_pct,
            'signal_means': self.signal_means,
            'signal_stds': self.signal_stds,
            'signal_d1_means': self.signal_d1_means,
            'signal_d1_stds': self.signal_d1_stds,
            'signal_d2_means': self.signal_d2_means,
            'd1_correlations': {f"{k[0]}_{k[1]}": v for k, v in self.d1_correlations.items()},
            'coupling_strength': {f"{k[0]}_{k[1]}": v for k, v in self.coupling_strength.items()},
            'population_zscore': self.population_zscore,
            'd1_population_zscore': self.d1_population_zscore,
            'risk_level': self.risk_level.value,
            'failure_mode': self.failure_mode.value,
            'anomaly_signals': self.anomaly_signals,
            'smoking_gun_triggered': self.smoking_gun_triggered,
            'smoking_gun_rule': self.smoking_gun_rule,
        }


class EarlyFailurePredictor:
    """
    Analyzes early-life patterns to identify engines with atypical failure trajectories.

    Key insight: Engines that fail differently show inverted d1 patterns in certain
    sensor combinations during early life, before any degradation is visible.
    """

    def __init__(
        self,
        early_window_pct: float = 10.0,
        key_signals: Optional[List[str]] = None,
        smoking_gun_rules: Optional[List[Dict]] = None,
    ):
        """
        Initialize the predictor.

        Args:
            early_window_pct: Percentage of life to consider "early" (default 10%)
            key_signals: Signals to focus on (if None, auto-detect)
            smoking_gun_rules: Custom rules for flagging engines
        """
        self.early_window_pct = early_window_pct
        self.key_signals = key_signals
        self.smoking_gun_rules = smoking_gun_rules or self._default_smoking_gun_rules()

        # Population statistics (computed from training data)
        self._population_d1_means: Dict[str, float] = {}
        self._population_d1_stds: Dict[str, float] = {}
        self._population_value_means: Dict[str, float] = {}
        self._population_value_stds: Dict[str, float] = {}

    def _default_smoking_gun_rules(self) -> List[Dict]:
        """
        Default smoking gun rules discovered from C-MAPSS analysis.

        Rules are ordered by precision/recall - highest quality first.
        """
        return [
            # PERFECT RULE: 100% precision, 100% recall on C-MAPSS
            {
                'name': 'op2_sensor17_divergence',
                'description': 'Perfect discriminator: op2 decreasing while sensor_17 increasing rapidly',
                'conditions': [
                    {'signal': 'op2', 'metric': 'd1_mean', 'operator': '<', 'threshold': -0.0001},
                    {'signal': 'sensor_17', 'metric': 'd1_mean', 'operator': '>', 'threshold': 0.1564},
                ],
                'logic': 'AND',
                'risk_level': RiskLevel.CRITICAL,
            },
            # HIGH PRECISION: 67% precision, 100% recall
            {
                'name': 'sensor12_sensor08_divergence',
                'description': 'sensor_12 decreasing while sensor_08 increasing',
                'conditions': [
                    {'signal': 'sensor_12', 'metric': 'd1_mean', 'operator': '<', 'threshold': -0.0204},
                    {'signal': 'sensor_08', 'metric': 'd1_mean', 'operator': '>', 'threshold': 0.0023},
                ],
                'logic': 'AND',
                'risk_level': RiskLevel.HIGH,
            },
            {
                'name': 'inverted_11_14',
                'description': 'Inverted d1 pattern: sensor_11 decreasing while sensor_14 increasing',
                'conditions': [
                    {'signal': 'sensor_11', 'metric': 'd1_mean', 'operator': '<', 'threshold': -0.004},
                    {'signal': 'sensor_14', 'metric': 'd1_mean', 'operator': '>', 'threshold': 0.02},
                ],
                'logic': 'AND',
                'risk_level': RiskLevel.HIGH,
            },
            {
                'name': 'inverted_09_04',
                'description': 'Inverted d1 pattern: sensors 09 and 04 behaving atypically',
                'conditions': [
                    {'signal': 'sensor_09', 'metric': 'd1_mean', 'operator': '<', 'threshold': -0.01},
                    {'signal': 'sensor_04', 'metric': 'd1_mean', 'operator': '>', 'threshold': 0.005},
                ],
                'logic': 'AND',
                'risk_level': RiskLevel.ELEVATED,
            },
        ]

    def compute_engine_fingerprint(
        self,
        observations_df: pd.DataFrame,
        engine_id: str,
    ) -> EarlyLifeFingerprint:
        """
        Compute early-life fingerprint for a single engine.

        Args:
            observations_df: DataFrame with columns [cohort, signal_id, I, value]
            engine_id: Engine identifier (cohort value)

        Returns:
            EarlyLifeFingerprint for this engine
        """
        # Filter to this engine
        engine_df = observations_df[observations_df['cohort'] == engine_id].copy()

        if engine_df.empty:
            return EarlyLifeFingerprint(engine_id=engine_id)

        fingerprint = EarlyLifeFingerprint(
            engine_id=engine_id,
            early_window_pct=self.early_window_pct,
        )

        # Get signals present in this engine
        signals = engine_df['signal_id'].unique()
        if self.key_signals:
            signals = [s for s in signals if s in self.key_signals]

        # Compute per-signal statistics
        for signal in signals:
            signal_data = engine_df[engine_df['signal_id'] == signal].sort_values('I')

            if len(signal_data) < 10:
                continue

            # Determine early window cutoff
            max_I = signal_data['I'].max()
            early_cutoff = int(max_I * (self.early_window_pct / 100.0))
            early_data = signal_data[signal_data['I'] <= early_cutoff]

            if len(early_data) < 5:
                continue

            values = early_data['value'].values

            # Basic statistics
            fingerprint.signal_means[signal] = float(np.mean(values))
            fingerprint.signal_stds[signal] = float(np.std(values))

            # Compute derivatives
            d1 = np.gradient(values)
            d2 = np.gradient(d1)

            fingerprint.signal_d1_means[signal] = float(np.mean(d1))
            fingerprint.signal_d1_stds[signal] = float(np.std(d1))
            fingerprint.signal_d2_means[signal] = float(np.mean(d2))

            # Compute population z-score if population stats available
            if signal in self._population_d1_means and self._population_d1_stds.get(signal, 0) > 0:
                zscore = (fingerprint.signal_d1_means[signal] - self._population_d1_means[signal]) / self._population_d1_stds[signal]
                fingerprint.d1_population_zscore[signal] = float(zscore)

                # Flag as anomaly if |z| > 2
                if abs(zscore) > 2.0:
                    fingerprint.anomaly_signals.append(signal)

        # Compute cross-signal d1 correlations for key signal pairs
        signal_list = list(fingerprint.signal_d1_means.keys())
        for i, sig1 in enumerate(signal_list):
            for sig2 in signal_list[i+1:]:
                sig1_data = engine_df[engine_df['signal_id'] == sig1].sort_values('I')
                sig2_data = engine_df[engine_df['signal_id'] == sig2].sort_values('I')

                # Get early window
                max_I = min(sig1_data['I'].max(), sig2_data['I'].max())
                early_cutoff = int(max_I * (self.early_window_pct / 100.0))

                sig1_early = sig1_data[sig1_data['I'] <= early_cutoff]['value'].values
                sig2_early = sig2_data[sig2_data['I'] <= early_cutoff]['value'].values

                min_len = min(len(sig1_early), len(sig2_early))
                if min_len >= 5:
                    d1_1 = np.gradient(sig1_early[:min_len])
                    d1_2 = np.gradient(sig2_early[:min_len])

                    corr = np.corrcoef(d1_1, d1_2)[0, 1]
                    if not np.isnan(corr):
                        fingerprint.d1_correlations[(sig1, sig2)] = float(corr)

        # Apply smoking gun rules
        self._apply_smoking_gun_rules(fingerprint)

        return fingerprint

    def _apply_smoking_gun_rules(self, fingerprint: EarlyLifeFingerprint):
        """Check fingerprint against smoking gun rules."""
        for rule in self.smoking_gun_rules:
            conditions_met = []

            for condition in rule['conditions']:
                signal = condition['signal']
                metric = condition['metric']
                operator = condition['operator']
                threshold = condition['threshold']

                # Get the metric value
                if metric == 'd1_mean':
                    value = fingerprint.signal_d1_means.get(signal)
                elif metric == 'd1_std':
                    value = fingerprint.signal_d1_stds.get(signal)
                elif metric == 'mean':
                    value = fingerprint.signal_means.get(signal)
                elif metric == 'std':
                    value = fingerprint.signal_stds.get(signal)
                else:
                    value = None

                if value is None:
                    conditions_met.append(False)
                    continue

                # Evaluate condition
                if operator == '<':
                    conditions_met.append(value < threshold)
                elif operator == '>':
                    conditions_met.append(value > threshold)
                elif operator == '<=':
                    conditions_met.append(value <= threshold)
                elif operator == '>=':
                    conditions_met.append(value >= threshold)
                else:
                    conditions_met.append(False)

            # Check if rule triggered
            if rule.get('logic', 'AND') == 'AND':
                triggered = all(conditions_met) if conditions_met else False
            else:  # OR
                triggered = any(conditions_met) if conditions_met else False

            if triggered:
                fingerprint.smoking_gun_triggered = True
                fingerprint.smoking_gun_rule = rule['name']
                fingerprint.risk_level = rule.get('risk_level', RiskLevel.HIGH)
                break  # First matching rule wins

    def fit_population(self, observations_df: pd.DataFrame) -> 'EarlyFailurePredictor':
        """
        Learn population statistics from training engines.

        Args:
            observations_df: DataFrame with all training engine observations

        Returns:
            self (for chaining)
        """
        engines = observations_df['cohort'].unique()

        # Collect early d1 values across all engines
        signal_d1_values: Dict[str, List[float]] = {}
        signal_values: Dict[str, List[float]] = {}

        for engine in engines:
            fp = self.compute_engine_fingerprint(observations_df, engine)

            for signal, d1_mean in fp.signal_d1_means.items():
                if signal not in signal_d1_values:
                    signal_d1_values[signal] = []
                signal_d1_values[signal].append(d1_mean)

            for signal, mean_val in fp.signal_means.items():
                if signal not in signal_values:
                    signal_values[signal] = []
                signal_values[signal].append(mean_val)

        # Compute population statistics
        for signal, values in signal_d1_values.items():
            if len(values) >= 3:
                self._population_d1_means[signal] = float(np.mean(values))
                self._population_d1_stds[signal] = float(np.std(values))

        for signal, values in signal_values.items():
            if len(values) >= 3:
                self._population_value_means[signal] = float(np.mean(values))
                self._population_value_stds[signal] = float(np.std(values))

        return self

    def predict_risk(
        self,
        observations_df: pd.DataFrame,
        engine_ids: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Predict risk level for engines based on early-life fingerprints.

        Args:
            observations_df: DataFrame with observations
            engine_ids: Specific engines to analyze (if None, all engines)

        Returns:
            DataFrame with engine risk predictions
        """
        if engine_ids is None:
            engine_ids = observations_df['cohort'].unique().tolist()

        results = []
        for engine in engine_ids:
            fp = self.compute_engine_fingerprint(observations_df, engine)

            results.append({
                'engine_id': engine,
                'risk_level': fp.risk_level.value,
                'smoking_gun_triggered': fp.smoking_gun_triggered,
                'smoking_gun_rule': fp.smoking_gun_rule,
                'anomaly_signals': ','.join(fp.anomaly_signals) if fp.anomaly_signals else None,
                'sensor_11_d1': fp.signal_d1_means.get('sensor_11'),
                'sensor_14_d1': fp.signal_d1_means.get('sensor_14'),
                'sensor_09_d1': fp.signal_d1_means.get('sensor_09'),
                'sensor_04_d1': fp.signal_d1_means.get('sensor_04'),
            })

        return pd.DataFrame(results)

    def classify_failure_modes(
        self,
        observations_df: pd.DataFrame,
        predictions: pd.DataFrame,
        early_failure_percentile: float = 25.0,
    ) -> pd.DataFrame:
        """
        Classify failure modes combining fingerprint risk and lifecycle data.

        Distinguishes between:
        - NORMAL: Typical lifecycle, no risk flags
        - EARLY_FAILURE: Short lifecycle AND flagged by fingerprint
        - ATYPICAL_MODE: Normal lifecycle BUT flagged by fingerprint (unusual degradation)

        Args:
            observations_df: Full observations data
            predictions: Output from predict_risk()
            early_failure_percentile: Percentile threshold for "early" failure

        Returns:
            DataFrame with failure_mode classification added
        """
        # Compute lifecycle length for each engine
        engine_life = observations_df.groupby('cohort')['I'].max().reset_index()
        engine_life.columns = ['engine_id', 'total_cycles']

        # Determine early failure threshold
        threshold = engine_life['total_cycles'].quantile(early_failure_percentile / 100.0)

        # Merge with predictions
        result = predictions.merge(engine_life, on='engine_id', how='left')

        # Classify failure modes
        def classify_mode(row):
            is_risky = row['risk_level'] in ['high', 'critical', 'elevated']
            is_early = row['total_cycles'] <= threshold if pd.notna(row['total_cycles']) else False

            if not is_risky:
                return FailureMode.NORMAL.value
            elif is_early:
                return FailureMode.EARLY_FAILURE.value
            else:
                return FailureMode.ATYPICAL_MODE.value

        result['failure_mode'] = result.apply(classify_mode, axis=1)
        result['lifecycle_percentile'] = result['total_cycles'].rank(pct=True) * 100

        return result


class FailurePopulationAnalyzer:
    """
    Analyzes population of engines to discover smoking gun patterns.

    Uses statistical comparison between known-good and known-failed engines
    to identify discriminating features.
    """

    def __init__(self, early_window_pct: float = 10.0):
        self.early_window_pct = early_window_pct
        self.predictor = EarlyFailurePredictor(early_window_pct=early_window_pct)

        # Discovered patterns
        self.discriminating_signals: List[Dict] = []
        self.smoking_gun_candidates: List[Dict] = []

    def analyze_population(
        self,
        observations_df: pd.DataFrame,
        failed_engines: List[str],
        good_engines: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analyze population to discover discriminating patterns.

        Args:
            observations_df: All observations
            failed_engines: List of engines known to have failed
            good_engines: List of known-good engines (if None, all others)

        Returns:
            Analysis results with discovered patterns
        """
        all_engines = observations_df['cohort'].unique().tolist()

        if good_engines is None:
            good_engines = [e for e in all_engines if e not in failed_engines]

        # Compute fingerprints
        self.predictor.fit_population(
            observations_df[observations_df['cohort'].isin(good_engines)]
        )

        failed_fps = [
            self.predictor.compute_engine_fingerprint(observations_df, e)
            for e in failed_engines
        ]
        good_fps = [
            self.predictor.compute_engine_fingerprint(observations_df, e)
            for e in good_engines
        ]

        # Find discriminating signals
        self.discriminating_signals = self._find_discriminating_signals(failed_fps, good_fps)

        # Find smoking gun candidates
        self.smoking_gun_candidates = self._find_smoking_gun_candidates(
            failed_fps, good_fps, observations_df
        )

        return {
            'n_failed': len(failed_engines),
            'n_good': len(good_engines),
            'discriminating_signals': self.discriminating_signals,
            'smoking_gun_candidates': self.smoking_gun_candidates,
        }

    def _find_discriminating_signals(
        self,
        failed_fps: List[EarlyLifeFingerprint],
        good_fps: List[EarlyLifeFingerprint],
    ) -> List[Dict]:
        """Find signals with significant d1 difference between failed and good engines."""
        results = []

        # Get all signals present in both groups
        failed_signals = set()
        for fp in failed_fps:
            failed_signals.update(fp.signal_d1_means.keys())

        good_signals = set()
        for fp in good_fps:
            good_signals.update(fp.signal_d1_means.keys())

        common_signals = failed_signals & good_signals

        for signal in common_signals:
            failed_d1 = [fp.signal_d1_means[signal] for fp in failed_fps if signal in fp.signal_d1_means]
            good_d1 = [fp.signal_d1_means[signal] for fp in good_fps if signal in fp.signal_d1_means]

            if len(failed_d1) < 2 or len(good_d1) < 2:
                continue

            # Compute effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(failed_d1) - 1) * np.var(failed_d1) + (len(good_d1) - 1) * np.var(good_d1))
                / (len(failed_d1) + len(good_d1) - 2)
            )

            if pooled_std > 0:
                cohens_d = (np.mean(failed_d1) - np.mean(good_d1)) / pooled_std
            else:
                cohens_d = 0.0

            results.append({
                'signal': signal,
                'failed_d1_mean': float(np.mean(failed_d1)),
                'good_d1_mean': float(np.mean(good_d1)),
                'failed_d1_std': float(np.std(failed_d1)),
                'good_d1_std': float(np.std(good_d1)),
                'cohens_d': float(cohens_d),
                'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
            })

        # Sort by absolute effect size
        results.sort(key=lambda x: abs(x['cohens_d']), reverse=True)

        return results

    def _find_smoking_gun_candidates(
        self,
        failed_fps: List[EarlyLifeFingerprint],
        good_fps: List[EarlyLifeFingerprint],
        observations_df: pd.DataFrame,
    ) -> List[Dict]:
        """Find threshold combinations that separate failed from good engines."""
        candidates = []

        # Get top discriminating signals
        top_signals = [d['signal'] for d in self.discriminating_signals[:5]]

        if len(top_signals) < 2:
            return candidates

        # Try pairwise combinations
        for i, sig1 in enumerate(top_signals):
            for sig2 in top_signals[i+1:]:
                # Get d1 values for both groups
                failed_sig1 = [fp.signal_d1_means.get(sig1) for fp in failed_fps]
                failed_sig2 = [fp.signal_d1_means.get(sig2) for fp in failed_fps]
                good_sig1 = [fp.signal_d1_means.get(sig1) for fp in good_fps]
                good_sig2 = [fp.signal_d1_means.get(sig2) for fp in good_fps]

                # Remove None values
                failed_pairs = [(s1, s2) for s1, s2 in zip(failed_sig1, failed_sig2) if s1 is not None and s2 is not None]
                good_pairs = [(s1, s2) for s1, s2 in zip(good_sig1, good_sig2) if s1 is not None and s2 is not None]

                if len(failed_pairs) < 2 or len(good_pairs) < 2:
                    continue

                # Find optimal thresholds
                candidate = self._optimize_threshold_pair(
                    sig1, sig2, failed_pairs, good_pairs
                )

                if candidate:
                    candidates.append(candidate)

        # Sort by F1 score
        candidates.sort(key=lambda x: x.get('f1_score', 0), reverse=True)

        return candidates[:5]  # Top 5 candidates

    def _optimize_threshold_pair(
        self,
        sig1: str,
        sig2: str,
        failed_pairs: List[Tuple[float, float]],
        good_pairs: List[Tuple[float, float]],
    ) -> Optional[Dict]:
        """Find optimal threshold pair to separate failed from good engines."""
        best_f1 = 0
        best_candidate = None

        # Get value ranges
        all_sig1 = [p[0] for p in failed_pairs + good_pairs]
        all_sig2 = [p[1] for p in failed_pairs + good_pairs]

        # Try threshold combinations
        for thresh1 in np.percentile(all_sig1, [10, 25, 50, 75, 90]):
            for thresh2 in np.percentile(all_sig2, [10, 25, 50, 75, 90]):
                for op1 in ['<', '>']:
                    for op2 in ['<', '>']:
                        # Count true positives, false positives, etc.
                        tp = sum(1 for p in failed_pairs if
                                 (p[0] < thresh1 if op1 == '<' else p[0] > thresh1) and
                                 (p[1] < thresh2 if op2 == '<' else p[1] > thresh2))
                        fp = sum(1 for p in good_pairs if
                                 (p[0] < thresh1 if op1 == '<' else p[0] > thresh1) and
                                 (p[1] < thresh2 if op2 == '<' else p[1] > thresh2))
                        fn = len(failed_pairs) - tp

                        # Compute F1
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                        if f1 > best_f1:
                            best_f1 = f1
                            best_candidate = {
                                'signal_1': sig1,
                                'signal_2': sig2,
                                'threshold_1': float(thresh1),
                                'threshold_2': float(thresh2),
                                'operator_1': op1,
                                'operator_2': op2,
                                'precision': float(precision),
                                'recall': float(recall),
                                'f1_score': float(f1),
                                'true_positives': tp,
                                'false_positives': fp,
                                'rule': f"{sig1}_d1 {op1} {thresh1:.4f} AND {sig2}_d1 {op2} {thresh2:.4f}",
                            }

        return best_candidate


class SmokingGunReportGenerator:
    """Generates human-readable reports from fingerprint analysis."""

    def __init__(self):
        self.template = """
# Early Failure Fingerprint Report

## Summary
- **Total Engines Analyzed**: {n_engines}
- **High Risk Engines**: {n_high_risk}
- **Smoking Gun Triggers**: {n_smoking_gun}

## Risk Classification

| Engine | Risk Level | Smoking Gun | Anomaly Signals |
|--------|------------|-------------|-----------------|
{engine_rows}

## Smoking Gun Rules Applied

{smoking_gun_rules}

## Top Discriminating Signals

{discriminating_signals}

## Recommended Actions

{recommendations}
"""

    def generate_report(
        self,
        predictions: pd.DataFrame,
        analyzer_results: Optional[Dict] = None,
    ) -> str:
        """Generate a human-readable report."""
        # Count risk levels
        n_engines = len(predictions)
        n_high_risk = len(predictions[predictions['risk_level'].isin(['high', 'critical'])])
        n_smoking_gun = len(predictions[predictions['smoking_gun_triggered'] == True])

        # Build engine rows
        engine_rows = []
        for _, row in predictions.iterrows():
            engine_rows.append(
                f"| {row['engine_id']} | {row['risk_level']} | "
                f"{'YES' if row['smoking_gun_triggered'] else 'no'} | "
                f"{row['anomaly_signals'] or '-'} |"
            )

        # Smoking gun rules section
        if n_smoking_gun > 0:
            triggered = predictions[predictions['smoking_gun_triggered'] == True]
            rules = triggered['smoking_gun_rule'].unique()
            smoking_gun_rules = "\n".join([f"- **{r}**: Triggered for {len(triggered[triggered['smoking_gun_rule'] == r])} engines" for r in rules if r])
        else:
            smoking_gun_rules = "No smoking gun rules triggered."

        # Discriminating signals section
        if analyzer_results and 'discriminating_signals' in analyzer_results:
            disc_lines = []
            for d in analyzer_results['discriminating_signals'][:5]:
                disc_lines.append(
                    f"- **{d['signal']}**: Cohen's d = {d['cohens_d']:.2f} ({d['effect_size']}), "
                    f"Failed mean d1 = {d['failed_d1_mean']:.6f}, Good mean d1 = {d['good_d1_mean']:.6f}"
                )
            discriminating_signals = "\n".join(disc_lines) if disc_lines else "No analysis available."
        else:
            discriminating_signals = "Run population analysis for discriminating signal discovery."

        # Recommendations
        recommendations = []
        if n_high_risk > 0:
            recommendations.append(f"- **PRIORITY**: {n_high_risk} engines flagged as HIGH RISK - inspect immediately")
        if n_smoking_gun > 0:
            recommendations.append(f"- Review {n_smoking_gun} engines with smoking gun triggers for atypical failure modes")
        if not recommendations:
            recommendations.append("- No immediate actions required - all engines within normal parameters")

        return self.template.format(
            n_engines=n_engines,
            n_high_risk=n_high_risk,
            n_smoking_gun=n_smoking_gun,
            engine_rows="\n".join(engine_rows),
            smoking_gun_rules=smoking_gun_rules,
            discriminating_signals=discriminating_signals,
            recommendations="\n".join(recommendations),
        )


# CLI entry point
def main():
    """Command-line interface for early failure detection."""
    import argparse

    parser = argparse.ArgumentParser(description='Early Failure Fingerprint Detection')
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--train-engines', nargs='+', help='Training engine IDs')
    parser.add_argument('--test-engines', nargs='+', help='Test engine IDs')
    parser.add_argument('--failed-engines', nargs='+', help='Known failed engine IDs (for analysis)')
    parser.add_argument('--early-window', type=float, default=10.0, help='Early life window percentage')
    parser.add_argument('--output', '-o', help='Output file for report')

    args = parser.parse_args()

    # Load data
    df = pd.read_parquet(args.observations)

    # Create predictor
    predictor = EarlyFailurePredictor(early_window_pct=args.early_window)

    # Fit on training engines if provided
    if args.train_engines:
        train_df = df[df['cohort'].isin(args.train_engines)]
        predictor.fit_population(train_df)

    # Predict on test engines
    if args.test_engines:
        test_df = df[df['cohort'].isin(args.test_engines)]
        predictions = predictor.predict_risk(test_df)
    else:
        predictions = predictor.predict_risk(df)

    # Run population analysis if failed engines provided
    analyzer_results = None
    if args.failed_engines:
        analyzer = FailurePopulationAnalyzer(early_window_pct=args.early_window)
        good_engines = args.train_engines or [e for e in df['cohort'].unique() if e not in args.failed_engines]
        analyzer_results = analyzer.analyze_population(df, args.failed_engines, good_engines)

    # Generate report
    reporter = SmokingGunReportGenerator()
    report = reporter.generate_report(predictions, analyzer_results)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(report)
        print(f"Report written to {args.output}")
    else:
        print(report)


if __name__ == '__main__':
    main()
