"""
Tests for Discrete & Sparse Signal Classification (PR5)
"""

import pytest

from prime.config.discrete_sparse_config import (
    DISCRETE_SPARSE_CONFIG,
    DISCRETE_SPARSE_SPECTRAL,
    DISCRETE_SPARSE_ENGINES,
)
from prime.typology.discrete_sparse import (
    is_constant,
    is_binary,
    is_discrete,
    is_impulsive,
    is_event,
    is_step,
    is_intermittent,
    classify_discrete_sparse,
    apply_discrete_sparse_classification,
)


# ============================================================
# Test: CONSTANT Detection
# ============================================================

class TestConstant:

    def test_zero_std_is_constant(self):
        """signal_std = 0 → CONSTANT"""
        row = {'signal_std': 0.0, 'signal_mean': 100.0,
               'unique_ratio': 0.001, 'n_samples': 1000}
        assert is_constant(row) is True

    def test_tiny_std_is_constant(self):
        """signal_std ≈ 0 → CONSTANT"""
        row = {'signal_std': 1e-12, 'signal_mean': 0.0,
               'unique_ratio': 0.01, 'n_samples': 1000}
        assert is_constant(row) is True

    def test_low_unique_ratio_is_constant(self):
        """unique_ratio < 0.001 + low CV → CONSTANT"""
        row = {'signal_std': 0.0005, 'signal_mean': 100.0,
               'unique_ratio': 0.0005, 'n_samples': 1000}
        assert is_constant(row) is True

    def test_missing_std_not_constant(self):
        """signal_std=None → not CONSTANT (can't determine)"""
        row = {'signal_std': None, 'unique_ratio': 0.01,
               'n_samples': 1000, 'hurst': 0.5, 'perm_entropy': 0.0}
        assert is_constant(row) is False

    def test_normal_signal_not_constant(self):
        """Normal signal → not CONSTANT"""
        row = {'signal_std': 1.5, 'unique_ratio': 0.8,
               'hurst': 0.6, 'perm_entropy': 0.95}
        assert is_constant(row) is False


# ============================================================
# Test: BINARY Detection
# ============================================================

class TestBinary:

    def test_two_values_is_binary(self):
        """exactly 2 unique values → BINARY"""
        row = {'unique_count': 2}
        assert is_binary(row) is True

    def test_inferred_unique_count(self):
        """infer unique_count from unique_ratio"""
        row = {'unique_ratio': 0.002, 'n_samples': 1000}  # 2 values
        assert is_binary(row) is True

    def test_three_values_not_binary(self):
        """3 values → not BINARY"""
        row = {'unique_count': 3}
        assert is_binary(row) is False

    def test_one_value_not_binary(self):
        """1 value → not BINARY (that's CONSTANT)"""
        row = {'unique_count': 1}
        assert is_binary(row) is False


# ============================================================
# Test: DISCRETE Detection
# ============================================================

class TestDiscrete:

    def test_integer_low_unique_ratio(self):
        """integer + low unique_ratio → DISCRETE"""
        row = {'is_integer': True, 'unique_ratio': 0.03, 'n_samples': 1000}
        assert is_discrete(row) is True

    def test_integer_few_unique_values(self):
        """integer + few unique values → DISCRETE"""
        row = {'is_integer': True, 'unique_ratio': 0.10,
               'unique_count': 20, 'n_samples': 200}
        assert is_discrete(row) is True

    def test_float_not_discrete(self):
        """float values → not DISCRETE"""
        row = {'is_integer': False, 'unique_ratio': 0.03, 'n_samples': 1000}
        assert is_discrete(row) is False

    def test_many_integer_values_not_discrete(self):
        """many integer values → not DISCRETE"""
        row = {'is_integer': True, 'unique_ratio': 0.50,
               'unique_count': 500, 'n_samples': 1000}
        assert is_discrete(row) is False


# ============================================================
# Test: IMPULSIVE Detection
# ============================================================

class TestImpulsive:

    def test_high_kurtosis_and_crest(self):
        """kurtosis > 20 + crest > 10 → IMPULSIVE"""
        row = {'kurtosis': 25.0, 'crest_factor': 12.0}
        assert is_impulsive(row) is True

    def test_high_kurtosis_low_crest(self):
        """kurtosis > 20 but crest < 10 → not IMPULSIVE"""
        row = {'kurtosis': 25.0, 'crest_factor': 5.0}
        assert is_impulsive(row) is False

    def test_low_kurtosis_high_crest(self):
        """kurtosis < 20 but crest > 10 → not IMPULSIVE"""
        row = {'kurtosis': 10.0, 'crest_factor': 15.0}
        assert is_impulsive(row) is False

    def test_normal_signal_not_impulsive(self):
        """normal signal → not IMPULSIVE"""
        row = {'kurtosis': 3.0, 'crest_factor': 3.5}
        assert is_impulsive(row) is False


# ============================================================
# Test: EVENT Detection
# ============================================================

class TestEvent:

    def test_sparse_with_high_kurtosis(self):
        """sparsity > 0.8 + kurtosis > 10 → EVENT"""
        row = {'sparsity': 0.85, 'kurtosis': 15.0}
        assert is_event(row) is True

    def test_sparse_low_kurtosis(self):
        """sparse but low kurtosis → not EVENT"""
        row = {'sparsity': 0.90, 'kurtosis': 5.0}
        assert is_event(row) is False

    def test_dense_high_kurtosis(self):
        """dense with high kurtosis → not EVENT (maybe IMPULSIVE)"""
        row = {'sparsity': 0.30, 'kurtosis': 15.0}
        assert is_event(row) is False


# ============================================================
# Test: STEP Detection
# ============================================================

class TestStep:

    def test_sparse_derivative(self):
        """derivative mostly zero → STEP"""
        row = {'derivative_sparsity': 0.95, 'unique_ratio': 0.05}
        assert is_step(row) is True

    def test_missing_derivative_sparsity(self):
        """missing derivative_sparsity → can't determine"""
        row = {'unique_ratio': 0.05}
        assert is_step(row) is False

    def test_dense_derivative(self):
        """derivative not sparse → not STEP"""
        row = {'derivative_sparsity': 0.50, 'unique_ratio': 0.05}
        assert is_step(row) is False


# ============================================================
# Test: INTERMITTENT Detection
# ============================================================

class TestIntermittent:

    def test_bursty_signal(self):
        """significant zero runs + mid sparsity → INTERMITTENT"""
        row = {'zero_run_ratio': 0.40, 'sparsity': 0.50}
        assert is_intermittent(row) is True

    def test_too_sparse_is_event(self):
        """sparsity > 0.8 → EVENT, not INTERMITTENT"""
        row = {'zero_run_ratio': 0.50, 'sparsity': 0.85}
        assert is_intermittent(row) is False

    def test_too_dense(self):
        """low sparsity → continuous, not INTERMITTENT"""
        row = {'zero_run_ratio': 0.40, 'sparsity': 0.20}
        assert is_intermittent(row) is False


# ============================================================
# Test: Classification Priority
# ============================================================

class TestClassificationPriority:

    def test_constant_beats_binary(self):
        """CONSTANT takes priority over BINARY"""
        row = {'signal_std': 0.0, 'signal_mean': 1.0,
               'unique_count': 2, 'unique_ratio': 0.001, 'n_samples': 1000}
        assert classify_discrete_sparse(row) == 'CONSTANT'

    def test_binary_beats_discrete(self):
        """BINARY takes priority over DISCRETE"""
        row = {'signal_std': 1.0, 'unique_count': 2,
               'is_integer': True, 'unique_ratio': 0.002, 'n_samples': 1000}
        assert classify_discrete_sparse(row) == 'BINARY'

    def test_impulsive_beats_event(self):
        """IMPULSIVE takes priority over EVENT"""
        row = {'kurtosis': 25.0, 'crest_factor': 12.0, 'sparsity': 0.85}
        assert classify_discrete_sparse(row) == 'IMPULSIVE'

    def test_continuous_returns_none(self):
        """Continuous signal returns None"""
        row = {'signal_std': 1.5, 'unique_ratio': 0.80,
               'kurtosis': 3.0, 'crest_factor': 3.5,
               'sparsity': 0.10, 'is_integer': False}
        assert classify_discrete_sparse(row) is None


# ============================================================
# Test: Real-World Integration
# ============================================================

# Electrochemistry Mn_II - all zeros
MN_II_ELECTROCHEMISTRY = {
    'signal_id': 'Mn_II',
    'cohort': 'CL10',
    'n_samples': 891,
    'signal_std': 0.0,
    'unique_ratio': 0.001,  # Only 1 unique value (0)
    'hurst': 0.5,
    'perm_entropy': 0.0,
    'spectral_flatness': 0.0,
}

# Binary switch signal
SMART_SWITCH = {
    'signal_id': 'light_switch',
    'cohort': 'living_room',
    'n_samples': 10000,
    'signal_std': 0.5,
    'unique_count': 2,
    'unique_ratio': 0.002,  # 2 values / 1000 effective = 0.2%
    'is_integer': True,
}

# Gear position (6-speed)
GEAR_POSITION = {
    'signal_id': 'gear',
    'cohort': 'vehicle_1',
    'n_samples': 5000,
    'signal_std': 1.8,
    'unique_count': 7,  # N, 1-6
    'unique_ratio': 0.0014,
    'is_integer': True,
}

# Earthquake events
EARTHQUAKE_CATALOG = {
    'signal_id': 'magnitude',
    'cohort': 'california',
    'n_samples': 100000,
    'signal_std': 1.2,
    'sparsity': 0.92,  # 92% no earthquake
    'kurtosis': 12.5,
    'is_integer': False,
}

# Bearing impact (early fault)
BEARING_IMPACT = {
    'signal_id': 'acc_z',
    'cohort': 'bearing_1',
    'n_samples': 10000,
    'signal_std': 0.5,
    'kurtosis': 28.0,
    'crest_factor': 15.0,
    'sparsity': 0.60,
}

# Continuous signal (should fall through)
TEMPERATURE_SENSOR = {
    'signal_id': 'temp_c',
    'cohort': 'room_1',
    'n_samples': 1000,
    'signal_std': 2.5,
    'unique_ratio': 0.85,
    'kurtosis': 2.8,
    'crest_factor': 3.0,
    'sparsity': 0.0,
    'is_integer': False,
}


class TestRealWorldIntegration:

    def test_mn_ii_is_constant(self):
        """Electrochemistry Mn_II (all zeros) → CONSTANT"""
        result = apply_discrete_sparse_classification(MN_II_ELECTROCHEMISTRY)
        assert result['temporal_pattern'] == 'CONSTANT'
        assert result['spectral'] == 'NONE'
        assert result['is_discrete_sparse'] is True

    def test_smart_switch_is_binary(self):
        """Smart home switch → BINARY"""
        result = apply_discrete_sparse_classification(SMART_SWITCH)
        assert result['temporal_pattern'] == 'BINARY'
        assert result['spectral'] == 'SWITCHING'

    def test_gear_is_discrete(self):
        """Gear position → DISCRETE"""
        result = apply_discrete_sparse_classification(GEAR_POSITION)
        assert result['temporal_pattern'] == 'DISCRETE'
        assert result['spectral'] == 'QUANTIZED'

    def test_earthquake_is_event(self):
        """Earthquake catalog → EVENT"""
        result = apply_discrete_sparse_classification(EARTHQUAKE_CATALOG)
        assert result['temporal_pattern'] == 'EVENT'
        assert result['spectral'] == 'SPARSE'

    def test_bearing_impact_is_impulsive(self):
        """Bearing impact → IMPULSIVE"""
        result = apply_discrete_sparse_classification(BEARING_IMPACT)
        assert result['temporal_pattern'] == 'IMPULSIVE'
        assert result['spectral'] == 'BROADBAND'

    def test_temperature_is_continuous(self):
        """Temperature sensor → None (continuous)"""
        result = apply_discrete_sparse_classification(TEMPERATURE_SENSOR)
        assert result['is_discrete_sparse'] is False
        assert 'temporal_pattern' not in result or result.get('temporal_pattern') is None


# ============================================================
# Test: Engine Adjustments
# ============================================================

class TestEngineAdjustments:

    def test_constant_removes_all_engines(self):
        """CONSTANT signal → no engines"""
        row = {'signal_std': 0.0, 'signal_mean': 1.0,
               'unique_ratio': 0.001, 'n_samples': 1000,
               'engines': ['hurst', 'kurtosis', 'trend_r2']}
        result = apply_discrete_sparse_classification(row)
        assert result['engines'] == []

    def test_binary_adds_transition_engines(self):
        """BINARY signal → adds transition analysis"""
        row = {'unique_count': 2, 'signal_std': 0.5, 'engines': ['kurtosis']}
        result = apply_discrete_sparse_classification(row)
        assert 'transition_count' in result['engines']
        assert 'duty_cycle' in result['engines']

    def test_event_adds_event_engines(self):
        """EVENT signal → adds event analysis"""
        row = {'sparsity': 0.90, 'kurtosis': 15.0, 'engines': ['hurst']}
        result = apply_discrete_sparse_classification(row)
        assert 'event_rate' in result['engines']
        assert 'hurst' not in result['engines']  # Removed
