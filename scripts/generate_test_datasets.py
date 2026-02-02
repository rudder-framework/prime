"""
Generate Test Datasets for ORTHON Validation

Creates observations.parquet files for multiple domains to test:
- 10-dimension typology classification
- Engine selection mapping
- Window recommendation

Domains:
1. bearing - Vibration signals (periodic, harmonic, impulsive)
2. cmapss - Turbofan degradation (trending, non-stationary)
3. fama_french - Financial returns (stationary, heavy-tailed)
4. chemical - Reaction kinetics (exponential, deterministic)
5. synthetic - Reference benchmarks (all signal types)

Run:
    python scripts/generate_test_datasets.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Dict, List, Tuple


def generate_bearing_data(n_samples: int = 10000, n_bearings: int = 3) -> pl.DataFrame:
    """
    Generate bearing vibration data (FEMTO-like).

    Characteristics:
    - acc_x, acc_y: Acceleration in m/s² (velocity derivative)
    - Periodic with harmonics (rotating machinery)
    - Increasing amplitude over life (degradation)
    - Impulsive events (defects)
    """
    np.random.seed(42)
    rows = []

    for bearing in range(1, n_bearings + 1):
        # Each bearing has different fundamental frequency
        f0 = 25 + bearing * 5  # 30, 35, 40 Hz (at 10kHz sample rate)

        # Degradation: amplitude increases over time
        degradation = 1 + 0.5 * np.linspace(0, 1, n_samples) ** 2

        for i in range(n_samples):
            t = i / 1000  # Sample at 1 kHz

            # Base rotation with harmonics
            fundamental = np.sin(2 * np.pi * f0 * t)
            harmonic_2 = 0.3 * np.sin(2 * np.pi * 2 * f0 * t)
            harmonic_3 = 0.15 * np.sin(2 * np.pi * 3 * f0 * t)

            # Add degradation-dependent noise
            noise = np.random.randn() * 0.1 * degradation[i]

            # Occasional impulses (bearing defects)
            impulse = 0
            if np.random.random() < 0.001 * degradation[i]:
                impulse = np.random.randn() * 5

            acc_x = degradation[i] * (fundamental + harmonic_2 + harmonic_3 + noise + impulse)
            # Y is 90 degrees phase shifted
            acc_y = degradation[i] * (
                np.cos(2 * np.pi * f0 * t) +
                0.3 * np.cos(2 * np.pi * 2 * f0 * t) +
                0.15 * np.cos(2 * np.pi * 3 * f0 * t) +
                noise
            )

            rows.append({
                'unit_id': f'bearing_{bearing}',
                'signal_id': 'acc_x',
                'I': i,
                'value': float(acc_x),
            })
            rows.append({
                'unit_id': f'bearing_{bearing}',
                'signal_id': 'acc_y',
                'I': i,
                'value': float(acc_y),
            })

    return pl.DataFrame(rows)


def generate_cmapss_data(n_cycles: int = 200, n_engines: int = 5) -> pl.DataFrame:
    """
    Generate turbofan degradation data (C-MAPSS-like).

    Characteristics:
    - 21 sensors with different behaviors
    - Non-stationary (trending toward failure)
    - Some sensors constant (operating settings)
    - Volatility clustering near end of life
    """
    np.random.seed(43)
    rows = []

    # Sensor characteristics
    sensors = {
        # Trending sensors (degradation)
        'sensor_02': {'base': 642, 'trend': 0.5, 'noise': 0.2},  # Total temp
        'sensor_03': {'base': 1580, 'trend': 2.0, 'noise': 1.0},  # HPC outlet temp
        'sensor_04': {'base': 1400, 'trend': 1.5, 'noise': 0.8},  # LPT outlet temp
        'sensor_07': {'base': 550, 'trend': 0.3, 'noise': 0.5},   # Total pressure
        'sensor_11': {'base': 47.5, 'trend': -0.02, 'noise': 0.1}, # Flow ratio
        'sensor_12': {'base': 520, 'trend': 0.2, 'noise': 0.3},   # Fuel flow
        'sensor_15': {'base': 8.5, 'trend': 0.01, 'noise': 0.05},  # Bleed enthalpy
        'sensor_17': {'base': 390, 'trend': 0.5, 'noise': 0.4},   # HPT coolant bleed
        'sensor_20': {'base': 38.5, 'trend': -0.015, 'noise': 0.08}, # HP turbine ratio
        'sensor_21': {'base': 23.3, 'trend': 0.01, 'noise': 0.05},  # Fan speed ratio

        # Near-constant sensors (operating conditions)
        'sensor_01': {'base': 518.67, 'trend': 0, 'noise': 0.01},  # Altitude
        'sensor_05': {'base': 14.62, 'trend': 0, 'noise': 0.01},   # Fan inlet temp
        'sensor_06': {'base': 21.61, 'trend': 0, 'noise': 0.01},   # LPC outlet pressure
        'sensor_09': {'base': 9046, 'trend': 0, 'noise': 1},       # Physical fan speed
        'sensor_10': {'base': 1.3, 'trend': 0, 'noise': 0.001},    # Corrected fan speed
        'sensor_14': {'base': 8.4, 'trend': 0, 'noise': 0.01},     # HPC bleed
        'sensor_16': {'base': 0.03, 'trend': 0, 'noise': 0.001},   # Fuel flow ratio
        'sensor_18': {'base': 2388, 'trend': 0, 'noise': 5},       # LPT speed
        'sensor_19': {'base': 100, 'trend': 0, 'noise': 0.5},      # Demanded speed

        # Volatile sensors
        'sensor_08': {'base': 2388, 'trend': 0.8, 'noise': 3, 'volatile': True},
        'sensor_13': {'base': 2388, 'trend': 0.6, 'noise': 2, 'volatile': True},
    }

    for engine in range(1, n_engines + 1):
        # Random RUL for this engine
        rul = np.random.randint(150, n_cycles)
        health = np.linspace(1, 0, rul)  # 1 = healthy, 0 = failure

        for cycle in range(min(rul, n_cycles)):
            h = health[cycle] if cycle < len(health) else 0

            for sensor_name, params in sensors.items():
                base = params['base']
                trend = params['trend']
                noise = params['noise']

                # Degradation effect (increases as health decreases)
                degradation = trend * (1 - h) * cycle

                # Noise increases near end of life
                noise_scale = noise * (1 + (1 - h) * 2)

                # Volatility clustering for some sensors
                if params.get('volatile') and h < 0.3:
                    noise_scale *= 3

                value = base + degradation + np.random.randn() * noise_scale

                rows.append({
                    'unit_id': f'engine_{engine}',
                    'signal_id': sensor_name,
                    'I': cycle,
                    'value': float(value),
                })

    return pl.DataFrame(rows)


def generate_fama_french_data(n_days: int = 2520, n_industries: int = 12) -> pl.DataFrame:
    """
    Generate financial returns data (Fama-French-like).

    Characteristics:
    - Daily returns (stationary)
    - Heavy tails (kurtosis > 3)
    - Volatility clustering (GARCH)
    - Low autocorrelation
    - Some cross-correlation between industries
    """
    np.random.seed(44)
    rows = []

    industries = [
        'NoDur', 'Durbl', 'Manuf', 'Enrgy', 'HiTec', 'Telcm',
        'Shops', 'Hlth', 'Utils', 'Other', 'Money', 'BusEq'
    ][:n_industries]

    # Market factor (common movement)
    market_vol = 0.01
    market_returns = np.zeros(n_days)
    for i in range(1, n_days):
        market_vol = 0.0001 + 0.85 * market_vol + 0.1 * market_returns[i-1]**2
        market_returns[i] = np.sqrt(market_vol) * np.random.standard_t(5) * 0.01

    for ind_idx, industry in enumerate(industries):
        # Industry-specific parameters
        beta = 0.8 + 0.4 * np.random.random()  # Market beta
        idio_vol = 0.005 + 0.01 * np.random.random()

        for i in range(n_days):
            # Return = beta * market + idiosyncratic
            idio = np.random.standard_t(4) * idio_vol
            ret = beta * market_returns[i] + idio

            rows.append({
                'unit_id': '',  # Single unit (market)
                'signal_id': industry,
                'I': i,
                'value': float(ret),
            })

    return pl.DataFrame(rows)


def generate_chemical_data(n_samples: int = 5000) -> pl.DataFrame:
    """
    Generate chemical reaction kinetics data.

    Characteristics:
    - Deterministic (exponential decay/growth)
    - Smooth trajectories
    - Multiple species with coupled dynamics
    - Units: mol/L/s for reaction rates
    """
    np.random.seed(45)
    rows = []

    # A → B → C (consecutive first-order reactions)
    k1 = 0.1  # Rate constant A→B
    k2 = 0.05  # Rate constant B→C

    dt = 0.01
    t = np.arange(n_samples) * dt

    # Initial concentrations
    A0 = 1.0
    B0 = 0.0
    C0 = 0.0

    # Analytical solutions with small measurement noise
    A = A0 * np.exp(-k1 * t) + np.random.randn(n_samples) * 0.001
    B = (A0 * k1 / (k2 - k1)) * (np.exp(-k1 * t) - np.exp(-k2 * t)) + np.random.randn(n_samples) * 0.001
    C = A0 * (1 + (k1 * np.exp(-k2 * t) - k2 * np.exp(-k1 * t)) / (k2 - k1)) + np.random.randn(n_samples) * 0.001

    # Reaction rates (derivatives) - these have velocity units
    rate_A = -k1 * A  # mol/L/s
    rate_B = k1 * A - k2 * B  # mol/L/s
    rate_C = k2 * B  # mol/L/s

    # Temperature (oscillating around setpoint)
    temp_setpoint = 350  # K
    temp = temp_setpoint + 5 * np.sin(0.02 * t) + np.random.randn(n_samples) * 0.5

    signals = {
        'conc_A': A,
        'conc_B': B,
        'conc_C': C,
        'rate_A': rate_A,  # Has /s unit (velocity)
        'rate_B': rate_B,  # Has /s unit (velocity)
        'rate_C': rate_C,  # Has /s unit (velocity)
        'temperature': temp,
    }

    for signal_id, values in signals.items():
        for i, v in enumerate(values):
            rows.append({
                'unit_id': 'reactor_1',
                'signal_id': signal_id,
                'I': i,
                'value': float(v),
            })

    return pl.DataFrame(rows)


def generate_synthetic_benchmarks(n_samples: int = 5000) -> pl.DataFrame:
    """
    Generate synthetic benchmark signals for typology validation.

    Signal types:
    - white_noise: STATIONARY, RANDOM, SHORT_MEMORY, STOCHASTIC
    - sine_wave: STATIONARY, PERIODIC, DETERMINISTIC
    - random_walk: NON_STATIONARY, TRENDING, LONG_MEMORY, STOCHASTIC
    - lorenz: STATIONARY, CHAOTIC, DETERMINISTIC
    - ar1_mean_revert: STATIONARY, MEAN_REVERTING, ANTI_PERSISTENT
    - impulse_train: STATIONARY, PERIODIC, IMPULSIVE, HARMONIC
    - heteroscedastic: VOLATILITY_CLUSTERING
    - constant: CONSTANT
    - discrete: DISCRETE
    """
    np.random.seed(46)
    rows = []

    # White noise
    white_noise = np.random.randn(n_samples)

    # Pure sine wave
    t = np.arange(n_samples)
    sine_wave = np.sin(2 * np.pi * 0.05 * t)

    # Random walk
    random_walk = np.cumsum(np.random.randn(n_samples))

    # Lorenz attractor (x component)
    sigma, rho, beta = 10.0, 28.0, 8.0/3.0
    dt = 0.01
    x, y, z = 1.0, 1.0, 1.0
    lorenz = []
    for _ in range(n_samples + 1000):
        dx = sigma * (y - x)
        dy = x * (rho - z) - y
        dz = x * y - beta * z
        x += dx * dt
        y += dy * dt
        z += dz * dt
        lorenz.append(x)
    lorenz = np.array(lorenz[1000:n_samples+1000])

    # AR(1) mean-reverting
    phi = 0.3
    ar1 = np.zeros(n_samples)
    for i in range(1, n_samples):
        ar1[i] = phi * ar1[i-1] + np.random.randn()

    # Impulse train
    impulse = np.random.randn(n_samples) * 0.1
    for i in range(0, n_samples, 100):
        impulse[i] += 10.0

    # Heteroscedastic (GARCH-like)
    hetero = np.zeros(n_samples)
    var = 1.0
    for i in range(1, n_samples):
        var = 0.1 + 0.85 * var + 0.1 * hetero[i-1]**2
        hetero[i] = np.sqrt(var) * np.random.randn()

    # Constant
    constant = np.ones(n_samples) * 42.0

    # Discrete
    discrete = np.random.randint(0, 5, n_samples).astype(float)

    signals = {
        'white_noise': white_noise,
        'sine_wave': sine_wave,
        'random_walk': random_walk,
        'lorenz': lorenz,
        'ar1_mean_revert': ar1,
        'impulse_train': impulse,
        'heteroscedastic': hetero,
        'constant': constant,
        'discrete': discrete,
    }

    for signal_id, values in signals.items():
        for i, v in enumerate(values):
            rows.append({
                'unit_id': 'benchmark',
                'signal_id': signal_id,
                'I': i,
                'value': float(v),
            })

    return pl.DataFrame(rows)


def main():
    """Generate all test datasets."""
    output_base = Path(__file__).parent.parent / 'data' / 'test_domains'

    datasets = {
        'bearing': generate_bearing_data,
        'cmapss': generate_cmapss_data,
        'fama_french': generate_fama_french_data,
        'chemical': generate_chemical_data,
        'synthetic': generate_synthetic_benchmarks,
    }

    for name, generator in datasets.items():
        output_dir = output_base / name
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Generating {name} dataset...")
        print(f"{'='*60}")

        df = generator()
        output_path = output_dir / 'observations.parquet'
        df.write_parquet(output_path)

        # Summary
        n_signals = df['signal_id'].n_unique()
        n_units = df['unit_id'].n_unique()
        n_rows = len(df)

        print(f"  Output: {output_path}")
        print(f"  Signals: {n_signals}")
        print(f"  Units: {n_units}")
        print(f"  Rows: {n_rows}")

        # Show signal names
        signals = df['signal_id'].unique().to_list()
        print(f"  Signal IDs: {signals}")

    print(f"\n{'='*60}")
    print("All datasets generated!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print("  1. Run typology classification:")
    print("     python -m orthon.ingest.typology_raw data/test_domains/<domain>/observations.parquet")
    print("  2. Run SQL classification:")
    print("     duckdb -c \".read orthon/sql/typology_v2.sql\"")
    print("  3. Generate manifest:")
    print("     python -m orthon.ingest.manifest_generator data/test_domains/<domain>/typology.parquet")


if __name__ == '__main__':
    main()
