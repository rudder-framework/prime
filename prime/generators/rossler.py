"""
Generate Rössler attractor dataset as observations.parquet.

Rössler ODE system:
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

Default parameters: a=0.2, b=0.2, c=5.7 (standard chaotic regime)

Outputs 4 signals in canonical long format:
    x, y, z   — raw ODE state variables
    pulse     — binary: 1 when z > mean(z) + 2*std(z), 0 otherwise
"""

import numpy as np
import polars as pl
from pathlib import Path
from scipy.integrate import solve_ivp


# Defaults
DEFAULT_N_SAMPLES = 24_000
DEFAULT_DT = 0.05
DEFAULT_A = 0.2
DEFAULT_B = 0.2
DEFAULT_C = 5.7
TRANSIENT = 500.0  # Time units to discard (let attractor settle)
INITIAL_CONDITIONS = [1.0, 1.0, 0.0]


def _rossler_ode(t, state, a, b, c):
    x, y, z = state
    return [-y - z, x + a * y, b + z * (x - c)]


def generate_rossler(
    output_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    dt: float = DEFAULT_DT,
    a: float = DEFAULT_A,
    b: float = DEFAULT_B,
    c: float = DEFAULT_C,
) -> Path:
    """
    Generate Rössler attractor and write observations.parquet.

    Args:
        output_dir: Directory to write observations.parquet
        n_samples: Number of samples to generate
        dt: Integration time step
        a, b, c: Rössler parameters

    Returns:
        Path to the written observations.parquet
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Integrate ODE
    t_end = TRANSIENT + n_samples * dt
    t_eval = np.linspace(TRANSIENT, t_end, n_samples, endpoint=False)

    print(f"Rössler ODE: a={a}, b={b}, c={c}, dt={dt}, n={n_samples}")
    print(f"Integrating {t_end:.0f} time units (discarding first {TRANSIENT:.0f})...")

    sol = solve_ivp(
        _rossler_ode, [0, t_end], INITIAL_CONDITIONS,
        args=(a, b, c),
        t_eval=t_eval,
        method="RK45",
        max_step=dt,
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success, f"ODE integration failed: {sol.message}"

    x, y, z = sol.y[0], sol.y[1], sol.y[2]

    # Pulse: binary threshold on z spikes
    z_threshold = z.mean() + 2.0 * z.std()
    pulse = (z > z_threshold).astype(np.float64)

    print(f"  x : [{x.min():.3f}, {x.max():.3f}]  mean={x.mean():.3f}")
    print(f"  y : [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")
    print(f"  z : [{z.min():.3f}, {z.max():.3f}]  mean={z.mean():.3f}")
    print(f"  z_threshold: {z_threshold:.3f}")
    print(f"  pulse: {int(pulse.sum())}/{len(pulse)} ({100 * pulse.mean():.1f}%)")

    # Build canonical long-format DataFrame
    signal_0 = np.arange(n_samples, dtype=np.float64)
    signals = {"x": x, "y": y, "z": z, "pulse": pulse}

    frames = []
    for name, values in signals.items():
        frames.append(pl.DataFrame({
            "cohort": [""] * n_samples,
            "signal_0": signal_0,
            "signal_id": [name] * n_samples,
            "value": values,
        }))

    df = pl.concat(frames).sort(["cohort", "signal_0", "signal_id"])

    # Ensure types
    df = df.with_columns([
        pl.col("cohort").cast(pl.String),
        pl.col("signal_0").cast(pl.Float64),
        pl.col("signal_id").cast(pl.String),
        pl.col("value").cast(pl.Float64),
    ])

    output_path = output_dir / "observations.parquet"
    df.write_parquet(output_path)

    # Write signals.parquet
    from prime.ingest.signal_metadata import write_signal_metadata
    write_signal_metadata(
        df, output_dir,
        descriptions={
            "x": "Rössler x state variable",
            "y": "Rössler y state variable",
            "z": "Rössler z state variable",
            "pulse": "Binary threshold on z spikes",
        },
    )

    n_signals = df["signal_id"].n_unique()
    print(f"\n  {n_signals} signals × {n_samples} samples = {len(df):,} rows")
    print(f"  → {output_path}")

    return output_path
