"""
Generate Rossler attractor dataset for Prime.

Rossler ODE system:
    dx/dt = -y - z
    dy/dt = x + a*y
    dz/dt = b + z*(x - c)

Parameters: a=0.2, b=0.2, c=5.7 (standard chaotic regime)
Initial conditions: [1.0, 1.0, 0.0]
Integration: RK45, dt=0.05, 500 time units transient discarded
Pulse: binary threshold on z > 17.0 (~1% spike rate)

Outputs 5 signals per CSV:
    x, y, z          - raw ODE state variables
    rossler pulse     - x (continuous chaotic oscillation)
    pulse             - (z > 17) as 0/1 (sparse binary events)

Usage:
    python scripts/generate_rossler.py
"""

import numpy as np
from scipy.integrate import solve_ivp
from pathlib import Path
import csv

# ---------------------------------------------------------------------------
# Parameters (all documented, nothing magic)
# ---------------------------------------------------------------------------
A, B, C = 0.2, 0.2, 5.7        # Standard chaotic Rossler
X0 = [1.0, 1.0, 0.0]           # Initial conditions
DT = 0.05                       # Sample spacing (time units)
TRANSIENT = 500.0               # Discard this much to settle onto attractor
TRAIN_N = 24_000                # Training samples
TEST_N = 6_000                  # Test samples
PULSE_THRESHOLD = 17.0          # z > this => pulse=1 (~1% rate)
DOMAINS_DIR = Path.home() / "domains" / "rossler"


def rossler(t, state):
    x, y, z = state
    return [-y - z, x + A * y, B + z * (x - C)]


def write_csv(path, x, y, z):
    """Write 5-column CSV: x, y, z, rossler pulse, pulse."""
    pulse = (z > PULSE_THRESHOLD).astype(float)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "rossler pulse", "pulse"])
        for i in range(len(x)):
            w.writerow([x[i], y[i], z[i], x[i], pulse[i]])


def main():
    total = TRAIN_N + TEST_N
    t_end = TRANSIENT + total * DT
    t_eval = np.arange(TRANSIENT, t_end, DT)[:total]

    print(f"Rossler ODE: a={A}, b={B}, c={C}")
    print(f"Integrating {TRANSIENT + total * DT:.0f} time units "
          f"(discarding first {TRANSIENT:.0f})...")

    sol = solve_ivp(
        rossler, [0, t_end], X0, t_eval=t_eval,
        method="RK45", max_step=0.01, rtol=1e-10, atol=1e-12,
    )
    assert sol.success, sol.message

    x, y, z = sol.y[0][:total], sol.y[1][:total], sol.y[2][:total]
    pulse = (z > PULSE_THRESHOLD).astype(float)

    print(f"  x : [{x.min():.3f}, {x.max():.3f}]  mean={x.mean():.3f}")
    print(f"  y : [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")
    print(f"  z : [{z.min():.3f}, {z.max():.3f}]  mean={z.mean():.3f}")
    print(f"  pulse: {pulse.sum():.0f}/{len(pulse)} "
          f"({100 * pulse.mean():.1f}%)")

    for split, start, n in [("train", 0, TRAIN_N), ("test", TRAIN_N, TEST_N)]:
        d = DOMAINS_DIR / split
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"rossler_{split}.csv"
        write_csv(p, x[start:start + n], y[start:start + n], z[start:start + n])
        print(f"  {p}  ({n} rows)")


if __name__ == "__main__":
    main()
