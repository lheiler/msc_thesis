#!/usr/bin/env python3
"""
Single-file Jansen–Rit neural mass model simulator
-------------------------------------------------

Usage
-----
Run from the command line, for example::

    python jansen_rit_simulator.py --t 1000 --dt 0.1 --p 120

This integrates the classic Jansen–Rit cortical column equations and plots the
pyramidal membrane potential.  The script also exposes a ``simulate`` function
so you can import it from your own analysis notebooks.

Dependencies: numpy, scipy, matplotlib (all pure-Python, widely available).
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from numpy import random as rng

# ───────────────────────────────────────────────── Parameters ────────────────────────────────────────────────
# Synaptic gains (mV)
A = 3.25  # excitatory
B = 22.0  # inhibitory

# Time constants (s⁻¹)
a = 100.0  # excitatory
b = 50.0   # inhibitory

# Average number of synaptic contacts
C = 135.0

# Sigmoid (firing response) parameters
E0 = 2.5   # maximum firing rate (s⁻¹)
V0 = 6.0    # PSP at 50 % of max. rate (mV)
R  = 0.56   # slope (mV⁻¹)

# Derived connectivity constants
C1 = C
C2 = 0.8 * C
C3 = 0.25 * C
C4 = 0.25 * C

# ───────────────────────────────────────────────── Model ────────────────────────────────────────────────────

def S(v: float | np.ndarray) -> float | np.ndarray:
    """Sigmoid – converts membrane potential (mV) to firing rate (s⁻¹)."""
    return 2.0 * E0 / (1.0 + np.exp(R * (V0 - v)))


def p_t(t):
    # 2 Hz square wave that toggles between 90 and 150 s⁻¹
    return 90.0 + 60.0 * (np.sin(2.0 * np.pi * t / 5.0) > 0.0)

def _jansen_rit_ode(t: float, y: np.ndarray, p: float) -> np.ndarray:
    """RHS of the Jansen–Rit ODE system.

    State vector ``y`` (length 6):
        y0 = x₀  pyramidal membrane potential
        y1 = x₁  excitatory interneuron membrane potential
        y2 = x₂  inhibitory interneuron membrane potential
        y3 = x₀̇ first derivative of x₀
        y4 = x₁̇ first derivative of x₁
        y5 = x₂̇ first derivative of x₂
    """
    y0, y1, y2, y3, y4, y5 = y

    dy0 = y3
    dy1 = y4
    dy2 = y5

    sigma = 0.0  # noise amplitude (mV)
    dy3 =   A * a * S(y1 - y2 + sigma * rng.normal())              - 2.0 * a * y3 - (a ** 2) * y0
    dy4 =   A * a * (p + C2 * S(C1 * y0))   - 2.0 * a * y4 - (a ** 2) * y1
    dy5 =   B * b * C4 * S(C3 * y0)         - 2.0 * b * y5 - (b ** 2) * y2

    return np.array([dy0, dy1, dy2, dy3, dy4, dy5])


# ───────────────────────────────────────────── Simulation helper ────────────────────────────────────────────

def simulate(
    t_ms: float = 1000.0,
    dt_ms: float = 0.1,
    p: float = 120.0,
    y0: np.ndarray | None = None,
    method: str = "RK45",
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the model.

    Parameters
    ----------
    t_ms
        Total simulated time (milliseconds).
    dt_ms
        Integration time-step (milliseconds) for dense output.
    p
        Average external excitatory input (pulses s⁻¹).
    y0
        Initial condition (length 6).  If *None*, a tiny random perturbation is used.
    method
        Any *scipy.integrate.solve_ivp* solver (``RK45``, ``BDF``, …).

    Returns
    -------
    t
        Time vector (ms).
    y
        Array of shape (6, len(t)) containing the state trajectory.
    """
    t_end = t_ms / 1000.0  # seconds
    dt    = dt_ms / 1000.0

    if y0 is None:
        rng = np.random.default_rng()
        y0 = rng.normal(0.0, 1e-4, size=6)

    t_eval = np.arange(0.0, t_end + dt, dt)

    sol = solve_ivp(
        _jansen_rit_ode,
        t_span=(0.0, t_end),
        y0=y0,
        t_eval=t_eval,
        args=(p,),
        method=method,
        rtol=1e-5,
        atol=1e-8,
    )

    if not sol.success:
        raise RuntimeError(f"Integrator failed: {sol.message}")

    return sol.t * 1000.0, sol.y  # back to ms


# ───────────────────────────────────────────── Command-line interface ───────────────────────────────────────

def _cli() -> None:
    

    t, y = simulate(t_ms=10000.0)
    x0   = y[0]  # pyramidal membrane potential

    plt.figure(figsize=(10, 3))
    plt.plot(t, x0)
    plt.xlabel("Time (ms)")
    plt.ylabel("Pyramidal membrane potential (mV)")
    plt.title("Jansen–Rit cortical column")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    _cli()
    
    
    
    