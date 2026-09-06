from typing import Tuple

import numpy as np


def make_random_initial_condition_1d(
    x: np.ndarray,
    rng: np.random.Generator,
    min_bumps: int = 1,
    max_bumps: int = 4,
) -> np.ndarray:
    """
    Create one smooth random 1D initial condition using Gaussian bumps.

    Args:
        x: spatial grid, shape (nx,)
        rng: NumPy random generator
        min_bumps: minimum number of Gaussian bumps
        max_bumps: maximum number of Gaussian bumps

    Returns:
        u0: initial field, shape (nx,)
    """
    u0 = np.zeros_like(x, dtype=np.float32)

    n_bumps = rng.integers(min_bumps, max_bumps + 1)

    for _ in range(n_bumps):
        center = rng.uniform(0.15, 0.85)
        width = rng.uniform(0.002, 0.025)
        amplitude = rng.uniform(0.4, 1.2)

        bump = amplitude * np.exp(-((x - center) ** 2) / width)
        u0 += bump.astype(np.float32)

    max_value = np.max(u0)
    if max_value > 0:
        u0 = u0 / max_value

    # Dirichlet boundary condition
    u0[0] = 0.0
    u0[-1] = 0.0

    return u0.astype(np.float32)


def solve_heat_equation_1d(
    u0: np.ndarray,
    alpha: float,
    total_time: float,
    dx: float,
    dt: float,
) -> np.ndarray:
    """
    Solve 1D heat equation using explicit finite differences.

    PDE:
        du/dt = alpha * d2u/dx2

    Update:
        u_new[i] = u[i] + r * (u[i-1] - 2u[i] + u[i+1])

    where:
        r = alpha * dt / dx^2

    Stability condition:
        r <= 0.5

    Args:
        u0: initial condition, shape (nx,)
        alpha: diffusion coefficient
        total_time: final simulation time
        dx: spatial grid spacing
        dt: time step

    Returns:
        u: final field, shape (nx,)
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if total_time <= 0:
        raise ValueError("total_time must be positive.")

    stability_number = alpha * dt / dx**2

    if stability_number > 0.5:
        raise ValueError(
            f"Unstable 1D solver: alpha*dt/dx^2 = {stability_number:.3f}. "
            "For explicit 1D heat equation, it should be <= 0.5."
        )

    u = u0.copy()
    n_steps = int(total_time / dt)

    for _ in range(n_steps):
        u_new = u.copy()

        u_new[1:-1] = u[1:-1] + stability_number * (
            u[:-2]
            - 2.0 * u[1:-1]
            + u[2:]
        )

        # Fixed zero boundary
        u_new[0] = 0.0
        u_new[-1] = 0.0

        u = u_new

    return u.astype(np.float32)


def make_1d_grid(
    nx: int,
    length: float,
) -> Tuple[np.ndarray, float]:
    """
    Create 1D grid and return grid spacing.
    """
    x = np.linspace(0.0, length, nx, dtype=np.float32)
    dx = float(x[1] - x[0])

    return x, dx