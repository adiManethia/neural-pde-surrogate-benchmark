from typing import Tuple

import numpy as np


def make_random_initial_condition_2d(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    rng: np.random.Generator,
    min_bumps: int = 1,
    max_bumps: int = 5,
) -> np.ndarray:
    """
    Create one smooth random 2D initial temperature field using Gaussian bumps.

    Args:
        x_grid: x-coordinate grid, shape (ny, nx)
        y_grid: y-coordinate grid, shape (ny, nx)
        rng: NumPy random generator
        min_bumps: minimum number of Gaussian bumps
        max_bumps: maximum number of Gaussian bumps

    Returns:
        u0: initial 2D field, shape (ny, nx)
    """
    u0 = np.zeros_like(x_grid, dtype=np.float32)

    n_bumps = rng.integers(min_bumps, max_bumps + 1)

    for _ in range(n_bumps):
        center_x = rng.uniform(0.15, 0.85)
        center_y = rng.uniform(0.15, 0.85)
        width = rng.uniform(0.005, 0.04)
        amplitude = rng.uniform(0.4, 1.2)

        bump = amplitude * np.exp(
            -(
                (x_grid - center_x) ** 2
                +
                (y_grid - center_y) ** 2
            )
            / width
        )

        u0 += bump.astype(np.float32)

    max_value = np.max(u0)
    if max_value > 0:
        u0 = u0 / max_value

    # Fixed zero boundary
    u0[0, :] = 0.0
    u0[-1, :] = 0.0
    u0[:, 0] = 0.0
    u0[:, -1] = 0.0

    return u0.astype(np.float32)


def solve_heat_equation_2d(
    u0: np.ndarray,
    alpha: float,
    total_time: float,
    dx: float,
    dt: float,
) -> np.ndarray:
    """
    Solve 2D heat equation using explicit finite differences.

    PDE:
        du/dt = alpha * (d2u/dx2 + d2u/dy2)

    Assumption:
        dx = dy

    Update:
        u_new[i,j] = u[i,j] + r * (
            u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4u[i,j]
        )

    where:
        r = alpha * dt / dx^2

    Stability condition:
        r <= 0.25

    Args:
        u0: initial field, shape (ny, nx)
        alpha: diffusion coefficient
        total_time: final simulation time
        dx: spatial grid spacing
        dt: time step

    Returns:
        u: final field, shape (ny, nx)
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    if total_time <= 0:
        raise ValueError("total_time must be positive.")

    stability_number = alpha * dt / dx**2

    if stability_number > 0.25:
        raise ValueError(
            f"Unstable 2D solver: alpha*dt/dx^2 = {stability_number:.3f}. "
            "For explicit 2D heat equation, it should be <= 0.25."
        )

    u = u0.copy()
    n_steps = int(total_time / dt)

    for _ in range(n_steps):
        u_new = u.copy()

        u_new[1:-1, 1:-1] = u[1:-1, 1:-1] + stability_number * (
            u[2:, 1:-1]
            + u[:-2, 1:-1]
            + u[1:-1, 2:]
            + u[1:-1, :-2]
            - 4.0 * u[1:-1, 1:-1]
        )

        # Fixed zero boundary
        u_new[0, :] = 0.0
        u_new[-1, :] = 0.0
        u_new[:, 0] = 0.0
        u_new[:, -1] = 0.0

        u = u_new

    return u.astype(np.float32)


def make_2d_grid(
    nx: int,
    ny: int,
    length: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Create 2D grid and return grid spacing.

    Returns:
        x: 1D x grid, shape (nx,)
        y: 1D y grid, shape (ny,)
        x_grid: 2D x grid, shape (ny, nx)
        y_grid: 2D y grid, shape (ny, nx)
        dx: grid spacing
    """
    x = np.linspace(0.0, length, nx, dtype=np.float32)
    y = np.linspace(0.0, length, ny, dtype=np.float32)

    dx = float(x[1] - x[0])

    x_grid, y_grid = np.meshgrid(x, y)

    return x, y, x_grid.astype(np.float32), y_grid.astype(np.float32), dx