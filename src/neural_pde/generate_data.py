import argparse
from pathlib import Path
from typing import Callable

import numpy as np
from tqdm import tqdm

from neural_pde.config import (
    load_config,
    get_data_dir,
    ensure_project_dirs,
)
from neural_pde.solvers.heat1d import (
    make_1d_grid,
    make_random_initial_condition_1d,
    solve_heat_equation_1d,
)
from neural_pde.solvers.heat2d import (
    make_2d_grid,
    make_random_initial_condition_2d,
    solve_heat_equation_2d,
)


def compute_stable_dt(
    dx: float,
    max_alpha: float,
    stability_factor: float,
) -> float:
    """
    Compute stable time step.

    For 1D:
        stability_factor should be <= 0.5

    For 2D:
        stability_factor should be <= 0.25

    We store the factor in YAML configs.
    """
    return stability_factor * dx**2 / max_alpha


def generate_split_1d(
    n_samples: int,
    x: np.ndarray,
    alpha_range: tuple[float, float],
    total_time: float,
    dx: float,
    dt: float,
    rng: np.random.Generator,
    split_name: str,
) -> dict[str, np.ndarray]:
    """
    Generate one 1D dataset split.
    """
    nx = len(x)

    u0_all = np.zeros((n_samples, nx), dtype=np.float32)
    alpha_all = np.zeros((n_samples, 1), dtype=np.float32)
    u_final_all = np.zeros((n_samples, nx), dtype=np.float32)

    alpha_min, alpha_max = alpha_range

    for idx in tqdm(range(n_samples), desc=f"Generating 1D {split_name}"):
        alpha = float(rng.uniform(alpha_min, alpha_max))

        u0 = make_random_initial_condition_1d(
            x=x,
            rng=rng,
        )

        u_final = solve_heat_equation_1d(
            u0=u0,
            alpha=alpha,
            total_time=total_time,
            dx=dx,
            dt=dt,
        )

        u0_all[idx] = u0
        alpha_all[idx, 0] = alpha
        u_final_all[idx] = u_final

    return {
        "x": x.astype(np.float32),
        "u0": u0_all,
        "alpha": alpha_all,
        "u_final": u_final_all,
    }


def generate_split_2d(
    n_samples: int,
    x: np.ndarray,
    y: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    alpha_range: tuple[float, float],
    total_time: float,
    dx: float,
    dt: float,
    rng: np.random.Generator,
    split_name: str,
) -> dict[str, np.ndarray]:
    """
    Generate one 2D dataset split.
    """
    ny, nx = x_grid.shape

    u0_all = np.zeros((n_samples, ny, nx), dtype=np.float32)
    alpha_all = np.zeros((n_samples, 1), dtype=np.float32)
    u_final_all = np.zeros((n_samples, ny, nx), dtype=np.float32)

    alpha_min, alpha_max = alpha_range

    for idx in tqdm(range(n_samples), desc=f"Generating 2D {split_name}"):
        alpha = float(rng.uniform(alpha_min, alpha_max))

        u0 = make_random_initial_condition_2d(
            x_grid=x_grid,
            y_grid=y_grid,
            rng=rng,
        )

        u_final = solve_heat_equation_2d(
            u0=u0,
            alpha=alpha,
            total_time=total_time,
            dx=dx,
            dt=dt,
        )

        u0_all[idx] = u0
        alpha_all[idx, 0] = alpha
        u_final_all[idx] = u_final

    return {
        "x": x.astype(np.float32),
        "y": y.astype(np.float32),
        "u0": u0_all,
        "alpha": alpha_all,
        "u_final": u_final_all,
    }


def save_split(
    split_data: dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """
    Save dataset split as compressed NPZ file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **split_data)


def generate_data(dim: int) -> None:
    """
    Generate train/val/test/OOD data for selected dimension.
    """
    if dim not in {1, 2}:
        raise ValueError(f"Unsupported dimension: {dim}. Expected 1 or 2.")

    config = load_config(dim)
    ensure_project_dirs(dim)

    problem_cfg = config["problem"]
    data_cfg = config["data"]
    alpha_cfg = config["alpha"]
    solver_cfg = config["solver"]
    training_cfg = config["training"]

    data_dir = get_data_dir(dim)

    seed = int(training_cfg["seed"])
    rng = np.random.default_rng(seed)

    alpha_train_range = (
        float(alpha_cfg["train_min"]),
        float(alpha_cfg["train_max"]),
    )

    alpha_ood_range = (
        float(alpha_cfg["ood_min"]),
        float(alpha_cfg["ood_max"]),
    )

    max_alpha = max(alpha_train_range[1], alpha_ood_range[1])

    total_time = float(problem_cfg["total_time"])
    length = float(problem_cfg["length"])
    stability_factor = float(solver_cfg["stability_factor"])

    print("\nDataset generation")
    print("==================")
    print(f"Dimension: {dim}D")
    print(f"Output directory: {data_dir}")
    print(f"Train alpha range: {alpha_train_range}")
    print(f"OOD alpha range: {alpha_ood_range}")

    if dim == 1:
        nx = int(problem_cfg["nx"])

        x, dx = make_1d_grid(
            nx=nx,
            length=length,
        )

        dt = compute_stable_dt(
            dx=dx,
            max_alpha=max_alpha,
            stability_factor=stability_factor,
        )

        print(f"Grid: nx={nx}")
        print(f"dx={dx:.6e}")
        print(f"dt={dt:.6e}")

        split_generator: Callable[..., dict[str, np.ndarray]] = generate_split_1d

        common_args = {
            "x": x,
            "total_time": total_time,
            "dx": dx,
            "dt": dt,
            "rng": rng,
        }

    else:
        nx = int(problem_cfg["nx"])
        ny = int(problem_cfg["ny"])

        x, y, x_grid, y_grid, dx = make_2d_grid(
            nx=nx,
            ny=ny,
            length=length,
        )

        dt = compute_stable_dt(
            dx=dx,
            max_alpha=max_alpha,
            stability_factor=stability_factor,
        )

        print(f"Grid: ny={ny}, nx={nx}")
        print(f"dx={dx:.6e}")
        print(f"dt={dt:.6e}")

        split_generator = generate_split_2d

        common_args = {
            "x": x,
            "y": y,
            "x_grid": x_grid,
            "y_grid": y_grid,
            "total_time": total_time,
            "dx": dx,
            "dt": dt,
            "rng": rng,
        }

    split_specs = [
        ("train", int(data_cfg["train_samples"]), alpha_train_range),
        ("val", int(data_cfg["val_samples"]), alpha_train_range),
        ("test", int(data_cfg["test_samples"]), alpha_train_range),
        ("ood_test", int(data_cfg["ood_samples"]), alpha_ood_range),
    ]

    for split_name, n_samples, alpha_range in split_specs:
        split_data = split_generator(
            n_samples=n_samples,
            alpha_range=alpha_range,
            split_name=split_name,
            **common_args,
        )

        output_path = data_dir / f"{split_name}.npz"
        save_split(split_data, output_path)

        print(f"Saved {split_name}: {output_path}")

    print("\nData generation complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate 1D or 2D heat-equation datasets."
    )

    parser.add_argument(
        "--dim",
        type=int,
        choices=[1, 2],
        required=True,
        help="Problem dimension: 1 or 2.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generate_data(dim=args.dim)


if __name__ == "__main__":
    main()