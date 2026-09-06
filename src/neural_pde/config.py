from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def get_config_path(dim: int) -> Path:
    """
    Return config path based on PDE dimension.
    """
    if dim == 1:
        return PROJECT_ROOT / "configs" / "heat1d.yaml"

    if dim == 2:
        return PROJECT_ROOT / "configs" / "heat2d.yaml"

    raise ValueError(f"Unsupported dimension: {dim}. Expected 1 or 2.")


def load_config(dim: int) -> dict[str, Any]:
    """
    Load YAML config for 1D or 2D heat-equation benchmark.
    """
    config_path = get_config_path(dim)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_problem_name(dim: int) -> str:
    """
    Return problem folder name.
    """
    if dim == 1:
        return "heat1d"

    if dim == 2:
        return "heat2d"

    raise ValueError(f"Unsupported dimension: {dim}. Expected 1 or 2.")


def get_data_dir(dim: int) -> Path:
    return PROJECT_ROOT / "data" / get_problem_name(dim)


def get_result_dir(dim: int) -> Path:
    return PROJECT_ROOT / "results" / get_problem_name(dim)


def get_plot_dir(dim: int) -> Path:
    return PROJECT_ROOT / "plots" / get_problem_name(dim)


def get_checkpoint_dir(dim: int) -> Path:
    return PROJECT_ROOT / "checkpoints" / get_problem_name(dim)


def ensure_project_dirs(dim: int) -> None:
    """
    Create required output directories for selected dimension.
    """
    get_data_dir(dim).mkdir(parents=True, exist_ok=True)
    get_result_dir(dim).mkdir(parents=True, exist_ok=True)
    get_plot_dir(dim).mkdir(parents=True, exist_ok=True)
    get_checkpoint_dir(dim).mkdir(parents=True, exist_ok=True)