import argparse
import ast
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from neural_pde.config import (
    get_result_dir,
    get_plot_dir,
    ensure_project_dirs,
)


def read_metric_file(path: Path) -> tuple[dict, dict, int | None]:
    """
    Read metrics saved by neural_pde.train.

    Expected file contains:
        Trainable parameters: ...
        In-distribution test:
        {...}

        OOD test:
        {...}
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    lines = path.read_text().splitlines()

    parameter_count = None
    dict_lines = []

    for line in lines:
        line = line.strip()

        if line.startswith("Trainable parameters:"):
            value = line.split(":", 1)[1].strip()
            try:
                parameter_count = int(value)
            except ValueError:
                parameter_count = None

        if line.startswith("{") and line.endswith("}"):
            dict_lines.append(line)

    if len(dict_lines) < 2:
        raise ValueError(
            f"Could not find two metric dictionaries in {path}. "
            f"Found {len(dict_lines)}."
        )

    test_metrics = ast.literal_eval(dict_lines[0])
    ood_metrics = ast.literal_eval(dict_lines[1])

    return test_metrics, ood_metrics, parameter_count


def build_comparison_table(dim: int) -> pd.DataFrame:
    """
    Build comparison table for selected dimension.
    """
    result_dir = get_result_dir(dim)

    rows = []

    for model_name in ["mlp", "cnn", "fno"]:
        metric_path = result_dir / f"{model_name}_metrics.txt"

        if not metric_path.exists():
            print(f"Skipping missing metrics file: {metric_path}")
            continue

        test_metrics, ood_metrics, parameter_count = read_metric_file(metric_path)

        for split_name, metrics in [
            ("test", test_metrics),
            ("ood_test", ood_metrics),
        ]:
            relative_l2 = float(metrics["relative_l2"])

            rows.append(
                {
                    "dim": dim,
                    "model": model_name.upper(),
                    "split": split_name,
                    "mse": float(metrics["mse"]),
                    "mae": float(metrics["mae"]),
                    "relative_l2": relative_l2,
                    "relative_l2_percent": 100.0 * relative_l2,
                    "parameters": parameter_count,
                }
            )

    if not rows:
        raise RuntimeError(
            f"No metrics found for dim={dim}. "
            f"Run training first, for example: python -m neural_pde.train --dim {dim} --model cnn"
        )

    return pd.DataFrame(rows)


def plot_grouped_bar(
    df: pd.DataFrame,
    value_column: str,
    ylabel: str,
    title: str,
    output_path: Path,
    logy: bool = False,
) -> None:
    """
    Create grouped bar plot by model and split.
    """
    pivot = df.pivot(
        index="model",
        columns="split",
        values=value_column,
    )

    preferred_order = ["MLP", "CNN", "FNO"]
    available_order = [m for m in preferred_order if m in pivot.index]
    pivot = pivot.loc[available_order]

    ax = pivot.plot(
        kind="bar",
        figsize=(8, 5),
        logy=logy,
    )

    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(title="Split")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.xticks(rotation=0)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def print_summary(df: pd.DataFrame, dim: int) -> None:
    """
    Print readable comparison summary.
    """
    print(f"\n{dim}D model comparison")
    print("=" * 24)
    print(df.to_string(index=False))

    print("\nRelative L2 error (%)")
    print("=====================")

    pivot = df.pivot(
        index="model",
        columns="split",
        values="relative_l2_percent",
    )

    preferred_order = ["MLP", "CNN", "FNO"]
    available_order = [m for m in preferred_order if m in pivot.index]
    pivot = pivot.loc[available_order]

    print(pivot.round(4).to_string())

    test_df = df[df["split"] == "test"].sort_values("relative_l2")
    ood_df = df[df["split"] == "ood_test"].sort_values("relative_l2")

    print("\nBest models")
    print("===========")
    print(f"Best test model: {test_df.iloc[0]['model']}")
    print(f"Best OOD model:  {ood_df.iloc[0]['model']}")


def compare_models(dim: int) -> None:
    """
    Compare all available models for selected dimension.
    """
    ensure_project_dirs(dim)

    result_dir = get_result_dir(dim)
    plot_dir = get_plot_dir(dim)

    df = build_comparison_table(dim)

    output_csv = result_dir / "model_comparison.csv"
    df.to_csv(output_csv, index=False)

    plot_grouped_bar(
        df=df,
        value_column="relative_l2_percent",
        ylabel="Relative L2 error (%)",
        title=f"{dim}D model comparison: relative L2 error",
        output_path=plot_dir / "model_comparison_relative_l2.png",
    )

    plot_grouped_bar(
        df=df,
        value_column="mse",
        ylabel="MSE, log scale",
        title=f"{dim}D model comparison: MSE",
        output_path=plot_dir / "model_comparison_mse_log.png",
        logy=True,
    )

    plot_grouped_bar(
        df=df,
        value_column="mae",
        ylabel="MAE",
        title=f"{dim}D model comparison: MAE",
        output_path=plot_dir / "model_comparison_mae.png",
    )

    print_summary(df, dim)

    print("\nSaved outputs:")
    print(f"- {output_csv}")
    print(f"- {plot_dir / 'model_comparison_relative_l2.png'}")
    print(f"- {plot_dir / 'model_comparison_mse_log.png'}")
    print(f"- {plot_dir / 'model_comparison_mae.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare trained surrogate models for 1D or 2D heat equation."
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
    compare_models(dim=args.dim)


if __name__ == "__main__":
    main()