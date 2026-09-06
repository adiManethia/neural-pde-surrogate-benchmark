import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neural_pde.config import (
    load_config,
    ensure_project_dirs,
    get_checkpoint_dir,
    get_result_dir,
    get_plot_dir,
)
from neural_pde.datasets import HeatEquationDataset
from neural_pde.metrics import (
    mse_metric,
    mae_metric,
    relative_l2_error,
)
from neural_pde.models.mlp import HeatMLP
from neural_pde.models.cnn import create_cnn_model
from neural_pde.models.fno import create_fno_model
from neural_pde.plotting import (
    plot_loss_curve,
    plot_prediction_examples,
)


def count_parameters(model: nn.Module) -> int:
    """
    Count trainable model parameters.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )


def create_model(
    dim: int,
    model_name: str,
    config: dict,
    train_dataset: HeatEquationDataset,
) -> nn.Module:
    """
    Create selected model from config.
    """
    model_cfg = config["models"][model_name]

    if model_name == "mlp":
        return HeatMLP(
            input_dim=train_dataset.input_dim,
            output_dim=train_dataset.output_dim,
            hidden_dim=int(model_cfg["hidden_dim"]),
            num_hidden_layers=int(model_cfg["num_hidden_layers"]),
        )

    if model_name == "cnn":
        return create_cnn_model(
            dim=dim,
            hidden_channels=int(model_cfg["hidden_channels"]),
            kernel_size=int(model_cfg["kernel_size"]),
        )

    if model_name == "fno":
        return create_fno_model(
            dim=dim,
            config=model_cfg,
        )

    raise ValueError(
        f"Unsupported model_name={model_name}. "
        "Expected one of: mlp, cnn, fno."
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """
    Train model for one epoch.
    """
    model.train()

    total_loss = 0.0
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """
    Evaluate model.
    """
    model.eval()

    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_rel_l2 = 0.0
    n_batches = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)

        y_pred = model(x_batch)
        loss = loss_fn(y_pred, y_batch)

        total_loss += loss.item()
        total_mse += mse_metric(y_batch, y_pred)
        total_mae += mae_metric(y_batch, y_pred)
        total_rel_l2 += relative_l2_error(y_batch, y_pred)
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "mse": total_mse / n_batches,
        "mae": total_mae / n_batches,
        "relative_l2": total_rel_l2 / n_batches,
    }


def save_metrics(
    output_path: Path,
    dim: int,
    model_name: str,
    parameter_count: int,
    best_val_loss: float,
    test_metrics: dict[str, float],
    ood_metrics: dict[str, float],
) -> None:
    """
    Save metrics to text file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"Model: {model_name}\n")
        f.write(f"Dimension: {dim}D\n")
        f.write(f"Trainable parameters: {parameter_count}\n")
        f.write(f"Best validation loss: {best_val_loss}\n")
        f.write("\nIn-distribution test:\n")
        f.write(str(test_metrics))
        f.write("\n\nOOD test:\n")
        f.write(str(ood_metrics))
        f.write("\n")


def run_training(
    dim: int,
    model_name: str,
) -> None:
    """
    Train selected model for selected dimension.
    """
    if dim not in {1, 2}:
        raise ValueError(f"Unsupported dim={dim}. Expected 1 or 2.")

    if model_name not in {"mlp", "cnn", "fno"}:
        raise ValueError(
            f"Unsupported model_name={model_name}. "
            "Expected one of: mlp, cnn, fno."
        )

    config = load_config(dim)
    ensure_project_dirs(dim)

    if model_name not in config["models"]:
        raise ValueError(
            f"Model '{model_name}' is not defined in config for dim={dim}."
        )

    alpha_cfg = config["alpha"]
    training_cfg = config["training"]

    alpha_min = float(alpha_cfg["train_min"])
    alpha_max = float(alpha_cfg["train_max"])

    seed = int(training_cfg["seed"])
    batch_size = int(training_cfg["batch_size"])
    learning_rate = float(training_cfg["learning_rate"])
    n_epochs = int(training_cfg["epochs"])

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset = HeatEquationDataset(
        dim=dim,
        split="train",
        model_name=model_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    val_dataset = HeatEquationDataset(
        dim=dim,
        split="val",
        model_name=model_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    test_dataset = HeatEquationDataset(
        dim=dim,
        split="test",
        model_name=model_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    ood_dataset = HeatEquationDataset(
        dim=dim,
        split="ood_test",
        model_name=model_name,
        alpha_min=alpha_min,
        alpha_max=alpha_max,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    ood_loader = DataLoader(
        ood_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    model = create_model(
        dim=dim,
        model_name=model_name,
        config=config,
        train_dataset=train_dataset,
    ).to(device)

    parameter_count = count_parameters(model)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    checkpoint_dir = get_checkpoint_dir(dim)
    result_dir = get_result_dir(dim)
    plot_dir = get_plot_dir(dim)

    best_model_path = checkpoint_dir / f"{model_name}_best.pt"
    metrics_path = result_dir / f"{model_name}_metrics.txt"

    print("\nTraining")
    print("========")
    print(f"Dimension: {dim}D")
    print(f"Model: {model_name}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {n_epochs}")
    print(f"Trainable parameters: {parameter_count:,}")
    print(f"Best checkpoint: {best_model_path}")

    train_losses = []
    val_losses = []

    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            loss_fn=loss_fn,
            device=device,
        )

        val_loss = val_metrics["loss"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "dim": dim,
                    "model_name": model_name,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "parameter_count": parameter_count,
                    "config": config,
                },
                best_model_path,
            )

        if epoch == 1 or epoch % 5 == 0:
            print(
                f"Epoch {epoch:03d}/{n_epochs} | "
                f"train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f} | "
                f"val_rel_l2={val_metrics['relative_l2']:.6f}"
            )

    print("\nTraining complete.")
    print(f"Best validation loss: {best_val_loss:.8f}")

    checkpoint = torch.load(
        best_model_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics = evaluate(
        model=model,
        dataloader=test_loader,
        loss_fn=loss_fn,
        device=device,
    )

    ood_metrics = evaluate(
        model=model,
        dataloader=ood_loader,
        loss_fn=loss_fn,
        device=device,
    )

    print("\nFinal evaluation")
    print("----------------")
    print("In-distribution test:")
    print(test_metrics)
    print("\nOOD test:")
    print(ood_metrics)

    save_metrics(
        output_path=metrics_path,
        dim=dim,
        model_name=model_name,
        parameter_count=parameter_count,
        best_val_loss=best_val_loss,
        test_metrics=test_metrics,
        ood_metrics=ood_metrics,
    )

    plot_loss_curve(
        train_losses=train_losses,
        val_losses=val_losses,
        output_path=plot_dir / f"{model_name}_loss_curve.png",
        title=f"{dim}D {model_name.upper()} training curve",
    )

    plot_prediction_examples(
        model=model,
        dataset=test_dataset,
        output_path=plot_dir / f"{model_name}_test_predictions.png",
        device=device,
        title=f"{dim}D {model_name.upper()} predictions on test data",
    )

    plot_prediction_examples(
        model=model,
        dataset=ood_dataset,
        output_path=plot_dir / f"{model_name}_ood_predictions.png",
        device=device,
        title=f"{dim}D {model_name.upper()} predictions on OOD data",
    )

    print("\nSaved outputs:")
    print(f"- {best_model_path}")
    print(f"- {metrics_path}")
    print(f"- {plot_dir / f'{model_name}_loss_curve.png'}")
    print(f"- {plot_dir / f'{model_name}_test_predictions.png'}")
    print(f"- {plot_dir / f'{model_name}_ood_predictions.png'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train neural surrogate model for 1D or 2D heat equation."
    )

    parser.add_argument(
        "--dim",
        type=int,
        choices=[1, 2],
        required=True,
        help="Problem dimension: 1 or 2.",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn", "fno"],
        required=True,
        help="Model type: mlp, cnn, or fno.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_training(
        dim=args.dim,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()