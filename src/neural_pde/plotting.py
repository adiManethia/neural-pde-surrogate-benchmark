from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_loss_curve(
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
    title: str,
) -> None:
    """
    Save train/validation loss curve.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="Train loss")
    plt.plot(val_losses, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def plot_prediction_examples(
    model: torch.nn.Module,
    dataset,
    output_path: Path,
    device: torch.device,
    title: str,
    n_examples: int = 4,
) -> None:
    """
    Plot prediction examples for 1D or 2D datasets.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()

    if dataset.dim == 1:
        _plot_1d_examples(
            model=model,
            dataset=dataset,
            output_path=output_path,
            device=device,
            title=title,
            n_examples=n_examples,
        )
    else:
        _plot_2d_examples(
            model=model,
            dataset=dataset,
            output_path=output_path,
            device=device,
            title=title,
            n_examples=min(n_examples, 3),
        )


@torch.no_grad()
def _plot_1d_examples(
    model: torch.nn.Module,
    dataset,
    output_path: Path,
    device: torch.device,
    title: str,
    n_examples: int = 4,
) -> None:
    plt.figure(figsize=(9, 6))

    for i in range(n_examples):
        x_input, y_true = dataset[i]

        x_batch = x_input.unsqueeze(0).to(device)
        y_pred = model(x_batch).squeeze(0).cpu()

        if dataset.model_name == "mlp":
            y_pred = y_pred.reshape(dataset.field_shape)
            y_true_plot = y_true.reshape(dataset.field_shape)
        else:
            y_true_plot = y_true

        u0 = dataset.u0[i]
        alpha = dataset.alpha[i, 0]

        plt.subplot(2, 2, i + 1)
        plt.plot(u0, label="Initial u0", linestyle="--")
        plt.plot(y_true_plot.numpy(), label="True final")
        plt.plot(y_pred.numpy(), label="Predicted final")
        plt.title(f"alpha={alpha:.4f}")
        plt.xlabel("Grid index")
        plt.ylabel("u")
        plt.legend(fontsize=8)

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


@torch.no_grad()
def _plot_2d_examples(
    model: torch.nn.Module,
    dataset,
    output_path: Path,
    device: torch.device,
    title: str,
    n_examples: int = 3,
) -> None:
    fig, axes = plt.subplots(
        n_examples,
        3,
        figsize=(10, 3.2 * n_examples),
    )

    if n_examples == 1:
        axes = axes.reshape(1, 3)

    for i in range(n_examples):
        x_input, y_true = dataset[i]

        x_batch = x_input.unsqueeze(0).to(device)
        y_pred = model(x_batch).squeeze(0).cpu()

        if dataset.model_name == "mlp":
            y_pred = y_pred.reshape(dataset.field_shape)
            y_true_plot = y_true.reshape(dataset.field_shape)
        else:
            y_true_plot = y_true

        u0 = dataset.u0[i]
        alpha = dataset.alpha[i, 0]

        fields = [
            (u0, "Initial"),
            (y_true_plot.numpy(), "True final"),
            (y_pred.numpy(), "Predicted final"),
        ]

        for col, (field, subtitle) in enumerate(fields):
            ax = axes[i, col]
            im = ax.imshow(field, origin="lower")
            ax.set_title(f"{subtitle}\nalpha={alpha:.4f}")
            ax.set_xticks([])
            ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()