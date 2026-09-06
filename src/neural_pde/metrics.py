import torch


def mse_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Mean squared error.
    Works for both 1D and 2D fields.
    """
    return torch.mean((y_true - y_pred) ** 2).item()


def mae_metric(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Mean absolute error.
    Works for both 1D and 2D fields.
    """
    return torch.mean(torch.abs(y_true - y_pred)).item()


def relative_l2_error(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """
    Relative L2 error.

    Formula:
        ||y_true - y_pred||_2 / ||y_true||_2

    Works for:
        1D: shape (batch, nx)
        2D: shape (batch, ny, nx)
    """
    numerator = torch.linalg.norm(y_true - y_pred)
    denominator = torch.linalg.norm(y_true) + 1e-8

    return (numerator / denominator).item()