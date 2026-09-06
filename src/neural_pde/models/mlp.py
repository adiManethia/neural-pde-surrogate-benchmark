import torch
import torch.nn as nn


class HeatMLP(nn.Module):
    """
    Fully connected baseline for 1D or 2D heat-equation surrogate modelling.

    The field is flattened before entering the MLP.

    Example:
        1D:
            input_dim = 128 + 1
            output_dim = 128

        2D:
            input_dim = 32*32 + 1
            output_dim = 32*32
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_hidden_layers: int = 3,
    ):
        super().__init__()

        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be >= 1.")

        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)