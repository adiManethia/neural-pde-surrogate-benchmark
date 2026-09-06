import torch
import torch.nn as nn


class HeatCNN1D(nn.Module):
    """
    CNN1D surrogate for 1D heat equation.

    Input:
        x shape = (batch, 2, nx)

    Channels:
        channel 0 = u0
        channel 1 = normalized alpha repeated along grid

    Output:
        y shape = (batch, nx)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.network = nn.Sequential(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv1d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        y = y.squeeze(1)

        return y


class HeatCNN2D(nn.Module):
    """
    CNN2D surrogate for 2D heat equation.

    Input:
        x shape = (batch, 2, ny, nx)

    Channels:
        channel 0 = u0
        channel 1 = normalized alpha repeated over grid

    Output:
        y shape = (batch, ny, nx)
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: int = 64,
        output_channels: int = 1,
        kernel_size: int = 5,
    ):
        super().__init__()

        padding = kernel_size // 2

        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.ReLU(),

            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.network(x)
        y = y.squeeze(1)

        return y


def create_cnn_model(
    dim: int,
    hidden_channels: int,
    kernel_size: int,
) -> nn.Module:
    """
    Factory function for CNN1D or CNN2D.
    """
    if dim == 1:
        return HeatCNN1D(
            in_channels=2,
            hidden_channels=hidden_channels,
            output_channels=1,
            kernel_size=kernel_size,
        )

    if dim == 2:
        return HeatCNN2D(
            in_channels=2,
            hidden_channels=hidden_channels,
            output_channels=1,
            kernel_size=kernel_size,
        )

    raise ValueError(f"Unsupported dim={dim}. Expected 1 or 2.")