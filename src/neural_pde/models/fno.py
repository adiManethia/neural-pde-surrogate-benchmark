import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1D spectral convolution
# ============================================================

class SpectralConv1d(nn.Module):
    """
    1D spectral convolution layer.

    Input:
        x shape = (batch, in_channels, nx)

    Output:
        out shape = (batch, out_channels, nx)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes

        scale = 1.0 / (in_channels * out_channels)

        self.weights = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                n_modes,
                dtype=torch.cfloat,
            )
        )

    def complex_multiply(
        self,
        x_ft: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_ft:
            (batch, in_channels, modes)

        weights:
            (in_channels, out_channels, modes)

        output:
            (batch, out_channels, modes)
        """
        return torch.einsum("bim,iom->bom", x_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        nx = x.shape[-1]

        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x_ft.shape[-1],
            device=x.device,
            dtype=torch.cfloat,
        )

        modes = min(self.n_modes, x_ft.shape[-1])

        out_ft[:, :, :modes] = self.complex_multiply(
            x_ft[:, :, :modes],
            self.weights[:, :, :modes],
        )

        out = torch.fft.irfft(
            out_ft,
            n=nx,
            dim=-1,
        )

        return out


class FNOBlock1d(nn.Module):
    """
    One 1D FNO block:
        spectral convolution + pointwise convolution
    """

    def __init__(
        self,
        width: int,
        n_modes: int,
    ):
        super().__init__()

        self.spectral_conv = SpectralConv1d(
            in_channels=width,
            out_channels=width,
            n_modes=n_modes,
        )

        self.pointwise_conv = nn.Conv1d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral_conv(x) + self.pointwise_conv(x)
        x = F.gelu(x)

        return x


class HeatFNO1D(nn.Module):
    """
    Simplified FNO-style model for 1D heat equation.

    Input:
        x shape = (batch, 3, nx)

    Channels:
        channel 0 = u0
        channel 1 = normalized alpha
        channel 2 = x coordinate

    Output:
        y shape = (batch, nx)
    """

    def __init__(
        self,
        in_channels: int = 3,
        width: int = 64,
        n_modes: int = 16,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.lift = nn.Conv1d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                FNOBlock1d(
                    width=width,
                    n_modes=n_modes,
                )
                for _ in range(n_blocks)
            ]
        )

        self.project = nn.Sequential(
            nn.Conv1d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)

        for block in self.blocks:
            x = block(x)

        y = self.project(x)
        y = y.squeeze(1)

        return y


# ============================================================
# 2D spectral convolution
# ============================================================

class SpectralConv2d(nn.Module):
    """
    2D spectral convolution layer.

    Input:
        x shape = (batch, in_channels, ny, nx)

    Output:
        out shape = (batch, out_channels, ny, nx)

    Note:
        rfft2 keeps only non-negative frequencies in the last dimension.
        We learn weights for low positive and low negative y-frequency modes.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        n_modes_y: int,
        n_modes_x: int,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes_y = n_modes_y
        self.n_modes_x = n_modes_x

        scale = 1.0 / (in_channels * out_channels)

        self.weights_pos = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                n_modes_y,
                n_modes_x,
                dtype=torch.cfloat,
            )
        )

        self.weights_neg = nn.Parameter(
            scale
            * torch.randn(
                in_channels,
                out_channels,
                n_modes_y,
                n_modes_x,
                dtype=torch.cfloat,
            )
        )

    def complex_multiply_2d(
        self,
        x_ft: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        x_ft:
            (batch, in_channels, modes_y, modes_x)

        weights:
            (in_channels, out_channels, modes_y, modes_x)

        output:
            (batch, out_channels, modes_y, modes_x)
        """
        return torch.einsum("bixy,ioxy->boxy", x_ft, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        ny = x.shape[-2]
        nx = x.shape[-1]

        x_ft = torch.fft.rfft2(
            x,
            dim=(-2, -1),
        )

        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            ny,
            nx // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )

        modes_y = min(self.n_modes_y, ny)
        modes_x = min(self.n_modes_x, nx // 2 + 1)

        # Low positive y-frequency modes
        out_ft[:, :, :modes_y, :modes_x] = self.complex_multiply_2d(
            x_ft[:, :, :modes_y, :modes_x],
            self.weights_pos[:, :, :modes_y, :modes_x],
        )

        # Low negative y-frequency modes
        out_ft[:, :, -modes_y:, :modes_x] = self.complex_multiply_2d(
            x_ft[:, :, -modes_y:, :modes_x],
            self.weights_neg[:, :, :modes_y, :modes_x],
        )

        out = torch.fft.irfft2(
            out_ft,
            s=(ny, nx),
            dim=(-2, -1),
        )

        return out


class FNOBlock2d(nn.Module):
    """
    One 2D FNO block:
        spectral convolution + pointwise convolution
    """

    def __init__(
        self,
        width: int,
        n_modes_y: int,
        n_modes_x: int,
    ):
        super().__init__()

        self.spectral_conv = SpectralConv2d(
            in_channels=width,
            out_channels=width,
            n_modes_y=n_modes_y,
            n_modes_x=n_modes_x,
        )

        self.pointwise_conv = nn.Conv2d(
            in_channels=width,
            out_channels=width,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spectral_conv(x) + self.pointwise_conv(x)
        x = F.gelu(x)

        return x


class HeatFNO2D(nn.Module):
    """
    Simplified FNO-style model for 2D heat equation.

    Input:
        x shape = (batch, 4, ny, nx)

    Channels:
        channel 0 = u0
        channel 1 = normalized alpha
        channel 2 = x coordinate
        channel 3 = y coordinate

    Output:
        y shape = (batch, ny, nx)
    """

    def __init__(
        self,
        in_channels: int = 4,
        width: int = 32,
        n_modes_y: int = 12,
        n_modes_x: int = 12,
        n_blocks: int = 4,
    ):
        super().__init__()

        self.lift = nn.Conv2d(
            in_channels=in_channels,
            out_channels=width,
            kernel_size=1,
        )

        self.blocks = nn.ModuleList(
            [
                FNOBlock2d(
                    width=width,
                    n_modes_y=n_modes_y,
                    n_modes_x=n_modes_x,
                )
                for _ in range(n_blocks)
            ]
        )

        self.project = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lift(x)

        for block in self.blocks:
            x = block(x)

        y = self.project(x)
        y = y.squeeze(1)

        return y


def create_fno_model(
    dim: int,
    config: dict,
) -> nn.Module:
    """
    Factory function for FNO1D or FNO2D.
    """
    if dim == 1:
        return HeatFNO1D(
            in_channels=3,
            width=int(config["width"]),
            n_modes=int(config["n_modes"]),
            n_blocks=int(config["n_blocks"]),
        )

    if dim == 2:
        return HeatFNO2D(
            in_channels=4,
            width=int(config["width"]),
            n_modes_y=int(config["n_modes_y"]),
            n_modes_x=int(config["n_modes_x"]),
            n_blocks=int(config["n_blocks"]),
        )

    raise ValueError(f"Unsupported dim={dim}. Expected 1 or 2.")