from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from neural_pde.config import get_data_dir


class HeatEquationDataset(Dataset):
    """
    Unified dataset class for 1D and 2D heat-equation surrogate modelling.

    Supported dimensions:
        dim = 1
        dim = 2

    Supported model types:
        mlp
        cnn
        fno

    Returns:
        x_input: model-specific input tensor
        y_target: model-specific target tensor
    """

    def __init__(
        self,
        dim: int,
        split: str,
        model_name: str,
        alpha_min: float,
        alpha_max: float,
    ):
        if dim not in {1, 2}:
            raise ValueError(f"Unsupported dim={dim}. Expected 1 or 2.")

        if model_name not in {"mlp", "cnn", "fno"}:
            raise ValueError(
                f"Unsupported model_name={model_name}. "
                "Expected one of: mlp, cnn, fno."
            )

        self.dim = dim
        self.split = split
        self.model_name = model_name
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        data_path = get_data_dir(dim) / f"{split}.npz"

        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_path}. "
                f"Run: python -m neural_pde.generate_data --dim {dim}"
            )

        data = np.load(data_path)

        self.u0 = data["u0"].astype(np.float32)
        self.alpha = data["alpha"].astype(np.float32)
        self.u_final = data["u_final"].astype(np.float32)

        self.alpha_norm = (
            (self.alpha - alpha_min)
            /
            (alpha_max - alpha_min)
        ).astype(np.float32)

        if dim == 1:
            self.x_grid = data["x"].astype(np.float32)
            self.field_shape = (self.u0.shape[1],)

        else:
            self.x_grid = data["x"].astype(np.float32)
            self.y_grid = data["y"].astype(np.float32)
            self.field_shape = (self.u0.shape[1], self.u0.shape[2])

        self.input_dim, self.output_dim = self._get_mlp_dims()

    def _get_mlp_dims(self) -> Tuple[int, int]:
        """
        Return flattened MLP input and output dimensions.
        """
        field_size = int(np.prod(self.field_shape))

        input_dim = field_size + 1
        output_dim = field_size

        return input_dim, output_dim

    def __len__(self) -> int:
        return len(self.u0)

    def _make_mlp_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MLP input:
            flattened u0 + normalized alpha

        MLP target:
            flattened u_final
        """
        u0_i = self.u0[idx].reshape(-1)
        alpha_i = self.alpha_norm[idx]  # shape: (1,)
        y_i = self.u_final[idx].reshape(-1)

        x_i = np.concatenate([u0_i, alpha_i], axis=0)

        return (
            torch.from_numpy(x_i).float(),
            torch.from_numpy(y_i).float(),
        )

    def _make_cnn_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        CNN input:
            1D: (2, nx)
            2D: (2, ny, nx)

        Channels:
            channel 0 = u0
            channel 1 = normalized alpha repeated over grid

        Target:
            1D: (nx,)
            2D: (ny, nx)
        """
        u0_i = self.u0[idx]
        alpha_i = float(self.alpha_norm[idx, 0])
        y_i = self.u_final[idx]

        alpha_channel = np.full_like(u0_i, fill_value=alpha_i)

        x_i = np.stack(
            [u0_i, alpha_channel],
            axis=0,
        )

        return (
            torch.from_numpy(x_i).float(),
            torch.from_numpy(y_i).float(),
        )

    def _make_fno_sample(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        FNO input:
            1D: (3, nx)
                channel 0 = u0
                channel 1 = normalized alpha repeated
                channel 2 = x coordinate

            2D: (4, ny, nx)
                channel 0 = u0
                channel 1 = normalized alpha repeated
                channel 2 = x coordinate grid
                channel 3 = y coordinate grid

        Target:
            1D: (nx,)
            2D: (ny, nx)
        """
        u0_i = self.u0[idx]
        alpha_i = float(self.alpha_norm[idx, 0])
        y_i = self.u_final[idx]

        alpha_channel = np.full_like(u0_i, fill_value=alpha_i)

        if self.dim == 1:
            x_channel = self.x_grid.copy()

            x_i = np.stack(
                [u0_i, alpha_channel, x_channel],
                axis=0,
            )

        else:
            x_grid, y_grid = np.meshgrid(
                self.x_grid,
                self.y_grid,
            )

            x_i = np.stack(
                [
                    u0_i,
                    alpha_channel,
                    x_grid.astype(np.float32),
                    y_grid.astype(np.float32),
                ],
                axis=0,
            )

        return (
            torch.from_numpy(x_i).float(),
            torch.from_numpy(y_i).float(),
        )

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.model_name == "mlp":
            return self._make_mlp_sample(idx)

        if self.model_name == "cnn":
            return self._make_cnn_sample(idx)

        if self.model_name == "fno":
            return self._make_fno_sample(idx)

        raise RuntimeError("Invalid dataset state.")