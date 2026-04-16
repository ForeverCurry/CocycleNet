"""Simple static normalizer storing mean and std as buffers.

The buffers are registered with the module using ``register_buffer`` so
they move with the module when calling ``.to(device)`` and are not
considered model parameters (no gradients).
"""

from typing import Union

import torch
import torch.nn as nn


class StaticNormalizer(nn.Module):
    """A minimal mean/std normalizer.

    Args:
        mu: Mean value(s). Can be a scalar or a sequence matching the
            feature dimensions.
        std: Standard deviation(s). Same shape requirements as ``mu``.
    """

    def __init__(self, mu: Union[float, list], std: Union[float, list]):
        super().__init__()
        # store as tensors so they move with the module (CPU/GPU)
        self.register_buffer('mu', torch.tensor(mu, dtype=torch.float32))
        self.register_buffer('std', torch.tensor(std, dtype=torch.float32))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize input using stored mean/std.

        Expected input shape: [B, T, D].
        """
        return (x - self.mu) / self.std

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Inverse transform: denormalize tensor."""
        return x * self.std + self.mu