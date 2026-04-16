"""Loss utilities used across models.

This module provides several reusable loss functions and metrics used in the
project. Strings and comments are English-only and the code is kept simple
and explicit for readability.
"""

from typing import Union

import torch
import torch.nn as nn


def compute_vpt(pred: torch.Tensor, true: torch.Tensor, threshold: float, dt: float = 1.0, mode: str = 'mean') -> Union[float, torch.Tensor]:
    """Compute the "valid prediction time" (VPT) for a batch of trajectories.

    VPT is defined as the time of the first timestep where the per-step RMSE
    exceeds a given threshold. If a trajectory never exceeds the threshold,
    its VPT is set to the trajectory length (i.e. maximum possible).

    Args:
        pred: Predicted tensor of shape (B, T, ...).
        true: Ground-truth tensor with the same shape as ``pred``.
        threshold: Scalar threshold for RMSE to define failure.
        dt: Time step size. The VPT returned is in the same time units.
        mode: If 'mean', return the mean VPT (scalar). If 'all', return the
            per-sample VPT tensor of shape (B,).

    Returns:
        Mean VPT (float) when ``mode=='mean'`` or per-sample VPT tensor when
        ``mode=='all'``.
    """

    # Compute per-timestep root-mean-square error across the trailing dims
    # Resulting shape: (B, T)
    error = torch.sqrt(torch.mean((pred - true) ** 2, dim=-1))

    # Boolean mask where error exceeds the threshold
    exceed = error >= threshold  # shape: (B, T)

    # Convert to int and find the first True index along time dim.
    # If no True exists, argmax returns 0; we handle that via any_exceed.
    exceed_int = exceed.to(torch.int64)
    first_idx = torch.argmax(exceed_int, dim=1)  # shape: (B,)

    any_exceed = torch.any(exceed, dim=1)
    seq_len = error.shape[1]

    # If no exceed, set index to seq_len (i.e., failure at the end)
    first_idx = torch.where(any_exceed, first_idx, torch.full_like(first_idx, seq_len))

    vpt_time = first_idx.to(torch.float32) * float(dt)

    if mode == 'mean':
        return vpt_time.mean().item()
    return vpt_time


class RMSELoss(nn.Module):
    """Root-mean-square error computed per time step.

    The forward method returns an array of RMSE values along the time
    dimension (shape depends on input shapes). This keeps compatibility with
    existing usage in the codebase.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute RMSE across batch and spatial dims, returning per-time RMSE.

        Expects inputs with shape (B, T, C) or similar; it reduces over batch
        and channel dims and returns a tensor indexed by the remaining dims.
        """
        mse = torch.mean((pred - target) ** 2, dim=(0, 2))
        rmse = torch.sqrt(mse)
        return rmse


class HsLoss1D:
    """Sobolev-type loss for 1D temporal signals.

    This class implements an H^s-like loss in the temporal frequency domain.
    It is intentionally a lightweight callable (not an ``nn.Module``) to
    maintain backward compatibility with existing code that instantiates and
    calls it directly.
    """

    def __init__(self, rel: bool = False, p: int = 2, k: int = 1, a=None, group: bool = False,
                 size_average: bool = True, reduction: bool = True):
        self.r = rel
        self.p = p
        self.k = k
        self.group = group
        self.reduction = reduction
        self.size_average = size_average

        if a is None:
            a = [1.0] * max(0, k)
        self.a = a

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute relative Lp-like norm between x and y per sample.

        Returns either a scalar (averaged) or per-sample vector depending on
        the ``reduction`` and ``size_average`` flags.
        """
        batch = x.size(0)
        diff_norms = torch.norm(x.reshape(batch, -1) - y.reshape(batch, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(batch, -1), p=self.p, dim=1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms) if self.r else torch.mean(diff_norms)
            return torch.sum(diff_norms / y_norms) if self.r else torch.sum(diff_norms)
        return diff_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor, a=None) -> torch.Tensor:
        """Compute the H^s-like loss between tensors x and y.

        Args:
            x, y: tensors with shape (B, T, C) or (B, T)
            a: optional weights for derivative terms
        """
        time_steps = x.size(1)
        a = self.a if a is None else a

        # Treat channel dims uniformly
        x = x.view(x.shape[0], time_steps, -1)
        y = y.view(y.shape[0], time_steps, -1)

        # Create 1D frequency vector along time dimension
        device = x.device
        k_t = torch.cat((torch.arange(0, time_steps // 2, device=device),
                         torch.arange(-time_steps // 2, 0, device=device)), dim=0).abs().float()
        k_t = k_t.view(1, time_steps, 1)

        # FFT along time dimension
        x_hat = torch.fft.fftn(x, dim=[1], norm='ortho')
        y_hat = torch.fft.fftn(y, dim=[1], norm='ortho')

        if not self.group:
            weight = torch.ones_like(k_t)
            if self.k >= 1:
                weight = weight + (a[0] ** 2) * (k_t ** 2)
            if self.k >= 2:
                weight = weight + (a[1] ** 2) * (k_t ** 4)
            weight = torch.sqrt(weight)
            return self._rel(x_hat * weight, y_hat * weight)

        loss = self._rel(x_hat, y_hat)
        if self.k >= 1:
            weight = a[0] * k_t
            loss = loss + self._rel(x_hat * weight, y_hat * weight)
        if self.k >= 2:
            weight = a[1] * (k_t ** 2)
            loss = loss + self._rel(x_hat * weight, y_hat * weight)
        return loss / (self.k + 1)


class HsLoss:
    """Sobolev-space loss for spatiotemporal data.

    This loss operates in the frequency domain over both time and space
    dimensions and supports multiple derivative orders.
    """

    def __init__(self, rel: bool = False, p: int = 2, k: int = 1, a=None, group: bool = False,
                 size_average: bool = True, reduction: bool = True):
        self.r = rel
        assert p > 0 and k >= 0
        self.p = p
        self.k = k
        self.group = group
        self.size_average = size_average
        self.reduction = reduction

        if a is None:
            a = [1.0] * max(0, k)
        self.a = a

    def _rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        diff_norms = torch.norm(x.reshape(batch, -1) - y.reshape(batch, -1), p=self.p, dim=1)
        y_norms = torch.norm(y.reshape(batch, -1), p=self.p, dim=1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms) if self.r else torch.mean(diff_norms)
            return torch.sum(diff_norms / y_norms) if self.r else torch.sum(diff_norms)
        return diff_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the H^s loss for tensors shaped (B, T, C).

        The function computes FFT over time and space, applies derivative
        weights, and aggregates the relative errors.
        """
        assert x.shape == y.shape, "x and y must have the same shape"
        B, T, C = x.shape
        device = x.device

        # Frequency coordinates along time and space
        k_t = torch.cat((torch.arange(0, T // 2, device=device),
                         torch.arange(-T // 2, 0, device=device)), dim=0).abs().view(1, T, 1)
        k_x = torch.cat((torch.arange(0, C // 2, device=device),
                         torch.arange(-C // 2, 0, device=device)), dim=0).abs().view(1, 1, C)

        k_t2 = k_t ** 2
        k_x2 = k_x ** 2

        # 2D FFT along time and space dims
        x_hat = torch.fft.fftn(x, dim=[1, 2], norm='ortho')
        y_hat = torch.fft.fftn(y, dim=[1, 2], norm='ortho')

        if not self.group:
            weight = torch.ones_like(x_hat.real)
            if self.k >= 1:
                weight = weight + (self.a[0] ** 2) * (k_t2 + k_x2)
            if self.k >= 2:
                weight = weight + (self.a[1] ** 2) * (k_t2 ** 2 + 2 * k_t2 * k_x2 + k_x2 ** 2)
            weight = torch.sqrt(weight)
            return self._rel(x_hat * weight, y_hat * weight)

        loss = self._rel(x_hat, y_hat)
        if self.k >= 1:
            weight = self.a[0] * torch.sqrt(k_t2 + k_x2)
            loss = loss + self._rel(x_hat * weight, y_hat * weight)
        if self.k >= 2:
            weight = self.a[1] * torch.sqrt(k_t2 ** 2 + 2 * k_t2 * k_x2 + k_x2 ** 2)
            loss = loss + self._rel(x_hat * weight, y_hat * weight)
        return loss / (self.k + 1)