"""Multivariate Variational Mode Decomposition (MVMD) module.

This module provides a pure-PyTorch implementation of the Multivariate
Variational Mode Decomposition algorithm, which is a multi-channel
extension of VMD [Dragomiretskiy & Zosso, 2014].

Reference:
    Rehman, N., & Aftab, H. (2019). Multivariate variational mode decomposition.
    IEEE Transactions on Signal Processing, 67(23), 6039-6052.
"""

from typing import Tuple

import torch
import torch.nn as nn


class MVMD(nn.Module):
    """Multivariate Variational Mode Decomposition (MVMD).

    Decomposes a multivariate signal into ``num_modes`` band-limited
    intrinsic mode functions (IMFs) via alternating direction method of
    multipliers (ADMM) in the frequency domain.

    Args:
        num_modes: Number of modes (IMFs) to extract.
        alpha: Bandwidth constraint (balancing parameter).
        tau: Noise tolerance (Lagrangian multiplier update step).
        tol: Convergence tolerance.
        max_iter: Maximum number of ADMM iterations.
    """

    def __init__(
        self,
        num_modes: int = 5,
        alpha: float = 2000.0,
        tau: float = 0.0,
        tol: float = 1e-7,
        max_iter: int = 500,
    ) -> None:
        super().__init__()
        self.num_modes = num_modes
        self.alpha = alpha
        self.tau = tau
        self.tol = tol
        self.max_iter = max_iter

    @torch.no_grad()
    def decompose(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose a multivariate signal into modes.

        Args:
            x: Input signal of shape ``(batch, channels, time)``.

        Returns:
            Tuple of:
                - ``imfs``: Intrinsic mode functions of shape
                  ``(batch, num_modes, channels, time)``.
                - ``omega``: Centre frequencies of shape
                  ``(batch, num_modes)``.
        """
        batch, C, T = x.shape
        device = x.device
        dtype = x.dtype

        # Mirror signal to reduce boundary effects
        T_ext = 2 * T
        f = torch.zeros(batch, C, T_ext, device=device, dtype=dtype)
        f[:, :, T // 2 : T // 2 + T] = x

        # One-sided frequency axis [0, 0.5)
        N = T_ext
        freqs = torch.arange(N, device=device, dtype=dtype) / N  # [0, 1)

        # Compute FFT of the extended signal
        f_hat = torch.fft.fft(f, dim=-1)  # (batch, C, N)
        f_hat_pos = f_hat.clone()
        f_hat_pos[:, :, N // 2 :] = 0.0  # keep only positive frequencies

        # Initialise mode spectra, centre frequencies, and Lagrange multiplier
        u_hat = torch.zeros(batch, self.num_modes, C, N, device=device, dtype=torch.complex64)
        omega = torch.zeros(batch, self.num_modes, device=device, dtype=dtype)
        # Spread initial centre frequencies uniformly
        for k in range(self.num_modes):
            omega[:, k] = 0.5 / self.num_modes * k

        lambda_hat = torch.zeros(batch, C, N, device=device, dtype=torch.complex64)

        f_hat_pos_c = f_hat_pos.to(torch.complex64)
        freqs_c = freqs.to(torch.complex64)
        alpha = float(self.alpha)

        for _ in range(self.max_iter):
            omega_prev = omega.clone()

            for k in range(self.num_modes):
                # Sum of all other modes
                sum_other = u_hat.sum(dim=1) - u_hat[:, k]  # (batch, C, N)

                # Denominator: 1 + 2*alpha*(freqs - omega_k)^2
                # Shape broadcast: (batch, 1, N) for omega[:, k] -> (batch, 1, 1)
                ok = omega[:, k].view(batch, 1, 1).to(dtype)  # (batch, 1, 1)
                denom = (1.0 + 2.0 * alpha * (freqs.view(1, 1, N) - ok) ** 2).to(
                    torch.complex64
                )

                u_hat[:, k] = (f_hat_pos_c - sum_other + lambda_hat / 2.0) / denom

                # Update centre frequency (power-weighted mean over positive freqs)
                power = (u_hat[:, k].abs() ** 2).sum(dim=1)  # (batch, N)
                pos_freqs = freqs[: N // 2].view(1, N // 2)  # (1, N/2)
                num = (pos_freqs * power[:, : N // 2]).sum(dim=-1)
                den = power[:, : N // 2].sum(dim=-1) + 1e-10
                omega[:, k] = num / den

            # Lagrange multiplier update
            residual = f_hat_pos_c - u_hat.sum(dim=1)
            lambda_hat = lambda_hat + self.tau * residual

            # Check convergence
            diff = ((omega - omega_prev) ** 2).sum(dim=-1).max()
            if diff < self.tol:
                break

        # Reconstruct real IMFs from one-sided spectra
        imfs_list = []
        for k in range(self.num_modes):
            # Full spectrum: add conjugate-symmetric negative frequencies
            u_full = torch.zeros(batch, C, N, device=device, dtype=torch.complex64)
            u_full[:, :, : N // 2] = u_hat[:, k, :, : N // 2]
            # Conjugate-symmetric part
            u_full[:, :, N // 2 + 1 :] = u_hat[:, k, :, 1 : N // 2].flip(-1).conj()
            imf_ext = torch.fft.ifft(u_full, dim=-1).real  # (batch, C, N)
            # Trim to original length
            imf = imf_ext[:, :, T // 2 : T // 2 + T]
            imfs_list.append(imf.unsqueeze(1))  # (batch, 1, C, T)

        imfs = torch.cat(imfs_list, dim=1)  # (batch, num_modes, C, T)
        return imfs, omega

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose input signal.

        Args:
            x: Input of shape ``(batch, channels, time)``.

        Returns:
            Same as :meth:`decompose`.
        """
        return self.decompose(x)
