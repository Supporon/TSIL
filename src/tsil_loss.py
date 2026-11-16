"""The fixed-coefficient, additive TSIL objective (original paper, Section 3.4)."""
import math
from collections.abc import Mapping

import torch
from torch import nn


class StatisticalInvariantLoss(nn.Module):
    """MSE + sum_s beta_s ||Phi_tilde_s.T @ residual / B||^2.

    Descriptors are fixed observations, not trainable parameters. No B x B
    matrix is constructed: the only projection is from B residuals to d_s
    moments. A scalar beta applies to every supplied predicate; a mapping
    specifies an independent, fixed coefficient for each one.
    """

    def __init__(self, beta=0.5):
        super().__init__()
        values = beta.values() if isinstance(beta, Mapping) else [beta]
        if any(not math.isfinite(float(b)) or float(b) < 0 for b in values):
            raise ValueError("predicate coefficients must be finite and nonnegative")
        self.beta = dict(beta) if isinstance(beta, Mapping) else float(beta)

    def forward(self, outputs, targets, descriptors):
        outputs, targets = outputs.reshape(-1), targets.reshape(-1)
        if outputs.shape != targets.shape or outputs.numel() == 0:
            raise ValueError("outputs and targets must contain the same nonzero number of samples")
        residual = outputs - targets
        mse = residual.square().mean()
        penalty = residual.new_zeros(())
        terms = {}
        for name, raw in descriptors.items():
            phi = raw.detach().to(device=residual.device, dtype=residual.dtype)
            if phi.ndim != 2 or phi.shape[0] != residual.numel() or not torch.isfinite(phi).all():
                raise ValueError(f"{name}: expected finite descriptor matrix [B, d]")
            phi = phi / (torch.linalg.vector_norm(phi) + 1e-8)
            moment = phi.T @ residual / residual.numel()
            beta = self.beta[name] if isinstance(self.beta, Mapping) else self.beta
            terms[name] = float(beta) * moment.square().sum()
            penalty = penalty + terms[name]
        return {"total": mse + penalty, "ori_mse": mse, "V_loss": mse,
                "P_loss": penalty, "predicate_losses": terms}
