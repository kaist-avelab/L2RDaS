"""Loss functions used by the RTNH anchor head."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SigmoidFocalClassificationLoss(nn.Module):
    """Anchor-wise sigmoid focal classification loss."""

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(
        inputs: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return (
            torch.clamp(inputs, min=0)
            - inputs * targets
            + torch.log1p(torch.exp(-torch.abs(inputs)))
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        pred_sigmoid = torch.sigmoid(inputs)
        alpha_weight = (
            targets * self.alpha + (1 - targets) * (1 - self.alpha)
        )
        pt = targets * (1.0 - pred_sigmoid) + (1.0 - targets) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)
        loss = focal_weight * self.sigmoid_cross_entropy_with_logits(
            inputs, targets
        )

        if len(weights.shape) == 2 or (
            len(weights.shape) == 1 and len(targets.shape) == 2
        ):
            weights = weights.unsqueeze(-1)
        assert len(weights.shape) == len(loss.shape)
        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """Code-wise and anchor-wise weighted Smooth L1 loss."""

    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        super().__init__()
        self.beta = beta
        self.code_weights = None
        if code_weights is not None:
            values = np.asarray(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(values).cuda()

    @staticmethod
    def smooth_l1_loss(diff: torch.Tensor, beta: float) -> torch.Tensor:
        if beta < 1e-5:
            return torch.abs(diff)
        absolute = torch.abs(diff)
        return torch.where(
            absolute < beta,
            0.5 * absolute**2 / beta,
            absolute - 0.5 * beta,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor = None,
    ) -> torch.Tensor:
        targets = torch.where(torch.isnan(targets), inputs, targets)
        diff = inputs - targets
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)
        if weights is not None:
            assert weights.shape[:2] == loss.shape[:2]
            loss = loss * weights.unsqueeze(-1)
        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """Anchor-wise weighted direction classification loss."""

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        inputs = inputs.permute(0, 2, 1)
        targets = targets.argmax(dim=-1)
        return F.cross_entropy(inputs, targets, reduction="none") * weights
