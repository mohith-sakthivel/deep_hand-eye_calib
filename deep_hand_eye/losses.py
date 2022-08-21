from typing import Tuple

import torch
import torch.nn as nn


class PoseNetCriterion(nn.Module):
    def __init__(
        self,
        translation_loss: torch.nn.Module = nn.L1Loss(),
        quaternion_loss: torch.nn.Module = nn.L1Loss(),
        beta: float = 0,
        gamma: float = 0,
        learn_beta: bool = True,
    ):
        super().__init__()
        self.translation_loss = translation_loss
        self.quaternion_loss = quaternion_loss
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)

    def forward(self, pred: torch.Tensor, targ: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param pred: N x 7
        :param targ: N x 7
        """
        # TODO: convert to log quaternion
        s = pred.size()
        if len(s) == 3:
            pred = pred.view(-1, *s[2:])
            targ = targ.view(-1, *s[2:])
        t_loss = self.translation_loss(pred[..., :3], targ[..., :3])
        q_loss = self.quaternion_loss(pred[..., 3:], targ[..., 3:])
        beta_exp = torch.exp(-self.beta)
        gamma_exp = torch.exp(-self.gamma)
        loss = beta_exp * t_loss + self.beta + gamma_exp * q_loss + self.gamma

        return loss, t_loss, q_loss
