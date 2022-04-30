import torch
import torch.nn as nn


class PoseNetCriterion(nn.Module):
    def __init__(self, trans_loss=nn.L1Loss(), quat_loss=nn.L1Loss(),
                 beta=0, gamma=0, learn_beta=True):
        super().__init__()
        self.trans_loss = trans_loss
        self.quat_loss = quat_loss
        self.beta = nn.Parameter(torch.Tensor([beta]), requires_grad=learn_beta)
        self.gamma = nn.Parameter(torch.Tensor([gamma]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        # TODO: convert to log quaternion
        s = pred.size()
        if len(s) == 3:
            pred = pred.view(-1, *s[2:])
            targ = targ.view(-1, *s[2:])
        t_loss = self.trans_loss(pred[..., :3], targ[..., :3])
        q_loss = self.quat_loss(pred[..., 3:], targ[..., 3:])
        beta_exp = torch.exp(-self.beta)
        gamma_exp = torch.exp(-self.gamma)
        loss = beta_exp * t_loss + self.beta + \
               gamma_exp * q_loss + self.gamma
        return loss, t_loss, q_loss
