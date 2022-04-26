import torch
import torch.nn as nn


class PoseNetCriterion(nn.Module):
    def __init__(self, trans_loss=nn.L1Loss(), quat_loss=nn.L1Loss(),
                 sax=0, saq=0, learn_beta=True):
        super().__init__()
        self.trans_loss = trans_loss
        self.quat_loss = quat_loss
        self.sax = nn.Parameter(torch.Tensor([sax]), requires_grad=learn_beta)
        self.saq = nn.Parameter(torch.Tensor([saq]), requires_grad=learn_beta)

    def forward(self, pred, targ):
        """
        :param pred: N x 7
        :param targ: N x 7
        :return:
        """
        s = pred.size()
        if len(s) == 3:
            pred = pred.view(-1, *s[2:])
            targ = targ.view(-1, *s[2:])
        t_loss = self.trans_loss(pred[..., :3], targ[..., :3])
        q_loss = self.quat_loss(pred[..., 3:], targ[..., 3:])
        sax_exp = torch.exp(-self.sax)
        saq_exp = torch.exp(-self.saq)
        loss = sax_exp * t_loss + self.sax + \
               saq_exp * q_loss + self.saq
        return loss, t_loss, q_loss
