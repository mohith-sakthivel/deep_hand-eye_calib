import os
import argparse

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torchvision.models import resnet34

import deep_hand_eye.utils as dhe_utils
from deep_hand_eye.model import GCNet
from deep_hand_eye.losses import PoseNetCriterion


config = dhe_utils.AttrDict()
config.seed = 0
config.epochs = 100
config.save_dir = ""
config.model_name = ""
config.save_model = True # save model parameters?
config.batch_size = 8



def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):

    def __init__(self, config) -> None:
        self.model = None
        self.train_dataloader = None
        self.config = config

        self.srx = 0.0  # initial relative translation loss coeff
        self.srq = -3   # initial relative rotation loss coeff
        self.weight_decay = 0.0005 
        self.lr = 5e-5 
        self.lr_decay = 0.1 # learning rate decay factor
        self.lr_decay_step = 20 
        self.learn_gamma = True

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tb_writer = tb.writer.SummaryWriter()

        # Define the model
        self.model = GCNet().to(self.device)
        self.model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Define loss
        self.train_criterion_R = PoseNetCriterion(sax=self.srx, saq=self.srq, learn_beta=True).to(self.device)
        self.val_criterion = PoseNetCriterion().to(self.device)

        # Define optimizer
        param_list = [{'params': self.model.parameters()}]
        if self.learn_gamma and hasattr(self.train_criterion_R, 'sax') \
          and hasattr(self.train_criterion_R, 'saq'):
            param_list.append({'params': [self.train_criterion_R.sax, self.train_criterion_R.saq]})

        self.optimizer = torch.optim.Adam(param_list, lr=self.lr, weight_decay=self.weight_decay)

    def train(self):

        for epoch in tqdm(range(self.config.epochs), desc='epoch', total=self.config.epochs):

            self.model.train()
            if epoch > 1 and epoch % self.lr_decay_step == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.lr_decay
                    print('LR: ', param_group['lr'])
            
            for batch_idx, data in tqdm(enumerate(self.train_loader),
                                        desc=f'[Epoch {epoch:04d}] train',
                                        total=len(self.train_loader)):

                if batch_idx == self.max_num_iter:
                    break

                target = data.y
                target = target.to(self.device)

                self.optimizer.zero_grad()

                pred_R, edge_index = self.model(data.to(self.device))

                target_R = self.model.compute_RP(target, edge_index)

                loss_R = self.train_criterion_R(
                    pred_R.view(1, pred_R.size(0), pred_R.size(1)),
                    target_R.view(1, target_R.size(0), target_R.size(1)))

                loss_total = loss_R[0]

                loss_total.backward()
                self.optimizer.step()



def main():
    seed_everything(0)
    
    trainer = Trainer()

