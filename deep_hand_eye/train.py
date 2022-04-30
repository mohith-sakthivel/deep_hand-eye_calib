import os
import argparse

import numpy as np
import random
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torch_geometric.loader import DataLoader

import deep_hand_eye.utils as dhe_utils
from deep_hand_eye.model import GCNet
from deep_hand_eye.losses import PoseNetCriterion


config = dhe_utils.AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 100
config.save_dir = ""
config.model_name = ""
config.save_model = True  # save model parameters?
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

        self.beta_loss_coeff = 0.0  # initial relative translation loss coeff
        self.gamma_loss_coeff = -3   # initial relative rotation loss coeff
        self.weight_decay = 0.0005
        self.lr = 5e-5
        self.lr_decay = 0.1  # learning rate decay factor
        self.lr_decay_step = 20
        self.learn_loss_coeffs = True

        if config.device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.tb_writer = tb.writer.SummaryWriter()

        # Setup Dataset
        self.train_dataset = None
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size,
                                           shuffle=True, num_workers=self.config.num_workers)

        # Define the model
        self.model = GCNet().to(self.device)
        self.model_parameters = filter(
            lambda p: p.requires_grad, self.model.parameters())
        self.params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Define loss
        self.train_criterion_R = PoseNetCriterion(
            beta=self.beta_loss_coeff, gamma=self.gamma_loss_coeff, learn_beta=True).to(self.device)
        self.train_criterion_he = PoseNetCriterion(
            beta=self.beta_loss_coeff, gamma=self.gamma_loss_coeff, learn_beta=True).to(self.device)
        self.val_criterion = PoseNetCriterion().to(self.device)

        # Define optimizer
        param_list = [{'params': self.model.parameters()}]
        if self.learn_loss_coeffs and hasattr(self.train_criterion_R, 'beta') \
                and hasattr(self.train_criterion_R, 'gamma'):
            param_list.append(
                {'params': [self.train_criterion_R.beta, self.train_criterion_R.gamma]})

        self.optimizer = torch.optim.Adam(
            param_list, lr=self.lr, weight_decay=self.weight_decay)

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
                
                target_R = data.y.to(self.device)
                target_he = data.y.to(self.device)

                self.optimizer.zero_grad()

                pred_he, pred_R, edge_index = self.model(data.to(self.device))

                loss_R = self.train_criterion_R(pred_R, target_R)
                loss_he = self.train_criterion_he(pred_he, target_he)

                loss_total = loss_R[0]

                loss_total.backward()
                self.optimizer.step()

                self.tb_writer.add_scalar("train/loss", loss_R)

    def eval_RP(self, dataloader, epoch, num_samples=None):
        self.model.eval()

        pred_poses = np.zeros((L, 7))  # store all predicted poses
        targ_poses = np.zeros((L, 7))  # store all target poses

        # loss functions
        t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
        q_criterion = dhe_utils.quaternion_angular_error

        # inference loop
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f'[Epoch {epoch:04d}] eval',
                                    total=len(dataloader)):

            # output : 1 x 6 or 1 x STEPS x 6
            output_he, output_R, edge_index = self.model(data.to(self.device))

            s = output.size()
            output_R = output_R.cpu().data.numpy().reshape((-1, s[-1]))

            target = data.y
            target = target.to('cpu').numpy().reshape((-1, s[-1]))

            edges = edge_index.cpu().data.numpy()

            # Choose one reference absolute pose and compute the absolute poses in the subgraph
            # using predicted relative poses
            valid_edges = edges[1] == 0

            ref_idx = np.argwhere(valid_edges)[ref_node, 0]
            RP_estimate = output_R[ref_idx, :]
            reference_AP = target[edges[0, ref_idx], :]
            output = reference_AP - RP_estimate
            output = np.expand_dims(output, axis=0)

            # normalize the predicted quaternions
            q = [qexp(p[3:]) for p in output]
            output = np.hstack((output[:, :3], np.asarray(q)))
            q = [qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            # take the first prediction
            pred_poses[batch_idx, :] = output[0]
            targ_poses[batch_idx, :] = target[0]

        # calculate losses
        t_loss = np.asarray([t_criterion(p, t)
                             for p, t in zip(pred_poses[:, :3], targ_poses[:, :3])])
        q_loss = np.asarray([q_criterion(p, t)
                             for p, t in zip(pred_poses[:, 3:], targ_poses[:, 3:])])

        median_t = np.median(t_loss)
        median_q = np.median(q_loss)
        mean_t = np.mean(t_loss)
        mean_q = np.mean(q_loss)

        logger.info(f'[Scene: {scene}, set: {set}, Epoch {epoch:04d}] Error in translation:'
                    f' median {median_t:3.2f} m,'
                    f' mean {mean_t:3.2f} m'
                    f'\tError in rotation:'
                    f' median {median_q:3.2f} degrees,'
                    f' mean {mean_q:3.2f} degrees')
        return median_t, mean_t, median_q, mean_q


def main():
    seed_everything(0)
    trainer = Trainer(config=config)
