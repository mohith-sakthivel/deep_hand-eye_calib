import os
import datetime
import argparse
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torch_geometric.loader import DataLoader

import deep_hand_eye.utils as utils
import deep_hand_eye.pose_utils as p_utils
from deep_hand_eye.model import GCNet
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.losses import PoseNetCriterion


config = utils.AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 25
config.save_dir = ""
config.model_name = ""
config.save_model = True  # save model parameters?
config.batch_size = 8
config.eval_freq = 1000
config.log_freq = 20
config.aux_coeffs = {
    "rel_cam_pose": 1
}
config.model_name = ""
config.log_dir = Path("runs")
config.model_save_dir = Path("models")


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Trainer(object):

    def __init__(self, config) -> None:
        self.config = config

        self.beta_loss_coeff = 0.0  # initial relative translation loss coeff
        self.gamma_loss_coeff = -3   # initial relative rotation loss coeff
        self.weight_decay = 0.0005
        self.lr = 5e-5
        self.lr_decay = 0.1  # learning rate decay factor
        self.lr_decay_step = 20
        self.learn_loss_coeffs = True
        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        if config.device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.tb_writer = tb.writer.SummaryWriter(
            log_dir=self.config.log_dir / self.config.model_name / self.run_id)

        # Setup Dataset
        self.train_dataset = MVSDataset()
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
        iter_no = 0
        for epoch in tqdm(range(self.config.epochs), desc='epoch', total=self.config.epochs):
            self.model.train()
            if epoch > 1 and epoch % self.lr_decay_step == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.lr_decay
                    print('LR: ', param_group['lr'])

            for batch_idx, data in tqdm(enumerate(self.train_dataloader),
                                        desc=f'[Epoch {epoch:04d}] train',
                                        total=len(self.train_dataloader)):
                self.optimizer.zero_grad()

                data = data.to(self.device)
                target_he, target_R = data.y, data.y_edge
                pred_he, pred_R, _ = self.model(data)

                loss_he, loss_he_t, loss_he_q = self.train_criterion_he(
                    pred_he, target_he)
                loss_R, loss_R_t, loss_R_q = self.train_criterion_R(
                    pred_R, target_R)

                loss_total = (loss_he +
                              self.config.aux_coeffs["rel_cam_pose"] * loss_R)

                loss_total.backward()
                self.optimizer.step()

                if iter_no % config.log_freq == 0:
                    log_data = {"epoch": epoch, "total_loss": loss_total,
                                "he_pose_loss": loss_he, "he_trans_loss": loss_he_t,
                                "he_rot_loss": loss_he_q, "he_beta": self.train_criterion_he.beta,
                                "he_gamma": self.train_criterion_he.gamma, "rel_pose_loss": loss_R,
                                "rel_trans_loss": loss_R_t, "rel_rot_loss": loss_R_q,
                                "rel_beta": self.train_criterion_R.beta, "rel_gamma": self.train_criterion_R.gamma}
                    self.tb_writer.add_scalars("train", log_data, iter_no)

                iter_no += 1

            if iter_no % config.eval_freq == 0:
                self.eval(self.train_dataloader, iter_no)

    def eval(self, dataloader, iter_no, max_samples=5000):
        self.model.eval()

        # loss functions
        def t_criterion(t_pred, t_gt): return np.linalg.norm(t_pred - t_gt)
        q_criterion = p_utils.quaternion_angular_error
        t_loss = []
        q_loss = []
        num_samples = 0

        # inference loop
        for batch_idx, data in tqdm(enumerate(dataloader), desc=f'[Iter {iter_no:04d}] eval',
                                    total=len(dataloader)):

            num_samples += data.num_graphs
            data = data.to(self.device)
            output_he, _, _ = self.model(data)
            output_he = output_he.cpu().data.numpy()
            target = data.y.to('cpu').numpy()

            # normalize the predicted quaternions
            q = [p_utils.qexp(p[3:]) for p in output_he]
            output_he = np.hstack((output_he[:, :3], np.asarray(q)))
            q = [p_utils.qexp(p[3:]) for p in target]
            target = np.hstack((target[:, :3], np.asarray(q)))

            # calculate losses
            for p, t in zip(output_he, target):
                t_loss.append(t_criterion(p[:3], t[:3]))
                q_loss.append(q_criterion(p[3:], t[3:]))

            if num_samples > max_samples:
                break

        median_t = np.median(t_loss)
        median_q = np.median(q_loss)
        mean_t = np.mean(t_loss)
        mean_q = np.mean(q_loss)

        print(f'Iter No [{iter_no:04d}] Error in translation:'
              f' median {median_t:3.2f} m,'
              f' mean {mean_t:3.2f} m'
              f'\tError in rotation:'
              f' median {median_q:3.2f} degrees,'
              f' mean {mean_q:3.2f} degrees')

        log_data = {"trans_median": median_t, "trans_mean": mean_t,
                    "rot_median": median_q, "rot_mean": mean_q}
        self.tb_writer.add_scalars("test", log_data, iter_no)

        return median_t, mean_t, median_q, mean_q

    def save_model(self):
        save_dir = self.config.model_save_dir / self.config.model_name / self.run_id
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model, save_dir / "model.pt")


def main():
    seed_everything(0)
    trainer = Trainer(config=config)
    trainer.train()
    trainer.save_model()


if __name__ == "__main__":
    main()
