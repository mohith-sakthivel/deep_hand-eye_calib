import os
import datetime
import random
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torch_geometric.loader import DataLoader

import deep_hand_eye.pose_utils as p_utils
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.model import GCNet
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.losses import PoseNetCriterion


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.epochs = 40
config.save_dir = ""
config.model_name = ""
config.batch_size = 16
config.eval_freq = 2000
config.log_freq = 20    # Iters to log after
config.save_freq = 5   # Epochs to save after. Set None to not save.
config.rel_pose_coeff = 1   # Set None to remove this auxiliary loss
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

    def __init__(self, config: AttrDict) -> None:
        self.config = config

        self.beta_loss_coeff = 0.0  # initial relative translation loss coeff
        self.gamma_loss_coeff = -3   # initial relative rotation loss coeff
        self.weight_decay = 0.0005
        self.lr = 5e-5
        self.lr_decay = 0.1  # learning rate decay factor
        self.lr_decay_step = 20
        self.learn_loss_coeffs = True
        self.run_id = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")

        if self.config.device == "cpu" or not torch.cuda.is_available():
            self.device = "cpu"
        else:
            self.device = "cuda"
        self.tb_writer = tb.writer.SummaryWriter(
            log_dir=self.config.log_dir / self.config.model_name / self.run_id)

        # Setup Dataset
        self.train_dataset = MVSDataset(image_folder="data/DTU_MVS_2014/Rectified/train/")
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

        self.test_dataset = MVSDataset(image_folder="data/DTU_MVS_2014/Rectified/test/")
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers
        )

        # Define the model
        self.model = GCNet(rel_pose=self.config.rel_pose_coeff is not None).to(self.device)
        self.model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        self.params = sum([np.prod(p.size()) for p in self.model_parameters])

        # Define loss
        self.train_criterion_R = PoseNetCriterion(
            beta=self.beta_loss_coeff,
            gamma=self.gamma_loss_coeff,
            learn_beta=True
        ).to(self.device)
        self.train_criterion_he = PoseNetCriterion(
            beta=self.beta_loss_coeff,
            gamma=self.gamma_loss_coeff,
            learn_beta=True
        ).to(self.device)
        self.val_criterion = PoseNetCriterion().to(self.device)

        # Define optimizer
        param_list = [{'params': self.model.parameters()}]
        if self.learn_loss_coeffs and hasattr(self.train_criterion_R, 'beta') \
                and hasattr(self.train_criterion_R, 'gamma'):
            param_list.append(
                {'params': [self.train_criterion_R.beta, self.train_criterion_R.gamma]})

        self.optimizer = torch.optim.Adam(
            param_list, lr=self.lr,
            weight_decay=self.weight_decay
        )

    def train(self):
        iter_no = 0
        for epoch in range(self.config.epochs):
            self.model.train()
            if epoch > 1 and epoch % self.lr_decay_step == 0:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = param_group['lr'] * self.lr_decay
                    print('LR: ', param_group['lr'])

            for batch_idx, data in tqdm(
                enumerate(self.train_dataloader),
                desc=f'[Epoch {epoch:04d}/{self.config.epochs}] train',
                total=len(self.train_dataloader)
            ):
                self.optimizer.zero_grad()

                data = data.to(self.device)
                target_he, target_R = data.y, data.y_edge
                pred_he, pred_R, _ = self.model(data)

                loss_he, loss_he_t, loss_he_q = self.train_criterion_he(pred_he, target_he)
                loss_total = loss_he

                if self.config.rel_pose_coeff is not None:
                    loss_R, loss_R_t, loss_R_q = self.train_criterion_R(pred_R, target_R)
                    loss_total += self.config.rel_pose_coeff * loss_R

                loss_total.backward()
                self.optimizer.step()

                if iter_no % self.config.log_freq == 0:
                    self.tb_writer.add_scalar("train/epoch", epoch, iter_no)
                    self.tb_writer.add_scalar("train/total_loss", loss_total, iter_no)
                    self.tb_writer.add_scalar("train/he_pose_loss", loss_he, iter_no)
                    self.tb_writer.add_scalar("train/he_trans_loss", loss_he_t, iter_no)
                    self.tb_writer.add_scalar("train/he_rot_loss", loss_he_q, iter_no)
                    self.tb_writer.add_scalar("train/he_beta", self.train_criterion_he.beta, iter_no)
                    self.tb_writer.add_scalar("train/he_gamma", self.train_criterion_he.gamma, iter_no)
                    if self.config.rel_pose_coeff is not None:
                        self.tb_writer.add_scalar("train/rel_pose_loss", loss_R, iter_no)
                        self.tb_writer.add_scalar("train/rel_trans_loss", loss_R_t, iter_no)
                        self.tb_writer.add_scalar("train/rel_rot_loss", loss_R_q, iter_no)
                        self.tb_writer.add_scalar("train/rel_beta", self.train_criterion_R.beta, iter_no)
                        self.tb_writer.add_scalar("train/rel_gamma", self.train_criterion_R.gamma, iter_no)

                if iter_no % self.config.eval_freq == 0:
                    self.eval(
                        dataloader=self.test_dataloader,
                        iter_no=iter_no,
                        eval_rel_pose=self.config.rel_pose_coeff is not None
                    )
                    self.model.train()
                iter_no += 1

            if epoch > 0 and self.config.save_freq is not None and (epoch+1) % self.config.save_freq == 0:
                self.save_model()

    @torch.no_grad()
    def eval(self, dataloader: DataLoader, iter_no: int, max_samples: int = 2000, eval_rel_pose: bool = True):
        self.model.eval()

        # loss functions
        def t_criterion(t_pred, t_gt): return np.linalg.norm(t_pred - t_gt)
        q_criterion = p_utils.quaternion_angular_error
        t_loss_he = []
        q_loss_he = []
        if eval_rel_pose:
            t_loss_R = []
            q_loss_R = []
        num_samples = 0

        # inference loop
        for batch_idx, data in tqdm(
            enumerate(dataloader),
            desc=f'[Iter {iter_no:04d}] eval',
            total=max_samples/self.config.batch_size
        ):

            num_samples += data.num_graphs
            data = data.to(self.device)
            output_he, output_R, _ = self.model(data)
            output_he = output_he.cpu().data.numpy()
            target_he = data.y.to('cpu').numpy()

            # normalize the predicted quaternions
            q = [p_utils.qexp(p[3:]) for p in output_he]
            output_he = np.hstack((output_he[:, :3], np.asarray(q)))
            q = [p_utils.qexp(p[3:]) for p in target_he]
            target_he = np.hstack((target_he[:, :3], np.asarray(q)))

            # calculate losses
            for p, t in zip(output_he, target_he):
                t_loss_he.append(t_criterion(p[:3], t[:3]))
                q_loss_he.append(q_criterion(p[3:], t[3:]))

            if eval_rel_pose:
                output_R = output_R.cpu().data.numpy()
                # normalize the predicted quaternions
                target_R = data.y_edge.to('cpu').numpy()

                q = [p_utils.qexp(p[3:]) for p in output_R]
                output_R = np.hstack((output_R[:, :3], np.asarray(q)))
                q = [p_utils.qexp(p[3:]) for p in target_R]
                target_R = np.hstack((target_R[:, :3], np.asarray(q)))

                for p, t in zip(output_R, target_R):
                    t_loss_R.append(t_criterion(p[:3], t[:3]))
                    q_loss_R.append(q_criterion(p[3:], t[3:]))

            if num_samples > max_samples:
                break

        median_t_he = np.median(t_loss_he)
        median_q_he = np.median(q_loss_he)
        mean_t_he = np.mean(t_loss_he)
        mean_q_he = np.mean(q_loss_he)
        max_t_he = np.max(t_loss_he)
        max_q_he = np.max(q_loss_he)

        print(f'Iter No [{iter_no:04d}] \n'
              f'Error in translation: \n'
              f'\t median {median_t_he:3.2f} m \n'
              f'\t mean {mean_t_he:3.2f} m \n'
              f'\t max {max_t_he:3.2f} m \n'
              f'Error in rotation: \n'
              f'\t median {median_q_he:3.2f} degrees \n'
              f'\t mean {mean_q_he:3.2f} degrees \n'
              f'\t max {max_q_he:3.2f} degrees \n')

        self.tb_writer.add_scalar("test/he_trans_median", median_t_he, iter_no)
        self.tb_writer.add_scalar("test/he_trans_mean", mean_t_he, iter_no)
        self.tb_writer.add_scalar("test/he_trans_max", max_t_he, iter_no)
        self.tb_writer.add_scalar("test/he_rot_median", median_q_he, iter_no)
        self.tb_writer.add_scalar("test/he_rot_mean", mean_q_he, iter_no)
        self.tb_writer.add_scalar("test/he_rot_max", max_q_he, iter_no)

        if eval_rel_pose:
            self.tb_writer.add_scalar("test/rel_trans_median", np.median(t_loss_R), iter_no)
            self.tb_writer.add_scalar("test/rel_trans_mean", np.mean(t_loss_R), iter_no)
            self.tb_writer.add_scalar("test/rel_trans_max", np.max(t_loss_R), iter_no)
            self.tb_writer.add_scalar("test/rel_rot_median", np.median(q_loss_R), iter_no)
            self.tb_writer.add_scalar("test/rel_rot_mean", np.mean(q_loss_R), iter_no)
            self.tb_writer.add_scalar("test/rel_rot_max", np.max(q_loss_R), iter_no)

    def save_model(self):
        save_dir = self.config.model_save_dir / self.config.model_name / self.run_id
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.model, save_dir / "model.pt")


def main():
    seed_everything(0)
    trainer = Trainer(config=config)
    trainer.train()


if __name__ == "__main__":
    main()
