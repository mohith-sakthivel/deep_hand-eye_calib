import random
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import torch
import torch.utils.tensorboard as tb
from torch_geometric.loader import DataLoader

from matplotlib.ticker import FormatStrFormatter
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


DEVICE = "cuda"


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def t_criterion(t_pred, t_gt):
    return np.linalg.norm(t_pred - t_gt)

seed_everything(0)


run_id = '2022-10-17--19-11-31'
model = torch.load(f'models/{run_id}/model.pt')

test_dataset = MVSDataset(image_folder="data/DTU_MVS_2014/Rectified/test/", get_raw_images=True)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=config.batch_size,
    shuffle=True,
    num_workers=config.num_workers
)


MAX_SAMPLES = 2000

model.eval()

# loss functions
q_criterion = p_utils.quaternion_angular_error
t_loss_he = []
q_loss_he = []
t_loss_R = []
q_loss_R = []
num_samples = 0

# inference loop
for batch_idx, data in tqdm(enumerate(test_dataloader), total=MAX_SAMPLES/config.batch_size):
    num_samples += data.num_graphs
    data = data.to(DEVICE)
    output_he, output_R, _ = model(data)
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

    # Visualization of Results
    for idx_in_batch, img_list in enumerate(data.raw_images): 
        print(f'Error in translation: \n'
            f'\t {t_loss_he[-config.batch_size + idx_in_batch]:3.2f} m \n'
            f'Error in rotation: \n'
            f'\t {q_loss_he[-config.batch_size + idx_in_batch]:3.2f} degrees \n'
            f'Error in relative translation: \n'
            f'\t {t_loss_R[-config.batch_size + idx_in_batch]:3.2f} m \n'
            f'Error in relative rotation: \n'
            f'\t {q_loss_R[-config.batch_size + idx_in_batch]:3.2f} degrees \n')    

        row1 = cv2.hconcat([cv2.resize(img_list[0], (600, 450)), cv2.resize(img_list[1], (600, 450)), cv2.resize(img_list[2], (600, 450))])
        row2 = cv2.hconcat([cv2.resize(img_list[3], (600, 450)), cv2.resize(img_list[4], (600, 450)), np.zeros_like(cv2.resize(img_list[3], (600, 450)))])
        combined = cv2.vconcat([row1, row2])

        angle = 2 * np.rad2deg(np.arccos(target_he[idx_in_batch, 3]))
        axis = target_he[idx_in_batch, 4:] / np.linalg.norm(target_he[idx_in_batch, 4:])

        pos_str = "  ".join([f"{1000 * item: 0.2f}" for item in target_he[idx_in_batch, :3]])
        axis_str = "  ".join([f"{item: 0.4f}" for item in axis])
        he_pos_str = "Hand Eye: Translation (mm):  " + pos_str + f"  Net: {1000 * np.linalg.norm(target_he[idx_in_batch, :3]): 0.2f}"
        he_rot_str = f"Hand Eye: Angle {angle: 0.2f} degrees   Axis: " + axis_str 


        cv2.putText(combined, he_pos_str, (1200, 500), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(combined, he_rot_str, (1200, 550), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(combined, f'Error in translation: {t_loss_he[-config.batch_size + idx_in_batch]*1000:3.2f} mm', (1200, 600), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(combined, f'Error in rotation: {q_loss_he[-config.batch_size + idx_in_batch]:3.2f} degrees', (1200, 650), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(combined, f'Error in relative translation: {t_loss_R[-config.batch_size + idx_in_batch]*1000:3.2f} mm', (1200, 700), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.putText(combined, f'Error in relative rotation: {q_loss_R[-config.batch_size + idx_in_batch]:3.2f} degrees', (1200, 750), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
        cv2.imshow("Images", combined)
        # cv2.imwrite(f"images/batch_{batch_idx}_{idx_in_batch}.jpg", combined)
        cv2.waitKey(0)

    if num_samples > MAX_SAMPLES:
        break

median_t_he = np.median(t_loss_he)
median_q_he = np.median(q_loss_he)
mean_t_he = np.mean(t_loss_he)
mean_q_he = np.mean(q_loss_he)
max_t_he = np.max(t_loss_he)
max_q_he = np.max(q_loss_he)

median_t_R = np.median(t_loss_R)
median_q_R = np.mean(t_loss_R)
mean_t_R = np.max(t_loss_R)
mean_q_R = np.median(q_loss_R)
max_t_R = np.mean(q_loss_R)
max_q_R = np.max(q_loss_R)

fig, axs = plt.subplots(2, 2, tight_layout=True)
axs[0, 0].hist(1000*np.array(t_loss_he), bins=125)
axs[0, 0].set_title('Hand-eye Translational Error')
axs[0, 0].set_xlabel('Error [mm]')
axs[0, 0].set_ylabel('Number of Instances')
axs[0, 0].minorticks_on()
axs[0, 1].hist(q_loss_he, bins=125)
axs[0, 1].set_title('Hand-eye Rotational Error')
axs[0, 1].set_xlabel('Error [deg]')
axs[0, 1].set_ylabel('Number of Instances')
axs[0, 1].minorticks_on()
axs[1, 0].hist(1000*np.array(t_loss_R), bins=125)
axs[1, 0].set_title('Relative Translational Error')
axs[1, 0].set_xlabel('Error [mm]')
axs[1, 0].set_ylabel('Number of Instances')
axs[1, 0].minorticks_on()
axs[1, 1].hist(q_loss_R, bins=125)
axs[1, 1].set_title('Relative Rotational Error')
axs[1, 1].set_xlabel('Error [deg]')
axs[1, 1].set_ylabel('Number of Instances')
axs[1, 1].minorticks_on()
plt.show()

print(f'Error in translation: \n'
        f'\t median {median_t_he:3.2f} m \n'
        f'\t mean {mean_t_he:3.2f} m \n'
        f'\t max {max_t_he:3.2f} m \n'
        f'Error in rotation: \n'
        f'\t median {median_q_he:3.2f} degrees \n'
        f'\t mean {mean_q_he:3.2f} degrees \n'
        f'\t max {max_q_he:3.2f} degrees \n'
        f'Error in relative translation: \n'
        f'\t median {median_t_R:3.2f} m \n'
        f'\t mean {median_q_R:3.2f} m \n'
        f'\t max {mean_t_R:3.2f} m \n'
        f'Error in relative rotation: \n'
        f'\t median {mean_q_R:3.2f} degrees \n'
        f'\t mean {max_t_R:3.2f} degrees \n'
        f'\t max {max_q_R:3.2f} degrees \n')
