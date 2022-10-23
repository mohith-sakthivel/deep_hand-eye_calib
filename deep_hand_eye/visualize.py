import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

import deep_hand_eye.pose_utils as p_utils
from deep_hand_eye.utils import AttrDict
from deep_hand_eye.dataset import MVSDataset
from deep_hand_eye.viz_utils import vizualize_poses


MODEL_PATH = 'models/2022-10-17--19-11-31/model.pt'
DATA_PATH = "data/DTU_MVS_2014/Rectified/test/"
MAX_SAMPLES = 20
LOG_DIR = "eval"
LOG_IMAGES = True


config = AttrDict()
config.seed = 0
config.device = "cuda"
config.num_workers = 8
config.batch_size = 2


def seed_everything(seed: int):
    """From https://pytorch-lightning.readthedocs.io/en/latest/_modules/pytorch_lightning/utilities/seed.html#seed_everything"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def run_eval():
    seed_everything(config.seed)
    log_dir = Path(LOG_DIR)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    model = torch.load(MODEL_PATH)
    model.eval()

    test_dataset = MVSDataset(image_folder=DATA_PATH, get_raw_images=True)
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # loss functions
    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt, axis=-1)
    q_criterion = p_utils.quaternion_angular_error

    t_loss_he = []
    q_loss_he = []
    t_loss_R = []
    q_loss_R = []
    count = 0

    # inference loop
    for data in tqdm(test_dataloader, total=MAX_SAMPLES // config.batch_size):
        data = data.to(config.device)
        output_he, output_R, _ = model(data)

        output_he = output_he.cpu().numpy()
        target_he = data.y.cpu().numpy()
        output_R = output_R.cpu().numpy()
        target_R = data.y_edge.cpu().numpy()

        # Generate poses from predictions and label for hand-eye pose
        q = [p_utils.qexp(p[3:]) for p in output_he]
        output_he = np.hstack((output_he[:, :3], np.asarray(q)))
        q = [p_utils.qexp(p[3:]) for p in target_he]
        target_he = np.hstack((target_he[:, :3], np.asarray(q)))

        # Calculate hand-eye losses
        t_loss_he_batch = t_criterion(output_he[:, :3], target_he[:, :3])
        q_loss_he_batch = q_criterion(output_he[:, 3:], target_he[:, 3:])

        # Generate poses from predictions and label for relative poses
        q = [p_utils.qexp(p[3:]) for p in output_R]
        output_R = np.hstack((output_R[:, :3], np.asarray(q)))
        q = [p_utils.qexp(p[3:]) for p in target_R]
        target_R = np.hstack((target_R[:, :3], np.asarray(q)))

        t_loss_R_batch = t_criterion(output_R[:, :3], target_R[:, :3])
        q_loss_R_batch = q_criterion(output_R[:, 3:], target_R[:, 3:])

        t_loss_he.append(t_loss_he_batch)
        q_loss_he.append(q_loss_he_batch)
        t_loss_R.append(t_loss_R_batch)
        q_loss_R.append(q_loss_R_batch)

        edge_batch_idx = data.batch[data.edge_index[0]].cpu().numpy()
        edge_first_nodes = data.edge_index[0].cpu().numpy()

        # Visualization of Results
        for graph_id in range(data.num_graphs):
            img_list = data.raw_images[graph_id] 
            count += 1

            axis, angle = p_utils.quaternion_to_axis_angle(target_he[graph_id, 3:], in_degrees=True)
            edge_mask = edge_batch_idx == graph_id
            rel_pos_error = t_loss_R_batch[edge_mask].mean(axis=0) * 1000
            rel_rot_error = q_loss_R_batch[edge_mask].mean(axis=0)

            row1 = cv2.hconcat([cv2.resize(img_list[0], (600, 450)), cv2.resize(img_list[1], (600, 450)), cv2.resize(img_list[2], (600, 450))])
            row2 = cv2.hconcat([cv2.resize(img_list[3], (600, 450)), cv2.resize(img_list[4], (600, 450)), np.zeros_like(cv2.resize(img_list[3], (600, 450)))])
            combined = cv2.vconcat([row1, row2])

            pos_str = "  ".join([f"{1000 * item: 0.2f}" for item in target_he[graph_id, :3]])
            he_pos_str = "HE Translation (mm):  " + pos_str + f"  Total: {1000 * np.linalg.norm(target_he[graph_id, :3]): 0.2f}"
            axis_str = "  ".join([f"{item: 0.4f}" for item in axis])
            he_rot_str = f"HE Angle {angle: 0.2f} degrees   Axis: " + axis_str

            cv2.putText(combined, he_pos_str, (1200, 550), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2.putText(combined, he_rot_str, (1200, 600), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2.putText(combined, f"HE Translation Error: {t_loss_he_batch[graph_id]*1000:3.2f} mm", (1200, 650), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2.putText(combined, f"HE Rotation Error: {q_loss_he_batch[graph_id]:3.2f} degrees", (1200, 700), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2.putText(combined, f"Relative Translation Error: {rel_pos_error:3.2f} mm", (1200, 750), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            cv2.putText(combined, f"Relative Rotation Error: {rel_rot_error:3.2f} degrees", (1200, 800), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 0, 0), 1)
            if LOG_IMAGES:
                plt.imsave(str(log_dir / f"data_{count}.jpg"), combined)
            plt.imshow(combined)
            plt.show()

            graph_first_node = min(edge_first_nodes[edge_mask])
            reqd_edges = edge_first_nodes == graph_first_node

            R_poses = target_R[reqd_edges]
            R_poses = np.vstack((np.array([[0, 0, 0, 1, 0, 0, 0]], dtype=float), R_poses))
            vizualize_poses(R_poses)


        if count > MAX_SAMPLES:
            break

    t_loss_he = np.concatenate(t_loss_he, axis=0)
    q_loss_he = np.concatenate(q_loss_he, axis=0)
    t_loss_R = np.concatenate(t_loss_R, axis=0).reshape(-1)
    q_loss_R = np.concatenate(q_loss_R, axis=0).reshape(-1)

    def get_stats(qty):
        return np.mean(qty, axis=0), np.median(qty, axis=0), np.max(qty, axis=0)

    mean_t_he, median_t_he, max_t_he = get_stats(t_loss_he)
    mean_q_he, median_q_he, max_q_he = get_stats(q_loss_he)

    mean_t_R, median_t_R, max_t_R = get_stats(t_loss_R)
    mean_q_R, median_q_R, max_q_R = get_stats(q_loss_R)

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


if __name__ == "__main__":
    run_eval()
