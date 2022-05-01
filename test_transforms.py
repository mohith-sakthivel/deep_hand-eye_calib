from turtle import numinput
import numpy as np
from scipy.spatial.transform import Rotation as R
from deep_hand_eye.dataset import MVSDataset
from torch_geometric.loader import DataLoader

import deep_hand_eye.pose_utils as p_utils

dataset = MVSDataset(max_rot_offset=0)

batch_size = 4
train_loader = DataLoader(dataset, batch_size=batch_size)
data = next(iter(train_loader))


for n in range(batch_size):

    hand_eye = data[n].y.squeeze()
    hand_eye_quat = hand_eye[3:][[1,2,3,0]]
    hand_eye_homogenous = R.from_quat(hand_eye_quat).as_matrix()
    hand_eye_translate = np.array(hand_eye[:3]).reshape(-1,1)
    hand_eye_homogenous = np.hstack([hand_eye_homogenous, hand_eye_translate])
    hand_eye_homogenous = np.vstack([hand_eye_homogenous, [0,0,0,1]])

    for i, (from_idx, to_idx) in enumerate(data[n].edge_index.T):

        rel_ee_transform = data[n].edge_attr[i]
        rel_cam_transform = data[n].y_edge[i]

        ee_quat = rel_ee_transform[3:][[1,2,3,0]]
        ee_homogenous = R.from_quat(ee_quat).as_matrix()
        ee_translate = np.array(rel_ee_transform[:3]).reshape(-1,1)
        ee_homogenous = np.hstack([ee_homogenous, ee_translate])
        ee_homogenous = np.vstack([ee_homogenous, [0,0,0,1]])

        cam_quat = rel_cam_transform[3:][[1,2,3,0]]
        cam_homogenous = R.from_quat(cam_quat).as_matrix()
        cam_translate = np.array(rel_cam_transform[:3]).reshape(-1,1)
        cam_homogenous = np.hstack([cam_homogenous, cam_translate])
        cam_homogenous = np.vstack([cam_homogenous, [0,0,0,1]])

        err_residuals = cam_homogenous - p_utils.invert_homo(hand_eye_homogenous) @ ee_homogenous @ hand_eye_homogenous
        print("max error : ", np.max(err_residuals))
        assert np.all(np.abs(err_residuals) < 1e-5)


