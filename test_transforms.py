from turtle import numinput
import numpy as np
from scipy.spatial.transform import Rotation as R
from deep_hand_eye.dataset import MVSDataset
from torch_geometric.loader import DataLoader

dataset = MVSDataset(max_rot_offset=0)

batch_size = 4
train_loader = DataLoader(dataset, batch_size=batch_size)
data = next(iter(train_loader))

num_nodes = data[0].edge_attr.shape[0]


for n in range(batch_size):
    for i in range(num_nodes):
        for j in range(num_nodes):

            if i == j:
                continue
            
            print(" iteration  ", i ,"  ", j)
            ee_transform = data[n].edge_attr[i][j]

            hand_eye = data[n].y

            cam_transform = data[n].y_edge[i][j]

            ee_quat = ee_transform[3:]
            ee_quat = ee_quat[[1,2,3,0]]

            ee_homogenous = R.from_quat(ee_quat).as_matrix()

            ee_translate = np.array(ee_transform[:3]).reshape(-1,1)
            
            ee_homogenous = np.hstack([ee_homogenous, ee_translate])

            ee_homogenous = np.vstack([ee_homogenous, [0,0,0,1]])

            cam_quat = cam_transform[3:]
            cam_quat = cam_quat[[1,2,3,0]]

            cam_homogenous = R.from_quat(cam_quat).as_matrix()
            
            cam_translate = np.array(cam_transform[:3]).reshape(-1,1)
            
            cam_homogenous = np.hstack([cam_homogenous, cam_translate])

            cam_homogenous = np.vstack([cam_homogenous, [0,0,0,1]])

            hand_eye_quat = hand_eye[3:]
            hand_eye_quat = hand_eye_quat[[1,2,3,0]]

            hand_eye_homogenous = R.from_quat(hand_eye_quat).as_matrix()
            hand_eye_translate = np.array(hand_eye[:3]).reshape(-1,1)

            hand_eye_homogenous = np.hstack([hand_eye_homogenous, hand_eye_translate])

            hand_eye_homogenous = np.vstack([hand_eye_homogenous, [0,0,0,1]])

            print("max error : ", np.max(cam_homogenous - hand_eye_homogenous @ ee_homogenous))
            print("max element in homogenous matrix : ", np.where(  cam_homogenous - hand_eye_homogenous @ ee_homogenous == np.max(cam_homogenous - hand_eye_homogenous @ ee_homogenous)  )  )

            assert np.all( abs(cam_homogenous - hand_eye_homogenous @ ee_homogenous) < 0.1)


