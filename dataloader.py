import json
import os
import torch
import random
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from glob import glob
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset


class MVSDataset(Dataset):
    def __init__(self, image_folder="Data/DTU_MVS_2014/Rectified/",
                 json_file="Data/DTU_MVS_2014/camera_pose.json", num_nodes=5,
                 max_trans_offset=0.1, max_rot_offset=0, transform=None, image_size=(256, 256)):
        # Storing the image folder
        self.image_folder = image_folder

        # Initialize idx used to loop through the folders
        self.folder_idx = 0

        # Get all scene names
        self.scans = os.listdir(image_folder)
        
        # Number of nodes in the graph
        self.num_nodes = int(num_nodes)

        # Camera positions dictionary
        self.camera_positions = json.load(open(json_file, 'r'))

        # Translational and rotational magnitude
        self.max_trans_offset = max_trans_offset
        self.max_rot_offset = max_rot_offset

        if transform == None:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.PILToTensor(),
            ])
        else:
            self.transform = transform

        # Image formats for the different brightnesses
        self.image_format = ['rect_{}_0_r5000.png', 'rect_{}_1_r5000.png', 'rect_{}_2_r5000.png',
                            'rect_{}_3_r5000.png', 'rect_{}_4_r5000.png', 'rect_{}_5_r5000.png',
                            'rect_{}_6_r5000.png', 'rect_{}_max.png']

        # Create edge index tensor
        self.edge_index = edge_idx_gen(self.num_nodes)
        
    def __len__(self):
        return len(self.scans)

    def __getitem__(self, id):
        """
        returns:
            image_list : list of images corresponding to those sampled from the given scene
            transforms : nxnx7 tensor capturing relative transforms between n sampled images
                         each entry contains 7 dim vector: [translation (1x3) | quaternion (1x4)]
                         [x y z | x y z w]
            hand_eye : hand eye translation vector used to generate data
        """
        ### Hand-eye transform matrix
        # Random translational offset
        hand_trans = np.random.uniform(-self.max_trans_offset, self.max_trans_offset, 3)
        # Random rotational angle
        hand_rpy = np.random.uniform(-self.max_rot_offset, self.max_rot_offset, 3)
        # Roll matrix
        hand_roll = np.array([[1, 0, 0],
                            [0, np.cos(hand_rpy[0]), -np.sin(hand_rpy[0])],
                            [0, np.sin(hand_rpy[0]), np.cos(hand_rpy[0])]])
        # Pitch matrix
        hand_pitch = np.array([[np.cos(hand_rpy[1]), 0, np.sin(hand_rpy[1])],
                            [0, 1, 0],
                            [-np.sin(hand_rpy[1]), 0, np.cos(hand_rpy[1])]])
        # Yaw matrix
        hand_yaw = np.array([[np.cos(hand_rpy[2]), -np.sin(hand_rpy[2]), 0],
                            [np.sin(hand_rpy[2]), np.cos(hand_rpy[2]), 0],
                            [0, 0, 1]])
        # Combined rotational matrix
        hand_rotation = hand_yaw @ hand_pitch @ hand_roll

        # Hand-eye transformation matrix
        hand_eye_matrix = np.zeros((4,4))
        hand_eye_matrix[:3,3] = hand_trans
        hand_eye_matrix[:3,:3] = hand_rotation
        hand_eye_matrix[3,3] = 1
        # Get inverse to multiply onto absolute positions
        hand_eye_inv = np.linalg.pinv(hand_eye_matrix)
        # Create ground truth 7-sized vector of the hand-eye calibration
        hand_eye = np.empty(7)
        hand_eye[:3] = hand_trans
        hand_eye[3:] = R.from_matrix(hand_rotation).as_quat()
        
        ### Get 5 images of a randomized brightness
        image_list = []
        # Set folder directory 
        folder = self.image_folder + self.scans[self.folder_idx % len(self.scans)] + '/'
        # Choose the brightness
        img_format = random.choice(self.image_format)
        # Find all images in the current scene with the selected brightness
        image_names = glob(folder + img_format.format('*'))
        # Sample N images = num_nodes
        index_list = random.sample(range(len(image_names)), self.num_nodes)
        # For each sampled index, get the image and append as a tensor
        for idx in index_list:
            # +1 because the dataset is 1-base rather than 0-base like Python indexing
            img_id = f'{idx+1:0>3}'
            file_name = folder + img_format.format(img_id)
            image = Image.open(file_name).convert('RGB')
            image = self.transform(image)
            image_list.append(image)
        image_list = torch.stack(image_list)

        # Initialize table for all relative transforms
        relative_transforms = np.zeros((self.num_nodes, self.num_nodes, 7))
        # List of end effector poses
        ee_poses = []
        # Obtain absolute end effector pose from the JSON file
        for idx in index_list:
            # Camera pose components from JSON
            abs_translation = np.array(self.camera_positions['pose']['trans'][idx])
            abs_rotation = np.array(self.camera_positions['pose']['rot'][idx])
            # Camera pose matrix
            abs_pose = np.zeros((4,4))
            abs_pose[:3,:3] = abs_rotation
            abs_pose[:3,3] = abs_translation.T
            abs_pose[3,3] = 1
            # Use inverse of hand-eye matrix to get end effector pose
            ee_pose = hand_eye_inv @ abs_pose
            ee_poses.append(ee_pose)

        # Loop through randomly sampled positions
        # Format is [x y z | x y z w] for [translation | rotation]
        for from_idx in range(self.num_nodes):
            for to_idx in range(self.num_nodes):
                # if relative to itself
                if from_idx == to_idx:
                    relative_transforms[from_idx, to_idx, :-1] = 0
                    relative_transforms[from_idx, to_idx, -1] = 1
                # If relative to another position
                else:
                    rel_rotation = ee_poses[from_idx][:3,:3].T @ ee_poses[to_idx][:3,:3]
                    rel_translation = ee_poses[to_idx][:3,3] - ee_poses[from_idx][:3,3]
                    # Obtain quaternion from relative rotation
                    rotation = R.from_matrix(rel_rotation)
                    rel_rotation_quaternion = rotation.as_quat()
                    # Populate relative transforms matrix
                    relative_transforms[from_idx, to_idx, :3] = rel_translation
                    relative_transforms[from_idx, to_idx, 3:] = rel_rotation_quaternion

        # Turn into a tensor
        relative_transforms = torch.from_numpy(relative_transforms)
        hand_eye = torch.from_numpy(hand_eye)

        # Return as a dictionary
        return {
            'images': image_list,
            'transforms': relative_transforms,
            'hand_eye': hand_eye
        }

def edge_idx_gen(num_nodes):
    edge_index_top = np.zeros((1,num_nodes*(num_nodes-1)))
    for i in range(num_nodes):
        edge_index_top[0, i*(num_nodes-1):(i+1)*(num_nodes-1)] = i
    edge_index_low = np.zeros_like(edge_index_top)
    idx = 0
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edge_index_low[0,j+i*(num_nodes-1)] = j
                idx += 1
    edge_index = torch.tensor(np.stack([edge_index_top, edge_index_low], dim=0))
    return edge_index