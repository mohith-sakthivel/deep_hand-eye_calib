import numpy as np
import json
import os
from PIL import Image
import torchvision.transforms as transforms
import torch
from scipy.spatial.transform import Rotation as R

class dataloader:
    def __init__(self, data_folder, data_file, image_folder):
        
        # inverse of the transform from eye to hand 
        # (negating the translation of the camera to arrive at hand frame)
        self.hand_eye_transform_inv = np.zeros((3,4))
        self.hand_eye_transform_inv[0,0] = 1
        self.hand_eye_transform_inv[1,1] = 1
        self.hand_eye_transform_inv[2,2] = 1

        self.image_folder = image_folder



        self.camera_positions = json.load(open(data_folder + data_file, 'r'))

    def __getitem__(self, idx):

        '''
        inputs:
            idx: scene id
        returns:
            idx: scene id
            imagelist : list of images corresponding to those sampled from the given scene
            transforms : nxnx7 tensor capturing relative transforms between n sampled images (in this case currently hardoded as 5)
                         each entry contains 7 dim vector: [quaternion (1x4) | translation (1x3)]
            hand_eye_transform : hand eye translation vector used to generate data
        '''
        # list of images corresponding to the samples taken from the chosen scene
        imagelist = []
        
        # creating a random vector to emulate hand-eye transform
        # currently imposing it to be a pure translation from the hand
        hand_offset = np.random((3,1))*50

        # computing the inverse of the translation from eye to hand frame
        self.hand_eye_transform_inv[:,-1] = -hand_offset

        folder = self.image_folder + "/scan" + str(idx)
        files = os.listdir(folder)

        # conatins a table of all relative transforms
        # currently hardcoded for 5 sampled images fromt he scene
        # TODO: update to generalize to any number of sampled images
        relative_transforms = torch.zeros(5,5,7)
        
        # caputing the end effector translation and rotations in each scene in the lists below  (end effector with hand eye offset compensated for)
        # for later use while computing relative transform quaternion and translation
        frame_translations = []
        frame_rotations = []
        
        
        # currently sampling 5 uniformly spaced images from the list of 49 for images to be used for one scene
        # TODO: random sampling from all 49 poses without replacement to generate the set of training images for one scene
        for i in range(0,50,10):
            filepath = os.path.join(folder)
            image = Image.open(filepath + files[i]).convert('RGB')
            image = transforms.ToTensor(image)
            imagelist.append(image)

            # load pose of camera for given idx image from json
            abs_translation = np.array(self.camera_positions['pose']['trans'][idx])
            abs_rotation = np.array(self.camera_positions['pose']['rot'][idx])

            # computation of end effector pose by removing the hand-eye offset being done below
            abs_pose = np.zeros(3,4)
            abs_pose[:,3] = abs_rotation
            abs_pose[:,-1] = abs_translation

            # end effector pose without camera
            end_effector_pose = self.hand_eye_transform_inv @ abs_pose

            frame_rotations.append(end_effector_pose[:3,:3])
            frame_translations.append(end_effector_pose[:3,-1])
            
        # iterating over all end effector poses to find relative transforms and populate relative_transform matrix
        for i in range(len(frame_translations)):
            for j in range(len(frame_rotations)):

                rel_transform = np.zeros(7)
                
                if i == j:
                    rel_rotation = np.eye(3)
                    rel_translation = np.zeros(3)
                else:
                    rel_rotation = frame_rotations[i].T @ frame_rotations[j]
                    rel_translation = frame_translations[j] - frame_translations[i]

                rotation = R.from_matrix(rel_rotation)
                rel_rotation_quaternion = rotation.as_quat()

                rel_transform[:4] = rel_rotation_quaternion
                rel_transform[4:] = rel_translation

                relative_transforms[i][j] = torch.from_numpy(rel_transform)




        return {
            'idx': idx,
            'images': imagelist,
            'transforms': relative_transforms,
            'hand_eye_transform': hand_offset
        }