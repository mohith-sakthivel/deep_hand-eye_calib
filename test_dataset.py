from deep_hand_eye.dataset import MVSDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = MVSDataset(max_rot_offset=0)

train_loader = DataLoader(dataset, batch_size=2)
data = next(iter(train_loader))

ee_pose = data['ee_transforms'][0,0,1]
cam_pose = data['cam_transforms'][0,0,1]
hand_eye = data['hand_eye'][0]

print(ee_pose)
print(cam_pose)
print(hand_eye)

input('')