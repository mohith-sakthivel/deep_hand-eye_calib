from deep_hand_eye.dataset import MVSDataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

dataset = MVSDataset(max_rot_offset=0)

train_loader = DataLoader(dataset, batch_size=1)
data = next(iter(train_loader))

print(data.edge)

print(0)