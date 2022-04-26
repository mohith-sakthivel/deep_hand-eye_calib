from dataloader import dataloader
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = dataloader()

train_loader = DataLoader(dataset, batch_size=2)
data = next(iter(train_loader))
print(data['images'].shape)
plt.imshow(data['images'][0][0].permute(1, 2, 0))
print(data['hand_eye'])
print(data['transforms'])

input('')