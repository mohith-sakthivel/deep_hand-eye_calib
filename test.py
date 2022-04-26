from dataloader import MVSDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = MVSDataset()

train_loader = DataLoader(dataset, batch_size=500)
data = next(iter(train_loader))

input('')