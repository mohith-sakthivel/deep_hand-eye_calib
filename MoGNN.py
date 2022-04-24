# PyTorch
import torch
from torch.nn import Linear, ReLU, Parameter
import torch.nn.functional as F

# PyTorch Geometric
from torch_geometric.nn import global_mean_pool

# Custom GNN Layer
from MoConv import MoConv

class MoGNN(torch.nn.Module):
    def __init__(self, data, embedding_size):
        super().__init__()
        # Set seed for consistent testing
        torch.manual_seed(2022)

        self.conv1 = MoConv(data.num_features + 7, data.num_features)
        self.conv2 = MoConv(data.num_features, data.num_features)
        self.relu = ReLU()

        self.beta = Parameter(torch.tensor(0.0))
        self.gamma = Parameter(torch.tensor(-3.0))

        self.classifier = Linear(embedding_size, 7)

    def forward(self, x, edge_index, edge_attr, batch_size):
        # 1. Obtain node embeddings - iterate twice (or more?)
        h = self.conv1(x, edge_index, edge_attr)
        h = h.relu()
        h = self.conv2(x, edge_index, edge_attr)
        h = h.relu()
        
        # 2. Pool graph features - I chose global mean pool but we can use something else
        x = global_mean_pool(x, batch_size)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier - output translation and log quaternion
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)

        return x

    # Assuming translation is first and log quaternion is second
    def dist_func(self, pred_y, gt_y):
        trans_error = (pred_y[0:3] - gt_y[0:3])**(-self.beta) + self.beta
        rot_error = (pred_y[3:] - gt_y[3:])**(-self.gamma) + self.gamma
        return trans_error + rot_error