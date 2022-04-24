import torch 
from MoGNN import MoGNN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

### TO-DO: The below lines should come from the dataloader ###
# Fully connected sample of 5
edge_index = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
                           [1, 2, 3, 4, 0, 2, 3, 4, 0, 1, 3, 4, 0, 1, 2, 4, 0, 1, 2, 3]], dtype=torch.long)

# 5 Nodes with random feature vector of one (to be replaced by output from Resnet)
x = torch.rand(5, 2048, dtype=torch.float)

# Random edge attributes
edge_attr = torch.rand(20, 7, dtype=torch.float)

# Random ground truth
y = torch.rand(1, 7, dtype=torch.float)

data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
### ###

# Initialize model
model = MoGNN(data, 2048)

# Create Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Define optimizer.

# Training
def train(data):
    # Clear gradients
    optimizer.zero_grad()

    # Perform a single forward pass
    out, h = model(data.x, data.edge_index, data.edge_attr, 1)

    # Compute the loss solely based on the training nodes
    loss = model.dist_func(out, data.y)

    # Derive gradients
    loss.backward()
    
    # Update parameters based on gradients
    optimizer.step()

    return loss, h

# TO-DO: Write Validation Test Function

# Main script to be run
if __name__ == "__main__":
    for epoch in range(100):
        loss, h = train(data)
        print(f'{epoch}: {loss}')