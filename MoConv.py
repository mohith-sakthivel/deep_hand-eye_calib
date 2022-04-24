import torch
from torch.nn import Linear, ReLU, Softmax, Parameter
from torch_geometric.nn import MessagePassing

class MoConv(MessagePassing):
    def __init__(self, in_channels=2055, embedding_size=2048):
        super().__init__(aggr='mean')  # "Add" aggregation (Step 5).
        self.linear1 = Linear(in_channels, embedding_size)
        self.linear2 = Linear(embedding_size, embedding_size)
        self.relu = ReLU()
        self.softmax = Softmax()

        # Attention mechanism
        self.attention_scaling = 8
        self.W_theta = Parameter(torch.zeros((int)(embedding_size/self.attention_scaling), embedding_size))
        torch.nn.init.xavier_uniform_(self.W_theta)
        self.W_phi = Parameter(torch.zeros((int)(embedding_size/self.attention_scaling), embedding_size))
        torch.nn.init.xavier_uniform_(self.W_phi)

        # Learnable up/down sampling operators and attention
        self.W_g = Parameter(torch.zeros(embedding_size, (int)(embedding_size/self.attention_scaling)))
        torch.nn.init.xavier_uniform_(self.W_g)
        self.W_f = Parameter(torch.zeros((int)(embedding_size/self.attention_scaling), embedding_size))
        torch.nn.init.xavier_uniform_(self.W_f)

    def forward(self, x, edge_index, edge_attr):

        # Start propagating messages.
        return self.propagate(edge_index, x=x, x_ji=edge_attr)

    # TO-DO: Unsure how to get edge_ij to pass into the message function properly
    def message(self, x_j, edge_ij):
        # 1. Combine neightbor feature and edge feature
        cat = torch.cat([edge_ij, x_j], dim=1)

        # 2. 2-layer MLP with ReLU
        h = self.linear1(cat)
        h = self.relu(h)
        h = self.linear(h)
        msg = self.relu(h)

        # 3. Apply attention mechanism as per paper
        W_ji = self.softmax(self.W_theta * msg * msg.t() * self.W_phi.t())
        a_ji = self.W_g * W_ji * self.W_f * msg
        msg_att = msg + a_ji  

        # 4. Return MLP output
        return msg_att