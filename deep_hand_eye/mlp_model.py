import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from torch_geometric.nn import MessagePassing


class SimpleEdgeModel(nn.Module):
    """
    Network to perform autoregressive edge update during Neural message passing
    """

    def __init__(self, node_channels, edge_in_channels, edge_out_channels):
        super().__init__()
        self.node_channels = node_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * node_channels + edge_in_channels, edge_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(edge_out_channels, edge_out_channels)
        )

    def forward(self, source, target, edge_attr):
        out = torch.cat([edge_attr, source, target], dim=1).contiguous()
        out = self.edge_mlp(out)
        return out


class AttentionBlock(nn.Module):
    """
    Network to apply non-local attention
    (Refer: https://arxiv.org/abs/1711.07971)
    """

    def __init__(self, in_channels, N=8):
        super().__init__()
        self.N = N
        self.in_channels = in_channels
        self.W_theta = nn.Linear(in_channels, in_channels // N)
        self.W_phi = nn.Linear(in_channels, in_channels // N)

        self.W_f = nn.Linear(in_channels, in_channels // N)
        self.W_g = nn.Linear(in_channels // N, in_channels)

    def forward(self, x):
        # Calculate attention map for low-dim features
        theta_x = self.W_theta(x).unsqueeze(dim=-1)
        phi_x = self.W_phi(x).unsqueeze(dim=-1)
        phi_x = phi_x.permute(0, 2, 1)
        W_ji = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)
        # Project to low-dim, apply attention, reproject to original dim
        t = self.W_f(x).unsqueeze(dim=-1)
        t = torch.matmul(W_ji, t).squeeze()
        a_ij = self.W_g(t)
        return x + a_ij


class SimpleEdgeUpdate(MessagePassing):
    """
    Network to pass messages and update the nodes
    """

    def __init__(self, node_in_channels, node_out_channels,
                 edge_in_channels, edge_out_channels, use_attention=True):
        super().__init__(aggr='mean')

        self.use_attention = use_attention

        self.edge_update_mlp = SimpleEdgeModel(
            node_in_channels, edge_in_channels, edge_out_channels)

        self.msg_mlp = nn.Sequential(
            nn.Linear(node_in_channels + edge_out_channels, node_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(node_out_channels, node_out_channels))

        self.node_update_mlp = nn.Sequential(
            nn.Linear(node_in_channels + node_out_channels, node_out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(node_out_channels, node_out_channels)
        )

        if use_attention:
            self.att = AttentionBlock(node_out_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        # Update edge features
        edge_attr = self.edge_update_mlp(x[row], x[col], edge_attr)

        # Propogate messages
        out = self.propagate(edge_index, size=(
            x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        # Generate message to target nodes
        msg = self.msg_mlp(torch.cat([x_j, edge_attr], dim=1))
        # Decide which features to attent for each individual message
        if self.use_attention:
            msg = self.att(msg)
        return msg

    def update(self, aggr_out, x):
        # Update target node with aggregated messages
        out = self.node_update_mlp(torch.cat([x, aggr_out], dim=1))
        return out


class MLPGCNet(nn.Module):

    def __init__(self, node_feat_dim=1024, edge_feat_dim=1024,
                 gnn_recursion=2, droprate=0.0, pose_proj_dim=32) -> None:

        super().__init__()
        self.gnn_recursion = gnn_recursion
        self.droprate = droprate
        self.pose_proj_dim = pose_proj_dim

        # Setup the feature extractor
        self.feature_extractor = resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                              node_feat_dim)

        # Project relative robot displacement
        self.proj_rel_disp = nn.Linear(6, self.pose_proj_dim)
        # Intial edge projection layer
        self.proj_init_edge = nn.Linear(
            2 * node_feat_dim + pose_proj_dim, edge_feat_dim)

        # Setup the message passing network
        self.gnn_layer = SimpleEdgeUpdate(
            node_feat_dim, node_feat_dim, edge_feat_dim + pose_proj_dim, edge_feat_dim)

        # Setup the relative pose regression networks
        self.fc_xyz_R = nn.Linear(node_feat_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_feat_dim, 3)

        # Setup the hand-eye regression networks
        self.edge_he = nn.Linear(edge_feat_dim + pose_proj_dim, edge_feat_dim)
        self.edge_attn_he = nn.Linear(edge_feat_dim, 1)
        self.fc_xyz_he = nn.Linear(edge_feat_dim, 3)
        self.fc_wpqr_he = nn.Linear(edge_feat_dim, 3)

        init_modules = [self.feature_extractor.fc,
                        self.proj_init_edge, self.gnn_layer,
                        self.fc_xyz_R, self.fc_wpqr_R]

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def join_node_edge_feat(self, node_feat, edge_index, edge_feat):
        # Join edge features with corresponding node features
        out_feat = torch.cat(
            (node_feat[edge_index[0], ...],
             node_feat[edge_index[1], ...],
             edge_feat), dim=1)
        return out_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.feature_extractor(x)

        # Compute edge features
        rel_disp_feat = F.relu(self.proj_rel_disp(edge_attr), inplace=True)
        edge_node_feat = self.join_node_edge_feat(x, edge_index, rel_disp_feat)
        edge_feat = F.relu(self.proj_init_edge(edge_node_feat), inplace=True)

        # Graph message passing step
        for _ in range(self.gnn_recursion):
            edge_feat = torch.cat([edge_feat, rel_disp_feat], dim=-1)
            x, edge_feat = self.gnn_layer(x, edge_index, edge_feat)
            x = F.relu(x, inplace=True)
            edge_feat = F.relu(edge_feat)

        # Drop node and edge features if necessary
        if self.droprate > 0:
            # x = F.dropout(x, p=self.droprate, training=self.training)
            edge_feat = F.dropout(
                edge_feat, p=self.droprate, training=self.training)

        # Predict the relative pose between images
        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        # Process edge features for regressing hand-eye parameters
        edge_he_feat = self.edge_he(
            torch.cat([edge_feat, rel_disp_feat], dim=-1))
        # Calculate the attention weights over the edges
        edge_he_logits = self.edge_attn_he(
            edge_he_feat).squeeze().repeat(data.num_graphs, 1)
        edge_graph_ids = data.batch[data.edge_index[0].cpu().numpy()]
        num_graphs = torch.arange(
            0, data.num_graphs).view(-1, 1).to(edge_graph_ids.device)
        edge_he_logits[num_graphs != edge_graph_ids] = -torch.inf
        edge_he_attn = F.softmax(edge_he_logits, dim=-1)
        # Apply attention
        edge_he_aggr = torch.matmul(edge_he_attn, edge_he_feat)

        # Predict the hand-eye parameters
        xyz_he = self.fc_xyz_he(edge_he_aggr)
        wpqr_he = self.fc_wpqr_he(edge_he_aggr)

        return torch.cat((xyz_he, wpqr_he), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index
