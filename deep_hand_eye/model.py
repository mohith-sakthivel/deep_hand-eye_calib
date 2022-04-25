import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from torch_geometric.nn import MessagePassing


class SimpleEdgeModel(nn.Module):
    """
    Network to perform autoregressive edge update during Neural message passing
    """

    def __init__(self, in_channels, edge_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * in_channels + edge_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, source, target, edge_attr):
        out = torch.cat([source, target, edge_attr], dim=1).contiguous()
        out = self.edge_mlp(out)
        return out


class AttentionBlock(nn.Module):
    """
    Network to apply non-local attention
    """

    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.W_theta = nn.Linear(in_channels, in_channels // 8)
        self.W_phi = nn.Linear(in_channels, in_channels // 8)

        self.W_f = nn.Linear(in_channels, in_channels // 8)
        self.W_g = nn.Linear(in_channels // 8, in_channels)

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        phi_x = self.W_phi(x).view(batch_size, out_channels // 8, 1)
        theta_x = self.W_theta(x).view(batch_size, out_channels // 8, 1)
        theta_x = theta_x.permute(0, 2, 1)
        W_ji = F.softmax(torch.matmul(phi_x, theta_x), dim=-1)

        t = self.W_f(x).view(batch_size, out_channels // 8, 1)
        t = torch.matmul(W_ji, t)
        t = t.view(batch_size, out_channels // 8)
        a_ij = self.W_g(t)
        return x + a_ij


class SimpleConvEdgeUpdate(MessagePassing):
    """
    Network to pass messages and update the nodes
    """

    def __init__(self, in_channels, edge_channels, out_channels, use_attention=True):
        super().__init__(aggr='mean')

        self.msg_mlp = nn.Sequential(
            nn.Linear(in_channels + edge_channels, out_channels),
            nn.eLU(),
            nn.Linear(out_channels, out_channels))

        self.node_update_mlp = nn.Sequential(
            nn.Linear(2 * in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )

        self.edge_update_mlp = SimpleEdgeModel(
            in_channels, edge_channels, edge_channels)

        if use_attention:
            self.att = AttentionBlock(in_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.edge_update_mlp(x[row], x[col], edge_attr)

        # x has shape [N, in_channels] and edge_index has shape [2, E]
        out = self.propagate(edge_index, size=(
            x.size(0), x.size(0)), x=x, edge_attr=edge_attr)
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr):
        msg = self.msg_mlp(torch.cat([x_j, edge_attr], dim=1))
        msg = self.att(msg)
        return msg

    def update(self, aggr_out, x):
        out = self.node_update_mlp(torch.cat([x, aggr_out], dim=1))
        return out


class GCNet(nn.Module):

    def __init__(self, feat_dim=1024, edge_feat_dim=2048, node_feat_dim=2048,
                 gnn_recursion=2, droprate=0.0) -> None:

        super().__init__()
        self.gnn_recursion = gnn_recursion
        self.droprate = droprate

        # setup the feature extractor
        self.feature_extractor = resnet34(pretrained=True)
        self.feature_extractor.avgpool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor.fc = nn.Linear(self.feature_extractor.fc.in_features,
                                              feat_dim)

        # intial edge project layer
        self.proj_init_edge = nn.Linear(2 * feat_dim, edge_feat_dim)

        # setup the message passing network
        self.gnn_layer = SimpleConvEdgeUpdate(
            node_feat_dim, edge_feat_dim, node_feat_dim)

        # setup the absolute(?) pose regression networks
        self.fc_xyz = nn.Linear(node_feat_dim, 3)
        self.fc_wpqr = nn.Linear(node_feat_dim, 3)

        # setup the relative(?) pose regression networks
        self.fc_xyz_R = nn.Linear(node_feat_dim, 3)
        self.fc_wpqr_R = nn.Linear(node_feat_dim, 3)

        init_modules = [self.feature_extractor.fc,
                        self.proj_init_edge, self.gnn_layer,
                        self.fc_xyz, self.fc_wpqr,
                        self.fc_xyz_R, self.fc_wpqr_R]

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def get_edge_node_features(self, x, edge_index):
        # cannot follow this ????
        edge_feat = torch.cat(
            (x[torch.min(edge_index, 0)[0], ...],
             x[torch.max(edge_index, 0)[0], ...]),
            dim=1)
        # torch.testing.assert_allclose(edge_feat, edge_feat2)
        return edge_feat

    def compute_RP(self, p, edge_index):
        num_edges = edge_index.size(1)
        RP = torch.zeros((num_edges, p.size(1)),
                         requires_grad=True).to(self.device)

        for i in range(num_edges):
            nodes = edge_index[:, i]
            node_s = nodes[0]
            node_t = nodes[1]
            RP[i] = p[node_s] - p[node_t]

        return RP

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = x.view(x.size(0), 3, self.input_img_height, -1).contiguous()

        x = self.feature_extractor(x)

        # Compute edge features
        edge_node_feat = self.get_edge_node_features(x, edge_index)
        edge_feat = self.proj_init_edge(edge_node_feat)
        edge_feat = F.relu(edge_feat)

        for r in range(self.gnn_recursion):
            if r == 0:
                x, edge_feat = self.gnn_layer(x, edge_index, edge_feat)
                x = F.relu(x)
                edge_feat = F.relu(edge_feat)
            else:
                x, edge_feat = self.gnn_layer(x, edge_index, edge_feat)
                x = F.relu(x)
                edge_feat = F.relu(edge_feat)

        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
            edge_feat = F.dropout(edge_feat, p=self.droprate)

        xyz = self.fc_xyz(x)
        wpqr = self.fc_wpqr(x)

        xyz_R = self.fc_xyz_R(edge_feat)
        wpqr_R = self.fc_wpqr_R(edge_feat)

        return torch.cat((xyz, wpqr), 1), torch.cat((xyz_R, wpqr_R), 1), edge_index
