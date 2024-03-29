import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing

from deep_hand_eye.resnet import resnet34
from deep_hand_eye.utils import unbatch


class SimpleEdgeModel(nn.Module):
    """
    Network to perform autoregressive edge update during Neural message passing
    """

    def __init__(self, node_channels, edge_in_channels, edge_out_channels):
        super().__init__()
        self.node_channels = node_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels

        self.edge_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=2 * node_channels + edge_in_channels,
                out_channels=edge_out_channels,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=edge_out_channels,
                out_channels=edge_out_channels,
                kernel_size=3,
                stride=1,
                padding=1)
        )

    def forward(self, source, target, edge_attr):
        out = torch.cat([edge_attr, source, target], dim=1).contiguous()
        out = self.edge_cnn(out)
        return out


class AttentionBlock(nn.Module):
    """
    Network to apply non-local attention
    """

    def __init__(self, in_channels, N=8):
        super().__init__()
        self.N = N
        self.W_theta = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // N,
            kernel_size=3
        )
        self.W_phi = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // N,
            kernel_size=3
        )

        self.W_f = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels // N,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.W_g = nn.Conv2d(
            in_channels=in_channels // N,
            out_channels=in_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        batch_size = x.size(0)
        out_channels = x.size(1)

        theta_x = self.W_theta(x)
        theta_x = F.adaptive_avg_pool2d(theta_x, 1).squeeze(dim=-1)
        phi_x = self.W_phi(x)
        phi_x = F.adaptive_avg_pool2d(phi_x, 1).squeeze(dim=-1)
        phi_x = phi_x.permute(0, 2, 1)
        W_ji = F.softmax(torch.matmul(theta_x, phi_x), dim=-1)

        t = self.W_f(x)
        t_shape = t.shape
        t = t.view(batch_size, out_channels // self.N, -1)
        t = torch.matmul(W_ji, t)
        t = t.view(*t_shape)
        a_ij = self.W_g(t)
        return x + a_ij


class SimpleConvEdgeUpdate(MessagePassing):
    """
    Network to pass messages and update the nodes
    """

    def __init__(self, node_in_channels, node_out_channels,
                 edge_in_channels, edge_out_channels, use_attention=True):
        super().__init__(aggr='mean')

        self.use_attention = use_attention

        self.edge_update_cnn = SimpleEdgeModel(
            node_channels=node_in_channels,
            edge_in_channels=edge_in_channels,
            edge_out_channels=edge_out_channels
        )

        self.msg_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=node_in_channels + edge_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=node_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            )
        )

        self.node_update_cnn = nn.Sequential(
            nn.Conv2d(
                in_channels=node_in_channels + node_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=node_out_channels,
                out_channels=node_out_channels,
                kernel_size=3,
                padding=1,
                stride=1
            )
        )

        if self.use_attention:
            self.att = AttentionBlock(node_out_channels)

    def forward(self, x, edge_index, edge_attr):
        row, col = edge_index
        edge_attr = self.edge_update_cnn(x[row], x[col], edge_attr)

        # x has shape [N, in_channels] and edge_index has shape [2, E]
        H, W = x.shape[-2:]
        num_nodes, num_edges = x.shape[0], edge_attr.shape[0]
        out = self.propagate(
            edge_index=edge_index,
            size=(x.size(0), x.size(0)),
            x=x.view(num_nodes, -1),
            edge_attr=edge_attr.view(num_edges, -1),
            H=H,
            W=W
        )
        return out, edge_attr

    def message(self, x_i, x_j, edge_attr, H, W):
        num_edges = edge_attr.shape[0]
        msg = self.msg_cnn(torch.cat(
            [x_j.view(num_edges, -1, H, W), edge_attr.view(num_edges, -1, H, W)], dim=-3))
        if self.use_attention:
            msg = self.att(msg)
        return msg.view(num_edges, -1)

    def update(self, aggr_out, x, H, W):
        num_nodes = x.shape[0]
        out = self.node_update_cnn(torch.cat(
            [x.view(num_nodes, -1, H, W), aggr_out.view(num_nodes, -1, H, W)], dim=-3))
        return out


class EdgeSelfAttention(nn.Module):

    def __init__(self, input_dim, feat_dim, key_dim=None, query_dim=None, value_dim=None):

        super().__init__()

        self.feat_dim = feat_dim
        self.value_dim = value_dim if value_dim is not None else feat_dim
        self.key_dim = key_dim if key_dim is not None else feat_dim
        self.query_dim = query_dim if query_dim is not None else feat_dim

        self.value_net = self.make_conv_block(input_dim, self.feat_dim, self.value_dim)
        self.key_net = self.make_conv_block(input_dim, self.feat_dim, self.key_dim)
        self.query_net = self.make_conv_block(input_dim, self.feat_dim, self.query_dim)

    @staticmethod
    def make_conv_block(input_dim, feat_dim, output_dim):
        block = nn.Sequential(
            nn.Conv2d(input_dim, feat_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, output_dim, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        return block

    def forward(self, edge_feat, edge_graph_id):
        query = self.query_net(edge_feat)
        key = self.key_net(edge_feat)
        value = self.value_net(edge_feat)
        feat_shape = query.shape[-3:]

        query = unbatch(query.flatten(start_dim=-3), edge_graph_id)
        key = unbatch(key.flatten(start_dim=-3), edge_graph_id)
        value = unbatch(value.flatten(start_dim=-3), edge_graph_id)

        output = []
        for q, k, v in zip(query, key, value):
            attn = F.softmax(torch.matmul(q, k.T) / np.sqrt(2 * self.feat_dim), dim=-1)
            output.append(torch.matmul(attn, v).view(-1, *feat_shape))
        return torch.concat(output, dim=0)


class GCNet(nn.Module):

    def __init__(self, node_feat_dim=512, edge_feat_dim=512,
                 gnn_recursion=2, droprate=0.0, pose_proj_dim=32,
                 rel_pose=True) -> None:

        super().__init__()
        self.gnn_recursion = gnn_recursion
        self.droprate = droprate
        self.pose_proj_dim = pose_proj_dim
        self.rel_pose = rel_pose
        self.edge_feat_dim = edge_feat_dim

        # Setup the feature extractor
        self.feature_extractor = resnet34(pretrained=True)
        self.process_feat = nn.Conv2d(
            in_channels=512,
            out_channels=node_feat_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Project relative robot displacement
        self.proj_rel_disp = nn.Linear(6, self.pose_proj_dim)
        # Intial edge project layer
        self.proj_init_edge = nn.Conv2d(
            in_channels=2 * node_feat_dim + pose_proj_dim,
            out_channels=edge_feat_dim,
            kernel_size=3,
            stride=1,
            padding=1
        )

        # Setup the message passing network
        self.gnn_layer = SimpleConvEdgeUpdate(
            node_feat_dim, node_feat_dim, edge_feat_dim + pose_proj_dim, edge_feat_dim)

        # Setup the relative pose regression networks
        if self.rel_pose:
            self.edge_R = nn.Sequential(
                nn.Conv2d(
                    in_channels=edge_feat_dim,
                    out_channels=edge_feat_dim // 2,
                    kernel_size=3
                ),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    in_channels=edge_feat_dim // 2,
                    out_channels=edge_feat_dim // 2,
                    kernel_size=3
                ),
                nn.ReLU(inplace=True)
            )
            self.xyz_R = nn.Conv2d(
                in_channels=edge_feat_dim // 2,
                out_channels=3,
                kernel_size=3
            )
            self.wpqr_R = nn.Conv2d(
                in_channels=edge_feat_dim // 2,
                out_channels=3,
                kernel_size=3
            )

        # Setup the hand-eye pose regression networks
        # Self-attention for edges to transfer information
        self.edge_self_attn_he = EdgeSelfAttention(
            input_dim=edge_feat_dim + pose_proj_dim + 2 * node_feat_dim,
            feat_dim=edge_feat_dim // 2
        )

        # Attention to combine information from all edges
        self.edge_attn_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2,
            out_channels=1,
            kernel_size=3,
            stride=1,
            padding=0
        )

        # Setup Regression heads
        self.xyz_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2,
            out_channels=3,
            kernel_size=3
        )
        self.wpqr_he = nn.Conv2d(
            in_channels=edge_feat_dim // 2,
            out_channels=3,
            kernel_size=3
        )

        # Initialize networks
        init_modules = [
            self.proj_rel_disp, self.process_feat, self.proj_init_edge, self.gnn_layer,
            self.edge_self_attn_he, self.edge_attn_he, self.xyz_he, self.wpqr_he
        ]
        if self.rel_pose:
            init_modules.extend([self.edge_R, self.xyz_R, self.wpqr_R])

        for m in init_modules:
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def join_node_edge_feat(self, node_feat, edge_index, edge_feat_list):
        # Join node features of a corresponding edge
        out_feat = torch.cat(
            (node_feat[edge_index[0], ...],
             node_feat[edge_index[1], ...],
             *edge_feat_list), dim=-3)
        return out_feat

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.feature_extractor(x)
        x = self.process_feat(x)

        # Compute edge features
        rel_disp_feat = F.relu(self.proj_rel_disp(edge_attr), inplace=True)
        rel_disp_feat = rel_disp_feat.view(*rel_disp_feat.shape, 1, 1).expand(-1, -1, *x.shape[-2:])
        edge_node_feat = self.join_node_edge_feat(x, edge_index, [rel_disp_feat])
        edge_feat = F.relu(self.proj_init_edge(edge_node_feat), inplace=True)

        # Graph message passing step
        for _ in range(self.gnn_recursion):
            edge_feat = torch.cat([edge_feat, rel_disp_feat], dim=-3)
            x, edge_feat = self.gnn_layer(x, edge_index, edge_feat)
            x = F.relu(x)
            edge_feat = F.relu(edge_feat)

        # Drop node and edge features if necessary
        if self.droprate > 0:
            # x = F.dropout(x, p=self.droprate, training=self.training)
            edge_feat = F.dropout(
                edge_feat, p=self.droprate, training=self.training)

        # Predict the relative pose between images
        if self.rel_pose:
            edge_R_feat = self.edge_R(edge_feat)
            xyz_R = self.xyz_R(edge_R_feat).squeeze()
            wpqr_R = self.wpqr_R(edge_R_feat).squeeze()
            rel_pose_out = torch.cat((xyz_R, wpqr_R), 1)
        else:
            rel_pose_out = None

        # Process edge features for regressing hand-eye parameters
        edge_he_feat = self.join_node_edge_feat(x, edge_index, [edge_feat, rel_disp_feat])
        # Find the graph id of each edge using the source node
        edge_graph_ids = data.batch[data.edge_index[0].cpu().numpy()]
        # Perform self-attention of edge features
        edge_he_feat = self.edge_self_attn_he(edge_he_feat, edge_graph_ids)

        # Calculate the attention weight over the edges
        edge_he_logits = self.edge_attn_he(edge_he_feat).squeeze().repeat(data.num_graphs, 1)
        num_graphs = torch.arange(0, data.num_graphs).view(-1, 1).to(edge_graph_ids.device)
        edge_he_logits[num_graphs != edge_graph_ids] = -torch.inf
        edge_he_attn = F.softmax(edge_he_logits, dim=-1)

        # Apply attention
        num_edges, feat_shape = edge_he_feat.shape[0], edge_he_feat.shape[1:]
        edge_he_aggr = torch.matmul(edge_he_attn, edge_he_feat.view(num_edges, -1))
        edge_he_aggr = edge_he_aggr.view(data.num_graphs, *feat_shape)

        # Predict the hand-eye parameters
        xyz_he = self.xyz_he(edge_he_aggr).reshape(-1, 3)
        wpqr_he = self.wpqr_he(edge_he_aggr).reshape(-1, 3)

        return torch.cat((xyz_he, wpqr_he), 1), rel_pose_out, edge_index
