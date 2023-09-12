# -*- coding:utf-8 -*-
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import GATConv, GCNConv, ChebConv, SAGEConv
from torch_geometric.utils import sort_edge_index


class GCN(torch.nn.Module):
    def __init__(self, num_node_features, h_feats, num_classes):
        super(GCN, self).__init__()
        self.use_edge_weight = False
        self.conv1 = GCNConv(num_node_features, h_feats)
        self.conv2 = GCNConv(h_feats, num_classes)

    def set_use_edge_weight(self, use_edge_weight):
        self.use_edge_weight = use_edge_weight

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if self.use_edge_weight:
            x, _ = self.conv1(x, edge_index, edge_weight=edge_weight)  # In the default gcn_conv.py file, modify the return values of the function forward(), out, edge_ Weight
            x = F.relu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x, _ = self.conv2(x, edge_index, edge_weight=edge_weight)
        else:
            x, out_edge_weight = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.6, training=self.training)
            x, out_edge_weight = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class ChebNet(torch.nn.Module):
    def __init__(self, num_node_features, h_feats, num_classes, k):
        super(ChebNet, self).__init__()
        self.use_edge_weight = False
        self.conv1 = ChebConv(num_node_features, h_feats, K=k)
        self.conv2 = ChebConv(h_feats, num_classes, K=k)

    def set_use_edge_weight(self, use_edge_weight):
        self.use_edge_weight = use_edge_weight

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        if self.use_edge_weight:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index, edge_weight)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.dropout(x, p=0.6, training=self.training)
            x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats, aggr="mean", normalize=True)
        self.conv2 = SAGEConv(h_feats, out_feats, aggr="mean", normalize=True)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


class GAT(torch.nn.Module):
    def __init__(self, in_feats, h_feats, out_feats, heads):
        super(GAT, self).__init__()
        self.conv1 = GATConv(in_feats, h_feats, heads=heads, concat=True)
        self.conv2 = GATConv(heads*h_feats, out_feats, heads=heads, concat=True)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        # x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        # x = F.dropout(x, p=0.6, training=self.training)
        # x = self.conv2(x, edge_index, edge_attr=edge_attr)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
