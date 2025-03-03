import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphSAGEConv, GINConv, global_mean_pool

class GCNLayer(nn.Module):
    """Graph Convolutional Network Layer using PyG"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GCNConv(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))

class GATLayer(nn.Module):
    """Graph Attention Network Layer using PyG"""
    def __init__(self, in_channels, out_channels, heads=4):
        super().__init__()
        self.conv = GATConv(in_channels, out_channels, heads=heads, concat=True)
        self.linear = nn.Linear(out_channels * heads, out_channels)
        
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return F.elu(self.linear(x))

class GraphSAGELayer(nn.Module):
    """GraphSAGE Layer using PyG"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = GraphSAGEConv(in_channels, out_channels)
        
    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))

class GINLayer(nn.Module):
    """Graph Isomorphism Network Layer using PyG"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        nn1 = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(),
            nn.Linear(out_channels, out_channels)
        )
        self.conv = GINConv(nn1)
        
    def forward(self, x, edge_index):
        return F.relu(self.conv(x, edge_index))

class MultiLayerGNN(nn.Module):
    """Multi-layer GNN with configurable layers"""
    def __init__(self, layer_type, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        if layer_type == "gcn":
            self.layers.append(GCNLayer(in_channels, hidden_channels))
        elif layer_type == "gat":
            self.layers.append(GATLayer(in_channels, hidden_channels))
        elif layer_type == "graphsage":
            self.layers.append(GraphSAGELayer(in_channels, hidden_channels))
        elif layer_type == "gin":
            self.layers.append(GINLayer(in_channels, hidden_channels))
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")
        
        # Hidden layers
        for _ in range(num_layers - 2):
            if layer_type == "gcn":
                self.layers.append(GCNLayer(hidden_channels, hidden_channels))
            elif layer_type == "gat":
                self.layers.append(GATLayer(hidden_channels, hidden_channels))
            elif layer_type == "graphsage":
                self.layers.append(GraphSAGELayer(hidden_channels, hidden_channels))
            elif layer_type == "gin":
                self.layers.append(GINLayer(hidden_channels, hidden_channels))
        
        # Output layer
        self.out_proj = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, x, edge_index, batch=None):
        for layer in self.layers:
            x = layer(x, edge_index)
        
        if batch is not None:
            x = global_mean_pool(x, batch)  # Graph-level pooling
        
        return self.out_proj(x)