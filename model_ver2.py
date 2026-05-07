import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

class BroadcastGNN(torch.nn.Module):
    def __init__(self, in_dim=6, hidden=32):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden)
        self.conv1 = GCNConv(hidden, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.head = Linear(hidden, 1)

    def forward(self, x, edge_index):
        x0 = F.relu(self.input_proj(x))
        h = F.relu(self.conv1(x0, edge_index))
        h = F.relu(self.conv2(h, edge_index))
        h = h + x0
        return self.head(h).squeeze(-1)
