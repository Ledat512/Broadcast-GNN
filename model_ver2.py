import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch.nn import Linear

class BroadcastGNN(torch.nn.Module):
    def __init__(self, in_dim=6, hidden=32, heads=4):
        super().__init__()
        self.input_proj = Linear(in_dim, hidden)
        self.conv1 = GATConv(in_dim, hidden, heads=heads, concat=True)
        self.conv2 = GATConv(hidden * heads, hidden, heads=1, concat=False)
        self.head  = Linear(hidden, 1)

    def forward(self, x, edge_index):
        x0 = F.relu(self.input_proj(x))          # residual branch

        h = F.relu(self.conv1(x, edge_index))
        h = F.dropout(h, p=0.3, training=self.training)
        h = F.relu(self.conv2(h, edge_index))
        h = h + x0                                # skip connection

        return self.head(h).squeeze(-1)

if __name__ == "__main__":
    x          = torch.rand(50, 6)
    edge_index = torch.randint(0, 50, (2, 308))
    model  = BroadcastGNN(in_dim=6, hidden=32, heads=4)
    logits = model(x, edge_index)
    print(f"Output logits: {logits.shape}")
    print(f"Scores       : {torch.sigmoid(logits)[:5]}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Tổng params  : {total}")
