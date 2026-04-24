import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

class BroadcastGNN(torch.nn.Module):
    def __init__(self, in_dim=4, hidden=32):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden)
        self.head  = Linear(hidden, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.head(x)
        return x.squeeze(-1)

if __name__ == "__main__":
    x          = torch.rand(50, 4)
    edge_index = torch.randint(0, 50, (2, 308))

    model  = BroadcastGNN(in_dim=4, hidden=32)
    logits = model(x, edge_index)

    print(f"Input  x     : {x.shape}")        # [50, 4]
    print(f"Output logits: {logits.shape}")    # [50]
    print(f"Scores       : {torch.sigmoid(logits)[:5]}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Tổng params  : {total}")