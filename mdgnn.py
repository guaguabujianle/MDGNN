import torch
import torch.nn as nn
from torch.nn import Linear, Sequential, GRU
import torch.nn.functional as F
from torch_geometric.nn import Set2Set
from utils import NodeLevelBatchNorm
from egat import EdgeGATConv


class Block(torch.nn.Module):
    def __init__(self, dim, edge_dim, time_step=6):
        super(Block, self).__init__()
        self.time_step = time_step
        self.conv = EdgeGATConv(dim, dim, edge_dim, heads=3)  # GraphMultiHeadAttention
        self.gru = GRU(dim, dim)
        self.norm1_list = nn.ModuleList([NodeLevelBatchNorm(dim) for _ in range(time_step)])
        self.lin = Linear(dim * (time_step + 1), dim)

    def forward(self, x, edge_index, edge_attr):
        h = x.unsqueeze(0)
        features = [x]
        for i in range(self.time_step):
            m = F.celu(self.conv(x, edge_index, edge_attr))
            x, h = self.gru(m.unsqueeze(0), h)
            x = self.norm1_list[i](x.squeeze(0))
            features.append(x)

        x = self.lin(torch.cat(features, dim=-1))

        return x

class MDGNN(torch.nn.Module):
    def __init__(self, in_dim, edge_in_dim, hidden_dim=32, depth=3, out_dim=12):
        super(MDGNN, self).__init__()
        self.depth = depth
        self.lin0 = Linear(in_dim, hidden_dim)
        self.convs = Sequential(*[
            Block(hidden_dim, edge_in_dim)
            for i in range(depth)
        ])

        self.set2set = Set2Set(hidden_dim, processing_steps=3)
        self.lin_share = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
        nn.CELU())
        self.lin_task = nn.ModuleList(
            [nn.Linear(hidden_dim, 1) for _ in range(out_dim)])


    def forward(self, data):
        x = F.celu(self.lin0(data.x))
        for conv in self.convs:
            x = x + conv(x, data.edge_index, data.edge_attr)

        x = self.set2set(x, data.batch)
        x = self.lin_share(x)
        x = torch.cat([task(x) for task in self.lin_task], dim=-1) 

        return x


# %%
if __name__ == '__main__':
    from dataset import *
    from utils import Complete
    import torch_geometric.transforms as T
    from torch_geometric.data import Batch

    transform = T.Compose([Complete(), T.Distance(norm=True)])
    train_dataset, valid_dataset, test_dataset = load_dataset(path="/data0/linan/data/Properties/qm9", transform=transform)
    data = Batch.from_data_list([train_dataset[0], train_dataset[1]])
    print(data)
    node_dim = train_dataset.num_node_features
    edge_dim = train_dataset.num_edge_features

    net = MDGNN(node_dim, edge_dim)
    res = net(data)
    print(res.shape)

    print(net.lin_out)
    
# %%
