# dynamic triplet-attentive mechanism
from typing import Union, Optional
from torch_geometric.typing import (Adj, Size, OptTensor, PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_sparse import SparseTensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import softmax
from torch_geometric.nn.inits import glorot


class EdgeGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: int,
                 out_channels: int, edge_dim: int, heads: int = 1, 
                 negative_slope: float = 0.2, dropout: float = 0.,
                 bias: bool = True, share_weights: bool = False,
                 **kwargs):
        super(EdgeGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights

        self.lin_l = Linear(in_channels, heads * out_channels, bias=bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(in_channels, heads * out_channels, bias=bias)

        self.att = Parameter(torch.Tensor(1, heads, out_channels))

        self.lin_edge = Linear(edge_dim, heads * out_channels, bias=bias)

        self.lin_out = Linear(heads * out_channels, out_channels)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_edge.weight)
        glorot(self.lin_out.weight)
        glorot(self.att)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj, edge_attr,
                size: Size = None, return_attention_weights: bool = None):

        H, C = self.heads, self.out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None

        edge_attr = self.lin_edge(edge_attr).view(-1, H, C)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, x=(x_l, x_r), edge_attr=edge_attr, size=size)

        alpha = self._alpha
        self._alpha = None

        out = out.view(-1, self.heads * self.out_channels)
        out = self.lin_out(out)

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_index.set_value(alpha, layout='coo')
        else:
            return out

    def message(self, x_j: Tensor, x_i: Tensor, edge_attr,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = x_i + x_j + edge_attr
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * edge_attr * alpha.unsqueeze(-1)

if __name__ == '__main__':
    from dataset import *
    from utils import Complete
    import torch_geometric.transforms as T
    from torch_geometric.data import Batch

    transform = T.Compose([SpecifyTarget(0), Complete(), T.Distance(norm=True)])
    train_dataset, valid_dataset, test_dataset = load_dataset(path="/data0/linan/data/Properties/qm9", transform=transform)
    data = Batch.from_data_list([train_dataset[0], train_dataset[1]])
    print(data)
    node_dim = train_dataset.num_node_features
    edge_dim = train_dataset.num_edge_features

    net = EdgeGATConv(node_dim, 32, edge_dim, heads=4)
    x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
    res = net(x, edge_index, edge_attr)

# %%