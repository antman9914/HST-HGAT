import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from typing import List, Optional
from typing import Union, Tuple


class DGATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        add_self_loops: bool = True,
        edge_dim: Optional[int] = None,
        fill_value: Union[float, Tensor, str] = 'mean',
        bias: bool = True,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super().__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops
        self.edge_dim = edge_dim
        self.fill_value = fill_value

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        self.att_src = nn.Parameter(torch.Tensor(heads, out_channels, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(heads, out_channels, out_channels))

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None):

        H, C = self.heads, self.out_channels

        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
        x_src = x_dst = self.lin_src(x).view(-1, H, C)

        x = (x_src, x_dst)

        # alpha_src = (x_src * self.att_src).sum(dim=-1)
        # alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha_src = torch.einsum('aij,ijk->aik', x_src, self.att_src)
        alpha_dst = None if x_dst is None else torch.einsum('aij,ijk->aik', x_dst, self.att_dst)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias

        return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha# .unsqueeze(-1)


class STP_UDGAT(nn.Module):
    def __init__(self, in_channel, hidden_channel, user_num, item_num, heads=4, dp_rate=0.5):
        super().__init__()
        self.in_channel = in_channel
        self.hidden_channel = hidden_channel
        self.heads = heads
        self.dropout_rate = dp_rate
        self.user_num, self.poi_num = user_num, item_num
        self.user_emb = nn.Parameter(torch.Tensor(user_num, in_channel))
        self.item_emb = nn.Parameter(torch.Tensor(item_num, in_channel))
        self.p_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.s_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.t_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.f_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.s_rw_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.t_rw_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.f_rw_dgat = DGATConv(in_channel, hidden_channel, heads)
        self.u_dgat = DGATConv(in_channel, hidden_channel, heads)

        self.a_rw_fc = nn.Linear(2*hidden_channel*heads, hidden_channel*heads)
        self.pp_stp_fc = nn.Linear(2*hidden_channel*heads, hidden_channel*heads)
        self.final_fc = nn.Linear(2*hidden_channel*heads, self.poi_num)

        self.reset_parameter()
    
    def reset_parameter(self):
        glorot(self.user_emb)
        glorot(self.item_emb)

    def forward(self, s_graph, t_graph, f_graph, s_rw_graph, t_rw_graph, f_rw_graph, pp_graph, u_eindex, anchors):
        s_x = self.item_emb[s_graph.node_idx]
        t_x = self.item_emb[t_graph.node_idx]
        f_x = self.item_emb[f_graph.node_idx]
        s_rw_x = self.item_emb[s_rw_graph.node_idx]
        t_rw_x = self.item_emb[t_rw_graph.node_idx]
        f_rw_x = self.item_emb[f_rw_graph.node_idx]
        pp_x = self.item_emb[pp_graph.node_idx]
        u_x = self.user_emb

        s_x = self.s_dgat(s_x, s_graph.edge_index)[s_graph.center_node]
        t_x = self.t_dgat(t_x, t_graph.edge_index)[t_graph.center_node]
        f_x = self.f_dgat(f_x, f_graph.edge_index)[f_graph.center_node]
        s_rw_x = self.s_rw_dgat(s_rw_x, s_rw_graph.edge_index)[s_rw_graph.center_node]
        t_rw_x = self.t_rw_dgat(t_rw_x, t_rw_graph.edge_index)[t_rw_graph.center_node]
        f_rw_x = self.f_rw_dgat(f_rw_x, f_rw_graph.edge_index)[f_rw_graph.center_node]
        pp_x = self.p_dgat(pp_x, pp_graph.edge_index)[pp_graph.center_node]
        u_x = self.u_dgat(u_x, u_eindex)[anchors]

        stp_1 = (s_x + t_x + f_x) / 3
        stp_rw = (s_rw_x + t_rw_x + f_rw_x) / 3
        stp = torch.cat([stp_1, stp_rw], dim=-1)
        stp = self.a_rw_fc(stp)
        y = self.pp_stp_fc(torch.cat([pp_x, stp], dim=-1))
        final_repr = torch.cat([y, u_x], dim=-1)
        final_repr = F.dropout(final_repr, p=self.dropout_rate, training=self.training)
        logits = self.final_fc(final_repr)
        return logits