from typing import List, Optional
from typing import Union, Tuple
import torch

from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.conv import GATConv
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.inits import glorot, zeros, reset
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor

class TADGAT_Conv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        edge_channel: int,
        heads: int = 1,
        num_etype: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        activation = None,
        residual: bool = False,
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
        self.fill_value = fill_value
        self.activation = activation

        self.edge_att_emb = nn.Parameter(Tensor(num_etype, heads, out_channels))
        self.edge_out_emb = nn.Parameter(Tensor(num_etype, heads, out_channels))

        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels,
                                  bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False,
                                  weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False,
                                  weight_initializer='glorot')

        # self.lin_short = Linear(in_channels[0], out_channels * heads, bias=False, weight_initializer='glorot')

        # The learnable parameters to compute attention coefficients:
        # TODO: separate the attention vector of different factors
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_e = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_time = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_time_2 = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_geo = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_geo_2 = nn.Parameter(torch.Tensor(1, heads, out_channels))

        # self.short_term_encoder = MultiHeadAttention(heads * out_channels, 360)
        # self.att_short_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        # self.emb_bias_proj = nn.Sequential(nn.Linear(out_channels*heads*2, out_channels*heads), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(out_channels*heads, out_channels * heads))

        if residual:
            self.lin_res = Linear(self.in_channels[0], heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('lin_res', None)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_dst.reset_parameters()
        self.lin_src.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        glorot(self.att_dst)
        glorot(self.att_src)
        glorot(self.att_e)
        glorot(self.edge_att_emb)
        glorot(self.edge_out_emb)
        zeros(self.bias)

        glorot(self.att_geo)
        glorot(self.att_geo_2)
        glorot(self.att_time)
        glorot(self.att_time_2)

        # self.lin_short.reset_parameters()
        # glorot(self.att_short_src)
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, edge_type: OptTensor, att_bias: List[Tensor], emb_bias: List[Tensor],
                att_bias_dist: List[Tensor]=[], emb_bias_dist: List[Tensor]=[], size: Size=None, **kwargs):
                #  seq_ids: OptTensor = None, short_emb: OptTensor = None, size: Size = None, **kwargs):
        H, C = self.heads, self.out_channels
        # TODO: add short-term preference for POI-POI edges
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            res_x = x
            # x_short = self.lin_short(x).view(-1, H, C)
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else: 
            x_src, x_dst = x
            res_x = x_dst
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            # x_short = self.lin_short(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)

        # long-term preference modeling
        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)
        # if short_emb is not None:
        #     seq_id, seq_mask, time_diff = short_emb
        #     seq_emb = x[0][seq_id].view(-1, H * C)
        #     short_emb = self.short_term_encoder(seq_emb, seq_emb, seq_emb, seq_mask, time_diff).view(-1, H, C)

            # short_emb = self.lin_short(short_emb).view(-1, H, C)
            # alpha_short = (short_emb * self.att_short_src).sum(-1)
        # else:
        #     alpha_short = None

        out = self.propagate(edge_index, x=x, edge_type=edge_type, alpha=alpha, att_bias=att_bias,
                            emb_bias=emb_bias, att_bias_dist=att_bias_dist, emb_bias_dist=emb_bias_dist, size=size)
                            #  emb_bias=emb_bias, short_emb=short_emb, seq_ids=seq_ids, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        # assert self._alpha_short is not None
        # self._alpha_short = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.lin_res is not None:
            res_val = self.lin_res(res_x)
            out = res_val + out

        if self.bias is not None:
            out += self.bias
        if self.activation:
            out = self.activation(out)
        
        return out
    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, edge_type: OptTensor, att_bias: List[Tensor], emb_bias: List[Tensor], 
                att_bias_dist: List[Tensor], emb_bias_dist: List[Tensor], index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
                #  x_short: OptTensor, short_emb: OptTensor, seq_ids: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:

        # Long-term preference
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = alpha + (att_bias[0] * self.att_time).sum(-1)
        alpha = alpha + (att_bias[1] * self.att_time_2).sum(-1)
        if len(att_bias_dist) != 0:
            cond = (edge_type == 0) | (edge_type == 2) | (edge_type == 3)
            alpha[cond] = alpha[cond] + (att_bias_dist[0] * self.att_geo).sum(-1)
            # alpha[cond] = alpha[cond] + (att_bias_dist[1] * self.att_geo_2).sum(-1)

        e_emb = self.edge_att_emb[edge_type]
        alpha_e = (e_emb * self.att_e).sum(-1)
        alpha = alpha + alpha_e

        # if seq_ids is not None:
        #     short_emb = x_short[short_emb]
        #     alpha_short = (short_emb * self.att_short_src).sum(-1)
        #     bias_idx = seq_ids != -1
        #     alpha[bias_idx] = alpha[bias_idx] + alpha_short[seq_ids[bias_idx]]

        # if short_emb is not None:
        #     alpha_short = (short_emb * self.att_short_src).sum(-1)
        #     bias_idx = seq_ids != -1
        #     alpha_short_bias = alpha_short[seq_ids[bias_idx]]
        #     alpha[bias_idx] = alpha[bias_idx] + alpha_short_bias

            # emb_bias_input = torch.cat([x_j[bias_idx].view(-1, self.heads * self.out_channels),
            #                             short_emb[seq_ids[bias_idx]].view(-1, self.heads * self.out_channels)], dim=-1)
            # emb_bias_short = self.emb_bias_proj(emb_bias_input).view(-1, self.heads, self.out_channels)

        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        x_j = x_j + self.edge_out_emb[edge_type] + emb_bias[0] + emb_bias[1]
        if len(emb_bias_dist) != 0:
            cond = (edge_type == 0) | (edge_type == 2) | (edge_type == 3)
            x_j[cond] = x_j[cond] + emb_bias_dist[0] # + emb_bias_dist[1]
        return x_j * alpha.unsqueeze(-1)
        

class LGCConv(MessagePassing):

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 normalize: bool = False, **kwargs):  # yapf: disable
        super(LGCConv, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.normalize = normalize
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
        self.improved = False
        self.add_self_loops = True
        self.reset_parameters()
    
    def reset_parameters(self):
        pass

    def forward(self, x: Tensor, edge_index: Adj,
                edge_weight: OptTensor = None) -> Tensor:
        
        edge_index, edge_weight = gcn_norm(edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j


class ResGATConv(MessagePassing):

    def __init__(
        self,
        in_channels: Union[int, Tuple[int, int]],
        out_channels: int,
        heads: int = 1,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        activation = None,
        residual: bool = False,
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
        self.fill_value = fill_value
        self.activation = activation

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
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))

        if residual:
            self.lin_res = Linear(self.in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
        else:
            self.register_parameter('lin_res', None)

        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin_dst.reset_parameters()
        self.lin_src.reset_parameters()
        if self.lin_res is not None:
            self.lin_res.reset_parameters()
        glorot(self.att_dst)
        glorot(self.att_src)
        zeros(self.bias)
    
    def forward(self, x: Union[Tensor, OptPairTensor], edge_index, size: Size=None, **kwargs):
        H, C = self.heads, self.out_channels
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            res_x = x
            x_src = x_dst = self.lin_src(x).view(-1, H, C)
        else: 
            x_src, x_dst = x
            res_x = x_dst
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)
        x = (x_src, x_dst)

        alpha_src = (x_src * self.att_src).sum(dim=-1)
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)
        
        if self.lin_res is not None:
            res_val = self.lin_res(res_x)
            out = res_val + out

        if self.bias is not None:
            out += self.bias
        if self.activation:
            out = self.activation(out)
        
        return out
    
    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor, index: Tensor, ptr: OptTensor, size_i: Optional[int]) -> Tensor:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha 
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)


class FFN(nn.Module):

    def __init__(self, hid_dim, filter_dim, dp_rate=0.2):
        super(FFN, self).__init__()

        self.lin_1 = nn.Linear(hid_dim, filter_dim)
        self.lin_2 = nn.Linear(filter_dim, hid_dim)
        self.dp_layer = nn.Dropout(dp_rate)
    
    def forward(self, x):
        x = self.lin_1(x)
        x = F.elu(x)
        x = self.dp_layer(x)
        x = self.lin_2(x)
        return x

    
class MultiHeadAttention(nn.Module):

    def __init__(self, hid_dim, out_dim, ubias_num=180, dp_rate=0.3, heads=8):
        super(MultiHeadAttention, self).__init__()
        self.heads = heads
        self.att_dim = hid_dim // heads
        self.scale = self.att_dim ** -0.5

        self.Q_lin = nn.Linear(hid_dim, self.att_dim * self.heads, bias=False)
        self.K_lin = nn.Linear(hid_dim, self.att_dim * self.heads, bias=False)
        self.V_lin = nn.Linear(hid_dim, self.att_dim * self.heads, bias=False)
        self.fc = nn.Linear(self.heads * self.att_dim, out_dim, bias=False)

        self.att_bias = nn.Parameter(torch.Tensor(ubias_num+3, self.att_dim))
        self.vec_bias = nn.Parameter(torch.Tensor(ubias_num+3, self.att_dim))

        self.att_dp = nn.Dropout(dp_rate)
        self.reset_parameters()

    def reset_parameters(self):
        zeros(self.att_bias)
        glorot(self.vec_bias)
    
    def forward(self, Q, K, V, mask, time_bias):
        orig_q_size = Q.size()
        batch_size = Q.size(0)
        seq_len = Q.size(1)

        Q = self.Q_lin(Q).view(batch_size, -1, self.heads, self.att_dim)
        K = self.K_lin(K).view(batch_size, -1, self.heads, self.att_dim)
        V = self.V_lin(V).view(batch_size, -1, self.heads, self.att_dim)

        att_bias = self.att_bias[time_bias]
        val_bias = self.vec_bias[time_bias]
        att_bias = att_bias.unsqueeze(1)
        val_bias = val_bias.unsqueeze(-2)

        Q, V = Q.transpose(1, 2), V.transpose(1, 2)
        K = K.transpose(1, 2).transpose(2, 3)       # [b, h, attn, k_len]
        val_bias = val_bias.transpose(2, 3).transpose(1, 2) # [b, 1, q_len, k_len, attn]
        
        Q.mul_(self.scale)
        x = torch.matmul(Q, K)  # [b, h, q_len, k_len]
        x = x + (Q.unsqueeze(3) * att_bias).sum(-1)
        x.masked_fill_(mask.unsqueeze(1), float("-inf"))
        # x = x + att_bias.unsqueeze(1)
        x = torch.softmax(x, dim=3)
        x = self.att_dp(x)
        x = x.matmul(V) + (x.unsqueeze(-1) * val_bias).sum(-2)  # [b, h, q_len, attn]

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
        x = x.view(batch_size, -1, self.heads * self.att_dim)

        x = self.fc(x)

        # assert x.size() == orig_q_size
        return x


class Transformer_layer(nn.Module):
    
    def __init__(self, hid_dim, filter_dim, ubias_num, dp_rate=0.3, heads=8):
        super(Transformer_layer, self).__init__()
        self.mha_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.mha = MultiHeadAttention(hid_dim, ubias_num=ubias_num, dp_rate=dp_rate, heads=heads)
        self.mha_dp = nn.Dropout(dp_rate)

        self.ffn_norm = nn.LayerNorm(hid_dim, eps=1e-6)
        self.ffn = FFN(hid_dim, filter_dim, dp_rate)
        self.ffn_dp = nn.Dropout(dp_rate)

    def forward(self, x, mask, ta_bias):
        # y = self.mha_norm(x)
        # y = self.mha(y, y, y, mask, ta_bias)
        y = self.mha(x, x, x, mask, ta_bias)
        y = self.mha_dp(y)
        x = x + y
        y = self.mha_norm(x)

        # y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dp(y)
        x = x + y
        x = self.ffn_norm(x)
        return x

