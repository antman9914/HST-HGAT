import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import init
from torch.nn.parameter import Parameter
from torch_geometric.nn.inits import glorot


class STGNCell(nn.Module):
    """
    A Spatial-Temporal Long Short Term Memory (ST-LSTM) cell.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTMCell(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hc = (torch.randn(3, 20), torch.randn(3, 20))
        >>> output = []
        >>> for i in range(6):
        >>>     hc = st_lstm(input_l[i], input_s[i], input_q[i], hc)
        >>>     output.append(hc[0])
    """

    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STGNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.w_i = Parameter(torch.Tensor(hidden_size, 2 * input_size))
        self.w_f = Parameter(torch.Tensor(hidden_size, 2 * input_size))
        self.w_c = Parameter(torch.Tensor(hidden_size, 2 * input_size))
        self.w_o = Parameter(torch.Tensor(hidden_size, 2 * input_size))

        self.w_xs1 = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_xs2 = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_s1 = Parameter(torch.Tensor(hidden_size, 1))
        self.w_s2 = Parameter(torch.Tensor(hidden_size, 1))
        self.w_so = Parameter(torch.Tensor(hidden_size, 1))
        self.w_xq1 = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_xq2 = Parameter(torch.Tensor(hidden_size, input_size))
        self.w_q1 = Parameter(torch.Tensor(hidden_size, 1))
        self.w_q2 = Parameter(torch.Tensor(hidden_size, 1))
        self.w_qo = Parameter(torch.Tensor(hidden_size, 1))

        self.b_i = Parameter(torch.Tensor(hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        self.b_c = Parameter(torch.Tensor(hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))
        self.b_s_1 = Parameter(torch.Tensor(hidden_size))
        self.b_s_2 = Parameter(torch.Tensor(hidden_size))
        self.b_q_1 = Parameter(torch.Tensor(hidden_size))
        self.b_q_2 = Parameter(torch.Tensor(hidden_size))
        # else:
        #     self.register_parameter('b_ih', None)
        #     self.register_parameter('b_hh', None)
        #     self.register_parameter('b_s', None)
        #     self.register_parameter('b_q', None)

        self.reset_parameters()

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            init.uniform_(weight, -stdv, stdv)

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed one step forward propagation of ST-LSTM.
        :param input_l: input of location embedding vector, shape (batch_size, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, input_size)
        :param hc: tuple containing hidden state and cell state of previous step.
        :return: hidden state and cell state of this step.
        """
        self.check_forward_input(input_l)
        # self.check_forward_input(input_s)
        # self.check_forward_input(input_q)
        if hc is None:
            zeros = torch.zeros(input_l.size(0), self.hidden_size, dtype=input_l.dtype, device=input_l.device)
            hc = (zeros, zeros)
        self.check_forward_hidden(input_l, hc[0], '[0]')
        self.check_forward_hidden(input_l, hc[1], '[0]')

        hidden, cell = hc
        input_vec = torch.cat([hidden, input_l], dim=-1)
        in_gate = torch.mm(input_vec, self.w_i.t()) + self.b_i
        # forget_gate = torch.mm(input_vec, self.w_f.t()) + self.b_f
        cell_gate = torch.mm(input_vec, self.w_c.t()) + self.b_c

        delta_t_1 = torch.mm(input_q, self.w_q1.t())
        delta_t_2 = torch.mm(input_q, self.w_q2.t())
        delta_d_1 = torch.mm(input_s, self.w_s1.t())
        delta_d_2 = torch.mm(input_s, self.w_s2.t())
        out_gate = torch.mm(input_vec, self.w_o.t()) + torch.mm(input_q, self.w_qo.t()) + torch.mm(input_s, self.w_so.t()) + self.b_o

        # with torch.no_grad():
        #     self.w_xq1.copy_(self.w_xq1.data.clamp(max=0))
        #     self.w_xs1.copy_(self.w_xs1.data.clamp(max=0))
        self.w_xq1.data.clamp_(max=0)
        self.w_xs1.data.clamp_(max=0)
        t1_gate = torch.mm(input_l, self.w_xq1.t()) + torch.sigmoid(delta_t_1) + self.b_q_1
        t2_gate = torch.mm(input_l, self.w_xq2.t()) + torch.sigmoid(delta_t_2) + self.b_q_2
        d1_gate = torch.mm(input_l, self.w_xs1.t()) + torch.sigmoid(delta_d_1) + self.b_s_1
        d2_gate = torch.mm(input_l, self.w_xs2.t()) + torch.sigmoid(delta_d_2) + self.b_s_2

        in_gate = torch.sigmoid(in_gate)
        # forget_gate = torch.sigmoid(forget_gate)
        cell_gate = torch.tanh(cell_gate)
        out_gate = torch.sigmoid(out_gate)

        # next_cell_new = (forget_gate * cell) + (in_gate * torch.sigmoid(t1_gate) * torch.sigmoid(d1_gate) * cell_gate)
        next_cell_new = (1 - in_gate * torch.sigmoid(t1_gate) * torch.sigmoid(d1_gate)) * cell + in_gate * torch.sigmoid(t1_gate) * torch.sigmoid(d1_gate) * cell_gate
        # next_cell = (forget_gate * cell) + (in_gate * torch.sigmoid(t2_gate) * torch.sigmoid(d2_gate) * cell_gate)
        next_cell = (1 - in_gate) * cell + in_gate * torch.sigmoid(t2_gate) * torch.sigmoid(d2_gate) * cell_gate
        next_hidden = out_gate * torch.tanh(next_cell_new)

        return next_hidden, next_cell
        

class STGCN(nn.Module):
    """
    One layer, batch-first Spatial-Temporal LSTM network.
    Kong D, Wu F. HST-LSTM: A Hierarchical Spatial-Temporal Long-Short Term Memory Network
    for Location Prediction[C]//IJCAI. 2018: 2341-2347.
    Examples:
        >>> st_lstm = STLSTM(10, 20)
        >>> input_l = torch.randn(6, 3, 10)
        >>> input_s = torch.randn(6, 3, 10)
        >>> input_q = torch.randn(6, 3, 10)
        >>> hidden_out, cell_out = st_lstm(input_l, input_s, input_q)
    """
    def __init__(self, input_size, hidden_size, bias=True):
        """
        :param input_size: The number of expected features in the input `x`
        :param hidden_size: The number of features in the hidden state `h`
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        """
        super(STGCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.cell = STGNCell(input_size, hidden_size, bias)

    def forward(self, input_l, input_s, input_q, hc=None):
        """
        Proceed forward propagation of ST-LSTM network.
        :param input_l: input of location embedding vector, shape (batch_size, step, input_size)
        :param input_s: input of spatial embedding vector, shape (batch_size, step, input_size)
        :param input_q: input of temporal embedding vector, shape (batch_size, step, input_size)
        :param hc: tuple containing initial hidden state and cell state, optional.
        :return: hidden states and cell states produced by iterate through the steps.
        """
        output_hidden, output_cell = [], []
        # self.check_forward_input(input_l, input_s, input_q)
        for step in range(input_l.size(1)):
            hc = self.cell(input_l[:,step,:], input_s[:,step, :], input_q[:,step, :], hc)
            output_hidden.append(hc[0])
            output_cell.append(hc[1])
        return torch.stack(output_hidden, dim=1), torch.stack(output_cell, dim=1)


class STGCN_CLS(nn.Module):
    def __init__(self, hidden_size, item_num):
        super(STGCN_CLS, self).__init__()
        self.item_num = item_num
        self.hidden_size = hidden_size
        self.item_emb = nn.Parameter(torch.Tensor(item_num, hidden_size))
        self.rnn = STGCN(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, item_num)
        self.reset_parameter()
    
    def reset_parameter(self):
        glorot(self.item_emb)
    
    def forward(self, seq_idx, t_seq, s_seq, state=None):
        x = self.item_emb[seq_idx]
        out, cell = self.rnn(x, s_seq, t_seq, state)
        out_for_prob = out.reshape(-1, self.hidden_size)
        prob = self.linear(out_for_prob)
        return prob, (out[:,-1,:], cell[:,-1,:])