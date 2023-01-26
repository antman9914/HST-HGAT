import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import init
from torch_geometric.nn.inits import glorot
from model.layer import LGCConv

# Code for basic MF
class MF(nn.Module):
    def __init__(self, in_channel, user_num, item_num):
        super(MF, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.in_channel = in_channel
        self.user_emb = nn.Parameter(torch.Tensor(user_num, in_channel))
        self.item_emb = nn.Parameter(torch.Tensor(item_num, in_channel))
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.xavier_uniform_(self.user_emb)
        nn.init.xavier_uniform_(self.item_emb)
    
    def forward(self, center_uid, seq):
        selected_user_emb = self.user_emb[center_uid]
        # selected_user_emb = F.normalize(selected_user_emb, p=2, dim=-1)
        # item_emb = F.normalize(self.item_emb, dim=-1, p=2)
        pref_score = torch.mm(selected_user_emb, self.item_emb.T)
        if self.training:
            mask = torch.zeros((len(center_uid), self.item_num))
            for i in range(len(center_uid)):
                mask[i, seq[i]] = 1
            loss = ((pref_score - 1) ** 2)[mask == 1].sum()
            loss += ((pref_score - 0) ** 2)[mask == 0].sum()
            return loss, pref_score
        else:
            return pref_score


class LSTM_Basic(nn.Module):
    def __init__(self, in_channel, item_num):
        super().__init__()
        self.item_num = item_num
        self.in_channel = in_channel
        self.item_emb = nn.Parameter(torch.Tensor(item_num, in_channel))
        self.lstm = nn.LSTM(in_channel, in_channel, 1)
        self.linear = nn.Linear(in_channel, item_num)
    
    def reset_paramter(self):
        nn.init.xavier_uniform_(self.item_emb)

    def forward(self, seq_idx, state=None):
        x = self.item_emb[seq_idx]
        out, state = self.lstm(x, state)
        out = out.reshape(-1, self.in_channel)
        prob = self.linear(out)
        return prob, state


class LightGCN(nn.Module):
    def __init__(self, in_channel, num_layer, user_num, item_num):
        super().__init__()
        self.item_num = item_num
        self.in_channel = in_channel
        self.num_layer = num_layer
        self.user_num = user_num
        self.id_emb = nn.Parameter(torch.Tensor(item_num + user_num, in_channel))
        self.convs = nn.ModuleList()
        for i in range(num_layer):
            self.convs.append(LGCConv(in_channel))
        self.reset_parameter()
    
    def reset_parameter(self):
        glorot(self.id_emb)

    def forward(self, edge_index, user_idx, seq):
        x = self.id_emb
        final_x = x
        for i in range(self.num_layer):
            x = self.convs[i](x, edge_index)
            final_x = final_x + x
        final_x = final_x / (self.num_layer + 1)
        target_u = final_x[user_idx]
        cand_poi = final_x[self.user_num:]
        # target_u, cand_poi = F.normalize(target_u, p=2, dim=-1), F.normalize(cand_poi, p=2, dim=-1)
        logits = target_u @ cand_poi.T
        if self.training:
            mask = torch.zeros((len(user_idx), self.item_num))
            loss = 0.
            for i in range(len(user_idx)):
                mask[i, seq[i]] = 1
                pos_logits = logits[i, mask[i] == 1]
                neg_logits = logits[i, mask[i] == 0]
                pos_loss = F.logsigmoid(pos_logits).mean()
                neg_loss = F.logsigmoid(-neg_logits).mean()
                loss += -pos_loss-neg_loss
            loss /= len(user_idx)
            # loss = ((logits - 1) ** 2)[mask == 1].sum()
            # loss += ((logits - 0) ** 2)[mask == 0].sum()
            return loss, logits
        else:
            return logits
        