import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv import GATConv, GCNConv
from model.layer import TADGAT_Conv, LGCConv, ResGATConv

class TADGAT(nn.Module):
    def __init__(self, in_channel, hidden_channel, num_layer, user_num, item_num, ubias_num, ibias_num, soc_layer=3, heads=4, ssl_temp=0.5, aux_feat=None):
        # TODO: add geographical distance mapping function and embedding
        super(TADGAT, self).__init__()
        self.in_channel = in_channel
        self.hid_channel = hidden_channel
        self.num_layer = num_layer
        self.soc_layer = soc_layer
        self.heads = heads
        self.user_num = user_num
        self.poi_num = item_num
        self.max_seq_len = 55
        self.ssl_temp = ssl_temp
        
        if aux_feat is None:
            self.id_emb = nn.Parameter(torch.Tensor(user_num + item_num + 1, in_channel))
            glorot(self.id_emb)
        else:
            self.id_emb = torch.Tensor(user_num, in_channel)
            glorot(self.id_emb)
            self.id_emb = nn.Parameter(torch.cat([self.id_emb, aux_feat], dim=0))
        
        self.ucenter_att_bias = nn.Parameter(torch.Tensor(ubias_num+2, heads * hidden_channel))
        self.icenter_att_bias = nn.Parameter(torch.Tensor(ibias_num+2, heads * hidden_channel))
        self.ucenter_emb_bias = nn.Parameter(torch.Tensor(ubias_num+2, heads * hidden_channel))
        self.icenter_emb_bias = nn.Parameter(torch.Tensor(ibias_num+2, heads * hidden_channel))

        self.trans_emb_bias = nn.Parameter(torch.Tensor(ibias_num+2, heads * hidden_channel))   # Last version: the last bias term is used for social relation
        self.trans_att_bias = nn.Parameter(torch.Tensor(ibias_num+2, heads * hidden_channel))

        self.ucenter_att_bias_2 = nn.Parameter(torch.Tensor(ubias_num*4+2, heads*hidden_channel))
        self.icenter_att_bias_2 = nn.Parameter(torch.Tensor(ibias_num*4+2, heads*hidden_channel))
        self.ucenter_emb_bias_2 = nn.Parameter(torch.Tensor(ubias_num*4+2, heads*hidden_channel))
        self.icenter_emb_bias_2 = nn.Parameter(torch.Tensor(ibias_num*4+2, heads*hidden_channel))
        self.trans_emb_bias_2 = nn.Parameter(torch.Tensor(ibias_num*4+2, heads*hidden_channel))
        self.trans_att_bias_2 = nn.Parameter(torch.Tensor(ibias_num*4+2, heads*hidden_channel))

        self.udist_emb_bias = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.udist_att_bias = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.idist_emb_bias = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.idist_att_bias = nn.Parameter(torch.Tensor(501, heads * hidden_channel))

        self.udist_emb_bias_2 = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.udist_att_bias_2 = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.idist_emb_bias_2 = nn.Parameter(torch.Tensor(501, heads * hidden_channel))
        self.idist_att_bias_2 = nn.Parameter(torch.Tensor(501, heads * hidden_channel))

        self.trans_head = nn.Parameter(torch.Tensor(in_channel, hidden_channel*heads))
        # self.soc_trans_head = nn.Parameter(torch.Tensor(hidden_channel*heads, hidden_channel*heads*num_layer+self.in_channel))
        # self.bound_vec = nn.Parameter(torch.Tensor(self.in_channel, 1))
        # self.bound_vec = nn.Parameter(torch.Tensor(heads*hidden_channel*num_layer+self.in_channel, 1))
        self.unified_map = nn.Linear(heads*hidden_channel*num_layer, in_channel, bias=False)
        self.rnn_update = nn.Linear(heads*hidden_channel*2, heads*hidden_channel, bias=False)
        # self.pred_head = nn.Sequential(nn.Linear(hidden_channel*heads*(num_layer)+self.in_channel, hidden_channel*heads), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(hidden_channel*heads, self.in_channel))
        
        self.pred_head = nn.Sequential(nn.Linear(hidden_channel*heads*(num_layer)+self.in_channel, hidden_channel*heads), nn.LeakyReLU(negative_slope=0.2), nn.Dropout(p=0.3), nn.Linear(hidden_channel*heads, item_num))
        # self.pred_head = nn.Linear(hidden_channel*heads*(num_layer)+self.in_channel, item_num)
        self.convs = nn.ModuleList()
        for i in range(num_layer):
            in_channel = in_channel if i == 0 else heads * hidden_channel
            self.convs.append(TADGAT_Conv((in_channel, in_channel), hidden_channel, edge_channel=self.in_channel, num_etype=6, heads=heads, residual=False))
            # self.convs.append(GATConv((in_channel, in_channel), hidden_channel, heads))

            # self.short_term_encoders.append(MultiHeadAttention(in_channel, hidden_channel*heads, ubias_num*2))
        # self.short_term_encoder = Transformer_layer(self.in_channel, 256, ubias_num*2)

        # self.soc_convs = nn.ModuleList()
        # for i in range(soc_layer):
        #     # self.soc_convs.append(LGCConv(self.in_channel))
        #     in_channel = self.in_channel if i == 0 else hidden_channel * heads
        #     self.soc_convs.append(ResGATConv(in_channel, hidden_channel, heads, residual=False))
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.trans_head)
        # glorot(self.soc_trans_head)
        # glorot(self.bound_vec)

        glorot(self.ucenter_att_bias)
        glorot(self.ucenter_emb_bias)
        glorot(self.icenter_att_bias)
        glorot(self.icenter_emb_bias)

        glorot(self.trans_att_bias)
        glorot(self.trans_emb_bias)

        glorot(self.ucenter_att_bias_2)
        glorot(self.ucenter_emb_bias_2)
        glorot(self.icenter_att_bias_2)
        glorot(self.icenter_emb_bias_2)
        glorot(self.trans_att_bias_2)
        glorot(self.trans_emb_bias_2)

        glorot(self.udist_att_bias)
        glorot(self.udist_emb_bias)
        glorot(self.idist_att_bias)
        glorot(self.idist_emb_bias)

        glorot(self.udist_att_bias_2)
        glorot(self.udist_emb_bias_2)
        glorot(self.idist_att_bias_2)
        glorot(self.idist_emb_bias_2)

    def forward(self, adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, device=None, soc_edges=None):#, 
                # soc_adjs=None, soc_node_idx=None, soc_cnid=None):
        x = self.id_emb[node_idx]
        x_backup = self.id_emb[node_idx]
        
        # edge_index, edge_type, t_offset, seq_ids, _ = adjs
        edge_index, edge_type, t_offset_full, dists_full, _ = adjs

        # # emask = edge_type != 1
        # emask = edge_type < 2
        # edge_index = edge_index[:, emask]
        # dist_mask = edge_type[edge_type != 1] < 2
        # edge_type = edge_type[emask]
        # t_offset_full = [t_offset_full[i][emask] for i in range(len(t_offset_full))]
        # dists_full = [dists_full[i][dist_mask] for i in range(len(dists_full))]

        hetero_idx, homo_idx = (edge_type >= 2), (edge_type < 2)
        att_bias_set, emb_bias_set = [], []
        att_dist_set, emb_dist_set = [], [] 

        for n, t_offset in enumerate(t_offset_full):
            edge_type_homo = edge_type[homo_idx]
            edge_type_hetero = edge_type[hetero_idx]
            t_offset_homo = t_offset[homo_idx]
            t_offset_hetero = t_offset[hetero_idx]
            edge_type_homo = edge_type_homo.unsqueeze(-1)
            edge_type_hetero = edge_type_hetero.unsqueeze(-1)
            if n == 0:
                att_bias = torch.where(edge_type_homo == 0, self.ucenter_att_bias[t_offset_homo], self.icenter_att_bias[t_offset_homo]).reshape(-1, self.heads, self.hid_channel)
                emb_bias = torch.where(edge_type_homo == 0, self.ucenter_emb_bias[t_offset_homo], self.icenter_emb_bias[t_offset_homo]).reshape(-1, self.heads, self.hid_channel)
                att_bias_trans = self.trans_att_bias[t_offset_hetero].reshape(-1, self.heads, self.hid_channel)
                emb_bias_trans = self.trans_emb_bias[t_offset_hetero].reshape(-1, self.heads, self.hid_channel)
            elif n == 1:
                att_bias = torch.where(edge_type_homo == 0, self.ucenter_att_bias_2[t_offset_homo], self.icenter_att_bias_2[t_offset_homo]).reshape(-1, self.heads, self.hid_channel)
                emb_bias = torch.where(edge_type_homo == 0, self.ucenter_emb_bias_2[t_offset_homo], self.icenter_emb_bias_2[t_offset_homo]).reshape(-1, self.heads, self.hid_channel)
                att_bias_trans = self.trans_att_bias_2[t_offset_hetero].reshape(-1, self.heads, self.hid_channel)
                emb_bias_trans = self.trans_emb_bias_2[t_offset_hetero].reshape(-1, self.heads, self.hid_channel)
            # att_bias, emb_bias = torch.cat([att_bias, att_bias_trans], dim=0), torch.cat([emb_bias, emb_bias_trans], dim=0)
            att_bias_set.append(torch.cat([att_bias, att_bias_trans], dim=0))
            emb_bias_set.append(torch.cat([emb_bias, emb_bias_trans], dim=0))
        if dists_full is not None:
            for n, dists in enumerate(dists_full):
                dist_involved_idx = (edge_type == 0) | (edge_type == 2) | (edge_type == 3)
                dist_edge_type = edge_type[dist_involved_idx]
                if n == 0:
                    att_bias_dist = torch.cat([self.udist_att_bias[dists[dist_edge_type == 0]], self.idist_att_bias[dists[dist_edge_type != 0]]]).reshape(-1, self.heads, self.hid_channel)
                    emb_bias_dist = torch.cat([self.udist_emb_bias[dists[dist_edge_type == 0]], self.idist_emb_bias[dists[dist_edge_type != 0]]]).reshape(-1, self.heads, self.hid_channel)
                elif n == 1:
                    att_bias_dist = torch.cat([self.udist_att_bias_2[dists[dist_edge_type == 0]], self.idist_att_bias_2[dists[dist_edge_type != 0]]]).reshape(-1, self.heads, self.hid_channel)
                    emb_bias_dist = torch.cat([self.udist_emb_bias_2[dists[dist_edge_type == 0]], self.idist_emb_bias_2[dists[dist_edge_type != 0]]]).reshape(-1, self.heads, self.hid_channel)
                att_dist_set.append(att_bias_dist)
                emb_dist_set.append(emb_bias_dist)
            
            # dists = torch.cat([dists[dist_edge_type == 0], dists[dist_edge_type != 0]])
            # att_bias_dist = self.dist_att_bias[dists].reshape(-1, self.heads, self.hid_channel)
            # emb_bias_dist = self.dist_emb_bias[dists].reshape(-1, self.heads, self.hid_channel)
        # else:
        #     att_bias_dist, emb_bias_dist = None, None
     
        edge_index = torch.cat([edge_index[:, homo_idx], edge_index[:, hetero_idx]], dim=-1)
        edge_type = torch.cat([edge_type_homo.squeeze(), edge_type_hetero.squeeze()], dim=-1)
        
        # # TODO: For Transformer-based short-term preference modeling
        # batch_size = orig_seqs.size(0)
        # avail_idx = torch.arange(0, self.max_seq_len, dtype=torch.int32, device=device).unsqueeze(0) \
        #                     .expand(batch_size,self.max_seq_len).lt(seq_lens.unsqueeze(1))
        # time_avail_idx = avail_idx.unsqueeze(1) & avail_idx.unsqueeze(-1)
        # time_mat = time_diff.unsqueeze(-1) - time_diff.unsqueeze(1)
        # time_mat = torch.abs(time_mat) // (3600 * 12)
        # time_mask = torch.full((time_diff.size(0), self.max_seq_len, self.max_seq_len), -1, device=device)
        # time_mask[time_avail_idx] = time_mat[time_avail_idx]
        # # for i in range(orig_seqs.size(0)):
        # #     time_mask[i, :seq_lens[i], :seq_lens[i]] = time_mat[i, :seq_lens[i], :seq_lens[i]] 
        # time_diff = torch.where(time_mask <= 360, time_mask, 361)

        # seq_mask = torch.ones((55, 55), dtype=torch.uint8, device=device)
        # seq_mask = torch.triu(seq_mask, diagonal=1).repeat(orig_seqs.size(0), 1).reshape(-1, 55, 55).bool()
        # coef_mat = torch.zeros((batch_size, self.max_seq_len), device=device)
        # coef_mat[avail_idx] = 1
        # coef_mat = coef_mat / seq_lens.unsqueeze(-1)
        # # seq_emb = self.id_emb[orig_seqs]
        # # seq_emb = self.short_term_encoder(seq_emb, seq_mask, time_diff)    # [B, len, emb]
        # # short_emb = (seq_emb * coef_mat.unsqueeze(-1)).sum(1)
        
        x_inter = [x]
        for i in range(self.num_layer):
            last_x = x
            # seq_emb = x[orig_seqs]
            # x_s = self.short_term_encoders[i](seq_emb, seq_emb, seq_emb, seq_mask, time_diff)
            # x_s = (x_s * coef_mat.unsqueeze(-1)).sum(1)
            # non_short_emb = torch.zeros((1, x_s.size(1)), device=device)
            # x_s = torch.cat([x_s, non_short_emb], dim=0)
            # x_s = x_s[seq_ids]
            if i == 0:
                last_x = last_x @ self.trans_head
            x = self.convs[i](x, edge_index, edge_type, att_bias_set, emb_bias_set, att_dist_set, emb_dist_set)
            x = torch.tanh(self.rnn_update(torch.cat([x, last_x], dim=-1)))
            x_inter.append(x)
            if i != self.num_layer - 1:
                x = x.relu()
                x = F.dropout(x, p=0.3, training=self.training)

        x = torch.cat(x_inter, dim=-1)
        x = x[center_nid]
        

        # # New: social diffusion introduction
        # soc_x = self.id_emb[:self.user_num]
        # # final_soc_x = soc_x
        # for i in range(self.soc_layer):
        #     soc_x = self.soc_convs[i](soc_x, soc_edges)
        #     if i != self.soc_layer - 1:
        #         soc_x = soc_x.relu()
        #         soc_x = F.dropout(soc_x, p=0.3, training=self.training)
        #     # final_soc_x = final_soc_x + soc_x
        # # final_soc_x = final_soc_x / (self.soc_layer + 1)
        # x_soc_bias = soc_x[node_idx[center_nid]]
        # # x_soc_bias = soc_x[node_idx[center_nid]]
        # x_w_soc = x + x_soc_bias @ self.soc_trans_head
        # # x = torch.cat([x, x_soc_bias], dim=-1)
        
        # # Code for social influence addition v2
        # if soc_adjs is not None:
        #     soc_x = self.id_emb[soc_node_idx]
        #     soc_eindex, soc_etype, soc_toff, soc_dists, _ = soc_adjs
        #     soc_att_bias, soc_emb_bias = self.soc_t_att_bias[soc_toff].reshape(-1, self.heads, self.hid_channel), self.soc_t_emb_bias[soc_toff].reshape(-1, self.heads, self.hid_channel)
        #     if soc_dists is not None:
        #         soc_att_dist_bias, soc_emb_dist_bias = self.soc_d_att_bias[soc_dists].reshape(-1, self.heads, self.hid_channel), self.soc_d_emb_bias[soc_dists].reshape(-1, self.heads, self.hid_channel)
        #     last_soc_x = soc_x @ self.trans_head
        #     soc_x = self.soc_convs(soc_x, soc_eindex, soc_etype, soc_att_bias, soc_emb_bias, soc_att_dist_bias, soc_emb_dist_bias)
        #     soc_x = torch.tanh(self.soc_update(torch.cat([soc_x, last_soc_x], dim=-1)))
        #     soc_idx, soc_bias_list = [], []
        #     for i, cnid in enumerate(soc_cnid):
        #         if len(cnid) == 0:
        #             soc_bias = torch.zeros((1, soc_x.size(1)), device=device)
        #             soc_bias_list.append(soc_bias)
        #             continue
        #         soc_bias = soc_x[cnid]
        #         # soc_idx.append(i)
        #         soc_bias_list.append(soc_bias.mean(0).unsqueeze(0))
        #     soc_bias = torch.cat(soc_bias_list, dim=0)
        #     # x[soc_idx] = x[soc_idx] + soc_bias @ self.soc_trans_head
        #     x = torch.cat([x, soc_bias], dim=-1)
        #     # x = x + soc_bias @ self.soc_trans_head
        
        logits = self.pred_head(x)
        # pref_bound = self.id_emb[node_idx[center_nid]] @ self.bound_vec
        # # pref_bound = x @ self.bound_vec
        # logits = logits @ self.id_emb[self.user_num:].T

        # # TODO: code for MTL
        # if self.training:
        #     # # Code for pretext task 1
        #     # local_x, global_x = x_inter[1][center_nid], x_inter[-1][center_nid]
        #     # local_x, global_x = F.normalize(local_x, dim=-1, p=2), F.normalize(global_x, dim=-1, p=2)
        #     # mtl_loss_1 = self.infonce(local_x, global_x)

        #     # Code for pretext task 2
        #     ssl_rate = 0.5
        #     # seed_1 = np.random.rand(edge_index.size(1))
        #     seed_2 = np.random.rand(edge_index.size(1))
        #     # seed_1 = torch.tensor(seed_1, device=device)
        #     seed_2 = torch.tensor(seed_2, device=device)
        #     # cond_inter = (edge_type != 1) | (seed_1 > ssl_rate)
        #     cond_trans = (edge_type != 2) & (edge_type != 3) | (seed_2 > ssl_rate)
        #     # inter_mask_eindex = edge_index[:, cond_inter]
        #     # inter_mask_etype = edge_type[cond_inter]
        #     # inter_emb_bias, inter_att_bias = emb_bias[cond_inter], att_bias[cond_inter]
        #     # inter_emb_dist_bias, inter_att_dist_bias = emb_bias_dist, att_bias_dist
        #     # hinter_x = x_backup
        #     # hinter_x_all = [hinter_x]
        #     # for i in range(self.num_layer):
        #     #     last_x = hinter_x
        #     #     if i == 0:
        #     #         last_x = last_x @ self.trans_head
        #     #     hinter_x = self.convs[i](hinter_x, inter_mask_eindex, inter_mask_etype, inter_att_bias, inter_emb_bias,
        #     #                                  inter_att_dist_bias, inter_emb_dist_bias)
        #     #     hinter_x = torch.tanh(self.rnn_update(torch.cat([hinter_x, last_x], dim=-1)))
        #     #     hinter_x_all.append(hinter_x)
        #     #     if i != self.num_layer - 1:
        #     #         hinter_x = hinter_x.relu()
        #     #         hinter_x = F.dropout(hinter_x, p=0.3, training=self.training)

        #     trans_mask_eindex = edge_index[:, cond_trans]
        #     trans_mask_etype = edge_type[cond_trans]
        #     trans_emb_bias, trans_att_bias = emb_bias[cond_trans], att_bias[cond_trans]
        #     dist_seed = seed_2[(edge_type == 0) | (edge_type == 2) | (edge_type == 3)]
        #     dist_etype = edge_type[(edge_type == 0) | (edge_type == 2) | (edge_type == 3)]
        #     trans_emb_dist_bias, trans_att_dist_bias = emb_bias_dist[(dist_etype != 2) & (dist_etype != 3) | (dist_seed > ssl_rate)], \
        #                                                  att_bias_dist[(dist_etype != 2) & (dist_etype != 3) | (dist_seed > ssl_rate)]
        #     trans_x = x_backup
        #     trans_x_all = [trans_x]
        #     for i in range(self.num_layer):
        #         last_x = trans_x
        #         if i == 0:
        #             last_x = last_x @ self.trans_head
        #         trans_x = self.convs[i](trans_x, trans_mask_eindex, trans_mask_etype, trans_att_bias, trans_emb_bias,
        #                                      trans_att_dist_bias, trans_emb_dist_bias)
        #         trans_x = torch.tanh(self.rnn_update(torch.cat([trans_x, last_x], dim=-1)))
        #         trans_x_all.append(trans_x)
        #         if i != self.num_layer - 1:
        #             trans_x = trans_x.relu()
        #             trans_x = F.dropout(trans_x, p=0.3, training=self.training)
            
        #     # hinter_x, trans_x = torch.cat(hinter_x_all, dim=-1)[center_nid], torch.cat(trans_x_all, dim=-1)[center_nid]
        #     # hinter_x, trans_x = F.normalize(hinter_x, dim=-1, p=2), F.normalize(trans_x, dim=-1, p=2)
        #     trans_x = torch.cat(trans_x_all, dim=-1)[center_nid]
        #     trans_x = F.normalize(trans_x, dim=-1, p=2)
        #     x_norm = F.normalize(x, dim=-1, p=2).detach()
        #     mtl_loss_1 = -(x_norm * trans_x).sum(-1).mean()
        #     # mtl_loss_2 = -(x_norm * hinter_x).sum(-1)
        #     # mtl_loss_1 = self.infonce(hinter_x, trans_x) + self.infonce(trans_x, hinter_x)
        #     return logits, mtl_loss_1 
        # else:

        
        # if self.training:
        #     return logits, pref_bound
        # else:
        return logits
    
    def infonce(self, out_1, out_2):
        pos_score = (out_1 * out_2).sum(-1)
        total_score = torch.matmul(out_1, out_2.T)
        pos_score = torch.exp(pos_score / self.ssl_temp)
        total_score = (torch.exp(total_score / self.ssl_temp)).sum(-1)
        return -torch.log(pos_score / total_score).mean()