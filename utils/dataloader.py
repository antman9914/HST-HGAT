import copy
import math
import random
import pickle
import numpy as np
import torch
from typing import  Optional, Tuple, NamedTuple
from haversine import haversine, Unit
from torch.utils.data import DataLoader, Dataset
from torch import Tensor
from torch_sparse import SparseTensor

from model.adj import Adj, Adj_v2, STP_Adj

GOW_ROOT = 'gowalla_processed/'
FOUR_ROOT = 'foursquare_processed/'

USER_SET_GOW = GOW_ROOT + 'seen_user.pkl'
POI_SET_GOW = GOW_ROOT + 'seen_poi.pkl'
LOC_INFO_GOW = GOW_ROOT + 'loc_info.pkl'
TRAIN_GOW = GOW_ROOT + 'train_seq_align.pkl'
TEST_GOW = GOW_ROOT + 'test_seq_align.pkl'
VAL_GOW = GOW_ROOT + 'val_seq_align.pkl'

USER_SET_FOUR = FOUR_ROOT + 'seen_user.pkl'
POI_SET_FOUR = FOUR_ROOT + 'seen_poi.pkl'
LOC_INFO_FOUR = FOUR_ROOT + 'loc_info.pkl'
TRAIN_FOUR = FOUR_ROOT + 'train_seq_align.pkl'
TEST_FOUR = FOUR_ROOT + 'test_seq_align.pkl'
VAL_FOUR = FOUR_ROOT + 'val_seq_align.pkl'


class POI_Loader(DataLoader):

    def __init__(self, max_len, layer_num, ubias_num, ibias_num, neg_sample_num, dataset='gowalla', mode='train', **kwargs):
        
        if dataset == 'gowalla':
            user_set, poi_set = pickle.load(open(USER_SET_GOW, 'rb')), pickle.load(open(POI_SET_GOW, 'rb'))
        else:
            user_set, poi_set = pickle.load(open(USER_SET_FOUR, 'rb')), pickle.load(open(POI_SET_FOUR, 'rb'))
        self.poi_num = len(poi_set)
        self.user_num = len(user_set)
        self.layer_num = layer_num
        self.sample_bound = max_len
        self.neg_sample_num = 9
        self.max_seq_len = 55
        self.mode = mode
        self.dataset = dataset
        root = GOW_ROOT if dataset == 'gowalla' else FOUR_ROOT
        self.ubias_num, self.ibias_num = ubias_num, ibias_num

        poi_info = pickle.load(open(LOC_INFO_GOW, 'rb')) if dataset == 'gowalla' else pickle.load(open(LOC_INFO_FOUR, 'rb'))
        poi_map = {}
        geo_info = np.zeros((len(poi_set), 2))
        min_lat, min_lon = 0, 0
        for i in range(len(poi_set)):
            poi_map[poi_set[i]] = i
        for id in poi_set:
            entry = poi_info[id]
            lat, lon = entry
            min_lat, min_lon = min(min_lat, lat), min(min_lon, lon)
            geo_info[poi_map[id]] = [lat, lon]
        self.geo_info = geo_info
        print(min_lat, min_lon)

        # TODO: change dataset setting
        if mode == 'train':
            egonet_data = pickle.load(open(root + 'train_ego_list_v2.pkl', 'rb'))
        elif mode == 'val':
            # egonet_data = pickle.load(open(root + 'val_ego_list_hetero_poi_geo_v2.pkl', 'rb'))
            egonet_data = pickle.load(open(root + 'val_ego_list_v2.pkl', 'rb'))
        else:
            egonet_data = pickle.load(open(root + 'test_ego_list_v2.pkl', 'rb'))
            # Last version: hetero_poi_geo_v2
        
        # ego_nets, involved_uid = egonet_data
        ego_nets = egonet_data
        self.ego_nets = ego_nets
        print(len(ego_nets))
        # self.ego_nets = []
        # for i, uid in enumerate(ego_nets.keys()):
        #     cur_out = ego_nets[uid]
        #     if i % 500 == 0:
        #         print(i)
        #     # if self.mode != 'test':
        #     for i in range(len(cur_out)):
        #         adjs, nid, sample_pair = cur_out[i]
        #         cur_target = sample_pair[1]
        #         neg_samples = np.random.choice(self.poi_num, self.neg_sample_num, replace=False)
        #         neg_samples = neg_samples + self.user_num
        #         cur_sample = [np.array([uid, cur_target]), neg_samples]
        #         cur_sample = np.concatenate(cur_sample).reshape(1, -1)
        #         self.ego_nets.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))
        # involved_uid = list(ego_nets.keys())

        # print(len(involved_uid))

        # New: add social diffusion
        soc_edges = pickle.load(open(root + 'soc_edge.pkl', 'rb'))
        # soc_eindex = []
        # for key in soc_edges:
        #     for tar in soc_edges[key]:
        #         soc_eindex.append([key, tar])
        self.soc_edges = torch.tensor(soc_edges, dtype=torch.int64).T

        # self.dataset = np.array(involved_uid).reshape(-1, 1)
        self.dataset = np.arange(len(self.ego_nets), dtype=np.int64).reshape(-1, 1)
        super(POI_Loader, self).__init__(self.dataset, collate_fn=self.sample, **kwargs)

    def __len__(self):
        return len(self.dataset)
    
    def sample(self, batch):
        # sample_size = len(batch[0])
        # batch = np.concatenate(batch).reshape(-1, sample_size)
        # orig_batch = torch.tensor(batch, dtype=torch.int64)
        # n_id = batch[:, 0].tolist()
        batch = np.concatenate(batch).reshape(-1)
        all_out = []
        for idx in batch:
            all_out.append(self.ego_nets[idx])
        # all_out = self.ego_nets[batch]

        # for uid in n_id:
        #     cur_out = self.ego_nets[uid]
        #     if self.mode != 'test':
        #         for i in range(len(cur_out)):
        #             adjs, nid, sample_pair = cur_out[i]
        #             cur_target = sample_pair[1]
        #             neg_samples = np.random.choice(self.poi_num, self.neg_sample_num, replace=False)
        #             while cur_target-self.user_num in neg_samples:
        #                 neg_samples = np.random.choice(self.poi_num, self.neg_sample_num, replace=False)
        #             neg_samples = neg_samples + self.user_num
        #             cur_sample = [np.array([uid, cur_target]), neg_samples]
        #             cur_sample = np.concatenate(cur_sample).reshape(1, -1)
        #             all_out.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))
        #     else:
        #         adjs, nid, sample_pair = cur_out[-1]
        #         cur_target = sample_pair[1]
        #         neg_samples = np.random.choice(self.poi_num, self.neg_sample_num, replace=False)
        #         while cur_target-self.user_num in neg_samples:
        #             neg_samples = np.random.choice(self.poi_num, self.neg_sample_num, replace=False)
        #         neg_samples = neg_samples + self.user_num
        #         cur_sample = [np.array([uid, cur_target]), neg_samples]
        #         cur_sample = np.concatenate(cur_sample).reshape(1, -1)
        #         all_out.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))

        ground_ref = torch.ones((self.max_seq_len,), dtype=torch.int64)
        full_edge_index, full_edge_type, full_t_offset, full_node_idx, full_time_diff = [], [], [], [], []
        full_t_offset_2 = []
        # full_orig_seqs, full_seq_len, full_seq_id = [], [], []
        full_dists = []
        full_dists_2 = []
        batch_samples = []
        center_nids = []
        last_node_offset = 0

        # soc_full_eindex, soc_full_etype, soc_full_toff, soc_full_node_idx = [], [], [], []
        # soc_full_dists = []
        # soc_center_nids = []
        # soc_node_offset = 0

        for entry in all_out:
            # 1. Align ensembled graph
            adjs, nid, cur_sample = entry
            # adjs, nid, soc_adjs, soc_nid, cur_sample = entry
            batch_samples.append(cur_sample)
            full_node_idx.append(nid)
            edge_index, edge_type, t_offset = adjs.edge_index, adjs.edge_type, copy.deepcopy(adjs.edge_weight)
            t_offset_2 = copy.deepcopy(t_offset)
            # assert adjs.orig_seq is not None
            # seq_endpoint = list(adjs.size)
            full_edge_type.append(edge_type)
            full_edge_index.append(edge_index + last_node_offset)
            ucenter_idx, icenter_idx = (edge_type == 0), (edge_type == 1) | (edge_type == 2) | (edge_type == 3)
            t_offset[ucenter_idx] = torch.clamp(t_offset[ucenter_idx] // (3600 * 12), max=self.ubias_num+1)
            t_offset[icenter_idx] = torch.clamp(t_offset[icenter_idx] // (3600 * 12), max=self.ibias_num+1)
            full_t_offset.append(t_offset)
            t_offset_2[ucenter_idx] = torch.clamp(t_offset_2[ucenter_idx] // (3600 * 3), max=self.ubias_num*4+1)
            t_offset_2[icenter_idx] = torch.clamp(t_offset_2[icenter_idx] // (3600 * 3), max=self.ibias_num*4+1)
            full_t_offset_2.append(t_offset_2)
            if adjs.orig_seq is not None:
                dists = copy.deepcopy(adjs.orig_seq)
                dists_1 = torch.clamp(dists * 10 // 1, max=500)
                dists_2 = torch.clamp(dists * 2 // 1, max=500)
                full_dists.append(dists_1)
                full_dists_2.append(dists_2)
            
            # if soc_adjs is not None:
            #     soc_nid = torch.tensor(soc_nid, dtype=torch.int64)
            #     soc_full_node_idx.append(soc_nid)
            #     soc_eindex, soc_etype, soc_toff = soc_adjs.edge_index, soc_adjs.edge_type, copy.deepcopy(soc_adjs.edge_weight)
            #     soc_full_etype.append(soc_etype)
            #     soc_full_eindex.append(soc_eindex + soc_node_offset)
            #     soc_full_toff.append(torch.clamp(soc_toff // (3600 * 12), max=self.ubias_num+1))
            #     if soc_adjs.orig_seq is not None:
            #         soc_dists = copy.deepcopy(soc_adjs.orig_seq)
            #         soc_dists = torch.clamp(soc_dists * 10 // 1, max=500)
            #         soc_full_dists.append(soc_dists)
            #     soc_center_nids.append(np.arange(len(soc_nid[soc_nid < self.user_num])) + soc_node_offset)
            #     soc_node_offset += len(soc_nid)
            # else:
            #     soc_center_nids.append([])


            # # 2. Separate individual short-term sequences from edge indexes
            # start = time.time()
            # cur_sequences, seq_timestamp = [ground_ref], [ground_ref]
            # t_offset = adjs.edge_weight
            # seq_id = torch.full((edge_index.size(1),), -1)
            # # real_edge_index = nid[edge_index]
            # last_tar = -1
            # for idx in seq_endpoint:
            #     src, tar = idx
            #     # seq_id[edge_index[1, src]] = len(full_orig_seqs) + len(cur_sequences) - 1
            #     seq_id[src:tar] = len(full_orig_seqs) + len(cur_sequences) - 1
            #     cur_sequences.append(edge_index[0, tar - 1] + last_node_offset)
            #     seq_timestamp.append(t_offset[tar - 1] - t_offset[src:tar])
            #     full_seq_len.append(tar - src)
            #     if full_seq_len[-1] >= 55:
            #         print(cur_sequences[-1].numpy())
            #         print(full_seq_len[-1])
            #     assert full_seq_len[-1] < 55
            #     # For heterogeneous sequence short-term preference
            #     if last_tar > 0 and last_tar != src:
            #         seq_id[last_tar:src] = len(full_orig_seqs) + len(cur_sequences) - 1
            #         cur_sequences.append(edge_index[0, src-1] + last_node_offset)
            #         seq_timestamp.append(t_offset[src - 1] - t_offset[last_tar:src])
            #         full_seq_len.append(src - last_tar)
            #         if full_seq_len[-1] >= 55:
            #             print(cur_sequences[-1].numpy())
            #             print(full_seq_len[-1])
            #         assert full_seq_len[-1] < 55
            #     last_tar = tar
            #     # break
            # if last_tar != edge_index.size(1):
            #     end = edge_index.size(1)
            #     seq_id[last_tar:end] = len(full_orig_seqs) + len(cur_sequences) - 1
            #     cur_sequences.append(edge_index[0, -1] + last_node_offset)
            #     seq_timestamp.append(t_offset[-1] - t_offset[last_tar:])
            #     full_seq_len.append(end - last_tar)
            #     if full_seq_len[-1] >= 55:
            #         print(cur_sequences[-1].numpy())
            #         print(full_seq_len[-1])
            #     assert full_seq_len[-1] < 55
            # assert seq_id[seq_id != -1].size() == seq_id.size()
            # # delta_time += end - start
            # # # First-layer sequence
            # # ucenter_idx = (edge_index[1] == 0) & (real_edge_index[0] >= self.user_num)
            # # seq_id[ucenter_idx] = len(full_orig_seqs) + len(cur_sequences)
            # # cur_sequences.append(real_edge_index[0, ucenter_idx])
            # # seq_timestamp.append(t_offset[ucenter_idx])
            # # seq_len.append(cur_sequences[-1].size(0))
            # # if seq_len[-1] >= 55:
            # #     print(cur_sequences[-1].numpy())
            # #     print(seq_len[-1])
            # # assert seq_len[-1] < 55
            # # # Second-layer sequence
            # # start = time.time()
            # # icenter_idx = real_edge_index[1] >= self.user_num
            # # icenters = torch.unique(real_edge_index[1, icenter_idx], sorted=False)
            # # for cnode in icenters:
            # #     tmp_idx = (real_edge_index[1] == cnode) & (real_edge_index[0] < self.user_num)
            # #     seq_id[tmp_idx] = len(full_orig_seqs) + len(cur_sequences)
            # #     cur_sequences.append(real_edge_index[0, tmp_idx])
            # #     seq_timestamp.append(t_offset[tmp_idx])
            # #     seq_len.append(cur_sequences[-1].size(0))
            # #     if seq_len[-1] >= 55:
            # #         print(cur_sequences[-1], cnode)
            # #     assert seq_len[-1] < 55
            # # end = time.time()
            # # delta_time += end - start
            # # print(end - start)
            # # # TODO: second-layer user-centered sequence
            # # add_ucenter_idx = (edge_index[1] != 0) & (real_edge_index[0] < self.user_num)
            # # add_ucenters = torch.unique(real_edge_index[1, add_ucenter_idx], sorted=False)
            # # for cnode in add_ucenters:
            # #     tmp_idx = (real_edge_index[1] == cnode) & (real_edge_index[0] >= self.user_num)
            # #     seq_id[tmp_idx] = len(full_orig_seqs) + len(cur_sequences)
            # #     cur_sequences.append(real_edge_index[0, tmp_idx])
            # #     seq_timestamp.append(t_offset[tmp_idx])
            # #     seq_len.append(cur_sequences[-1].size(0))
            # #     if seq_len[-1] >= 55:
            # #         print(cur_sequences[-1], cnode)
            # #     assert seq_len[-1] < 55
            
            # # 3. Sequence padding and time difference based mask
            # # pad_seqs = pad_sequence(cur_sequences, batch_first=True)
            # pad_toffset = pad_sequence(seq_timestamp, batch_first=True)
            # # full_orig_seqs.append(pad_seqs[1:])#.unsqueeze(0))
            # full_orig_seqs.append(torch.tensor(cur_sequences[1:], dtype=torch.int64))
            # full_time_diff.append(pad_toffset[1:])#.unsqueeze(0))
            # # start = time.time()
            # # for i in range(len(cur_sequences)):
            # #     cur_seq, cur_toffset = cur_sequences[i], seq_timestamp[i]
            # #     cur_seq_len = cur_seq.shape[0]
            # #     formal_seq = torch.full((self.max_seq_len,), -1, dtype=torch.int64)
            # #     formal_seq[:cur_seq_len] = cur_seq
            # #     full_orig_seqs.append(formal_seq.unsqueeze(0))
            # #     full_seq_len.append(seq_len[i])
                
            # #     formal_toff = torch.full((self.max_seq_len,), -1, dtype=torch.int64)
            # #     formal_toff[:cur_seq_len] = cur_toffset
            # #     full_time_diff.append(formal_toff.unsqueeze(0))

            # start = time.time()
            # time_mat = short_toffset.unsqueeze(-1) - short_toffset.unsqueeze(1)
            # time_mat = torch.abs(time_mat)
            # time_mask = torch.full((short_toffset.size(0), self.max_seq_len, self.max_seq_len), -1)
            # time_mask[:, :cur_seq_len, :cur_seq_len] = time_mat[:, :cur_seq_len, :cur_seq_len] // (3600 * 12)
            # time_mask = torch.where(time_mask <= self.ubias_num, time_mask, self.ubias_num + 1)
            # # full_time_diff.append(time_mask.unsqueeze(0))
            # full_time_diff.append(time_mask)
            # end = time.time()
            # delta_time += end - start
            # print(end - start)

            # full_seq_id.append(seq_id)
            center_nids.append(last_node_offset)
            last_node_offset += len(nid)
            
            # delta_time += end - start
            # # For spatial information
            # lat_embidx, lon_embidx = torch.full(nid.size(), -1, dtype=torch.int64), torch.full(nid.size(), -1, dtype=torch.int64)
            # poi_nid = (nid[nid >= self.user_num] - self.user_num).numpy()
            # lat_embidx[nid >= self.user_num] = torch.tensor(self.geo_info[poi_nid, 0], dtype=torch.int64)
            # lon_embidx[nid >= self.user_num] = torch.tensor(self.geo_info[poi_nid, 1], dtype=torch.int64)

            # full_lat_embidx.append(lat_embidx)
            # full_lon_embidx.append(lon_embidx)
        
        # full_time_diff = torch.cat(full_time_diff, dim=0)
        full_edge_index = torch.cat(full_edge_index, dim=-1)
        full_edge_type = torch.cat(full_edge_type, dim=-1)
        full_t_offset = torch.cat(full_t_offset)
        full_t_offset_2 = torch.cat(full_t_offset_2)
        full_node_idx = torch.cat(full_node_idx)
        if len(full_dists) != 0:
            full_dists = torch.cat(full_dists)
            full_dists_2 = torch.cat(full_dists_2)
        else:
            full_dists, full_dists_2 = None, None
        # full_orig_seqs = torch.cat(full_orig_seqs, dim=0)
        # full_seq_id = torch.cat(full_seq_id)
        # full_seq_len = torch.tensor(full_seq_len, dtype=torch.int32)
        batch_samples = torch.cat(batch_samples, dim=0)
        # full_adj = Adj_v2(full_edge_index, full_edge_type, [full_t_offset, full_t_offset_2], [full_dists, full_dists_2], None)
        full_adj = Adj_v2(full_edge_index, full_edge_type, [full_t_offset, full_t_offset_2], [full_dists], None)
        # full_adj = Adj_v2(full_edge_index, full_edge_type, [full_t_offset], full_dists, None)
        center_nids = torch.tensor(center_nids, dtype=torch.int64)

        # soc_full_eindex = torch.cat(soc_full_eindex, dim=-1)
        # soc_full_etype = torch.cat(soc_full_etype)
        # soc_full_toff = torch.cat(soc_full_toff)
        # soc_full_node_idx = torch.cat(soc_full_node_idx)
        # if len(soc_full_dists) != 0:
        #     soc_full_dists = torch.cat(soc_full_dists)
        # else:
        #     soc_full_dists = None
        # soc_full_adj = Adj(soc_full_eindex, soc_full_etype, soc_full_toff, soc_full_dists, None)
        
        # full_out = (full_adj, full_node_idx, soc_full_adj, soc_full_node_idx, None, None, None, center_nids, soc_center_nids, batch_samples)
        full_out = (full_adj, full_node_idx, None, None, None, center_nids, batch_samples)
        return full_out



class MF_Loader(DataLoader):
    def __init__(self, mode='train', dataset='gowalla', **kwargs):
        self.mode = mode
        if dataset == 'gowalla':
            user_set, poi_set = pickle.load(open(USER_SET_GOW, 'rb')), pickle.load(open(POI_SET_GOW, 'rb'))
        else:
            user_set, poi_set = pickle.load(open(USER_SET_FOUR, 'rb')), pickle.load(open(POI_SET_FOUR, 'rb'))
        self.poi_num = len(poi_set)
        self.user_num = len(user_set)
        if mode == 'train':
            seq_data = pickle.load(open(TRAIN_GOW, 'rb')) if dataset == 'gowalla' else pickle.load(open(TRAIN_FOUR, 'rb'))
        elif mode == 'val':
            seq_data = pickle.load(open(VAL_GOW, 'rb')) if dataset == 'gowalla' else pickle.load(open(VAL_FOUR, 'rb'))
        elif mode == 'test':
            seq_data = pickle.load(open(TEST_GOW, 'rb')) if dataset == 'gowalla' else pickle.load(open(VAL_FOUR, 'rb'))
        self.seq_data = seq_data
        user_key = list(seq_data.keys())
        self.dataset = np.array(user_key, dtype=np.int64).reshape(-1, 1)
        super(MF_Loader, self).__init__(self.dataset, collate_fn=self.sample, **kwargs)
    
    def sample(self, batch):
        batch = np.concatenate(batch).reshape(-1)
        out_seqs = []
        for key in batch:
            full_seq = np.array(self.seq_data[key])[:, 0].astype(np.int64)
            out_seqs.append(full_seq)
        return batch, out_seqs


class RNN_Dataset(Dataset):
    def reset(self):
        random.shuffle(self.user_set)    
        self.next_user_idx = 0
        self.active_users = []
        self.active_user_seq = []
        for i in range(self.bs):            
            self.next_user_idx = (self.next_user_idx + 1) % len(self.user_set)
            self.active_users.append(self.user_set[i]) 
            self.active_user_seq.append(0)
    
    def __init__(self, max_seq_len, batch_size, include_st=False, dataset='gowalla', mode='train'):
        self.mode = mode
        self.dataset = dataset
        self.include_st = include_st
        self.bs = batch_size
        self.max_seq_len = max_seq_len
        self.t_num, self.s_num = 168, 300
        if dataset == 'gowalla':
            user_set, poi_set = pickle.load(open(USER_SET_GOW, 'rb')), pickle.load(open(POI_SET_GOW, 'rb'))
            self.poi_num = len(poi_set)
            self.user_num = len(user_set)
            train_seq = pickle.load(open(TRAIN_GOW, 'rb'))
            val_seq = pickle.load(open(VAL_GOW, 'rb'))
            test_seq = pickle.load(open(TEST_GOW, 'rb'))
            poi_info = pickle.load(open(LOC_INFO_GOW, 'rb'))
        else:
            user_set, poi_set = pickle.load(open(USER_SET_FOUR, 'rb')), pickle.load(open(POI_SET_FOUR, 'rb'))
            self.poi_num = len(poi_set)
            self.user_num = len(user_set)
            train_seq = pickle.load(open(TRAIN_FOUR, 'rb'))
            val_seq = pickle.load(open(VAL_FOUR, 'rb'))
            test_seq = pickle.load(open(TEST_FOUR, 'rb'))
            poi_info = pickle.load(open(LOC_INFO_FOUR   , 'rb'))
        poi_map = {}
        geo_info = np.zeros((len(poi_set), 2))
        for i in range(len(poi_set)):
            poi_map[poi_set[i]] = i
        for id in poi_set:
            entry = poi_info[id]
            lat, lon = entry
            geo_info[poi_map[id]] = [lat, lon]
        self.geo_info = geo_info
        
        if mode == 'train':
            seq_len = None
        elif mode == 'val':
            seq_len = {}
            for key in val_seq:
                train_seq[key] = train_seq[key] + val_seq[key]
                seq_len[key] = len(val_seq[key])
        elif mode == 'test':
            seq_len = {}
            for key in val_seq:
                train_seq[key] = train_seq[key] + val_seq[key]
            for key in test_seq:
                train_seq[key] = train_seq[key] + test_seq[key]
                seq_len[key] = len(test_seq[key])
        
        seq_dict, lbl_dict = {}, {}
        self.user_set = []
        self.seq_count = 0
        if seq_len is None:
            for key in train_seq:
                self.user_set.append(key)
                cur_seq = np.array(train_seq[key])
                full_seq, full_lbl = cur_seq[:-1], cur_seq[1:]
                cur_len = len(full_seq)
                seq_num = cur_len // max_seq_len
                start_idx = cur_len - seq_num * max_seq_len
                self.seq_count += seq_num
                seq_dict[key] = []
                lbl_dict[key] = []
                for j in range(seq_num):
                    seq_dict[key].append(full_seq[j*max_seq_len+start_idx:(j+1)*max_seq_len+start_idx])
                    lbl_dict[key].append(full_lbl[j*max_seq_len+start_idx:(j+1)*max_seq_len+start_idx, 0].astype(np.int64))
        else:
            for key in train_seq:
                if mode == 'val' and key not in val_seq or mode == 'test' and key not in test_seq:
                    continue
                cur_seq = np.array(train_seq[key])
                full_seq, full_lbl = cur_seq[:-1], cur_seq[1:]
                cur_len = seq_len[key] - 1
                seq_num = math.ceil(cur_len / max_seq_len)
                if seq_num == 0:
                    continue
                self.user_set.append(key)
                full_len = len(full_seq)
                start_idx = full_len - seq_num * max_seq_len
                self.seq_count += seq_num
                seq_dict[key] = []
                lbl_dict[key] = []
                for j in range(seq_num):
                    seq_dict[key].append(full_seq[j*max_seq_len+start_idx:(j+1)*max_seq_len+start_idx])
                    lbl_dict[key].append(full_lbl[j*max_seq_len+start_idx:(j+1)*max_seq_len+start_idx, 0].astype(np.int64))
        
        self.seqs, self.lbls = seq_dict, lbl_dict
        self.active_users = [] 
        self.active_user_seq = []
        random.shuffle(self.user_set)
        for i in range(self.bs):
            self.active_users.append(self.user_set[i])
            self.active_user_seq.append(0)
        self.next_user_idx = batch_size

    def __len__(self):
        return self.seq_count // self.bs

    def __getitem__(self, index):
        seqs = []
        seq_t_u, seq_s_u, seq_t_l, seq_s_l = [], [], [], []
        coef_t, coef_s = [], []
        lbls = []
        reset_h = []
        for i in range(self.bs):
            i_user = self.active_users[i]
            j = self.active_user_seq[i]
            max_j = len(self.seqs[i_user])
            if (j >= max_j):
                # repalce this user in current sequence:
                i_user = self.user_set[self.next_user_idx]
                j = 0
                self.active_users[i] = i_user
                self.active_user_seq[i] = j
                self.next_user_idx = (self.next_user_idx + 1) % len(self.user_set)
                while self.user_set[self.next_user_idx] in self.active_users:
                    self.next_user_idx = (self.next_user_idx + 1) % len(self.user_set)
            reset_h.append(j == 0)
            if j >= len(self.seqs[i_user]):
                print(j, i_user)
            cur_seq = self.seqs[i_user][j]
            cur_lbl = self.lbls[i_user][j]
            seqs.append(torch.tensor(cur_seq[:, 0].astype(np.int64), dtype=torch.int64))
            lbls.append(torch.tensor(cur_lbl, dtype=torch.int64))
            self.active_user_seq[i] += 1

            # Code for spatial information
            cur_t = cur_seq[:, 1]
            t_offset = np.zeros(cur_t.shape[0])
            t_offset_u = np.zeros(cur_t.shape[0])
            t_coef = np.zeros(cur_t.shape[0])
            t_delta = cur_t[1:]-cur_t[:-1]
            for n in range(1, len(t_offset)):
                # # Code for ST-LSTM 
                interval = t_delta[n-1].total_seconds() / 3600
                interval = min(interval, self.t_num)
                t_offset[n] = math.floor(interval)
                t_offset_u[n] = math.ceil(interval)
                t_coef[n] = interval - t_offset[n]

                # Code for STGCN
                # t_offset[n] = interval
            
            s_offset, s_offset_u = np.zeros(cur_t.shape[0]), np.zeros(cur_t.shape[0])
            s_coef = np.zeros_like(s_offset)
            poi_id_seq = cur_seq[:, 0]
            for n in range(1, len(s_offset)):
                src, tar = poi_id_seq[n], poi_id_seq[n-1]
                src_lat, src_lon = self.geo_info[src]
                tar_lat, tar_lon = self.geo_info[tar]
                dist = haversine((src_lat, src_lon), (tar_lat, tar_lon), unit=Unit.KILOMETERS)
                # # Code for ST-LSTM
                dist = min(dist * 10, self.s_num)
                s_offset[n] = math.floor(dist)
                s_offset_u[n] = math.ceil(dist)
                s_coef[n] = dist - s_offset[n]

                # Code for STGN
                # s_offset[n] = dist
            
            seq_t_u.append(torch.tensor(t_offset_u, dtype=torch.int64))
            seq_t_l.append(torch.tensor(t_offset, dtype=torch.int64))
            # seq_t_l.append(torch.tensor(t_offset).float())
            coef_t.append(torch.tensor(t_coef).float())
            seq_s_l.append(torch.tensor(s_offset, dtype=torch.int64))
            # seq_s_l.append(torch.tensor(s_offset).float())
            seq_s_u.append(torch.tensor(s_offset_u, dtype=torch.int64))
            coef_s.append(torch.tensor(s_coef).float())
            
        x = torch.stack(seqs, dim=1)
        y = torch.stack(lbls, dim=1)  
        seq_t_l, seq_t_u = torch.stack(seq_t_l), torch.stack(seq_t_u)
        seq_s_l, seq_s_u = torch.stack(seq_s_l), torch.stack(seq_s_u)
        coef_s, coef_t = torch.stack(coef_s), torch.stack(coef_t)

        if self.include_st:
            return x, y, seq_t_l, seq_t_u, coef_t, seq_s_l, seq_s_u, coef_s, reset_h
        else:
            return x, y, reset_h


class STP_Dataloader(DataLoader):
    def __init__(self, mode='train', dataset='gowalla', **kwargs):
        self.u_eindex = pickle.load(open(dataset + '_stp/u_graph.pkl', 'rb'))
        self.u_eindex = torch.tensor(self.u_eindex, dtype=torch.int64)
        self.raw_dataset = pickle.load(open(dataset + '_stp/%s_data.pkl' % mode, 'rb'))
        if dataset == 'gowalla':
            user_set, poi_set = pickle.load(open(USER_SET_GOW, 'rb')), pickle.load(open(POI_SET_GOW, 'rb'))
        else:
            user_set, poi_set = pickle.load(open(USER_SET_FOUR, 'rb')), pickle.load(open(POI_SET_FOUR, 'rb'))
        self.user_num, self.poi_num = len(user_set), len(poi_set)
        self.dataset = np.arange(len(self.raw_dataset), dtype=np.int64).reshape(-1, 1)
        print(len(self.raw_dataset))
        super(STP_Dataloader, self).__init__(self.dataset, collate_fn=self.sample, **kwargs)
    
    def sample(self, batch):
        batch = np.concatenate(batch).reshape(-1)
        full_pp_eindex, full_pp_nid = [], []
        full_s_eindex, full_s_nid = [], []
        full_t_eindex, full_t_nid = [], []
        full_f_eindex, full_f_nid = [], []
        full_s_rw_eindex, full_s_rw_nid = [], []
        full_t_rw_eindex, full_t_rw_nid = [], []
        full_f_rw_eindex, full_f_rw_nid = [], []
        center_pp_nid, center_s_nid, center_t_nid, center_f_nid = [], [], [], []
        center_s_rw_nid, center_t_rw_nid, center_f_rw_nid = [], [], []
        last_pp_offset, last_s_offset, last_t_offset, last_f_offset = 0, 0, 0, 0
        last_s_rw_offset, last_t_rw_offset, last_f_rw_offset = 0, 0, 0
        labels, anchors = [], []
        for idx in batch:
            last_poi, label, tar_user, pp_eindex, s_tar, t_tar, f_tar, s_rw_tar, t_rw_tar, f_rw_tar = self.raw_dataset[idx]
            labels.append(label)
            anchors.append(tar_user)
            pp_eindex = pp_eindex.astype(np.int64)
            # tmp_idx = pp_eindex[0] != pp_eindex[1]
            # pp_eindex = pp_eindex[:, tmp_idx]
            if len(pp_eindex) == 0:
                pp_nid = np.array([last_poi], dtype=np.int64)
            else:
                pp_nid = np.unique(pp_eindex)
            s_nid, t_nid, f_nid, s_rw_nid, t_rw_nid, f_rw_nid = [last_poi], [last_poi], [last_poi], [last_poi], [last_poi], [last_poi]
            s_nid.extend(s_tar.tolist())
            t_nid.extend(t_tar.tolist())
            f_nid.extend(f_tar.tolist())
            s_rw_nid.extend(s_rw_tar.tolist())
            t_rw_nid.extend(t_rw_tar.tolist())
            f_rw_nid.extend(f_rw_tar.tolist())
            s_eindex = np.stack([s_tar, np.full(len(s_tar), last_poi)])
            t_eindex = np.stack([t_tar, np.full(len(t_tar), last_poi)])
            f_eindex = np.stack([f_tar, np.full(len(f_tar), last_poi)])
            s_rw_eindex = np.stack([s_rw_tar, np.full(len(s_rw_tar), last_poi)])
            t_rw_eindex = np.stack([t_rw_tar, np.full(len(t_rw_tar), last_poi)])
            f_rw_eindex = np.stack([f_rw_tar, np.full(len(f_rw_tar), last_poi)])
            s_map, t_map, f_map = np.zeros(self.poi_num), np.zeros(self.poi_num), np.zeros(self.poi_num)
            s_rw_map, t_rw_map, f_rw_map = np.zeros(self.poi_num), np.zeros(self.poi_num), np.zeros(self.poi_num)
            pp_map = np.zeros(self.poi_num)
            s_map[s_nid] = np.arange(len(s_nid))
            t_map[t_nid] = np.arange(len(t_nid))
            f_map[f_nid] = np.arange(len(f_nid))
            s_rw_map[s_rw_nid] = np.arange(len(s_rw_nid))
            t_rw_map[t_rw_nid] = np.arange(len(t_rw_nid))
            f_rw_map[f_rw_nid] = np.arange(len(f_rw_nid))
            pp_map[pp_nid] = np.arange(len(pp_nid))
            s_eindex = s_map[s_eindex]
            t_eindex = t_map[t_eindex]
            f_eindex = f_map[f_eindex]
            s_rw_eindex = s_rw_map[s_rw_eindex]
            t_rw_eindex = t_rw_map[t_rw_eindex]
            f_rw_eindex = f_rw_map[f_rw_eindex]

            full_s_eindex.append(s_eindex + last_s_offset)
            full_f_eindex.append(f_eindex + last_f_offset)
            full_t_eindex.append(t_eindex + last_t_offset)
            full_s_rw_eindex.append(s_rw_eindex + last_s_rw_offset)
            full_t_rw_eindex.append(t_rw_eindex + last_t_rw_offset)
            full_f_rw_eindex.append(f_rw_eindex + last_f_rw_offset)
            if len(pp_eindex) > 0:
                pp_eindex = pp_map[pp_eindex]
                full_pp_eindex.append(pp_eindex + last_pp_offset)
            full_s_nid += s_nid
            full_t_nid += t_nid
            full_f_nid += f_nid
            full_s_rw_nid += s_rw_nid
            full_t_rw_nid += t_rw_nid
            full_f_rw_nid += f_rw_nid
            full_pp_nid += pp_nid.tolist()

            center_s_nid.append(last_s_offset)
            center_t_nid.append(last_t_offset)
            center_f_nid.append(last_f_offset)
            center_s_rw_nid.append(last_s_rw_offset)
            center_t_rw_nid.append(last_t_rw_offset)
            center_f_rw_nid.append(last_f_rw_offset)
            center_pp_nid.append(last_pp_offset + int(np.argwhere(pp_nid == last_poi).squeeze()))

            last_s_offset += len(s_nid)
            last_t_offset += len(t_nid)
            last_f_offset += len(f_nid)
            last_s_rw_offset += len(s_rw_nid)
            last_t_rw_offset += len(t_rw_nid)
            last_f_rw_offset += len(f_rw_nid)
            last_pp_offset += len(pp_nid)

        full_s_eindex = np.concatenate(full_s_eindex, axis=-1)
        full_f_eindex = np.concatenate(full_f_eindex, axis=-1)
        full_t_eindex = np.concatenate(full_t_eindex, axis=-1)
        full_s_rw_eindex = np.concatenate(full_s_rw_eindex, axis=-1)
        full_f_rw_eindex = np.concatenate(full_f_rw_eindex, axis=-1)
        full_t_rw_eindex = np.concatenate(full_t_rw_eindex, axis=-1)
        full_pp_eindex = np.concatenate(full_pp_eindex, axis=-1)

        full_s_eindex, full_f_eindex, full_t_eindex = torch.tensor(full_s_eindex, dtype=torch.int64), torch.tensor(full_f_eindex, dtype=torch.int64), torch.tensor(full_t_eindex, dtype=torch.int64)
        full_s_rw_eindex, full_f_rw_eindex, full_t_rw_eindex = torch.tensor(full_s_rw_eindex, dtype=torch.int64), torch.tensor(full_f_rw_eindex, dtype=torch.int64),  torch.tensor(full_t_rw_eindex, dtype=torch.int64)
        full_pp_eindex = torch.tensor(full_pp_eindex, dtype=torch.int64)
        full_s_nid, full_f_nid, full_t_nid = torch.tensor(full_s_nid, dtype=torch.int64), torch.tensor(full_f_nid, dtype=torch.int64), torch.tensor(full_t_nid, dtype=torch.int64)
        full_s_rw_nid, full_f_rw_nid, full_t_rw_nid = torch.tensor(full_s_rw_nid, dtype=torch.int64), torch.tensor(full_f_rw_nid, dtype=torch.int64), torch.tensor(full_t_rw_nid, dtype=torch.int64)
        full_pp_nid = torch.tensor(full_pp_nid, dtype=torch.int64)

        center_s_nid, center_f_nid, center_t_nid = torch.tensor(center_s_nid, dtype=torch.int64), torch.tensor(center_f_nid, dtype=torch.int64), torch.tensor(center_t_nid, dtype=torch.int64)
        center_s_rw_nid, center_f_rw_nid, center_t_rw_nid = torch.tensor(center_s_rw_nid, dtype=torch.int64),torch.tensor(center_f_rw_nid, dtype=torch.int64),torch.tensor(center_t_rw_nid, dtype=torch.int64)
        center_pp_nid = torch.tensor(center_pp_nid, dtype=torch.int64)

        labels = torch.tensor(labels, dtype=torch.int64)
        anchors = torch.tensor(anchors, dtype=torch.int64)

        return (STP_Adj(full_s_eindex, full_s_nid, center_s_nid), STP_Adj(full_t_eindex, full_t_nid, center_t_nid), STP_Adj(full_f_eindex, full_f_nid, center_f_nid),
                STP_Adj(full_s_rw_eindex, full_s_rw_nid, center_s_rw_nid), STP_Adj(full_t_rw_eindex, full_t_rw_nid, center_t_rw_nid), STP_Adj(full_f_rw_eindex, full_f_rw_nid, center_f_rw_nid),
                STP_Adj(full_pp_eindex, full_pp_nid, center_pp_nid), anchors, labels)