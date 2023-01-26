from datetime import datetime, timedelta
import copy
import time
import math
import pickle
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from model.adj import Adj
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from geopy.distance import geodesic
from haversine import haversine, Unit

def process_gowalla():
    user_ids, loc_info, check_ins = set(), dict(), []
    lower_bound = datetime.strptime('2009-11-01', "%Y-%m-%d")
    upper_bound = datetime.strptime('2010-10-31', "%Y-%m-%d")
    for line in tqdm(open('gowalla/loc-gowalla_totalCheckins.txt', 'r')):
        data = line.strip().split('\t')
        uid, timestamp, lat, lon, lid = data
        if float(lat) > 90 or float(lat) < -90 or float(lon) > 180 or float(lon) < -180:
            continue
        timestamp = datetime.strptime(timestamp.strip(), '%Y-%m-%dT%H:%M:%SZ')
        if timestamp > upper_bound or timestamp < lower_bound:
            continue
        user_ids.add(int(uid))
        loc_info[int(lid)] = [float(lat), float(lon)]
        check_ins.append([int(uid), int(lid), timestamp])
    user_ids = np.array(list(user_ids))
    loc_ids = np.array(list(loc_info.keys()))
    print(len(user_ids), len(loc_ids))
    print(np.max(user_ids), np.max(loc_ids))
    user_map, loc_map = {}, {}
    for i in range(len(user_ids)):
        user_map[user_ids[i]] = i
    for i in range(len(loc_ids)):
        loc_map[loc_ids[i]] = i
    loc_info_aligned = {}
    for i in range(len(loc_ids)):
        loc_info_aligned[i] = loc_info[loc_ids[i]]
    for i in range(len(check_ins)):
        check_ins[i][0], check_ins[i][1] = user_map[check_ins[i][0]], loc_map[check_ins[i][1]]
    print('start data storage')
    np.save('gowalla_processed/user_id.npy', user_ids)
    pickle.dump(loc_info_aligned, open('gowalla_processed/loc_info.pkl', 'wb'))
    pickle.dump(check_ins, open('gowalla_processed/check_ins.pkl', 'wb'))


def gowalla_filter():
    check_ins = pickle.load(open('gowalla_processed/check_ins.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(106812), np.zeros(1279123)
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    uidx = user_freq >= 50
    lidx = loc_freq >= 10
    user_id, loc_id = np.arange(len(user_freq)), np.arange(len(loc_freq))
    filtered_user, filtered_loc = set(user_id[uidx].tolist()), set(loc_id[lidx].tolist())
    filtered_checkins = []
    for entry in check_ins:
        if entry[0] in filtered_user and entry[1] in filtered_loc:
            filtered_checkins.append(entry)
    print(len(filtered_checkins))
    involved_user, involved_poi = set(), set()
    for entry in filtered_checkins:
        involved_poi.add(entry[1])
        involved_user.add(entry[0])
    print(len(involved_user))
    total_seq = {}
    train_seq, val_seq, test_seq = {}, {}, {}
    for entry in filtered_checkins:
        if entry[0] not in total_seq:
            total_seq[entry[0]] = [[entry[1], entry[2]]]
        else:
            total_seq[entry[0]].append([entry[1], entry[2]])
    seen_poi = []
    seen_user = []
    total_checkin = 0
    for key in total_seq.keys():
        if len(total_seq[key]) < 50:
            continue
        seen_user.append(key)
        sorted_list = sorted(total_seq[key], key=lambda x:x[1], reverse=False)
        total_checkin += len(sorted_list)
        train_split, val_split = math.floor(len(sorted_list) * 0.7), math.ceil(len(sorted_list) * 0.8)
        train_list, val_list, test_list = sorted_list[:train_split], sorted_list[train_split:val_split], sorted_list[val_split:]
        train_seq[key] = train_list
        for entry in train_list:
            seen_poi.append(entry[0])
        val_seq[key] = val_list
        test_seq[key] = test_list
    seen_poi = set(seen_poi)
    seen_user = set(seen_user)
    print(len(seen_poi), len(seen_user))
    print(total_checkin)
    val_popkey, test_popkey = [], []
    for n, key in enumerate(val_seq):
        val_list = val_seq[key]
        index = []
        for i in range(len(val_list)):
            if val_list[i][0] not in seen_poi:
                index.append(i)
        val_list = np.delete(val_list, index, axis=0).tolist()
        if len(val_list) > 0:
            val_seq[key] = val_list
        else:
            val_popkey.append(key)
    for key in test_seq:
        test_list = test_seq[key]
        index = []
        for i in range(len(test_list)):
            if test_list[i][0] not in seen_poi:
                index.append(i)
        test_list = np.delete(test_list, index, axis=0).tolist()
        if len(test_list) > 0:
            test_seq[key] = test_list
        else:
            test_popkey.append(key)
    [val_seq.pop(k) for k in val_popkey]
    [test_seq.pop(k) for k in test_popkey]
    # involved_user, seen_poi = list(involved_user), list(seen_poi)
    seen_user, seen_poi = list(seen_user), list(seen_poi)
    pickle.dump(train_seq, open('gowalla_processed/train_seq.pkl', 'wb'))
    pickle.dump(val_seq, open('gowalla_processed/val_seq.pkl', 'wb'))
    pickle.dump(test_seq, open('gowalla_processed/test_seq.pkl', 'wb'))
    pickle.dump(filtered_checkins, open('gowalla_processed/check_ins_pop.pkl', 'wb'))
    pickle.dump(seen_user, open('gowalla_processed/seen_user.pkl', 'wb'))
    pickle.dump(seen_poi, open('gowalla_processed/seen_poi.pkl', 'wb'))
    # pickle.dump([len(involved_user), len(seen_poi), 50], open('gowalla_processed/basic_stat.pkl', 'wb'))



def gowalla_freq_analysis():
    check_ins = pickle.load(open('gowalla_processed/check_ins.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(106812), np.zeros(1279123)  # Original: 10.7w, 128w 
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    user_freq, loc_freq = Counter(user_freq), Counter(loc_freq)
    user_freq = dict(sorted(user_freq.items(),key=lambda x:x[0], reverse=False))
    loc_freq = dict(sorted(loc_freq.items(), key=lambda x:x[0], reverse=False))
    x_user, y_user = list(user_freq.keys()), np.array(list(user_freq.values()), dtype=np.float64)
    for i in range(1, len(y_user)):
        y_user[i] += y_user[i-1]
    y_user /= y_user[-1]
    x_loc, y_loc = list(loc_freq.keys()), np.array(list(loc_freq.values()), dtype=np.float64)
    for i in range(1, len(y_loc)):
        y_loc[i] += y_loc[i-1]
    y_loc /= y_loc[-1]
    plt.figure()
    plt.plot(x_user[:100], y_user[:100])
    plt.xlabel('visit count')
    plt.ylabel('accumulated user ratio')
    plt.grid('--')
    plt.title('User Visit Frequency in Gowalla')
    plt.savefig('gowalla_user_freq.png')
    plt.figure()
    plt.plot(x_loc[:100], y_loc[:100])
    plt.xlabel('visit count')
    plt.ylabel('accumulated loc ratio')
    plt.grid('--')
    plt.title('Location Visit Frequency in Gowalla')
    plt.savefig('gowalla_loc_freq_pop.png')


def gowalla_social():
    edges = []
    user_id = np.load('gowalla_processed/user_id.npy')
    # check_ins_pop = pickle.load(open('gowalla_processed/check_ins_pop.pkl', 'rb'))
    user_set = pickle.load(open('gowalla_processed/seen_user.pkl', 'rb'))
    for line in open('gowalla/loc-gowalla_edges.txt', 'r'):
        splits = line.strip().split('\t')
        edges.append([int(splits[0]), int(splits[1])])
    print(len(edges))
    user_map = {}
    for i in range(len(user_id)):
        user_map[user_id[i]] = i
    for i in range(len(edges)):
        try:
            edges[i][0], edges[i][1] = user_map[edges[i][0]], user_map[edges[i][1]]
        except KeyError:
            # print(edges[i])
            continue

    # involved_user = set()
    # for entry in check_ins_pop:
    #     involved_user.add(entry[0])
    involved_user = set(user_set)
    print(len(involved_user))
    filtered_edges = []
    for edge in edges:
        if edge[0] in involved_user and edge[1] in involved_user:
            filtered_edges.append(edge)
    print(len(filtered_edges))
    pickle.dump(filtered_edges, open('gowalla_processed/soc_edge.pkl', 'wb'))

    # TODO: Align user id
    alpha = 0.1
    k = 10
    seen_users = pickle.load(open('gowalla_processed/seen_user.pkl', 'rb'))
    soc_edges = pickle.load(open('gowalla_processed/soc_edge.pkl', 'rb'))
    aligned_soc_edges = []
    user_map = {}
    for i in range(len(seen_users)):
        user_map[seen_users[i]] = i
    for edge in soc_edges:
        aligned_soc_edges.append([user_map[edge[0]], user_map[edge[1]]])
    print(len(aligned_soc_edges))
    pickle.dump(aligned_soc_edges, open('gowalla_processed/soc_edge.pkl', 'wb'))

    adj_map = {}
    for edge in aligned_soc_edges:
        src, tar = edge
        if src not in adj_map:
            adj_map[src] = [tar]
        else:
            adj_map[src].append(tar)
        # if tar not in adj_map:
        #     adj_map[tar] = [src]
        # else:
        #     adj_map[tar].append(src)
    degs = np.zeros(len(seen_users))
    for key in adj_map:
        degs[key] = len(adj_map[key])
    # # Max degree: 8193, median degree: 7
    print("Degree Data:")
    print(len(degs[degs >= 10]))
    print(len(degs[degs >= 20]))
    print(np.max(degs))
    print(np.median(degs))
    print(np.min(degs))

    num_nodes = len(seen_users)
    aligned_soc_edges = torch.tensor(aligned_soc_edges, dtype=torch.int64).t()
    ppr_eindex, ppr_eweight = gcn_norm(aligned_soc_edges, None, num_nodes, add_self_loops=False)
    soc_adj = SparseTensor(row=aligned_soc_edges[0], col=aligned_soc_edges[1], value=ppr_eweight, sparse_sizes=(num_nodes, num_nodes)).to_torch_sparse_coo_tensor().to_dense()
    unit_adj = torch.sparse_coo_tensor((range(num_nodes), range(num_nodes)), [1.]*num_nodes).to_dense()
    diffusion_adj = alpha * unit_adj # + alpha * (1-alpha) * soc_adj + alpha * (1-alpha) * (1-alpha) * torch.sparse.mm(ppr_adj, ppr_adj)
    coef, cur_adj = alpha, unit_adj
    print(diffusion_adj.size(), unit_adj.size())
    print("Start APPR")
    for i in range(1, k):
        print(i)
        diffusion_adj = diffusion_adj + coef * (1 - alpha) * (torch.mm(cur_adj, soc_adj))# torch.sparse.mm(cur_adj, soc_adj)
        cur_adj = torch.mm(cur_adj, soc_adj)
    # diffusion_adj = diffusion_adj.coalesce().to_dense()
    np.save('gowalla_processed/diff_soc_graph.npy', diffusion_adj.numpy())
    # diffusion_adj = torch.tensor(np.load('gowalla_processed/diff_soc_graph.npy'))
    sorted_idx = torch.argsort(diffusion_adj, dim=-1, descending=True)
    print(sorted_idx.size())
    diffusion_adj[diffusion_adj < 0.002] = 0
    row = torch.arange(num_nodes, dtype=torch.int64)
    col = sorted_idx[:, 30]
    # end_idx = torch.cat([torch.arange(num_nodes, dtype=torch.int64), sorted_idx[:, 30]]).reshape(2, -1)
    # print(end_idx.size())
    print(torch.nonzero(diffusion_adj[row, sorted_idx[:, 20]]).size())
    print(torch.nonzero(diffusion_adj[row, col]).size())
    print(torch.nonzero(diffusion_adj[row, sorted_idx[:, 10]]).size())
    diff_soc_edges = []
    sorted_idx = sorted_idx.numpy()
    for i in range(num_nodes):
        cur_num = 0
        for j in range(num_nodes):
            if cur_num == 10:
                break
            cur_y = sorted_idx[i, j]
            if i == cur_y:
                continue
            if diffusion_adj[i, cur_y] == 0:
                break
            diff_soc_edges.append([i, cur_y])
            cur_num += 1
    print(len(diff_soc_edges))
    soc_edges = {}
    for edge in diff_soc_edges:
        src, tar = edge
        if src not in soc_edges:
            soc_edges[src] = [tar]
        else:
            soc_edges[src].append(tar)
    pickle.dump(soc_edges, open('gowalla_processed/diff_soc_edge.pkl', 'wb'))
    # eindex, eweight = diffusion_adj.indices(), diffusion_adj.values()
    # print(torch.min(eweight), torch.max(eweight))

    # row, col = eindex[0], eindex[1]
    
    # idx = torch.abs(eweight) > 0.005
    # row, col = row[idx], col[idx]

def gowalla_traj_analysis():
    check_ins = pickle.load(open('gowalla_processed/check_ins.pkl', 'rb'))
    individual_checkin, timestamps = {}, []
    for entry in check_ins:
        timestamps.append(entry[2].date())
        if entry[0] in individual_checkin:
            individual_checkin[entry[0]].append([entry[1], entry[2]])
        else:
            individual_checkin[entry[0]] = [[entry[1], entry[2]]]
    time_count = Counter(timestamps)
    time_count = dict(sorted(time_count.items(),key=lambda x:x[0], reverse=False))
    x_time, y_time = list(time_count.keys()), np.array(list(time_count.values()), dtype=np.float64)
    plt.figure()
    plt.plot(x_time, y_time)
    plt.gcf().autofmt_xdate()
    plt.xlabel('date')
    plt.ylabel('record number')
    plt.grid('--')
    plt.title('Check-in Timestamp in Gowalla')
    plt.savefig('gowalla_timestamp.png')

    traj_len = []
    lower_bound = datetime.strptime('2009-11-01', "%Y-%m-%d")
    upper_bound = datetime.strptime('2010-11-01', "%Y-%m-%d")
    maximum_delta = timedelta(hours=8)
    for _, records in tqdm(individual_checkin.items()):
        records = sorted(records, key=lambda x:x[1], reverse=False)
        last_ts = records[0][1]
        cur_traj_len = 1
        for i in range(1, len(records)):
            cur_ts = records[i][1]
            if cur_ts.date() < lower_bound.date():
                continue
            elif cur_ts.date() > upper_bound.date():
                break
            
            # # For Daily trajectories
            # if last_ts.date() == cur_ts.date():
            #     cur_traj_len += 1
            # elif last_ts.date() != cur_ts.date() and last_ts.date() >= lower_bound.date():
            #     traj_len.append(cur_traj_len)
            #     last_ts = cur_ts
            #     cur_traj_len = 1
            # else:
            #     last_ts = cur_ts

            # For timespan based trajectories
            if cur_ts - last_ts <= maximum_delta:
                cur_traj_len += 1
            elif cur_ts - last_ts > maximum_delta and last_ts.date() >= lower_bound.date():
                traj_len.append(cur_traj_len)
                cur_traj_len = 1
            last_ts = cur_ts
        traj_len.append(cur_traj_len)
    traj_len = Counter(traj_len)
    traj_len = dict(sorted(traj_len.items(),key=lambda x:x[0], reverse=False))
    x_traj, y_traj = list(traj_len.keys()), np.array(list(traj_len.values()), dtype=np.float64)
    for i in range(1, len(y_traj)):
        y_traj[i] += y_traj[i-1]
    y_traj /= y_traj[-1]
    plt.figure()
    plt.plot(x_traj[:20], y_traj[:20])
    # plt.gcf().autofmt_xdate()
    plt.xlabel('trajectory length')
    plt.ylabel('accmulated ratio')
    plt.grid('--')
    plt.title('Timespan Based Trajectory Length in Gowalla')
    plt.savefig('gowalla_traj_timespan.png')


def poi_geography_analysis():
    check_ins = pickle.load(open('gowalla_processed/check_ins.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(107092), np.zeros(1280969)
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    loc_info = pickle.load(open('gowalla_processed/loc_info.pkl', 'rb'))
    x_loc, y_loc, label = [], [], []
    for key, val in loc_info.items():
        x_loc.append(val[0])
        y_loc.append(val[1])
        if loc_freq[key] > 5:
            label.append(0)
        else:
            label.append(1)
    x_loc, y_loc, label = np.array(x_loc), np.array(y_loc), np.array(label)
    plt.figure()
    # plt.scatter(y_loc[label == 0], x_loc[label == 0], s=1)
    # plt.scatter(y_loc[label == 1], x_loc[label == 1], s=1)
    plt.scatter(y_loc, x_loc, s=1)
    plt.grid('--')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Gowalla POI')
    plt.savefig('loc_geo_gowalla.png')

    check_ins = pickle.load(open('foursquare_processed/check_ins.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(114324), np.zeros(3820891)
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    loc_info = pickle.load(open('foursquare_processed/loc_info.pkl', 'rb'))
    x_loc, y_loc, label = [], [], []
    for key, val in loc_info.items():
        x_loc.append(val[0])
        y_loc.append(val[1])
        if loc_freq[key] > 5:
            label.append(0)
        else:
            label.append(1)
    plt.figure()
    x_loc, y_loc, label = np.array(x_loc), np.array(y_loc), np.array(label)
    # plt.scatter(y_loc[label == 0], x_loc[label == 0], s=1)
    # plt.scatter(y_loc[label == 1], x_loc[label == 1], s=1)
    plt.scatter(y_loc, x_loc, s=1)
    plt.grid('--')
    plt.xlabel('longitude')
    plt.ylabel('latitude')
    plt.title('Foursquare POI')
    plt.savefig('loc_geo_foursquare.png')



def process_foursquare():
    user_ids, loc_ids, check_ins = set(), set(), []
    lower_bound = datetime.strptime('2012-04-01', "%Y-%m-%d")
    upper_bound = datetime.strptime('2013-03-31', "%Y-%m-%d")
    loc_info = {}
    for line in tqdm(open('foursquare/checkins.txt')):
        data = line.strip().split('\t')
        uid, lid, timestamp, offset = data
        try:
            timestamp = datetime.strptime(timestamp.strip(), '%a %b %d %H:%M:%S +0000 %Y')
        except ValueError:
            print(timestamp)
            continue
        if timestamp > upper_bound or timestamp < lower_bound:
            continue
        seed = random.random()
        if seed > 0.4:
            continue
        user_ids.add(int(uid))
        loc_ids.add(lid)
        delta = timedelta(seconds=float(offset))
        timestamp = timestamp + delta
        check_ins.append([int(uid), lid, timestamp])
    
    for line in tqdm(open('foursquare/raw_POIs.txt', 'r')):
        data = line.strip().split('\t')
        lid, lat, lon, category, country = data
        if lid in loc_ids:
            loc_info[lid] = [float(lat), float(lon)]# , category, country]
    
    user_ids, loc_ids = np.array(list(user_ids)), list(loc_ids)
    print(len(user_ids))
    print(len(loc_ids))
    print(np.max(user_ids))
    user_map, loc_map = {}, {}
    for i in range(len(user_ids)):
        user_map[user_ids[i]] = i
    for i in range(len(loc_ids)):
        loc_map[loc_ids[i]] = i
    loc_info_aligned = {}
    for i in range(len(loc_ids)):
        loc_info_aligned[i] = loc_info[loc_ids[i]]
    for i in range(len(check_ins)):
        check_ins[i][0], check_ins[i][1] = user_map[check_ins[i][0]], loc_map[check_ins[i][1]]
    print('start data storage')
    np.save('foursquare_processed/user_id.npy', user_ids)
    pickle.dump(loc_ids, open('foursquare_processed/involved_loc_id.pkl', 'wb'))
    pickle.dump(loc_info_aligned, open('foursquare_processed/loc_info.pkl', 'wb'))
    pickle.dump(check_ins, open('foursquare_processed/check_ins.pkl', 'wb'))


def foursquare_freq_analysis():
    check_ins = pickle.load(open('foursquare_processed/check_ins_pop.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(113666), np.zeros(3166484)   # Original: 114324, 3820891
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    user_freq, loc_freq = Counter(user_freq), Counter(loc_freq)
    user_freq = dict(sorted(user_freq.items(),key=lambda x:x[0], reverse=False))
    loc_freq = dict(sorted(loc_freq.items(), key=lambda x:x[0], reverse=False))
    x_user, y_user = list(user_freq.keys()), np.array(list(user_freq.values()), dtype=np.float64)
    for i in range(1, len(y_user)):
        y_user[i] += y_user[i-1]
    y_user /= y_user[-1]
    x_loc, y_loc = list(loc_freq.keys()), np.array(list(loc_freq.values()), dtype=np.float64)
    for i in range(1, len(y_loc)):
        y_loc[i] += y_loc[i-1]
    y_loc /= y_loc[-1]
    plt.figure()
    plt.plot(x_user[:100], y_user[:100])
    plt.xlabel('visit count')
    plt.ylabel('accumulated user ratio')
    plt.grid('--')
    plt.title('User Visit Frequency in Foursquare')
    plt.savefig('foursquare_user_freq.png')
    plt.figure()
    plt.plot(x_loc[:100], y_loc[:100])
    plt.xlabel('visit count')
    plt.ylabel('accumulated loc ratio')
    plt.grid('--')
    plt.title('Location Visit Frequency in Foursquare')
    plt.savefig('foursquare_loc_freq_pop.png')


def foursquare_filter():
    check_ins = pickle.load(open('foursquare_processed/check_ins.pkl', 'rb'))
    user_freq, loc_freq = np.zeros(113666), np.zeros(3166484)
    for entry in check_ins:
        user_freq[entry[0]] += 1
        loc_freq[entry[1]] += 1
    uidx = user_freq >= 50
    lidx = loc_freq >= 10
    user_id, loc_id = np.arange(len(user_freq)), np.arange(len(loc_freq))
    filtered_user, filtered_loc = set(user_id[uidx].tolist()), set(loc_id[lidx].tolist())
    filtered_checkins = []
    for entry in check_ins:
        if entry[0] in filtered_user and entry[1] in filtered_loc:
            filtered_checkins.append(entry)
    print(len(filtered_checkins))
    involved_user, involved_poi = set(), set()
    for entry in filtered_checkins:
        involved_poi.add(entry[1])
        involved_user.add(entry[0])
    # print(len(involved_user))
    total_seq = {}
    train_seq, val_seq, test_seq = {}, {}, {}
    for entry in filtered_checkins:
        if entry[0] not in total_seq:
            total_seq[entry[0]] = [[entry[1], entry[2]]]
        else:
            total_seq[entry[0]].append([entry[1], entry[2]])
    seen_poi = []
    total_checkin = 0
    for key in total_seq.keys():
        if len(total_seq[key]) < 50:
            continue
        sorted_list = sorted(total_seq[key], key=lambda x:x[1], reverse=False)
        total_checkin += len(sorted_list)
        train_split, val_split = math.floor(len(sorted_list) * 0.7), math.ceil(len(sorted_list) * 0.8)
        train_list, val_list, test_list = sorted_list[:train_split], sorted_list[train_split:val_split], sorted_list[val_split:]
        train_seq[key] = train_list
        for entry in train_list:
            seen_poi.append(entry[0])
        val_seq[key] = val_list
        test_seq[key] = test_list
    seen_poi = set(seen_poi)
    print(len(seen_poi))
    print(len(list(train_seq.keys())))
    print(total_checkin)
    val_popkey, test_popkey = [], []
    for key in val_seq:
        val_list = val_seq[key]
        index = []
        for i in range(len(val_list)):
            if val_list[i][0] not in seen_poi:
                index.append(i)
        val_list = np.delete(val_list, index, axis=0).tolist()
        if len(val_list) > 0:
            val_seq[key] = val_list
        else:
            val_popkey.append(key)
    for key in test_seq:
        test_list = test_seq[key]
        index = []
        for i in range(len(test_list)):
            if test_list[i][0] not in seen_poi:
                index.append(i)
        test_list = np.delete(test_list, index, axis=0).tolist()
        if len(test_list) > 0:
            test_seq[key] = test_list
        else:
            test_popkey.append(key)
    [val_seq.pop(k) for k in val_popkey]
    [test_seq.pop(k) for k in test_popkey]
    involved_user, seen_poi = list(involved_user), list(seen_poi)
    pickle.dump(train_seq, open('foursquare_processed/train_seq.pkl', 'wb'))
    pickle.dump(val_seq, open('foursquare_processed/val_seq.pkl', 'wb'))
    pickle.dump(test_seq, open('foursquare_processed/test_seq.pkl', 'wb'))
    pickle.dump(filtered_checkins, open('foursquare_processed/check_ins_pop.pkl', 'wb'))
    pickle.dump(list(train_seq.keys()), open('foursquare_processed/seen_user.pkl', 'wb'))
    pickle.dump(seen_poi, open('foursquare_processed/seen_poi.pkl', 'wb'))


def foursquare_traj_analysis():
    check_ins = pickle.load(open('foursquare_processed/check_ins.pkl', 'rb'))
    individual_checkin, timestamps = {}, []
    for entry in check_ins:
        timestamps.append(entry[2].date())
        if entry[0] in individual_checkin:
            individual_checkin[entry[0]].append([entry[1], entry[2]])
        else:
            individual_checkin[entry[0]] = [[entry[1], entry[2]]]
    time_count = Counter(timestamps)
    time_count = dict(sorted(time_count.items(),key=lambda x:x[0], reverse=False))
    x_time, y_time = list(time_count.keys()), np.array(list(time_count.values()), dtype=np.float64)
    plt.figure()
    plt.plot(x_time, y_time)
    plt.gcf().autofmt_xdate()
    plt.xlabel('date')
    plt.ylabel('record number')
    plt.grid('--')
    plt.title('Check-in Timestamp in Foursquare')
    plt.savefig('foursquare_timestamp.png')

    traj_len = []
    lower_bound = datetime.strptime('2012-04-01', "%Y-%m-%d")
    upper_bound = datetime.strptime('2013-04-01', "%Y-%m-%d")
    maximum_delta = timedelta(hours=8)
    for _, records in tqdm(individual_checkin.items()):
        records = sorted(records, key=lambda x:x[1], reverse=False)
        last_ts = records[0][1]
        cur_traj_len = 1
        for i in range(1, len(records)):
            cur_ts = records[i][1]
            if cur_ts.date() < lower_bound.date():
                continue
            elif cur_ts.date() > upper_bound.date():
                break
            
            # # For Daily trajectories
            # if last_ts.date() == cur_ts.date():
            #     cur_traj_len += 1
            # elif last_ts.date() != cur_ts.date() and last_ts.date() >= lower_bound.date():
            #     traj_len.append(cur_traj_len)
            #     last_ts = cur_ts
            #     cur_traj_len = 1
            # else:
            #     last_ts = cur_ts

            # For timespan based trajectories
            if cur_ts - last_ts <= maximum_delta:
                cur_traj_len += 1
            elif cur_ts - last_ts > maximum_delta and last_ts.date() >= lower_bound.date():
                traj_len.append(cur_traj_len)
                cur_traj_len = 1
            last_ts = cur_ts
        traj_len.append(cur_traj_len)
    traj_len = Counter(traj_len)
    traj_len = dict(sorted(traj_len.items(),key=lambda x:x[0], reverse=False))
    x_traj, y_traj = list(traj_len.keys()), np.array(list(traj_len.values()), dtype=np.float64)
    for i in range(1, len(y_traj)):
        y_traj[i] += y_traj[i-1]
    y_traj /= y_traj[-1]
    plt.figure()
    plt.plot(x_traj[:20], y_traj[:20])
    # plt.gcf().autofmt_xdate()
    plt.xlabel('trajectory length')
    plt.ylabel('accmulated ratio')
    plt.grid('--')
    plt.title('Timespan Based Trajectory Length in Foursquare')
    plt.savefig('foursquare_traj_timespan.png')


def foursquare_social():
    edges = []
    user_id = np.load('foursquare_processed/user_id.npy')
    user_set = pickle.load(open('foursquare_processed/seen_user.pkl', 'rb'))
    for line in open('foursquare/friendship_old.txt', 'r'):
        splits = line.strip().split('\t')
        edges.append([int(splits[0]), int(splits[1])])
    print(len(edges))
    user_map = {}
    for i in range(len(user_id)):
        user_map[user_id[i]] = i
    for i in range(len(edges)):
        try:
            edges[i][0], edges[i][1] = user_map[edges[i][0]], user_map[edges[i][1]]
        except KeyError:
            continue

    involved_user = set(user_set)
    print(len(involved_user))
    filtered_edges = []
    for edge in edges:
        if edge[0] in involved_user and edge[1] in involved_user:
            filtered_edges.append(edge)
    print(len(filtered_edges))
    pickle.dump(filtered_edges, open('foursquare_processed/soc_edge.pkl', 'wb'))

    seen_users = pickle.load(open('foursquare_processed/seen_user.pkl', 'rb'))
    soc_edges = pickle.load(open('foursquare_processed/soc_edge.pkl', 'rb'))
    aligned_soc_edges = []
    user_map = {}
    for i in range(len(seen_users)):
        user_map[seen_users[i]] = i
    for edge in soc_edges:
        aligned_soc_edges.append([user_map[edge[0]], user_map[edge[1]]])
    print(len(aligned_soc_edges))
    pickle.dump(aligned_soc_edges, open('foursquare_processed/soc_edge.pkl', 'wb'))


def seq_align(dataset):
    user_set, poi_set = pickle.load(open(dataset + '_processed/seen_user.pkl', 'rb')), pickle.load(open(dataset+'_processed/seen_poi.pkl', 'rb'))
    user_map, poi_map = {}, {}
    for i in range(len(user_set)):
        user_map[user_set[i]] = i
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    train_seq = pickle.load(open(dataset+'_processed/train_seq.pkl', 'rb'))
    val_seq, test_seq = pickle.load(open(dataset+'_processed/val_seq.pkl', 'rb')), pickle.load(open(dataset+'_processed/test_seq.pkl', 'rb'))
    train_align, val_align, test_align = {}, {}, {}
    for key in train_seq:
        align_key = user_map[key]
        align_seq = []
        for entry in train_seq[key]:
            entry[0] = poi_map[entry[0]]
            align_seq.append(entry)
        train_align[align_key] = align_seq
    for key in val_seq:
        align_key = user_map[key]
        align_seq = []
        for entry in val_seq[key]:
            entry[0] = poi_map[entry[0]]
            align_seq.append(entry)
        val_align[align_key] = align_seq
    for key in test_seq:
        align_key = user_map[key]
        align_seq = []
        for entry in test_seq[key]:
            entry[0] = poi_map[entry[0]]
            align_seq.append(entry)
        test_align[align_key] = align_seq
    pickle.dump(train_align, open(dataset+'_processed/train_seq_align.pkl', 'wb'))
    pickle.dump(val_align, open(dataset+'_processed/val_seq_align.pkl', 'wb'))
    pickle.dump(test_align, open(dataset+'_processed/test_seq_align.pkl', 'wb'))

    # # Negative Sampling
    # neg_num, neg_num_test = 99, 99
    # train_neg, val_neg, test_neg = {}, {}, {}
    # for key in tqdm(train_align.keys()):
    #     seq = train_align[key]
    #     if len(seq) == 0:
    #         continue
    #     tar = seq[-1][0]
    #     neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     while tar in neg_sample:
    #         neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     neg_sample = neg_sample + len(user_set)
    #     train_neg[key] = neg_sample.tolist()
    # for key in tqdm(val_align.keys()):
    #     seq = val_align[key]
    #     tar = seq[-1][0]
    #     neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     while tar in neg_sample:
    #         neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     neg_sample = neg_sample + len(user_set)
    #     val_neg[key] = neg_sample.tolist()
    # for key in tqdm(test_align.keys()):
    #     seq = test_align[key]
    #     tar = seq[-1][0]
    #     neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     while tar in neg_sample:
    #         neg_sample = np.random.choice(len(poi_set), (neg_num_test, ), replace=False)
    #     neg_sample = neg_sample + len(user_set)
    #     test_neg[key] = neg_sample.tolist()
    # pickle.dump(train_neg, open(dataset+'_processed/train_neg.pkl', 'wb'))
    # pickle.dump(val_neg, open(dataset+'_processed/val_neg.pkl', 'wb'))
    # pickle.dump(test_neg, open(dataset + '_processed/test_neg.pkl', 'wb'))


def interval_statistics():
    train_seq = pickle.load(open('foursquare_processed/train_seq_align.pkl', 'rb'))
    poi_set, user_set = pickle.load(open('foursquare_processed/seen_poi.pkl', 'rb')), pickle.load(open('foursquare_processed/seen_user.pkl', 'rb'))

    max_len = 51
    poi_num, user_num = len(poi_set), len(user_set)
    sequences, timestamp, uid = [], [], []
    for i, key in enumerate(train_seq.keys()):
        cur_seq = np.array(train_seq[key][-max_len:])
        if len(cur_seq) == 0:
            continue
        seq, ts = cur_seq[:, 0].tolist(), cur_seq[:, 1].tolist()
        sequences.append(seq[:-1])
        timestamp.append(ts[:-1])
        uid.append(key)

    user_seq, user_ts = [[] for _ in range(poi_num)], [[] for _ in range(poi_num)]
    for user_id, seq, ts in zip(uid, sequences, timestamp):
        for i in range(len(seq)):
            user_seq[seq[i]].append(user_id)
            user_ts[seq[i]].append(ts[i])
    
    ucenter_time_offset, icenter_time_offset = [], []
    for user_id, seq, ts in zip(uid, sequences, timestamp):
        for i in range(1, len(seq)-1):
            # td = ts[i] - ts[i-1]
            # num_hour = td.days * 24 + td.seconds // 3600
            td = ts[-1] - ts[i]
            num_hour = (td.days * 24 + td.seconds // 3600) // 12
            ucenter_time_offset.append(num_hour)  
    for ts in user_ts:
        ts = sorted(ts, reverse=False)
        for i in range(1, len(ts)-1):
            # td = ts[i] - ts[i-1]
            # num_hour = td.days * 24 + td.seconds // 3600
            td = ts[-1] - ts[i]
            # num_hour = td.days
            num_hour = (td.days * 24 + td.seconds // 3600) // 24
            icenter_time_offset.append(num_hour)  

    # print(ucenter_time_offset[:5])
    hour_count = Counter(icenter_time_offset)
    hour_count = dict(sorted(hour_count.items(),key=lambda x:x[0], reverse=False))
    x_u, y_u = list(hour_count.keys()), np.array(list(hour_count.values()), dtype=np.float64)
    for i in range(1, len(y_u)):
        y_u[i] += y_u[i-1]
    y_u /= y_u[-1]
    plt.figure()
    plt.plot(x_u[:360], y_u[:360]) 
    plt.xlabel('interval(hour)')
    plt.ylabel('accmulated ratio')
    plt.grid('--')
    plt.title('Intra-Sequence Time Interval in Gowalla')
    plt.savefig('4sq_interval_i.png')


# For Flashback and LSTPM
def pkl2txt(dataset):
    train_seq, val_seq, test_seq = pickle.load(open(dataset+'_processed/train_seq.pkl', 'rb')), pickle.load(open(dataset+'_processed/val_seq.pkl', 'rb')), pickle.load(open(dataset+'_processed/test_seq.pkl', 'rb'))
    loc_info = pickle.load(open(dataset+'_processed/loc_info.pkl', 'rb'))
    for key in val_seq.keys():
        train_seq[key] = train_seq[key] + val_seq[key]
    for key in test_seq.keys():
        if key not in train_seq:
            continue
        else:
            train_seq[key] = train_seq[key] + test_seq[key]
    with open(dataset+'_customized.txt', 'w') as f:
        for key in tqdm(train_seq.keys()):
            check_ins = train_seq[key]
            uid = str(key)
            for poi_id, ts in check_ins:
                ts_in_date = ts.strftime('%Y-%m-%dT%H:%M:%SZ')
                lat, lon = loc_info[poi_id]
                f.write("\t".join([uid, ts_in_date, str(lat), str(lon), str(poi_id)])+'\n')


# For STAN
def pkl2npy(dataset):
    train_seq, val_seq, test_seq = pickle.load(open(dataset+'_processed/train_seq_align.pkl', 'rb')), pickle.load(open(dataset+'_processed/val_seq_align.pkl', 'rb')), pickle.load(open(dataset+'_processed/test_seq_align.pkl', 'rb'))
    loc_info = pickle.load(open(dataset+'_processed/loc_info.pkl', 'rb'))
    poi_set = pickle.load(open(dataset + '_processed/seen_poi.pkl', 'rb'))
    print(len(poi_set))
    poi_map = {}
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    
    for key in val_seq.keys():
        train_seq[key] = train_seq[key] + val_seq[key]
    for key in test_seq.keys():
        if key not in train_seq:
            continue
        else:
            train_seq[key] = train_seq[key] + test_seq[key]
    check_ins, pois = [], []
    included_user, included_poi = set(), set()
    for key in train_seq.keys():
        seed = random.random()
        if seed > 0.01:
            continue
        included_user.add(key)
        traj = train_seq[key]
        for poi_id, ts in traj:
            included_poi.add(poi_id)
            ts_delta = ts - datetime(2000, 1, 1)
            ts_in_min = ts_delta.total_seconds() // 60
            check_ins.append([key, poi_id, ts_in_min])
    check_ins = np.array(check_ins, dtype=np.int32)
    sampled_user_map, sampled_poi_map = {}, {}
    included_user = list(included_user)
    for i in range(len(included_user)):
        sampled_user_map[included_user[i]] = i + 1
    seen_poi_list = list(included_poi)
    for i in range(len(seen_poi_list)):
        sampled_poi_map[seen_poi_list[i]] = i + 1
    
    for key in loc_info.keys():
        if key in poi_map and poi_map[key] in included_poi:
            pois.append([sampled_poi_map[poi_map[key]], float(loc_info[key][0]), float(loc_info[key][1])])
    print(len(pois))
    print(len(included_user))
    for i in range(len(check_ins)):
        check_ins[i, 0] = sampled_user_map[check_ins[i, 0]]
        check_ins[i, 1] = sampled_poi_map[check_ins[i, 1]]
    idx = np.argsort(check_ins[:, 0])
    check_ins = check_ins[idx, :]
    print(check_ins[-1])
    np.save(dataset + '.npy', check_ins)
    np.save(dataset + '_POI.npy', pois)


def precompute_egonet(dataset, n_layer, seq_max_len, mode):
    '''
    dataset: ['gowalla', 'foursquare']
    n_layer: fixed to 2 (If layer number > 2, the time complexity will be intolerable)
    seq_max_len: the maximum neighbor sampled for each anchor node
    mode: ['train', 'test']. Both mode need to be conducted for once
    '''
    # TODO: add transition graph
    user_set, poi_set = pickle.load(open(dataset+'_processed/seen_user.pkl', 'rb')), pickle.load(open(dataset+'_processed/seen_poi.pkl', 'rb'))
    user_num, poi_num = len(user_set), len(poi_set)

    poi_info = pickle.load(open(dataset + '_processed/loc_info.pkl', 'rb'))
    poi_map = {}
    geo_info = np.zeros((len(poi_set), 2))
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    for id in poi_set:
        entry = poi_info[id]
        lat, lon = entry
        geo_info[poi_map[id]] = [lat, lon]
    
    poi_seqs = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb'))
    val_patch = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
    test_patch = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))
    train_seq_len = {}
    for key in poi_seqs.keys():
        train_seq_len[key] = len(poi_seqs[key])
    involved_users = list(poi_seqs.keys())

    trans_graph = pickle.load(open(dataset + '_processed/transition_graph.pkl', 'rb'))
    # soc_edges = pickle.load(open(dataset + '_processed/diff_soc_edge.pkl', 'rb'))
    # soc_edges = pickle.load(open(dataset + '_processed/soc_edge.pkl', 'rb'))

    # soc_edges = pickle.load(open(dataset + '_processed/user_trans_graph.pkl', 'rb'))

    if mode == 'train':
        all_train_egos, train_uid = subgraph_sample(poi_seqs, user_num, poi_num, seq_max_len, n_layer,
                                                    involved_users, trans_graph=trans_graph, geo_info=geo_info)
    else:
        involved_users = []
        applied_seq_len = {}
        for key in val_patch.keys():
            if len(val_patch[key]) == 0:
                continue
            involved_users.append(key)
            poi_seqs[key] = poi_seqs[key] + val_patch[key]
        for key in poi_seqs.keys():
            applied_seq_len[key] = len(poi_seqs[key])
        all_val_egos, val_uid = subgraph_sample(poi_seqs, user_num, poi_num, seq_max_len, n_layer, involved_users,
                                                train_seq_len, applied_seq_len, trans_graph=trans_graph, geo_info=geo_info)

        train_seq_len = applied_seq_len
        applied_seq_len = {}
        involved_users = []
        for key in test_patch.keys():
            if key not in poi_seqs or len(test_patch[key]) == 0:
                continue
            involved_users.append(key)
            poi_seqs[key] = poi_seqs[key] + test_patch[key]
        for key in poi_seqs.keys():
            applied_seq_len[key] = len(poi_seqs[key])
        all_test_egos, test_uid = subgraph_sample(poi_seqs, user_num, poi_num, seq_max_len, n_layer, involved_users, 
                                                    train_seq_len, applied_seq_len, trans_graph=trans_graph, geo_info=geo_info)
    
    # pickle.dump(all_train_egos, open(dataset+'_processed/train_ego_nets_hetero_poi_geo_v2.pkl', 'wb'))
    # print('Done')
    # pickle.dump(all_val_egos, open(dataset + '_processed/val_ego_nets_hetero_poi_geo_v2.pkl', 'wb'))
    # print('Val Done')
    # pickle.dump(all_test_egos, open(dataset + '_processed/test_ego_nets_hetero_poi_geo_v2.pkl', 'wb'))
    # print('Test Done')

    # all_train_egos = pickle.load(open(dataset + '_processed/train_ego_nets_hetero_poi_geo_v2.pkl', 'rb'))
    # all_val_egos = pickle.load(open(dataset + '_processed/val_ego_nets_hetero_poi_geo_v2.pkl', 'rb'))
    # all_test_egos = pickle.load(open(dataset + '_processed/test_ego_nets_hetero_poi_geo_v2.pkl', 'rb'))

    train_egos_list, val_egos_list, test_egos_list = [], [], []
    if mode == 'train':
        for i, uid in tqdm(enumerate(all_train_egos.keys())):
            cur_out = all_train_egos[uid]
            for i in range(len(cur_out)):
                adjs, nid, sample_pair, _ = cur_out[i]
                cur_target = sample_pair[1]
                neg_samples = np.random.choice(poi_num, 2, replace=False)
                neg_samples = neg_samples + user_num
                cur_sample = [np.array([uid, cur_target]), neg_samples]
                cur_sample = np.concatenate(cur_sample).reshape(1, -1)
                train_egos_list.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))
        print(len(train_egos_list))
        pickle.dump(train_egos_list, open(dataset+'_processed/train_ego_list_v2.pkl', 'wb'))
    else:
        # # val_contain_soc, test_contain_soc = 0, 0
        for i, uid in tqdm(enumerate(all_val_egos.keys())):
            cur_out = all_val_egos[uid]
            for i in range(len(cur_out)):
                adjs, nid, sample_pair, _ = cur_out[i]
                # if soc_adjs is not None:
                #     val_contain_soc += 1
                cur_target = sample_pair[1]
                neg_samples = np.random.choice(poi_num, 2, replace=False)
                neg_samples = neg_samples + user_num
                cur_sample = [np.array([uid, cur_target]), neg_samples]
                cur_sample = np.concatenate(cur_sample).reshape(1, -1)
                val_egos_list.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))

        for i, uid in tqdm(enumerate(all_test_egos.keys())):
            cur_out = all_test_egos[uid]
            for i in range(len(cur_out)):
                adjs, nid, sample_pair, _ = cur_out[i]
                # if soc_adjs is not None:
                #     test_contain_soc += 1
                cur_target = sample_pair[1]
                neg_samples = np.random.choice(poi_num, 2, replace=False)
                neg_samples = neg_samples + user_num
                cur_sample = [np.array([uid, cur_target]), neg_samples]
                cur_sample = np.concatenate(cur_sample).reshape(1, -1)
                test_egos_list.append((adjs, nid, torch.tensor(cur_sample, dtype=torch.int64)))
        print(len(val_egos_list), len(test_egos_list))
        # print(val_contain_soc, test_contain_soc)
        pickle.dump(val_egos_list, open(dataset + '_processed/val_ego_list_v2.pkl', 'wb'))
        pickle.dump(test_egos_list, open(dataset + '_processed/test_ego_list_v2.pkl', 'wb'))


def subgraph_sample(poi_seqs, user_num, poi_num, seq_max_len, n_layer, involved_users,
                    base_seq_lens=None, applied_seq_lens=None, trans_graph=None, soc_edges=None, geo_info=None):
    train_ucenter_seqs, train_ucenter_ts = {}, {}
    involved_users = set(involved_users)
    for key in poi_seqs.keys():
        cur_seq = np.array(poi_seqs[key])
        if len(cur_seq) == 0:
            if key in involved_users:
                involved_users.remove(key)
            continue
        seq, ts = cur_seq[:, 0].tolist(), cur_seq[:, 1].tolist()
        train_ucenter_seqs[key] = seq
        train_ucenter_ts[key] = ts
    involved_users = list(involved_users)

    train_icenter_seqs, train_icenter_ts = {}, {}
    for uid in train_ucenter_seqs.keys():
        seq = train_ucenter_seqs[uid]
        for i in range(len(seq)):
            key = seq[i]
            if key not in train_icenter_seqs:
                train_icenter_seqs[key] = [uid]
                train_icenter_ts[key] = [train_ucenter_ts[uid][i]]
            else:
                train_icenter_seqs[key].append(uid)
                train_icenter_ts[key].append(train_ucenter_ts[uid][i])
    
    for key in train_icenter_ts.keys():
        ts = train_icenter_ts[key]
        ts = sorted(enumerate(ts), key=lambda x:x[1], reverse=False)
        idx, sorted_ts = [], []
        for ele in ts:
            idx.append(ele[0]); sorted_ts.append(ele[1])
        train_icenter_seqs[key] = np.array(train_icenter_seqs[key])[idx].tolist()
        train_icenter_ts[key] = sorted_ts

    assert len(list(train_icenter_seqs.keys())) == poi_num

    geo_avail = geo_info is not None
    print(geo_avail)
    all_train_egos = {}
    for sn, uid in enumerate(train_ucenter_seqs.keys()):
        if sn % 1000 == 0:
            print(sn)
        orig_traj, orig_ts_line = train_ucenter_seqs[uid], train_ucenter_ts[uid]
        cur_traj, cur_ts_line = orig_traj[:-1], orig_ts_line[:-1]
        lbl_traj, lbl_ts = orig_traj[1:], orig_ts_line[1:]
        
        start_idx = 0
        if base_seq_lens is not None:
            base_seq_len, applied_seq_len = base_seq_lens[uid], applied_seq_lens[uid]
            if base_seq_len != applied_seq_len:
                start_idx = base_seq_len
            else:
                continue

        for idx in range(start_idx, len(cur_traj)):
            anchored_nodes = set()
            nid_in_layer = [[uid]]

            cur_pred_ts, cur_target = lbl_ts[idx], lbl_traj[idx]
            real_start = idx - seq_max_len if idx >= seq_max_len else 0
            used_traj, used_ts_known = cur_traj[real_start:(idx+1)], cur_ts_line[real_start:(idx+1)]
            assert len(used_traj) < 55

            edge_index, t_offset, edge_type = [], [], []
            cur_adjs = []
            seq_endpoint = []
            dists = []
            # last_id, last_time_diff = [], []
            # last_time_diff = []
            for n in range(n_layer):
                nid_in_layer.append(set())
                for node in nid_in_layer[-2]:
                    if node in anchored_nodes:
                        continue
                    anchored_nodes.add(node)
                    start = len(edge_index)
                    if n == 0:
                        for i in range(len(used_traj)):
                            poi_id = used_traj[i]
                            edge_index.append([poi_id + user_num, node])
                            edge_type.append(0)
                            td = cur_pred_ts - used_ts_known[i]
                            # num_half_day = (td.days * 24 + td.seconds // 3600) // 12
                            # t_offset.append(num_half_day if num_half_day <= 180 else 181)
                            t_offset.append(int(td.total_seconds()))
                            if geo_avail:
                                last_poi = used_traj[-1]
                                anchor_lat, anchor_lon = geo_info[last_poi]
                                tar_lat, tar_lon = geo_info[poi_id]
                                cur_dist = haversine((tar_lat, tar_lon), (anchor_lat, anchor_lon), unit=Unit.KILOMETERS)
                                # dist_mapid = cur_dist * 10 // 1
                                # dists.append(min(dist_mapid, 500))
                                dists.append(cur_dist)

                            # # For short-term preference
                            # last_id.append(used_traj[-1])

                            # cur_time_diff = used_ts_known[-1] - used_ts_known[i]
                            # last_time_diff.append(cur_time_diff.total_seconds())
                            # num_half_day = (cur_time_diff.days * 24 + cur_time_diff.seconds // 3600) // 12
                            # last_time_diff.append(num_half_day if num_half_day <= 180 else 181)

                            nid_in_layer[-1].add(poi_id + user_num)
                        end = len(edge_index)
                        seq_endpoint.append([start, end])
                        # if soc_edges is not None and node in soc_edges:
                        #     # # Code for social graph integratation
                        #     # friends = soc_edges[node]
                        #     # for user in friends:
                        #     #     edge_index.append([user, node])
                        #     #     edge_type.append(3)
                        #     #     t_offset.append(-1)
                        #     #     nid_in_layer[-1].add(user)

                        #     trans_edges = soc_edges[node]
                        #     for e_idx in range(len(trans_edges)):
                        #         if trans_edges[e_idx][1] > cur_pred_ts:
                        #             break
                        #     if e_idx >= 10:
                        #         trans_edges = trans_edges[(e_idx - 10):e_idx]
                        #     else:
                        #         trans_edges = trans_edges[:e_idx]
                        #     for i in range(len(trans_edges)):
                        #         next_user, arrive_ts, etype = trans_edges[i]
                        #         edge_index.append([next_user, node])
                        #         edge_type.append(etype)
                        #         td = cur_pred_ts - arrive_ts
                        #         t_offset.append(int(td.total_seconds()))
                        #         nid_in_layer[-1].add(next_user)

                    elif n % 2 == 1:
                        if node >= user_num:
                            node = node - user_num
                            cur_seq, cur_ts = train_icenter_seqs[node], train_icenter_ts[node]
                            for t_idx in range(len(cur_ts)):
                                if cur_ts[t_idx] >= cur_pred_ts:
                                    break
                            if t_idx == 0:
                                continue
                            real_start = t_idx - seq_max_len if t_idx >= seq_max_len else 0
                            cur_seq, cur_ts = cur_seq[real_start:t_idx], cur_ts[real_start:t_idx]
                            assert len(cur_seq) < 55
                            for i in range(len(cur_seq)):
                                edge_index.append([cur_seq[i], node + user_num])
                                edge_type.append(1)
                                td = cur_pred_ts - cur_ts[i]
                                # num_half_day = (td.days * 24 + td.seconds // 3600) // 24
                                # t_offset.append(num_half_day if num_half_day <= 180 else 181)
                                t_offset.append(int(td.total_seconds()))

                                # For short-term preference
                                # last_id.append(cur_seq[-1])
                                # cur_time_diff = cur_ts[-1] - cur_ts[i]
                                # last_time_diff.append(cur_time_diff.total_seconds())
                                # num_half_day = (cur_time_diff.days * 24 + cur_time_diff.seconds // 3600) // 24
                                # last_time_diff.append(num_half_day if num_half_day <= 180 else 181)

                                nid_in_layer[-1].add(cur_seq[i])
                            end = len(edge_index)
                            seq_endpoint.append([start, end])
                            
                            if trans_graph is not None and node in trans_graph:
                                trans_edges = trans_graph[node]
                                for e_idx in range(len(trans_edges)):
                                    if trans_edges[e_idx][1] >= cur_pred_ts:
                                        break
                                if e_idx >= seq_max_len:
                                    trans_edges = trans_edges[(e_idx - seq_max_len):e_idx]
                                else:
                                    trans_edges = trans_edges[:e_idx]
                                for i in range(len(trans_edges)):
                                    next_poi, arrive_ts, etype = trans_edges[i]
                                    edge_index.append([next_poi+user_num, node+user_num])
                                    edge_type.append(etype)
                                    td = cur_pred_ts - arrive_ts
                                    # num_half_day = (td.days * 24 + td.seconds // 3600) // 24
                                    # t_offset.append(num_half_day if num_half_day <= 180 else 181)
                                    t_offset.append(int(td.total_seconds()))
                                    nid_in_layer[-1].add(next_poi+user_num)
                                    if geo_avail:
                                        src_lat, src_lon = geo_info[next_poi]
                                        # tar_lat, tar_lon = geo_info[node]
                                        # cur_dist = geodesic((src_lat, src_lon), (tar_lat, tar_lon)).km
                                        cur_dist = haversine((src_lat, src_lon), (anchor_lat, anchor_lon), unit=Unit.KILOMETERS)
                                        # cur_dist = haversine((src_lat, src_lon), (tar_lat, tar_lon), unit=Unit.KILOMETERS)
                                        # dist_mapid = cur_dist * 10 // 1
                                        # dists.append(min(dist_mapid, 500))
                                        dists.append(cur_dist)
                        else:
                            if node not in train_ucenter_seqs:
                                continue
                            cur_seq, cur_ts = train_ucenter_seqs[node], train_ucenter_ts[node]
                            for t_idx in range(len(cur_ts)):
                                if cur_ts[t_idx] >= cur_pred_ts:
                                    break
                            if t_idx == 0:
                                continue
                            real_start = max(t_idx - seq_max_len, 0)
                            cur_seq, cur_ts = cur_seq[real_start:t_idx], cur_ts[real_start:t_idx]
                            assert len(cur_seq) < 55
                            for i in range(len(cur_seq)):
                                poi_id = cur_seq[i]
                                edge_index.append([poi_id + user_num, node])
                                edge_type.append(0)
                                td = cur_pred_ts - cur_ts[i]
                                # num_half_day = (td.days * 24 + td.seconds // 3600) // 12
                                # t_offset.append(num_half_day if num_half_day <= 180 else 181)
                                t_offset.append(int(td.total_seconds()))
                                if geo_avail:
                                    # anchor_poi = cur_seq[-1]
                                    anchor_poi = used_traj[-1]
                                    anchor_lat, anchor_lon = geo_info[anchor_poi]
                                    tar_lat, tar_lon = geo_info[poi_id]
                                    cur_dist = haversine((tar_lat, tar_lon), (anchor_lat, anchor_lon), unit=Unit.KILOMETERS)
                                    # dist_mapid = cur_dist * 10 // 1
                                    # dists.append(min(dist_mapid, 500))
                                    dists.append(cur_dist)
                                nid_in_layer[-1].add(cur_seq[i] + user_num)
                            end = len(edge_index)
                            seq_endpoint.append([start, end])
                            # if soc_edges is not None and node in soc_edges:
                            #     # Code for social graph fusion
                            #     # friends = soc_edges[node]
                            #     # for user in friends:
                            #     #     edge_index.append([user, node])
                            #     #     edge_type.append(3)
                            #     #     t_offset.append(-1)
                            #     #     nid_in_layer[-1].add(user)

                            #     trans_edges = soc_edges[node]
                            #     for e_idx in range(len(trans_edges)):
                            #         if trans_edges[e_idx][1] > cur_pred_ts:
                            #             break
                            #     if e_idx >= 10:
                            #         trans_edges = trans_edges[(e_idx - 10):e_idx]
                            #     else:
                            #         trans_edges = trans_edges[:e_idx]
                            #     for i in range(len(trans_edges)):
                            #         next_user, arrive_ts, etype = trans_edges[i]
                            #         edge_index.append([next_user, node])
                            #         edge_type.append(etype)
                            #         td = cur_pred_ts - arrive_ts
                            #         t_offset.append(int(td.total_seconds()))
                            #         nid_in_layer[-1].add(next_user)
                    
                    # Code for layer larger than 2, Not useful for now
                    else:
                        cur_seq, cur_ts = train_ucenter_seqs[node], train_ucenter_ts[node]
                        for t_idx in range(len(cur_ts)):
                            if cur_ts[t_idx] > cur_pred_ts:
                                break
                        if t_idx == 0:
                            continue
                        real_start = max(t_idx - seq_max_len, 0)
                        cur_seq, cur_ts = cur_seq[real_start:t_idx], cur_ts[real_start:t_idx]
                        for i in range(len(cur_seq)):
                            poi_id = cur_seq[i]
                            edge_index.append([poi_id + user_num, node])
                            edge_type.append(0)
                            td = cur_pred_ts - cur_ts[i]
                            # num_half_day = (td.days * 24 + td.seconds // 3600) // 12
                            # t_offset.append(num_half_day if num_half_day <= 180 else 181)
                            t_offset.append(int(td.total_seconds()))

                            # For short-term preference
                            # last_id.append(cur_seq[-1])

                            # cur_time_diff = cur_ts[-1] - cur_ts[i]
                            # last_time_diff.append(cur_time_diff.total_seconds())
                            # num_half_day = (cur_time_diff.days * 24 + cur_time_diff.seconds // 3600) // 12
                            # last_time_diff.append(num_half_day if num_half_day <= 180 else 181)

                            nid_in_layer[-1].add(cur_seq[i] + user_num)

                # if n != 0 and soc_edges is not None:
                #     center_user = nid_in_layer[0][0]
                #     if center_user in soc_edges:
                #         friends = set(soc_edges[center_user])
                #         sampled_nid = set()
                #         for node in nid_in_layer[-1]:
                #             if node < user_num and node not in anchored_nodes:
                #                 anchored_nodes.add(node)
                #                 if node not in friends:
                #                     continue
                #                 edge_index.append([node, center_user])
                #                 edge_type.append(4)
                #                 t_offset.append(-1)

                #                 cur_seq, cur_ts = train_ucenter_seqs[node], train_ucenter_ts[node]
                #                 for t_idx in range(len(cur_ts)):
                #                     if cur_ts[t_idx] > cur_pred_ts:
                #                         break
                #                 if t_idx == 0:
                #                     continue
                #                 real_start = max(t_idx - seq_max_len, 0)
                #                 cur_seq, cur_ts = cur_seq[real_start:t_idx], cur_ts[real_start:t_idx]
                #                 for i in range(len(cur_seq)):
                #                     poi_id = cur_seq[i]
                #                     edge_index.append([poi_id + user_num, node])
                #                     edge_type.append(0)
                #                     td = cur_pred_ts - cur_ts[i]
                #                     t_offset.append(int(td.total_seconds()))
                #                     if geo_avail:
                #                         last_poi = used_traj[-1]
                #                         anchor_lat, anchor_lon = geo_info[last_poi]
                #                         tar_lat, tar_lon = geo_info[poi_id]
                #                         cur_dist = haversine((tar_lat, tar_lon), (anchor_lat, anchor_lon), unit=Unit.KILOMETERS)
                #                         dists.append(cur_dist)
                #                     # nid_in_layer[-1].add(cur_seq[i] + user_num)
                #                     sampled_nid.add(cur_seq[i] + user_num)
                #         nid_in_layer[-1] |= sampled_nid


                nid_in_layer[-1] = list(nid_in_layer[-1])
                cur_nid = np.concatenate(nid_in_layer)
                cur_nid, index = np.unique(cur_nid, return_index=True)
                cur_nid = cur_nid[np.argsort(index)].tolist()
                mapped_nid = torch.zeros(user_num + poi_num, dtype=torch.int64)
                cur_nid = np.array(cur_nid)
                mapped_nid[cur_nid] = torch.arange(cur_nid.shape[0])
                cur_index = torch.tensor(edge_index, dtype=torch.int64).T
                cur_index = mapped_nid[cur_index]
                cur_etype = torch.tensor(edge_type, dtype=torch.int64)
                cur_toff = torch.tensor(t_offset, dtype=torch.int64)
                cur_dists = torch.tensor(dists, dtype=torch.int64) if geo_avail else None
                # size = (len(cur_nid), last_size)
                size = tuple(seq_endpoint)
                
                assert torch.min(cur_index) >= 0 and torch.max(cur_index) < len(cur_nid)
                cur_adjs.append(Adj(cur_index, cur_etype, cur_toff, cur_dists, size))
                # last_size = len(cur_nid)
            
            if soc_edges is not None:
                center_user = nid_in_layer[0][0]
                if center_user not in soc_edges:
                    soc_adjs, soc_nid = None, None
                elif center_user in soc_edges:
                    friends = soc_edges[center_user]
                    soc_eindex, soc_etype, soc_toff, soc_dists = [], [], [], []
                    soc_nid = set(friends)
                    for node in friends:
                        cur_seq, cur_ts = train_ucenter_seqs[node], train_ucenter_ts[node]
                        for t_idx in range(len(cur_ts)):
                            if cur_ts[t_idx] > cur_pred_ts:
                                break
                        if t_idx == 0:
                            soc_nid.remove(node)
                            continue
                        real_start = max(t_idx - seq_max_len, 0)
                        cur_seq, cur_ts = cur_seq[real_start:t_idx], cur_ts[real_start:t_idx]
                        for i in range(len(cur_seq)):
                            poi_id = cur_seq[i]
                            soc_nid.add(poi_id + user_num)
                            soc_eindex.append([poi_id + user_num, node])
                            soc_etype.append(0)
                            td = cur_pred_ts - cur_ts[i]
                            soc_toff.append(int(td.total_seconds()))
                            if geo_avail:
                                last_poi = used_traj[-1]
                                anchor_lat, anchor_lon = geo_info[last_poi]
                                tar_lat, tar_lon = geo_info[poi_id]
                                cur_dist = haversine((tar_lat, tar_lon), (anchor_lat, anchor_lon), unit=Unit.KILOMETERS)
                                soc_dists.append(cur_dist)
                    mapped_nid = torch.zeros(user_num + poi_num, dtype=torch.int64)
                    soc_nid = np.array(list(soc_nid))
                    if len(soc_nid) == 0:
                        soc_adjs, soc_nid = None, None
                    else:
                        mapped_nid[soc_nid] = torch.arange(soc_nid.shape[0])
                        soc_eindex = torch.tensor(soc_eindex, dtype=torch.int64).T
                        soc_eindex = mapped_nid[soc_eindex]
                        soc_etype = torch.tensor(soc_etype, dtype=torch.int64)
                        soc_toff = torch.tensor(soc_toff, dtype=torch.int64)
                        soc_dists = torch.tensor(soc_dists, dtype=torch.int64) if geo_avail else None
                        soc_adjs = Adj(soc_eindex, soc_etype, soc_toff, soc_dists, None)

            cur_adjs = cur_adjs[-1]
            cur_nid = torch.tensor(list(cur_nid), dtype=torch.int64)
            if soc_edges is None:
                if uid in all_train_egos:
                    all_train_egos[uid].append((cur_adjs, cur_nid, np.array([uid, cur_target]) , cur_pred_ts))
                else:
                    all_train_egos[uid] = [(cur_adjs, cur_nid, np.array([uid, cur_target]) , cur_pred_ts)]
            else:
                if uid in all_train_egos:
                    all_train_egos[uid].append((cur_adjs, cur_nid, soc_adjs, soc_nid, np.array([uid, cur_target]) , cur_pred_ts))
                else:
                    all_train_egos[uid] = [(cur_adjs, cur_nid, soc_adjs, soc_nid, np.array([uid, cur_target]) , cur_pred_ts)]
    
    print(len(list(all_train_egos.keys())), len(involved_users))
    return all_train_egos, involved_users


def spa_time_interval(dataset):
    poi_set = pickle.load(open(dataset + '_processed/seen_poi.pkl', 'rb'))
    poi_seqs = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb'))
    val_patch = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
    test_patch = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))

    poi_info = pickle.load(open(dataset + '_processed/loc_info.pkl', 'rb'))
    poi_map = {}
    geo_info = np.zeros((len(poi_set), 2))
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    for id in poi_set:
        entry = poi_info[id]
        lat, lon = entry
        geo_info[poi_map[id]] = [lat, lon]
    
    intervals = []
    for key in val_patch:
        poi_seqs[key] = poi_seqs[key] + val_patch[key]
    for key in test_patch:
        poi_seqs[key] = poi_seqs[key] + test_patch[key]
    for key in poi_seqs:
        for i in range(1, len(poi_seqs[key])):
            prev, cur = poi_seqs[key][i-1][1], poi_seqs[key][i][1]
            intervals.append((cur - prev).total_seconds())
    print(np.median(intervals))

    s_intervals = []
    for key in poi_seqs:
        for i in range(1, len(poi_seqs[key])):
            prev, cur = poi_seqs[key][i-1][0], poi_seqs[key][i][0]
            prev_lat, prev_lon = geo_info[prev]
            cur_lat, cur_lon = geo_info[cur]
            s_intervals.append(haversine((prev_lat, prev_lon), (cur_lat, cur_lon), unit=Unit.KILOMETERS))
    print(np.median(s_intervals))


def pkl2graph(dataset):
    poi_set = pickle.load(open(dataset + '_processed/seen_poi.pkl', 'rb'))
    user_set = pickle.load(open(dataset + '_processed/seen_user.pkl', 'rb'))
    poi_seqs = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb'))
    val_patch = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
    test_patch = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))
    
    user_num, poi_num = len(user_set), len(poi_set)
    train_eindex = []
    val_eindex = []
    for key in poi_seqs:
        for ele in poi_seqs[key]:
            train_eindex.append([key, ele[0] + user_num])
            val_eindex.append([key, ele[0] + user_num])
    train_eindex = np.array(train_eindex).T
    pickle.dump(train_eindex, open(dataset + '_processed/train_lgcn_eindex.pkl', 'wb'))
    for key in val_patch:
        for ele in val_patch[key]:
            val_eindex.append([key, ele[0] + user_num])
    val_eindex = np.array(val_eindex).T
    pickle.dump(val_eindex, open(dataset + '_processed/val_lgcn_eindex.pkl', 'wb'))


if __name__ == '__main__':
    # process_gowalla()
    # gowalla_freq_analysis()
    # gowalla_traj_analysis()
    # gowalla_filter()
    # seq_align('gowalla')
    # gowalla_social()
    # poi_geography_analysis()
    # process_foursquare()
    # foursquare_freq_analysis()
    # foursquare_traj_analysis()
    # foursquare_filter()
    # seq_align('foursquare')
    # foursquare_social()

    # interval_statistics()
    # pkl2txt('foursquare')
    # pkl2npy('gowalla')
    # pkl2npy('foursquare')
    # pkl2graph('gowalla')
    # pkl2graph('foursquare')

    precompute_egonet('foursquare', 2, 50, 'train')
    # spa_time_interval('gowalla')