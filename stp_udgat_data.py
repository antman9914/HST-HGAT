import pickle, time, random, copy
import numpy as np
from haversine import haversine, Unit
from tqdm import tqdm

dataset='foursquare'

train_seq = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb')) 
val_seq = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
test_seq = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))
user_set = pickle.load(open(dataset + '_processed/seen_user.pkl', 'rb'))
poi_set = pickle.load(open(dataset + '_processed/seen_poi.pkl', 'rb'))
poi_info = pickle.load(open(dataset + '_processed/loc_info.pkl', 'rb'))
val_full_seq, test_full_seq = {}, {}
for key in val_seq:
    val_full_seq[key] = train_seq[key] + val_seq[key]
for key in test_seq:
    if key not in val_seq:
        test_full_seq[key] = train_seq[key] + test_seq[key]
    else:
        test_full_seq[key] = train_seq[key] + val_seq[key] + test_seq[key]
poi_map = {}
for i in range(len(poi_set)):
    poi_map[poi_set[i]] = i
geo_info = {}
for key in poi_set:
    geo_info[poi_map[key]] = poi_info[key]

def spa_glob_graph():
    s_eindex, s_eweight = [], []
    # For spatial graph
    for i in range(len(poi_set)):
        dist = np.zeros(len(poi_set))
        src_lat, src_lon = geo_info[i]
        min_dist = 10000
        start = time.time()
        for j in range(len(poi_set)):
            if i == j:
                continue
            tar_lat, tar_lon = geo_info[j]
            dist[j] = haversine((src_lat, src_lon), (tar_lat, tar_lon), unit=Unit.KILOMETERS)
            if dist[j] != 0:
                min_dist = min(dist[j], min_dist)
        sorted_idx = np.argsort(dist)[:5]
        for ele in sorted_idx:
            s_eindex.append([i, ele])
            if dist[ele] == 0:
                s_eweight.append(1/min_dist)
            else:
                s_eweight.append(1/dist[ele])
        end = time.time()
        if i % 500 == 0:
            print(i, end-start)
    s_eindex, s_eweight = np.array(s_eindex).T, np.array(s_eweight)
    inv_eindex = np.stack([s_eindex[1], s_eindex[0]], axis=0)
    s_eindex, s_eweight = np.concatenate([s_eindex, inv_eindex], axis=-1), np.concatenate([s_eweight, s_eweight], axis=-1)
    s_eindex, uni_idx = np.unique(s_eindex, return_index=True, axis=-1)
    s_eweight = s_eweight[uni_idx]
    s_graph = {
        'edge_index': s_eindex,
        'edge_weight': s_eweight
    }
    pickle.dump(s_graph, open(dataset + '_stp/s_graph.pkl', 'wb'))
    return s_eindex, s_eweight

def time_glob_graph():
    t_eindex, t_eweight = [], []
    t_info = {}
    full_seq = []
    for key in train_seq:
        full_seq.extend(train_seq[key])
    full_seq = sorted(full_seq, key=lambda x:x[1], reverse=False)
    for i in tqdm(range(len(full_seq)-1)):
        src, tar = full_seq[i][0], full_seq[i+1][0]
        if src == tar:
            continue
        src_ts, tar_ts = full_seq[i][1], full_seq[i+1][1]
        timespan = (tar_ts - src_ts).total_seconds()
        if src not in t_info:
            t_info[src] = {}
        if tar not in t_info[src]:
            t_info[src][tar] = []
        t_info[src][tar].append(timespan)
        if tar not in t_info:
            t_info[tar] = {}
        if src not in t_info[tar]:
            t_info[tar][src] = []
        t_info[tar][src].append(timespan)
    for src in t_info:
        for tar in t_info[src]:
            weight = np.mean(t_info[src][tar]) / 3600
            t_eindex.append([src, tar])
            t_eweight.append(1 / weight if weight != 0 else 1e5)
    t_eindex, t_eweight = np.array(t_eindex).T, np.array(t_eweight)
    t_graph = {
        'edge_index': t_eindex,
        'edge_weight': t_eweight,
    }
    pickle.dump(t_graph, open(dataset + '_stp/t_graph.pkl', 'wb'))
    return t_eindex, t_eweight

def freq_glob_graph():
    f_eindex, f_eweight = [], []
    f_info = {}
    for key in tqdm(train_seq):
        cur_seq = np.array(train_seq[key])[:, 0]
        for i in range(len(cur_seq) - 1):
            src, tar = cur_seq[i], cur_seq[i+1]
            if src not in f_info:
                f_info[src] = {}
            if tar not in f_info[src]:
                f_info[src][tar] = 0
            f_info[src][tar] += 1
            if tar not in f_info:
                f_info[tar] = {}
            if src not in f_info[tar]:
                f_info[tar][src] = 0
            f_info[tar][src] += 1
    for src in f_info:
        for tar in f_info[src]:
            f_eindex.append([src, tar])
            f_eweight.append(f_info[src][tar])
    f_eindex, f_eweight = np.array(f_eindex).T, np.array(f_eweight)
    f_graph = {
        'edge_index': f_eindex,
        'edge_weight': f_eweight
    }
    pickle.dump(f_graph, open(dataset + '_stp/f_graph.pkl', 'wb'))
    return f_eindex, f_eweight

def user_glob_graph():
    u_eindex = []
    user_list = list(train_seq.keys())
    for i in range(len(user_list)):
        for j in range(i+1, len(user_list)):
            src, tar = user_list[i], user_list[j]
            src_seq = np.array(train_seq[src])[:, 0].tolist()
            tar_seq = np.array(train_seq[tar])[:, 0].tolist()
            src_seq, tar_seq = set(src_seq), set(tar_seq)
            jaccard_sim = len(src_seq & tar_seq) / len(src_seq | tar_seq)
            if jaccard_sim > 0.2:
                u_eindex.append([src, tar])
                u_eindex.append([tar, src])
        if i % 50 == 0:
            print(i, len(u_eindex))
    u_eindex = np.array(u_eindex).T
    pickle.dump(u_eindex, open(dataset + '_stp/u_graph.pkl', 'wb'))

def dataset_generate(s_dict, t_dict, f_dict, mode='train'):
    cur_seqs = None
    if mode == 'train':
        cur_seqs = train_seq
        ref_seqs = None
    elif mode == 'val':
        cur_seqs = val_full_seq
        ref_seqs = val_seq
    else:
        cur_seqs = test_full_seq
        ref_seqs = test_seq
    final_data = []
    for key in tqdm(cur_seqs):
        # if n % 1000 == 0:
        #     print(n)
        selected_seq = np.array(cur_seqs[key])[:, 0]
        seed_poi_set = set(selected_seq.tolist())
        seed_poi_list = list(seed_poi_set)

        # First-hop neighbor extraction
        start = time.time()
        s_poi_freq, t_poi_freq, f_poi_freq = np.zeros(len(poi_set)), np.zeros(len(poi_set)), np.zeros(len(poi_set))
        s_neigh_set, t_neigh_set, f_neigh_set = copy.deepcopy(seed_poi_list), copy.deepcopy(seed_poi_list), copy.deepcopy(seed_poi_list)
        for ele in seed_poi_set:
            neighbors = s_dict[ele][:, 0].astype(int)
            s_poi_freq[neighbors] += 1
            s_neigh_set.extend(neighbors.tolist())
            neighbors = t_dict[ele][:, 0].astype(int)
            t_poi_freq[neighbors] += 1
            t_neigh_set.extend(neighbors.tolist())
            neighbors = f_dict[ele][:, 0].astype(int)
            f_poi_freq[neighbors] += 1
            f_neigh_set.extend(neighbors.tolist())
        s_neigh_set, t_neigh_set, f_neigh_set = list(set(s_neigh_set)), list(set(t_neigh_set)), list(set(f_neigh_set))
        basic_idx = np.arange(len(poi_set))
        s_neighbors = basic_idx[s_poi_freq != 0]
        t_neighbors = basic_idx[t_poi_freq != 0]
        f_neighbors = basic_idx[f_poi_freq != 0]
        if len(s_neighbors) > 25:
            freqs = s_poi_freq[s_poi_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            s_neighbors = s_neighbors[sorted_idx]
        if len(t_neighbors) > 25:
            freqs = t_poi_freq[t_poi_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            t_neighbors = t_neighbors[sorted_idx]
        if len(f_neighbors) > 25:
            freqs = f_poi_freq[f_poi_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            f_neighbors = f_neighbors[sorted_idx]
        end = time.time()
        # print(end - start)
        
        start = time.time()
        s_rw_freq, t_rw_freq, f_rw_freq = np.zeros(len(poi_set)), np.zeros(len(poi_set)), np.zeros(len(poi_set))
        for ele in seed_poi_set:
            s_rw_freq = s_rw_freq + random_walk(s_dict, ele, 's')
            t_rw_freq = t_rw_freq + random_walk(t_dict, ele, 't')
            f_rw_freq = f_rw_freq + random_walk(f_dict, ele, 'f')
        end = time.time()
        # print(end - start)
        s_rw_freq[s_neigh_set] = 0
        t_rw_freq[t_neigh_set] = 0
        f_rw_freq[f_neigh_set] = 0
        s_rw_neighbors = basic_idx[s_rw_freq != 0]
        t_rw_neighbors = basic_idx[t_rw_freq != 0]
        f_rw_neighbors = basic_idx[f_rw_freq != 0]
        if len(s_rw_neighbors) > 25:
            freqs = s_rw_freq[s_rw_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            s_rw_neighbors = s_rw_neighbors[sorted_idx]
        if len(t_rw_neighbors) > 25:
            freqs = t_rw_freq[t_rw_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            t_rw_neighbors = t_rw_neighbors[sorted_idx]
        if len(f_rw_neighbors) > 25:
            freqs = f_rw_freq[f_rw_freq != 0]
            sorted_idx = np.argsort(freqs)[::-1][:25]
            f_rw_neighbors = f_rw_neighbors[sorted_idx]
        if ref_seqs is not None:
            start_idx = len(selected_seq) - len(ref_seqs[key]) - 1
        else:
            start_idx = 0
        for i in range(start_idx, len(selected_seq)-1):
            pp_eindex = []
            for j in range(len(seed_poi_list)):
                if seed_poi_list[j] != selected_seq[i]:
                    pp_eindex.append([seed_poi_list[j], selected_seq[i]])
            pp_eindex = np.array(pp_eindex).T
            final_data.append((selected_seq[i], selected_seq[i+1], key, pp_eindex, s_neighbors, t_neighbors, f_neighbors, s_rw_neighbors, t_rw_neighbors, f_rw_neighbors))
    pickle.dump(final_data, open(dataset + '_stp/%s_data.pkl' % mode, 'wb'))


def random_walk(graph_dict, start_node, etype):
    mu, beta = 5, 5
    freq = np.zeros(len(poi_set))
    if start_node not in graph_dict:
        return freq
    for n in range(beta):
        cur_node = start_node
        for i in range(mu):
            info = graph_dict[cur_node]
            try:
                tars, prob = info[:, 0].astype(int), info[:, 1]
                basic_idx = np.arange(len(tars))
                prob = prob / np.sum(prob)
                prob = np.cumsum(prob)
                seed = random.random()
                next_node = basic_idx[prob > seed][0]
            except:
                print(cur_node, prob, info[:,1], etype)
                exit()
            cur_node = tars[next_node]
            freq[cur_node] += 1
    return freq

# time_glob_graph()
# freq_glob_graph()
# user_glob_graph()
# spa_glob_graph()
s_graph = pickle.load(open(dataset + '_stp/s_graph.pkl', 'rb'))
t_graph = pickle.load(open(dataset + '_stp/t_graph.pkl', 'rb'))
f_graph = pickle.load(open(dataset + '_stp/f_graph.pkl', 'rb'))
s_eindex, s_eweight = s_graph['edge_index'], s_graph['edge_weight']
t_eindex, t_eweight = t_graph['edge_index'], t_graph['edge_weight']
f_eindex, f_eweight = f_graph['edge_index'], f_graph['edge_weight']
s_dict, t_dict, f_dict = {}, {}, {}
for i in range(s_eindex.shape[1]):
    src, tar, w = s_eindex[0, i], s_eindex[1, i], s_eweight[i]
    if src not in s_dict:
        s_dict[src] = []
    s_dict[src].append([tar, w])
for i in range(t_eindex.shape[1]):
    src, tar, w = t_eindex[0, i], t_eindex[1, i], t_eweight[i]
    if src not in t_dict:
        t_dict[src] = []
    t_dict[src].append([tar, w])
for i in range(f_eindex.shape[1]):
    src, tar, w = f_eindex[0, i], f_eindex[1, i], f_eweight[i]
    if src not in f_dict:
        f_dict[src] = []
    f_dict[src].append([tar, w])
for key in s_dict:
    s_dict[key] = np.array(s_dict[key])
for key in t_dict:
    t_dict[key] = np.array(t_dict[key])
for key in f_dict:
    f_dict[key] = np.array(f_dict[key])

dataset_generate(s_dict, t_dict, f_dict, 'train')
dataset_generate(s_dict, t_dict, f_dict, 'val')
dataset_generate(s_dict, t_dict, f_dict, 'test')