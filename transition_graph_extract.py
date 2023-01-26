import numpy as np
import pickle
import time
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
from geopy.distance import geodesic
from utils.FastNode2Vec import FastNode2Vec

emb_dim = 64
num_walks = 30
walk_length = 15
window = 3
epochs = 3
p = 1.0
q = 10.0


# def getDistance(latA, lonA, latB, lonB):
#     if latA == latB and lonA == lonB:
#         return 0
#     ra = 6378140  
#     rb = 6356755 
#     flatten = (ra - rb) / ra 
#     radLatA = np.radians(latA)
#     radLonA = np.radians(lonA)
#     radLatB = np.radians(latB)
#     radLonB = np.radians(lonB)

#     pA = np.arctan(rb / ra * np.tan(radLatA))
#     pB = np.arctan(rb / ra * np.tan(radLatB))
#     tmp = np.sin(pA) * np.sin(pB) + np.cos(pA) * np.cos(pB) * np.cos(radLonA - radLonB)
#     x = np.arccos(tmp)
#     c1 = (np.sin(x) - x) * (np.sin(pA) + np.sin(pB)) ** 2 / np.cos(x / 2) ** 2
#     c2 = (np.sin(x) + x) * (np.sin(pA) - np.sin(pB)) ** 2 / np.sin(x / 2) ** 2
#     dr = flatten / 8 * (c1 - c2)
#     distance = ra * (x + dr)
#     return distance / 1000


def distance_statistics(dataset, poi_set):
    poi_info = pickle.load(open(dataset + '_processed/loc_info.pkl', 'rb'))
    trans_graph = pickle.load(open(dataset + '_processed/transition_graph.pkl', 'rb'))
    poi_map = {}
    geo_info = np.zeros((len(poi_set), 2))
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    for id in poi_set:
        entry = poi_info[id]
        lat, lon = entry
        geo_info[poi_map[id]] = [lat, lon]
    max_dist, min_dist = 0, 1000
    dists = []
    for key in tqdm(trans_graph.keys()):
        trans_edges = trans_graph[key]
        src_lat, src_lon = geo_info[key]
        for edge in trans_edges:
            tar_poi = edge[0]
            tar_lat, tar_lon = geo_info[tar_poi]
            # cur_dist = getDistance(src_lat, src_lon, tar_lat, tar_lon)
            cur_dist = geodesic((src_lat, src_lon), (tar_lat, tar_lon)).km
            dists.append(cur_dist)
            max_dist = max(max_dist, cur_dist)
            min_dist = min(min_dist, cur_dist)
    print(max_dist, min_dist)
    
    dist_map = np.zeros(1001)
    dists = np.array(dists)
    print(len(dists))
    dists = dists[~np.isnan(dists)]
    print(len(dists))
    dists_id = np.floor(dists * 20)
    dists_id[dists_id >= 1000] = 1000
    for i in range(len(dists_id)):
        dist_map[int(dists_id[i])] += 1
    for i in range(1, 1001):
        dist_map[i] += dist_map[i-1]
    dist_map /= dist_map[-1]
    plt.figure()
    plt.plot(range(1001), dist_map)

    # dist_freq = dict(sorted(dists.items(),key=lambda x:x[0], reverse=False))
    # print(dist_freq)
    # x_user, y_user = list(dist_freq.keys()), np.array(list(dist_freq.values()), dtype=np.float64)
    # for i in range(1, len(y_user)):
    #     y_user[i] += y_user[i-1]
    # y_user /= y_user[-1]
    # plt.figure()
    # plt.plot(x_user, y_user)
    plt.xlabel('distance')
    plt.ylabel('accumulated user ratio')
    plt.grid('--')
    plt.title('Geographical distance distribution in Gowalla')
    plt.savefig('gowalla_distance_2.png')


def spatial_graph_extract(dataset, poi_set):
    poi_info = pickle.load(open(dataset + '_processed/loc_info.pkl', 'rb'))
    poi_map = {}
    geo_info = np.zeros((len(poi_set), 2))
    for i in range(len(poi_set)):
        poi_map[poi_set[i]] = i
    
    for id in poi_set:
        entry = poi_info[id]
        lat, lon = entry
        geo_info[poi_map[id]] = [lat, lon]
    
    edge_index, edge_weight = [], []
    for i in range(len(poi_set)):
        if i % 1000 == 0:
            print(i)
        cur_lat, cur_lon = geo_info[i]
        distances = np.zeros(len(poi_set))
        for j in range(len(poi_set)):
            if j == i:
                continue
            dst_lat, dst_lon = geo_info[j]
            distances[j] = (cur_lat - dst_lat) ** 2 + (cur_lon - dst_lon) ** 2
        idx = np.argsort(distances)
        start = 0
        for j in range(len(idx)):
            if distances[idx[j]] == 0:
                start += 1
            else:
                break
        idx = idx[start:start+5]
        for j in idx:
            edge_index.append([i, j])
            edge_weight.append(1 / distances[j])
    pickle.dump((edge_index, edge_weight), open(dataset+'_processed/spatial_graph.pkl', 'wb'))
    return edge_index, edge_weight


def transition_graph_extract(dataset):
    poi_seqs = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb'))
    val_patch = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
    test_patch = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))
    actural_seqs = {}
    for key in poi_seqs.keys():
        actural_seqs[key] = poi_seqs[key]
    for key in val_patch.keys():
        if len(val_patch[key]) == 0:
            continue
        actural_seqs[key] += val_patch[key]
    for key in test_patch.keys():
        if key not in actural_seqs or len(test_patch[key]) == 0:
            continue
        actural_seqs[key] += test_patch[key]
    
    temp_graph, trans_graph = {}, {}
    for key in tqdm(actural_seqs.keys()):
        cur_seq = actural_seqs[key]
        if len(cur_seq) == 0:
            continue
        for i in range(1, len(cur_seq)):
            src, dst = cur_seq[i][0], cur_seq[i-1][0]
            if src == dst:
                continue
            src_ts, dst_ts = cur_seq[i][1], cur_seq[i-1][1]
            if src not in trans_graph:
                trans_graph[src] = {}
                trans_graph[src][dst] = 1
            else:
                if dst not in trans_graph[src]:
                    trans_graph[src][dst] = 1
                else:
                    trans_graph[src][dst] += 1
            if dst not in trans_graph:
                trans_graph[dst] = {}
                trans_graph[dst][src] = 1
            else:
                if src not in trans_graph[dst]:
                    trans_graph[dst][src] = 1
                else:
                    trans_graph[dst][src] += 1
            
            delta_t = src_ts - dst_ts
            delta_t = delta_t.days * 24 + delta_t.seconds // 3600
            eweight = 1.0 / delta_t if delta_t > 0 else 1.
            if src not in temp_graph:
                temp_graph[src] = {}
                temp_graph[src][dst] = [eweight]
            else:
                if dst not in temp_graph[src]:
                    temp_graph[src][dst] = [eweight]
                else:
                    temp_graph[src][dst].append(eweight)
            if dst not in temp_graph:
                temp_graph[dst] = {}
                temp_graph[dst][src] = [eweight]
            else:
                if src not in temp_graph[dst]:
                    temp_graph[dst][src] = [eweight]
                else:
                    temp_graph[dst][src].append(eweight)
    temp_edge_index, temp_edge_weight, trans_edge_index, trans_edge_weight = [], [], [], []
    for src in temp_graph:
        for dst in temp_graph[src]:
            temp_edge_index.append([src, dst])
            temp_edge_weight.append(np.median(temp_graph[src][dst]))
    for src in trans_graph:
        for dst in trans_graph[src]:
            trans_edge_index.append([src, dst])
            trans_edge_weight.append(trans_graph[src][dst])
    pickle.dump((temp_edge_index, temp_edge_weight), open(dataset + '_processed/temporal_graph.pkl', 'wb'))
    pickle.dump((trans_edge_index, trans_edge_weight), open(dataset + '_processed/transition_graph.pkl', 'wb'))
    return temp_edge_index, temp_edge_weight, trans_edge_index, trans_edge_weight


def dynamic_transition_graph(dataset):
    poi_seqs = pickle.load(open(dataset + '_processed/train_seq_align.pkl', 'rb'))
    val_patch = pickle.load(open(dataset + '_processed/val_seq_align.pkl', 'rb'))
    test_patch = pickle.load(open(dataset + '_processed/test_seq_align.pkl', 'rb'))
    actural_seqs = {}
    for key in poi_seqs.keys():
        actural_seqs[key] = poi_seqs[key]
    for key in val_patch.keys():
        if len(val_patch[key]) == 0:
            continue
        actural_seqs[key] += val_patch[key]
    for key in test_patch.keys():
        if key not in actural_seqs or len(test_patch[key]) == 0:
            continue
        actural_seqs[key] += test_patch[key]

    train_icenter_seqs, train_icenter_ts = {}, {}
    for uid in actural_seqs.keys():
        cur_seq = np.array(actural_seqs[uid])
        seq, ts = cur_seq[:, 0].tolist(), cur_seq[:, 1].tolist()
        for i in range(len(seq)):
            key = seq[i]
            if key not in train_icenter_seqs:
                train_icenter_seqs[key] = [uid]
                train_icenter_ts[key] = [ts[i]]
            else:
                train_icenter_seqs[key].append(uid)
                train_icenter_ts[key].append(ts[i])
    
    trans_graph = {}
    for key in actural_seqs.keys():
        cur_seq = actural_seqs[key]
        if len(cur_seq) == 0:
            continue
        for i in range(1, len(cur_seq)):
            src, dst = cur_seq[i][0], cur_seq[i-1][0]
            if src == dst:
                continue
            src_ts, dst_ts = cur_seq[i][1], cur_seq[i-1][1]
            if src not in trans_graph:
                trans_graph[src] = [[dst, dst_ts, 2]]
            else:
                trans_graph[src].append([dst, dst_ts, 2])

            if dst not in trans_graph:
                # trans_graph[dst] = {}
                # trans_graph[dst][src] = [src_ts]
                trans_graph[dst] = [[src, src_ts, 3]]
            else:
                trans_graph[dst].append([src, src_ts, 3])
                # if src not in trans_graph[dst]:
                #     trans_graph[dst][src] = [src_ts]
                # else:
                #     trans_graph[dst][src].append(src_ts)
    total_num = 0
    for src in trans_graph.keys():
        trans_graph[src] = sorted(trans_graph[src], key=lambda x:x[1])
        total_num += len(trans_graph[src])
    print(total_num)
    pickle.dump(trans_graph, open(dataset + '_processed/transition_graph.pkl', 'wb'))

    # user_trans_graph = {}
    # for key in train_icenter_seqs.keys():
    #     cur_seq, cur_ts = train_icenter_seqs[key], train_icenter_ts[key]
    #     if len(cur_seq) <= 1:
    #         continue
    #     for i in range(1, len(cur_seq)):
    #         src, dst = cur_seq[i], cur_seq[i-1]
    #         if src == dst:
    #             continue
    #         src_ts, dst_ts = cur_ts[i], cur_ts[i-1]
    #         if src not in user_trans_graph:
    #             user_trans_graph[src] = [[dst, dst_ts, 4]]
    #         else:
    #             user_trans_graph[src].append([dst, dst_ts, 4])

    #         if dst not in user_trans_graph:
    #             user_trans_graph[dst] = [[src, src_ts, 5]]
    #         else:
    #             user_trans_graph[dst].append([src, src_ts, 5])
    # total_num = 0
    # for src in user_trans_graph.keys():
    #     user_trans_graph[src] = sorted(user_trans_graph[src], key=lambda x:x[1])
    #     total_num += len(user_trans_graph[src])
    # print(total_num)
    # pickle.dump(user_trans_graph, open(dataset + '_processed/user_trans_graph.pkl', 'wb'))


def dense_feat_train(edge_index, edge_weight, node_num, dataset, comment):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    edge_weight, edge_index = np.array(edge_weight), np.array(edge_index).T
    row, col = edge_index[0], edge_index[1]
    
    n2v_model = FastNode2Vec(node_num, np.array(row), np.array(col))
    alpha_schedule = [[1,2,2,301], [0.05, 0.05, 0.005, 0.005]]
    n2v_model.run_node2vec(
        dim=emb_dim, 
        epochs=epochs, 
        num_walks=num_walks, 
        walk_length=walk_length, 
        window=window, 
        alpha_schedule=alpha_schedule,
        p=p, 
        q=q,
    )
    embs = n2v_model.get_embeddings()
    print(embs.shape)
    np.save('poi_emb_{}_{}.npy'.format(dataset, comment), embs)


if __name__ == '__main__':
    dataset = 'foursquare'
    poi_set = pickle.load(open(dataset + '_processed/seen_poi.pkl', 'rb'))
    node_num = len(poi_set)
    # s_graph, s_eweight = spatial_graph_extract(dataset, poi_set)
    # print('Spatial graph loaded')
    # t_graph, t_eweight, trans_graph, trans_eweight = transition_graph_extract(dataset)
    # print('Temporal and Transition graph loaded')
    # # dense_feat_train(s_graph, s_eweight, node_num, dataset, 'spatial')
    # dense_feat_train(t_graph, t_eweight, node_num, dataset, 'temporal')
    # dense_feat_train(trans_graph, trans_eweight, node_num, dataset, 'transition')

    dynamic_transition_graph(dataset)
    # distance_statistics(dataset, poi_set)
