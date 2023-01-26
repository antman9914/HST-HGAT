import pickle
import numpy as np
from tqdm import tqdm

dataset = 'gowalla'

GOW_TRAIN = dataset + '_processed/train_seq_align.pkl'
GOW_VAL = dataset + '_processed/val_seq_align.pkl'
GOW_TEST = dataset + '_processed/test_seq_align.pkl'
GOW_USER = dataset + '_processed/seen_user.pkl'
GOW_POI = dataset + '_processed/seen_poi.pkl'


def hit_rate(logits, label, k=10):
    sorted_idx = logits[:k]
    tot = 0.
    if label in sorted_idx:
        tot += 1
    return tot


def load_data():
    user_set, item_set = pickle.load(open(GOW_USER, 'rb')), pickle.load(open(GOW_POI, 'rb'))
    train_seq = pickle.load(open(GOW_TRAIN, 'rb'))
    val_seq, test_seq = pickle.load(open(GOW_VAL, 'rb')), pickle.load(open(GOW_TEST, 'rb'))
    for key in val_seq:
        train_seq[key] = train_seq[key] + val_seq[key]
    for key in train_seq:
        cur_seq = np.array(train_seq[key])[:, 0]
        train_seq[key] = cur_seq.tolist()
    for key in test_seq:
        cur_seq = np.array(test_seq[key])[:, 0]
        test_seq[key] = cur_seq.tolist()
    return user_set, item_set, train_seq, test_seq


def top():
    hr_1_tot, hr_5_tot, hr_10_tot = 0, 0, 0
    mrr = 0
    _, poi_set, train_seq, test_seq = load_data()
    poi_freq = np.zeros_like(poi_set)
    for key in train_seq:
        for ele in train_seq[key]:
            poi_freq[ele] += 1
    sorted_idx = np.argsort(poi_freq)[::-1]
    pred_idx = sorted_idx[:10]
    step = 0
    for key in test_seq:
        for ele in test_seq[key]:
            hr_1 = hit_rate(pred_idx, ele, 1)
            hr_5 = hit_rate(pred_idx, ele, 5)
            hr_10 = hit_rate(pred_idx, ele, 10)
            hr_1_tot += hr_1
            hr_5_tot += hr_5
            hr_10_tot += hr_10
            step += 1
            y_true = np.zeros(sorted_idx.shape[0])
            y_true[ele] = 1
            y_true = y_true[sorted_idx]
            rr_score = y_true / (np.arange(np.shape(y_true)[0]) + 1)
            mrr += np.sum(rr_score) / np.sum(y_true)
    print(hr_1_tot / step, hr_5_tot / step, hr_10_tot / step, mrr / step)


def utop():
    hr_1_tot, hr_5_tot, hr_10_tot = 0, 0, 0
    mrr = 0
    _, poi_set, train_seq, test_seq = load_data()
    poi_freq = {}
    step = 0
    for key in tqdm(train_seq):
        cur_freq = np.zeros_like(poi_set)
        for ele in train_seq[key]:
            cur_freq[ele] += 1
        sorted_idx = np.argsort(cur_freq)[::-1]
        pred_idx = sorted_idx[:10]
        poi_freq[key] = pred_idx
        if key in test_seq:
            for ele in test_seq[key]:
                y_true = np.zeros(sorted_idx.shape[0])
                y_true[ele] = 1
                y_true = y_true[sorted_idx]
                rr_score = y_true / (np.arange(np.shape(y_true)[0]) + 1)
                mrr += np.sum(rr_score) / np.sum(y_true)
    step = 0
    for key in test_seq:
        pred_idx = poi_freq[key]
        for ele in test_seq[key]:
            hr_1 = hit_rate(pred_idx, ele, 1)
            hr_5 = hit_rate(pred_idx, ele, 5)
            hr_10 = hit_rate(pred_idx, ele, 10)
            hr_1_tot += hr_1
            hr_5_tot += hr_5
            hr_10_tot += hr_10
            step += 1
    print(hr_1_tot / step, hr_5_tot / step, hr_10_tot / step, mrr / step)


if __name__ == '__main__':
    top()
    utop()