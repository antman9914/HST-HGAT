import os
import time, pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.cuda import amp
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils.dataloader import POI_Loader, MF_Loader, RNN_Dataset, STP_Dataloader
from utils.metric import *
from model.poiformer import TADGAT
from model.baseline import *
from model.hst_lstm import HSTLSTM_CLS
from model.stgcn import STGCN_CLS
from model.stp_udgat import STP_UDGAT


class FixNoiseStrategy():
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        mu = 0
        sd = 1/self.hidden_size
        self.h0 = torch.randn(self.hidden_size, requires_grad=False) * sd + mu
    
    def on_init(self, user_len, device):
        hs = []
        for i in range(user_len):
            hs.append(self.h0)
        if not st_flag:
            return torch.stack(hs, dim=0).view(1, user_len, self.hidden_size).to(device)
        else:
            return torch.stack(hs, dim=0).to(device)
    
    def on_reset(self):
        return self.h0


class LstmStrategy():
    
    def __init__(self, hidden_size):
        self.hidden_size=hidden_size
        self.h_strategy = FixNoiseStrategy(hidden_size)
        self.c_strategy = FixNoiseStrategy(hidden_size)
    
    def on_init(self, user_len, device):
        h = self.h_strategy.on_init(user_len, device)
        c = self.c_strategy.on_init(user_len, device)
        return (h,c)
    
    def on_reset(self):
        h = self.h_strategy.on_reset()
        c = self.c_strategy.on_reset()
        return (h,c)


class CELoss(nn.Module):
    def __init__(self, label_smooth=None):
        super().__init__()
        self.label_smooth = label_smooth

    def forward(self, pred, target):
        eps = 1e-12
        class_num = pred.size(-1)
        if self.label_smooth is not None:
            logprobs = F.log_softmax(pred, dim=1)
            target = F.one_hot(target, pred.size(-1)).to(device)
            target = torch.clamp(target.float(), min=self.label_smooth/(class_num-1), max=1.0-self.label_smooth)
            loss = -1*torch.sum(target*logprobs, 1)
        
        else:
            loss = -1.*pred.gather(1, target.unsqueeze(-1)) + torch.log(torch.exp(pred+eps).sum(dim=1))
        return loss.mean()


parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='train', help='running mode, choose between [train, test]')
parser.add_argument('--dataset', type=str, help='dataset chosen for training')
parser.add_argument('--model', type=str, default='poiformer', help='model under verification in this run')
parser.add_argument('--checkpoint_dir', '-cd', type=str, default='checkpoint', help='path of checkpoint file')
parser.add_argument('--gpu_id', type=int, default=-1, help='gpu id chosen to run train/test')
parser.add_argument('--layer_num', type=int, default=1, help='number of stacked model layer')
parser.add_argument('--max_len', type=int, default=50, help='maximum length of item/user sequence')
parser.add_argument('--ubias_num', type=int, default=180, help='number of user ego-nets attention bias')
parser.add_argument('--ibias_num', type=int, default=360, help='number of item ego-nets attention bias')
parser.add_argument('--input_dim', type=int, default=32, help='dimension of input feature')
parser.add_argument('--hidden_channel', type=int, default=32, help='dimension of hidden layer')
parser.add_argument('--weight_decay', type=float, default=0, help='hyperparameter of weight decay')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--test_bs', type=int, default=1024, help='batch size for test and validation')
parser.add_argument('--neg_sample_num', type=int, default=19, help='number of negative samples during training/validation')
parser.add_argument('--eval_per_n', type=int, default=1000, help='evaluate per n steps')
parser.add_argument('--epoch_num', type=int, default=30, help='training epoch number')
parser.add_argument('--ssl_temp', type=float, default=0.5, help='temperature hyperparameter for MTL')
parser.add_argument('--mtl_coef_1', type=float, default=0.5, help='coefficient of auxiliary task 1')
parser.add_argument('--mtl_coef_2', type=float, default=0.5, help='coefficient of auxiliary task 2')

args, _ = parser.parse_known_args()
checkpoint_path = '%s/%s.pth' % (args.checkpoint_dir, args.model)

# use_aux_init = args.add_aux_init == 1
# use_gt = args.add_gt == 1

# Print current experimental setting
print("Run %s" % time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
print("Current process ID: %d" % os.getpid())
print("Current Model: %s" % args.model)
print("Current Dataset: %s" % args.dataset)
print("Running Mode: %s" % args.mode)
print("------HyperParameter Settings------")
print("Number of Stacking Layer: %d" % args.layer_num)
print("In Channel: %d" % args.input_dim)
print("Hidden Channel: %d" % args.hidden_channel)
# print("Use Graph Embedding for POI Init: %s" % str(use_aux_init))
# print('Use Graph Transformer Layer: %s' % str(use_gt))
if args.mode == 'train':
    print("Maximum Sequence Length: %d" % args.max_len)
    print("Learning Rate: %f" % args.lr)
    print("Weight Decay: %f" % args.weight_decay)
    print("Batch Size: %d" % args.batch_size)
    print("Epoch Number: %d" % args.epoch_num)
    print("Evaluate per %d step" % args.eval_per_n)
    print("Multi-task coefficient: %.2f, %.2f" % (args.mtl_coef_1, args.mtl_coef_2))
    print("\n\n")
else:
    print("Batch Size: %d" % args.batch_size)
    print("\n\n")

if args.gpu_id != -1:
    device = torch.device('cuda:%d' % args.gpu_id if torch.cuda.is_available() else 'cpu')
else:
    device = torch.device('cpu')
enable_amp = True if "cuda" in device.type else False
enable_amp = False

# TODO: ucenter need 80 day slot, icenter need 150 day slot for gowalla
if 'TADGAT' in args.model:
    if args.mode == 'train':
        train_loader = POI_Loader(args.max_len, args.layer_num, args.ubias_num, args.ibias_num, args.neg_sample_num, dataset=args.dataset, mode='train', batch_size=args.batch_size, shuffle=True)    #, num_workers=1)
        print('train data loaded')
        val_loader = POI_Loader(args.max_len, args.layer_num, args.ubias_num, args.ibias_num, args.neg_sample_num, dataset=args.dataset, mode='val', batch_size=args.batch_size, shuffle=False)   #, num_workers=1)
        print('valid data loaded')
        user_num, item_num = train_loader.user_num, train_loader.poi_num
        # test_loader = POI_Loader(args.max_len, args.layer_num, args.ubias_num, args.ibias_num, args.neg_sample_num, mode='test', batch_size=args.batch_size, shuffle=False)  #, num_workers=1)
    else:
        test_loader = POI_Loader(args.max_len, args.layer_num, args.ubias_num, args.ibias_num, args.neg_sample_num, dataset=args.dataset, mode='test', batch_size=args.batch_size, shuffle=True)  #, num_workers=1)
        user_num, item_num = test_loader.user_num, test_loader.poi_num
        print('test data loaded')
elif args.model == 'MF' or args.model == 'LightGCN':
    if args.mode == 'train':
        train_loader = MF_Loader('train', batch_size=args.batch_size, dataset=args.dataset, shuffle=True)
        val_loader = MF_Loader('val', batch_size=args.batch_size, dataset=args.dataset, shuffle=False)
        user_num, item_num = train_loader.user_num, train_loader.poi_num
        edge_index = pickle.load(open('gowalla_processed/train_lgcn_eindex.pkl', 'rb'))
        edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
    else:
        test_loader = MF_Loader('test', batch_size=args.batch_size, dataset=args.dataset, shuffle=False)
        user_num, item_num = test_loader.user_num, test_loader.poi_num
        edge_index = pickle.load(open('gowalla_processed/val_lgcn_eindex.pkl', 'rb'))
        edge_index = torch.tensor(edge_index, dtype=torch.int64).to(device)
    print('Data Loaded')
elif 'LSTM' in args.model or args.model == 'STGCN':
    st_flag = 'ST' in args.model and args.model != 'LSTM'
    if args.mode == 'train':
        train_dataset = RNN_Dataset(15, args.batch_size, st_flag, args.dataset, 'train')
        val_dataset = RNN_Dataset(15, args.batch_size, st_flag, args.dataset, 'val')
        train_loader, val_loader = DataLoader(train_dataset, batch_size=1), DataLoader(val_dataset, batch_size=1)
        user_num, item_num = train_dataset.user_num, train_dataset.poi_num
    else:
        test_dataset = RNN_Dataset(15, args.batch_size, st_flag, args.dataset, 'test')
        test_loader = DataLoader(test_dataset, batch_size=1)
        user_num, item_num = test_dataset.user_num, test_dataset.poi_num
elif args.model == 'STP_UDGAT':
    if args.mode == 'train':
        train_loader = STP_Dataloader('train', args.dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = STP_Dataloader('val', args.dataset, batch_size=args.batch_size, shuffle=False)
        user_num, item_num = train_loader.user_num, train_loader.poi_num
    else:
        test_loader = STP_Dataloader('test', args.dataset, batch_size=args.batch_size, shuffle=False)
        user_num, item_num = test_loader.user_num, test_loader.poi_num

print(user_num, item_num)
neg_sample_num = args.neg_sample_num
if 'TADGAT' in args.model:  
    model = TADGAT(args.input_dim, args.hidden_channel, args.layer_num, user_num, item_num, args.ubias_num, args.ibias_num, ssl_temp=args.ssl_temp, aux_feat=None)
elif args.model == 'MF':
    model = MF(args.input_dim, user_num, item_num)
elif 'LSTM' in args.model:
    if st_flag:
        model = HSTLSTM_CLS(args.input_dim, item_num, 168, 300)
    else:
        print('LSTM')
        model = LSTM_Basic(args.input_dim, item_num)
elif args.model == 'STGCN':
    model = STGCN_CLS(args.input_dim, item_num)
elif args.model == 'STP_UDGAT':
    model = STP_UDGAT(args.input_dim, args.hidden_channel, user_num, item_num)
elif args.model == 'LightGCN':
    model = LightGCN(args.input_dim, args.layer_num, user_num, item_num)
model = model.to(device)

if args.mode == 'train':
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20, 30], gamma=0.4)
    scaler = amp.GradScaler(enabled=enable_amp)

    criterion = CELoss(label_smooth=0.05)


def train(best_ndcg, best_hr):
    model.train()
    step = 0
    train_loader.soc_edges = train_loader.soc_edges.to(device)
    for all_out in train_loader:
        optimizer.zero_grad()
        adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, sampled_edges = all_out
        # adjs, node_idx, soc_adjs, soc_node_idx, orig_seqs, time_diff, seq_lens, center_nid, soc_cnid, sampled_edges = all_out
        adjs = adjs.to(device)
        node_idx = node_idx.to(device)
        center_nid = center_nid.to(device)

        # soc_adjs = soc_adjs.to(device)
        # soc_node_idx = soc_node_idx.to(device)

        # orig_seqs = orig_seqs.to(device)
        # time_diff = time_diff.to(device)
        # seq_lens = seq_lens.to(device)

        # sampled_idx = sampled_edges[:, 1:]
        # sampled_idx = sampled_idx.to(device)
        # label = torch.zeros(sampled_edges.size(0), dtype=torch.int64)
        label = sampled_edges[:, 1]
        # sampled_idx = torch.arange(item_num) + user_num
        # sampled_idx = sampled_idx.repeat(label.size(0)).reshape(-1, item_num)
        # sampled_idx = sampled_idx.to(device)
        label = label.to(device)

        # lat_emb_idx, lon_emb_idx = lat_emb_idx.to(device), lon_emb_idx.to(device)

        with amp.autocast(enabled=enable_amp):
            logits, pref_bound = model(adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, device, train_loader.soc_edges)
            # logits = model(adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, device, soc_adjs, soc_node_idx, soc_cnid)
            
            # Code for ranking-based loss
            loss = 0.
            for k in range(logits.size(0)):
                pos_logit = logits[k, label[k]]
                neg_logit = torch.cat([logits[k, :label[k]], logits[k, (label[k]+1):]])
                # pos_loss = F.logsigmoid(pos_logit - pref_bound[k])
                # neg_loss = F.logsigmoid(pref_bound[k] - neg_logit).mean()
                pos_loss = F.logsigmoid(pos_logit)
                neg_loss = F.logsigmoid(-neg_logit).mean()
                loss += - pos_loss - neg_loss
            loss /= logits.size(0)
            # loss = criterion(logits, label) # + args.mtl_coef_1 * mtl_loss_1
            # new_end = time.time()
            # print(new_end - end)
        if np.isnan(loss.detach().cpu()):
            print("NaN detected!")
            exit(1)
        label = label.cpu()
        if step % 200 == 0:
            label = label.numpy()
            logits = logits.detach()
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
            # hr_1 += hit_rate(sorted_idx, label, k=1)
            # hr_5 += hit_rate(sorted_idx, label, k=5)
            # hr_10 += hit_rate(sorted_idx, label, k=10)
            hr_1 = hit_rate(sorted_idx, label, k=1) / args.batch_size
            hr_5 = hit_rate(sorted_idx, label, k=5) / args.batch_size
            hr_10 = hit_rate(sorted_idx, label, k=10) / args.batch_size
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        # start = time.time()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # end = time.time()
        # print(end - start)

        # if step % 20 == 0:
        #     hr_1 /= len(all_out) ; hr_5 /= len(all_out) ; hr_10 /= len(all_out)
        #     print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))

        # if step != 0 and step % args.eval_per_n == 0:
        #     start = time.time()
        #     hr_1, hr_5, hr_10, mrr, ndcg = test('val')
        #     end = time.time()
        #     print("time consumption: %.4f" % (end-start))
        #     print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
        #     if ndcg > best_ndcg:
        #         best_ndcg = ndcg
        #         print("New best model saved")
        #         torch.save(model.state_dict(), checkpoint_path)
        step += 1
        # end = time.time()
        # print(end - all_start)
    scheduler.step()

    # if step % args.eval_per_n > 20:
    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if ndcg > best_ndcg or hr_10 > best_hr:
        if ndcg > best_ndcg:
            best_ndcg = ndcg
        else:
            best_hr = hr_10
        print("New best model saved")
        torch.save(model.state_dict(), checkpoint_path)    
    return best_ndcg, best_hr


def train_stp(best_ndcg, best_hr):
    model.train()
    step = 0
    train_loader.u_eindex = train_loader.u_eindex.to(device)
    for out in train_loader:
        optimizer.zero_grad()
        s_adj, t_adj, f_adj, s_rw_adj, t_rw_adj, f_rw_adj, pp_adj, anchors, label = out
        s_adj, t_adj, f_adj = s_adj.to(device), t_adj.to(device), f_adj.to(device)
        s_rw_adj, t_rw_adj, f_rw_adj = s_rw_adj.to(device), t_rw_adj.to(device), f_rw_adj.to(device)
        pp_adj = pp_adj.to(device)
        anchors = anchors.to(device)
        label = label.to(device)
        with amp.autocast(enabled=enable_amp):
            logits = model(s_adj, t_adj, f_adj, s_rw_adj, t_rw_adj, f_rw_adj, pp_adj, train_loader.u_eindex, anchors)
            loss = criterion(logits, label)
        label = label.cpu()
        if step % 200 == 0:
            label = label.numpy()
            logits = logits.detach()
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
            hr_1 = hit_rate(sorted_idx, label, k=1) / args.batch_size
            hr_5 = hit_rate(sorted_idx, label, k=5) / args.batch_size
            hr_10 = hit_rate(sorted_idx, label, k=10) / args.batch_size
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        step += 1
    scheduler.step()
    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test_stp('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if ndcg > best_ndcg or hr_10 > best_hr:
        if ndcg > best_ndcg:
            best_ndcg = ndcg
        else:
            best_hr = hr_10
        print("New best model saved")
        torch.save(model.state_dict(), checkpoint_path)    
    return best_ndcg, best_hr


def train_mf(best_ndcg, best_hr):
    model.train()
    step = 0
    for out in train_loader:
        optimizer.zero_grad()
        center_uid, seq = out
        with amp.autocast(enabled=enable_amp):
            if args.model == 'LightGCN':
                loss, logits = model(edge_index, center_uid, seq)
            else:
                loss, logits = model(center_uid, seq)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step % 5 == 0:
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
            hr_1 = top_k_acc(sorted_idx, seq, k=1) / args.batch_size
            hr_5 = top_k_acc(sorted_idx, seq, k=5) / args.batch_size
            hr_10 = top_k_acc(sorted_idx, seq, k=10) / args.batch_size
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        step += 1
    scheduler.step()
    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test_mf('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if hr_10 > best_hr:
        if ndcg > best_ndcg:
            best_ndcg = ndcg
        else:
            best_hr = hr_10
        print("New best model saved")
        torch.save(model.state_dict(), checkpoint_path)    
    return best_ndcg, best_hr


def train_rnn(best_ndcg, best_hr):
    model.train()
    step = 0
    h0_strategy = LstmStrategy(args.input_dim)
    h = h0_strategy.on_init(args.batch_size, device)
    for out in train_loader:
        optimizer.zero_grad()
        with torch.autograd.set_detect_anomaly(True):
            if not st_flag:
                seqs, lbls, reset_h = out
                seqs, lbls = seqs.to(device).squeeze(), lbls.to(device).squeeze()
            else:
                seqs, lbls, t_l, t_u, t_coef, s_l, s_u, s_coef, reset_h = out
                seqs, lbls = seqs.to(device).squeeze().T, lbls.to(device).squeeze().T
                # Note: the unsqueeze is for STGCN only
                t_l, t_u, t_coef = t_l.to(device).squeeze(), t_u.to(device).squeeze(), t_coef.to(device).squeeze()
                s_l, s_u, s_coef = s_l.to(device).squeeze(), s_u.to(device).squeeze(), s_coef.to(device).squeeze()
            lbls = lbls.reshape(-1)
            for j, reset in enumerate(reset_h):
                if reset:
                    hc = h0_strategy.on_reset()
                    if st_flag:
                        h[0][j] = hc[0]
                        h[1][j] = hc[1]
                    else:
                        h[0][0, j] = hc[0]
                        h[1][0, j] = hc[1]
            with amp.autocast(enabled=enable_amp):
                if not st_flag:
                    logits, h = model(seqs, h)
                else:
                    if args.model == 'ST_LSTM':
                        logits, h = model(seqs, t_l, t_u, s_l, s_u, t_coef, s_coef, h)
                    elif args.model == 'STGCN':
                        logits, h = model(seqs, t_l, s_l, h)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward(retain_graph=True)
            scaler.step(optimizer)
            scaler.update()
        h = (h[0].detach(), h[1].detach())
        if step % 100 == 0:
            sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
            hr_1 = hit_rate(sorted_idx, lbls, k=1) / logits.size(0)
            hr_5 = hit_rate(sorted_idx, lbls, k=5) / logits.size(0)
            hr_10 = hit_rate(sorted_idx, lbls, k=10) / logits.size(0)
            print("step %d : loss %.4f, HR@1 %.4f, HR@5 %.4f, HR@10 %.4f" % (step, float(loss), hr_1, hr_5, hr_10))
        step += 1
    scheduler.step()
    start = time.time()
    hr_1, hr_5, hr_10, mrr, ndcg = test_rnn('val')
    end = time.time()
    print("time consumption: %.4f" % (end - start))
    print("Test result on validation set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    if ndcg > best_ndcg or hr_10 > best_hr:
        if ndcg > best_ndcg:
            best_ndcg = ndcg
        else:
            best_hr = hr_10
        print("New best model saved")
        torch.save(model.state_dict(), checkpoint_path)    
    return best_ndcg, best_hr

@torch.no_grad()
def test_mf(mode):
    print('Start Test...')
    model.eval()
    hr_1_tot, hr_5_tot, hr_10_tot = [], [], []
    if mode == 'test':
        model.load_state_dict(torch.load(checkpoint_path))
    cur_loader = val_loader if mode == 'val' else test_loader
    mrr = []
    for out in cur_loader:
        center_uid, seq = out
        if args.model == 'LightGCN':
            logits = model(edge_index, center_uid, seq)
        else:
            logits = model(center_uid, seq)
        sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        hr_1 = top_k_acc(sorted_idx, seq, k=1) / args.batch_size
        hr_5 = top_k_acc(sorted_idx, seq, k=5) / args.batch_size
        hr_10 = top_k_acc(sorted_idx, seq, k=10) / args.batch_size
        hr_1_tot.append(hr_1)
        hr_5_tot.append(hr_5)
        hr_10_tot.append(hr_10)

        y_true = np.zeros_like(sorted_idx)
        for i in range(sorted_idx.shape[0]):
            for j in range(len(seq[i])):
                y_true[i, seq[i][j]] = 1
        for i in range(sorted_idx.shape[0]):
            y_true[i] = y_true[i][sorted_idx[i]]
        
        rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
        mrr.append(np.sum(rr_score) / np.sum(y_true))
    return np.mean(hr_1_tot), np.mean(hr_5_tot), np.mean(hr_10_tot), np.mean(mrr), 0


@torch.no_grad()
def test_rnn(mode):
    print('Start Test...')
    hr_1_tot, hr_5_tot, hr_10_tot = [], [], []
    mrr = []
    if mode == 'test':
        model.load_state_dict(torch.load(checkpoint_path))
    cur_loader = val_loader if mode == 'val' else test_loader
    h0_strategy = LstmStrategy(args.input_dim)
    h = h0_strategy.on_init(args.batch_size, device)
    for out in cur_loader:
        if not st_flag:
            seqs, lbls, reset_h = out
            seqs = seqs.to(device)
            seqs, lbls = seqs.squeeze(), lbls.squeeze()
        else:
            seqs, lbls, t_l, t_u, t_coef, s_l, s_u, s_coef, reset_h = out
            seqs, lbls = seqs.to(device).squeeze().T, lbls.squeeze().T
            t_l, t_u, t_coef = t_l.to(device).squeeze(), t_u.to(device).squeeze(), t_coef.to(device).squeeze()
            s_l, s_u, s_coef = s_l.to(device).squeeze(), s_u.to(device).squeeze(), s_coef.to(device).squeeze()
        lbls = lbls.reshape(-1)
        for j, reset in enumerate(reset_h):
            if reset:
                hc = h0_strategy.on_reset()
                if st_flag:
                    h[0][j] = hc[0]
                    h[1][j] = hc[1]
                else:
                    h[0][0, j] = hc[0]
                    h[1][0, j] = hc[1]
        if st_flag:
            if args.model == 'STGCN':
                logits, h = model(seqs, t_l, s_l, h)
            else:
                logits, h = model(seqs, t_l, t_u, s_l, s_u, t_coef, s_coef, h)
        else:
            logits, h = model(seqs, h)
        sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        label = lbls.numpy()
        hr_1 = hit_rate(sorted_idx, label, k=1) / logits.size(0)
        hr_5 = hit_rate(sorted_idx, label, k=5) / logits.size(0)
        hr_10 = hit_rate(sorted_idx, label, k=10) / logits.size(0)
        hr_1_tot.append(hr_1)
        hr_5_tot.append(hr_5)
        hr_10_tot.append(hr_10)

        y_true = np.zeros_like(sorted_idx)
        for i in range(sorted_idx.shape[0]):
            y_true[i, label[i]] = 1
        for i in range(sorted_idx.shape[0]):
            y_true[i] = y_true[i][sorted_idx[i]]
        
        rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
        mrr.append(np.sum(rr_score) / np.sum(y_true))
    return np.mean(hr_1_tot), np.mean(hr_5_tot), np.mean(hr_10_tot), np.mean(mrr), 0


@torch.no_grad()
def test_stp(mode):
    print('Start Test...')
    model.eval()
    hr_1_tot, hr_5_tot, hr_10_tot = [], [], []
    mrr = []
    if mode == 'test':
        model.load_state_dict(torch.load(checkpoint_path))
    cur_loader = val_loader if mode == 'val' else test_loader
    cur_loader.u_eindex = cur_loader.u_eindex.to(device)
    for out in cur_loader:
        s_adj, t_adj, f_adj, s_rw_adj, t_rw_adj, f_rw_adj, pp_adj, anchors, label = out
        s_adj, t_adj, f_adj = s_adj.to(device), t_adj.to(device), f_adj.to(device)
        s_rw_adj, t_rw_adj, f_rw_adj = s_rw_adj.to(device), t_rw_adj.to(device), f_rw_adj.to(device)
        pp_adj = pp_adj.to(device)
        anchors = anchors.to(device)
        logits = model(s_adj, t_adj, f_adj, s_rw_adj, t_rw_adj, f_rw_adj, pp_adj, cur_loader.u_eindex, anchors)
        sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        label = label.numpy()
        hr_1 = hit_rate(sorted_idx, label, k=1) / logits.size(0)
        hr_5 = hit_rate(sorted_idx, label, k=5) / logits.size(0)
        hr_10 = hit_rate(sorted_idx, label, k=10) / logits.size(0)
        hr_1_tot.append(hr_1)
        hr_5_tot.append(hr_5)
        hr_10_tot.append(hr_10)
        y_true = np.zeros_like(sorted_idx)
        for i in range(sorted_idx.shape[0]):
            y_true[i, label[i]] = 1
        for i in range(sorted_idx.shape[0]):
            y_true[i] = y_true[i][sorted_idx[i]]
        
        rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
        mrr.append(np.sum(rr_score) / np.sum(y_true))
    
    return np.mean(hr_1_tot), np.mean(hr_5_tot), np.mean(hr_10_tot), np.mean(mrr), 0


@torch.no_grad()
def test(mode):
    print('Start Test...')
    model.eval()
    hr_1_tot, hr_5_tot, hr_10_tot, mrr, ndcg_tot = [], [], [], [], []

    # TODO: uncomment the checkpoint loading operation during real test
    if mode == 'test':
        model.load_state_dict(torch.load(checkpoint_path))
    cur_loader = val_loader if mode == 'val' else test_loader
    cur_loader.soc_edges = cur_loader.soc_edges.to(device)
    # step = 0
    for n, all_out in enumerate(cur_loader): 
        # adjs, node_idx, center_nid, sampled_edges = all_out
        # adjs, node_idx, soc_adjs, soc_node_idx, orig_seqs, time_diff, seq_lens, center_nid, soc_cnid, sampled_edges = all_out
        adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, sampled_edges = all_out

        # if mode == 'val':
        #     sampled_idx = sampled_edges[:, 1:]
        #     label = torch.zeros(sampled_edges.size(0), dtype=torch.int64)
        # else:
        label = sampled_edges[:, 1]
        # sampled_idx = torch.arange(item_num) + user_num
        # sampled_idx = sampled_idx.repeat(label.size(0)).reshape(-1, item_num)
        # adjs = [adj.to(device) for adj in adjs]
        adjs = adjs.to(device)
        node_idx = node_idx.to(device)
        center_nid = center_nid.to(device)

        # soc_adjs = soc_adjs.to(device)
        # soc_node_idx = soc_node_idx.to(device)

        # orig_seqs = orig_seqs.to(device)
        # time_diff = time_diff.to(device)
        # seq_lens = seq_lens.to(device)

        # logits = model(adjs, node_idx, center_nid, device)
        logits = model(adjs, node_idx, orig_seqs, time_diff, seq_lens, center_nid, device, cur_loader.soc_edges)# , soc_adjs, soc_node_idx, soc_cnid)

        # full_logits.append(logits)
        # full_labels.append(label)
        # if mode == 'test':
        #     full_logits.append(logits)

        # logits = torch.cat(full_logits, dim=0)
        # label = torch.cat(full_labels)
        # random_noise = torch.rand(logits.numpy().shape) * 1e-7
        # logits = logits + random_noise
        sorted_idx = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        label = label.numpy()
        # if mode == 'test':
        hr_1 = hit_rate(sorted_idx, label, k=1) / logits.size(0)
        hr_5 = hit_rate(sorted_idx, label, k=5) / logits.size(0)
        hr_10 = hit_rate(sorted_idx, label, k=10) / logits.size(0)
        # else:
        #     hr_1 = hit_rate(sorted_idx, label, k=1)
        #     hr_5 = hit_rate(sorted_idx, label, k=5)
        #     hr_10 = hit_rate(sorted_idx, label, k=10)
        # hr_50 = hit_rate(sorted_idx, k=50)
        hr_1_tot.append(hr_1)
        hr_5_tot.append(hr_5)
        hr_10_tot.append(hr_10)
        y_true = np.zeros_like(sorted_idx)
        for i in range(sorted_idx.shape[0]):
            y_true[i, label[i]] = 1
        for i in range(sorted_idx.shape[0]):
            y_true[i] = y_true[i][sorted_idx[i]]
        
        rr_score = y_true / (np.arange(np.shape(y_true)[1]) + 1)
        mrr.append(np.sum(rr_score) / np.sum(y_true))
        
        # y_true = np.zeros_like(sorted_idx)
        # for i in range(len(label)):
        #     y_true[i, label[i]] = 1
        
        # mrr_true = np.take(y_true, sorted_idx)
        # rr_score = mrr_true / (np.arange(np.shape(mrr_true)[1]) + 1)
        # mrr += np.sum(np.sum(rr_score, axis=1) / np.sum(mrr_true, axis=1))

        # ndcg = ndcg_score(y_true, sorted_idx, k=10)
        # if np.nan not in ndcg:
        #     # ndcg_tot += np.mean(ndcg)
        #     ndcg_tot += np.sum(ndcg)
        # # del full_logits, full_labels
        # # full_logits, full_labels = [], []
        
        # if n % 100 == 0 and n != 0:
        #     print(n)
        # # step += 1
    
    return np.mean(hr_1_tot), np.mean(hr_5_tot), np.mean(hr_10_tot), np.mean(mrr), 0 # mrr / step, ndcg_tot / step


np.random.seed(1)
torch.manual_seed(1)
if args.mode == 'train':
    ndcg = 0.
    hr_10 = 0.
    # start = time.time()
    # hr_1, hr_5, hr_10, mrr, ndcg = test_stp('val')
    # end = time.time()
    # print("time consumption: %.4f" % (end-start))
    # print("Test result on val set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))
    for epoch in range(1, args.epoch_num + 1):
        print("Currently epoch %d:" % epoch)
        print("Learning rate: %f" % scheduler.get_lr()[0])
        if 'TADGAT' in args.model:
            ndcg, hr_10 = train(ndcg, hr_10)
        elif args.model == 'MF' or args.model == 'LightGCN':
            ndcg, hr_10 = train_mf(ndcg, hr_10)
        elif 'LSTM' in args.model or args.model == 'STGCN':
            train_dataset.reset()
            ndcg, hr_10 = train_rnn(ndcg, hr_10)
        elif args.model == 'STP_UDGAT':
            ndcg, hr_10 = train_stp(ndcg, hr_10)
else:
    start = time.time()
    if 'TADGAT' in args.model:
        hr_1, hr_5, hr_10, mrr, ndcg = test('test')
    elif args.model == 'MF' or args.model == 'LightGCN':
        hr_1, hr_5, hr_10, mrr, ndcg = test_mf('test')
    elif 'LSTM' in args.model or args.model == 'STGCN':
        hr_1, hr_5, hr_10, mrr, ndcg = test_rnn('test')
    elif args.model == 'STP_UDGAT':
        hr_1, hr_5, hr_10, mrr, ndcg = test_stp('test')
    end = time.time()
    print("time consumption: %.4f" % (end-start))
    print("Test result on test set: HR@1 %.4f, HR@5 %.4f, HR@10 %.4f, MRR: %.4f, NDCG@10: %.4f" % (hr_1, hr_5, hr_10, mrr, ndcg))

