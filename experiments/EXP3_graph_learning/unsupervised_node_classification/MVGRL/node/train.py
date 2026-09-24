import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from pathlib import Path
import sys

_MVGRL_ROOT = Path(__file__).resolve().parents[1]
if str(_MVGRL_ROOT) not in sys.path:
    sys.path.insert(0, str(_MVGRL_ROOT))
_EXPERIMENT_DIR = Path(__file__).resolve().parents[2]
if str(_EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(_EXPERIMENT_DIR))

try:
    from ..utils import sparse_mx_to_torch_sparse_tensor
    from .dataset import load
except ImportError:  # Support direct execution from the original repository layout.
    from utils import sparse_mx_to_torch_sparse_tensor
    from node.dataset import load
from evaluation_protocol import get_split


# Borrowed from https://github.com/PetarV-/DGI
class GCN(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(GCN, self).__init__()
        self.fc = nn.Linear(in_ft, out_ft, bias=False)
        self.act = nn.PReLU()

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_ft))
            self.bias.data.fill_(0.0)
        else:
            self.register_parameter('bias', None)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    # Shape of seq: (batch, nodes, features)
    def forward(self, seq, adj, sparse=False):
        seq_fts = self.fc(seq)
        if sparse:
            out = torch.unsqueeze(torch.spmm(adj, torch.squeeze(seq_fts, 0)), 0)
        else:
            out = torch.bmm(adj, seq_fts)
        if self.bias is not None:
            out += self.bias
        return self.act(out)


# Borrowed from https://github.com/PetarV-/DGI
class Readout(nn.Module):
    def __init__(self):
        super(Readout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.mean(seq * msk, 1) / torch.sum(msk)


# Borrowed from https://github.com/PetarV-/DGI
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c1, c2, h1, h2, h3, h4, s_bias1=None, s_bias2=None):
        c_x1 = torch.unsqueeze(c1, 1)
        c_x1 = c_x1.expand_as(h1).contiguous()
        c_x2 = torch.unsqueeze(c2, 1)
        c_x2 = c_x2.expand_as(h2).contiguous()

        # positive
        sc_1 = torch.squeeze(self.f_k(h2, c_x1), 2)
        sc_2 = torch.squeeze(self.f_k(h1, c_x2), 2)

        # negetive
        sc_3 = torch.squeeze(self.f_k(h4, c_x1), 2)
        sc_4 = torch.squeeze(self.f_k(h3, c_x2), 2)

        logits = torch.cat((sc_1, sc_2, sc_3, sc_4), 1)
        return logits


class Model(nn.Module):
    def __init__(self, n_in, n_h):
        super(Model, self).__init__()
        self.gcn1 = GCN(n_in, n_h)
        self.gcn2 = GCN(n_in, n_h)
        self.read = Readout()

        self.sigm = nn.Sigmoid()

        self.disc = Discriminator(n_h)

    def forward(self, seq1, seq2, adj, diff, sparse, msk, samp_bias1, samp_bias2):
        h_1 = self.gcn1(seq1, adj, sparse)
        c_1 = self.read(h_1, msk)
        c_1 = self.sigm(c_1)

        h_2 = self.gcn2(seq1, diff, sparse)
        c_2 = self.read(h_2, msk)
        c_2 = self.sigm(c_2)

        h_3 = self.gcn1(seq2, adj, sparse)
        h_4 = self.gcn2(seq2, diff, sparse)

        ret = self.disc(c_1, c_2, h_1, h_2, h_3, h_4, samp_bias1, samp_bias2)

        return ret, h_1, h_2

    def embed(self, seq, adj, diff, sparse, msk):
        h_1 = self.gcn1(seq, adj, sparse)
        c = self.read(h_1, msk)

        h_2 = self.gcn2(seq, diff, sparse)
        return (h_1 + h_2).detach(), c.detach()


class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)
        self.sigm = nn.Sigmoid()

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = torch.log_softmax(self.fc(seq), dim=-1)
        return ret


def train(
    dataset,
    data_root,
    device,
    checkpoint_path,
    split_seed,
    verbose=False,
    *,
    views=None,
    return_metrics=False,
    nb_epochs=3000,
    patience=20,
    lr=0.001,
    l2_coef=0.0,
    hid_units=512,
    sample_size=2000,
    batch_size=4,
    classifier_epochs=300,
    classifier_lr=1e-2,
    classifier_weight_decays=(0.0, 0.001, 0.005, 0.01, 0.1),
    log_interval=1,
):
    sparse = False

    if views is None:
        adj, diff, features, labels, *_ = load(dataset, data_root)
    else:
        adj, diff, features, labels = views
    split = get_split(len(labels), split_seed)
    idx_train = split['train']
    idx_val = split['valid']
    idx_test = split['test']

    ft_size = features.shape[1]
    nb_classes = np.unique(labels).shape[0]

    if not 1 <= sample_size <= adj.shape[-1]:
        raise ValueError(
            f'sample_size must be in [1, {adj.shape[-1]}], got {sample_size}.'
        )

    labels = torch.LongTensor(labels)
    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    lbl_1 = torch.ones(batch_size, sample_size * 2)
    lbl_2 = torch.zeros(batch_size, sample_size * 2)
    lbl = torch.cat((lbl_1, lbl_2), 1)

    model = Model(ft_size, hid_units)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)

    model = model.to(device)
    labels = labels.to(device)
    lbl = lbl.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    b_xent = nn.BCEWithLogitsLoss()
    xent = nn.CrossEntropyLoss()
    cnt_wait = 0
    best = 1e9
    best_t = 0

    for epoch in range(nb_epochs):

        idx = np.random.randint(0, adj.shape[-1] - sample_size + 1, batch_size)
        ba, bd, bf = [], [], []
        for i in idx:
            ba.append(adj[i: i + sample_size, i: i + sample_size])
            bd.append(diff[i: i + sample_size, i: i + sample_size])
            bf.append(features[i: i + sample_size])

        ba = np.array(ba).reshape(batch_size, sample_size, sample_size)
        bd = np.array(bd).reshape(batch_size, sample_size, sample_size)
        bf = np.array(bf).reshape(batch_size, sample_size, ft_size)

        if sparse:
            ba = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(ba))
            bd = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(bd))
        else:
            ba = torch.FloatTensor(ba)
            bd = torch.FloatTensor(bd)

        bf = torch.FloatTensor(bf)
        idx = np.random.permutation(sample_size)
        shuf_fts = bf[:, idx, :]

        bf = bf.to(device)
        ba = ba.to(device)
        bd = bd.to(device)
        shuf_fts = shuf_fts.to(device)

        model.train()
        optimiser.zero_grad()

        logits, __, __ = model(bf, shuf_fts, ba, bd, sparse, None, None, None)

        loss = b_xent(logits, lbl)

        loss.backward()
        optimiser.step()

        if verbose and (epoch == 0 or (epoch + 1) % log_interval == 0):
            print(
                'Epoch: {0}, Loss: {1:0.4f}'.format(epoch, loss.item()),
                flush=True,
            )

        if loss < best:
            best = loss
            best_t = epoch
            cnt_wait = 0
            torch.save(model.state_dict(), checkpoint_path)
        else:
            cnt_wait += 1

        if cnt_wait == patience:
            if verbose:
                print('Early stopping!')
            break

    if verbose:
        print('Loading {}th epoch'.format(best_t))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if sparse:
        adj = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(adj))
        diff = sparse_mx_to_torch_sparse_tensor(sp.coo_matrix(diff))

    features = torch.FloatTensor(features[np.newaxis])
    adj = torch.FloatTensor(adj[np.newaxis])
    diff = torch.FloatTensor(diff[np.newaxis])
    features = features.to(device)
    adj = adj.to(device)
    diff = diff.to(device)

    embeds, _ = model.embed(features, adj, diff, sparse, None)
    train_embs = embeds[0, idx_train]
    val_embs = embeds[0, idx_val]
    test_embs = embeds[0, idx_test]

    train_lbls = labels[idx_train]
    val_lbls = labels[idx_val]
    test_lbls = labels[idx_test]

    best_classifier = None
    best_val_acc = -1.0
    best_weight_decay = None
    best_epoch = None
    for wd in classifier_weight_decays:
        torch.manual_seed(split_seed)
        torch.cuda.manual_seed_all(split_seed)
        log = LogReg(hid_units, nb_classes)
        opt = torch.optim.Adam(
            log.parameters(), lr=classifier_lr, weight_decay=wd
        )
        log = log.to(device)
        wd_best_val_acc = -1.0
        wd_best_state = None
        wd_best_epoch = None
        for epoch in range(classifier_epochs):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

            log.eval()
            with torch.no_grad():
                val_preds = torch.argmax(log(val_embs), dim=1)
                val_acc = (val_preds == val_lbls).float().mean().item()
                if val_acc > wd_best_val_acc:
                    wd_best_val_acc = val_acc
                    wd_best_epoch = epoch + 1
                    wd_best_state = {
                        key: value.detach().clone()
                        for key, value in log.state_dict().items()
                    }
        if wd_best_val_acc > best_val_acc:
            best_val_acc = wd_best_val_acc
            best_weight_decay = wd
            best_epoch = wd_best_epoch
            best_classifier = wd_best_state

    log = LogReg(hid_units, nb_classes).to(device)
    log.load_state_dict(best_classifier)
    log.eval()
    with torch.no_grad():
        test_preds = torch.argmax(log(test_embs), dim=1)
        test_acc = (test_preds == test_lbls).float().mean().item() * 100
    print(
        f'(E) | TestAcc={test_acc:.4f}, ValAcc={best_val_acc * 100:.4f}, '
        f'weight_decay={best_weight_decay}, epoch={best_epoch}',
        flush=True,
    )
    if return_metrics:
        return {
            'test_accuracy': test_acc / 100.0,
            'val_accuracy': best_val_acc,
            'eval_weight_decay': best_weight_decay,
            'eval_epoch': best_epoch,
        }
    return test_acc


if __name__ == '__main__':
    import argparse
    import warnings
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['cora', 'citeseer'], default='cora')
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('MVGRL requires a visible CUDA device.')
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_accuracies = []
    for run in range(args.runs):
        run_seed = args.seed + run
        np.random.seed(run_seed)
        torch.manual_seed(run_seed)
        torch.cuda.manual_seed_all(run_seed)
        checkpoint = args.checkpoint_dir / f'{args.dataset}_run{run:02d}.pt'
        print(
            f'[run] dataset={args.dataset} run={run + 1}/{args.runs} '
            f'seed={run_seed}',
            flush=True,
        )
        accuracy = train(
            args.dataset, args.data_root, device, checkpoint, run_seed,
            args.verbose,
        )
        run_accuracies.append(accuracy)
    print(
        f'(E-final) | TestAcc={np.mean(run_accuracies):.4f}+-'
        f'{np.std(run_accuracies):.4f} ({args.runs} pretraining runs)',
        flush=True,
    )
