# This is the usecase for node classification task with full batch training

import argparse
import numpy as np
import random
import os
import os.path as osp
import sys
sys.path.append('../')

import torch
from torch import nn
import torch_geometric.transforms as T
from tqdm import tqdm
from torch.optim import Adam
from torch_geometric.nn import GCNConv
from torch_geometric.nn.inits import uniform
from torch_geometric.datasets import Planetoid

from utils import seed_everything    
from Loss import JSD 
from Evaluator import get_split, LREvaluator
from ContrastMode import DualBranchContrast
from Augmentor import Compose, FeatureAugmentor, SpectralAugmentor


###################### Backbone Model ######################

class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, edge_weight=None):
        z = x
        for conv in self.layers:
            z = conv(z, edge_index, edge_weight)
            z = self.activation(z)
        return z


###################### GCL Encoder ######################

class Encoder(torch.nn.Module):
    def __init__(self, encoder1, encoder2, augmentor, hidden_dim):
        super(Encoder, self).__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.augmentor = augmentor
        self.project = torch.nn.Linear(hidden_dim, hidden_dim)
        uniform(hidden_dim, self.project.weight)

    @staticmethod
    def corruption(x, edge_index, edge_weight=None):
        return x[torch.randperm(x.size(0))], edge_index, edge_weight

    def forward(self, data):  
        x, edge_index = data.x, data.edge_index
        ptb_prob1 = data.max
        ptb_prob2 = data.min
        
        aug1, aug2 = self.augmentor
        x1, edge_index1, _ = aug1(x, edge_index, ptb_prob1, batch=None)
        x2, edge_index2, _ = aug2(x, edge_index, ptb_prob2, batch=None)
        z1 = self.encoder1(x1, edge_index1)
        z2 = self.encoder2(x2, edge_index2)
        g1 = self.project(torch.sigmoid(z1.mean(dim=0, keepdim=True)))
        g2 = self.project(torch.sigmoid(z2.mean(dim=0, keepdim=True)))
        z1n = self.encoder1(*self.corruption(x1, edge_index1))
        z2n = self.encoder2(*self.corruption(x2, edge_index2))
        return z1, z2, g1, g2, z1n, z2n

    def encode_clean(self, data):
        """Encode the same unaugmented graph with both trained towers."""
        return (
            self.encoder1(data.x, data.edge_index),
            self.encoder2(data.x, data.edge_index),
        )


###################### GCL Training and Testing ######################

def train(encoder_model, contrast_model, data, optimizer):
    encoder_model.train()
    optimizer.zero_grad()
    z1, z2, g1, g2, z1n, z2n = encoder_model(data)
    loss = contrast_model(h1=z1, h2=z2, g1=g1, g2=g2, h3=z1n, h4=z2n)
    loss.backward()
    optimizer.step()
    return loss.item()


def print_edit_summary(label, history):
    for view_name, values in history.items():
        stats = {
            key: np.asarray([value[key] for value in values], dtype=float)
            for key in values[0]
        }
        print(
            f'({label}): {view_name} '
            f'add_pairs={stats["add_pairs"].mean():.1f}+-{stats["add_pairs"].std():.1f}, '
            f'delete_pairs={stats["delete_pairs"].mean():.1f}+-{stats["delete_pairs"].std():.1f}, '
            f'directed_add={stats["directed_add"].mean():.1f}+-{stats["directed_add"].std():.1f}, '
            f'directed_delete={stats["directed_delete"].mean():.1f}+-{stats["directed_delete"].std():.1f}, '
            f'asymmetric_pairs={stats["asymmetric_pairs"].mean():.1f}+-'
            f'{stats["asymmetric_pairs"].std():.1f}',
            flush=True,
        )

def test(encoder_model, data, eval_seed, classifier_lr, classifier_epoch,
         eval_view):
    encoder_model.eval()
    with torch.no_grad():
        if eval_view == 'official':
            z1, z2, _, _, _, _ = encoder_model(data)
        else:
            z1, z2 = encoder_model.encode_clean(data)
    # z = z1 + z2
    z = torch.cat((z1, z2), 1)
    split = get_split(
        num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8,
        seed=eval_seed,
    )
    
    best_result = {
        'accuracy': 0,
        'micro_f1': 0,
        'macro_f1': 0,
        'accuracy_val': 0,
        'micro_f1_val': 0,
        'macro_f1_val': 0,
        'eval_weight_decay': None,
    }
    for decay in [0.0, 0.001, 0.005, 0.01, 0.1]:
        result = LREvaluator(
            num_epochs=classifier_epoch,
            learning_rate=classifier_lr,
            weight_decay=decay,
            seed=eval_seed if eval_view == 'clean' else None,
        )(z, data.y, split)
        print(
            f'(E-wd): weight_decay={decay:g}, epoch={result["best_epoch"]}, '
            f'Val accuracy={result["accuracy_val"]:.4f}, '
            f'Test accuracy={result["accuracy"]:.4f}, '
            f'F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}',
            flush=True,
        )
        if result['accuracy_val'] > best_result['accuracy_val']:
            best_result = {**result, 'eval_weight_decay': decay}
    return best_result

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=15, help='Random seed')
    parser.add_argument('--device', type=int, default=0, help='cuda')
    parser.add_argument('--dataset', type=str, default='Cora') 
    parser.add_argument('--epoch', type=int, default=200, help='GCL training epoch')
    parser.add_argument('--aug_lr1', type=float, default=100, help='augmentation learning rate for spectral max')
    parser.add_argument('--aug_lr2', type=float, default=0.1, help='augmentation learning rate for spectral min')
    parser.add_argument('--aug_iter', type=int, default=40, help='iteration for augmentation')
    parser.add_argument('--pf', type=float, default=0.4, help='feature probability')
    parser.add_argument('--pe', type=float, default=0.3, help='edge probability')
    parser.add_argument('--runs', type=int, default=10,
                        help='number of repeated experiments')
    parser.add_argument('--classifier_lr', type=float, default=0.001,
                        help='linear classifier learning rate')
    parser.add_argument('--classifier_epoch', type=int, default=5000,
                        help='linear classifier training epochs')
    parser.add_argument(
        '--eval-view', choices=('official', 'clean'), default='official',
        help='Official stochastic-view evaluation or clean original-graph evaluation.',
    )
    default_data_root = osp.abspath(
        osp.join(osp.dirname(__file__), '..', '..', '..', '..', 'data')
    )
    parser.add_argument('--data-root', type=str, default=default_data_root,
                        help='root directory for PyG datasets')
    return parser.parse_args()

def main():
    args = arg_parse()
    if args.runs < 1:
        raise ValueError('--runs must be at least 1')
    if args.classifier_lr <= 0:
        raise ValueError('--classifier_lr must be positive')
    if args.classifier_epoch < 1:
        raise ValueError('--classifier_epoch must be at least 1')
    
    seed_everything(args.seed)
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(1)
    
    # Load dataset
    path = osp.abspath(osp.expanduser(args.data_root))
    dataset = Planetoid(path, name=args.dataset, transform=T.NormalizeFeatures())
    print(
        f'(S): protocol=10% train / 10% validation / 80% test, '
        f'runs={args.runs}, classifier_lr={args.classifier_lr:g}, '
        f'classifier_epoch={args.classifier_epoch}, eval_view={args.eval_view}'
    )
    
    # Initialize augmentor
    span1 = SpectralAugmentor(ratio=args.pe, 
                             lr=args.aug_lr1,
                             iteration=args.aug_iter,
                             dis_type='max',
                             device=device,
                             threshold=0.8)
    span2 = SpectralAugmentor(ratio=args.pe, 
                             lr=args.aug_lr2,
                             iteration=args.aug_iter,
                             dis_type='min',
                             device=device,
                             threshold=0.8)
    
    # Pre-compute the perturbation probability or load the saved probability
    update_data_path = osp.join(
        path,
        args.dataset
        + '/updated_lr1_{}_lr2_{}_iter_{}_pe_{}.pt'.format(
            args.aug_lr1, args.aug_lr2, args.aug_iter, args.pe
        ),
    )
    if os.path.exists(update_data_path):  # Load precomputed probability matrix
        data = torch.load(update_data_path)
        print('(A): perturbation probability loaded!')
    else: # Precompute probability matrix and save it to file
        print('(A): precomputing probability ...')
        assert dataset.len() == 1  # should only have one data object for node classification
        data = dataset.get(0)        
        span1.calc_prob(data) # note that data is updated with data['max']=ptb_prob1
        span2.calc_prob(data) # note that data is further updated with data['min']=ptb_prob2
        # Save the updated data to reduce recomputation
        torch.save(data, update_data_path)
        
    # print(data.max)
    # print(data.min)
    # exit()
    
    # Start repeated GCL training. The paper reports mean/std over 10 runs.
    data = data.to(device)
    aug1 = Compose([span1, FeatureAugmentor(pf=args.pf)])
    aug2 = Compose([span2, FeatureAugmentor(pf=args.pf)])
    results = []
    for run in range(args.runs):
        run_seed = args.seed + run
        seed_everything(run_seed)
        gconv1 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
        gconv2 = GConv(input_dim=dataset.num_features, hidden_dim=512, num_layers=2).to(device)
        encoder_model = Encoder(
            encoder1=gconv1, encoder2=gconv2, augmentor=(aug1, aug2),
            hidden_dim=512,
        ).to(device)
        contrast_model = DualBranchContrast(loss=JSD(), mode='G2L').to(device)
        optimizer = Adam(encoder_model.parameters(), lr=0.0001)
        edit_history = {'view1_max': [], 'view2_min': []}
        with tqdm(total=args.epoch, desc=f'(T run {run + 1}/{args.runs})') as pbar:
            for _ in range(args.epoch):
                loss = train(encoder_model, contrast_model, data, optimizer)
                edit_history['view1_max'].append(span1.last_edit_counts)
                edit_history['view2_min'].append(span2.last_edit_counts)
                pbar.set_postfix({'loss': loss})
                pbar.update()

        print_edit_summary('A-train', edit_history)
        result = test(
            encoder_model,
            data,
            eval_seed=run_seed,
            classifier_lr=args.classifier_lr,
            classifier_epoch=args.classifier_epoch,
            eval_view=args.eval_view,
        )
        if args.eval_view == 'official':
            print_edit_summary(
                'A-eval',
                {
                    'view1_max': [span1.last_edit_counts],
                    'view2_min': [span2.last_edit_counts],
                },
            )
        results.append(result)
        print(
            f'(E-run): run={run + 1} seed={run_seed} '
            f'classifier_epoch={result["best_epoch"]}, '
            f'Val accuracy={result["accuracy_val"]:.4f}, '
            f'Test accuracy={result["accuracy"]:.4f}, '
            f'F1Mi={result["micro_f1"]:.4f}, F1Ma={result["macro_f1"]:.4f}, '
            f'weight_decay={result["eval_weight_decay"]:g}'
        )

    for key, label in (
        ('accuracy', 'Test accuracy'),
        ('micro_f1', 'F1Mi'),
        ('macro_f1', 'F1Ma'),
    ):
        values = np.asarray([result[key] for result in results])
        print(f'(E-final): {label}={values.mean():.4f}+-{values.std():.4f}')
    selected_decays = [result['eval_weight_decay'] for result in results]
    print(f'(E-final): Selected weight decays={selected_decays}')


if __name__ == '__main__':
    main()
