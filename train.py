import os
import os.path as osp
import math
import tqdm

import numpy as np
import torch
import gc

import warnings
warnings.simplefilter('ignore')

import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T

from torch.utils.checkpoint import checkpoint
from torch_cluster import knn_graph

from torch_geometric.nn import EdgeConv, NNConv
from torch_geometric.nn.pool.edge_pool import EdgePooling

from torch_geometric.utils import normalized_cut
from torch_geometric.utils import remove_self_loops
from torch_geometric.utils.undirected import to_undirected
from torch_geometric.nn import (graclus, max_pool, max_pool_x,
                                global_mean_pool, global_max_pool,
                                global_add_pool)

from torch_geometric.datasets import MNISTSuperpixels
from torch_geometric.data import DataLoader

transform = T.Cartesian(cat=False)

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import math
import torch
import sys


class ReduceMaxLROnRestart:
    def __init__(self, ratio=0.75):
        self.ratio = ratio
        
        def __call__(self, eta_min, eta_max):
            return eta_min, eta_max * self.ratio
        
        
class ExpReduceMaxLROnIteration:
    def __init__(self, gamma=1):
        self.gamma = gamma
        
    def __call__(self, eta_min, eta_max, iterations):
        return eta_min, eta_max * self.gamma ** iterations


class CosinePolicy:
    def __call__(self, t_cur, restart_period):
        return 0.5 * (1. + math.cos(math.pi *
                                    (t_cur / restart_period)))
    
    
class ArccosinePolicy:
    def __call__(self, t_cur, restart_period):
        return (math.acos(max(-1, min(1, 2 * t_cur
                                      / restart_period - 1))) / math.pi)
    
    
class TriangularPolicy:
    def __init__(self, triangular_step=0.5):
        self.triangular_step = triangular_step
        
    def __call__(self, t_cur, restart_period):
        inflection_point = self.triangular_step * restart_period
        point_of_triangle = (t_cur / inflection_point
                             if t_cur < inflection_point
                             else 1.0 - (t_cur - inflection_point)
                             / (restart_period - inflection_point))
        return point_of_triangle
    
    
class CyclicLRWithRestarts(_LRScheduler):
    """Decays learning rate with cosine annealing, normalizes weight decay
    hyperparameter value, implements restarts.
    https://arxiv.org/abs/1711.05101
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        batch_size: minibatch size
        epoch_size: training samples per epoch
        restart_period: epoch count in the first restart period
        t_mult: multiplication factor by which the next restart period will expand/shrink
        policy: ["cosine", "arccosine", "triangular", "triangular2", "exp_range"]
        min_lr: minimum allowed learning rate
        verbose: print a message on every restart
        gamma: exponent used in "exp_range" policy
        eta_on_restart_cb: callback executed on every restart, adjusts max or min lr
        eta_on_iteration_cb: callback executed on every iteration, adjusts max or min lr
        triangular_step: adjusts ratio of increasing/decreasing phases for triangular policy
    Example:
        >>> scheduler = CyclicLRWithRestarts(optimizer, 32, 1024, restart_period=5, t_mult=1.2)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>         ...
        >>>         optimizer.zero_grad()
        >>>         loss.backward()
        >>>         optimizer.step()
        >>>         scheduler.batch_step()
        >>>     validate(...)
    """
    
    def __init__(self, optimizer, batch_size, epoch_size, restart_period=100,
                 t_mult=2, last_epoch=-1, verbose=False,
                 policy="cosine", policy_fn=None, min_lr=1e-7,
                 eta_on_restart_cb=None, eta_on_iteration_cb=None,
                 gamma=1.0, triangular_step=0.5):
        
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(
                type(optimizer).__name__))
        
        self.optimizer = optimizer
        
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
                group.setdefault('minimum_lr', min_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an"
                                   " optimizer".format(i))
                
        self.base_lrs = [group['initial_lr'] for group
                         in optimizer.param_groups]
        
        self.min_lrs = [group['minimum_lr'] for group
                        in optimizer.param_groups]
        
        self.base_weight_decays = [group['weight_decay'] for group
                                   in optimizer.param_groups]
        
        self.policy = policy
        self.eta_on_restart_cb = eta_on_restart_cb
        self.eta_on_iteration_cb = eta_on_iteration_cb
        if policy_fn is not None:
            self.policy_fn = policy_fn
        elif self.policy == "cosine":
            self.policy_fn = CosinePolicy()
        elif self.policy == "arccosine":
            self.policy_fn = ArccosinePolicy()
        elif self.policy == "triangular":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
        elif self.policy == "triangular2":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_restart_cb = ReduceMaxLROnRestart(ratio=0.5)
        elif self.policy == "exp_range":
            self.policy_fn = TriangularPolicy(triangular_step=triangular_step)
            self.eta_on_iteration_cb = ExpReduceMaxLROnIteration(gamma=gamma)
            
        self.last_epoch = last_epoch
        self.batch_size = batch_size
        self.epoch_size = epoch_size
        
        self.iteration = 0
        self.total_iterations = 0
        
        self.t_mult = t_mult
        self.verbose = verbose
        self.restart_period = math.ceil(restart_period)
        self.restarts = 0
        self.t_epoch = -1
        self.epoch = -1
        
        self.eta_min = 0
        self.eta_max = 1
        
        self.end_of_period = False
        self.batch_increments = []
        self._set_batch_increment()
        
    def _on_restart(self):
        if self.eta_on_restart_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_restart_cb(self.eta_min,
                                                                self.eta_max)
            
    def _on_iteration(self):
        if self.eta_on_iteration_cb is not None:
            self.eta_min, self.eta_max = self.eta_on_iteration_cb(self.eta_min,
                                                                  self.eta_max,
                                                                  self.total_iterations)
            
    def get_lr(self, t_cur):
        eta_t = (self.eta_min + (self.eta_max - self.eta_min)
                 * self.policy_fn(t_cur, self.restart_period))
        
        weight_decay_norm_multi = math.sqrt(self.batch_size /
                                            (self.epoch_size *
                                             self.restart_period))
        
        lrs = [min_lr + (base_lr - min_lr) * eta_t for base_lr, min_lr
               in zip(self.base_lrs, self.min_lrs)]
        weight_decays = [base_weight_decay #* eta_t * weight_decay_norm_multi
                         for base_weight_decay in self.base_weight_decays]
        
        if (self.t_epoch + 1) % self.restart_period < self.t_epoch:
            self.end_of_period = True
            
        if self.t_epoch % self.restart_period < self.t_epoch:
            if self.verbose:
                print("Restart {} at epoch {}".format(self.restarts + 1,
                                                      self.last_epoch))
            self.restart_period = math.ceil(self.restart_period * self.t_mult)
            self.restarts += 1
            self.t_epoch = 0
            self._on_restart()
            self.end_of_period = False
            
        return zip(lrs, weight_decays)
        
    def _set_batch_increment(self):
        d, r = divmod(self.epoch_size, self.batch_size)
        batches_in_epoch = d + 2 if r > 0 else d + 1
        self.iteration = 0
        self.batch_increments = torch.linspace(0, 1, batches_in_epoch).tolist()
        
    def step(self):
        self.last_epoch += 1
        self.t_epoch += 1
        self._set_batch_increment()
        self.batch_step()
        
    def batch_step(self):
        try:
            t_cur = self.t_epoch + self.batch_increments[self.iteration]
            self._on_iteration()
            self.iteration += 1
            self.total_iterations += 1
        except (IndexError):
            raise StopIteration("Epoch size and batch size used in the "
                                "training loop and while initializing "
                                "scheduler should be the same.")
        
        for param_group, (lr, weight_decay) in zip(self.optimizer.param_groups,
                                                   self.get_lr(t_cur)):
            param_group['lr'] = lr
            param_group['weight_decay'] = weight_decay


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class DynamicReductionNetwork(nn.Module):
    # This model clusters nearest neighbour graphs
    # in two steps.
    # The latent space trained to group useful features at each level
    # of aggregration.
    # This allows single quantities to be regressed from complex point counts
    # in a location and orientation invariant way.
    # One encoding layer is used to abstract away the input features.
    def __init__(self, input_dim=4, hidden_dim=64, output_dim=2, k=16, aggr='add',
                 norm=torch.tensor([1./1000., 1./10., 1./3.15, 1/3000.])):
        super(DynamicReductionNetwork, self).__init__()

        self.datanorm = nn.Parameter(norm)

        self.k = k
        start_width = 2 * hidden_dim
        middle_width = 3 * hidden_dim // 2

        self.inputnet =  nn.Sequential(
            nn.Linear(input_dim, hidden_dim//2),
            nn.ELU(),
            nn.Linear(hidden_dim//2, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU()
        )

        convnn1 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),
                                nn.ELU(),
                                )
        convnn2 = nn.Sequential(nn.Linear(start_width, middle_width),
                                nn.ELU(),
                                nn.Linear(middle_width, hidden_dim),
                                nn.ELU(),
                                )
        self.edgeconv1 = EdgeConv(nn=convnn1, aggr=aggr)
        self.edgeconv2 = EdgeConv(nn=convnn2, aggr=aggr)
        
        self.output = nn.Sequential(nn.Linear(hidden_dim, hidden_dim),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim, hidden_dim//2),
                                    nn.ELU(),
                                    nn.Linear(hidden_dim//2, output_dim))

    def forward(self, data):
        data.x = self.datanorm * data.x
        data.x = self.inputnet(data.x)

        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv1.flow))
        data.x = self.edgeconv1(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data)

        data.edge_index = to_undirected(knn_graph(data.x, self.k, data.batch, loop=False, flow=self.edgeconv2.flow))
        data.x = self.edgeconv2(data.x, data.edge_index)

        weight = normalized_cut_2d(data.edge_index, data.x)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        x, batch = max_pool_x(cluster, data.x, data.batch)

        x = global_max_pool(x, batch)

        logits = self.output(x).squeeze(-1)
        return logits

        # print(logits)
        # return F.log_softmax(logits, dim=1)


def print_model_summary(model):
    """Override as needed"""
    print(
        'Model: \n%s\nParameters: %i' %
        (model, sum(p.numel() for p in model.parameters()))
    )

def main():
    from data import ZPrimeDataset
    dataset = ZPrimeDataset('data')

    batch_size = 24
    n_validation = math.ceil(len(dataset)*0.2)
    n_train = len(dataset) - n_validation

    # batch_size = 4
    # n_train = 50
    # n_validation = 20
    
    train_batch_size = batch_size
    test_batch_size = batch_size
    train_dataset = torch.utils.data.Subset(dataset, list(range(n_train)))
    test_dataset = torch.utils.data.Subset(dataset, list(range(n_train, n_train+n_validation)))
    train_dataset.indices = list(train_dataset.indices)
    test_dataset.indices = list(test_dataset.indices)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=True)

    epoch_size = len(train_dataset)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DynamicReductionNetwork().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = CyclicLRWithRestarts(optimizer, batch_size, epoch_size, restart_period=400, t_mult=1.2, policy="cosine")

    print_model_summary(model)

    def train(epoch):
        model.train()
        scheduler.step()

        print('Looping over train data')

        for data in tqdm.tqdm(train_loader, total=len(train_loader)):
            data = data.to(device)
            optimizer.zero_grad()
            result = model(data)

            # print('y =', data.y)
            # print('result =', result)
            # print('Sizes of y and result:', data.y.size(), result.size())

            loss = F.nll_loss(result, data.y.type(torch.LongTensor))
            loss.backward()

            #print(torch.unique(torch.argmax(result, dim=-1)))
            #print(torch.unique(data.y))

            optimizer.step()
            scheduler.batch_step()

    def test():
        model.eval()
        correct = 0        
        for data in test_loader:
            data = data.to(device)
            result = model(data)
            probabilities = torch.nn.functional.softmax(result, dim=1)
            pred = probabilities.argmax()

            # print('probabilities=', probabilities)
            # print('pred=', pred)
            # print('data.y=', data.y)

            correct += pred.eq(data.y).sum().item()
        return correct / len(test_dataset)


    best_test_acc = 0.0
    for epoch in range(1, 401):
        train(epoch)
        test_acc = test()
        # write_checkpoint(epoch)
        print('Epoch: {:02d}, Test: {:.4f}'.format(epoch, test_acc))
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            # write_checkpoint(epoch, best=True)

if __name__ == '__main__':
    main()

