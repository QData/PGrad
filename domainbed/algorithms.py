# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import random
import copy
import numpy as np
from collections import defaultdict, OrderedDict
try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except:
    backpack = None

from domainbed import networks
from domainbed.lib.misc import (
    random_pairs_of_minibatches, ParamDict, MovingAverage, l2_between_dicts
)


ALGORITHMS = [
    'Fish', 
    'Fishr',
    'PGrad',
    'PGradParallel'
]

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]

class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain generalization algorithm.
    Subclasses should implement the following:
    - update()
    - predict()
    """
    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Algorithm, self).__init__()
        self.hparams = hparams

    def update(self, minibatches, unlabeled=None):
        """
        Perform one update step, given a list of (x, y) tuples for all
        environments.

        Admits an optional list of unlabeled minibatches from the test domains,
        when task is domain_adaptation.
        """
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError



class Fish(Algorithm):
    """
    Implementation of Fish, as seen in Gradient Matching for Domain 
    Generalization, Shi et al. 2021.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(Fish, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes

        self.network = networks.WholeFish(input_shape, num_classes, hparams)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams["lr"],
            weight_decay=self.hparams['weight_decay']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def fish(self, meta_weights, inner_weights, lr_meta):
        meta_weights = ParamDict(meta_weights)
        inner_weights = ParamDict(inner_weights)
        meta_weights += lr_meta * (inner_weights - meta_weights)
        return meta_weights

    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)

        for x, y in minibatches:
            loss = F.cross_entropy(self.network_inner(x), y)
            self.optimizer_inner.zero_grad()
            loss.backward()
            self.optimizer_inner.step()

        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        meta_weights = self.fish(
            meta_weights=self.network.state_dict(),
            inner_weights=self.network_inner.state_dict(),
            lr_meta=self.hparams["meta_lr"]
        )
        self.network.reset_weights(meta_weights)

        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



class Fishr(Algorithm):
    "Invariant Gradients variances for Out-of-distribution Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        assert backpack is not None, "Install backpack with: 'pip install backpack-for-pytorch==1.3.0'"
        super(Fishr, self).__init__(input_shape, num_classes, num_domains, hparams)
        self.num_domains = num_domains

        self.featurizer = networks.Featurizer(input_shape, self.hparams)
        self.classifier = extend(
            networks.Classifier(
                self.featurizer.n_outputs,
                num_classes,
                self.hparams['nonlinear_classifier'],
            )
        )
        self.network = nn.Sequential(self.featurizer, self.classifier)

        self.register_buffer("update_count", torch.tensor([0]))
        self.bce_extended = extend(nn.CrossEntropyLoss(reduction='none'))
        self.ema_per_domain = [
            MovingAverage(ema=self.hparams["ema"], oneminusema_correction=True)
            for _ in range(self.num_domains)
        ]
        self._init_optimizer()

    def _init_optimizer(self):
        self.optimizer = torch.optim.Adam(
            list(self.featurizer.parameters()) + list(self.classifier.parameters()),
            lr=self.hparams["lr"],
            weight_decay=self.hparams["weight_decay"],
        )

    def update(self, minibatches, unlabeled=False):
        assert len(minibatches) == self.num_domains
        all_x = torch.cat([x for x, y in minibatches])
        all_y = torch.cat([y for x, y in minibatches])
        len_minibatches = [x.shape[0] for x, y in minibatches]

        all_z = self.featurizer(all_x)
        all_logits = self.classifier(all_z)

        penalty = self.compute_fishr_penalty(all_logits, all_y, len_minibatches)
        all_nll = F.cross_entropy(all_logits, all_y)

        penalty_weight = 0
        if self.update_count >= self.hparams["penalty_anneal_iters"]:
            penalty_weight = self.hparams["lambda"]
            if self.update_count == self.hparams["penalty_anneal_iters"] != 0:
                # Reset Adam as in IRM or V-REx, because it may not like the sharp jump in
                # gradient magnitudes that happens at this step.
                self._init_optimizer()
        self.update_count += 1

        objective = all_nll + penalty_weight * penalty
        self.optimizer.zero_grad()
        objective.backward()
        self.optimizer.step()

        return {'loss': objective.item(), 'nll': all_nll.item(), 'penalty': penalty.item()}

    def compute_fishr_penalty(self, all_logits, all_y, len_minibatches):
        dict_grads = self._get_grads(all_logits, all_y)
        grads_var_per_domain = self._get_grads_var_per_domain(dict_grads, len_minibatches)
        return self._compute_distance_grads_var(grads_var_per_domain)

    def _get_grads(self, logits, y):
        self.optimizer.zero_grad()
        loss = self.bce_extended(logits, y).sum()
        with backpack(BatchGrad()):
            loss.backward(
                inputs=list(self.classifier.parameters()), retain_graph=True, create_graph=True
            )

        # compute individual grads for all samples across all domains simultaneously
        dict_grads = OrderedDict(
            [
                (name, weights.grad_batch.clone().view(weights.grad_batch.size(0), -1))
                for name, weights in self.classifier.named_parameters()
            ]
        )
        return dict_grads

    def _get_grads_var_per_domain(self, dict_grads, len_minibatches):
        # grads var per domain
        grads_var_per_domain = [{} for _ in range(self.num_domains)]
        for name, _grads in dict_grads.items():
            all_idx = 0
            for domain_id, bsize in enumerate(len_minibatches):
                env_grads = _grads[all_idx:all_idx + bsize]
                all_idx += bsize
                env_mean = env_grads.mean(dim=0, keepdim=True)
                env_grads_centered = env_grads - env_mean
                grads_var_per_domain[domain_id][name] = (env_grads_centered).pow(2).mean(dim=0)

        # moving average
        for domain_id in range(self.num_domains):
            grads_var_per_domain[domain_id] = self.ema_per_domain[domain_id].update(
                grads_var_per_domain[domain_id]
            )

        return grads_var_per_domain

    def _compute_distance_grads_var(self, grads_var_per_domain):

        # compute gradient variances averaged across domains
        grads_var = OrderedDict(
            [
                (
                    name,
                    torch.stack(
                        [
                            grads_var_per_domain[domain_id][name]
                            for domain_id in range(self.num_domains)
                        ],
                        dim=0
                    ).mean(dim=0)
                )
                for name in grads_var_per_domain[0].keys()
            ]
        )

        penalty = 0
        for domain_id in range(self.num_domains):
            penalty += l2_between_dicts(grads_var_per_domain[domain_id], grads_var)
        return penalty / self.num_domains

    def predict(self, x):
        return self.network(x)



class PGrad(Fish):
    " Learning Principal Gradients for Domain Generalization"

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PGrad, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.network = networks.WholeFish(self.input_shape, self.num_classes, self.hparams)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.1,
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None
        self.split_num = 3
        self.global_update = 0

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams['lr']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def transpose_pca(self, stack_classifier, comb=True):
        stack_classifier = torch.cat(stack_classifier)
        mean_classifier = stack_classifier.mean(dim=0)
        centerized_classifier = stack_classifier - mean_classifier
        cov_classifier = centerized_classifier@centerized_classifier.T/(stack_classifier.size(1)-1)
        pr_direction, pr_value, _ = torch.svd(cov_classifier)
        pr_direction = centerized_classifier.T@pr_direction
        pr_direction = pr_direction/pr_direction.norm(dim=0)
        return pr_direction, pr_value

    def stack_and_pca(self, envs_weights):
        new_stack = [[] for _ in range(self.split_num*self.num_domains+1)]
        for i in range(self.split_num*self.num_domains+1):
            new_stack[i] = [envs_weights[ele]['value'][i].view(1, -1) for ele in envs_weights.keys()]
            new_stack[i] = torch.cat(new_stack[i], dim=1)
        pr_direction, pr_value = self.transpose_pca(new_stack)
        return pr_direction, pr_value

        
    
    def principal_grad(self, meta_weights, inner_weights, params_stack):
    
        if True:
            meta_weights = ParamDict(meta_weights)
            inner_weights = ParamDict(inner_weights)
            diff_weights = meta_weights - inner_weights
            norm_diff = sum([ele.pow(2).sum() for ele in diff_weights.values()])
            norm_diff = norm_diff.sqrt()
            principle_dir, pr_value = self.stack_and_pca(params_stack)
            principle_dir *= norm_diff # length calibration
            grad_mask = torch.zeros_like(pr_value)
            start_index = 0
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index+param_size
                pra_grad = principle_dir[start_index:end_index, :]
                cali_direction = diff_weights[name] 
                cali_mask = (cali_direction.flatten().unsqueeze(1)*pra_grad).sum(0)
                grad_mask += cali_mask
                start_index= end_index
            
            cali_mask = 2*(grad_mask > 0).float()-1
            pra_grad = cali_mask*principle_dir # direction calibration
            comb_num = 4
            comb_coef = pr_value[:comb_num]/pr_value[:comb_num].norm()
            pra_grad =  (comb_coef*pra_grad[:, :comb_num]).sum(1) # direction ensemble
            start_index = 0
            # Learning PGrad for model update
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index+param_size
                value_grad = pra_grad[start_index:end_index].view(value.size())
                start_index= end_index
                value.grad = value_grad.clone()


    def update(self, minibatches, unlabeled=None):

        self.create_clone(minibatches[0][0].device)
        params_stack = {key: {'value':[]} for key, _ in self.network.named_parameters()}
        range_list = list(range(0, len(minibatches)))
        random.shuffle(range_list)
        for i in range(self.split_num):
            "Trajectory Sampling"
            for num, index in enumerate(range_list):
                x, y = minibatches[index]
                x = x[(i*x.size(0)//self.split_num):((i+1)*x.size(0)//self.split_num)]
                y = y[i*y.size(0)//self.split_num:(i+1)*y.size(0)//self.split_num]
                loss = F.cross_entropy(self.network_inner(x), y)
                self.optimizer_inner.zero_grad()
                loss.backward()
                for (key, _), env_value in zip(params_stack.items(), self.network_inner.parameters()):
                    params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(dim=0))
                self.optimizer_inner.step()
            random.shuffle(range_list)
        for (key, _), env_value in zip(params_stack.items(), self.network_inner.parameters()):
                params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(0))

            
        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        self.optimizer.zero_grad()

        "PGrad Learning"
        self.principal_grad(self.network.named_parameters(), self.network_inner.named_parameters(), params_stack)
        self.optimizer.step()
        self.global_update += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)




class PGradParallel(Fish):
    """
    Parallel training will suppress robust directions and maximize domain-specific noise.
    """

    def __init__(self, input_shape, num_classes, num_domains, hparams):
        super(PGradParallel, self).__init__(input_shape, num_classes, num_domains,
                                   hparams)
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_domains = num_domains
        self.network = networks.WholeFish(self.input_shape, self.num_classes, self.hparams)
        self.optimizer = torch.optim.SGD(
            self.network.parameters(),
            lr=0.1,
            weight_decay=self.hparams['weight_decay']
        )
        self.optimizer_inner_state = None
        self.split_num = 3
        self.global_update = 0

    def create_clone(self, device):
        self.network_inner = networks.WholeFish(self.input_shape, self.num_classes, self.hparams,
                                            weights=self.network.state_dict()).to(device)
        self.optimizer_inner = torch.optim.Adam(
            self.network_inner.parameters(),
            lr=self.hparams['lr']
        )
        if self.optimizer_inner_state is not None:
            self.optimizer_inner.load_state_dict(self.optimizer_inner_state)

    def transpose_pca(self, stack_classifier, comb=True):
        stack_classifier = torch.cat(stack_classifier)
        mean_classifier = stack_classifier.mean(dim=0)
        centerized_classifier = stack_classifier - mean_classifier
        cov_classifier = centerized_classifier@centerized_classifier.T/(stack_classifier.size(1)-1)
        pr_direction, pr_value, _ = torch.svd(cov_classifier)
        pr_direction = centerized_classifier.T@pr_direction
        pr_direction = pr_direction/pr_direction.norm(dim=0)
        return pr_direction, pr_value

    def stack_and_pca(self, envs_weights):
        new_stack = [[] for _ in range(self.split_num*self.num_domains+1)]
        for i in range(self.split_num*self.num_domains+1):
            new_stack[i] = [envs_weights[ele]['value'][i].view(1, -1) for ele in envs_weights.keys()]
            new_stack[i] = torch.cat(new_stack[i], dim=1)
        pr_direction, pr_value = self.transpose_pca(new_stack)
        return pr_direction, pr_value

        
    
    def principal_grad(self, meta_weights, inner_weights, params_stack):
    
        if True:
            meta_weights = ParamDict(meta_weights)
            inner_weights = ParamDict(inner_weights)
            diff_weights = meta_weights - inner_weights
            norm_diff = sum([ele.pow(2).sum() for ele in diff_weights.values()])
            norm_diff = norm_diff.sqrt()
            principle_dir, pr_value = self.stack_and_pca(params_stack)
            principle_dir *= norm_diff
            grad_mask = torch.zeros_like(pr_value)
            start_index = 0
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index+param_size
                pra_grad = principle_dir[start_index:end_index, :]
                cali_direction = diff_weights[name] ## error here
                cali_mask = (cali_direction.flatten().unsqueeze(1)*pra_grad).sum(0)
                grad_mask += cali_mask
                start_index= end_index
            
            cali_mask = 2*(grad_mask > 0).float()-1
            pra_grad = cali_mask*principle_dir
            comb_num = 4
            comb_coef = pr_value[:comb_num]/pr_value[:comb_num].norm()
            pra_grad =  (comb_coef*pra_grad[:, :comb_num]).sum(1)
            start_index = 0
            
            for name, value in self.network.named_parameters():
                param_size = value.numel()
                end_index = start_index+param_size
                value_grad = pra_grad[start_index:end_index].view(value.size())
                start_index= end_index
                value.grad = value_grad.clone()



    def update(self, minibatches, unlabeled=None):
        self.create_clone(minibatches[0][0].device)
        params_stack = {key: {'value':[]} for key, _ in self.network.named_parameters()}
        range_list = list(range(0, len(minibatches)))
        random.shuffle(range_list)
        for i in range(self.split_num):
            for num, index in enumerate(range_list):
                self.network_inner_copy = copy.deepcopy(self.network)
                self.optimizer_inner_copy = torch.optim.Adam(
                    self.network_inner_copy.parameters(),
                    lr=self.hparams['lr']
                )
                x, y = minibatches[index]
                x = x[(i*x.size(0)//self.split_num):((i+1)*x.size(0)//self.split_num)]
                y = y[i*y.size(0)//self.split_num:(i+1)*y.size(0)//self.split_num]
                loss = F.cross_entropy(self.network_inner_copy(x), y)
                self.optimizer_inner_copy.zero_grad()
                loss.backward()
                for (key, _), env_value in zip(params_stack.items(), self.network_inner_copy.parameters()):
                    params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(dim=0))
                self.optimizer_inner_copy.step()
                
            random.shuffle(range_list)
        for (key, _), env_value in zip(params_stack.items(), self.network_inner_copy.parameters()):
                params_stack[key]['value'].append(copy.deepcopy(env_value).unsqueeze(0))

            
        self.optimizer_inner_state = self.optimizer_inner.state_dict()
        self.optimizer.zero_grad()
        self.principal_grad(self.network.named_parameters(), self.network_inner_copy.named_parameters(), params_stack)
        self.optimizer.step()
        self.global_update += 1
        return {'loss': loss.item()}

    def predict(self, x):
        return self.network(x)



