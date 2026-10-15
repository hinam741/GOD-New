import json
import os
import numpy as np
import random
import argparse
import functools
import torch
import torch.nn as nn


import math

import torch.nn.functional as F

torch.autograd.set_detect_anomaly(True)
eps = 1e-10
infty = 1e10

def get_distance(p1, p2, type, slice=1000, temperature=None):
    eps = 1e-10
    if type == "dot_exp":
        assert temperature is not None
        dist = get_distance(p1, p2, "cosine", slice)
        dot = (1- dist) / temperature
        return torch.exp(-dot)
    if len(p1.shape) > 1:
        if len(p2.shape) == 1:
            # p1 (n, dim)
            # p2 (dim)
            p2 = p2.unsqueeze(0)  # (1, dim)
            if type == "cosine":
                p1 = F.normalize(p1, p=2, dim=1)
                p2 = F.normalize(p2, p=2, dim=1)
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    dist_ = 1 - torch.sum(p1[slice*i:slice*(i+1)]*p2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            elif type == "euclidean":
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)]-p2, p=2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            elif type == "product":
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    dist_ = torch.sum(p1[slice*i:slice*(i+1)]*p2, dim=1)  # (slice, )
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
            else:
                assert type == "KLDiv"
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    p1_ = p1[slice*i:slice*(i+1)]
                    dist1 = p1_ * torch.log(p1_ / (p2 + eps) + eps)  # (slice, dim)
                    dist2 = p2 * torch.log(p2 / (p1_ + eps) + eps)  # (slice, dim)
                    dist_ = dist1 + dist2
                    dist_ = torch.sum(dist_, dim=1)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)  # (n, dim)

        else:
            # p1 (n, dim)
            # p2 (m, dim)
            if type == "cosine":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                # p1 = F.normalize(p1, p=2, dim=2)
                p2 = F.normalize(p2, p=2, dim=2)
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    p1[slice * i:slice * (i + 1)] = F.normalize(p1[slice*i:slice*(i+1)], p=2, dim=2)
                    dist_ = 1 - torch.sum(p1[slice*i:slice*(i+1)] * p2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            elif type == "euclidean":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    dist_ = torch.norm(p1[slice*i:slice*(i+1)] - p2, p=2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            elif type == "product":
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    print(i)
                    dist_ = torch.sum(p1[slice*i:slice*(i+1)] * p2, dim=2)  # (slice, m)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)

            else:
                assert type == "KLDiv"
                p1 = p1.unsqueeze(1)  # (n, 1, dim)
                p2 = p2.unsqueeze(0)  # (1, m, dim)
                dist = []
                iter = math.ceil(1.0 * p1.shape[0] / slice)
                for i in range(iter):
                    p1_ = p1[slice*i:slice*(i+1)]
                    dist1 = p1_ * torch.log(p1_ / (p2 + eps) + eps)  # (slice, m, dim)
                    dist2 = p2 * torch.log(p2 / (p1_ + eps) + eps)  # (slice, m, dim)
                    dist_ = dist1 + dist2
                    dist_ = torch.sum(dist_, dim=2)
                    dist.append(dist_)
                dist = torch.cat(dist, dim=0)
    else:
        # p1 (dim, )
        # p2 (dim, )
        if type == "cosine":
            dist = 1 - torch.sum(p1 * p2)
        elif type == "euclidean":
            dist = torch.norm(p1 - p2, p=2)
        elif type == "product":
            dist = torch.sum(p1 * p2)
        else:
            assert type == "KLDiv"
            dist1 = p1 * torch.log(p1 / (p2 + eps) + eps)
            dist2 = p2 * torch.log(p2 / (p1 + eps) + eps)
            dist = dist1 + dist2
            dist = torch.sum(dist, dim=0)  # (n, )

    return dist

def update_distance(distances, all_features, cfeature, dist_func):
    new_dist = dist_func(all_features, cfeature)
    distances = torch.where(distances < new_dist, distances, new_dist)
    return distances

def farthest_distance_sample(all_features, sample_num, dist_func, init_ids=[]):
    if len(init_ids) >= sample_num:
        print("Initial samples are enough")
        return init_ids

    if all_features.shape[0] <= sample_num:
        print("No enough features")
        return list(range(all_features.shape[0]))

    total_num = all_features.shape[0]
    if len(init_ids) == 0:
        sample_ids = random.sample(range(total_num), 1)
    else:
        sample_ids = init_ids

    distances = torch.zeros(total_num).cuda() + 1e20
    print(torch.max(distances, dim=0)[0])

    for i, init_id in enumerate(sample_ids):
        distances = update_distance(distances, all_features, all_features[init_id], dist_func)
        print(i, torch.max(distances, dim=0)[0])

    while len(sample_ids) < sample_num:
        if len(sample_ids) % 100 == 1:
            print(len(sample_ids))
        new_id = torch.max(distances, dim=0)[1]
        print(len(sample_ids), torch.max(distances, dim=0)[0])
        distances = update_distance(distances, all_features, all_features[new_id], dist_func)
        sample_ids.append(new_id.item())

    # assert len(set(sample_ids)) == sample_num
    return sample_ids


class SampleModel(nn.Module):
    def __init__(self, features, sample_num, temperature, init, distance, balance=1.0):
        super(SampleModel, self).__init__()
        self.features = features
        self.total_num = features.shape[0]
        self.temperature = temperature
        self.sample_num = sample_num
        self.balance = balance

        self.init = init
        self.distance = distance

        centroids = self.init_centroids()
        self.centroids = nn.Parameter(centroids)
        print(self.centroids.shape)

    def init_centroids(self):
        if self.init == "random":
            sample_ids = list(range(self.total_num))
            sample_ids = random.sample(sample_ids, self.sample_num)
        elif self.init == "fps":
            dist_func = functools.partial(get_distance, type=self.distance)
            sample_ids = farthest_distance_sample(self.features, self.sample_num, dist_func)

        centroids = self.features[sample_ids].clone()
        return centroids

    def get_loss(self):
        centroids = F.normalize(self.centroids, dim=1)
        prod = torch.matmul(self.features, centroids.transpose(1, 0))  # (n, k)
        prod = prod / self.temperature
        prod_exp = torch.exp(prod)
        prod_exp_pos, pos_k = torch.max(prod_exp, dim=1)  # (n, )

        cent_prod = torch.matmul(centroids.detach(), centroids.transpose(1, 0))  # (k, k)
        cent_prod = cent_prod / self.temperature
        cent_prod_exp = torch.exp(cent_prod)
        cent_prob_exp_sum = torch.sum(cent_prod_exp, dim=0)  # (k, )

        J = torch.log(prod_exp_pos) - torch.log(prod_exp_pos + cent_prob_exp_sum[pos_k] * self.balance)
        J = -torch.mean(J)

        return J

