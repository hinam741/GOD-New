
import math
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.layers.weight_init import trunc_normal_
import numpy as np
from scipy.optimize import linear_sum_assignment


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.40, easy_margin=False, bias=False):

        super(ArcFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)

        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)

    def forward(self, input, label=None):
        input_norm = F.normalize(input)
        weight_norm = F.normalize(self.weight)
        cosine = F.linear(input_norm, weight_norm)
        if label is not None:
            sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
            phi = cosine * self.cos_m - sine * self.sin_m
            if self.easy_margin:
                phi = torch.where(cosine > 0, phi, cosine)
            else:
                phi = torch.where(cosine > self.th, phi, cosine - self.mm)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, label.view(-1, 1), 1)
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
            output = output * self.s
        else:
            output = cosine * self.s

        return {'logits': output}

    def loss(self, logits, labels):
        return F.cross_entropy(logits, labels)
class Multi_lora_etf(nn.Module):
    def __init__(self, num_classes: int, in_channels: int, Hiddensize, increment) -> None:
        super(Multi_lora_etf, self).__init__()
        assert num_classes > 0, f'num_classes={num_classes} must be a positive integer'
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.hidden = Hiddensize
        self.hidden_layer = nn.Linear(self.in_channels, self.hidden, bias=False)
        nn.init.xavier_uniform_(self.hidden_layer.weight)
        for param in self.hidden_layer.parameters():
            param.requires_grad = False
        self.classes_per_task = increment
        self.mapping_layer = nn.ModuleList()
        orth_vec = self.generate_random_orthogonal_matrix(self.in_channels, self.num_classes)
        i_nc_nc = torch.eye(self.num_classes)
        one_nc_nc: torch.Tensor = torch.mul(torch.ones(self.num_classes, self.num_classes), (1 / self.num_classes))
        etf_vec = torch.mul(torch.matmul(orth_vec, i_nc_nc - one_nc_nc),
                            math.sqrt(self.num_classes / (self.num_classes - 1)))
        self.register_buffer('etf_vec', etf_vec)  # 768 *200
        etf_rect = torch.ones((1, num_classes), dtype=torch.float32)  # 1 * N
        self.etf_rect = etf_rect

    def add_task_layer(self):
        new_layer = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.hidden, self.in_channels),
        )
        new_layer.apply(init_weights)
        self.mapping_layer.append(new_layer)
        print(f"Adding a new mapping layer for task_id: {len(self.mapping_layer) - 1}")

    def generate_random_orthogonal_matrix(self, feat_in, num_classes):
        rand_mat = np.random.random(size=(feat_in, num_classes))  # 768*20
        orth_vec, _ = np.linalg.qr(rand_mat)
        orth_vec = torch.tensor(orth_vec).float()
        assert torch.allclose(torch.matmul(orth_vec.T, orth_vec), torch.eye(num_classes), atol=1.e-7), \
            "The max irregular value is : {}".format(
                torch.max(torch.abs(torch.matmul(orth_vec.T, orth_vec) - torch.eye(num_classes))))
        return orth_vec

    def pre_logits(self, x):
        x = x / torch.norm(x, p=2, dim=1, keepdim=True)
        return x

    def forward(self, x, layer_id):
        x = self.hidden_layer(x)
        processed_x = self.mapping_layer[layer_id](x)
        normalized_x = self.pre_logits(processed_x)
        etf_vec = F.normalize(self.etf_vec.T, p=2, dim=1)
        sliced_logits = F.linear(normalized_x, etf_vec) / 0.1
        return sliced_logits