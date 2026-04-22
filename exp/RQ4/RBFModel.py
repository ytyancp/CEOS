import numpy as np
import torch
from torch import nn


class RBFKernel(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, x1, x2):
        '''

        :param x1: x (N x D)
        :param x2: xis (SVs x D)
        :return: (N x SVs)
        '''
        dist = torch.sum((x1[:, None] - x2) ** 2, dim=2)
        return torch.exp(-self.gamma * dist)


class RBFKernelNumpy:
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, xis, x):
        return np.exp(-self.gamma * np.sum((xis - x) ** 2, axis=1))


class RBFSVMModel:
    def __init__(self, alphas, xis, yis, b, gamma):
        self.alphas = alphas
        self.xis = xis
        self.yis = yis
        self.b = b
        self.gamma = gamma
        self.kernel = RBFKernelNumpy(gamma)

    def decision_function(self, x):
        return np.sum(self.alphas * self.kernel(self.xis, x)) + self.b


class DifferentiableRBFSVMModel(nn.Module):
    def __init__(self, alphas, support_vectors, yis, intercept, gamma):
        super(DifferentiableRBFSVMModel, self).__init__()
        self.alphas = torch.tensor(alphas, dtype=torch.float32)
        self.xis = torch.tensor(support_vectors, dtype=torch.float32)
        self.yis = torch.tensor(yis, dtype=torch.float32)
        self.intercept = torch.tensor(intercept, dtype=torch.float32)
        self.gamma = gamma
        self.kernel = RBFKernel(gamma)

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.alphas.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)

        pairwise_dist = torch.cdist(x, self.xis, p=2)
        kernel_vals = torch.exp(-self.gamma * pairwise_dist.pow(2))
        decision = torch.matmul(kernel_vals, (self.alphas * self.yis)) + self.intercept

        prob_positive = torch.sigmoid(decision).unsqueeze(1)
        prob_negative = 1 - prob_positive
        return torch.cat([prob_negative, prob_positive], dim=1)
