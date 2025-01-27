import math, random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from cfg import get_cfg
cfg = get_cfg()
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs)
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=cfg.sigma_init):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x, training = True):

        weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
        bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        return F.linear(x, weight, bias)


    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def _zeros_noise(self, size):
        x = torch.zeros(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

    def remove_noise(self):
        epsilon_in  = self._zeros_noise(self.in_features)
        epsilon_out = self._zeros_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._zeros_noise(self.out_features))



# import math
# import torch
# from torch import nn
# from torch.nn import init, Parameter
# from torch.nn import functional as F
# from torch.autograd import Variable
# # Noisy linear layer with independent Gaussian noise
# device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# class NoisyLinear(nn.Linear):
#   def __init__(self, in_features, out_features, sigma_init=0.5, bias=True):
#     super(NoisyLinear, self).__init__(in_features, out_features, bias=True)
#     # µ^w and µ^b reuse self.weight and self.bias
#     self.sigma_init = sigma_init
#     self.sigma_weight = Parameter(torch.Tensor(out_features, in_features))  # σ^w
#     self.sigma_bias = Parameter(torch.Tensor(out_features))  # σ^b
#     self.register_buffer('epsilon_weight', torch.zeros(out_features, in_features))
#     self.register_buffer('epsilon_bias', torch.zeros(out_features))
#     self.reset_parameters()
#
#   def reset_parameters(self):
#     if hasattr(self, 'sigma_weight'):  # Only init after all params added (otherwise super().__init__() fails)
#       init.uniform_(self.weight, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#       init.uniform_(self.bias, -math.sqrt(3 / self.in_features), math.sqrt(3 / self.in_features))
#       init.constant_(self.sigma_weight, self.sigma_init)
#       init.constant_(self.sigma_bias, self.sigma_init)
#
#   def forward(self, input):
#
#     return F.linear(input, self.weight + self.sigma_weight * Variable(self.epsilon_weight), self.bias + self.sigma_bias * Variable(self.epsilon_bias))
#
#   def sample_noise(self):
#     self.epsilon_weight = torch.randn(self.out_features, self.in_features).to(device)
#     self.epsilon_bias = torch.randn(self.out_features).to(device)
#
#   def remove_noise(self):
#     self.epsilon_weight = torch.zeros(self.out_features, self.in_features).to(device)
#     self.epsilon_bias = torch.zeros(self.out_features).to(device)