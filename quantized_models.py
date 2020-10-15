
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import numpy as np
from math import *

from common.datasets import get_data_loaders
from thrifty.modules import *
import common.utils as utils
import Quan_layer
from ShiftBatchNorm2d import *

class QuantizedThriftyNet(nn.Module):
    """
    Residual Thrifty Network
    """
    def __init__(self, input_shape, n_classes, n_filters, n_iter, n_history, pool_strategy, activ="relu",
                conv_mode="classic", out_mode="pool", bn_mode = "classic", n_bits_weight=16, n_bits_activ=16, bias=False):
        super(QuantizedThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.n_history = n_history
        self.activ = activ
        self.conv_mode = conv_mode
        self.bn_mode = bn_mode
        self.bias = bias

        self.out_mode = out_mode

        self.n_bits_weight = n_bits_weight
        self.n_bits_activ = n_bits_activ

        self.n_tests = 10
        self.max = 19.6690
        self.max_weight = 0.7286


        self.pool_strategy = [False]*self.n_iter
        assert isinstance(pool_strategy, list) or isinstance(pool_strategy, tuple)
        if len(pool_strategy)==1:
            self.n_pool = 0
            freq = pool_strategy[0]
            for i in range(self.n_iter):
                if (i%freq == freq-1):
                    self.pool_strategy[i] = True
                    self.n_pool +=1
        else:
            self.n_pool = len(pool_strategy)
            for x in pool_strategy:
                self.pool_strategy[x] = True

        self.Lactiv = get_activ(activ)

        if self.bn_mode=="classic":
            self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])
        elif self.bn_mode=="shift":
            self.Lnormalization = nn.ModuleList([ShiftBatchNorm2d(n_filters) for x in range(n_iter)])

        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        elif self.conv_mode=="quan":
            self.Lconv = Quan_layer.Quan_layer(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias, bits=self.n_bits_activ)
        elif self.conv_mode == "quan_fixed":
            self.Lconv = Quan_layer.Quan_layer_fixed(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias, bits=self.n_bits_activ)

        self.activ = get_activ(activ)

        self.alpha = torch.zeros((n_iter, n_history+1))
        for t in range(n_iter):
            self.alpha[t,0] = 0.1
            self.alpha[t,1] = 0.9
        self.alpha = nn.Parameter(self.alpha)

        if out_mode == "pool":
            out_size = n_filters
        elif out_mode == "flatten":
            out_size = np.prod(self.Lblock.out_shape(input_shape))
        self.LOutput = nn.Linear(out_size, n_classes)

        self.n_parameters = sum(p.numel() for p in self.parameters())


    def forward(self, x):
        x = F.pad(x, (0, 0, 0, 0, 0, self.n_filters - self.input_shape[0]))
        hist = [None for _ in range(self.n_history-1)] + [x]
        for t in range(self.n_iter):
            a = self.Lconv(hist[-1])
            fm_height = hist[-1].size(-1)

            a = self.Lactiv(a)
            a = self.alpha[t,0] * a
            for i, x in enumerate(hist):
                if x is not None:
                    a = a + self.alpha[t,i+1] * x
            a = self.Lnormalization[t](a)
            for i in range(1, self.n_history-1):
                hist[i] = hist[i+1]
            hist[self.n_history-1] = a

            if self.pool_strategy[t]:
                for i in range(len(hist)):
                    if hist[i] is not None:
                        hist[i] = F.max_pool2d(hist[i], 2)

        if self.out_mode=="pool" and hist[-1].size()[-1]>1:
            out = F.adaptive_max_pool2d(hist[-1], (1,1))[:,:,0,0]
        elif self.out_mode=="flatten":
            out = hist[-1].view(hist[-1].size()[0], -1)
        else:
            out = hist[-1][:,:,0,0]
        return self.LOutput(out)
