from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *
import pdb

import numpy as np
from math import *

from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

import random
import time
import os
import sys
from common.datasets import get_data_loaders
from thrifty.modules import *
import common.utils as utils
# from Quan_layer import *
import Quan_layer

# class IntNoGradient(torch.autograd.Function):
#
#     @staticmethod
#     def forward(ctx, x):
#         return x.int().float()
#
#     @staticmethod
#     def backward(ctx, g):
#         return g

# def truncate(w):
#     v = torch.flatten(w)
#     for i in range(v.shape[0]):
#         if v[i].item() > 0:
#             v[i] = v[i].ceil()
#         else:
#             v[i] = v[i].floor()
#     return v.reshape(w.shape)


# def quantifier(w, n_bit,maxi):
#
#     int_w = ceil(log2(maxi)) + 1;
#     frac_w = n_bit - int_w;
#     a = w.shape
#     v = torch.zeros(a)
#     v = v + pow(2, frac_w)
#     v = v.float() #FloatNoGradient.apply(v)
#     v = v.cuda()
#     w = w*v
#     # w = IntNoGradient.apply(w)
#     w = w.floor()
#     #w = FloatNoGradient.apply(w)
#     w = w/v
#     return w

# def maxWeight(weight,maxi):
#     #liste_max = []
#     #index=0
#     #maxi = 0
#     global train
#     w = weight
#     v = w.view(-1)
#     if(maxi==0 or train==1):
#         maxi=torch.max(torch.abs(v))
#     #maxi = torch.max(torch.abs(v)) #.cpu().data.numpy()
#     n=0
#     if(maxi<1):
#         while(maxi<1):
#             maxi*=2
#             n+=1
#         return n-1
#     elif(maxi>=2):
#         while(maxi>=2):
#            maxi/=2
#            n-=1
#         return n-1
#     else:
#         return n-1
#
# def quantifier(w, n_bit,maxi=0):
#
#     maxi=maxWeight(w,maxi)
#
#     #w = weight.clone().cuda()
#     a = w.shape
#     v = torch.zeros(a)
#     v = v + pow(2, n_bit-1 + maxi)
#     v = v.float() #FloatNoGradient.apply(v)
#     v = v.cuda()
#     w = w*v
#     w = IntNoGradient.apply(w)
#     #w = FloatNoGradient.apply(w)
#     w = w/v
#     return w


class QuantizedThriftyNet(nn.Module):
    """
    Residual Thrifty Network
    """
    def __init__(self, input_shape, n_classes, n_filters, n_iter, n_history, pool_strategy, activ="relu",
                conv_mode="classic", out_mode="pool", n_bits_weight=8, n_bits_activ=8, bias=False):
        super(QuantizedThriftyNet, self).__init__()
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.n_iter = n_iter
        self.n_history = n_history
        self.activ = activ
        self.conv_mode = conv_mode
        self.bias = bias

        self.out_mode = out_mode

        self.n_bits_weight = n_bits_weight
        self.n_bits_activ = n_bits_activ

        # self.n_tests = 10
        # self.max = 17.6794
        # self.max_weight = 0.3638


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
        self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])
        # self.Lnormalization = nn.ModuleList([ShiftBatchNorm2d(n_filters) for x in range(n_iter)])
        #
        # if self.conv_mode=="classic":
        #     self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        # elif self.conv_mode=="mb1":
        #     self.Lconv = MBConv(n_filters, n_filters, bias=self.bias)
        # elif self.conv_mode=="mb2":
        #     self.Lconv = MBConv(n_filters, n_filters//2, bias=self.bias)
        # elif self.conv_mode=="mb4":
        #     self.Lconv = MBConv(n_filters, n_filters//4, bias=self.bias)

        self.Lconv = Quan_layer.Quan_layer(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias, bits=self.n_bits_activ)

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

## _____________________________________________________________________________________________

if __name__ == '__main__':
    # print("Starting")
    parser = utils.args()
    parser.add_argument("-n-bits-weight", "--n-bits-weight", default=8, type=int)
    parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)

    if args.topk is not None:
        topk = tuple(args.topk)
    else:
        if args.dataset=="imagenet":
            topk=(1,5)
        else:
            topk=(1,)

    # model = get_model(args, metadata)
    model = QuantizedThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
            pool_strategy=args.pool, conv_mode=args.conv_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)
    #model = UnfactorThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, args.iter, args.pool, args.activ, args.conv_mode, args.bias)
    #model = factorized_resnet18(metadata["n_classes"])
    #model = resnet18()

    if args.n_params is not None and args.model not in ["block_thrifty", "blockthrifty"]:
        n = model.n_parameters
        if n<args.n_params:
            while n<args.n_params:
                args.filters += 1
                model = get_model(args, metadata)
                n = model.n_parameters
        if n>args.n_params:
            while n>args.n_params:
                args.filters -= 1
                model = get_model(args,metadata)
                n = model.n_parameters

    n_parameters = sum(p.numel() for p in model.parameters())
    print("N parameters : ", n_parameters)
    if (hasattr(model, "n_filters")):
        print("N filters : ", model.n_filters)
    if (hasattr(model, "pool_stategy")):
        print("Pool strategy : ", model.pool_strategy)

    if args.resume is not None:
        model.load_state_dict(torch.load(args.resume)["state_dict"])

    model = model.to(device)

    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
        # scheduler = StepLR(optimizer, 100, gamma=0.1)
    elif args.optimizer=="adam":
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    try:
        os.mkdir("logs")
    except:
        pass
    logger = utils.Logger("logs/{}.log".format(args.name))

    with open("logs/{}.log".format(args.name), "a") as f:
        f.write(str(args))
        f.write("\nParameters : " + str(n_parameters))
        if hasattr(model, "n_filters"):
            f.write("\nFilters : " + str(model.n_filters))
        else:
            f.write("\nFilters : _ ")
        f.write("\n*******\n")

    print("-"*80 + "\n")
    test_loss = 0
    test_acc = torch.zeros(len(topk))
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        logger.update({"Epoch" :  epoch, "lr" : lr})

        ## TRAINING
        model.train()
        accuracies = torch.zeros(len(topk))
        loss = 0
        avg_loss = 0


        Quan_layer.first_batch = 1
        for batch_idx, (data, target) in tqdm(enumerate(train_loader),
                                              total=len(train_loader),
                                              position=1,
                                              leave=False,
                                              ncols=100,
                                              unit="batch"):

            Quan_layer.train = 1
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            Quan_layer.first_batch = 0

            loss = F.cross_entropy(output, target)
            avg_loss += loss.item()

            """
            alpha_loss = F.relu(model.Lblock.alpha - 2) + F.relu(-model.Lblock.alpha)
            loss += alpha_loss.sum()

            l1_loss = 0
            for i in range(model.Lblock.alpha.size()[0]):
                line_i = model.Lblock.alpha[i,1:]
                l1_loss += F.l1_loss(line_i, torch.zeros_like(line_i))
            """

            loss.backward()
            optimizer.step()
            accuracies += utils.accuracy(output, target, topk=topk)
            acc_score = accuracies / (1+batch_idx)

            tqdm_log = prefix+"Epoch {}/{}, LR: {:.1E}, Train_Loss: {:.3f}, Test_loss: {:.3f}, ".format(epoch, args.epochs, lr, avg_loss/(1+batch_idx), test_loss)
            for i,k in enumerate(topk):
                tqdm_log += "Train_acc(top{}): {:.3f}, Test_acc(top{}): {:.3f}, ".format(k, acc_score[i], k, test_acc[i])
            tqdm.write(tqdm_log)

        logger.update({"epoch_time" : (time.time() - t0)/60 })
        logger.update({"train_loss" : loss.item()})
        for i,k in enumerate(topk):
            logger.update({"train_acc(top{})".format(k) : acc_score[i]})

        ## TESTING

        Quan_layer.train = 0
        test_loss = 0
        test_acc = torch.zeros(len(topk))
        model.eval()
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
                test_acc += utils.accuracy(output, target, topk=topk)

        test_loss /= len(test_loader.dataset)
        test_acc /= len(test_loader)

        # plot_alphas(model.Lblock, "alpha_e{}.txt".format(epoch))

        logger.update({"test_loss" : test_loss})
        for i,k in enumerate(topk):
            logger.update({"test_acc(top{})".format(k) : test_acc[i]})

        if scheduler is not None:
            scheduler.step(logger["test_loss"])
        lr = optimizer.state_dict()["param_groups"][0]["lr"]
        print()

        if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
            name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc(top1)"]))
            torch.save(model.state_dict(), name)

        logger.log()

    torch.save(model.state_dict(), args.name+".model")
