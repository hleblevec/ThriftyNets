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

from train_utils import progress_bar

# from tensorboard_utils import*


from torch.utils.tensorboard import SummaryWriter



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
                conv_mode="classic", out_mode="pool", bn_mode = "classic", n_bits_weight=8, n_bits_activ=8, bias=False):
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
        # breakpoint()

        self.Lactiv = get_activ(activ)

        if self.bn_mode=="classic":
            self.Lnormalization = nn.ModuleList([nn.BatchNorm2d(n_filters) for x in range(n_iter)])
        elif self.bn_mode=="shift":
            self.Lnormalization = nn.ModuleList([Quan_layer.ShiftBatchNorm2d(n_filters) for x in range(n_iter)])
        #
        if self.conv_mode=="classic":
            self.Lconv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=self.bias)
        # elif self.conv_mode=="mb1":
        #     self.Lconv = MBConv(n_filters, n_filters, bias=self.bias)
        # elif self.conv_mode=="mb2":
        #     self.Lconv = MBConv(n_filters, n_filters//2, bias=self.bias)
        # elif self.conv_mode=="mb4":
        #     self.Lconv = MBConv(n_filters, n_filters//4, bias=self.bias)
        elif self.conv_mode=="quan":
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
            
            # print('Mean of activations:', torch.mean(hist[-1]))
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


# Training
def train(epoch):
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    print('\nEpoch: %d, lr: %f' % (epoch,lr))
    t0 = time.time()
    logger.update({"Epoch" :  epoch, "lr" : lr})
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    Quan_layer.first_batch = 1
    Quan_layer.train = 1
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        Quan_layer.first_batch = 0
        loss = criterion(outputs, targets)
        print('Mean of weights:', torch.mean(model.Lconv.weight.data))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        writer.add_scalar('training loss',
                    train_loss/(1+batch_idx),
                    epoch * len(train_loader) + batch_idx)
        writer.add_scalar('training accuracy',
                    100.*correct/total,
                    epoch * len(train_loader) + batch_idx)

        progress_bar(batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    logger.update({"epoch_time" : (time.time() - t0)/60 })
    logger.update({"train_loss" : loss.item()})
    logger.update({"train_acc" : 100.*correct/total})


def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    Quan_layer.train = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(test_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
            writer.add_scalar('test loss',
                                test_loss/(batch_idx+1),
                                epoch)
            writer.add_scalar('test accuracy',
                            100.*correct/total,
                            epoch)


    logger.update({"test_loss" : loss.item()})
    logger.update({"test_acc" : 100.*correct/total})
    if scheduler is not None:
        print("Step")
        scheduler.step()
    logger.log()
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'model': model.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+ args.name +'_ckpt.pth')
        best_acc = acc




## _____________________________________________________________________________________________

if __name__ == '__main__':
    # print("Starting")
    parser = utils.args()
    parser.add_argument("-n-bits-weight", "--n-bits-weight", default=8, type=int)
    parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
    # parser.add_argument('--min-lr', type=float, default=1e-4)
    # parser.add_argument('--patience', type=int, default=7)
    parser.add_argument("-tid", "--tid", default=0, type=int)
    parser.add_argument("-bn-mode", "--bn-mode", default="classic", type=str)
    # parser.add_argument('--resume', '-r', action='store_true',
                    # help='resume from checkpoint')
    args = parser.parse_args()
    print(args)

    writer = SummaryWriter('runs/'+args.name+'_'+str(args.tid))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # model = get_model(args, metadata)
    model = QuantizedThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
            pool_strategy=args.pool, conv_mode=args.conv_mode, bn_mode = args.bn_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)
    #model = UnfactorThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, args.iter, args.pool, args.activ, args.conv_mode, args.bias)
    #model = factorized_resnet18(metadata["n_classes"])
    #model = resnet18()

    n_parameters = sum(p.numel() for p in model.parameters())
    print("N parameters : ", n_parameters)
    if (hasattr(model, "n_filters")):
        print("N filters : ", model.n_filters)
    if (hasattr(model, "pool_stategy")):
        print("Pool strategy : ", model.pool_strategy)

    # if args.resume is not None:
    #     model.load_state_dict(torch.load(args.resume)["state_dict"])


    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+ args.name +'_ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
        # torch.save(model.state_dict(), args.name+".model")

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    scheduler = None
    if args.optimizer=="sgd":
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
        # schedule_fun = lambda epoch, gamma=args.gamma, steps=args.steps : utils.reduceLR(epoch, gamma, steps)
        # scheduler = LambdaLR(optimizer, lr_lambda= schedule_fun)
        # scheduler = ReduceLROnPlateau(optimizer, factor=args.gamma, patience=args.patience, min_lr=args.min_lr)
        scheduler = StepLR(optimizer, 50, gamma=0.1)
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




    for epoch in range(start_epoch, start_epoch+args.epochs):
        train(epoch)
        test(epoch)
