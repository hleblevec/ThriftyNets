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

import Quan_layer
from quantized_models import *
from ShiftBatchNorm2d import *

from train_utils import progress_bar

#from torch.utils.tensorboard import SummaryWriter


# Training
def train(epoch):
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
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
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # writer.add_scalar('training loss',
        #             train_loss/(1+batch_idx),
        #             epoch * len(train_loader) + batch_idx)
        # writer.add_scalar('training accuracy',
        #             100.*correct/total,
        #             epoch * len(train_loader) + batch_idx)

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
            # writer.add_scalar('test loss',
            #                     test_loss/(batch_idx+1),
            #                     epoch)
            # writer.add_scalar('test accuracy',
            #                 100.*correct/total,
            #                 epoch)


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
    parser = utils.args()
    parser.add_argument("-n-bits-weight", "--n-bits-weight", default=16, type=int)
    parser.add_argument("-n-bits-activ", "--n-bits-activ", default=16, type=int)
    parser.add_argument("-tid", "--tid", default=0, type=int)
    parser.add_argument("-bn-mode", "--bn-mode", default="classic", type=str)
    args = parser.parse_args()
    print(args)

    # writer = SummaryWriter('runs/'+args.name+'_'+str(args.tid))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    train_loader, test_loader, metadata = get_data_loaders(args)

    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    model = QuantizedThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
            pool_strategy=args.pool, conv_mode=args.conv_mode, bn_mode = args.bn_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)

    n_parameters = sum(p.numel() for p in model.parameters())
    print("N parameters : ", n_parameters)
    if (hasattr(model, "n_filters")):
        print("N filters : ", model.n_filters)
    if (hasattr(model, "pool_stategy")):
        print("Pool strategy : ", model.pool_strategy)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/'+ args.name +'_ckpt.pth')
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

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
    print('==> Saving best acc..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+ args.name +'_ckpt.pth')
    model.load_state_dict(checkpoint['model'])
    torch.save(model.state_dict(), args.name+".model")
