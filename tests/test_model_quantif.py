from __future__ import print_function
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import *

from tqdm import tqdm, trange
from tqdm._utils import _term_move_up
prefix = _term_move_up() + '\r'

import random
import time
import os
import sys

import sys
sys.path.insert(1, r"C:\Users\Hugo\Documents\IMT_Atlantique\internship\ThriftyNets")

# from thrifty.models import get_model
from common.datasets import *
from common import utils
from train import plot_alphas
from train_quantif import *

DATA_PATH = r"C:\Users\Hugo\torch_datasets"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = utils.args()
parser.add_argument("-n-bits-weight", "--n-bits-weight", default=8, type=int)
parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
args = parser.parse_args()

args.activ = "relu"
args.auto_augment = False
args.batch_size = 100
args.bias = False
args.checkpoint_freq = 1
args.conv_mode = "classic"
args.cutout = 0
args.dataset = ["mnist"]
args.epochs = 200
args.filters = 16
args.gamma = 0.5
args.history = 1
args.iter = 7
args.learning_rate = 0.1
args.min_lr = 0.0001
args.model = "quantif"
args.momentum = 0.9
args.n_mini_batch = 1
args.n_params = None
args.name = "mnist8bit"
args.nesterov = True
args.optimizer = "sgd"
args.out_mode = "pool"
args.patience = 5
args.pool = [2]
args.resume = None
args.seed = 358756
args.test_batch_size = 100
args.topk  =None
args.weight_decay = 0.0005


_, test_loader, metadata = load_mnist(args)

model = QuantitizedRCNN(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
        pool_strategy=args.pool, conv_mode=args.conv_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)

model.load_state_dict(torch.load("mnist8bit.model", map_location=device))

# print(model.Lconv.weight.data.size())
# W = (model.Lconv.weight.data*(2**(7))).int()
#
#
# with open('conv2d_weights.csv', 'a') as f:
#     f.write('(')
#     for c1 in range(args.filters):
#         f.write('(')
#         for c2 in range(args.filters):
#             f.write('(')
#             for k1 in range(3):
#                 f.write('(')
#                 for k2 in range(2):
#                     f.write('"%s",' % format(W[c1][c2][k1][k2] & 0xff, '08b'))
#                 f.write('"%s"' % format(W[c1][c2][k1][2] & 0xff, '08b'))
#                 if k1 < 2:
#                     f.write('),')
#                 else:
#                     f.write(')')
#             if c2 < args.filters-1:
#                 f.write('),\n')
#             else:
#                 f.write(')')
#         if c1 < args.filters-1:
#             f.write('),\n')
#         else:
#             f.write(')')
#     f.write(')')


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
print("N filters : ", model.n_filters)
print("Pool strategy : ", model.pool_strategy)

#if args.resume is not None:
#    model.load_state_dict(torch.load(args.resume)["state_dict"])

if args.topk is not None:
    topk = tuple(args.topk)
else:
    if args.dataset=="imagenet":
        topk=(1,5)
    else:
        topk=(1,)

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

    ## TESTING
    test_loss = 0
    test_acc = 0
    correct = 0
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_acc = correct / len(test_loader.dataset)

    logger.update({ "test_loss" : test_loss, "test_acc" : test_acc })

    if scheduler is not None:
        scheduler.step(logger["test_loss"])
    lr = optimizer.state_dict()["param_groups"][0]["lr"]
    print("Epoch:", epoch)

    # if args.checkpoint_freq != 0 and epoch%args.checkpoint_freq == 0:
    #     name = args.name+ "_e" + str(epoch) + "_acc{:d}.model".format(int(10000*logger["test_acc"]))
    #     torch.save(model.state_dict(), name)

    logger.log()
