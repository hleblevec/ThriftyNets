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
sys.path.insert(1, r"./../")

# from thrifty.models import get_model
from common.datasets import *
from common import utils
from train_quantized2 import*
import Quan_layer


# DATA_PATH = r"C:\Users\Hugo\torch_datasets"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = utils.args()
parser.add_argument("-n-bits-weight", "--n-bits-weight", default=8, type=int)
parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
args = parser.parse_args()

args.activ = "relu"
args.auto_augment = False
args.batch_size = 100
args.bias = False
args.checkpoint_freq = 10
args.conv_mode = "classic"
args.bn_mode = "classic"
args.cutout = 8
args.dataset = ["cifar10"]
args.epochs = 200
args.filters = 256
args.gamma = 0.1
args.history = 1
args.iter = 20
args.learning_rate = 0.1
args.model = "res_thrifty"
args.momentum = 0.9
# args.n_mini_batch = 1
args.n_params = None
args.name = "cifar10_no_quan"
args.nesterov = True
args.optimizer = "sgd"
args.out_mode = "pool"
args.patience = 8
args.pool = [5]
args.resume = None
args.seed = 437546
args.test_batch_size = 100
args.topk  =None
args.weight_decay = 0.0005


train_loader, test_loader, metadata = load_cifar10(args)

model = QuantizedThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
        pool_strategy=args.pool, conv_mode=args.conv_mode, bn_mode = args.bn_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)


model.load_state_dict(torch.load("cifar10_conv_quan_16bit.model", map_location=device))

model.to(device)

model.eval()

Quan_layer.train = 0

max_weight = torch.max(model.Lconv.weight.data)
print("Max weight:", max_weight)
with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

print('Maximum:', model.max)
