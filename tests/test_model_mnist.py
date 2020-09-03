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
from quantized_models import QuantizedThriftyNet


DATA_PATH = r"C:\Users\Hugo\torch_datasets"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = utils.args()
parser.add_argument("-n-bits-weight", "--n-bits-weight", default=8, type=int)
parser.add_argument("-n-bits-activ", "--n-bits-activ", default=8, type=int)
parser.add_argument("-tid", "--tid", default=0, type=int)
parser.add_argument("-bn-mode", "--bn-mode", default="classic", type=str)
args = parser.parse_args()


args.n_bits_activ = 16
args.n_bits_weight = 16
args.tid = 1
args.bn_mode = "shift"

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
args.model = "res_thrifty"
args.momentum = 0.9
args.n_mini_batch = 1
args.n_params = None
args.name = "mnist_16_fixed_bn_shift"
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



train_loader, test_loader, metadata = load_mnist(args)

model = QuantizedThriftyNet(metadata["input_shape"], metadata["n_classes"], args.filters, n_iter=args.iter, n_history=args.history,
        pool_strategy=args.pool, conv_mode=args.conv_mode, bn_mode = args.bn_mode, n_bits_weight=args.n_bits_weight, n_bits_activ = args.n_bits_activ)

model.load_state_dict(torch.load("mnist_16_fixed_bn_shift.model", map_location=device))

max_weight = torch.max(model.Lblock.Lconv.weight.data)
print("Max weight:", max_weight)

for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    output = model(data)
print('Maximum:', model.Lblock.max)
