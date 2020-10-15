import torch
import torch.nn


class ShiftBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(ShiftBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.num_features = num_features

    def forward(self, input):
        b_weight = Ap2NoGradient.apply(self.weight/torch.sqrt(self.running_var))*torch.sqrt(self.running_var)
        return torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, b_weight, self.bias,
            self.training, self.momentum, self.eps)


class Ap2NoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)*2**(torch.round(torch.log2(torch.abs(x))))

    @staticmethod
    def backward(ctx, g):
        return g

def ap2(x):
    return torch.sign(x)*2**(torch.round(torch.log2(torch.abs(x))))