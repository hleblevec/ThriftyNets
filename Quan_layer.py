import torch
import torch.nn
import sys
from math import *
first_batch=0
train=0
class Quan_layer(torch.nn.Conv2d):

    def __init__(self, in_features, out_features, kernel_size=3,stride=1, padding = 0,dilation=1, groups=1, bias=False,bits=8):
        super(Quan_layer, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size,groups=groups)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_width = kernel_size
        self.groups = groups
        self.max_activation = 0
        self.bits=bits

        #self.n_bit_w = torch.nn.parameter.Parameter(torch.FloatTensor([8]))
        #self.n_bit_w.data.fill_(8)
        #self.n_bit_a = torch.nn.parameter.Parameter(torch.FloatTensor([8]))
        #self.n_bit_a.data.fill_(8)
                #b=torch.max(self.attentionWeights,dim=2)
        #print(b.shape)
        #self.attentionWeights.data[:,:,4]=b[0][:,:]

    def forward(self,input):
        global train
        global first_batch

        if train==1:
            maxi = torch.max(torch.abs(input)).item()
            if first_batch==1 or  self.max_activation < maxi:
                self.max_activation = maxi
        else:
        #print(torch.max(torch.abs(input)))
        #print(torch.max(torch.abs(input)).item())
            input = torch.clamp(input,-self.max_activation,self.max_activation)
        return torch.nn.functional.conv2d(quantifier(input,self.bits,self.max_activation), quantifier(self.weight,self.bits), bias = self.bias, stride = self.stride, padding = self.padding,groups=self.groups)
        #return torch.nn.functional.conv2d(quantifier(input,self.bits,0), quantifier(self.weight,self.bits), bias = self.bias, stride = self.stride, padding = self.padding,groups=self.groups)


class Quan_layer_fixed(torch.nn.Conv2d):

    def __init__(self, in_features, out_features, kernel_size=3,stride=1, padding = 0,dilation=1, groups=1, bias=False,bits=8):
        super(Quan_layer_fixed, self).__init__(in_features, out_features, stride = stride, bias = bias, padding = padding, kernel_size = kernel_size,groups=groups)
        self.in_features = in_features
        self.out_features = out_features
        self.kernel_width = kernel_size
        self.groups = groups
        self.max_activation = 47.7038
        self.max_weight = 0.6026
        self.bits=bits

        #self.n_bit_w = torch.nn.parameter.Parameter(torch.FloatTensor([8]))
        #self.n_bit_w.data.fill_(8)
        #self.n_bit_a = torch.nn.parameter.Parameter(torch.FloatTensor([8]))
        #self.n_bit_a.data.fill_(8)
                #b=torch.max(self.attentionWeights,dim=2)
        #print(b.shape)
        #self.attentionWeights.data[:,:,4]=b[0][:,:]

    def forward(self,input):
        global train
        global first_batch

        if train==1:
            maxi = torch.max(torch.abs(input)).item()
            if first_batch==1 or  self.max_activation < maxi:
                self.max_activation = maxi
        else:
        #print(torch.max(torch.abs(input)))
        #print(torch.max(torch.abs(input)).item())
            input = torch.clamp(input,-self.max_activation,self.max_activation)
        return torch.nn.functional.conv2d(fixed_quantifier(input,self.bits,self.max_activation), fixed_quantifier(self.weight,self.bits, self.max_weight), bias = self.bias, stride = self.stride, padding = self.padding,groups=self.groups)
        #return torch.nn.functional.conv2d(quantifier(input,self.bits,0), quantifier(self.weight,self.bits), bias = self.bias, stride = self.stride, padding = self.padding,groups=self.groups)


class ShiftBatchNorm2d(torch.nn.BatchNorm2d):

    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super(ShiftBatchNorm2d, self).__init__(num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)
        self.num_features = num_features

    def forward(self, input):
        #self.weight.data = ap2(self.weight.data/self.running_var)*self.running_var
        b_weight = ap2(self.weight.data/torch.sqrt(self.running_var))*torch.sqrt(self.running_var)
        # b_weight = Ap2NoGradient.apply(self.weight/torch.sqrt(self.running_var))*torch.sqrt(self.running_var)
        # print(torch.min(weight))
        # print(torch.min(self.bias.data))
        # print(torch.min(self.bias.data))
        # breakpoint()
        output = torch.nn.functional.batch_norm(
            input, self.running_mean, self.running_var, b_weight, self.bias,
            self.training, self.momentum, self.eps)
        # breakpoint()
        return output


class IntNoGradient(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.int().float()

    @staticmethod
    def backward(ctx, g):
        return g

class IntNoGradientFloor(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return x.floor()

    @staticmethod
    def backward(ctx, g):
        return g

class Ap2NoGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.sign(x)*2**(torch.round(torch.log2(torch.abs(x))))

    @staticmethod
    def backward(ctx, g):
        return g

def ap2(x):
    return torch.sign(x)*2**(torch.round(torch.log2(torch.abs(x))))

def maxWeight(weight,maxi):
    #liste_max = []
    #index=0
    #maxi = 0
    global train
    w = weight
    v = w.view(-1)
    if(maxi==0 or train==1):
        maxi=torch.max(torch.abs(v))
    #maxi = torch.max(torch.abs(v)) #.cpu().data.numpy()
    n=0
    if(maxi<1):
        while(maxi<1):
            maxi*=2
            n+=1
        return n-1
    elif(maxi>=2):
        while(maxi>=2):
           maxi/=2
           n-=1
        return n-1
    else:
        return n-1

def quantifier(w, n_bit,maxi=0):

    maxi=maxWeight(w,maxi)

    #w = weight.clone().cuda()
    a = w.shape
    v = torch.zeros(a)
    v = v + pow(2, n_bit-1 + maxi)
    v = v.float() #FloatNoGradient.apply(v)
    v = v.cuda()
    w = w*v
    w = IntNoGradient.apply(w)
    #w = FloatNoGradient.apply(w)
    w = w/v
    return w

def fixed_quantifier(w, n_bit,maxi):

    if abs(maxi) < 1:
        int_w = 1
    else:
        int_w = ceil(log2(maxi)) + 1
    frac_w = n_bit - int_w
    a = w.shape
    v = torch.zeros(a)
    v = v + pow(2, frac_w)
    v = v.float() #FloatNoGradient.apply(v)
    v = v.cuda()
    w = w*v
    w = IntNoGradientFloor.apply(w)
    #w = FloatNoGradient.apply(w)
    w = w/v
    return w
