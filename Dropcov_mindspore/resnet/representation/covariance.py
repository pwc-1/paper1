#import torch
#import torch.nn as nn
#import torch.nn.functional as F
#from torch.autograd import Function
import numpy as np
import math
from src.model_utils.config import config
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as ini
import mindspore
import mindspore.common.dtype as mstype
from mindspore import Parameter
def conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    if config.net_name == "resnet152":
        stddev = (scale ** 0.5)
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    weight = np.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))
    return Tensor(weight, dtype=mstype.float32)


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def calculate_gain(nonlinearity, param=None):
    """calculate_gain"""
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    res = 0
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        res = 1
    elif nonlinearity == 'tanh':
        res = 5.0 / 3
    elif nonlinearity == 'relu':
        res = math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            neg_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            neg_slope = param
        else:
            raise ValueError("neg_slope {} not a valid number".format(param))
        res = math.sqrt(2.0 / (1 + neg_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
    return res


def _calculate_fan_in_and_fan_out(tensor):
    """_calculate_fan_in_and_fan_out"""
    dimensions = len(tensor)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:  # Linear
        fan_in = tensor[1]
        fan_out = tensor[0]
    else:
        num_input_fmaps = tensor[1]
        num_output_fmaps = tensor[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = tensor[2] * tensor[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out


def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Unsupported mode {}, please use one of {}".format(mode, valid_modes))
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out
def kaiming_normal(inputs_shape, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    return np.random.normal(0, std, size=inputs_shape).astype(np.float32)


def kaiming_uniform(inputs_shape, a=0., mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(inputs_shape, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    return np.random.uniform(-bound, bound, size=inputs_shape).astype(np.float32)

def _conv1x1(in_channel, out_channel, stride=1, use_se=False, res_base=False,padding=0):
    if use_se:
        weight = conv_variance_scaling_initializer(in_channel, out_channel, kernel_size=1)
    else:
        weight_shape = (out_channel, in_channel, 1)
        weight = Tensor(kaiming_normal(weight_shape, mode="fan_out", nonlinearity='relu'))
        if config.net_name == "resnet152":
            weight = _weight_variable(weight_shape)
    if res_base:
        return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride,
                         padding=0, pad_mode='pad', weight_init=weight)
    return nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride,
                     padding=padding, pad_mode='pad', weight_init=weight)

def _bn(channel, res_base=False):
    if res_base:
        return nn.BatchNorm2d(channel, eps=1e-5, momentum=0.1,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
    return nn.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                          gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
class Covariance(nn.Cell):
    def __init__(self, 
                cov_type='norm',
                remove_mean=True,
                dimension_reduction=None,
                input_dim=2048,
        ):
        super(Covariance, self).__init__()
        self.cov_type = cov_type
        self.remove_mean = remove_mean
        self.dr = dimension_reduction
        if self.dr is not None:
            if self.cov_type == 'norm':
                self.conv_dr_block = nn.SequentialCell(
                    _conv1x1(input_dim, self.dr[0], stride=1),
                    _bn(self.dr[0]),
                    nn.ReLU()
                )
            elif self.cov_type == 'cross':
                self.conv_dr_block = nn.SequentialCell(
                    nn.SequentialCell(
                        _conv1x1(input_dim, self.dr[0], stride=1),
                        _bn(self.dr[0]),
                        nn.ReLU()
                    ),
                    nn.SequentialCell(
                        _conv1x1(input_dim, self.dr[1],stride=1),
                        _bn(self.dr[1]),
                        nn.ReLU()
                    )
                )


                
    

    def _remove_mean(self, x):
        mean = ops.AdaptiveAvgPool2D((1,1))
        _mean=mean(x)

        x = x - _mean

        return x

    def _cov(self, x):
        # channel

        batchsize, d, h, w = x.shape
        N = h*w
        x = x.view(batchsize, d, N)

        y = (1. / N ) * (x.bmm(x.transpose(0,2,1)))
        
        return y
    
    def _cross_cov(self, x1, x2):
        # channel
        batchsize1, d1, h1, w1 = x1.shape
        batchsize2, d2, h2, w2 = x2.shape
        N1 = h1*w1
        N2 = h2*w2
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.view(batchsize1, d1, N1)
        x2 = x2.view(batchsize2, d2, N2)

        y = (1. / N1) * (x1.bmm(x2.transpose(0,2, 1)))
        return y
    
    def construct(self, x, y=None):
        #import pdb;pdb.set_trace()
        if self.dr is not None:
            if self.cov_type == 'norm':
                x = self.conv_dr_block(x)

            elif self.cov_type == 'cross':
                if y is not None:
                    x = self.conv_dr_block[0](x)
                    y = self.conv_dr_block[1](y)

                else:
                    ori = x
                    x = self.conv_dr_block[0](ori)
                    y = self.conv_dr_block[1](ori)

        if self.remove_mean:
            x = self._remove_mean(x)

            if y is not None:
                y = self._remove_mean(y)
          
        if y is not None:
            x = self._cross_cov(x, y)

        else:

            x = self._cov(x)
   
        return x

class Triuvec(nn.Cell):
     def __init__(self):
            super(Triuvec,self).__init__()
            self.ind=mindspore.Tensor(np.ones(8256),mindspore.float32)
            #self.ind=-1
     def construct(self,x):
         batchSize = x.shape[0]
         dim = x.shape[1]
         dtype=x.dtype
         x = x.reshape(batchSize, dim*dim)
         one=ops.Ones()
         I = mindspore.numpy.triu(one((dim,dim),dtype)).reshape(dim*dim)
         index =ops.nonzero(I)
         index=index.view(int(dim*(dim+1)/2))
         zero=ops.Zeros()
         y = zero((batchSize,int(dim*(dim+1)/2)),dtype)
         y = x[:,index]
         self.ind=index
         #ctx.save_for_backward(input,index)
         return y

     def bprop(self,x,out,dout):
         #global bprop_debug
         #bprop_debug = True
         index=self.ind
         batchSize = x.shape[0]
         dim = x.shape[1]
         dtype=x.dtype
         zer=ops.Zeros()
         grad_input = zer((batchSize,dim*dim),dtype)
         grad_input[:,index] = dout
         output= grad_input.reshape(batchSize,dim,dim)
         return output
# if __name__=='__main__' :
#     one=np.arange(0.00,321126.4, 0.1)
#     x=mindspore.Tensor(one,mindspore.float32).reshape(128,128,14,14)
#     cov=Covariance()
#     x=cov(x)
    
