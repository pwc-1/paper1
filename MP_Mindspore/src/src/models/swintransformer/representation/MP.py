import numpy as np
import math
from scipy.stats import truncnorm
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.common.initializer as ini
import mindspore.common.dtype as mstype
import math
import mindspore
import mindspore as ms
# def _is_contiguous(tensor: torch.Tensor) -> bool:
#     # jit is oh so lovely :/
#     # if torch.jit.is_tracing():
#     #     return True
#     if torch.jit.is_scripting():
#         return tensor.is_contiguous()
#     else:
#         return tensor.is_contiguous(memory_format=torch.contiguous_format)

# @register_notrace_module
# class LayerNorm2d(nn.LayerNorm):
#     r""" LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
#     """

#     def __init__(self, normalized_shape, eps=1e-6):
#         super().__init__(normalized_shape, eps=eps)

#     def forward(self, x) -> torch.Tensor:
#         if _is_contiguous(x):
#             return F.layer_norm(
#                 x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)
#         else:
#             s, u = torch.var_mean(x, dim=1, keepdim=True)
#             x = (x - u) * torch.rsqrt(s + self.eps)
#             x = x * self.weight[:, None, None] + self.bias[:, None, None]
#             return x

class Covariance(nn.Cell):
    def __init__(self, 
                conv=False,
                remove_mean=True,
        ):
        super(Covariance, self).__init__()
        self.conv = conv
        self.remove_mean = remove_mean
    def _remove_mean(self, x):
        x=ops.swapaxes(x,axis0=-1,axis1=-2)
        mean = ops.AdaptiveAvgPool2D((1,1))
        _mean=mean(x)
        x = x - _mean
        x=ops.swapaxes(x,axis0=-1,axis1=-2)
        return x
    def remove_mean(self, x):
        mean = ops.AdaptiveAvgPool2D((1,1))
        _mean=mean(x)

        x = x - _mean

        return x

    def _cov(self, x):
        # channel

        batchsize, d, N = x.shape
        y = (1. / N ) * (x.bmm(x.transpose(0,2,1)))
        return y
    
    def cross_cov(self, x1, x2):
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
    def _cross_cov(self, x1, x2):
        # channel
        batchsize1,   N1,d1  = x1.shape
        batchsize2,   N2 ,d2 = x2.shape
        assert batchsize1 == batchsize2
        assert N1 == N2
        x1 = x1.transpose(0,2, 1)
        x2 = x2.transpose(0,2, 1)

        y = (1. / N1) * (x1.bmm(x2.transpose(0,2, 1)))
        return y    
    def construct(self, x, y=None):

        if self.remove_mean:
            if self.conv:
                self.remove_mean(x)
            else:
                x = self._remove_mean(x)

            if y is not None:
                if self.conv:
                    self.remove_mean(x)
                else:
                    y = self._remove_mean(y)
          
        if y is not None:
            if self.conv:
                x = self.cross_cov(x, y)
            else:
                x = self._cross_cov(x, y)
        else:

            x = self._cov(x)
   
        return x

class Moment_Probing_ViT(nn.Cell):
    def __init__(self, cross_type='near', in_dim=768, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2, num_classes=1000):
        super().__init__()
        self.cross_type = cross_type
        self.num_heads = num_heads

        self.proj = nn.Dense(in_dim, hidden_dim,has_bias=True)
        self.ln = nn.LayerNorm((hidden_dim//num_heads,))
        self.cov = Covariance()
        self.classifier1 = nn.Dense(in_dim, num_classes,has_bias=True)
        self.classifier2 = nn.Dense(2883, num_classes,has_bias=True)
        self.downblocks = nn.CellList()
        self.downblocks.append(nn.SequentialCell(
            nn.Conv2d(3,3, kernel_size=3, stride=2,pad_mode='valid',has_bias=False),
            nn.GELU(),
            nn.Conv2d(3,3,kernel_size=3, stride=2,pad_mode='valid',has_bias=False),
            ))
        #print(self.downblocks)
    def _l2norm(self, x):
        l2_normalize = ops.L2Normalize(axis=2)
        x = l2_normalize(x)
        return x
    
    def construct(self, cls_token, x):
        x = self.proj(x[:,1:,:])
        B, L, D = x.shape
        # divide head
        heads = ops.permute(x.reshape(B, L, self.num_heads, D//self.num_heads),(2, 0, 1, 3))
                
        if self.cross_type == 'n/2':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[self.num_heads//2])).unsqueeze(1)
            cov_list = self._l2norm(cov_list)

            for i in range(1, self.num_heads//2):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+self.num_heads//2])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = ops.cat([cov_list, cov], axis=1)

        elif self.cross_type == 'near':
            cov_list = self.cov(self.ln(heads[0]),self.ln(heads[1])).unsqueeze(1)
            #print("cov"+str(cov_list))
            cov_list = self._l2norm(cov_list)
            #print("l2"+str(cov_list))           
            for i in range(1, self.num_heads-1):
                cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
                cov = self._l2norm(cov)
                cov_list = ops.cat([cov_list, cov], axis=1)
                #print("near"+str(i)+str(cov_list))
        elif self.cross_type == 'cn2':
            cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
            for i in range(0, self.num_heads-1):
                for j in range(i+1, self.num_heads):
                    cov = self.cov(self.ln(heads[i]), self.ln(heads[j])).unsqueeze(1)
                    cov_list = ops.cat([cov_list, cov], axis=1)
                    
            cov_list = cov_list[:,1:]
        else:
            assert 0, 'Please choose from [one, near, cn2] !'

        
        for layer in self.downblocks:
            cov_list = layer(cov_list)
            #print("layer"+str(i)+str(cov_list))
            #cls=32*3*32*32
            #sequence = mindspore.numpy.arange(0.0001, 0.0001*cls + 0.0001, 0.0001,dtype=mindspore.float32)
            #tensor = sequence.reshape(32,3,32,32)
            #x = mindspore.Tensor(tensor, mindspore.float32)
            #x=layer(x)
            #print("layer"+str(i)+str(x))    
        cross_cov = cov_list.view(B, -1)
        #print("precross"+str(cross_cov))
        cls_token = self.classifier1(cls_token)
        #print("cls"+str(cls_token))        
        cross_cov = self.classifier2(cross_cov)
        #print("cross"+str(cross_cov))
        #print("cross+cls"+str((cls_token+cross_cov)/2))  
        return (cls_token + cross_cov)/2

# class Moment_Probing_CNN(nn.Module):
#     def __init__(self, cross_type='near', in_dim=1280, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2,num_classes=1000):
#         super().__init__()
#         self.cross_type = cross_type
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.in_dim = in_dim
#         self.proj = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0)
#         self.ln = nn.LayerNorm(hidden_dim//num_heads)
#         self.cov = Covariance()
#         norm_layer = partial(LayerNorm2d, eps=1e-6)
#         self.cls_head = nn.Sequential(OrderedDict([
#             ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
#             ('norm', norm_layer(self.in_dim)),
#             ('flatten', nn.Flatten(1) if 'avg' else nn.Identity()),
#             ('drop', nn.Dropout(0)),
#             ('fc', nn.Linear(self.in_dim, num_classes) if num_classes > 0 else nn.Identity())
#         ]))

#         self.classifier2 = nn.Linear(2883, num_classes)
        
#         self.downblocks = nn.ModuleList()
#         self.downblocks.append(nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.GELU(),
#             nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
#             ))

#     def _l2norm(self, x):
#         x = nn.functional.normalize(x, dim=2)
#         return x
    
#     def forward(self, x):
#         b, c, h, w = x.shape
#         cls_token = self.cls_head(x)
#         x = self.proj(x)
#         x = x.reshape(b, self.hidden_dim, -1)
#         B, C, D = x.shape
#         # divide head
#         heads = x.reshape(B, C//self.num_heads, self.num_heads, D).permute(2, 0, 3, 1)
                        
#         if self.cross_type == 'one':
#             cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(2, self.num_heads):
#                 cov = self.cov(self.ln(heads[0]), self.ln(heads[i])).unsqueeze(1)
#                 cov = self._l2norm(cov)
#                 cov_list = torch.cat([cov_list, cov], dim=1)

#         elif self.cross_type == 'near':
#             cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(1, self.num_heads-1):
#                 cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
#                 cov = self._l2norm(cov)
#                 cov_list = torch.cat([cov_list, cov], dim=1)

#         elif self.cross_type == 'cn2':
#             cov_list = self.cov(self.bn(heads[0]), self.bn(heads[self.num_heads//2])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(1, self.num_heads//2):
#                 cov = self.cov(self.bn(heads[i]), self.bn(heads[i+self.num_heads//2])).unsqueeze(1)
#                 cov_list = self._l2norm(cov_list)
#                 cov_list = torch.cat([cov_list, cov], dim=1)
#         else:
#             assert 0, 'Please choose from [one, near, cn2] !'

        
#         for layer in self.downblocks:
#             cov_list = layer(cov_list)
#         cross_cov = cov_list.view(B, -1)
        
#         cross_cov = self.classifier2(cross_cov)

#         return (cls_token + cross_cov)/2

# class Moment_Probing_MLP(nn.Module):
#     def __init__(self, cross_type='near', in_dim=1280, hidden_dim=512, num_heads=4, num_tokens=197, num_blocks=2, num_classes=100):
#         super().__init__()
#         self.cross_type = cross_type
#         self.num_heads = num_heads
#         self.hidden_dim = hidden_dim
#         self.in_dim = in_dim
#         self.proj = nn.Conv2d(in_dim, hidden_dim, 1, 1, 0)
#         self.ln = nn.LayerNorm(hidden_dim//num_heads)
#         self.cov = Covariance()
#         self.cls_head = nn.Sequential(OrderedDict([
#             ('global_pool', SelectAdaptivePool2d(pool_type='avg')),
#             ('flatten', nn.Flatten(1) if 'avg' else nn.Identity()),
#             ('drop', nn.Dropout(0)),
#             ('fc', nn.Linear(self.in_dim, num_classes) if num_classes > 0 else nn.Identity())
#         ]))

#         self.classifier2 = nn.Linear(2883, num_classes)
        
#         self.downblocks = nn.ModuleList()
#         self.downblocks.append(nn.Sequential(
#             nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
#             nn.GELU(),
#             nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=0, bias=False),
#             ))
        
#     def _l2norm(self, x):
#         x = nn.functional.normalize(x, dim=2)
#         return x
    
#     def forward(self, x):
#         b, c, h, w = x.shape
#         cls_token = self.cls_head(x)
#         x = self.proj(x)
#         x = x.reshape(b, self.hidden_dim, -1)
#         B, C, D = x.shape
#         # divide head
#         heads = x.reshape(B, C//self.num_heads, self.num_heads, D).permute(2, 0, 3, 1)
                        
#         if self.cross_type == 'one':
#             cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(2, self.num_heads):
#                 cov = self.cov(self.ln(heads[0]), self.ln(heads[i])).unsqueeze(1)
#                 cov = self._l2norm(cov)
#                 cov_list = torch.cat([cov_list, cov], dim=1)

#         elif self.cross_type == 'near':
#             cov_list = self.cov(self.ln(heads[0]), self.ln(heads[1])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(1, self.num_heads-1):
#                 cov = self.cov(self.ln(heads[i]), self.ln(heads[i+1])).unsqueeze(1)
#                 cov = self._l2norm(cov)
#                 cov_list = torch.cat([cov_list, cov], dim=1)

#         elif self.cross_type == 'cn2':
#             cov_list = self.cov(self.bn(heads[0]), self.bn(heads[self.num_heads//2])).unsqueeze(1)
#             cov_list = self._l2norm(cov_list)

#             for i in range(1, self.num_heads//2):
#                 cov = self.cov(self.bn(heads[i]), self.bn(heads[i+self.num_heads//2])).unsqueeze(1)
#                 cov_list = self._l2norm(cov_list)
#                 cov_list = torch.cat([cov_list, cov], dim=1)
#         else:
#             assert 0, 'Please choose from [one, near, cn2] !'

        
#         for layer in self.downblocks:
#             cov_list = layer(cov_list)
#         cross_cov = cov_list.view(B, -1)
        
#         cross_cov = self.classifier2(cross_cov)

#         return (cls_token + cross_cov)/2

if __name__ == '__main__':
    # load model
    cls=2*768
    sequence = mindspore.numpy.arange(0.0001, 0.0001*cls + 0.0001, 0.0001,dtype=mindspore.float32)
    tensor = sequence.reshape(2,768)
    cls = mindspore.Tensor(tensor, mindspore.float32)
    num_elements = 2*197*768
    sequence = mindspore.numpy.arange(0.0001, 0.0001*num_elements + 0.0001, 0.0001,dtype=mindspore.float32)
    tensor = sequence.reshape(2, 197, 768)
    #fixed_value = np.full((64, 3, 224, 224), 1, dtype=np.int32) #固定输入为42
    x = mindspore.Tensor(tensor, mindspore.float32)
    print(x)
    num_elements = 2*197*768
    sequence = mindspore.numpy.arange(0.01, 0.0001*num_elements + 0.010, 0.0001,dtype=mindspore.float32)
    tensor = sequence.reshape(2, 197, 768)
    x2 = mindspore.Tensor(tensor, mindspore.float32)
    model = Moment_Probing_ViT(in_dim=768,)
    y = model(cls, x)
    print(y)
    cov=Covariance()
    y=cov(x2,x)
    print(y)