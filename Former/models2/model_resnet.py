import math

from os.path import join as pjoin

from collections import OrderedDict  # pylint: disable=g-importing-member

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self,pretrained=True):
        super(ResNet18, self).__init__()
        if pretrained == True:
            resnet = models.resnet18()
            _parameterDir = 'models2/Resnet18_MS1M_pytorch.pth.tar'
            checkpoint = torch.load(_parameterDir)
            pretrained_state_dict = checkpoint['state_dict']
            model_state_dict = resnet.state_dict()
            for key in pretrained_state_dict:
                if ((key == 'module.fc.weight') | (key == 'module.fc.bias') |(key == 'module.feature.weight') | (key == 'module.feature.bias')):
                    pass
                else:
                    model_state_dict[key.replace('module.', '')] = pretrained_state_dict[key]
            resnet.load_state_dict(model_state_dict)
        else:
            resnet = models.resnet18(pretrained=False)
        self.features = nn.Sequential(*list(resnet.children())[:-3])




    def forward(self,x):
        b, f, _, _, _ = x.shape
        x = x.contiguous().view(-1,3,112,112)
        x = self.features(x)
        _, c, h, w = x.shape
        x = x.contiguous().view(b,f,c,h,w)
        return x

if __name__ == '__main__':
    input = torch.rand(2, 16, 3, 112, 112)
    model = ResNet18()
    print(model(input).shape)
