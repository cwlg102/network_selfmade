import math#####
import torch
import torch.nn as nn
import torch.nn.functional as F#####
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class DensenetGenerator(nn.Module):
    #densenet임~
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect', nb_layers=6, growth_rate=12):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(DensenetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout)]
            model += [TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout)]

        

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_dropout):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropout = nn.Dropout(0.5)
        
        if use_dropout:
            layer1 = [self.bn1, self.relu, self.conv1, self.dropout]
        else:
            layer1 = [self.bn1, self.relu, self.conv1]

        self.layer1 = nn.Sequential(*layer1)

        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if use_dropout:
            layer2 = [self.bn2, self.relu, self.conv2, self.dropout]
        else:
            layer2 = [self.bn2, self.relu, self.conv2]
        
        self.layer2 = nn.Sequential(*layer2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_dropout):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropout = nn.Dropout(0.5)
        if use_dropout:
            layer = [self.bn1, self.relu, self.conv1, self.dropout]
        else:
            layer = [self.bn1, self.relu, self.conv1]
        self.layers = nn.Sequential(*layer)
        #self.droprate = dropRate
    def forward(self, x):
        out = self.layers(x)
        return out

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, growth_rate, block, use_dropout):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layer(block, in_planes, growth_rate, nb_layers, use_dropout)
    def _make_layer(self, block, in_planes, growth_rate, nb_layers, use_dropout):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_planes+i*growth_rate, growth_rate, use_dropout))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.layer(x)
