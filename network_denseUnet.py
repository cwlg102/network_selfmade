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

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,n_blocks=9, padding_type='reflect', nb_layers=6, growth_rate=16):
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
        #assert(n_blocks >= 0)
        super(DensenetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_encode = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_encode += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        self.model_encode = nn.Sequential(*model_encode)

        num_updown_block = 5
        mult = 2 ** n_downsampling

        self.de1 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.de2 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.de3 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.de4 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.de5 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))

        
        model_deepest_dense = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
         #2 elements
        
        self.dd1 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.dd2 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.dd3 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.dd4 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        self.dd5 = nn.Sequential(
        DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
        
        self.deepest_dense = model_deepest_dense
        


        #for i in range(n_blocks):       # add ResNet blocks
        #    model_denseU += [DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout)]
        #    model_denseU += [TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout)]

        
        model_decode = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_decode += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model_decode += [nn.ReflectionPad2d(3)]
        model_decode += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_decode += [nn.Tanh()]

        self.model_decode = nn.Sequential(*model_decode)

    def forward(self, input):
        """Standard forward"""
        start = self.model_encode(input)
        #even - DenseBlock, odd - Transition Block

        #encoding layers
        skip1 = self.de1(start)
        skip2 = self.de2(skip1)
        skip3 = self.de3(skip2)
        skip4 = self.de4(skip3)
        skip5 = self.de5(skip4)

        #deepest layers
        deep = self.deepest_dense(skip5)

        #skipconnection and decoding layers
        out6 = self.dd1(torch.cat((skip5, deep), 1))
        out7 = self.dd2(torch.cat((skip4, out6), 1))
        out8 = self.dd3(torch.cat((skip3, out7), 1))
        out9 = self.dd4(torch.cat((skip2, out8), 1))
        tout = self.dd5(torch.cat((skip1, out9), 1))


        return self.model_decode(tout)

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