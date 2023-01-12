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

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,n_blocks=9, padding_type='reflect', nb_layers=7, growth_rate=16):
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
        enc_list = []
        for i in range(num_updown_block):
            enc_list.append(nn.Sequential(
                DenseBlock(nb_layers=nb_layers, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
                TransitionBlock(ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout)))
        
        self.enc_list = nn.ModuleList(enc_list)

        
        model_deepest_dense = nn.Sequential(
        DenseBlock(nb_layers=nb_layers*2, in_planes=ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionBlock(ngf * mult + 2*nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout))
         #2 elements
        self.deepest_dense = model_deepest_dense

        dec_list = []
        for i in range(num_updown_block):
            dec_list.append(nn.Sequential(
                DenseBlock(nb_layers=nb_layers, in_planes=2 * ngf * mult, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
                TransitionBlock(2 * ngf * mult + nb_layers*growth_rate, ngf * mult, use_dropout=use_dropout)))
        
        
        self.dec_list = nn.ModuleList(dec_list)
        
        
        


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
        skip_list = []
        for i, module in enumerate(self.enc_list):
            skip_list.append(module(start))
            start = skip_list[-1]

        #deepest layers
        output = self.deepest_dense(start)

        #skipconnection and decoding layers
        for i, module in enumerate(self.dec_list):
            tout = module(torch.cat((skip_list[-(i+1)], output), 1))
            output = tout


        return self.model_decode(output)

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