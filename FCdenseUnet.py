import math#####
import torch
import torch.nn as nn
import torch.nn.functional as F#####
from torch.nn import init
import functools
from torch.optim import lr_scheduler


class DensenetGenerator(nn.Module):
    #densenet임~
    #num_updown_block 매개변수화 시키면 layer 자유자재로 
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.
    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,n_blocks=9, padding_type='reflect', nb_layers=3, growth_rate=12):
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

        model_encode = [
                nn.ReflectionPad2d(1),
                nn.Conv2d(input_nc, ngf, kernel_size=3, padding=0, bias=use_bias),
                norm_layer(ngf),
                nn.LeakyReLU(0.25)
                ]

        self.model_encode = nn.Sequential(*model_encode)

        deep_chan = 512
        layer_num = [4, 5, 7, 10, 12]
        ch_num = [ngf, 128, 256, 512, deep_chan]
        deep_layer = 15

        num_updown_block = 5
        self.num_iters = num_updown_block*2

        
        enc_list = []
        for i in range(num_updown_block):
            enc_list.append(DenseBlock(nb_layers=layer_num[i], in_planes=ch_num[i], growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout))
            if i < num_updown_block-1:
                enc_list.append(TransitionDownBlock(ch_num[i] + layer_num[i]*growth_rate, ch_num[i+1], use_dropout=use_dropout))
            else:enc_list.append(TransitionDownBlock(ch_num[i] + layer_num[i]*growth_rate, deep_chan, use_dropout=use_dropout))
        
        self.enc_list = nn.ModuleList(enc_list)

        
        model_deepest_dense = nn.Sequential(
        DenseBlock(nb_layers=deep_layer, in_planes=deep_chan, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
        TransitionUpBlock(deep_chan + deep_layer*growth_rate, deep_chan, use_dropout=use_dropout))
         #2 elements
        self.deepest_dense = model_deepest_dense

        dec_list = []
        for i in range(num_updown_block):
            if i < num_updown_block -1:
                dec_list.append(
                    nn.Sequential(
                    DenseBlock(nb_layers=layer_num[-(i+1)], in_planes=2 * ch_num[-(i+1)] + layer_num[-(i+1)]*growth_rate, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout),
                    TransitionUpBlock(2 * ch_num[-(i+1)] + 2*layer_num[-(i+1)]*growth_rate, ch_num[-(i+2)], use_dropout=use_dropout)
                    )
                    )
            else:
                dec_list.append(
                    DenseBlock(nb_layers=layer_num[-(i+1)], in_planes=2 * ch_num[-(i+1)] + layer_num[-(i+1)]*growth_rate, growth_rate=growth_rate, block=BottleneckBlock,use_dropout=use_dropout)
                    )

        self.dec_list = nn.ModuleList(dec_list)
        
        model_decode = [
        nn.Conv2d(2 * ch_num[0] + 2*layer_num[0]*growth_rate, output_nc, kernel_size=3, padding=1),
        nn.Tanh()
        ]

        self.model_decode = nn.Sequential(*model_decode)

    def forward(self, input):
        """Standard forward"""
        start = self.model_encode(input)
        #even - DenseBlock, odd - Transition Block

        #encoding layers
        skip_list = []

        for idx in range(0, self.num_iters, 2):
            skip_list.append(self.enc_list[idx](start))
            start = self.enc_list[idx+1](skip_list[-1])

        #deepest layers
        output = self.deepest_dense(start)

        #skipconnection and decoding layers
        for i, module in enumerate(self.dec_list):
            tout = module(torch.cat((skip_list[-(i+1)], output), 1))
            output = tout

        last_out = self.model_decode(output)

        return last_out

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_dropout):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropout = nn.Dropout(0.5)
        
        if use_dropout:
            layer1 = [self.bn1, self.relu, self.conv1, self.dropout]
        else:
            layer1 = [self.bn1, self.relu, self.conv1]

        self.layer1 = nn.Sequential(*layer1)

        self.bn2 = nn.InstanceNorm2d(inter_planes)
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

class TransitionUpBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_dropout):
        super(TransitionUpBlock, self).__init__()
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2,
                               padding=1, output_padding = 1, bias=False)
        layer = [self.conv1]
        self.layers = nn.Sequential(*layer)
        
        
    def forward(self, x):
        out = self.layers(x)
        return out

class TransitionDownBlock(nn.Module):
    def __init__(self, in_planes, out_planes, use_dropout):
        super(TransitionDownBlock, self).__init__()
        self.bn1 = nn.InstanceNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.max_pool = nn.MaxPool2d(2)
        if use_dropout:
            layer = [self.bn1, self.relu, self.conv1, self.dropout, self.max_pool]
        else:
            layer = [self.bn1, self.relu, self.conv1, self.max_pool]
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