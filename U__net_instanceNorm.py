import torch
from torch import nn

class EncodeLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        layer = [
            nn.Conv2d(in_chan, out_chan, 4, 2, 1),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_chan, out_chan, 1, 1),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(0.2)
        ]

        self.layer = nn.Sequential(*layer)

    def forward(self, x):
        return self.layer(x)

class DecodeLayer(nn.Module):
    def __init__(self, in_chan, out_chan):
        super().__init__()
        layer = [
            nn.ConvTranspose2d(in_chan, out_chan, 2, 2),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_chan, out_chan, 1, 1),
            nn.InstanceNorm2d(out_chan),
            nn.LeakyReLU(0.2)
        ]
        self.layer = nn.Sequential(*layer)
    def forward(self, x):
        return self.layer(x)

        
class Unet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Unet, self).__init__()

        self.prep = nn.Conv2d(in_channel, 64, 1, 1) # 1 64 256 256
        self.prepbat = nn.InstanceNorm2d(64)
        self.rel = nn.LeakyReLU(0.2)
        #prepone
        self.layer1 = EncodeLayer(64, 64) #1 64 128 128
        #skipcon1
        self.layer2 = EncodeLayer(64, 128) #1 128 64 64
        #skipcon2
        self.layer3 = EncodeLayer(128, 256) #1 256 32 32
        #skipcon3
        self.layer4 = EncodeLayer(256, 512) #1 512 16 16
        #skipcon4
        self.layer5 = EncodeLayer(512, 512) # 1 512 8 8
        #skipcon5
        self.layer6 = EncodeLayer(512, 1024) # 1 1024 4 4

        self.layer7 = DecodeLayer(1024, 512) # 1 512 8 8
        #skipcon5를 concat 1 1024 8 8
        self.layer8 = DecodeLayer(1024, 512) # 1 512 16 16
        #skipcon4를 concat  1 1024 16 16
        self.layer9 = DecodeLayer(1024, 256) # 1 256 32 32
        #skipcon3를 concat 1 512 32 32
        self.layer10 = DecodeLayer(512, 128) # 1 128 64 64
        #skipcon2를 concat 1 256 64 64
        self.layer11 = DecodeLayer(256, 64) # 1 64 128 128
        #skipcon1 concat 1 128 128 128
        self.layer12 = DecodeLayer(128, 64) # 1 64 256 256
        #prepone concat 1 128 256 256

        self.post = nn.Conv2d(128, out_channel, 1, 1)
        self.last = nn.Tanh()

    def forward(self, x):
        prepone = self.rel(self.prepbat(self.prep(x)))
        skipcon1 = self.layer1(prepone)
        skipcon2 = self.layer2(skipcon1)
        skipcon3 = self.layer3(skipcon2)
        skipcon4 = self.layer4(skipcon3)
        skipcon5 = self.layer5(skipcon4)
        bottleneck = self.layer6(skipcon5)
        deco1 = self.layer7(bottleneck)
        deco1_1 = torch.cat((skipcon5, deco1), 1)
        deco2 = self.layer8(deco1_1)
        deco2_1 = torch.cat((skipcon4, deco2), 1)
        deco3 = self.layer9(deco2_1)
        deco3_1 = torch.cat((skipcon3, deco3), 1)
        deco4 = self.layer10(deco3_1)
        deco4_1 = torch.cat((skipcon2, deco4), 1)
        deco5 = self.layer11(deco4_1)
        deco5_1 = torch.cat((skipcon1, deco5), 1)
        deco6 = self.layer12(deco5_1)
        deco6_1 = torch.cat((prepone, deco6), 1)
        postone = self.post(deco6_1)
        lastone = self.last(postone)

        return lastone
        








        








        