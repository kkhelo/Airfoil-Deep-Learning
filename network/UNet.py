##################
#
# Contributor : B.Y. You
# 2022/07/18
# UNet architecture modified from thunil's DfpNet
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/DfpNet.py
#
##################

from turtle import forward
import torch 
import torch.nn as nn

class DownSamplingblock(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, relu=True, size=4, pad=1) -> None:
        super().__init__()

        net = []
        net.append(nn.ReLU(inplace=True) if relu else nn.LeakyReLU(0.2, inplace=True))
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=size, stride=2, padding=pad, bias=True))
        if bn : net.append(nn.BatchNorm2d(out_channel))
        
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class UpSamplingBlock(nn.Module):
    def __init__(self, in_channel, out_channel, bn=True, relu=True, size=4, pad=1) -> None:
        super().__init__()

        net = []
        net.append(nn.ReLU(inplace=True) if relu else nn.LeakyReLU(0.2, inplace=True))
        net.append(nn.UpsamplingBilinear2d(scale_factor=2))
        net.append(nn.Conv2d(in_channel, out_channel, kernel_size=size-1, stride=1, padding=pad, bias=True))
        if bn : net.append(nn.BatchNorm2d(out_channel))
        
        self.net = nn.Sequential(*net)

    def forward(self, x):
        out = self.net(x)
        return out


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel, expo = 6) -> None:
        super().__init__()

        channel = int(2 ** expo + 0.5)
        self.downLayer1 = nn.Sequential(nn.Conv2d(in_channel, channel, kernel_size=4, stride=2, padding=1, bias=True))
        self.downLayer2 = DownSamplingblock(channel  , channel*2, bn=True , relu=False)
        self.downLayer3 = DownSamplingblock(channel*2, channel*2, bn=True , relu=False)
        self.downLayer4 = DownSamplingblock(channel*2, channel*4, bn=True , relu=False)
        self.downLayer5 = DownSamplingblock(channel*4, channel*8, bn=True , relu=False)
        self.downLayer6 = DownSamplingblock(channel*8, channel*8, bn=True , relu=False, size=2, pad=0)
        self.downLayer7 = DownSamplingblock(channel*8, channel*8, bn=False, relu=False, size=2, pad=0)

        self.upLayer7 = UpSamplingBlock(channel*8, channel*8, bn=True, relu=True, size=2, pad=0)
        self.upLayer6 = UpSamplingBlock(channel*16, channel*8, bn=True, relu=True, size=2, pad=0)
        self.upLayer5 = UpSamplingBlock(channel*16, channel*4, bn=True, relu=True)
        self.upLayer4 = UpSamplingBlock(channel*8, channel*2, bn=True, relu=True)
        self.upLayer3 = UpSamplingBlock(channel*4, channel*2, bn=True, relu=True)
        self.upLayer2 = UpSamplingBlock(channel*4, channel, bn=True, relu=True)

        upLayer1 = []
        upLayer1.append(nn.ReLU(inplace=True))
        upLayer1.append(nn.ConvTranspose2d(channel*2, out_channel, kernel_size=4, stride=2, padding=1, bias=True))
        self.upLayer1 = nn.Sequential(*upLayer1)

    def forward(self, x):
        downOut1 = self.downLayer1(x)
        downOut2 = self.downLayer2(downOut1)
        downOut3 = self.downLayer3(downOut2)
        downOut4 = self.downLayer4(downOut3)
        downOut5 = self.downLayer5(downOut4)
        downOut6 = self.downLayer6(downOut5)
        downOut7 = self.downLayer7(downOut6)
        upOut7   = self.upLayer7(downOut7)
        upOut6   = self.upLayer6(torch.cat([upOut7, downOut6]))
        upOuT5   = self.upLayer5(torch.cat([upOut6, downOut5]))
        upOut4   = self.upLayer4(torch.cat([upOuT5, downOut4]))
        upOut3   = self.upLayer3(torch.cat([upOut4, downOut3]))
        upOut2   = self.upLayer2(torch.cat([upOut3, downOut2]))
        upOut1   = self.upLayer1(torch.cat([upOut2, downOut1]))

        return upOut1
        
    
        


