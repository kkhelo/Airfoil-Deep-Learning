# Contributor : B.Y. You
# 2022/02/09
# Resnet50 no Relu for SDF dataset

import torch.nn as nn


class Identical_Block(nn.Module):

    def __init__(self, in_channel, filter) -> None:
        super().__init__()

        net = []
        f1, f2, f3 = filter
        self.relu = nn.ReLU(inplace=True)

        net.append(nn.Conv2d(in_channel, f1, 1, 1, 0, bias = False))
        net.append(nn.BatchNorm2d(f1))
        # net.append(nn.ReLU(inplace=True))

        net.append(nn.Conv2d(f1, f2, 3, 1, 1, bias = False))
        net.append(nn.BatchNorm2d(f2))
        # net.append(nn.ReLU(inplace=True))

        net.append(nn.Conv2d(f2, f3, 1, 1, 0, bias = False))
        net.append(nn.BatchNorm2d(f3))

        self.net = nn.Sequential(*net)
        

    def forward(self, x):
        
        residual = x
        out = self.net(x)
        out += residual
        # out = self.relu(out)
        

        return out


class Douwn_Sample_Block(nn.Module):
    def __init__(self, in_channel, filter, s : int) -> None:
        super().__init__()

        net = []
        shortcut = []
        f1, f2, f3 = filter
        self.relu = nn.ReLU(inplace=True)

        net.append(nn.Conv2d(in_channel, f1, 1, s, 0, bias = False))
        net.append(nn.BatchNorm2d(f1))
        # net.append(nn.ReLU(inplace=True))

        net.append(nn.Conv2d(f1, f2, 3, 1, 1, bias = False))
        net.append(nn.BatchNorm2d(f2))
        # net.append(nn.ReLU(inplace=True))

        net.append(nn.Conv2d(f2, f3, 1, 1, 0, bias = False))
        net.append(nn.BatchNorm2d(f3))

        self.net = nn.Sequential(*net)

        shortcut.append(nn.Conv2d(in_channel, f3, 1, s, 0, bias = False))
        shortcut.append(nn.BatchNorm2d(f3))

        self.shortcut_net = nn.Sequential(*shortcut)

    def forward(self, x):

        residual = self.shortcut_net(x)
        out = self.net(x)
        out += residual
        # out = self.relu(out)

        return out


class ResNet50_noRelu(nn.Module):
    def __init__(self, in_channel : int, out_classes : int)->None:
        super().__init__()

        net = []

        ## Block 1  
        net.append(nn.Conv2d( in_channels=in_channel, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False))
        net.append(nn.BatchNorm2d(64))
        # net.append(nn.ReLU(True))
        net.append(nn.MaxPool2d(3, 2, padding=1))

        ## Block 2 
        net.append(Douwn_Sample_Block(64, [64, 64, 256], s=1))
        net.append(Identical_Block(256, [64, 64, 256]))
        net.append(Identical_Block(256, [64, 64, 256]))

        ## Block 3 
        net.append(Douwn_Sample_Block(256, [128, 128, 512], s=2))
        net.append(Identical_Block(512, [128, 128, 512]))
        net.append(Identical_Block(512, [128, 128, 512]))
        net.append(Identical_Block(512, [128, 128, 512]))

        # Block 4
        net.append(Douwn_Sample_Block(512, [256, 256, 1024], s=2))
        net.append(Identical_Block(1024, [256, 256, 1024]))
        net.append(Identical_Block(1024, [256, 256, 1024]))
        net.append(Identical_Block(1024, [256, 256, 1024]))
        net.append(Identical_Block(1024, [256, 256, 1024]))
        net.append(Identical_Block(1024, [256, 256, 1024]))

        # Block 5 
        net.append(Douwn_Sample_Block(1024, [512, 512, 2048], s=2))
        net.append(Identical_Block(2048, [512, 512, 2048]))
        net.append(Identical_Block(2048, [512, 512, 2048]))
        net.append(nn.AdaptiveAvgPool2d((7, 7)))

        self.net = nn.Sequential(*net)
        
        # Fully Connected Layer
        fc = []

        fc.append(nn.Linear(7*7*2048, out_classes))
        
        self.fc = nn.Sequential(*fc)
        

    def forward(self, x):
        
        out = self.net(x)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out


def main()->None:
    pass

if __name__ == '__main__':
    main()