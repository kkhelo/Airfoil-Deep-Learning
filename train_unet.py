##################
#
# Contributor : B.Y. You
# 2022/07/19
# Training script for compressible airfoil.
# Modified from thunil's DFP runTrain
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/runTrain.py
#
##################

import os, sys, time
import numpy as np
import torch 
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torchsummary import summary_string
from dataset.comAirfoilDataset import ComAirfoilDataset
import helper.comAiroiflUtils as utils
from helper.comAiroiflUtils import resultImagesGenerator, logWriter
from network.UNet import UNet

####### Training settings ########

# Numbers of training epochs
epochs = 5000
# Batch size
bathcSize = 32
# Learning rate
lr = 0.0005
# Inputs channels, outputs channels
in_channel, out_channel = 3, 4
# Channel exponent to control network parameters amount
expo = 6
# Network
network = UNet(in_channel, out_channel, expo)

result, params_info = summary_string(network, (in_channel, 128, 128))
print(result)
# print(params_info)
with open('test.txt', 'w+') as of:
    of.write(result)
# print(network)

######## Dataset settings ########

dataDir='dataset/OpenFOAM_com_airfoil/'

##################################

# dataset = ComAirfoilDataset()

# logWriter = logWriter(logDir='./log/trainingLog/', dataset=None)