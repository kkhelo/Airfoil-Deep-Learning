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
from dataset import comAirfoilDataset
import helper.comAiroiflUtils as utils
from helper.comAiroiflUtils import resultImagesGenerator

######### Basic settings #########

# Numbers of training epochs
epochs = 5000
# Batch size
bathcSize = 32
# Learning rate
lr = 0.0005
# Channel exponent to control network parameters amount
expo = 6

##################################

# logFile = 