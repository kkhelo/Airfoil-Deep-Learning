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
from dataset.comAirfoilDataset import ComAirfoilDataset, ValComAirfoilDataset
import helper.comAiroiflUtils as utils
from helper.comAiroiflUtils import resultImagesGenerator, logWriter
from network.UNet import UNet

####### Training settings ########

# Numbers of training epochs
epochs = 5000
# Batch size
batchSize = 32
# Learning rate
lr = 0.0005
# Inputs channels, outputs channels
in_channel, out_channel = 3, 4
# Channel exponent to control network parameters amount
expo = 6
# Network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
network = UNet(in_channel, out_channel, expo).to(device)
networkSummary, _ = summary_string(network, (in_channel, 128, 128), device=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)
# Loss function
criterion = nn.L1Loss().to(device)
# Optimizer 
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# Data in validation dataset to demo
demoindex = 5

######## Dataset settings ########

# Dataset directory.
dataDir = 'dataset/OpenFOAM_com_airfoil/train/'
# Validation dataset directory (optional).
valDaraDir = 'dataset/OpenFOAM_com_airfoil/val/' 
# Dataset preprocessing mode
preprocessingMode = ComAirfoilDataset.OFFSETREMOVAL
# Dataset usage mode, train or test.
mode = ComAirfoilDataset.TRAIN
# Dataset and the train loader declaration.
dataset = ComAirfoilDataset(dataDir, preprocessingMode, mode)
trainLoader = DataLoader(dataset, batchSize, shuffle=True)
valDataset = ValComAirfoilDataset(valDaraDir, dataset)
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

########## Log settings ##########

# Log writer declaration (record basic info).
textFileWriter = logWriter(logDir='./log/trainingLog/', dataset=dataset)
# Sumarry writer declaration (record loss per epoch).
lossHistoryWriter = SummaryWriter(log_dir=f'log/SummaryWriterLog/bs{batchSize}_{epochs}ep/')
# Write train parameters setting
trainSetting = [f'Runing epochs : {epochs}', f'Batch size : {batchSize}', f'Learning rate : {lr}']
trainSetting += [f'Loss function : {criterion._get_name()}', f'Optimizaer : {optimizer.__class__.__name__}']
textFileWriter.writeLog(trainSetting)
textFileWriter.writeLog('*' * 64)
# Write network architecture and number of parameters 
textFileWriter.writeLog(networkSummary)
# Images output generator
imgagesGenerator = resultImagesGenerator(channels=4, resolution=128, root=f'./log/resultImages/{batchSize}_{epochs}/')

######## Training script #########

def train():
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        loss_sum = 0

        for _, data in enumerate(trainLoader):
            inputs, targets = data
            inputs, targets = inputs.float.to(device), targets.float.to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            loss_sum += loss.item()

            optimizer.step()

        loss_val_sum = 0
        with network.eval():
            for i, data in enumerate(valLoader):
                inputs, targets = data
                inputs, targets = inputs.float.to(device), targets.float.to(device)
                outputs = network(inputs)
            
                loss = criterion(outputs, targets)
                loss_val_sum += loss.item()

                if not (epoch+1)%100:
                    if not i :
                        demoInputs = inputs.data.cpu().numpy()[demoindex].copy()
                        demoPrediction = outputs.data.cpu().numpy()[demoindex].copy()
                        demoGroundTruth = targets.data.cpu().numpy()[demoindex].copy()
                        _, demoGroundTruth = dataset.recoverTrueValues(demoInputs, demoGroundTruth)
                        _, demoPrediction = dataset.recoverTrueValues(demoInputs, demoPrediction)

                        folderName = f'epoch_{epoch+1}'
                        imgagesGenerator.setPredAndGround(demoPrediction, demoGroundTruth, folderName)
                        imgagesGenerator.predVsGround()
                        imgagesGenerator.globalDiff()

        loss_sum /= len(trainLoader)
        loss_val_sum /= len(valLoader)

        print()
        lossHistoryWriter.add_scalars('Loss', {'Train' : loss_sum, 'Validation' : loss_val_sum}, epoch+1)

            
if __name__ == '__main__':
    # train()
    pass
