################
#
# Contributor : B.Y. You
# 2022/07/19
# Script used to evaluate model  
# Modified from thunil's DFP runTestCpu
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/runTestCpu.py
#
################

import os, sys, math
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary_string
from network.UNet import UNet
from torch.utils.data import DataLoader
from dataset.comAirfoilDataset import ComAirfoilDataset
from helper.comAiroiflUtils import mkdir, logWriter, resultImagesGenerator
from glob import glob


####### network settings ########

# Batch size
batchSize = 64
# Inputs channels, outputs channels
in_channel, out_channel = 3, 4
# Channel exponent to control network parameters amount
expo = 7
# Network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = sys.argv[1]
# Automatically detect model type and load
try :
    network = UNet(in_channel, out_channel, expo)
    network.load_state_dict(torch.load(model, map_location=device), strict=True)
except:
    network = torch.load(model, map_location=device)
# Get model info
networkSummary, _ = summary_string(network, (in_channel, 128, 128), device=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)
# Loss function
criterion = nn.L1Loss().to(device)
# Data in testing dataset to demo
demoindex = 0

######## Dataset settings ########

# Dataset name
# datasetName = 'OpenFOAM_com_airfoil_4619'
datasetName = sys.argv[1][8:33]
# Dataset directory.
dataDir = f'dataset/{datasetName}/train/'
# Test dataset directory .
testDataDir = f'dataset/{datasetName}/test/' 
# Dataset preprocessing mode
preprocessingMode = ComAirfoilDataset.OFFSETREMOVAL
# Dataset usage mode, train or test.
mode = ComAirfoilDataset.TEST
# Dataset and the train loader declaration.
dataset = ComAirfoilDataset(dataDir, preprocessingMode, mode)
dataset.testDataBuild(testDataDir=testDataDir)
testLoader = DataLoader(dataset, batchSize, shuffle=False)

########## Log settings ##########

# Log writer declaration (record basic info).
textFileWriter = logWriter(logDir='./log/testingLog/', dataset=dataset)
# Write train parameters setting
testSetting = [f'Batch size : {batchSize}', f'Loss function : {criterion._get_name()}']
textFileWriter.writeLog(testSetting)
textFileWriter.writeLog('*' * 64)
# Write network architecture and number of parameters 
textFileWriter.writeLog(networkSummary)
# Images output generator
imgagesGenerator = resultImagesGenerator(channels=4, resolution=128, root=f'./log/resultImages/DEMO/')

######## Evaluation script ########

loss_test_sum = 0

network.eval()
with torch.no_grad():
    for i,data in enumerate(testLoader):
        inputs, targets = data
        inputs, targets = inputs.float().to(device), targets.float().to(device)
        outputs = network(inputs)
        loss = criterion(outputs, targets)
        loss_test_sum += loss.item()

        # Denormalize data and make images
        if not i:
            demoInputs = inputs.data.cpu().numpy()[demoindex].copy()
            demoPrediction = outputs.data.cpu().numpy()[demoindex].copy()
            demoGroundTruth = targets.data.cpu().numpy()[demoindex].copy()
            _, demoGroundTruth = dataset.recoverTrueValues(demoInputs, demoGroundTruth)
            _, demoPrediction = dataset.recoverTrueValues(demoInputs, demoPrediction)

            imgagesGenerator.setPredAndGround(demoPrediction, demoGroundTruth, folderName=model.split('\\')[-1])
            imgagesGenerator.predVsGround()
            imgagesGenerator.globalDiff()
            imgagesGenerator.localDiff()
            imgagesGenerator.Diff()