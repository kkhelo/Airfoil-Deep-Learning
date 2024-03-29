################
#
# Contributor : B.Y. You
# 2022/07/19
# Script used to evaluate model  
# Modified from thunil's DFP runTestCpu
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/runTestCpu.py
#
################

import sys, os
import numpy as np
import torch
import torch.nn as nn
from torchsummary import summary_string
from network.UNet import UNet
from torch.utils.data import DataLoader
from dataset.comAirfoilDataset import ComAirfoilDataset
from helper.comAiroiflUtils import logWriter, resultImagesGenerator


####### network settings ########

# Batch size
# batchSize = int(sys.argv[1].split('_batchSize')[0].split('_')[-1])
batchSize = 1
# Inputs channels, outputs channels
in_channel, out_channel = 3, 4
# Channel exponent to control network parameters amount
expo = 7
# Network
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = sys.argv[1]
# Automatically detect model type and load
try :
    network = torch.load(model, map_location=device)
except:
    for expo in [6, 7, 8]:
        try:
            network = UNet(in_channel, out_channel, expo)
            network.load_state_dict(torch.load(model, map_location=device), strict=True)
            break
        except:
            pass
    
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
datasetName = '_'.join(sys.argv[1].split('_batchSize')[0].split('.\model\\')[-1].split('_')[:-1])
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
# AverageValueMap
averageGroundTruthMap = np.zeros((4,128,128))
averagePredMap = np.zeros((4,128,128))
# Value array used for test result diff percentage normalization
Mach = dataset.meanUBC/(1.16*287*300)**0.5
POff = (1+0.2*Mach**2)**3.5*100000
UOff = dataset.meanUBC
TOff = (1+0.2*Mach**2)*300
testOffset_total = [POff, UOff, UOff, TOff]

POff = 0.5*1.16*dataset.meanUBC**2
UOff = dataset.meanUBC
TOff = 300
testOffset = [POff, UOff, UOff, TOff]

textFileWriter.writeLog('*' * 64)
textFileWriter.writeLog(f'Normalization value of test data demo result (total pressure) : [{testOffset_total[0]:.2f}, {UOff:.2f}, {testOffset_total[3]}]')
textFileWriter.writeLog(f'Normalization value of test data demo result (dynamic pressure) : [{POff:.2f}, {UOff:.2f}, {TOff}]')

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
        
        demoInputs = inputs.data.cpu().numpy()[demoindex].copy()
        demoPrediction = outputs.data.cpu().numpy()[demoindex].copy()
        demoGroundTruth = targets.data.cpu().numpy()[demoindex].copy()
        _, demoGroundTruth = dataset.recoverTrueValues(demoInputs, demoGroundTruth)
        demoInputs, demoPrediction = dataset.recoverTrueValues(demoInputs, demoPrediction)
        averageGroundTruthMap += demoGroundTruth
        averagePredMap += demoPrediction
        

        if not i:
            # Single data result images generation 
            imgagesGenerator.setPredAndGround(demoPrediction, demoGroundTruth, folderName=os.path.join(model.split('\\')[-1], 'Single_self_total'))
            imgagesGenerator.predVsGround()
            selfU = (demoInputs[0,0,0]**2+demoInputs[1,0,0])**0.5
            selfMach = selfU/(1.16*287*300)**0.5
            selfOffset_total = [(1+0.2*selfMach**2)**3.5*100000, selfU, selfU, (1+0.2*selfMach**2)*300]
            imgagesGenerator.globalDiff(normalizeValue=selfOffset_total)
            imgagesGenerator.localDiff()
            imgagesGenerator.Diff()
            imgagesGenerator.saveNP()

            imgagesGenerator.setPredAndGround(demoPrediction, demoGroundTruth, folderName=os.path.join(model.split('\\')[-1], 'Single_self'))
            imgagesGenerator.predVsGround()
            selfU = (demoInputs[0,0,0]**2+demoInputs[1,0,0])**0.5
            selfOffset = [0.5*1.16*selfU**2, selfU, selfU, 300]
            imgagesGenerator.globalDiff(normalizeValue=selfOffset)
            imgagesGenerator.localDiff()
            imgagesGenerator.Diff()
            imgagesGenerator.saveNP()

# Calculation for mean loss in testing dataset
loss_test_sum /= len(testLoader)
textFileWriter.writeLog(f'Average loss for this model is {loss_test_sum:.5f}')
print(selfOffset)
print(selfOffset_total)

# # Get mean loss map
# averageGroundTruthMap /= dataset.length
# averagePredMap /= dataset.length

# # Average data result images generation 
# imgagesGenerator.setPredAndGround(averagePredMap, averageGroundTruthMap, folderName=os.path.join(model.split('\\')[-1], 'Average_total'))
# imgagesGenerator.predVsGround()
# imgagesGenerator.globalDiff(normalizeValue=testOffset_total)
# imgagesGenerator.Diff()
# imgagesGenerator.saveNP()

# imgagesGenerator.setPredAndGround(averagePredMap, averageGroundTruthMap, folderName=os.path.join(model.split('\\')[-1], 'Average'))
# imgagesGenerator.predVsGround()
# imgagesGenerator.globalDiff(normalizeValue=testOffset)
# imgagesGenerator.Diff()
# imgagesGenerator.saveNP()

# print(testOffset_total)
# print(testOffset)