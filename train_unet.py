##################
#
# Contributor : B.Y. You
# 2022/07/19
# Training script for compressible airfoil.
# Modified from thunil's DFP runTrain
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/runTrain.py
#
##################

import time, datetime
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
batchSize = 128
# Learning rate
lr = 0.00005
# Inputs channels, outputs channels
in_channel, out_channel = 3, 4
# Channel exponent to control network parameters amount
expo = 8
# Network　　
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = UNet(in_channel, out_channel, expo)
network = torch.nn.DataParallel(network)
network = network.to(device)
networkSummary, _ = summary_string(network, (in_channel, 128, 128), device=device)
# CPU maximum number
cpuMax = 12
torch.set_num_threads(cpuMax)
# Loss function
criterion = nn.L1Loss().to(device)
# Optimizer 
optimizer = torch.optim.Adam(network.parameters(), lr=lr)
# Data in validation dataset to demo
demoindex = 0

######## Dataset settings ########

# Dataset name
datasetName = 'OpenFOAM_com_airfoil_8866'
# Dataset directory.
dataDir = f'dataset/{datasetName}/train/'
# Validation dataset directory .
valDataDir = f'dataset/{datasetName}/val/' 
# Dataset preprocessing mode
preprocessingMode = ComAirfoilDataset.OFFSETREMOVAL
# Dataset usage mode, train or test.
mode = ComAirfoilDataset.TRAIN
# Dataset and the train loader declaration.
dataset = ComAirfoilDataset(dataDir, preprocessingMode, mode)
trainLoader = DataLoader(dataset, batchSize, shuffle=True)
valDataset = ValComAirfoilDataset(valDataDir, dataset)
valLoader = DataLoader(valDataset, batchSize, shuffle=False)

########## Log settings ##########

# Log writer declaration (record basic info).
textFileWriter = logWriter(logDir='./log/trainingLog/', dataset=dataset)
# Sumarry writer declaration (record loss per epoch).
networkName = 'UNet'
lossHistoryWriter = SummaryWriter(log_dir=f'log/SummaryWriterLog/{networkName}/bs{batchSize}_{epochs}ep_{lr}lr_{expo}expo/')
# Write train parameters setting
trainSetting = [f'Runing epochs : {epochs}', f'Batch size : {batchSize}', f'Learning rate : {lr}']
trainSetting += [f'Loss function : {criterion._get_name()}', f'Optimizaer : {optimizer.__class__.__name__}']
textFileWriter.writeLog(trainSetting)
textFileWriter.writeLog('*' * 64)
# Write network architecture and number of parameters 
textFileWriter.writeLog(networkSummary)
# Images output generator
imgagesGenerator = resultImagesGenerator(channels=4, resolution=128, root=f'./log/resultImages/{batchSize}_batchSize_{epochs}_epochs_{lr}lr_{expo}expo/')

######## Training script #########

def train():
    longPeriodTime = shortPeriodTime= startTime = time.time()
    for epoch in range(epochs):
        print(f'Starting epoch {epoch+1}/{epochs}')
        network.train()
        loss_sum = 0

        for _, data in enumerate(trainLoader):
            inputs, targets = data
            inputs, targets = inputs.float().to(device), targets.float().to(device)

            optimizer.zero_grad()
            outputs = network(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            loss_sum += loss.item()

            optimizer.step()

        loss_val_sum = 0
        network.eval()
        with torch.no_grad():
            for i, data in enumerate(valLoader):
                inputs, targets = data
                inputs, targets = inputs.float().to(device), targets.float().to(device)
                outputs = network(inputs)
            
                loss = criterion(outputs, targets)
                loss_val_sum += loss.item()

                if not i and not (epoch+1)%100 :
                    demoInputs = inputs.data.cpu().numpy()[demoindex].copy()
                    demoPrediction = outputs.data.cpu().numpy()[demoindex].copy()
                    demoGroundTruth = targets.data.cpu().numpy()[demoindex].copy()
                    _, demoGroundTruth = dataset.recoverTrueValues(demoInputs, demoGroundTruth)
                    _, demoPrediction = dataset.recoverTrueValues(demoInputs, demoPrediction)

                    folderName = f'epoch_{epoch+1}'
                    imgagesGenerator.setPredAndGround(demoPrediction, demoGroundTruth, folderName)
                    imgagesGenerator.predVsGround()
                    imgagesGenerator.globalDiff()
        
        loss = loss_sum / len(trainLoader)
        lossVal = loss_val_sum / len(valLoader)

        logLine = f'Epoch {epoch+1:04d} finished | Time duration : {(time.time()-shortPeriodTime)/60:.2f} minutes\n'
        shortPeriodTime = time.time()
        logLine += f'Traning loss : {loss:.4f} | Validation loss : {lossVal:.4f}'
        print(logLine)
        print('-'*30)
        if not (epoch+1)%100 :
            textFileWriter.writeLog(f' *** Time duration for last 100 epochs : {(time.time()-longPeriodTime)/60:.2f} minutes *** ')
            textFileWriter.writeLog(f'Epoch {epoch+1:04d} finished ')
            textFileWriter.writeLog(f'Traning loss : {loss:.4f} | Validation loss : {lossVal:.4f}')
            textFileWriter.writeLog('-'*64)
            longPeriodTime = time.time()
        lossHistoryWriter.add_scalars('Loss', {'Train' : loss, 'Validation' : lossVal}, epoch+1)

    # Record 
    totalTime = (time.time()-startTime)/60
    print(f'Training completed | Total time duration : {totalTime:.2f} minutes')
    nowTime = datetime.datetime.now()
    nowTime = str(nowTime).split(' ')[-1]
    nowTime = nowTime.split('.')[0]
    textFileWriter.writeLog(f'*** Training completed at {nowTime} ***')
    textFileWriter.writeLog(f'Total training time : {totalTime:.2f} minutes.')
    torch.save(network.module.state_dict(), f'{datasetName}_{batchSize}_batchSize_{epochs}_epochs_{lr}lr_{expo}expo')

       
if __name__ == '__main__':
    train()
    # pass
