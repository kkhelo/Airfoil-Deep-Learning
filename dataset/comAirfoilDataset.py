##################
#
# Contributor : B.Y. You
# 2022/07/19
# Dataset for compressible airfoil, used for UNet.
# Modified from thunil's TurbDataset
# https://github.com/thunil/Deep-Flow-Prediction/blob/master/train/dataset.py
#
##################

import numpy as np
import math
from glob import glob
from torch.utils.data import DataLoader, Dataset


class ComAirfoilDataset(Dataset):
    """
    Class for compressible airfoil simulation data, include data processing.
    ** Dimensionless and offset removal can not apply to the dataset together. ** 

    Args:
        dataDir : Directory where the dataset is, ex: 'dataset/train/'. ** Noticed the backslash at the end **
        dataChannel : (number of input data channels, number of target data channels).
        preprocessingMode : Choose the preprocessing method, offset removal or dimensionless.
        mode : Taining dataset or testing(evaluation) dataset.
    """
    TRAIN, TEST = 0, 1
    DIMENSIONLESS, OFFSETREMOVAL = 2, 3

    def __init__(self, dataDir:str,preprocessingMode=OFFSETREMOVAL, mode=TRAIN) -> None:
        super().__init__()
        self.baseDataDir = dataDir
        self.dataDir = dataDir
        self.mode = mode
        self.preprocessingMode = preprocessingMode

        self.__loadData()
        self.baseDataLength = self.length
        print('\n' + '*'*25)
        print(f' Load base dataset completed. ')
        print(f' Total data amount : {self.length:d}')
        print('*'*25 + '\n')

        if self.preprocessingMode==self.OFFSETREMOVAL: 
            self.__getPTMeanValue()
            if self.mode==self.TRAIN: 
                self.__removeOffset()
                print('\n' + '*'*25)
                print(f' Base dataset offset removal completed. ')
                print('*'*25 + '\n')
        elif self.preprocessingMode==self.DIMENSIONLESS:
            if self.mode==self.TRAIN: 
                self.__dimensionless()
                print('\n' + '*'*25)
                print(f' Base dataset dimensionless completed. ')
                print('*'*25 + '\n')
        else:
            print('\n' + '*'*25)
            print(' Preprocessing mode code error, no prprocessing is applied. ')
            print('*'*25 + '\n')
            return
        
        self.__getNormalizationVlaue()
        if self.mode == self.TRAIN : self.__normalization()
            
    def __loadData(self):
        fileList = glob(self.dataDir+'*')
        self.length = len(fileList)
        self.inputs = np.zeros((self.length, 3, 128, 128))
        self.targets = np.zeros((self.length, 4, 128, 128))

        for i in range(self.length):
            data = np.load(fileList[i])['a']
            self.inputs = data[0:3]
            self.targets = data[3:]

    def __getPTMeanValue(self):
        self.Offset = np.zeros((4,1))
        
        for i in range(self.length):
            temp = self.targets[i,:,:,:].copy()
            validPoint = 16384-len(np.where(temp[0]==0)[0])
            for j in range(4):
                self.Offset[j] += np.sum(temp[j])/validPoint
        
        for i in range(4):
            self.Offset[i] /= self.length
        
        print('\n' + '*'*25)
        print(f' Mean value aquired from base dataset as below : ')
        print(f' Pressure : {self.Offset[0]:.2f}')
        print(f' X-direction velocity : {self.Offset[1]:.2f}')
        print(f' Y-direction velocity : {self.Offset[2]:.2f}')
        print(f' Temperature : {self.Offset[3]:.2f}')
        print('*'*25 + '\n')

    def __removeOffset(self):
        OffseyArray = np.ones((4,128,128))
        for i in range(4):
            OffseyArray[i] *= self.Offset[i]

        for i in range(self.length):
            for j in range(4):
                self.targets[i,j,:,:] -= OffseyArray[j]
                self.targets[i,j,:,:] -= self.targets[i,j,:,:] * self.inputs[i,2,:,:]

    def __dimensionless(self):
        PDLess, TDLess = 100000, 300
        for i in range(self.length):
            VDLess = math.sqrt(self.inputs[i,0,0,0]**2 + self.inputs[i,1,0,0]**2)
            self.targets[i,0,:,:] /= PDLess
            self.targets[i,1,:,:] /= VDLess
            self.targets[i,2,:,:] /= VDLess
            self.targets[i,3,:,:] /= TDLess

    def __getNormalizationVlaue(self):
        self.inputsNorm = np.zeros((2, 1))
        self.targetsNorm = np.zeros((4, 1))

        for i in range(2):
            self.inputsNorm[i] = np.max(np.abs(self.inputs[:,i,:,:]))
        for i in range(4):
            self.targetsNorm[i] = np.max(np.abs(self.targets[:,i,:,:]))
        
        print('\n' + '*'*25)
        print(f' Normalization value aquired from base dataset as below : ')
        print(f' Initial condition X-direction velocity : {self.inputsNorm[0]:.2f}')
        print(f' Initial condition Y-direction velocity : {self.inputsNorm[1]:.2f}')
        print(f' Pressure : {self.targetsNorm[0]:.2f}')
        print(f' X-direction velocity : {self.targetsNorm[1]:.2f}')
        print(f' Y-direction velocity : {self.targetsNorm[2]:.2f}')
        print(f' Temperature : {self.targetsNorm[3]:.2f}')
        print('*'*25 + '\n')

    def __normalization(self):
        for i in range(2):
            self.inputs[:,i,:,:] /= self.inputsNorm[i]
        for i in range(4):
            self.targets[:,i,:,:] /= self.targetsNorm[i]
        
    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def recoverTrueValues(self, inputs, targets):
        """
        Return field values before preprocessing 
        (True inputs, True targets)
        """
        a, b = inputs.copy(), targets.copy()
        for i in range(2):
            a[i] *= self.inputsNorm[i]
        for i in range(4):
            b[i] *= self.targetsNorm[i]

        if self.preprocessingMode==self.OFFSETREMOVAL: 
            OffseyArray = np.ones((4,128,128))
            for i in range(4):
                OffseyArray[i] *= self.Offset[i]
            for i in range(4):
                b[i] += OffseyArray[i]
                b[i] -= OffseyArray[i] * a[2]
        else:
            PDLess, TDLess = 100000, 300
            VDLess = math.sqrt(a[0,0,0]**2 + a[1,0,0]**2)
            b[0] *= PDLess
            b[1] *= VDLess
            b[2] *= VDLess
            b[3] *= TDLess

        return a, b

    def testDataBuild(self, testDataDir):
        """
        Build test dataset 
        """
        self.dataDir = testDataDir
        self.__loadData()

        print('\n' + '*'*25)
        print(f' Load test dataset completed. ')
        print(f' Total test data amount : {self.length:d} ')
        print('*'*25 + '\n')

        if self.preprocessingMode==self.OFFSETREMOVAL: 
            self.__removeOffset()
            print('\n' + '*'*25)
            print(f' Test dataset offset removal completed. ')
            print(f' Mean value used for test dataset as below : ')
            print(f' Pressure : {self.Offset[0]:.2f}')
            print(f' X-direction velocity : {self.Offset[1]:.2f}')
            print(f' Y-direction velocity : {self.Offset[2]:.2f}')
            print(f' Temperature : {self.Offset[3]:.2f}')
            print('*'*25 + '\n')
        elif self.preprocessingMode==self.DIMENSIONLESS:
            self.__dimensionless()
            print('\n' + '*'*25)
            print(f' Test dataset dimensionless completed. ')
            print('*'*25 + '\n')
        else:
            print('\n' + '*'*25)
            print(' Preprocessing mode code error, no prprocessing is applied. ')
            print('*'*25 + '\n')
            return

        self.__normalization()
        print(f' Normalization value used for test dataset as below : ')
        print(f' Initial condition X-direction velocity : {self.inputsNorm[0]:.2f}')
        print(f' Initial condition Y-direction velocity : {self.inputsNorm[1]:.2f}')
        print(f' Pressure : {self.targetsNorm[0]:.2f}')
        print(f' X-direction velocity : {self.targetsNorm[1]:.2f}')
        print(f' Y-direction velocity : {self.targetsNorm[2]:.2f}')
        print(f' Temperature : {self.targetsNorm[3]:.2f}')