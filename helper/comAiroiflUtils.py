import os, math, datetime, sys
import numpy as np, matplotlib.pyplot as plt
# sys.path.append('..')
from dataset.comAirfoilDataset import ComAirfoilDataset

def mkdir(dirList : list):
    for dir in dirList:
        if not os.path.exists(dir):
            os.makedirs(dir)

class logWriter():
    def __init__(self, logDir:str='./log/trainingLog/', dataset:ComAirfoilDataset=None) -> None:
        self.__makeUniqueLogFile(logDir)
        self.__writeTitle()
        if dataset is not None : self.__writeDatasetInfo(dataset)

    def __makeUniqueLogFile(self, logDir):
        self.date = str(datetime.date.today())
        self.__logName = logDir + self.date + '/'
        mkdir([self.__logName])
        duplicate = 1
        self.__logName += f'train-{duplicate}.txt'

        while os.path.exists(self.__logName):
            duplicate += 1
            self.__logName = self.__logName.split('train-')[0] + f'train-{duplicate}.txt'

        with open(self.__logName, 'w+') as of:
            print(f'Log file created : {self.__logName}')

    def __writeTitle(self):
        time = datetime.datetime.now()
        time = str(time).split(' ')[-1]
        time = time.split('.')[0]
        trainSet = self.__logName.split('-')[-1][0]

        with open(self.__logName, 'a+') as of:
            of.write(f'***** Train-{trainSet} *****\n')
            of.write(f'Starting Date : {self.date}\n')
            of.write(f'Starting time : {time}\n')
            of.write('*' * 64 + '\n')

    def __writeDatasetInfo(self, dataset:ComAirfoilDataset):
        """
        Write dataset info into log file, passing comAirfoilDataset as parameter.
        """
        mode = 'train' if dataset.mode == ComAirfoilDataset.TRAIN else 'test'
        with open(self.__logName, 'a+') as of:
            of.write(f'Using base dataset : {dataset.baseDataDir}\n')
            of.write(f'Base dataset size : {dataset.baseDataLength}\n')
            of.write(f'Dataset usage mode : {mode}\n')
            if dataset.preprocessingMode == ComAirfoilDataset.OFFSETREMOVAL:
                of.write('Applying preprocessing mode : Offset removal \n')
                of.write('*' * 64 + '\n')
                of.write(f'Mean value aquired from base dataset as below : \n')
                of.write(f'Pressure : {dataset.Offset[0]:.2f}\n')
                of.write(f'X-direction velocity : {dataset.Offset[1]:.2f}\n')
                of.write(f'Y-direction velocity : {dataset.Offset[2]:.2f}\n')
                of.write(f'Temperature : {dataset.Offset[3]:.2f}\n')
                of.write('*' * 64 + '\n')
            else:
                of.write('Applying preprocessing mode : Dimensionless\n')
                of.write('*' * 64 + '\n')

            of.write(f'Normalization value aquired from base dataset as below : \n')
            of.write(f'Initial X-direction velocity : {dataset.inputsNorm[0]:.2f}\n')
            of.write(f'Initial Y-direction velocity : {dataset.inputsNorm[1]:.2f}\n')
            of.write(f'Pressure : {dataset.targetsNorm[0]:.2f}\n')
            of.write(f'X-direction velocity : {dataset.targetsNorm[1]:.2f}\n')
            of.write(f'Y-direction velocity : {dataset.targetsNorm[2]:.2f}\n')
            of.write(f'Temperature : {dataset.targetsNorm[3]:.2f}\n')
            of.write('*' * 64 + '\n')

    def writeLog(self, lines):
        with open(self.__logName, 'a+') as of:
            if type(lines) is str : lines = [lines]
            for line in lines:
                of.write(line + '\n')

        
class resultImagesGenerator():
    """
    Generator class makes three type of images :
        * Predition vs Ground Truth
        * Global differnece
        * Local difference
    """
    def __init__(self, channels:tuple=4, resolution:int=128, root:str='./log/resultImages/') -> None:
        self.outputs, self.targets = np.zeros((channels, resolution, resolution)), np.zeros((channels, resolution, resolution))
        self.channels = channels
        self.root = root
        mkdir([root])
        
    def setPredAndGround(self, _outputs, _targets, folderName):
        outputs, targets = _outputs, _targets
        self.folderName = os.path.join(self.root, folderName)
        mkdir([self.folderName])
        for i in range(self.channels):
            self.outputs[i] = np.flipud(outputs[i].transpose())
            self.targets[i] = np.flipud(targets[i].transpose())
        
    def predVsGround(self):
        """
        Save contour filled plots of model prediction and ground truth as following order :
        pressure, x-dir velocity, y-dir velocity, temperature 
        """
        plt.figure(figsize=(12,6))
        for i in range(self.channels):
            plt.subplot(2,self.channels,i+1)
            plt.contourf(self.outputs[i], 100)
            plt.axis('off')
            plt.colorbar()
            plt.subplot(2,self.channels,i+1+self.channels)
            plt.contourf(self.targets[i], 100)
            plt.axis('off')
            plt.colorbar()

        plt.tight_layout()
        plt.savefig(os.path.join(self.folderName, 'Pred vs Ground'))

    def globalDiff(self):
        """
        Save difference contour filled plot divided by globalground truth maximum value.
        """
        plt.figure(figsize=(12,3))
        for i in range(self.channels):
            M = np.max(self.targets[i])
            diff = np.abs(self.targets[i]-self.outputs[i])
            diff /= M
            plt.subplot(1,self.channels,i+1)
            plt.contourf(diff, 200, cmap='Greens')
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.folderName, 'Diff by Global'))
        
    def localDiff(self):
        """
        Save difference contour filled plot divided by local ground truth value.
        """
        plt.figure(figsize=(12,3))
        diff = np.zeros((4,128,128))
        for i in range(self.channels):
            M = np.max(self.targets[i])
            diff[i] = np.abs(self.targets[i]-self.outputs[i])
            diff[i] = np.divide(diff[i], np.abs(self.targets[i]))

        for i in range(128):
            for j in range(128):
                if self.targets[1,i,j] == 0:
                    diff[:,i,j] = 0

        for i in range(self.channels):
            plt.subplot(1,self.channels,i+1)
            plt.contourf(diff[i], 200, cmap='Greys')
            plt.colorbar()
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(self.folderName, 'Diff by Local'))


if __name__ == '__main__':
    logger = logWriter(logDir='../log/trainingLog/')