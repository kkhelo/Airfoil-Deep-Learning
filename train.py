from random import shuffle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import cv2 as cv
import os, glob
from datetime import datetime
import time
from network.ResNet50 import ResNet50
from network.VGG16 import vgg16


class Airfoil_Dataset(Dataset):
    def __init__(self, img_list, label_list) -> None:
        self.imgs = img_list
        self.label = label_list
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn)
        img = img.convert('1')
        img = img.resize((224, 224))
        img = self.transform(img)
        label = self.label[index]

        return img, label

    def __len__(self):
        return len(self.imgs)


class Airfoil_3channel_Dataset(Airfoil_Dataset):
    def __init__(self, img_list, label_list) -> None:
        super().__init__(img_list, label_list)
        self.Gx = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])
        self.Gy = np.array([[1, 2, 1],
                           [0, 0, 0],
                           [-1, -2, -1]])
                                               

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = cv.imread(fn, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (224, 224))/255
        img_pad = cv.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv.BORDER_REPLICATE)
        img_x = np.zeros(img.shape)
        img_y = np.zeros(img.shape)

        for x in range(224):
            for y in range(224):
                roi = img_pad[x:x+3, y:y+3]
                img_x[x,y] = (roi * self.Gx).sum()  
                img_y[x,y] = (roi * self.Gy).sum()
        
        img = cv.merge([img, img_x, img_y])
        img = self.transform(img)
        label = self.label[index]

        return img, label


def main():

    batch_size = 32
    epoch_num = 20
    test_name = 'channel1_test1'

    # classes = ('10th', '1st', '2nd', '3rd', '4th', '5th',
    #                 '6th', '7th', '8th', '9th')
    
    train_set = Airfoil_3channel_Dataset(np.load('dataset/NACAUIUC_10C_filldf1_1123_3channel/train_img.npy'), 
                                np.load('dataset/NACAUIUC_10C_filldf1_1123_3channel/train_label.npy'))

    val_set = Airfoil_3channel_Dataset(np.load('dataset/NACAUIUC_10C_filldf1_1123_3channel/val_img.npy'), 
                                np.load('dataset/NACAUIUC_10C_filldf1_1123_3channel/val_label.npy'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # train

    writer_path = f'log/{test_name}'
    writer_loss = SummaryWriter(os.path.join(writer_path, 'loss/'))
    writer_acc  =SummaryWriter(os.path.join(writer_path, 'acc/'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    # model = torch.load('model/channe3_test1.pk1')
    model = ResNet50(3, 10)
    model = model.to(device)


    lr = 0.01
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=6, eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3)
    record_batch = 15

    start = time.time()/60
    for epoch in range(epoch_num):
        print('*' * 25, f'Epoch {epoch+1:03d} Start :', '*' * 25)

        training_loss = 0
        training_acc = 0
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            result = model(images)
            loss = criterion(result, labels)
            training_loss += loss
            loss.backward()
            optimizer.step()

            _, pred = torch.max(result, 1)
            acc = (pred == labels).sum()
            training_acc += acc.item()

            if (not (i+1) % record_batch) or ( i+1 == len(train_loader)):
                print(f'Epoch : {epoch+1:03d}/{epoch_num:03d} | Batch : {i+1:04d}/{len(train_loader):04d} | Loss : {loss:.2f}')       

        # scheduler.step()

        # Validation 
        validation_loss = 0
        validation_acc = 0
        model.eval()
        with torch.no_grad():
            for i, (images, labels) in enumerate(val_loader):
                images, labels = images.float().to(device), labels.long().to(device)
                result = model(images)
                loss = criterion(result, labels)
                validation_loss += loss
                _, pred = torch.max(result, 1)
                acc = (pred == labels).sum()
                validation_acc += acc.item()
        
        print(f'Epoch {epoch+1:03d} finished | Traning acc : {training_acc/8992*100:.2f}% | Validation acc : {validation_acc/2239*100:.2f}%')
        print(f'Time elapsed : {time.time()/60 - start:.2f} min')
        

        writer_loss.add_scalars('Loss', {'Train' : training_loss/len(train_loader), 'Validation' : validation_loss/len(val_loader)}, epoch + 1)
        writer_acc.add_scalars('Accuracy', {'Train' : training_acc/8992*100, 'Validation' : validation_acc/2239*100}, epoch + 1)

    print(f'Training finished!!')
    print(f'Total time : {time.time()/60 - start:.2f} min')
    model_name = f'model/{test_name}.pk1'
    torch.save(model, model_name)
    writer_loss.flush()
    writer_acc.flush()
    writer_loss.close()
    writer_acc.close()


if __name__ == '__main__':
    main()