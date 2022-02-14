from random import shuffle
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils
import numpy as np
from PIL import Image
import os
import time
from network.ResNet50 import ResNet50
from network.VGG16 import vgg16
from network.ResNet50_noRelu import ResNet50_noRelu
from network.ResNet50_PRelu import ResNet50_PRelu


class Airfoil_Dataset_From_Images(Dataset):
    def __init__(self, img_list, label_list, transform) -> None:
        self.imgs = img_list
        self.label = label_list
        self.transform = transform

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


class Airfoil_Dataset_From_NPY(Airfoil_Dataset_From_Images):
    def __init__(self, img_list, label_list, transform) -> None:
        super().__init__(img_list, label_list, transform)

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = np.load(fn)
        # img = img[:,:,1:]       # for two channels
        img = self.transform(img)
        label = self.label[index]

        return img, label


def main():

    torch.set_num_threads(10)

    test_name = 'sdf3_lr000005c_50ep_80data'
    dataset_channel = 3

    batch_size = 32
    epoch_num = 50
    lr = 0.00005
    if_scheduler = False
    # model = torch.load('model/3channel_lr00008c_21-50ep.pk1')
    model = ResNet50_PRelu(dataset_channel, 10)


    transform = transforms.Compose([transforms.ToTensor()])

    if dataset_channel == 1:
        # 1 channel dataset
        train_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_sdf1_1123/train_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_sdf1_1123/train_label.npy'), transform)
        val_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_sdf1_1123/val_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_sdf1_1123/val_label.npy'), transform)
    elif dataset_channel == 3:
        # 3 channels dataset
        train_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_sdf1_1123_3channel/train_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_sdf1_1123_3channel/train_label.npy'), transform)
        val_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_sdf1_1123_3channel/val_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_sdf1_1123_3channel/val_label.npy'), transform)
    elif dataset_channel == 7:
    # 7 channels dataset
        train_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_filldf1_1123_7channel/train_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_filldf1_1123_7channel/train_label.npy'), transform)
        val_set = Airfoil_Dataset_From_NPY(np.load('dataset/NACAUIUC_10C_filldf1_1123_7channel/val_img.npy'), 
                                    np.load('dataset/NACAUIUC_10C_filldf1_1123_7channel/val_label.npy'), transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # train

    writer_path = f'log/{test_name}'
    writer_loss = SummaryWriter(os.path.join(writer_path, 'loss/'))
    writer_acc  =SummaryWriter(os.path.join(writer_path, 'acc/'))

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model = model.to(device)

    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.8)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=3, eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3)
    record_batch = 30

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
        
        if if_scheduler:
            scheduler.step()

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
        
        print(f'Epoch {epoch+1:03d} finished | Traning acc : {training_acc/len(train_set)*100:.2f}% | Validation acc : {validation_acc/len(val_set)*100:.2f}%')
        print(f'Time elapsed : {time.time()/60 - start:.2f} min')
        

        writer_loss.add_scalars('Loss', {'Train' : training_loss/len(train_loader), 'Validation' : validation_loss/len(val_loader)}, epoch + 1)
        writer_acc.add_scalars('Accuracy', {'Train' : training_acc/len(train_set)*100, 'Validation' : validation_acc/len(val_set)*100}, epoch + 1)

    print(f'Training finished!!')
    print(f'Total time : {time.time()/60 - start:.2f} min')
    model_name = f'model/SDF_Resnet50/{test_name}.pk1'
    torch.save(model, model_name)
    writer_loss.flush()
    writer_acc.flush()
    writer_loss.close()
    writer_acc.close()


if __name__ == '__main__':
    main()
    
    