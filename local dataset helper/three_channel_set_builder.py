import os, glob
import random
import numpy as np
import cv2 as cv

root = '../dataset/NACAUIUC_10C_filldf1_1123/'

class_folder = os.path.join(root, '*')

class_folder = glob.glob(class_folder)

train_img = []
train_label = []
val_img = []
val_label = []

train_ratio = 0.8
val_ratio = 0.2

Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])


for Class_list in class_folder:

    target_class_folder = Class_list.replace('NACAUIUC_10C_filldf1_1123', 'NACAUIUC_10C_filldf1_1123_3channel')
    if not os.path.exists(target_class_folder):
        os.makedirs(target_class_folder)

    Class = Class_list.split('\\')[-1]
    Class = int(Class[:-2:1]) - 1
    img_list = os.path.join(Class_list, '*.png')   
    img_list = glob.glob(img_list)    
    random.shuffle(img_list)

    train_len = int(len(img_list)*train_ratio+1)
    tra_img_list = img_list[0:train_len]
    val_img_list = img_list[train_len:]

    for img_path in tra_img_list:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (224, 224))/255
        img_pad = cv.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv.BORDER_REPLICATE)
        img_x = np.zeros(img.shape)
        img_y = np.zeros(img.shape)

        for x in range(224):
            for y in range(224):
                roi = img_pad[x:x+3, y:y+3]
                img_x[x,y] = (roi * Gx).sum()  
                img_y[x,y] = (roi * Gy).sum()
        
        img = cv.merge([img, img_x, img_y])
        img_path = img_path.replace('NACAUIUC_10C_filldf1_1123', 'NACAUIUC_10C_filldf1_1123_3channel')
        img_path = img_path.replace('.png', '.npy')
        np.save(img_path, img)
        train_img.append(img_path)
        train_label.append(Class)
    
    for img_path in val_img_list:
        img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (224, 224))/255
        img_pad = cv.copyMakeBorder(img, 1, 1, 1, 1, borderType=cv.BORDER_REPLICATE)
        img_x = np.zeros(img.shape)
        img_y = np.zeros(img.shape)

        for x in range(224):
            for y in range(224):
                roi = img_pad[x:x+3, y:y+3]
                img_x[x,y] = (roi * Gx).sum()  
                img_y[x,y] = (roi * Gy).sum()
        
        img = cv.merge([img, img_x, img_y])
        img_path = img_path.replace('NACAUIUC_10C_filldf1_1123', 'NACAUIUC_10C_filldf1_1123_3channel')
        img_path = img_path.replace('.png', '.npy')
        np.save(img_path, img)
        val_img.append(img_path)
        val_label.append(Class)
    
np.save('../dataset order/NACAUIUC_10C_filldf1_1123_3channel/train_img.npy', train_img)
np.save('../dataset order/NACAUIUC_10C_filldf1_1123_3channel/train_label.npy', train_label)
np.save('../dataset order/NACAUIUC_10C_filldf1_1123_3channel/val_img.npy', val_img)
np.save('../dataset order/NACAUIUC_10C_filldf1_1123_3channel/val_label.npy', val_label)