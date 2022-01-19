import os, glob
import random
import numpy as np

root = 'D:\\BYY\\Airfoil DL\\NACAUIUC_TRAINING_1123\\NACAUIUC_10C_filldf1_1123\\'
class_folder = os.path.join(root, '*')

class_folder = glob.glob(class_folder)

train_img = []
train_label = []
val_img = []
val_label = []

train_ratio = 0.8
val_ratio = 0.2


for Class_list in class_folder:

    Class = Class_list.split('\\')[-1]
    Class = int(Class[:-2:1]) - 1
    img_list = os.path.join(Class_list, '*.png')   
    img_list = glob.glob(img_list)    
    random.shuffle(img_list)

    train_len = int(len(img_list)*train_ratio+1)
    tra_img_list = img_list[0:train_len]
    val_img_list = img_list[train_len:]

    for img in tra_img_list:
        train_img.append(img)
        train_label.append(Class)
    
    for img in val_img_list:
        val_img.append(img)
        val_label.append(Class)
    
np.save('D:\\BYY\\Airfoil DL\\dataset_order\\train_img.npy', train_img)
np.save('D:\\BYY\\Airfoil DL\\dataset_order\\train_label.npy', train_label)
np.save('D:\\BYY\\Airfoil DL\\dataset_order\\val_img.npy', val_img)
np.save('D:\\BYY\\Airfoil DL\\dataset_order\\val_label.npy', val_label)