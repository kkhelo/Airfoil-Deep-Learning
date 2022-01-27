import os, glob
import random
import numpy as np
import cv2 as cv
import multiprocessing as mp


# filters to get x-dir and y-dir slope
Gx = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])
Gy = np.array([[1, 2, 1],
               [0, 0, 0],
               [-1, -2, -1]])
                   

# function to expand df plot from 1 channel to 3 channels
def get_first_order_slope(img_path):
    
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


def get_second_order_slope(img_path):
    img = np.load(img_path)
    x = img[:,:,1]
    y = img[:,:,2]
    xx = np.zeros(x.shape)
    xy = np.zeros(x.shape)
    yx = np.zeros(x.shape)
    yy = np.zeros(x.shape)

    x_pad = cv.copyMakeBorder(x, 1, 1, 1, 1, borderType=cv.BORDER_REPLICATE)
    y_pad = cv.copyMakeBorder(y, 1, 1, 1, 1, borderType=cv.BORDER_REPLICATE)

    for x in range(224):
        for y in range(224):
            roi = x_pad[x:x+3, y:y+3]
            xx[x,y] = (roi * Gx).sum()  
            xy[x,y] = (roi * Gy).sum()
            roi = y_pad[x:x+3, y:y+3]
            yx[x,y] = (roi * Gx).sum()  
            yy[x,y] = (roi * Gy).sum()

    img = np.concatenate((img, xx[..., None], xy[..., None], yx[..., None], yy[..., None]), axis=2)
    img_path = img_path.replace('NACAUIUC_10C_filldf1_1123_3channel', 'NACAUIUC_10C_filldf1_1123_7channel')
    np.save(img_path, img)


def build_three_channel_npy(root, cpu_num : int = os.cpu_count()//2):

    class_folder = os.path.join(root, '*')

    class_folder = glob.glob(class_folder)

    for Class_list in class_folder:

        if os.path.isfile(Class_list):
            continue

        target_class_folder = Class_list.replace('NACAUIUC_10C_filldf1_1123', 'NACAUIUC_10C_filldf1_1123_3channel')

        if not os.path.exists(target_class_folder):
            os.makedirs(target_class_folder)

        img_list = os.path.join(Class_list, '*.png')   
        img_list = glob.glob(img_list)    

        # multiprocessing
        p = mp.Pool(cpu_num)
        p.map(get_first_order_slope, img_list)
        p.close()
        p.join()

        # for img_path in img_list:
        #     get_df_slope(img_path)

def build_seven_channel_npy(root, cpu_num : int = os.cpu_count()//2):
    class_folder = os.path.join(root, '*')

    class_folder = glob.glob(class_folder)

    for Class_list in class_folder:

        if os.path.isfile(Class_list):
            continue

        target_class_folder = Class_list.replace('NACAUIUC_10C_filldf1_1123_3channel', 'NACAUIUC_10C_filldf1_1123_7channel')

        if not os.path.exists(target_class_folder):
            os.makedirs(target_class_folder)

        img_list = os.path.join(Class_list, '*.npy')   
        img_list = glob.glob(img_list)    

        
        # multiprocessing
        p = mp.Pool(cpu_num)
        p.map(get_second_order_slope, img_list)
        p.close()
        p.join()

        # for img_path in img_list:
        #     get_second_order_slope(img_path)
        #     break
    

# function to build dataset order 
def build_dataset_order(root, train_ratio = 0.8, format = '*'):

    class_folder = os.path.join(root, '*')

    class_folder = glob.glob(class_folder)


    train_img = []
    train_label = []
    val_img = []
    val_label = []

    for Class_list in class_folder:

        if os.path.isfile(Class_list):
            continue

        Class = Class_list.split('\\')[-1]
        Class = int(Class[:-2:1]) - 1
        img_list = os.path.join(Class_list, format)   
        img_list = glob.glob(img_list)    
        random.shuffle(img_list)

        train_len = int(len(img_list)*train_ratio+1)
        tra_img_list = img_list[0:train_len]
        val_img_list = img_list[train_len:]

        for img in tra_img_list:
            img = img.split('\\', 1)[1]
            train_img.append(img)
            train_label.append(Class)
        
        for img in val_img_list:
            img = img.split('\\', 1)[1]
            val_img.append(img)
            val_label.append(Class)
    
        np.save(f'{root}\\train_img.npy', train_img)
        np.save(f'{root}\\train_label.npy', train_label)
        np.save(f'{root}\\val_img.npy', val_img)
        np.save(f'{root}\\val_label.npy', val_label)


if __name__ == '__main__':

    # root = '..\\dataset\\NACAUIUC_10C_filldf1_1123\\'
    root = '..\\dataset\\NACAUIUC_10C_filldf1_1123_3channel\\'
    # root = '..\\dataset\\NACAUIUC_10C_filldf1_1123_7channel\\'

    # format = '*.npy'
    # format = '*.png'

    # build_seven_channel_npy(root=root,cpu_num=12)
    # build_three_channel_npy(root=root, cpu_num=16)
    build_dataset_order(root, train_ratio=0.8)