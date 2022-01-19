
import glob
import os

root = 'D:\\BYY\\Airfoil DL\\NACAUIUC_TRAINING_1123\\NACAUIUC_10C_filldf1_1123\\'
class_folder = os.path.join(root, '*')

class_folder = glob.glob(class_folder)

for Class in class_folder:
    img_list = os.path.join(Class, '*')
    img_list = glob.glob(img_list)
    for img in img_list:
        img_new = img.split('.png')[0] + '.png'
        os.rename(img, img_new)

