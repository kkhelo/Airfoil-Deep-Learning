import numpy as np 
import multiprocessing as mp
import cv2 as cv 
import os, glob
import math


# define function that calculate for the shortest distance to edge
def find_shortest(i_:int,j_:int, edge_domain_min, edge_domain_max, upper, lower)->float:

    distance_to_upper = np.zeros(224, dtype=np.double)
    distance_to_lower = np.zeros(224, dtype=np.double)
    for j in range(edge_domain_min, edge_domain_max+1):
        distance_to_upper[j] = (j-j_)**2 + (upper[j]-i_)**2
        distance_to_lower[j] = (j-j_)**2 + (lower[j]-i_)**2

    answer = min(np.min(distance_to_upper[edge_domain_min:edge_domain_max+1]), np.min(distance_to_lower[edge_domain_min:edge_domain_max+1]))
    return answer


def make_sdf(img_path, target_path : str = None):

    img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
    img = cv.resize(img, (224, 224))
    # find the edge of airfoil
    upper = np.zeros(224, dtype=np.double)
    lower = np.zeros(224, dtype=np.double)

    for j in range(224):
        for i in range(224):
            if img[i,j] != 255:
                upper[j] = i
                for k in range(i,224):
                    if img[k,j] == 255:
                        lower[j] = k-1
                        break
                break

    edge_domain_min = np.where(upper != 0)[0][0]
    edge_domain_max = np.where(upper != 0)[0][-1]

    # find distance between bound and each pixel

    sdf = np.zeros(img.shape, dtype=float)

    for j in range(224):
        for i in range(224):
            if j in range(edge_domain_min, edge_domain_max+1):
                if i == upper[j] or i == lower[j]:
                    sdf[i,j] = 0.0
                elif i > upper[j] and i < lower[j]:
                    sdf[i,j] = -math.sqrt(find_shortest(i,j,edge_domain_min,edge_domain_max,upper,lower))
                else:
                    sdf[i,j] = math.sqrt(find_shortest(i,j,edge_domain_min,edge_domain_max,upper,lower))
            else:
                sdf[i,j] = math.sqrt(find_shortest(i,j,edge_domain_min,edge_domain_max,upper,lower))

    if target_path is None:
        temp = img_path.replace('NACAUIUC_10C_fill1_1123', 'NACAUIUC_10C_sdf1_1123')
        target_path = temp.replace('.png', '.npy')

    np.save(target_path, sdf)
    return sdf
    
def main(root, cpu_num : int = os.cpu_count()//2):

    class_folder = os.path.join(root, '*')

    class_folder = glob.glob(class_folder)

    for Class_list in class_folder:

        if os.path.isfile(Class_list):
            continue

        target_class_folder = Class_list.replace('NACAUIUC_10C_fill1_1123', 'NACAUIUC_10C_sdf1_1123')

        if not os.path.exists(target_class_folder):
            os.makedirs(target_class_folder)

        img_list = os.path.join(Class_list, '*.png')   
        img_list = glob.glob(img_list)    

        # multiprocessing
        p = mp.Pool(cpu_num)
        p.map(make_sdf, img_list)
        p.close()
        p.join()

        # for img_path in img_list:
        #     get_df_slope(img_path)

if __name__ == '__main__':
    root = '..\\dataset\\NACAUIUC_10C_fill1_1123\\'
    main(root=root, cpu_num=25)