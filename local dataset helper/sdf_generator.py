import numpy as np 
import matplotlib.pyplot as plt
import cv2 as cv 
import os, glob
import math

img = cv.imread('..\\dataset\\empty_airfoil_all\\-0.0005_goe311te15.png', cv.IMREAD_GRAYSCALE)
img = cv.resize(img, (224, 224))

# find the edge of airfoil
upper = np.zeros(224, dtype=np.double)
lower = np.zeros(224, dtype=np.double)


for j in range(224):
    for i in range(224):
        if img[i,j] == 0:
            upper[j] = i
            for k in range(i,224):
                if img[k,j] == 255:
                    lower[j] = k-1
                    break
            break

edge_domain_min = np.where(upper != 0)[0][0]
edge_domain_max = np.where(upper != 0)[0][-1]
distance_to_upper = np.zeros(224, dtype=np.double)
distance_to_lower = np.zeros(224, dtype=np.double)

# define function that calculate for the shortest distance to edge
def find_shortest(i_:int,j_:int)->float:
    for j in range(edge_domain_min, edge_domain_max+1):
        distance_to_upper[j] = (j-j_)**2 + (upper[j]-i_)**2
        distance_to_lower[j] = (j-j_)**2 + (lower[j]-i_)**2

    answer = min(np.min(distance_to_upper[edge_domain_min:edge_domain_max+1]), np.min(distance_to_lower[edge_domain_min:edge_domain_max+1]))
    return answer

# find distance between bound and each pixel

sdf = np.zeros(img.shape, dtype=float)

for j in range(224):
    for i in range(224):
        if j in range(edge_domain_min, edge_domain_max+1):
            if i == upper[j] or i == lower[j]:
                sdf[i,j] = 0.0
            elif i > upper[j] and i < lower[j]:
                sdf[i,j] = -math.sqrt(find_shortest(i,j))
            else:
                sdf[i,j] = math.sqrt(find_shortest(i,j))
        else:
            sdf[i,j] = math.sqrt(find_shortest(i,j))


plt.figure()
plt.contourf(sdf, cmap='jet')
plt.colorbar()
plt.axis('off')
plt.show()