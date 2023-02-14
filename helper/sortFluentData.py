import numpy as np
import matplotlib.pyplot as plt
import time

data = np.loadtxt('y=0_pressure', skiprows=1)

length = data.shape[0]

data = np.delete(data, 0, 1)
data = np.delete(data, 1, 1)

dx = 1/256
dz = 0.5/256

xList = np.linspace(-0.25+dx, 0.75-dx, 256)
zList = np.linspace(dz, 0.5-dz, 256)
pressureContour = np.ones((256,256))
pressureContourMask = np.ones((256,256))

c, k = 0.1, 0.5
for i in range(64, 192):
    x = xList[i]*2*np.pi
    
    temp_x = x**2*np.tan(np.pi/28)**2 + c
    y = 0
    temp = temp_x/(1/np.cos(np.arctan(y/k)))**2
    z = np.sqrt(temp) * np.sin(x) * np.sin((y+np.pi)/2)
    z /= 2
    
    bound = int(z/dz-1)

    pressureContourMask[i, :bound+1] = 0

# pressureContourMask = np.flipud(pressureContourMask.transpose())
# pressureContourMask = pressureContourMask.transpose()

def dist2(p1, p2):
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2


def nearestNeighborValue(p1, pList):
    
    min = dist2(p1, pList[0, :2])
    value = pList[0, -1]
    for i in range(1, pList.shape[0]):
        localDist = dist2(p1, pList[i, :2])
        if  localDist < min :
            min = localDist
            value = pList[i, -1]

    return value

start = time.time()
print(start)
for i in range(256):
    for j in range(256):
        loopTimeStart = time.time()
        if pressureContourMask[i, j]:
            pressureContour[i, j] = nearestNeighborValue((xList[i], zList[j]), data[np.where(abs(data[:,0]-xList[i]) < 0.1)])
        print(f'Loop {i*255+j}')
        print(time.time() - loopTimeStart)

end = time.time()
print(f'{end-start:.2f}')
np.save('y=0_pressure', pressureContour)

# plt.figure()
# plt.contourf(pressureContour)
# plt.savefig('y=0_pressure')
# plt.show()
