'''
Created on Jul 18, 2013

@author: guoguibing
'''

import matplotlib as mpl
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10

def read_data():
    x = []
    y = []
    z = []
    with open('data.txt', 'r') as f:
        for line in f:
            data = line.strip().split('\t')
            x.append(float(data[0]))
            y.append(float(data[1]))
            z.append(float(data[2]))
    return x, y, z


x, y, z = read_data()
#X, Y=np.meshgrid(x, y)
#Z=np.transpose(z)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_wireframe(x, y, z, label='line')  

plt.show()
