'''
Created on Mar 29, 2013

@author: guoguibing
'''
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

def plot_results():
    xs = []
    ys = []
    zs = []
    with open('data.txt', 'r') as f:
        for line in f:
            x, y, z = line.strip().split('\t')
            xs.append(float(x))
            ys.append(float(y))
            zs.append(float(z))
    
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(xs, ys, zs)
    
    plt.show()

plot_results() 
