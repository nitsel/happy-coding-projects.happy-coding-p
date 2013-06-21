'''
This code is modified from http://blog.csdn.net/xuy1202/article/details/6818205, 
where the author exemplifies the usage of NMF for clustering based on latent features. 
'''
from numpy import *
import random


# utility
def Unit(aa):
    return_l = []
    for line in aa:
        line_2 = [ x ** 2 for x in line]
        v = reduce(lambda x, y: x + y, line_2) ** 0.5
        line = [x / v for x in line]
        return_l.append(line)
    return return_l


def difcost(a, b):
    dif = 0
    for i in range(shape(a)[0]):
        for j in range(shape(a)[1]):
            dif += pow(a[i, j] - b[i, j], 2)
    return dif

# supposed there has pc features
def nmf(v, pc=10, itera=50):
    '''
    Implementation of the basic gradient descent non-negative matrix factorization
    '''
    ic, fc = shape(v)
    w = matrix([[random.random() for j in range(pc)] for i in range(ic)])
    h = matrix([[random.random() for i in range(fc)] for i in range(pc)])
    for i in range(itera):
        wh = w * h
        cost = difcost(v, wh)
        #print 'cost =', cost
        if cost < 1e-5:
            print 'Get out at iteration', i
            break
        hn = (transpose(w) * v)
        hd = (transpose(w) * w * h)
        h = matrix(array(h) * array(hn) / array(hd))
        wn = (v * transpose(h))
        wd = (w * h * transpose(h))
        w = matrix(array(w) * array(wn) / array(wd))
    return w, h

def example():
    '''The expected groups are (0, 1, 2), (3, 4, 5), (6, 7) 
       and (0, 1, 2, 6, 7), (3, 4, 5, 6, 7)'''
    l1 = [
        [5, 5, 0, 0],
        [6, 6, 0, 0],
        [7, 6, 1, 0],
        
        [0, 1, 9, 9],
        [1, 1, 9, 9],
        [0, 0, 5, 5],
        
        [9, 9, 9, 9],
        [4, 5, 6, 7] ]
    
    l1 = Unit(l1)
    m1 = matrix(l1)
    w, h = nmf(m1, 3, 2000)
    
    rows = shape(w)[0]
    for i in range(rows):
        for j in range(i + 1, rows):
            print '%s-%s:%s' % (i, j, difcost(w[i], w[j]))
    
    print w
    print h
    print w * h
    print m1
    
if __name__ == '__main__':
    example()
