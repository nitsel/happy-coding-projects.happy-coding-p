'''
Created on Feb 13, 2013

A module for similarity computation

@author: Felix
'''

import numpy as py
import math

def pcc(a, b):
    vas, vbs = pairs(a, b)
    if len(vas) > 1:
        mu_a = py.mean(vas)
        mu_b = py.mean(vbs)
        ras = [x - mu_a for x in vas]
        rbs = [y - mu_b for y in vbs]
        num = sum([x * y for x, y in zip(ras, rbs)])
        den1 = sum([x ** 2 for x in ras])
        den2 = sum([y ** 2 for y in rbs])
        
        return num / (math.sqrt(den1) * math.sqrt(den2))
    else:
        return None

def cos(a, b):
    vas, vbs = pairs(a, b)
    if len(vas) > 0:
        num = sum([x * y for x, y in zip(vas, vbs)])
        den1 = sum([x ** 2 for x in vas])
        den2 = sum([y ** 2 for y in vbs])
        
        return num / (math.sqrt(den1) * math.sqrt(den2))
    else:
        return None

def pairs(a, b):
    vas = []
    vbs = []
    for item in a: 
        if item in b:
            vas.append(a[item])
            vbs.append(b[item])
    return (vas, vbs)
    
def test():
    a = {'1':1, '2':3, '3':5, '4':6, '5':8, '6':9, '7':6, '8':4, '9':3, '10':2, '11':3}
    b = {'1':2, '2':5, '3':6, '4':6, '5':7, '6':7, '7':5, '8':3, '9':1, '10':1, '12':5}
    print 'pcc =', pcc(a, b)  # 0.85471
    print 'cos =', cos(a, b)  # 0.96897

