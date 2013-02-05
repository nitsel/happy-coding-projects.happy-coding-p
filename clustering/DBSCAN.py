# -*- coding: utf-8 -*-
"""
Modified from scikit-learn module

"""
import numpy as np
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from sklearn import metrics


##############################################################################
# Read data from file
X=[]
labels_true=[]
with open('Data/R15.txt', mode='r') as afile:
    for line in afile:
        strs = line.split('\t')
        X.append([float(strs[0]), float(strs[1])])
        labels_true.append(int(strs[2])-1)
        

##############################################################################
# Compute similarities
D = distance.squareform(distance.pdist(X))
S = 1 - (D / np.max(D))

##############################################################################
# Compute DBSCAN
db = DBSCAN(eps=0.5, min_samples=10).fit(S)
core_samples = db.core_sample_indices_
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print 'Estimated number of clusters: %d' % n_clusters_
print "Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels)
print "Completeness: %0.3f" % metrics.completeness_score(labels_true, labels)
print "V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels)
print "Adjusted Rand Index: %0.3f" % \
    metrics.adjusted_rand_score(labels_true, labels)
print "Adjusted Mutual Information: %0.3f" % \
    metrics.adjusted_mutual_info_score(labels_true, labels)
print ("Silhouette Coefficient: %0.3f" %
       metrics.silhouette_score(D, labels, metric='precomputed'))

##############################################################################
# Plot result
import pylab as pl
from itertools import cycle

pl.close('all')
pl.figure(1)
pl.clf()

# Black removed and is used for noise instead.
colors = cycle('bgrcmybgrcmybgrcmybgrcmy')
for k, col in zip(set(labels), colors):
    if k == -1:
        # Black used for noise.
        col = 'k'
        markersize = 6
    class_members = [index[0] for index in np.argwhere(labels == k)]
    cluster_core_samples = [index for index in core_samples
                            if labels[index] == k]
    for index in class_members:
        x = X[index]
        if index in core_samples and k != -1:
            markersize = 14
        else:
            markersize = 6
        pl.plot(x[0], x[1], 'o', markerfacecolor=col,
                markeredgecolor='k', markersize=markersize)

pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()
