'''
Created on Feb 15, 2013

@author: guoguibing
'''
import sims
import math
import numpy as py
from scipy.spatial import distance
from sklearn.cluster import DBSCAN
from data import Dataset

class AbstractCF(object):
    '''
    classdocs
    '''
    similairty_methods = ['pcc', 'cos', 'constant']
    dataset_modes = ['all', 'coldUsers']
    rating_set = "ratings.txt"
    trust_set = 'trust.txt'
    peek_interval = 10

    def __init__(self):
        self.load_config()
    
    def load_config(self):
        with open('cf.conf', 'r') as f:
            for line in f:
                if line.find('=') == -1:
                    continue
                params = line.strip().split('=')
                param = params[0]
                value = params[1]
                if param == 'similarity.method':
                    self.similarity_method = value.lower()
                    if self.similarity_method not in self.similairty_methods:
                        raise ValueError('invalid similarity measures')
                elif param == 'dataset.mode':
                    self.dataset_mode = value
                    if self.dataset_mode not in self.dataset_modes:
                        raise ValueError('invalid test dataset mode')
                elif param == 'run.dataset':
                    self.run_dataset = value
                elif param == 'dataset.directory':
                    self.dataset_directory = value.replace('$run.dataset$', self.run_dataset)
                elif param == 'predicting.method':
                    self.prediction_method = value
                else:
                    continue
        
        print 'similarity method =', self.similarity_method
        print 'prediction method =', self.prediction_method
    
    def prep_test(self, data):
        if self.dataset_mode == 'coldUsers':
            self.test = {user:item_ratings for user, item_ratings in data.items() if len(data[user]) < 5}
        elif self.dataset_mode == 'all':
            self.test = data.copy()
        self.total_test = sum([len(value) for value in self.test.viewvalues() ])
            
    def execute(self):
        ds = Dataset()
        ds.load_ratings(self.dataset_directory + self.rating_set)
        ds.load_trust(self.dataset_directory + self.rating_set)
        
        self.prep_test(ds.ratings)
        self.perform(ds.ratings, self.test)
        self.print_performance();
    
    def perform(self, train, test):
        pass
    
    def print_performance(self):
        MAE = py.mean(self.errors)
        print 'MAE = {0:.6f}\t'.format(MAE),
        predictable = len(self.errors)
        print 'RC = {0:d}/{1:d} = {2:2.2f}%\t'.format(predictable, self.total_test, float(predictable) / self.total_test * 100),
        RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
        print 'RMSE = {0:.6f}'.format(RMSE)
        
    def similarity(self, a, b):
        '''
        compute user or item similarity
        '''
        if self.similarity_method == 'pcc':
            return sims.pcc(a, b)
        elif self.similarity_method == 'cos':
            return sims.cos(a, b)
        elif self.similarity_method == 'constant':
            return 1.0
        
class ClassicCF(AbstractCF):
    '''
    classdocs
    '''
    def __init__(self):
        AbstractCF.__init__(self)
        self.similarity_method = 'constant'
        print 'Run ClassicCF method'
        
    def perform(self, train, test):
        errors = []
        count = 0
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
                
            # predict test item's rating
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                a = {item: rate for item, rate in train[test_user].items() if item != test_item}
                mu_a = 0.0
                if self.prediction_method == 'resnick_formula':
                    if not a: continue
                    mu_a = py.mean(a.values())
                    # print test_user, test_item, mu_a
                
                # find similar users, train, weights
                votes = 0.0
                weights = 0.0
                for user in train.viewkeys():
                    if user == test_user: continue
                    b = {item: rate for item, rate in train[user].items() if test_item in train[user]}
                    if not b: continue
                    
                    sim = self.similarity(a, b)
                    mu_b = py.mean(b.values()) if self.prediction_method == 'resnick_formula' else 0.0
                    
                    votes += sim * (train[user][test_item] - mu_b)
                    weights += abs(sim)
                
                # prediction
                if weights == 0.0:continue
                
                pred = mu_a + votes / weights
                errors.append(abs(truth - pred))
                
                # print test_user, test_item, truth, pred
        self.errors = errors      

class Trusties(AbstractCF):
    '''
    Trusties prototype for DBSCAN clustering method-based CF
    '''

    def __init__(self):
        AbstractCF.__init__(self)
        print 'Run Trusties method'
        
    def cluster(self, train):
        users = len(train.viewkeys())
        similarities = []
        for userA_index in range(users):
            for userB_index in range(userA_index + 1, users):
                userA = train.viewkeys()[userA_index]
                userB = train.viewkeys()[userB_index]
                a = train[userA]
                b = train[userB]
                sim = self.similarity(a, b)
                similarities.append(sim)
        S = distance.squareform(similarities)
        
        # Compute DBSCAN
        db = DBSCAN(eps=0.5, min_samples=10).fit(S)
        # core_samples = db.core_sample_indices_
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Estimated number of clusters: %d' % n_clusters_
            
        
    def perform(self, train, test):
        errors = []
        count = 0
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
            # predict test item's rating
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                a = {item: float(rate) for item, rate in train[test_user].items() if item != test_item}
                mu_a = py.mean(a.values()) if self.prediction_method == 'resnick_formula' else 0
                if py.isnan(mu_a):
                    continue
                
                # find similar users, train, weights
                rates = []
                weights = []
                for user in train.viewkeys():
                    if user == test_user: 
                        continue
                    b = {item: float(rate) for item, rate in train[user].items() if (item != test_item) and (test_item in train[user])}
                    if len(b) == 0: 
                        continue
                    sim = self.similarity(a, b)
                    mu_b = py.mean(b.values()) if self.prediction_method == 'resnick_formula' else 0
                    if py.isnan(mu_b):
                        continue
                
                    rates.append(train[user][test_item] - mu_b)
                    weights.append(abs(sim))
                
                # prediction
                if len(rates) == 0:
                    continue
                pred = [mu_a + rate * weight for rate, weight in zip(rates, weights)]
                pred = sum(pred) / len(rates)
                errors.append(abs(truth - pred))
        self.errors = errors      
