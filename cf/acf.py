'''
Created on Feb 15, 2013

@author: guoguibing
'''
import sims
import math
import numpy as py
import logging as logs
import operator
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
    debug_file = 'debug.txt'
    peek_interval = 10

    def __init__(self):
        self.load_config()
        logs.basicConfig(filename=self.debug_file, filemode='w', level=logs.DEBUG, format="%(message)s")
    
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
                elif param == 'kNN':
                    self.knn = int(value)
                elif param == 'similarity.threshold':
                    self.similarity_threashold = float(value)
                
        
        print 'similarity method =', self.similarity_method
        print 'similarity threshold =', self.similarity_threashold
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
        mae = 'MAE = {0:.6f}\t'.format(MAE)
        predictable = len(self.errors)
        rc = 'RC = {0:d}/{1:d} = {2:2.2f}%\t'.format(predictable, self.total_test, float(predictable) / self.total_test * 100)
        RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
        rmse = 'RMSE = {0:.6f}'.format(RMSE)
        
        result = mae + rc + rmse
        print result
        logs.debug(result)
        
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
    classic collaborative filtering algorithm
    '''
    def __init__(self):
        AbstractCF.__init__(self)
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
                    mu_a = py.mean(a.values(), dtype=py.float64)
                
                # find the ratings of similar users and their weights
                votes = []
                weights = []
                for user in train.viewkeys():
                    if user == test_user: continue
                    b = train[user] if test_item in train[user] else {}
                    if not b: continue
                    
                    weight = self.similarity(a, b)
                    if py.isnan(weight): continue
                    # make sure the prediction will always be positive
                    if weight <= self.similarity_threashold: continue 
                    
                    mu_b = py.mean(b.values(), dtype=py.float64) if self.prediction_method == 'resnick_formula' else 0.0
                    
                    votes.append(train[user][test_item] - mu_b)
                    weights.append(weight)
                
                if not votes:continue
                
                # k-NN methods: find top-k most similar users according to their weights
                if self.knn > 0:
                    sorted_ws = sorted(enumerate(weights), reverse=True, key=operator.itemgetter(1))[:self.knn]
                    indeces = [item[0] for item in sorted_ws]
                    weights = [weights[index] for index in indeces]
                    votes = [votes[index] for index in indeces]
                
                # prediction
                pred = mu_a + py.average(votes, weights=weights)
                errors.append(abs(truth - pred))
                
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
