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
    rating_set = "ratings.txt"
    trust_set = 'trust.txt'
    debug_file = 'debug.txt'
    peek_interval = 10
    method_id = ''

    def __init__(self):
        self.load_config()
        logs.basicConfig(filename=self.debug_file, filemode='w', level=logs.DEBUG, format="%(message)s")
    
    def load_config(self):
        self.config = {}
        with open('cf.conf', 'r') as f:
            for line in f:
                if line.find('=') == -1:
                    continue
                params = line.strip().split('=')
                self.config[params[0]] = params[1]
        
        # some commonly used configurations 
        self.prediction_method = self.config['predicting.method']
        self.similarity_method = self.config['similarity.method'].lower()
        self.similarity_threashold = float(self.config['similarity.threshold'])
        
        self.dataset = self.config['run.dataset']
        self.dataset_mode = self.config['dataset.mode']
        self.dataset_directory = self.config['dataset.directory'].replace('$run.dataset$', self.dataset)
        
        self.knn = int(self.config['kNN'])
        
        self.print_config()
        
    def print_config(self):
        print 'Run', self.method_id, 'method'
        print 'prediction method =', self.prediction_method
        print 'similarity method =', self.similarity_method
        print 'similarity threshold =', self.similarity_threashold
        if self.knn > 0: print 'kNN =', self.knn
        # print self.config
        
    def prep_test(self, data):
        if self.dataset_mode == 'coldUsers':
            self.test = {user:item_ratings for user, item_ratings in data.items() if len(data[user]) < 5}
        elif self.dataset_mode == 'all':
            self.test = data.copy()
        else:
            raise ValueError('invalid test dataset mode')
        self.total_test = sum([len(value) for value in self.test.viewvalues() ])
            
    def execute(self):
        ds = Dataset()
        ds.load_ratings(self.dataset_directory + self.rating_set)
        ds.load_trust(self.dataset_directory + self.rating_set)
        
        self.prep_test(ds.ratings)
        self.perform(ds.ratings, self.test)
        self.collect_results();
    
    def perform(self, train, test):
        validate_method = self.config['validating.method']
        if validate_method == 'cross_validation':
            self.cross_over(train, test)
        elif validate_method == 'leave_one_out':
            self.leave_one_out(train, test)
        else:
            raise ValueError('invalid validation method')
    
    def cross_over(self, train, test):
        pass
    
    def leave_one_out(self, train, test):
        pass
    
    def collect_results(self):
        # print performance
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
        else: 
            raise ValueError('invalid similarity measures')
        
class ClassicCF(AbstractCF):
    '''
    classic collaborative filtering algorithm
    '''
    def __init__(self):
        self.method_id = 'ClassicCF'
        AbstractCF.__init__(self)
        
    def cross_over(self, train, test):
        pass
    
    def leave_one_out(self, train, test):
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
                    if py.isnan(weight) or weight <= self.similarity_threashold: continue 
                    
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

class Trusties(ClassicCF):
    '''
    Trusties prototype for DBSCAN clustering method-based CF
    '''

    def __init__(self):
        self.method_id = 'Trusties'
        ClassicCF.__init__(self)
        
    def cluster(self, train):
        '''
        cluster the training users
        '''
        keys = train.keys()
        users = len(keys)
        similarities = []
        for userA_index in range(users):
            for userB_index in range(userA_index + 1, users):
                userA = keys[userA_index]
                userB = keys[userB_index]
                a = train[userA]
                b = train[userB]
                sim = self.similarity(a, b)
                if py.isnan(sim) or sim < 0.0: sim = 0.0
                similarities.append(sim)
        S = distance.squareform(similarities)
        
        # Compute DBSCAN
        self.db = DBSCAN(eps=0.5, min_samples=100).fit(S)
        # core_samples = db.core_sample_indices_
        labels = self.db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print 'Estimated number of clusters: %d' % n_clusters_
            
    def leave_one_out(self, train, test):
        pass
    
    def cross_over(self, train, test):
        self.cluster(train)
             
