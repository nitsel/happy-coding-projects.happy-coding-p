'''
Created on Feb 15, 2013

@author: guoguibing
'''
import sims
import numpy as py
from data import Dataset

class AbstractCF(object):
    '''
    classdocs
    '''
    similairty_methods = ['pcc', 'cos', 'constant']
    dataset_modes = ['all', 'cold']
    rating_set = "ratings.txt"
    trust_set = 'trust.txt'

    def __init__(self):
        '''
        Constructor
        '''
        self.load_config()
    
    def load_config(self):
        with open('cf.properties', 'r') as f:
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
                    self.dataset_mode = value.lower()
                    if self.dataset_mode not in self.dataset_modes:
                        raise ValueError('invalid test dataset mode')
                elif param == 'run.dataset':
                    self.run_dataset = value
                elif param == 'dataset.directory':
                    self.dataset_directory = value.replace('$run.dataset$', self.run_dataset)
                else:
                    continue
        
        print 'similarity method =', self.similarity_method
    
    def prep_test(self, data):
        if self.dataset_mode == 'cold':
            self.test = {user:item_ratings for user, item_ratings in data.items() if len(data[user]) < 5}
        elif self.dataset_mode == 'all':
            self.test = data.copy()
        self.total_test=sum([len(value) for value in data.viewvalues() ])
            
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
        print 'MAE = {0:.4f}'.format(MAE)
        predictable = len(self.errors)
        all_test = self.total_test
        print 'RC  = {0:d}/{1:d} = {2:2.2f}%'.format(predictable, all_test, float(predictable) / all_test * 100)
        
    def similarity(self, a, b):
        '''
        compute user or item similarity
        '''
        if self.similarity_method == 'pcc':
            sim = sims.pcc(a, b)
        elif self.similarity_method == 'cos':
            sim = sims.cos(a, b)
        else:
            sim = 1.0
        
        return sim
        
                
class TrustAll(AbstractCF):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        AbstractCF.__init__(self)
        print 'Run TrustAll method'
        
    def perform(self, train, test):
        errors = []
        count = 0
        for test_user in test.viewkeys():
            count += 1
            if count % 10 == 0:
                print 'current progress =', count, 'out of', len(test)
            # predict test item's rating
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                a = {item: float(rate) for item, rate in train[test_user].items() if item != test_item}
                
                # find similar users, train, weights
                users = []
                rates = []
                weights = []
                for user in train.viewkeys():
                    if user == test_user: 
                        continue
                    b = {item: float(rate) for item, rate in train[user].items() if (item != test_item) and (test_item in train[user])}
                    if len(b) == 0: 
                        continue
                    sim = self.similarity(a, b)
                    users.append(user)
                    rates.append(train[user][test_item])
                    weights.append(sim)
                
                # prediction
                if len(rates) == 0:
                    continue
                pred = [rate * weight for rate, weight in zip(rates, weights)]
                pred = sum(pred) / len(rates)
                errors.append(abs(truth - pred))
        self.errors = errors      
