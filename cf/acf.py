'''
Created on Feb 15, 2013

@author: guoguibing
'''
import math, os, pickle
import numpy as py
import logging as logs
import operator
from scipy.spatial import distance
from data import Dataset
from sklearn.cluster.k_means_ import MiniBatchKMeans
from sklearn.cluster import DBSCAN

cross_validation = 'cross_validation'
leave_one_out = 'leave_one_out'
resnick_formula = 'resnick_formula'

def load_config():
    config = {}
    with open('cf.conf', 'r') as f:
            for line in f:
                if line.find('=') == -1:
                    continue
                params = line.strip().split('=')
                config[params[0]] = params[1]
    return config

class AbstractCF(object):
    '''
    classdocs
    '''
    rating_set = "ratings.txt"
    trust_set = 'trust.txt'
    debug_file = 'debug.txt'
    peek_interval = 10
    method_id = ''
    config = {}

    def __init__(self):
        self.config_cf()
        logs.basicConfig(filename=self.debug_file, filemode='w', level=logs.DEBUG, format="%(message)s")
    
    def config_cf(self):
        if not self.config: self.config = load_config()
        
        # some commonly used configurations 
        self.prediction_method = self.config['predicting.method']
        self.similarity_method = self.config['similarity.method'].lower()
        self.similarity_threashold = float(self.config['similarity.threshold'])
        self.validate_method = self.config['validate.method']
        
        self.dataset = self.config['run.dataset']
        self.dataset_mode = self.config['dataset.mode']
        
        self.dataset_directory = self.config['dataset.directory'].replace('$run.dataset$', self.dataset)
        if self.validate_method == cross_validation:
            self.dataset_directory += '5fold/'
            self.rating_set = self.config['train.set']
            self.test_set = self.config['test.set']
        
        self.knn = int(self.config['kNN'])
        
        self.print_config()
        
    def print_config(self):
        print 'Run', self.method_id, 'method'
        print 'prediction method =', self.prediction_method
        print 'similarity method =', self.similarity_method
        print 'similarity threshold =', self.similarity_threashold
        if self.knn > 0: print 'kNN =', self.knn
        # print self.config
        
    def prep_test(self, test):
        '''
        prepare test rating dictionary for leave-one-out method
        '''
        if self.dataset_mode == 'coldUsers':
            self.test = {user:item_ratings for user, item_ratings in test.items() if len(test[user]) < 5}
        elif self.dataset_mode == 'all':
            self.test = test.copy()
        else:
            raise ValueError('invalid test dataset mode')
        
        return self.test
            
    def execute(self):
        ds = Dataset()
        self.train = ds.load_ratings(self.dataset_directory + self.rating_set)
        self.trust = ds.load_trust(self.dataset_directory + self.trust_set)
        self.test = ds.load_ratings(self.dataset_directory + self.test_set) if self.validate_method == cross_validation else self.prep_test(self.train)
        
        self.perform(self.train, self.test)
        self.collect_results();
    
    def perform(self, train, test):
        if self.validate_method == cross_validation:
            self.cross_over(train, test)
        elif self.validate_method == leave_one_out:
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
        pred_test = len(self.errors)
        total_test = sum([len(value) for value in self.test.viewvalues() ])
        rc = 'RC = {0:d}/{1:d} = {2:2.2f}%\t'.format(pred_test, total_test, float(pred_test) / total_test * 100)
        RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
        rmse = 'RMSE = {0:.6f}'.format(RMSE)
        
        result = mae + rc + rmse
        print result
        logs.debug(result)
    
    def pairs(self, a, b):
        vas = []
        vbs = []
        for item in a: 
            if item in b:
                vas.append(a[item])
                vbs.append(b[item])
        return (vas, vbs)
    
    def similarity(self, u, v, paired=False):
        '''
        compute user or item similarity
        '''
        if not paired: u, v = self.pairs(u, v)
        
        if self.similarity_method == 'pcc':
            return 1 - distance.correlation(u, v)
        elif self.similarity_method == 'wpcc':
            n = len(u)
            r = 10.0
            w = n / r if n <= r else 1.0
            return w * (1 - distance.correlation(u, v))
        elif self.similarity_method == 'cos':
            return 1 - distance.cosine(u, v)
        elif self.similarity_method == 'constant':
            return 1.0
        elif self.similarity_method == 'euclidean':
            return distance.euclidean(u, v) / len(u)
        else: 
            raise ValueError('invalid similarity measures')
        
class ClassicCF(AbstractCF):
    '''
    classic collaborative filtering algorithm
    '''
    method_id = "ClassicCF"
    def __init__(self):
        AbstractCF.__init__(self)
        
    def cross_over(self, train, test):
        errors = []
        # {user, {user, similarity}}
        sim_dist = {}
        # {user, mu}
        mu_dist = {}
        
        count = 0
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
                
            # predict test item's rating
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                a = train[test_user] if test_user in train else {}
                if not a: continue
                
                mu_a = 0.0
                if self.prediction_method == 'resnick_formula':
                    if not a: continue
                    mu_a = py.mean(a.values(), dtype=py.float64)
                
                # find the ratings of similar users and their weights
                votes = []
                weights = []
                for user in train.viewkeys():
                    if user == test_user: continue
                    if test_item not in train[user]: continue
                    # find or compute user similarity
                    user_sims = sim_dist[test_user] if test_user in sim_dist else {}
                    
                    weight = 0.0
                    mu_b = 0.0
                    if not user_sims or user not in user_sims: 
                        b = train[user]
                        
                        # weight
                        weight = self.similarity(a, b)
                        user_sims[user] = weight
                        sim_dist[test_user] = user_sims
                        
                        # mu_b
                        mu_b = py.mean(b.values(), dtype=py.float64) if self.prediction_method == resnick_formula else 0.0
                        mu_dist[user] = mu_b
                    else:
                        weight = user_sims[user]
                        mu_b = mu_dist[user]
                    
                    if py.isnan(weight) or weight <= self.similarity_threashold: continue
                    
                    # print user, test_item
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
                if self.prediction_method == resnick_formula:
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
                    
                    mu_b = py.mean(b.values(), dtype=py.float64) if self.prediction_method == resnick_formula else 0.0
                    
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
    method_id = 'Trusties'
    
    def __init__(self):
        ClassicCF.__init__(self)
        self.similarity_file = 'user-' + self.similarity_method + '-corr.txt'
        
    def compute_similarities(self, train):
        keys = train.keys()
        users = len(keys)
        count = 0
        for userA_index in range(users):
            similarities = {}
            count += 1
            if count % 20 == 0:
                print 'current progress =', userA_index + 1, 'out of', users
            for userB_index in range(userA_index + 1, users):
                userA = keys[userA_index]
                userB = keys[userB_index]
                a = train[userA]
                b = train[userB]
                
                sim = self.similarity(a, b)
                similarities[userB_index] = sim
            with open(self.similarity_file, 'a') as f:
                for key, value in similarities.items():
                    f.write(str(userA_index) + ' ' + str(key) + ' ' + str(value) + '\n')  
    
    def cluster(self, train):
        '''
        cluster the training users
        '''
        if not os.path.exists(self.similarity_file):
            self.compute_similarities(train)
        
        D = []
        with open(self.similarity_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                sim = float(line.split()[2])
                dist = sim if self.similarity_method == 'euclidean' else 1 - sim
                
                D.append(dist)
        D = distance.squareform(D)
        # D = D/py.max(D)
        
        # Compute DBSCAN
        cluster_method = 'DBSCAN'
        
        if cluster_method == 'DBSCAN':
            ''' recommended settings for FilmTrust
            
            wpcc (r=10): eps=0.30, minpts=5 => n_clusters=5
            wpcc (r=10): eps=0.25, minpts=5 => n_clusters=7
            wpcc (r=10): eps=0.20, minpts=5 => n_clusters=16
            '''
            eps = 0.5
            n_clusters = 0
            while eps > 0:
                minpts = 5
                while minpts <= 100: 
                    self.db = DBSCAN(eps=eps, min_samples=minpts, metric='precomputed').fit(D)
                    # core_samples = db.core_sample_indices_
                    labels = self.db.labels_
        
                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    if n_clusters >= 5:
                        raw_input('wait for an input to continue?')
                    
                    print 'min_sample = {0:d}, eps={1:2.2f},'.format(minpts, eps),
                    print 'Estimated number of clusters: %d' % n_clusters
                    
                    minpts += 5
                eps -= 0.05
            
            '''eps=0.44
            minpts=75
            self.db = DBSCAN(eps=eps, min_samples=minpts, metric='precomputed').fit(D)
            # core_samples = db.core_sample_indices_
            labels = self.db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            print 'min_sample = {0:d}, eps={1:2.2f},'.format(minpts, eps),
            print 'Estimated number of clusters: %d' % n_clusters'''
                    
        elif cluster_method == 'minibatch_kmeans':
            batch_size = 45
            self.db = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
                      n_init=10, max_no_improvement=10, verbose=0).fit(D)
            labels = self.db.labels_
            n_clusters = len(set(labels))
            print 'Estimated number of clusters: %d' % n_clusters
            
    def perform(self, train, test):
        self.cluster(train)
    
def main():
    config = load_config()
    AbstractCF.config = config
    
    run_method = config['run.method'].lower()
    if run_method == 'classic_cf':
        ClassicCF().execute()
    elif run_method == 'trusties':
        Trusties().execute()
    else:
        raise ValueError('invalid method to run')

def test():
    lp = [0.5, 0.3, 0.6]
    pickle.dump(lp, open('list.txt', 'wb'))
    lp = pickle.load(open('list.txt', 'rb'))
    print lp
    
if __name__ == '__main__':
    main()
