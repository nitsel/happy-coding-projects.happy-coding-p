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
from sklearn.cluster import DBSCAN
from graph import Graph, Vertex, Edge
import random
from sklearn.cluster.k_means_ import KMeans

cross_validation = 'cross_validation'
leave_one_out = 'leave_one_out'
resnick_formula = 'resnick_formula'

on = 'on'
off = 'off'

debug = not True
verbose = not True

def load_config():
    config = {}
    with open('cf.conf', 'r') as f:
            for line in f:
                if line.find('=') == -1:
                    continue
                params = line.strip().split('=')
                config[params[0]] = params[1]
    return config

"""
@INPUT:
    R     : a matrix to be factorized, dimension N x M
    U     : an initial matrix of dimension N x K
    V     : an initial matrix of dimension K x M
    K     : the number of latent features
    steps : the maximum number of steps to perform the optimisation
    alpha : the learning rate
    beta  : the regularization parameter
@OUTPUT:
    the final matrices P and Q
"""
def matrix_factorization(R, U, V, K, steps=5000, lrate=0.0002, lam=0.02, tol=0.0001):
    for step in xrange(steps):
        e = 0
        N = len(R)
        M = len(R[0])
        for i in xrange(N):
            for j in xrange(M):
                if R[i][j] > 0:
                    eij = R[i][j] - py.dot(U[i, :], V[:, j])
                    for k in xrange(K):
                        newPik = U[i][k] + lrate * (2 * eij * V[k][j] - lam * U[i][k])
                        newQkj = V[k][j] + lrate * (2 * eij * U[i][k] - lam * V[k][j])
                        
                        U[i][k] = newPik
                        V[k][j] = newQkj
                
                e = e + pow(R[i][j] - py.dot(U[i, :], V[:, j]), 2)    
        
        e += 0.5 * lam * sum([U[i][k] ** 2 for i in range(N) for k in range(K)])
        e += 0.5 * lam * sum([V[k][j] ** 2 for k in range(K) for j in range(M)])
        
        if e < tol: break
        else:
            print 'current progress =', (step + 1), ', errors =', e
    return U, V

class Prediction(object):
    def __init__(self, user, item, pred, truth):
        self.user = user
        self.item = item
        self.pred = pred
        self.truth = truth

class AbstractCF(object):
    '''
    classdocs
    '''
    rating_set = "ratings.txt"
    trust_set = 'trust.txt'
    debug_file = 'results.csv'
    peek_interval = 20
    config = {}

    def __init__(self):
        self.method_id = 'abstract_cf'
    
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
        
    def prep_test(self, train, test=None):
        '''
        prepare test rating dictionary
        
        parameters
        ---------------------
        train: training set
        test: if test=None, generate test ratings from training set; else select ratings from test set according to the properties from the training set
        '''
        cold_len = 5
        heavy_len = 10
        if test is None:
            if self.dataset_mode == 'all':
                return train.copy()
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in train.items() if len(train[user]) < cold_len}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in train.items() if len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
        else:
            if self.dataset_mode == 'all':
                return self.test
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) < cold_len}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
            pass
        
    def execute(self):
        self.config_cf()
        logs.basicConfig(filename=self.debug_file, filemode='a', level=logs.DEBUG, format="%(message)s")
        
        # run multiple times at one shot
        batch_run = self.config['cross.validation.batch'] == on or int(self.config['kmeans.init']) > 1
        if batch_run: 
            self.multiple_run()
        else:
            self.single_run()
            
    def multiple_run(self):
        if self.config['cross.validation.batch'] == on:
            for i in range(1, 6):
                self.rating_set = 'u' + str(i) + '.base'
                self.test_set = 'u' + str(i) + '.test'
                self.single_run()
        elif int(self.config['kmeans.init']) > 1:
            for i in range(int(self.config['kmeans.init'])):
                self.single_run()
            
    def single_run(self):    
        # import training rating and trust data
        ds = Dataset()
        self.train, self.items = ds.load_ratings(self.dataset_directory + self.rating_set)
        
        trust_file = self.dataset_directory + self.trust_set
        if os.path.exists(trust_file): 
            self.trust = ds.load_trust(trust_file)
        
        # prepare test set
        test_data = ds.load_ratings(self.dataset_directory + self.test_set)  if self.validate_method == cross_validation else None
        self.test = test_data[0]  if test_data is not None else None
        self.test_items = test_data[1]  if test_data is not None else None
        self.test = self.prep_test(self.train, self.test)
        
        # execute recommendations
        self.perform(self.train, self.test)
        
        # collect results
        self.collect_results();
    
    def perform(self, train, test):
        if self.validate_method == cross_validation:
            # self.cross_over(train, test)
            self.cross_over_test_items(train, test)
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
        write_out = self.config['write.results'] == on
        if self.errors:
            MAE = py.mean(self.errors)
            mae = 'MAE = {0:.6f}, '.format(MAE)
            pred_test = len(self.errors)
            total_test = sum([len(value) for value in self.test.viewvalues() ])
            RC = float(pred_test) / total_test * 100
            rc = 'RC = {0:d}/{1:d} = {2:2.2f}%, '.format(pred_test, total_test, RC)
            RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
            rmse = 'RMSE = {0:.6f}'.format(RMSE)
            
            print mae + rc + rmse + ', ' + self.dataset_directory + self.rating_set
            
            if write_out:
                results = '{0},{1},{2:.6f},{3:2.2f}%,{4:.6f},{5}'\
                        .format(self.method_id, self.dataset_mode, MAE, RC, RMSE, self.dataset_directory + self.rating_set)
                logs.debug(results)
        elif self.user_preds:
            precisions = []
            nDCGs = []
            top_n = 5
            covered = 0
            for preds in self.user_preds.viewvalues():
                sorted_preds = sorted(preds, key=lambda x: x.pred, reverse=True)
                sorted_truths = sorted(preds, key=lambda x: x.truth, reverse=True)
                
                predicted = [pred.truth for pred in preds if pred.truth > 0]
                covered += len(predicted)
                
                list_rec = sorted_preds[:top_n] if len(sorted_preds) > top_n else sorted_preds
                list_truth = sorted_truths[:top_n] if len(sorted_truths) > top_n else sorted_truths
                 
                tp = 0
                n_rec = len(list_rec)
                
                DCG = 0
                for i in range(n_rec):
                    truth = list_rec[i].truth
                    if truth > 0:
                        tp += 1
                        rank = i + 1
                        DCG += 1.0 / math.log(rank + 1, 2)
                
                iDCG = 0
                for i in range(len(list_truth)):
                    truth = list_truth[i].truth
                    rank = i + 1
                    iDCG += (1.0 / math.log(rank + 1, 2) if truth > 0 else 0)
                
                if iDCG>0:
                    nDCG = DCG / iDCG
                    nDCGs.append(nDCG)
                        
                precision = float(tp) / n_rec
                precisions.append(precision)
            
            print 'P@{0:d} = {1:f}'.format(top_n, py.average(precisions))
            print 'nDCG@{0:d} = {1:f}'.format(top_n, py.average(nDCGs))
            
            total_test = sum([len(value) for value in self.test.viewvalues() ])
            print 'Coverage = {0:2.2f}%'.format(float(covered) / total_test * 100)
            
            if write_out:
                results = '{0},{1},{2:.6f},{3:6f},{4:2.2f},{5}'\
                        .format(self.method_id, self.dataset_mode, py.average(precisions), 
                                py.average(nDCGs), float(covered) / total_test * 100, self.dataset_directory + self.rating_set)
                logs.debug(results)
    
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
            r = float(self.config['similarity.wpcc.gamma'])
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
    def __init__(self):
        self.method_id = "ClassicCF"
        
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
            
            # trust information
            # if test_user in self.trust:continue
            
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
        
    def cross_over_test_items(self, train, test):
        # {test_user, {train_user, similarity}}
        sim_dist = {}
        # {train_user, mu}
        mu_dist = {}
        
        count = 0
        # {test_user: [prediction object]}
        user_preds = {}
        
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
            
            a = train[test_user] if test_user in train else {}
            
            # predict test item's rating
            for test_item in self.test_items:
                if test_item in a: continue
                
                truth = test[test_user][test_item] if test_item in test[test_user] else 0.0
                
                mu_a = 0.0
                if self.prediction_method == 'resnick_formula':
                    if not a: continue
                    mu_a = py.mean(a.values(), dtype=py.float64)
                
                # find the ratings of similar users and their weights
                votes = []
                weights = []
                for user in train.viewkeys():
                    if test_item not in train[user]: continue
                    if test_user == user: continue
                    # find or compute user similarity
                    user_sims = sim_dist[test_user] if test_user in sim_dist else {}
                    
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
                
                prediction = Prediction(test_user, test_item, pred, truth)
                predictions = user_preds[test_user] if test_user in user_preds else []
                predictions.append(prediction)
                user_preds[test_user] = predictions
                
        self.user_preds = user_preds 
        self.errors = []
    
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

class ClusterCF(ClassicCF):
    
    def __init__(self):
        self.method_id = 'Abstract Cluster CF'
    
    def calc_user_distances(self, train):
        prefix = self.test_set.rindex('.test')
        prefix = self.test_set[0:prefix]
        self.similarity_file = self.dataset_directory + prefix + '-' + self.similarity_method + '.txt'
        
        if not os.path.exists(self.similarity_file): 
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
        D = []
        with open(self.similarity_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                sim = float(line.split()[2])
                dist = sim if self.similarity_method == 'euclidean' else 1 - sim
                
                D.append(dist)
        return distance.squareform(D)
    
    def perform(self, train, test):
        pass

class Trusties(ClusterCF):
    '''
    Trusties prototype for DBSCAN clustering method-based CF
    '''
    def __init__(self):
        self.method_id = 'Trusties'
        
    def kmeans(self, train):
        '''
        kmeans the training users
        '''
        D = self.calc_user_distances(train)
        
        ''' recommended settings for FilmTrust
        
        wpcc (r=10):
        
        eps=0.30, minpts = 5, clusters: 5, clustered points: 754, core points: 460 [noted]
        eps=0.29, minpts = 2, clusters: 5, clustered points: 778, core points: 778
        eps=0.29, minpts = 10, clusters: 5, clustered points: 456, core points: 101
        eps=0.28, minpts = 2, clusters: 6, clustered points: 753, core points: 753
        eps=0.28, minpts = 8, clusters: 5, clustered points: 501, core points: 131
        eps=0.28, minpts = 9, clusters: 5, clustered points: 438, core points: 97
        eps=0.27, minpts = 2, clusters: 8, clustered points: 731, core points: 731
        eps=0.27, minpts = 6, clusters: 6, clustered points: 552, core points: 216
        eps=0.27, minpts = 10, clusters: 7, clustered points: 309, core points: 52
        eps=0.26, minpts = 2, clusters: 15, clustered points: 693, core points: 693
        eps=0.26, minpts = 3, clusters: 6, clustered points: 675, core points: 517 [noted]
        eps=0.26, minpts = 4, clusters: 7, clustered points: 629, core points: 371
        eps=0.26, minpts = 5, clusters: 5, clustered points: 554, core points: 258
        eps=0.26, minpts = 6, clusters: 7, clustered points: 489, core points: 171
        eps=0.26, minpts = 7, clusters: 5, clustered points: 417, core points: 113
        eps=0.26, minpts = 8, clusters: 6, clustered points: 356, core points: 79
        eps=0.26, minpts = 9, clusters: 6, clustered points: 298, core points: 54
        eps=0.26, minpts = 10, clusters: 7, clustered points: 242, core points: 36
        eps=0.25, minpts = 2, clusters: 20, clustered points: 652, core points: 652
        eps=0.25, minpts = 3, clusters: 7, clustered points: 626, core points: 470 [noted]
        eps=0.25, minpts = 4, clusters: 7, clustered points: 573, core points: 315
        eps=0.25, minpts = 5, clusters: 7, clustered points: 488, core points: 203
        eps=0.25, minpts = 6, clusters: 7, clustered points: 420, core points: 131
        eps=0.25, minpts = 8, clusters: 6, clustered points: 284, core points: 55
        eps=0.24, minpts = 2, clusters: 27, clustered points: 628, core points: 628
        eps=0.24, minpts = 3, clusters: 11, clustered points: 596, core points: 423 [noted]
        eps=0.24, minpts = 4, clusters: 8, clustered points: 513, core points: 258
        eps=0.24, minpts = 5, clusters: 12, clustered points: 425, core points: 156
        eps=0.24, minpts = 6, clusters: 6, clustered points: 356, core points: 105
        eps=0.24, minpts = 9, clusters: 5, clustered points: 142, core points: 19
        eps=0.23, minpts = 2, clusters: 36, clustered points: 590, core points: 590
        eps=0.23, minpts = 3, clusters: 12, clustered points: 542, core points: 364
        eps=0.23, minpts = 4, clusters: 14, clustered points: 456, core points: 212
        eps=0.23, minpts = 5, clusters: 10, clustered points: 354, core points: 123
        eps=0.23, minpts = 6, clusters: 9, clustered points: 279, core points: 77
        eps=0.23, minpts = 8, clusters: 5, clustered points: 133, core points: 21
        eps=0.22, minpts = 2, clusters: 48, clustered points: 544, core points: 544
        eps=0.22, minpts = 3, clusters: 13, clustered points: 474, core points: 296
        eps=0.22, minpts = 4, clusters: 15, clustered points: 381, core points: 161
        eps=0.22, minpts = 5, clusters: 16, clustered points: 288, core points: 89
        eps=0.22, minpts = 6, clusters: 7, clustered points: 206, core points: 53
        eps=0.22, minpts = 7, clusters: 8, clustered points: 133, core points: 22
        eps=0.22, minpts = 8, clusters: 6, clustered points: 86, core points: 10
        eps=0.21, minpts = 2, clusters: 57, clustered points: 504, core points: 504
        eps=0.21, minpts = 3, clusters: 17, clustered points: 424, core points: 253
        eps=0.21, minpts = 4, clusters: 20, clustered points: 325, core points: 133
        eps=0.21, minpts = 5, clusters: 14, clustered points: 230, core points: 70
        eps=0.21, minpts = 6, clusters: 10, clustered points: 146, core points: 32
        eps=0.21, minpts = 7, clusters: 8, clustered points: 90, core points: 12
        eps=0.20, minpts = 2, clusters: 63, clustered points: 465, core points: 465
        eps=0.20, minpts = 3, clusters: 20, clustered points: 379, core points: 218
        eps=0.20, minpts = 4, clusters: 23, clustered points: 277, core points: 102
        eps=0.20, minpts = 5, clusters: 16, clustered points: 189, core points: 52
        eps=0.20, minpts = 6, clusters: 10, clustered points: 108, core points: 20
        eps=0.20, minpts = 7, clusters: 5, clustered points: 57, core points: 8
        eps=0.19, minpts = 2, clusters: 71, clustered points: 409, core points: 409
        eps=0.19, minpts = 3, clusters: 23, clustered points: 313, core points: 171
        eps=0.19, minpts = 4, clusters: 24, clustered points: 217, core points: 76
        eps=0.19, minpts = 5, clusters: 14, clustered points: 114, core points: 26
        eps=0.19, minpts = 6, clusters: 6, clustered points: 56, core points: 8
        eps=0.18, minpts = 2, clusters: 82, clustered points: 370, core points: 370
        eps=0.18, minpts = 3, clusters: 26, clustered points: 258, core points: 135
        eps=0.18, minpts = 4, clusters: 20, clustered points: 168, core points: 57
        eps=0.18, minpts = 5, clusters: 8, clustered points: 70, core points: 14
        eps=0.18, minpts = 6, clusters: 5, clustered points: 48, core points: 7
        eps=0.17, minpts = 2, clusters: 78, clustered points: 317, core points: 317
        eps=0.17, minpts = 3, clusters: 24, clustered points: 209, core points: 108
        eps=0.17, minpts = 4, clusters: 15, clustered points: 123, core points: 40
        eps=0.17, minpts = 5, clusters: 7, clustered points: 61, core points: 12
        eps=0.16, minpts = 2, clusters: 74, clustered points: 262, core points: 262
        eps=0.16, minpts = 3, clusters: 25, clustered points: 164, core points: 81
        eps=0.16, minpts = 4, clusters: 10, clustered points: 83, core points: 28
        eps=0.15, minpts = 2, clusters: 71, clustered points: 222, core points: 222
        eps=0.15, minpts = 3, clusters: 22, clustered points: 124, core points: 59
        eps=0.15, minpts = 4, clusters: 8, clustered points: 50, core points: 14
        eps=0.14, minpts = 2, clusters: 65, clustered points: 180, core points: 180
        eps=0.14, minpts = 3, clusters: 17, clustered points: 84, core points: 36
        eps=0.13, minpts = 2, clusters: 55, clustered points: 145, core points: 145
        eps=0.13, minpts = 3, clusters: 13, clustered points: 61, core points: 27
        eps=0.13, minpts = 4, clusters: 5, clustered points: 24, core points: 6
        eps=0.12, minpts = 2, clusters: 47, clustered points: 121, core points: 121
        eps=0.12, minpts = 3, clusters: 11, clustered points: 49, core points: 21
        eps=0.11, minpts = 2, clusters: 32, clustered points: 85, core points: 85
        eps=0.11, minpts = 3, clusters: 10, clustered points: 41, core points: 18
        eps=0.10, minpts = 2, clusters: 25, clustered points: 62, core points: 62
        eps=0.10, minpts = 3, clusters: 6, clustered points: 24, core points: 10
        eps=0.09, minpts = 2, clusters: 20, clustered points: 46, core points: 46
        eps=0.08, minpts = 2, clusters: 14, clustered points: 32, core points: 32
        eps=0.07, minpts = 2, clusters: 11, clustered points: 24, core points: 24
        eps=0.06, minpts = 2, clusters: 8, clustered points: 16, core points: 16
        eps=0.05, minpts = 2, clusters: 7, clustered points: 14, core points: 14
        eps=0.04, minpts = 2, clusters: 6, clustered points: 12, core points: 12
        '''
        is_test = not True
        if is_test:
            eps = 0.30
            n_clusters = 0
            while eps > 0:
                minpts = 2
                while minpts <= 10: 
                    db = DBSCAN(eps=eps, min_samples=minpts, metric='precomputed').fit(D)
                    core_points = db.core_sample_indices_
                    labels = self.db.labels_
        
                    # Number of clusters in labels, ignoring noise if present.
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    
                    n_points = len(labels) - len([x for x in labels if x == -1])
                    
                    if n_clusters >= 5:
                        # raw_input('wait for an input to continue?')
                        result = 'eps={:2.2f}, minpts = {:d}, clusters: {:d}, clustered points: {:d}, core points: {:d}'\
                                .format(eps, minpts, n_clusters, n_points, len(core_points))
                        print result
                        logs.debug(result)
                    
                    minpts += 1
                eps -= 0.01
        else:
            self.eps = 0.26
            self.minpts = 3
            return DBSCAN(eps=self.eps, min_samples=self.minpts, metric='precomputed').fit(D)
                    
    def gen_cluster_graph(self, db):
        labels = db.labels_
        users = self.train.keys()
        
        g = Graph()
        
        # generate vertices
        clusters = list(set(labels))
        if -1 in clusters: clusters.remove(-1)
        vs = {c: Vertex(c) for c in clusters}
        
        # genrate edges
        for c1 in clusters: 
            c1ms = [users[x] for x in py.where(labels == c1)[0]]
            for c2 in clusters: 
                # currently, we ignore the circular links
                if c1 == c2: continue
                c2ms = [users[x] for x in py.where(labels == c2)[0]]
                links = {c1m:c2m for c1m in c1ms for c2m in c2ms if c1m in self.trust and c2m in self.trust[c1m]}
                
                if links:
                    edge = Edge(vs[c1], vs[c2])
                    edge.weight = float(len(links))
                    g.add_edge(edge)
        
        if debug: g.print_graph()
        return g.page_rank(d=0.85, normalized=True)
    
    def calc_trust_distance(self, current_depth_users, trustee, depth=0, visited_users=[]):
        ''' calculate the distance between trustor and trustee, if trustee not in the WOT of trustor, return -1, using breath first searching 
        '''
        if not current_depth_users: return -1
        
        depth += 1
        if trustee in current_depth_users: 
            return depth
        else:
            visited_users.extend(current_depth_users)
            
            next_depth_users = []
            trustees = [self.trust[tn].keys() for tn in current_depth_users if tn in self.trust]
            for n in trustees:
                next_depth_users.extend([x for x in n if x not in visited_users])
            
            return self.calc_trust_distance(next_depth_users, trustee, depth, visited_users)
        
    def perform(self, train, test):
        db = self.kmeans(train)
        
        # for test purpose
        if db is None: os.abort()
        
        core_components = db.components_
        core_indices = db.core_sample_indices_
        labels = db.labels_
        
        core_lens = {}
        for index, i in zip(core_indices, range(len(core_components))):
            lens = len(py.where(core_components[i] < self.eps)[0])
            core_lens[index] = lens
        max_len = float(py.max(core_lens.values()))
        cor_len = float(len(core_indices))

        noises = py.where(labels == -1)[0]
        users = train.keys()
        
        prs = self.gen_cluster_graph(db)
        if debug: print prs
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        
        # trust_weight = 1.0
        # core_weight = 1.0
        
        # procedure: trusties
        if pred_method == 'wtrust':
            for test_user in test: 
                user_index = users.index(test_user) if test_user in users else -1
                tns = self.trust[test_user] if test_user in self.trust else {}
                
                for test_item in test[test_user]:

                    preds = []
                    wpred = []
                    cs = -1  # cluster label of active users                  
                    # step 1: prediction from clusters that test_user belongs to
                    if user_index > -1 and user_index not in noises:
                        # predict according to cluster members
                        cs = labels[user_index]  # cl is not -1
                        
                        member_indices = py.where(labels == cs)[0]
                        members = [users[x] for x in member_indices]
                        rates = []
                        weights = []
                        
                        for member, index in zip(members, member_indices):
                            if test_item in train[member]:
                                u = train[test_user]
                                v = train[member]
                                sim = self.similarity(u, v)
                                # if py.isnan(sim): sim = 0
                                if py.isnan(sim) or sim <= self.similarity_threashold: continue
                                
                                # I think merging two many weights is not a good thing
                                # we should only use sim or member weights 
                                rates.append(train[member][test_item])
                                '''trust = trust_weight if member in tns else 0.0
                                weight = core_weight if index in core_indices else 1.0
                                weights.append(weight * (1 + sim) + trust)'''
                                weights.append(sim)                            
                        if not rates: continue
                        pred = py.average(rates, weights=weights)
                        
                        preds.append(pred)
                        
                        weight_cluster = 0 if cs not in prs else (prs[str(cs)] if cs > -1 else 0)
                        wpred.append(1.0 + weight_cluster)
                
                    # step 2: prediction from the clusters that trusted neighbors belong to
                    if not preds and tns and not False:
                        
                        # print user_index in noises # true
                        # find tns' clusters
                        tn_indices = [users.index(tn) for tn in tns if tn in users]
                        tns_cs = {users[index]: labels[index] for index in tn_indices}
                        
                        cls = list(set(tns_cs.values()))
                        for cl in list(set(labels)): 
                            pred = 0.0
                            # if cl == -1: continue
                            if cl != -1 and cl == cs:continue
                            if cl < -1: 
                                # nns = [key for key, val in tns_cs.items() if val == -1 and test_item in train[key]]
                                nns = [users[x] for x in py.where(labels == cl)[0]]
                                if not nns: continue
                                
                                rates = [train[nn][test_item] for nn in nns if test_item in train[nn]]
                                if not rates: continue
                                pred = py.average(rates)
                            else: 
                                member_indices = py.where(labels == cl)[0]
                                members = [users[x] for x in member_indices]
                                rates = []
                                weights = []
                                for member, index in zip(members, member_indices):
                                    if test_item in train[member]:
                                        rates.append(train[member][test_item])
                                        
                                        dist = self.calc_trust_distance(tns, member)
                                        if dist > 1: print dist
                                        trust = 1.0 / dist if dist > 0 else 0.0
                                        
                                        # trust2 = trust_weight if member in tns else 0.0
                                        # if trust!= trust2: print trust, trust2
                                        # weight = core_weight if index in core_indices else 1.0
                                        weight = core_lens[index] / max_len if index in core_indices else 0.0
                                        weights.append((1 + weight) * (1 + trust))                            
                                if not rates: continue
                                pred = py.average(rates, weights=weights)
                            
                            weight_cluster = 0.0 
                            if cl == -1: weight_cluster = 1.0
                            elif str(cl) in prs: weight_cluster = prs[str(cl)]
                            else: weight_cluster = len(member) / cor_len
                            
                            weight_trust = tns_cs.values().count(cl) / float(len(tns_cs)) if cl in cls else 0
                            w = weight_cluster + weight_trust
                            
                            if w == 0:continue
                            
                            wpred.append(w)
                            preds.append(pred)
                    
                    # step 3: for users without any trusted neighbors and do not have any predictions from the cluster that he blongs to 
                    if not tns and not preds and not False:
                        for cl in list(set(labels)):
                            # if cl == -1: continue
                            members_indices = py.where(labels == cl)[0]
                            members = [users[x] for x in members_indices]
                            
                            rates = []
                            weights = []
                            for member, index in zip(members, members_indices):
                                if test_item in train[member]:
                                    ''' for these users: core points are much further away from the active users, 
                                        hence we should put more weights on the border users than the core users, 
                                        and the (noise) cluster weight should be the most
                                    '''
                                    # weight = (1 - core_lens[index] / max_len) if index in core_indices else 1.0
                                    weight = 0.01 if index in core_indices else 1.0
                                    rates.append(train[member][test_item])  
                                    weights.append(weight) 
                            if not rates: continue
                            pred = py.average(rates, weights=weights)
                            preds.append(pred)
                            # wpred.append(1.0 if cl == -1 else len(member) / cor_len)
                            wpred.append(prs[str(cl)] if str(cl) in prs else 1.0)
                        if not preds: continue
                        pred = py.average(preds, weights=wpred)
                        error = abs(pred - test[test_user][test_item])
                        errors.append(error)
                    else:                                                          
                    # step 3: final predictions aggregated from different clusters
                        if not preds: continue
                        pred = py.average(preds, weights=wpred)
                        error = abs(pred - test[test_user][test_item])
                        errors.append(error)
            self.errors = errors
            return 
        
        # procedure: plain clustering methods
        for test_user in test:
            user_index = users.index(test_user) if test_user in users else -1
            # not found in train
            if user_index == -1: continue
            # noise user
            if user_index in noises: continue
            for test_item in test[test_user]:
                member_indices = py.where(labels == labels[user_index])[0]
                members = [users[x] for x in member_indices]
                
                pred = 0
                if pred_method == 'mean':
                    rates = [train[m][test_item] for m in members if test_item in train[m]]
                    if not rates: continue
                    pred = py.mean(rates)
                    
                elif pred_method == 'wmean':
                    rates = []
                    weights = []
                    for member, index in zip(members, member_indices):
                        if test_item in train[member]:
                            rates.append(train[member][test_item])
                            weight = 5.0 if index in core_indices else 1.0
                            weights.append(weight)                            
                    if not rates: continue
                    pred = py.average(rates, weights=weights)
                
                elif pred_method == 'wcf':
                    rates = []
                    weights = []
                    for member, index in zip(members, member_indices):
                        if test_item in train[member]:
                            u = train[test_user]
                            v = train[member]
                            weight = self.similarity(u, v)
                            
                            if py.isnan(weight) or weight <= self.similarity_threashold: continue
                            rates.append(train[member][test_item])
                            weights.append(weight)                            
                    if not rates: continue
                    pred = py.average(rates, weights=weights)
                
                elif pred_method == 'wmcf':
                    rates = []
                    weights = []
                    for member, index in zip(members, member_indices):
                        if test_item in train[member]:
                            u = train[test_user]
                            v = train[member]
                            weight = self.similarity(u, v)
                            
                            if py.isnan(weight) or weight <= self.similarity_threashold: continue
                            rates.append(train[member][test_item])
                            
                            w = 5.0 if index in core_indices else 1.0
                            weights.append(w + weight)                            
                    if not rates: continue
                    pred = py.average(rates, weights=weights)
                    
                error = abs(pred - test[test_user][test_item])
                errors.append(error)
        self.errors = errors

class KmeansCF(AbstractCF):
    def __init__(self):
        self.method_id = 'kmeans_cf'

    def cluster_users(self, train):
        # D = self.calc_user_distances(train)
        verbose = True
        n_clusters = 7
        
        items = self.items.keys()
        users = train.keys()
        
        X = []
        for user in users:
            xi = []
            for item in items:
                val = train[user][item] if item in train[user] else 0
                xi.append(val)
            X.append(xi)
        k_means = KMeans(init='k-means++', n_clusters=n_clusters, n_init=10)
        k_means.fit(X)
        k_means_labels = k_means.labels_
        k_means_cluster_centers = k_means.cluster_centers_
        k_means_labels_unique = py.unique(k_means_labels)
        
        if verbose:
            print 'number of cluster centers =', k_means_cluster_centers
            print 'number of lables =', k_means_labels_unique
            print 'stop here'
        
    def kmeans(self, train, n_clusters):
        ''' A good reference: http://home.dei.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html
        '''
        last_errors = 0
        tol = 0.0000001
        
        # items = self.items.keys()
        users = train.keys()
        
        # random.seed(100)
        
        # initial k-means clusters
        centroids_indices = random.sample(range(len(users)), n_clusters)
        centroids_users = [users[index] for index in centroids_indices]
        centroids = {index: train[user] for index, user in zip(range(len(centroids_users)), centroids_users)}
        clusters = {}
        
        iteration = 100
        for i in range(iteration):
            clustered_users = []
            clusters = {}
            
            # cluster users
            for user in users:
                if user in clustered_users: continue
                
                min_dist = py.Infinity
                cluster_index = -1
                for c_index, centroid in centroids.viewitems():
                    dist = self.similarity(train[user], centroid) if self.similarity_method == 'euclidean' else 1 - self.similarity(train[user], centroid)
                    # u, v = self.pairs(train[user], centroid)
                    # we need to divide by len(u) in order to compare distances from different dimentional spaces
                    # dist = distance.euclidean(u, v) / len(u) if len(u) > 0 else py.Infinity 
                    if dist < min_dist:
                        min_dist = dist
                        cluster_index = c_index
                # if cluster_index == -1: this user is not able to be clustered, usually due to 
                # that this user only rated one item and this item is only rated by this user or few users,
                # which make this user is not able to compute distance with others
                cluster_members = clusters[cluster_index] if cluster_index in clusters else []
                cluster_members.append(user)
                clusters[cluster_index] = cluster_members
                
                clustered_users.append(user)
            
            # recompute cluster centroids
            new_centroids = {}
            for cluster_index, cluster_members in clusters.viewitems():
                if cluster_index == -1: continue
                
                item_ratings = {}
                for cluster_member in cluster_members:
                    for item in train[cluster_member]:
                        ratings = item_ratings[item] if item in item_ratings else []
                        ratings.append(train[cluster_member][item])
                        item_ratings[item] = ratings
                item_means = {item: py.mean(ratings) for item, ratings in item_ratings.items()}
                new_centroids[cluster_index] = item_means
            
            # compute the errors of centroids
            errors = 0.0
            for index, centroid in centroids.viewitems():
                if index not in new_centroids: 
                    # print 'centroids cannot be updated'
                    # new_centroids[index] = centroids[index]
                    return 
                new_centroid = new_centroids[index]
                error = [(centroid[item] - new_centroid[item]) ** 2 for item in new_centroid if item in centroid ]
                errors += sum(error)
            
            if verbose: 
                print 'iteration', (i + 1),
                print 'errors =', errors
            
            if (last_errors - errors) ** 2 < tol or i >= iteration - 1:
                break
            else:
                last_errors = errors
                centroids = new_centroids
        if verbose:
            print 'centroids:', centroids
        
        return centroids, clusters
    
    def cross_over_test_items(self, train, test):
        n_clusters = int(self.config['kmeans.clusters'])
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        print 'finished clustering'
        
        sim_dist = {}
        mu_dist = {}
        count = 0
        # {test_user: [prediction object]}
        user_preds = {}
        
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
            
            # identity the clusters of this test_user
            cluster = -1
            members = []
            for cluster_index, cluster_members in clusters.viewitems():
                if test_user in cluster_members:
                    cluster = cluster_index
                    members = [member for member in cluster_members if member != test_user]
                    break
            if cluster == -1:
                if verbose: 
                    print 'cannot identify the cluster for user:', test_user
                continue
            
            a = train[test_user] if test_user in train else {}
            
            # predict test item's rating
            for test_item in self.test_items:
                if test_item in a: continue
                
                truth = test[test_user][test_item] if test_item in test[test_user] else 0.0
                
                mu_a = 0.0
                if self.prediction_method == 'resnick_formula':
                    if not a: continue
                    mu_a = py.mean(a.values(), dtype=py.float64)
                    
                # find the ratings of similar users and their weights
                votes = []
                weights = []
                for user in members:
                    if test_item not in train[user]: continue
                    if test_user == user: continue
                    # find or compute user similarity
                    user_sims = sim_dist[test_user] if test_user in sim_dist else {}
                    
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
                
                prediction = Prediction(test_user, test_item, pred, truth)
                predictions = user_preds[test_user] if test_user in user_preds else []
                predictions.append(prediction)
                user_preds[test_user] = predictions
                
        self.user_preds = user_preds 
        self.errors = []
        
    def cross_over(self, train, test):
        n_clusters = int(self.config['kmeans.clusters'])
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        for test_user in test:
            # identity the clusters of this test_user
            cluster = -1
            members = []
            for cluster_index, cluster_members in clusters.viewitems():
                if test_user in cluster_members:
                    cluster = cluster_index
                    members = [member for member in cluster_members if member != test_user]
                    break
            if cluster == -1:
                if verbose: 
                    print 'cannot identify the cluster for user:', test_user
                continue
            
            for test_item in test[test_user]:
                
                if pred_method == 'mean':
                    rates = [train[member][test_item] for member in members if test_item in train[member]]
                    if not rates: continue
                    pred = py.mean(rates)
                
                elif pred_method == 'wcf':
                    rates = []
                    weights = []
                    for member in members:
                        if test_item in train[member]:
                            u = train[test_user]
                            v = train[member]
                            weight = self.similarity(u, v)
                            
                            if py.isnan(weight) or weight <= self.similarity_threashold: continue
                            
                            rates.append(train[member][test_item])
                            weights.append(weight)                            
                    if not rates: continue
                    pred = py.average(rates, weights=weights)
                
                errors.append(abs(pred - test[test_user][test_item]))
        self.errors = errors

class MF(AbstractCF):
    
    def __init__(self):
        self.method_id = 'Matrix Factorization'
    
    def training(self, train):
        users = train.keys()
        items = self.items.keys()
        features = 50 
        lrate = 0.005
        lam = 0.02
        steps = 5000
        tol = 0.0001
        
        rows = len(users)
        cols = len(items)
        
        R = []  # rating matrix
        for i in range(rows):
            user = users[i]
            r = []
            for j in range(cols):
                item = items[j]
                r.append(train[user][item] if item in train[user] else 0)
            R.append(r)
            
        R = py.array(R)
        
        U = py.random.rand(rows, features)
        V = py.random.rand(features, cols)
        
        nU, nV = matrix_factorization(R, U, V, features,
                                      steps=steps, lrate=lrate,
                                      lam=lam, tol=tol)
        return py.dot(nU, nV)
    
    def cross_over_test_items(self, train, test):
        R = self.training(train)
        
        users = train.keys()
        items = self.items.keys()
        
        count = 0
        # {test_user: [prediction object]}
        user_preds = {}
        
        for test_user in test.viewkeys():
            count += 1
            if count % self.peek_interval == 0:
                print 'current progress =', count, 'out of', len(test)
            
            a = train[test_user] if test_user in train else {}
            
            # predict test item's rating
            for test_item in self.test_items:
                if test_item in a: continue
                
                # truth
                truth = test[test_user][test_item] if test_item in test[test_user] else 0.0
                
                # prediction
                row = users.index(test_user)
                col = items.index(test_item)
                pred = R[row][col]
                
                prediction = Prediction(test_user, test_item, pred, truth)
                predictions = user_preds[test_user] if test_user in user_preds else []
                predictions.append(prediction)
                user_preds[test_user] = predictions
                
        self.user_preds = user_preds 
        self.errors = []

class SlopeOne(AbstractCF):
    
    def __init__(self):
        self.method_id = 'Slope One'
    
    def perform(self, train, test):
        '''http://www.serpentine.com/blog/2006/12/12/collaborative-filtering-made-easy/'''
        pass
    
def main():
    config = load_config()
    AbstractCF.config = config
    
    run_method = config['run.method'].lower()
    if run_method == 'cf':
        ClassicCF().execute()
    elif run_method == 'trusties':
        Trusties().execute()
    elif run_method == 'kmeans':
        KmeansCF().execute()
    elif run_method == 'mf':
        MF().execute()
    else:
        raise ValueError('invalid method to run')

def test_io():
    lp = [0.5, 0.3, 0.6]
    pickle.dump(lp, open('list.txt', 'wb'))
    lp = pickle.load(open('list.txt', 'rb'))
    print lp
    
def test_mf():
    R = [[5, 3, 0, 1],
         [4, 0, 0, 1],
         [1, 1, 0, 5],
         [1, 0, 0, 4],
         [0, 1, 5, 4], ]

    R = py.array(R)

    N = len(R)
    M = len(R[0])
    K = 2

    P = py.random.rand(N, K)
    Q = py.random.rand(K, M)

    nP, nQ = matrix_factorization(R, P, Q, K)
    nR = py.dot(nP, nQ)
    
    print nR
    
if __name__ == '__main__':
    main()
    # test_mf()
