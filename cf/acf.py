'''
Created on Feb 15, 2013

@author: guoguibing
'''
import math, os, pickle, sys, shutil, socket, datetime
import numpy as py
import logging as logs
import operator
from scipy.spatial import distance
from data import Dataset
# from sklearn.cluster import DBSCAN
from graph import Graph, Vertex, Edge
import random, emailer
from scipy import stats
import copy

cross_validation = 'cross_validation'
leave_one_out = 'leave_one_out'
resnick_formula = 'resnick_formula'

on = 'on'
off = 'off'

debug = not True
verbose = True

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
    last_e = 0
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
        
        if (last_e - e) ** 2 < tol: break
        else:
            last_e = e
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
    debug_file = 'results.txt'  # results.csv used for formal results
    peek_interval = 20
    cold_len = 5
    heavy_len = 10
    config = {}
    errors = []
    user_preds = {}
    results = ''

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
        
        if sys.platform.startswith('win32'):
            self.temp_directory = 'D:/Data/' + self.dataset + "/"
        else:
            self.temp_directory = self.config['dataset.temp.directory'].replace('$run.dataset$', self.dataset)
        
        self.knn = int(self.config['kNN'])
        self.trust_len = int(self.config['trust.propagation.len'])
        
        self.print_config()
        
    def print_config(self):
        print 'Run [{0}] method'.format(self.method_id)
        print 'prediction method =', self.prediction_method
        print 'similarity method =', self.similarity_method
        print 'similarity threshold =', self.similarity_threashold
        print 'trust propagation len =', self.trust_len
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
        
    def execute(self):
        self.config_cf()
        
        # clean result file
        open(self.debug_file, 'w').close()
        logs.basicConfig(filename=self.debug_file, filemode='a', level=logs.DEBUG, format="%(message)s")
        
        # run multiple times at one shot
        batch_run = self.config['cross.validation.batch'] == on or int(self.config['kmeans.init']) > 1
        if batch_run: 
            self.multiple_run()
        else:
            self.single_run()
        
        # copy file to back up results
        dst = 'Results/' + self.dataset
        if not os.path.isdir(dst): 
            os.makedirs(dst)
        new_file = self.method_id + '@' + str(datetime.datetime.now()).replace(':', '-') + '.txt'
        shutil.copy2(self.debug_file, dst + '/' + new_file)
        
        # notify me when it is finished
        if self.config['results.email.notification'] == on:
            subject = 'Program is finished @ {0:s}'.format(socket.gethostname())
            emailer.send_email(file=self.debug_file, Subject=subject)
            print 'An email with results has been sent to you.' 
        
    def multiple_run(self):
        if self.config['cross.validation.batch'] == on:
            if self.config['kmeans.clusters'] == 'batch':
                n_clusters = [10 * (i + 1) for i in range(9)]
            else:
                n_clusters = [int(self.config['kmeans.clusters'])]
            
            for n_cluster in n_clusters:
                self.n_clusters = n_cluster
                 
                run_times = int(self.config['kmeans.run.times'])
                if run_times < 1:
                    run_times = 1
                for j in range(run_times):
                    print 'current running time is', j + 1
                    for i in range(1, 6):
                        self.rating_set = 'u' + str(i) + '.base'
                        self.test_set = 'u' + str(i) + '.test'
                        self.single_run()
        elif int(self.config['kmeans.init']) > 1:
            for i in range(int(self.config['kmeans.init'])):
                self.single_run()
    
    def read_trust(self, trust, user):
        if self.trust_len == 1:
            tns = trust[user] if user in trust else []
        else:
            file_path = self.temp_directory + '/MT' + str(self.trust_len) + '/MoleTrust/' + user + '.txt'
            tns = self.ds.read_trust(file_path)
        return tns
            
    def single_run(self):
        self.results = '{0},{1}'.format(self.method_id, self.dataset_mode)
        
        # import training rating and trust data
        ds = Dataset()
        self.data_file = self.dataset_directory + self.rating_set
        self.train, self.items = ds.load_ratings(self.data_file)
        
        trust_file = self.dataset_directory + self.trust_set
        if os.path.exists(trust_file): 
            self.trust = ds.load_trust(trust_file)
        
        # prepare test set
        test_data = ds.load_ratings(self.dataset_directory + self.test_set)  if self.validate_method == cross_validation else None
        self.test = test_data[0]  if test_data is not None else None
        self.test_items = test_data[1]  if test_data is not None else None
        self.test = self.prep_test(self.train, self.test)
        
        self.ds = ds
        
        # execute recommendations
        self.perform(self.train, self.test)
        
        # collect results
        self.collect_results();
    
    def perform(self, train, test):
        if self.validate_method == cross_validation:
            top_n = int(self.config['top.n'])
            if top_n > 0:
                self.cross_over_top_n(train, test)
            else:
                self.cross_over(train, test)
                
        elif self.validate_method == leave_one_out:
            self.leave_one_out(train, test)
            
        else:
            raise ValueError('invalid validation method')
    
    def cross_over(self, train, test):
        pass
    
    def cross_over_top_n(self, train, test):
        pass
    
    def leave_one_out(self, train, test):
        pass
    
    def collect_results(self):
        '''collect and print the final results'''
        
        total_test = sum([len(value) for value in self.test.viewvalues() ])
        if self.user_preds:
            precisions = []
            recalls = []
            nDCGs = []
            APs = []
            RRs = []
            errors = []
            top_n = int(self.config['top.n'])
            covered = 0
            for test_user, preds in self.user_preds.viewitems():
                # complete ranked (by prediction) and ideal list
                sorted_preds = sorted(preds, key=lambda x: x.pred, reverse=True)
                sorted_truths = sorted(preds, key=lambda x: x.truth, reverse=True)
                
                # predictable (recovered) items
                es = [abs(pred.pred - pred.truth) for pred in preds if pred.truth > 0]
                errors.extend(es)
                covered += len(es)
                
                # cut-off at position N
                list_recom = sorted_preds[:top_n]
                list_truth = sorted_truths[:top_n]
                
                n_recom = len(list_recom)
                precisions_at_k = []
                tp = 0
                DCG = 0
                
                first_relevant = False
                RR = 0  # Reciprocal Rank
                for i in range(n_recom):
                    truth = list_recom[i].truth
                    if truth > 0:  # rated as a hit item
                        tp += 1
                        rank = i + 1
                        DCG += 1.0 / math.log(rank + 1, 2)
                        
                        precision_at_k = float(tp) / rank
                        precisions_at_k.append(precision_at_k)
                        
                        if not first_relevant:
                            RR = 1.0 / rank
                            first_relevant = True
                            
                precisions.append(float(tp) / n_recom)
                recalls.append(float(tp) / len(self.test[test_user]))
                RRs.append(RR)
                
                # average prediction
                AP = py.mean(precisions_at_k) if precisions_at_k else 0
                APs.append(AP)
                
                iDCG = 0
                for i in range(len(list_truth)):
                    truth = list_truth[i].truth
                    rank = i + 1
                    iDCG += (1.0 / math.log(rank + 1, 2) if truth > 0 else 0)
                
                if iDCG > 0:
                    nDCG = DCG / iDCG
                    nDCGs.append(nDCG)
                        
            self.errors = errors
            
            Precision_at_n = py.mean(precisions)
            Recall_at_n = py.mean(recalls)
            F1_at_n = stats.hmean([Precision_at_n, Recall_at_n])
            nDCG_at_n = py.mean(nDCGs)
            MAP_at_n = py.mean(APs)
            MRR_at_n = py.mean(RRs)
            RC = float(covered) / total_test * 100
            
            print 'P@{0:d} = {1:f},'.format(top_n, Precision_at_n),
            print 'R@{0:d} = {1:f},'.format(top_n, Recall_at_n),
            print 'F1@{0:d} = {1:f},'.format(top_n, F1_at_n),
            print 'NDCG@{0:d} = {1:f},'.format(top_n, nDCG_at_n),
            print 'MAP@{0:d} = {1:f},'.format(top_n, MAP_at_n),
            print 'MRR@{0:d} = {1:f},'.format(top_n, MRR_at_n),
            print 'RC = {0:2.2f}%'.format(RC)
            
            self.results += ',{0:.6f},{1:.6f},{2:.6f},{3:.6f},{4:.6f},{5:.6f}'\
                .format(Precision_at_n, Recall_at_n, F1_at_n, nDCG_at_n, MAP_at_n, MRR_at_n)
                
        if self.errors:
            MAE = py.mean(self.errors)
            mae = 'MAE = {0:.6f}, '.format(MAE)
           
            RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
            rmse = 'RMSE = {0:.6f}, '.format(RMSE)
            
            pred_test = len(self.errors)
            RC = float(pred_test) / total_test * 100
            rc = 'RC = {0:2.2f}%, '.format(RC)
            
            print mae + rmse + rc + self.data_file
            self.results += ',{0:.6f},{1:.6f},{2:2.2f}%'.format(MAE, RMSE, RC)
        
        self.results += ',' + self.data_file
        logs.debug(self.results)
    
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
        
    def cross_over_top_n(self, train, test):
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
                if self.knn > 0 and len(weights) > self.knn:
                    sorted_ws = sorted(enumerate(weights), reverse=True, key=operator.itemgetter(1))[:self.knn]
                    indeces = [item[0] for item in sorted_ws]
                    weights = [item[1] for item in sorted_ws]
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

class CCF(ClassicCF):
    
    def __init__(self):
        self.method_id = 'Cold Classic CF'
    
    def prep_test(self, train, test=None):
        cold_len = 5
        heavy_len = 10
        if test is not None:
            # self.train2 = train
            # self.train = {user:item_ratings for user, item_ratings in train.items() if len(train[user]) >= cold_len}
            
            if self.dataset_mode == 'all':
                return self.test
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in test.items() if (user not in train or len(train[user]) < cold_len) and user in self.trust}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
    
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
    
class Trusties(ClusterCF):
    '''
    Trusties: the implementation of trust-based clustering methods
    '''
    def __init__(self):
        self.method_id = 'Trusties'
        
    '''def kmeans(self, train):
        
        kmeans the training users
        
        D = self.calc_user_distances(train)
        
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
            return DBSCAN(eps=self.eps, min_samples=self.minpts, metric='precomputed').fit(D)'''
                    
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
        
    def cross_over(self, train, test):
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

class KmedoidsCF(AbstractCF):
    
    def __init__(self):
        self.cluster_by = self.config['kmedoids.cluster.by']
        self.method_id = 'Kmedoids'
        
        self.alpha = float(self.config['kmedoids.trust.alpha'])
        self.trust_len = int(self.config['trust.propagation.len'])
        
        self.max_depth = int(self.config['kmedoids.trust.max_depth'])
        
    def prep_test(self, train, test=None):
        
        self.train = {user: train[user] for user in train if len(train[user]) >= self.cold_len or (user in self.trust and len(self.trust[user]) >= 5)} 
        
        return {user: test[user] for user in test if user in self.train}
    
    def trust_distance(self, source, trustee, max_depth=2):
        ''' calculate the distance between trustor and trustee, if trustee not in the WOT of trustor, return -1, using breath first searching 
        '''
        visited_users = []
        visited_users.append(source)
        depth_users = self.trust[source] if source in self.trust else []
        if not depth_users: return -1
        
        depth = 0
        while depth < max_depth:
            depth += 1
            if trustee in depth_users: return depth
            visited_users.extend(depth_users)
            
            next_depth_users = []            
            trustees = [self.trust[tn].keys() for tn in depth_users if tn in self.trust]
            for n in trustees:
                next_depth_users.extend([x for x in n if x not in visited_users])
            
            depth_users = next_depth_users[:]
            if not depth_users: break
            
        return -1   
        
    def user_dists(self, train):
        rating_dist = {} 
        trust_dist = {}
        users = train.keys()
        
        prefix = self.test_set[0:self.test_set.find('.')]
        rating_dist_file = 'rating_dist_' + prefix + '.txt'
        trust_dist_file = 'trust_dist_' + prefix + '.txt'
        
        if os.path.exists(rating_dist_file) and os.path.exists(trust_dist_file):
            print 'reading rating distance data from', rating_dist_file
            with open(rating_dist_file, 'r') as f:
                for line in f: 
                    userA, userB, dist = line.strip().split(',')
                    
                    user_dist = rating_dist[userA] if userA in rating_dist else {}
                    user_dist[userB] = float(dist)
                    rating_dist[userA] = user_dist
            
            print 'reading trust distance data from', trust_dist_file
            with open(trust_dist_file, 'r') as f:
                for line in f: 
                    userA, userB, dist = line.strip().split(',')
                    
                    user_dist = trust_dist[userA] if userA in trust_dist else {}
                    user_dist[userB] = float(dist)
                    trust_dist[userA] = user_dist
            
            self.rating_dist = rating_dist
            self.trust_dist = trust_dist
            
            return rating_dist, trust_dist        
                    
        for u in users:
            print len(rating_dist) + 1, 'out of', len(users)
            urs = train[u]
            utn = self.trust[u] if u in self.trust else {}
            for v in users:
                if u == v: continue
                
                # reduce repeating computation since dist(u, v)=dist(v, u)
                if v in rating_dist or v in trust_dist:
                    computed = False
                    
                    if v in rating_dist and u in rating_dist[v]:
                        sim_dist = rating_dist[v][u]
                        
                        user_dist = rating_dist[u] if u in rating_dist else {}
                        user_dist[v] = sim_dist
                        rating_dist[u] = user_dist
                        computed = True
                    
                    if v in trust_dist and u in trust_dist[v]:
                        tsim_dist = trust_dist[v][u]
                        
                        social_dist = trust_dist[u] if u in trust_dist else {}
                        social_dist[v] = tsim_dist
                        trust_dist[u] = social_dist
                        computed = True
                    
                    if computed: continue
                
                # compute from scratch
                vrs = train[v]
                vtn = self.trust[v] if v in self.trust else {}
                
                # rating similarity
                sim = self.similarity(urs, vrs)
                if not py.isnan(sim):
                    user_dist = rating_dist[u] if u in rating_dist else {}
                    user_dist[v] = 1 - sim
                    rating_dist[u] = user_dist
                
                # compute social similarity in 3 parts
                if not utn: continue 
                if not vtn: continue               
                
                # part 1: direct trust
                d = self.trust_distance(u, v, self.max_depth)
                # if d > 0: print u, v, d
                trust = 1.0 / d if d > 0 else 0                
                
                # part 2: overlapping trusted neighbors
                cnt = 0
                cnt_all = 0
                for tn in vtn:
                    if tn in utn:
                        cnt += 1
                cnt_all = len(utn) + len(vtn) - cnt
                jaccard = float(cnt) / cnt_all
                
                # part 3: overall social distance
                tsim = self.alpha * trust + (1 - self.alpha) * jaccard
                '''
                Note:
                    if cannot connect in the trust network, and no commonly overlapping, 
                    then relationship cannot be determined, because the real situation is unknown, could be close
                However, 
                    if we don't do this, the trust-based approach will become very unreliable in terms of MAE and RC, according to experiments.
                Hence, 
                    overall, I decided to continue allow tsim=0''' 
                # if tsim == 0.0: continue
                
                social_dist = trust_dist[u] if u in trust_dist else {}
                social_dist[v] = 1 - tsim
                trust_dist[u] = social_dist
        
        self.rating_dist = rating_dist
        self.trust_dist = trust_dist 
        
        # output to the disk to save running time
        with open(rating_dist_file, 'w') as f: 
            for userA, user_dist in rating_dist.viewitems():
                for userB, dist in user_dist.viewitems():
                    line = userA + ',' + userB + ',' + str(dist) + '\n'
                    f.write(line)
        
        with open(trust_dist_file, 'w') as f: 
            for userA, user_dist in trust_dist.viewitems():
                for userB, dist in user_dist.viewitems():
                    line = userA + ',' + userB + ',' + str(dist) + '\n'
                    f.write(line)
                    
        return rating_dist, trust_dist
    
    def Kmedoids(self, train, K):
        '''K-medoids clustering methods: http://en.wikipedia.org/wiki/K-medoids'''

        print 'Start to cluster users by K-medoids...'
        tol = 0.00001
        
        # cluster by sim, trust, or both
        cluster_by = self.cluster_by
        users = train.keys()
        
        '''Pre-processing: pre-compute user distance (rating similarity, social similarity) matrix '''
        rating_dist, trust_dist = self.user_dists(train)
        
        # random.seed(100)
        
        '''Initialization: initial k medoids: randomly; implement other initialization methods -- ranked medoids'''
        medoid_indices = random.sample(range(len(users)), K)
        medoids = {k:users[index] for k, index in zip(range(K), medoid_indices)}
        
        iteration = 100
        deltas = {}
        deltas_stable_times = 0
        
        for i in range(iteration):
            
            '''Assignment step: associate each data point o to the closest medoid'''
            cluster_errors = {}
            clusters = {}
            for user in train:
                min_dist = py.inf
                cluster_id = -1
                for c_id, medoid in medoids.viewitems():
                    
                    # the medoid point itself
                    if user == medoid:
                        min_dist = 0
                        cluster_id = c_id
                        break
                    
                    # other data points
                    if cluster_by == 'sim':
                        dist = rating_dist[user][medoid] if user in rating_dist and medoid in rating_dist[user] else py.nan
    
                    elif cluster_by == 'trust':
                        dist = trust_dist[user][medoid] if user in trust_dist and medoid in trust_dist[user] else py.nan
                    
                    if not py.isnan(dist):
                        if min_dist > dist:
                            min_dist = dist
                            cluster_id = c_id

                # if cluster_by == 'trust' and min_dist == 1.0: cluster_id = -1                
                if cluster_id == -1: continue
                
                user_cluster = clusters[cluster_id] if cluster_id in clusters else []
                user_cluster.append(user)
                clusters[cluster_id] = user_cluster
                
                errors = cluster_errors[cluster_id] if cluster_id in cluster_errors else 0
                errors += min_dist
                cluster_errors[cluster_id] = errors
            
            
            ''' Update step: for each medoid m and each data point o associated to m, 
                swap m and o and compute the total cost of the configuration, 
                that is, the average dissimilarity of o to all the data points associated to m. 
                Select the medoid o with the lowest cost of the configuration.'''
            
            best_medoids = copy.deepcopy(medoids)
            delta = 0
            for cluster_id, cluster_ms in clusters.viewitems():
                medoid = medoids[cluster_id]
                last_errors = cluster_errors[cluster_id]
                best_errors = last_errors
                best_errors_medoid = medoid
                
                for m in cluster_ms:
                    if m == medoid: continue
                    
                    new_medoid = m
                    new_errors = 0
                    for user in cluster_ms:
                        if user == new_medoid: continue
                        
                        if cluster_by == 'sim':
                            dist = rating_dist[user][new_medoid] if user in rating_dist and new_medoid in rating_dist[user] else py.nan
        
                        elif cluster_by == 'trust':
                            dist = trust_dist[user][new_medoid] if user in trust_dist and new_medoid in trust_dist[user] else py.nan
                        
                        if not py.isnan(dist):
                            new_errors += dist
                    
                    if new_errors < best_errors:
                        best_errors = new_errors
                        best_errors_medoid = new_medoid
                
                delta += (last_errors - best_errors)
                best_medoids[cluster_id] = best_errors_medoid
            
            medoids = copy.deepcopy(best_medoids)
            
            deltas[i % 4] = delta
            
            if verbose: print 'iteration', (i + 1), 'delta =', delta
            if delta < tol: 
                break
            if len(deltas) == 4:
                odd_delta = deltas[0] - deltas[2]
                even_delta = deltas[1] - deltas[3]
                if odd_delta == 0 and even_delta == 0:
                    deltas_stable_times += 1
                if deltas_stable_times >= 4 and delta == min(deltas.values()):
                    break
            
        print 'Done!'
        return clusters
    
    def cross_over(self, train, test):
        
        clusters = self.Kmedoids(train, self.n_clusters)
        
        self.results += ',' + self.cluster_by + ',' + str(self.n_clusters) + (',' + str(self.alpha) if self.cluster_by == 'trust' else '')
        
        errors = []
        for test_user in test:
            if test_user not in train: continue     
            
            cluster_ms = []
            for c_ms in clusters.viewvalues():
                if test_user in c_ms:
                    cluster_ms = [c_m for c_m in c_ms if c_m != test_user]
                    break
                
            if not cluster_ms:
                # print 'cannot find the cluster for test user', test_user
                continue
                
            for test_item in test[test_user]:
                candidates = [m for m in cluster_ms if test_item in train[m]]
                
                rates = []
                ws = []
                for v in candidates:
                    '''no matter clustered by sim or trust, for prediction, all based on similarity value'''
                    sim = 1 - self.rating_dist[test_user][v] if test_user in self.rating_dist and v in self.rating_dist[test_user] else py.nan
                    
                    if not py.isnan(sim) and sim > 0.0:
                        rates.append(train[v][test_item])
                        ws.append(sim)
                
                if rates:
                    pred = py.average(rates, weights=ws)
                    truth = test[test_user][test_item]
                    error = abs(pred - truth)
                    errors.append(error)
        self.errors = errors

class MultiViewKmedoidsCF(KmedoidsCF):
    
    def __init__(self):
        KmedoidsCF.__init__(self)
        self.method_id = 'Multiview Kmedoids CF'
        self.beta = float(self.config['multiview.trust.beta'])
        
    def Multiview_Kmedoids(self, train, K):
        '''Multiview K-medoids clustering methods, refers to paper --- Multi-View Clustering
        
        --------------------------
        view 1: similarity; view 2: social trust'''

        print 'Start to cluster users by Multi-View K-medoids clustering ...'
        
        tol = 0.00001
        users = train.keys()
        
        '''Pre-processing: pre-compute user distance (rating similarity, social similarity) matrix '''
        rating_dist, trust_dist = self.user_dists(train)
        
        # random.seed(500)
        
        '''Initialization: initial k medoids for view 2, i.e. trust, what if we start with view 1: sim?'''
        medoid_indices = random.sample(range(len(users)), K)
        
        '''E-step for view 2: associate each data point o to the closest medoid'''
        start_with_trust = True
        if start_with_trust:
            
            trust_medoids = {k:users[index] for k, index in zip(range(K), medoid_indices)}
            trust_clusters = {}
            for user in train:
                min_dist = py.inf
                cluster_id = -1
                for c_id, medoid in trust_medoids.viewitems():
                    
                    # the medoid point itself
                    if user == medoid:
                        min_dist = 0
                        cluster_id = c_id
                        break
                    
                    dist = trust_dist[user][medoid] if user in trust_dist and medoid in trust_dist[user] else py.nan
                    
                    if not py.isnan(dist):
                        if min_dist > dist:
                            min_dist = dist
                            cluster_id = c_id
                if cluster_id == -1: continue
                
                user_cluster = trust_clusters[cluster_id] if cluster_id in trust_clusters else []
                user_cluster.append(user)
                trust_clusters[cluster_id] = user_cluster
        
        else:
            sim_medoids = {k:users[index] for k, index in zip(range(K), medoid_indices)}
            sim_clusters = {}
            for user in train:
                min_dist = py.inf
                cluster_id = -1
                for c_id, medoid in sim_medoids.viewitems():
                    
                    # the medoid point itself
                    if user == medoid:
                        min_dist = 0
                        cluster_id = c_id
                        break
                    
                    dist = rating_dist[user][medoid] if user in rating_dist and medoid in rating_dist[user] else py.nan
                    
                    if not py.isnan(dist):
                        if min_dist > dist:
                            min_dist = dist
                            cluster_id = c_id
                
                if cluster_id == -1: continue
                
                user_cluster = sim_clusters[cluster_id] if cluster_id in sim_clusters else []
                user_cluster.append(user)
                sim_clusters[cluster_id] = user_cluster
            
        iteration = 100
        deltas = {}
        deltas_stable_times = 0
        
        for t in range(0, 2 * iteration, 2):
            
            for v in range(2):
                if v == (0 if start_with_trust else 1):
                    '''M-step for view 1: for each medoid m and each data point o associated to m, 
                        swap m and o and compute the total cost of the configuration, 
                        that is, the average dissimilarity of o to all the data points associated to m. 
                        Select the medoid o with the lowest cost of the configuration.'''
                    
                    sim_medoids = copy.deepcopy(trust_medoids)
                    sim_delta = 0
                    for cluster_id, cluster_ms in trust_clusters.viewitems():
                        medoid = trust_medoids[cluster_id]
                        
                        # compute last errors based on this medoid
                        last_errors = 0
                        for user in cluster_ms:
                            if user == medoid: continue
                            dist = rating_dist[user][medoid] if user in rating_dist and medoid in rating_dist[user] else py.nan
                            if not py.isnan(dist):
                                last_errors += dist
                        
                        best_errors = last_errors
                        best_errors_medoid = medoid
                        
                        # compute total cost by using other users in the clusters
                        for m in cluster_ms:
                            if m == medoid: continue
                            
                            new_medoid = m
                            new_errors = 0
                            for user in cluster_ms:
                                if user == new_medoid: continue
                                dist = rating_dist[user][new_medoid] if user in rating_dist and new_medoid in rating_dist[user] else py.nan
                                if not py.isnan(dist):
                                    new_errors += dist
                            
                            if new_errors < best_errors:
                                best_errors = new_errors
                                best_errors_medoid = new_medoid
                        
                        sim_delta += (last_errors - best_errors)
                        sim_medoids[cluster_id] = best_errors_medoid
                    
                    '''E-step for view 1: associate each data point o to the closest medoid'''
                    if t > 0 and sim_delta == 0: continue  # no need to re-associate as it becomes stable, but also need to ensure sim_clusters exist
                    sim_clusters = {}
                    for user in train:
                        min_dist = py.inf
                        cluster_id = -1
                        for c_id, medoid in sim_medoids.viewitems():
                            
                            # the medoid point itself
                            if user == medoid:
                                min_dist = 0
                                cluster_id = c_id
                                break
                            
                            dist = rating_dist[user][medoid] if user in rating_dist and medoid in rating_dist[user] else py.nan
                            
                            if not py.isnan(dist):
                                if min_dist > dist:
                                    min_dist = dist
                                    cluster_id = c_id
                        
                        if cluster_id == -1: continue
                        
                        user_cluster = sim_clusters[cluster_id] if cluster_id in sim_clusters else []
                        user_cluster.append(user)
                        sim_clusters[cluster_id] = user_cluster
                        
                elif v == (1 if start_with_trust else 0):
                    '''M-step for view 2: for each medoid m and each data point o associated to m, 
                        swap m and o and compute the total cost of the configuration, 
                        that is, the average dissimilarity of o to all the data points associated to m. 
                        Select the medoid o with the lowest cost of the configuration.'''
                    
                    trust_medoids = copy.deepcopy(sim_medoids)
                    trust_delta = 0
                    for cluster_id, cluster_ms in sim_clusters.viewitems():
                        medoid = sim_medoids[cluster_id]
                        
                        # compute last errors based on this medoid
                        last_errors = 0
                        for user in cluster_ms:
                            if user == medoid: continue
                            dist = trust_dist[user][medoid] if user in trust_dist and medoid in trust_dist[user] else py.nan
                            if not py.isnan(dist):
                                last_errors += dist
                        
                        best_errors = last_errors
                        best_errors_medoid = medoid
                        
                        # compute total cost by using other users in the clusters
                        for m in cluster_ms:
                            if m == medoid: continue
                            
                            new_medoid = m
                            new_errors = 0
                            for user in cluster_ms:
                                if user == new_medoid: continue
                                dist = trust_dist[user][new_medoid] if user in trust_dist and new_medoid in trust_dist[user] else py.nan
                                if not py.isnan(dist):
                                    new_errors += dist
                            
                            if new_errors < best_errors:
                                best_errors = new_errors
                                best_errors_medoid = new_medoid
                        
                        trust_delta += (last_errors - best_errors)
                        trust_medoids[cluster_id] = best_errors_medoid
                    
                    '''E-step for view 2: associate each data point o to the closest medoid'''
                    if t > 0 and trust_delta == 0: continue
                    trust_clusters = {}
                    for user in train:
                        min_dist = py.inf
                        cluster_id = -1
                        for c_id, medoid in trust_medoids.viewitems():
                            
                            # the medoid point itself
                            if user == medoid:
                                min_dist = 0
                                cluster_id = c_id
                                break
                            
                            dist = trust_dist[user][medoid] if user in trust_dist and medoid in trust_dist[user] else py.nan
                            
                            if not py.isnan(dist):
                                if min_dist > dist:
                                    min_dist = dist
                                    cluster_id = c_id
                        
                        if min_dist == 1.0 or cluster_id == -1: continue
                        
                        user_cluster = trust_clusters[cluster_id] if cluster_id in trust_clusters else []
                        user_cluster.append(user)
                        trust_clusters[cluster_id] = user_cluster
            
            delta = sim_delta + trust_delta
            deltas[t % 4] = delta
            
            if verbose: print 'iteration', (t + 1), 'delta =', delta
            if delta < tol: 
                break
            if len(deltas) == 4:
                odd_delta = deltas[0] - deltas[2]
                even_delta = deltas[1] - deltas[3]
                if odd_delta == 0 and even_delta == 0:
                    deltas_stable_times += 1
                if deltas_stable_times >= 4 and delta == min(deltas.values()):
                    break
            
        print 'Done!'
        
        return {k: list(set(sim_clusters[k]) | set(trust_clusters[k])) for k in range(K)}
    
    def cross_over(self, train, test):
        
        clusters = self.Multiview_Kmedoids(train, self.n_clusters)
        
        self.results += ',' + self.cluster_by + ',' + str(self.n_clusters) + ',' + str(self.alpha) + ',' + str(self.beta)
        
        errors = []
        for test_user in test:
            if test_user not in train: continue     
            
            # it is possible that one user occurs in two clusters
            cluster_ids = [] 
            for c_id, c_ms in clusters.viewitems():
                if test_user in c_ms:
                    cluster_ids.append(c_id)
                
            if not cluster_ids:
                # print 'cannot find the cluster for test user', test_user
                continue
                
            for test_item in test[test_user]:
                preds = []
                for cluster_id in cluster_ids:
                    candidates = [m for m in clusters[cluster_id] if m != test_user and test_item in train[m]]
                    rates = []
                    ws = []
                    for m in candidates:
                        sim = 1 - self.rating_dist[test_user][m] if test_user in self.rating_dist and m in self.rating_dist[test_user] else py.nan
                        if not py.isnan(sim) and sim > 0:
                            t = 1 - self.trust_dist[test_user][m] if test_user in self.trust_dist and m in self.trust_dist[test_user] else 0
                            
                            rates.append(train[m][test_item])
                            
                            w = 1 - self.beta * t # if t==1 => sim=1 which is not correct. so beta in [0, 1)
                            ws.append(math.pow(sim, w))
                    if rates:
                        pred = py.average(rates, weights=ws)
                        preds.append(pred)
                if preds:
                    truth = test[test_user][test_item]
                    error = abs(py.mean(preds) - truth)
                    errors.append(error)
        self.errors = errors
        
class KmeansCF(AbstractCF):
    def __init__(self):
        self.method_id = 'Kmeans CF'

    def kmeans(self, train, n_clusters):
        ''' A good reference: http://home.dei.polimi.it/matteucc/Clustering/tutorial_html/kmeans.html
        '''
        print 'Start to cluster users ...',
        last_errors = 0
        tol = 0.00001
        
        # items = self.items.keys()
        users = train.keys()
        
        # random.seed(100)
        
        # initial k-means clusters
        centroids_indices = random.sample(range(len(users)), n_clusters)
        centroids_users = [users[index] for index in centroids_indices]
        centroids = {index: train[user] for index, user in zip(range(len(centroids_users)), centroids_users)}
        
        iteration = 50
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
        # if verbose: print 'centroids:', centroids
        print 'Done!'
        return centroids, clusters
    
    def cross_over_top_n(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        sim_dist = {}
        mu_dist = {}
        count = 0
        # {test_user: [prediction object]}
        user_preds = {}
        pred_method = self.config['cluster.pred.method']
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
                
                if pred_method == 'mean':
                    rates = [train[member][test_item] for member in members if test_item in train[member]]
                    if not rates: continue
                    pred = py.mean(rates)
                
                elif pred_method == 'wcf':
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
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters) 
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
        
class KCF_1(KmeansCF):
    ''' This class is to train KmeansCF method using non-cold users to train the model, 
        only the cluster with the highest similarity will be chosen as the neighborhood.
    '''
    
    def __init__(self):
        self.method_id = 'KCF-1'
    
    def prep_test(self, train, test=None):
        cold_len = 5
        heavy_len = 10
        if test is not None:
            self.train2 = train
            self.train = {user:item_ratings for user, item_ratings in train.items() if len(train[user]) >= cold_len}
            
            if self.dataset_mode == 'all':
                return self.test
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in test.items() if (user not in train or len(train[user]) < cold_len) and user in self.trust}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
            
    def cross_over(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters) 
        for test_user in test:
            # identity the clusters of this test_user
            cluster = -1
            u = self.train2[test_user] if test_user in self.train2 else {}
            if not u: continue
            
            max = -10
            c_index = -1
            for index, centroid in centroids.viewitems():
                sim = self.similarity(u, centroid)
                if py.isnan(sim): continue
                if max < sim:
                    max = sim
                    c_index = index
            
            if c_index == -1:
                if verbose: 
                    print 'cannot identify the cluster for user:', test_user
                continue
            else:
                c_members = clusters[c_index]
                members = [member for member in c_members if member != test_user]
            
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
        
class KAverage(KCF_1):
    '''Kmeans with average: average the values of all different clusters. '''
    
    def __init__(self):
        self.method_id = 'Kmeans Average'
    
    def cross_over(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters) 
        
        for test_user in test:
            for test_item in test[test_user]:
                
                rates = [centroid[test_item] for centroid in centroids.viewvalues() if test_item in centroid]
                if not rates: continue
                pred = py.mean(rates)
                
                errors.append(abs(pred - test[test_user][test_item]))
        self.errors = errors

class KCF_all(KCF_1):
    '''use all the similarity clusters as the clusters of the cold users, 
       the final prediction is weighted by the similarities
    '''
    def __init__(self):
        self.method_id = 'KCF-all'
        
    def cross_over(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters)
        for test_user in test:
            # identity the clusters of this test_user
            cluster = -1
            u = self.train2[test_user] if test_user in self.train2 else {}
            if not u: continue
            
            cluster_sims = {}
            for index, centroid in centroids.viewitems():
                sim = self.similarity(u, centroid)
                if py.isnan(sim): continue
                if sim > 0:
                    cluster_sims[index] = sim
            
            if not cluster_sims:
                print 'cannot identify the cluster for user:', test_user
                continue
            
            for test_item in test[test_user]:
                
                preds = []
                ws = []
                for cluster, sim in cluster_sims.viewitems():
                    members = [m for m in clusters[cluster] if m != test_user]
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
                    preds.append(pred)
                    ws.append(sim)
                if not ws: 
                    continue
                pred = py.average(preds, weights=ws)
                errors.append(abs(pred - test[test_user][test_item]))
        self.errors = errors

class KmeansTrust(KmeansCF):
    ''' We claim that in cold conditions [cold_users]:
    1) trust information is more useful to determine the clusters of cold users than the rating information;
       two cases: (a) cold users can be clustered, but the identified cluster may not be reliable (due to the un-reliabl compouted similarity); 
                  (b) cold users cannot be clustered due to the non-computable user similarity; 
    To verify this: 
       (1) kmeans-cf uses rating information to determine clusters
       (2) kmeans-trust uses trust information to determine clusters
       (3) kmeans-tcf may use both trust and rating information to determine clusters
    
    2) through connecting different clusters based on trust links, the coverage can be improved; 
    3) distrust information may help refine the cluster connections; '''
    
    def __init__(self):
        self.method_id = 'Kmeans Trust'
    
    def gen_edge_id(self, source, target):
        return '<' + str(source.identity) + ', ' + str(target.identity) + '>'
    
    def gen_cluster_graph(self, clusters, trust):
        
        g = Graph()
        nodes = {v:Vertex(v) for v in clusters.keys() if v > -1}
        edges = {self.gen_edge_id(s, t): Edge(s, t) for s in nodes.viewvalues() for t in nodes.viewvalues() if s != t}
        
        for c1, c1ms in clusters.viewitems():
            if c1 <= -1:continue
            for c2, c2ms in clusters.viewitems():
                if c1 == c2: continue
                if c2 <= -1: continue
                s = nodes[c1]
                t = nodes[c2]
                e_id = self.gen_edge_id(s, t)
                edge = edges[e_id]
                
                for u in c1ms: 
                    tns = trust[u] if u in trust else []
                    if not tns: continue
                    for v in c2ms:
                        if v in tns:
                            g.add_edge(edge)
        g.print_graph()
    
    def cross_over(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                clusters = result[1]
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        trust = self.trust
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters)
        for test_user in test:
            # identity the clusters of this test_user
            cluster = -1
            for cluster_index, cluster_members in clusters.viewitems():
                if test_user in cluster_members:
                    cluster = cluster_index
                    members = [member for member in cluster_members if member != test_user]
                    break
            if cluster == -1:
                if verbose: 
                    print 'cannot identify the cluster for user:', test_user
                '''use trust to determine its clusters'''
                tns = self.read_trust(trust, test_user)
                if not tns: continue
                cluster_tns = {}
                total = 0
                for tn in tns:
                    for cluster_index, cluster_members in clusters.viewitems():
                        if tn in cluster_members:
                            cnt = cluster_tns[cluster_index] if cluster_index in cluster_tns else 0
                            cnt += 1
                            cluster_tns[cluster_index] = cnt
                            total += 1
                            break
                        
                if not cluster_tns: continue
            
            for test_item in test[test_user]:
                
                if pred_method == 'mean':
                    if cluster > -1:
                        rates = [train[member][test_item] for member in members if test_item in train[member]]
                        if not rates: continue
                        pred = py.mean(rates)
                    else: 
                        preds = []
                        weights = []
                        for cluster_index, cluster_cnt in cluster_tns.viewitems():
                            members = [member for member in clusters[cluster_index] if member != test_user]
                            rates = [train[member][test_item] for member in members if test_item in train[member]]
                            if not rates: continue
                            preds.append(py.mean(rates))
                            weights.append(float(cluster_cnt) / total)
                        if not preds: continue
                        pred = py.average(preds, weights=weights)
                
                elif pred_method == 'wcf':
                    
                    if cluster > -1:
                        rates = []
                        weights = []
                        for member in members:
                            if member == test_user:continue
                            if test_item in train[member]:
                                u = train[test_user]
                                v = train[member]
                                weight = self.similarity(u, v)
                                
                                if py.isnan(weight) or weight <= self.similarity_threashold: continue
                                
                                rates.append(train[member][test_item])
                                weights.append(weight)                            
                        if not rates: continue
                        pred = py.average(rates, weights=weights)
                    
                    else:
                        preds = []
                        weights = []
                        for cluster_index, cluster_cnt in cluster_tns.viewitems():
                            rates = []
                            ws = []
                            for member in members:
                                if member == test_user:continue
                                if test_item in train[member]:
                                    u = train[test_user]
                                    v = train[member]
                                    weight = self.similarity(u, v)
                                    
                                    if py.isnan(weight) or weight <= self.similarity_threashold: continue
                                    
                                    rates.append(train[member][test_item])
                                    ws.append(weight)                            
                            if not rates: continue
                            pred = py.average(rates, weights=ws)   
                            weights.append(float(cluster_cnt) / total)
                        if not preds: continue
                        pred = py.average(preds, weights=weights)
                
                errors.append(abs(pred - test[test_user][test_item]))
        self.errors = errors
        
    def cross_over_top_n(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                centroids, clusters = result
                break
            else:
                print 're-try different initial centroids'
        
        self.gen_cluster_graph(clusters, self.trust)
        
        sim_dist = {}
        mu_dist = {}
        count = 0
        # {test_user: [prediction object]}
        user_preds = {}
        pred_method = self.config['cluster.pred.method']
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
                
                if pred_method == 'mean':
                    rates = [train[member][test_item] for member in members if test_item in train[member]]
                    if not rates: continue
                    pred = py.mean(rates)
                
                elif pred_method == 'wcf':
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
        
class Trusties2(KmeansCF):
    
    def __init__(self):
        self.method_id = 'Trusties2'
        self.base = 2.0
    
    def prep_test(self, train, test=None):
        if test is not None:
            self.train2 = train
            self.train = {user:item_ratings for user, item_ratings in train.items() if len(train[user]) >= self.cold_len}
            
            if self.dataset_mode == 'all':
                return self.test
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in test.items() if (user not in train or len(train[user]) < self.cold_len) and user in self.trust}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) > self.heavy_len}
            else:
                raise ValueError('invalid test data set mode')
            
    def gen_cluster_graph(self, cluster_result):
        centroids = cluster_result[0]
        clusters = cluster_result[1]
        
        g = Graph()
        
        # generate vertices
        vs = {c: Vertex(c) for c in centroids.viewkeys()}
        
        # genrate edges
        for c1 in vs.viewkeys(): 
            c1ms = clusters[c1]
            for c2 in vs.viewkeys(): 
                # currently, we ignore the circular links
                if c1 == c2: continue
                
                c2ms = clusters[c2]
                links = {c1m:c2m for c1m in c1ms for c2m in c2ms if c1m in self.trust and c2m in self.trust[c1m]}
                
                if links:
                    edge = Edge(vs[c1], vs[c2])
                    r = len(links)
                    edge.weight = 2.0 / (1 + math.exp(-r / self.base)) - 1.0
                    g.add_edge(edge)
        
        if debug: g.print_graph()
        return g.page_rank(d=0.85, normalized=True)
    
    def gen_cluster_trust(self, cluster_result):
        # centroids = cluster_result[0]
        clusters = cluster_result[1]
        
        cluster_trust = {}
        # genrate edges
        for c1, c1ms in clusters.viewitems(): 
            if c1 == -1:continue
            cluster_rs = {}
            for c2, c2ms in clusters.viewitems(): 
                # currently, we ignore the circular links
                if c1 == c2: continue
                
                links = {c1m:c2m for c1m in c1ms for c2m in c2ms if c1m in self.trust and c2m in self.trust[c1m]}
                
                
                if links:
                    r = len(links)
                    cluster_rs[c2] = r
            
            sum_rs = sum(cluster_rs.values())
            for cluster_id, r in cluster_rs.viewitems():
                t = 2.0 / (1 + math.exp(-r / 5.0)) - 1.0
                # t = float(r) / sum_rs
                c_trust = cluster_trust[c1] if c1 in cluster_trust else {}
                c_trust[cluster_id] = t
                cluster_trust[c1] = c_trust
        
        return cluster_trust
    
    def propagate_trust(self, source, cluster_ws, cluster_trust, prop_len):
        ''' apply MoleTrust to infer trust value of cluster
            
            ------------------
            cluster_ws: directly trusted cluster
            cluster_trust: all the trust information among clusters
            prop_len: propagated length'''
        # len 1
        dist = 1
        visited = []
        cluster_ts = {key:val for key, val in cluster_ws.viewitems()}
        visited.extend(cluster_ts.keys())
        
        cluster_dist = {}
        cluster_dist[dist] = cluster_ws.keys()
        
        trust_threshold = 0.2
        
        while dist < prop_len:
            c_processors = {}
            dist += 1
            
            new_nodes = []
            if (dist - 1) not in cluster_dist:
                print 'stop here'
                return cluster_ts
            
            for cluster in cluster_dist[dist - 1]:
                next_clusters = cluster_trust[cluster] if cluster in cluster_trust else {}
                for c in next_clusters.viewkeys():
                    if c not in visited:
                        processors = c_processors[cluster] if cluster in c_processors else []
                        processors.append(cluster)
                        c_processors[c] = processors
                        
                        cs = cluster_dist[dist] if dist in cluster_dist else []
                        if c not in cs:
                            cs.append(c)
                            cluster_dist[dist] = cs
                        
                        if c not in new_nodes:
                            new_nodes.append(c)
            visited.extend(new_nodes)
            
            for cluster, processors in c_processors.viewitems():
                nodes = [cluster_ts[processor] for processor in processors if processor in cluster_ts and cluster_ts[processor] > trust_threshold]
                edges = [cluster_trust[processor][cluster] for processor in processors if processor in cluster_ts and cluster_ts[processor] > trust_threshold]
                
                '''if len(nodes) == 1:
                    cluster_ts[cluster] = nodes[0] * edges[0]
                else:'''
                if nodes:
                    cluster_ts[cluster] = py.average(edges, weights=nodes)
        
        return cluster_ts
    
    def get_wot(self, user):
        wot = []
        
        current_list = [u for u in self.trust[user]]
        next_list = []
        
        while current_list:
            for u in current_list:
                wot.append(u)
                if u in self.trust:
                    for v in self.trust[u]:
                        if v == user:continue
                        if v not in next_list and v not in current_list and v not in wot:
                            next_list.append(v) 
            current_list = [m for m in next_list]
            next_list = []
            
        return wot
        
    def cross_over(self, train, test):
        
        while(True):
            cluster_result = self.kmeans(train, n_clusters=self.n_clusters)
            if cluster_result is not None:
                centroids, clusters = cluster_result
                break
            else:
                print 're-try different initial centroids'
        
        global_importance = not True
        
        if global_importance:
            # global importance
            global_weight = self.gen_cluster_graph(cluster_result)
            global_weight = {int(key):val for key, val in global_weight.viewitems()}
        else:
            cluster_trust = self.gen_cluster_trust(cluster_result)
        
        errors = []
        cluster_overlaps = []
        
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(self.n_clusters) 
        
        alpha = float(self.config['trusties.trust.alpha'])
        beta = float(self.config['trusties.local.beta'])
        # record (alpha, beta) values
        self.results += ',' + str(alpha) + ',' + str(beta)
        
        for test_user in test:
            
            # trust value
            tns = self.read_trust(self.trust, test_user)
            
            cluster_tns = {}
            cluster_ws = {}  
            if tns:
                # len_tns = len(tns)
                for cluster_id, cluster_ms in clusters.viewitems():
                    r = 0
                    for tn in tns: 
                        if tn in cluster_ms:
                            r += 1
                    if r > 0:
                        cluster_tns[cluster_id] = r
                
                sum_r = sum(cluster_tns.values())
                for cluster_id, r in cluster_tns.viewitems():
                    cluster_ws[cluster_id] = 2.0 / (1 + math.exp(-r / self.base)) - 1.0
                    # cluster_ws[cluster_id] = float(r) / sum_r
                
                # propagate trust
                prop_len = 3
                if global_importance and cluster_ws:
                    cluster_ws = self.propagate_trust(test_user, cluster_ws, cluster_trust, prop_len)
            
            # similarity value
            cluster_ss = {}
            rs = self.train2[test_user] if test_user in self.train2 else []
            if rs: 
                for cluster_id, centroid in centroids.viewitems():
                    sim = self.similarity(centroid, rs)
                    if not py.isnan(sim) and sim > 0:
                        cluster_ss[cluster_id] = sim
            
            # local importance
            total = 0
            overlap = 0
            local_weight = {}
            if tns or rs:
                for cluster_id in centroids.viewkeys():
                    w = cluster_ws[cluster_id] if cluster_id in cluster_ws else 0
                    s = cluster_ss[cluster_id] if cluster_id in cluster_ss else 0
                    if w > 0 or s > 0:
                        if s > 0:
                            if w == 0:
                                val = s
                            elif s == 0:
                                val = 0
                            elif w > 0 and s > 0:
                                val = math.pow(s, 1 - w) if w > 0.5 and s > 0.5 else w * s
                        else:
                            val = 0

                        local_weight[cluster_id] = alpha * w + (1 - alpha) * val
                    
                        if w > 0 and s > 0:
                            # print s, 2 * w - 1, val
                            overlap += 1
                            
                        total += 1
                if total > 0:
                    cluster_overlaps.append(float(overlap) / total)
            
            # predict item's rating
            wot = self.get_wot(test_user)
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                
                preds = []
                weights = []
                for cluster_id, centroid in centroids.viewitems():
                    if test_item in centroid:
                        lw = local_weight[cluster_id] if cluster_id in local_weight else 0
                        
                        if global_importance:
                            gw = global_weight[cluster_id] if cluster_id in global_weight else 0
                        else:
                            gw = 0
                        # cluster weight
                        cw = beta * lw + (1 - beta) * gw
                        if cw > 0:
                            ms = [m for m in clusters[cluster_id] if test_item in train[m]]
                            rates = [train[m][test_item] for m in ms]
                            ws = []
                            for m in ms:
                                if m in wot:
                                    ws.append(1.05)
                                else:
                                    ws.append(1.0)
                            if not rates: continue
                            pred = py.average(rates, weights=ws)
                            preds.append(pred)
                            weights.append(cw)
                if not preds:
                    continue
                
                pred = py.average(preds, weights=weights)
                errors.append(abs(pred - truth))
        
        self.results += ',{0:.6f}'.format(py.mean(cluster_overlaps))
        self.errors = errors
  
class Trusties3(Trusties2):
    
    def __init__(self):
        self.method_id = 'Trusties3'
            
    def gen_cluster_graph(self, cluster_result):
        centroids = cluster_result[0]
        clusters = cluster_result[1]
        
        g = Graph()
        
        # generate vertices
        vs = {c: Vertex(c) for c in centroids.viewkeys()}
        
        # genrate edges
        for c1 in vs.viewkeys(): 
            c1ms = clusters[c1]
            for c2 in vs.viewkeys(): 
                # currently, we ignore the circular links
                if c1 == c2: continue
                
                c2ms = clusters[c2]
                links = {c1m:c2m for c1m in c1ms for c2m in c2ms if c1m in self.trust and c2m in self.trust[c1m]}
                
                if links:
                    edge = Edge(vs[c1], vs[c2])
                    r = len(links)
                    edge.weight = 2.0 / (1 + math.exp(-r / 2.0)) - 1.0
                    g.add_edge(edge)
        
        if debug: g.print_graph()
        return g.page_rank(d=0.85, normalized=True)
    
    def gen_cluster_trust(self, cluster_result):
        # centroids = cluster_result[0]
        clusters = cluster_result[1]
        
        cluster_trust = {}
        # genrate edges
        for c1, c1ms in clusters.viewitems(): 
            cluster_rs = {}
            for c2, c2ms in clusters.viewitems(): 
                # currently, we ignore the circular links
                if c1 == c2: continue
                
                links = {c1m:c2m for c1m in c1ms for c2m in c2ms if c1m in self.trust and c2m in self.trust[c1m]}
                
                
                if links:
                    r = len(links)
                    cluster_rs[c2] = r
            
            sum_rs = sum(cluster_rs.values())
            for cluster_id, r in cluster_rs.viewitems():
                # t = 2.0 / (1 + math.exp(-r / 2.0)) - 1.0
                t = float(r) / sum_rs
                c_trust = cluster_trust[c1] if c1 in cluster_trust else {}
                c_trust[cluster_id] = t
                cluster_trust[c1] = c_trust
        
        return cluster_trust
    
    def propagate_trust(self, source, cluster_ws, cluster_trust, prop_len):
        ''' apply MoleTrust to infer trust value of cluster
            
            ------------------
            cluster_ws: directly trusted cluster
            cluster_trust: all the trust information among clusters
            prop_len: propagated length'''
        # len 1
        dist = 1
        visited = []
        cluster_ts = {key:val for key, val in cluster_ws.viewitems()}
        visited.extend(cluster_ts.keys())
        
        cluster_dist = {}
        cluster_dist[dist] = cluster_ws.keys()
        
        trust_threshold = 0
        
        while dist < prop_len:
            c_processors = {}
            dist += 1
            
            new_nodes = []
            if (dist - 1) not in cluster_dist:
                print 'stop here'
                return cluster_ts
            
            for cluster in cluster_dist[dist - 1]:
                next_clusters = cluster_trust[cluster] if cluster in cluster_trust else {}
                for c in next_clusters.viewkeys():
                    if c not in visited:
                        processors = c_processors[cluster] if cluster in c_processors else []
                        processors.append(cluster)
                        c_processors[c] = processors
                        
                        cs = cluster_dist[dist] if dist in cluster_dist else []
                        if c not in cs:
                            cs.append(c)
                            cluster_dist[dist] = cs
                        
                        if c not in new_nodes:
                            new_nodes.append(c)
            visited.extend(new_nodes)
            
            for cluster, processors in c_processors.viewitems():
                nodes = [cluster_ts[processor] for processor in processors if processor in cluster_ts and cluster_ts[processor] > trust_threshold]
                edges = [cluster_trust[processor][cluster] for processor in processors if processor in cluster_ts and cluster_ts[processor] > trust_threshold]
                
                if nodes:
                    if len(nodes) == 1:
                        cluster_ts[cluster] = nodes[0] * edges[0]
                    else:
                    # if nodes:
                        cluster_ts[cluster] = py.average(edges, weights=nodes)
        
        return cluster_ts
        
    def cross_over(self, train, test):
        
        while(True):
            cluster_result = self.kmeans(train, n_clusters=self.n_clusters)
            if cluster_result is not None:
                centroids, clusters = cluster_result
                break
            else:
                print 're-try different initial centroids'
        
        global_importance = not True
        
        if global_importance:
            # global importance
            global_weight = self.gen_cluster_graph(cluster_result)
            global_weight = {int(key):val for key, val in global_weight.viewitems()}
        else:
            cluster_trust = self.gen_cluster_trust(cluster_result)
        
        errors = []
        cluster_overlaps = []
        
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(self.n_clusters) 
        
        alpha = float(self.config['trusties.trust.alpha'])
        beta = float(self.config['trusties.local.beta'])
        # record (alpha, beta) values
        self.results += ',' + str(alpha) + ',' + str(beta)
        
        # for training 
        trusts = []
        sims = []
        for test_user in test:
            
            # trust value
            tns = self.read_trust(self.trust, test_user)
            
            cluster_tns = {}
            cluster_ws = {}  
            if tns:
                # len_tns = len(tns)
                for cluster_id, cluster_ms in clusters.viewitems():
                    r = 0
                    for tn in tns: 
                        if tn in cluster_ms:
                            r += 1
                    if r > 0:
                        cluster_tns[cluster_id] = r
                
                sum_r = sum(cluster_tns.values())
                for cluster_id, r in cluster_tns.viewitems():
                    cluster_ws[cluster_id] = 2.0 / (1 + math.exp(-r / 2.0)) - 1.0
                    # cluster_ws[cluster_id] = float(r) / sum_r
                
                # propagate trust
                prop_len = 3
                if not global_importance and cluster_ws:
                    cluster_ws = self.propagate_trust(test_user, cluster_ws, cluster_trust, prop_len)
            
            # similarity value
            cluster_ss = {}
            rs = self.train2[test_user] if test_user in self.train2 else []
            if rs: 
                for cluster_id, centroid in centroids.viewitems():
                    sim = self.similarity(centroid, rs)
                    if not py.isnan(sim) and sim > 0:
                        cluster_ss[cluster_id] = sim
            
            # local importance
            if tns or rs:
                for cluster_id in centroids.viewkeys():
                    w = cluster_ws[cluster_id] if cluster_id in cluster_ws else 0
                    s = cluster_ss[cluster_id] if cluster_id in cluster_ss else 0
                    if w > 0 and s > 0:
                        trusts.append(w)
                        sims.append(s)
                            
        # least squar to learn: sim = a + b * t
        slope, intercept, r_val, p_val, std_err = stats.linregress(trusts, sims)
        print slope, intercept, r_val, p_val, std_err
        
        # for testing
        for test_user in test:
            # trust value
            tns = self.read_trust(self.trust, test_user)
            
            cluster_tns = {}
            cluster_ws = {}  
            if tns:
                # len_tns = len(tns)
                for cluster_id, cluster_ms in clusters.viewitems():
                    r = 0
                    for tn in tns: 
                        if tn in cluster_ms:
                            r += 1
                    if r > 0:
                        cluster_tns[cluster_id] = r
                
                sum_r = sum(cluster_tns.values())
                for cluster_id, r in cluster_tns.viewitems():
                    # cluster_ws[cluster_id] = 2.0 / (1 + math.exp(-r / 2.0)) - 1.0
                    cluster_ws[cluster_id] = float(r) / sum_r
                
                # propagate trust
                prop_len = 3
                if global_importance and cluster_ws:
                    cluster_ws = self.propagate_trust(test_user, cluster_ws, cluster_trust, prop_len)
            
            # similarity value
            cluster_ss = {}
            rs = self.train2[test_user] if test_user in self.train2 else []
            if rs: 
                for cluster_id, centroid in centroids.viewitems():
                    sim = self.similarity(centroid, rs)
                    if not py.isnan(sim) and sim > 0:
                        cluster_ss[cluster_id] = sim
            
            # local importance
            total = 0
            overlap = 0
            local_weight = {}
            if tns or rs:
                for cluster_id in centroids.viewkeys():
                    w = cluster_ws[cluster_id] if cluster_id in cluster_ws else 0
                    s = cluster_ss[cluster_id] if cluster_id in cluster_ss else 0
                    if w > 0 or s > 0:
                            
                        local_weight[cluster_id] = alpha * (slope * w + intercept) + (1 - alpha) * s
                    
                        if w > 0 and s > 0:
                            # print s, 2 * w - 1, val
                            overlap += 1
                            trusts.append(w)
                            sims.append(s)
                            
                        total += 1
                if total > 0:
                    cluster_overlaps.append(float(overlap) / total)
            # predict item's rating
            for test_item in test[test_user]:
                truth = test[test_user][test_item]
                
                preds = []
                weights = []
                for cluster_id, centroid in centroids.viewitems():
                    if test_item in centroid:
                        lw = local_weight[cluster_id] if cluster_id in local_weight else 0
                        
                        if global_importance:
                            gw = global_weight[cluster_id] if cluster_id in global_weight else 0
                        else:
                            gw = 0
                        # cluster weight
                        cw = beta * lw + (1 - beta) * gw
                        if cw > 0:
                            preds.append(centroid[test_item])
                            weights.append(cw)
                if not preds:
                    continue
                
                pred = py.average(preds, weights=weights)
                errors.append(abs(pred - truth))
        
        self.results += ',{0:.6f}'.format(py.mean(cluster_overlaps))
        self.errors = errors
  
class KMT_all(KmeansTrust):
    ''' This class is to train Kmeans method using non-cold users to train the model'''
    
    def __init__(self):
        self.method_id = 'KMT-all'
    
    def prep_test(self, train, test=None):
        cold_len = 5
        heavy_len = 10
        if test is not None:
            self.train = {user:item_ratings for user, item_ratings in train.items() if len(train[user]) >= cold_len}
            
            if self.dataset_mode == 'all':
                return self.test
            elif self.dataset_mode == 'cold_users':
                return {user:item_ratings for user, item_ratings in test.items() if (user not in train or len(train[user]) < cold_len) and user in self.trust}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user not in train or len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
            
class KMT_1(KMT_all):
    '''use the cluster with the most trusted neighbors as the neighborhood'''
    
    def __init__(self):
        self.method_id = 'KMT-1'
    
    def cross_over(self, train, test):
        n_clusters = self.n_clusters
        while(True):
            result = self.kmeans(train, n_clusters=n_clusters)
            if result is not None:
                clusters = result[1]
                break
            else:
                print 're-try different initial centroids'
        
        errors = []
        trust = self.trust
        pred_method = self.config['cluster.pred.method']
        self.results += ',' + pred_method + ',' + str(n_clusters)
        for test_user in test:
            # identity the clusters of this test_user
            cluster = -1
            '''In this setting, the cold users are not clustered, hence this step will not be useful'''
            for cluster_index, cluster_members in clusters.viewitems():
                if test_user in cluster_members:
                    cluster = cluster_index
                    members = [member for member in cluster_members if member != test_user]
                    break
            if cluster == -1:
                if verbose: 
                    print 'cannot identify the cluster for user:', test_user
                '''use trust to determine its clusters'''
                tns = self.read_trust(trust, test_user)
                
                if not tns: continue
                cluster_tns = {}
                total = 0
                for tn in tns:
                    for cluster_index, cluster_members in clusters.viewitems():
                        if tn in cluster_members:
                            cnt = cluster_tns[cluster_index] if cluster_index in cluster_tns else 0
                            cnt += 1
                            cluster_tns[cluster_index] = cnt
                            total += 1
                            break
                        
                if not cluster_tns: continue
                max = 0
                cluster = -1
                for cluster_index, n_tns in cluster_tns.viewitems():
                    if max < n_tns:
                        max = n_tns
                        cluster = cluster_index
            
            for test_item in test[test_user]:
                
                if pred_method == 'mean':
                    if cluster > -1:
                        rates = [train[member][test_item] for member in clusters[cluster] if test_item in train[member]]
                        if not rates: continue
                        pred = py.mean(rates)
                    else: 
                        preds = []
                        weights = []
                        for cluster_index, cluster_cnt in cluster_tns.viewitems():
                            members = [member for member in clusters[cluster_index] if member != test_user]
                            rates = [train[member][test_item] for member in members if test_item in train[member]]
                            if not rates: continue
                            preds.append(py.mean(rates))
                            # recompute weight using the other formula
                            weights.append(float(cluster_cnt) / total)
                        if not preds: continue
                        pred = py.average(preds, weights=weights)
                
                elif pred_method == 'wcf':
                    
                    if cluster > -1:
                        rates = []
                        weights = []
                        for member in members:
                            if member == test_user:continue
                            if test_item in train[member]:
                                u = train[test_user]
                                v = train[member]
                                weight = self.similarity(u, v)
                                
                                if py.isnan(weight) or weight <= self.similarity_threashold: continue
                                
                                rates.append(train[member][test_item])
                                weights.append(weight)                            
                        if not rates: continue
                        pred = py.average(rates, weights=weights)
                    
                    else:
                        preds = []
                        weights = []
                        for cluster_index, cluster_cnt in cluster_tns.viewitems():
                            rates = []
                            ws = []
                            for member in members:
                                if member == test_user:continue
                                if test_item in train[member]:
                                    u = train[test_user]
                                    v = train[member]
                                    weight = self.similarity(u, v)
                                    
                                    if py.isnan(weight) or weight <= self.similarity_threashold: continue
                                    
                                    rates.append(train[member][test_item])
                                    ws.append(weight)                            
                            if not rates: continue
                            pred = py.average(rates, weights=ws)   
                            weights.append(float(cluster_cnt) / total)
                        if not preds: continue
                        pred = py.average(preds, weights=weights)
                
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
    
    def cross_over_top_n(self, train, test):
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

def test_math():
    for i in range(20):
        print i, 2.0 / (1 + math.exp(-i / 2.0)) - 1
        print i, 2.0 / (1 + math.exp(-i / 3.0)) - 1
        print i, 2.0 / (1 + math.exp(-i / 4.0)) - 1
        print i, 2.0 / (1 + math.exp(-i / 5.0)) - 1
        print i, 2.0 / (1 + math.exp(-i / 6.0)) - 1
    
    
    xs = [0.5, 0.6]
    for x in xs:
        for i in range(10):
            y = 0.1 * i
            print y, math.pow(x, 1 - y)   
        
def main():
    
    if not True: test_math()
    
    config = load_config()
    AbstractCF.config = config
    
    methods = config['run.method'].lower().strip().split(',')
    
    for method in methods:
        if method == 'cf':
            ClassicCF().execute()
        elif method == 'ccf':
            CCF().execute()
        elif method == 'kavg':
            KAverage().execute()
        elif method == 'trusties':
            Trusties().execute()
        elif method == 'trusties2':
            Trusties2().execute()
        elif method == 'trusties3':
            Trusties3().execute()
        elif method == 'kmeans':
            KmeansCF().execute()
        elif method == 'kmtrust':
            KmeansTrust().execute()
        elif method == 'kcf-1':
            KCF_1().execute()
        elif method == 'kcf-all':
            KCF_all().execute()
        elif method == 'kmedoids':
            KmedoidsCF().execute()
        elif method == 'mv_kmedoids':
            MultiViewKmedoidsCF().execute()
        elif method == 'kmt-all':
            KMT_all().execute()
        elif method == 'kmt-1':
            KMT_1().execute()    
        elif method == 'mf':
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
