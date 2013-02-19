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
from graph import Graph, Vertex, Edge

cross_validation = 'cross_validation'
leave_one_out = 'leave_one_out'
resnick_formula = 'resnick_formula'

on = 'on'
off = 'off'

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
                return {user:item_ratings for user, item_ratings in test.items() if user in train and len(train[user]) < cold_len}
            elif self.dataset_mode == 'heavy_users':
                return {user:item_ratings for user, item_ratings in test.items() if user in train and len(train[user]) > heavy_len}
            else:
                raise ValueError('invalid test data set mode')
            pass
        
    def execute(self):
        self.config_cf()
        logs.basicConfig(filename=self.debug_file, filemode='a', level=logs.DEBUG, format="%(message)s")
        
        # run multiple times at one shot
        batch_run = self.config['cross.validation.batch']
        if batch_run == on: 
            self.multiple_run()
        else:
            self.single_run()
            
    def multiple_run(self):
        if self.config['cross.validation.batch'] == on:
            for i in range(1, 6):
                self.rating_set = 'u' + str(i) + '.base'
                self.test_set = 'u' + str(i) + '.test'
                self.single_run()
            
    def single_run(self):    
        # import training rating and trust data
        ds = Dataset()
        self.train = ds.load_ratings(self.dataset_directory + self.rating_set)
        self.trust = ds.load_trust(self.dataset_directory + self.trust_set)
        
        # prepare test set
        self.test = ds.load_ratings(self.dataset_directory + self.test_set)  if self.validate_method == cross_validation else None
        self.test = self.prep_test(self.train, self.test)
        
        # execute recommendations
        self.perform(self.train, self.test)
        
        # collect results
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
        mae = 'MAE = {0:.6f}, '.format(MAE)
        pred_test = len(self.errors)
        total_test = sum([len(value) for value in self.test.viewvalues() ])
        RC = float(pred_test) / total_test * 100
        rc = 'RC = {0:d}/{1:d} = {2:2.2f}%, '.format(pred_test, total_test, RC)
        RMSE = math.sqrt(py.mean([e ** 2 for e in self.errors]))
        rmse = 'RMSE = {0:.6f}'.format(RMSE)
        
        print mae + rc + rmse + ', ' + self.dataset_directory + self.rating_set
        
        if self.config['write.results'] == on:
            results = '{0},{1},{2:.6f},{3:2.2f}%,{4:.6f},{5}'\
                    .format(self.method_id, self.dataset_mode, MAE, RC, RMSE, self.dataset_directory + self.rating_set)
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
        self.method_id='Abstract Cluster CF'
    
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
        
    def cluster_users(self, train):
        '''
        cluster_users the training users
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
            eps = 0.26
            minpts = 3
            return DBSCAN(eps=eps, min_samples=minpts, metric='precomputed').fit(D)
                    
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
        
        g.print_graph()
        return g.page_rank(d=0.85, normalized=True)
        
    def perform(self, train, test):
        db = self.cluster_users(train)
        
        # for test purpose
        if db is None: os.abort()
        
        core_indices = db.core_sample_indices_
        labels = db.labels_

        noises = py.where(labels == -1)[0]
        users = train.keys()
        
        prs = self.gen_cluster_graph(db)
        
        print prs
        
        ''' test properties of noise users
        
        noise_users = [users[x] for x in noises]
        sum=0
        cnt=0
        for user in noise_users:
            sum+=len(train[user])
            cnt+=1
        print 'average rating =', float(sum)/cnt'''
        
        errors = []
        pred_method = self.config['cluster.pred.method']
        
        trust_weight = 1.0
        core_weight = 2.0
        
        # procedure: trusties
        if pred_method == 'wtrust':
            for test_user in test: 
                user_index = users.index(test_user) if test_user in users else -1
                tns = self.trust[test_user] if test_user in self.trust else {}
                
                for test_item in test[test_user]:

                    preds = []
                    wpred = []
                    cs = -1  # cluster_users label of active users                  
                    # step 1: prediction from cluster_users that test_user belongs to
                    if user_index > -1 and user_index not in noises:
                        # predict according to cluster_users members
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
                                if py.isnan(sim): sim = 0
                                
                                # I think merging two many weights is not a good thing
                                # we should only use sim or member weights 
                                rates.append(train[member][test_item])
                                trust = trust_weight if member in tns else 0.0
                                weight = core_weight if index in core_indices else 1.0
                                weights.append(weight * (1 + sim) + trust)                            
                        if not rates: continue
                        pred = py.average(rates, weights=weights)
                        
                        preds.append(pred)
                        
                        weight_cluster = 0 if cs not in prs else (prs[str(cs)] if cs > -1 else 0)
                        wpred.append(1.0 + weight_cluster)
                
                    # step 2: prediction from clusters that the trusted neighbors belong to
                    if not preds and tns:
                        # find tns' clusters
                        tn_indices = [users.index(tn) for tn in tns if tn in users]
                        tns_cs = {users[index]: labels[index] for index in tn_indices}
                        
                        cls = list(set(tns_cs.values()))
                        for cl in cls: 
                            pred = 0.0
                            # if cl == -1: continue
                            if cl != -1 and cl == cs:continue
                            if cl == -1: 
                                nns = [key for key, val in tns_cs.items() if val == -1 and test_item in train[key]]
                                if not nns: continue
                                
                                rates = [train[nn][test_item] for nn in nns]
                                pred = py.average(rates)
                            else: 
                                member_indices = py.where(labels == cl)[0]
                                members = [users[x] for x in member_indices]
                                rates = []
                                weights = []
                                for member, index in zip(members, member_indices):
                                    if test_item in train[member]:
                                        rates.append(train[member][test_item])
                                        trust = trust_weight if member in self.trust[test_user] else 0.0
                                        weight = core_weight if index in core_indices else 1.0
                                        weights.append(weight * (1 + trust))                            
                                if not rates: continue
                                pred = py.average(rates, weights=weights)
                            
                            weight_cluster = 0 if str(cl) not in prs else (prs[str(cl)] if cl > -1 else 0)
                            weight_trust = tns_cs.values().count(cl) / float(len(tns_cs))
                            wpred.append(weight_cluster + weight_trust)
                            preds.append(pred)
                    
                    # step 3: for users without any trusted neighbors and do not have any predictions from the cluster that he blongs to 
                    if not tns and not preds and not False:
                        for cl in list(set(labels)):
                            # if cl == -1: continue
                            members = [users[x] for x in py.where(labels == cl)[0]]
                            rates = [train[member][test_item] for member in members if test_item in train[member]]
                            if not rates: continue
                            pred = py.average(rates)
                            preds.append(pred)
                            # wpred.append(prs[str(cl)] if str(cl) in prs else 1.0)
                        if not preds: continue
                        pred = py.average(preds)
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

class KmeansCF(Trusties):
    def __init__(self):
        self.method_id = 'kmeans_cf'

    def cluster_users(self, train):
        D=self.calc_user_distances(train)
        batch_size = 45
        db = MiniBatchKMeans(init='k-means++', n_clusters=10, batch_size=batch_size,
                  n_init=10, max_no_improvement=10, verbose=0).fit(D)
        labels = db.labels_
        n_clusters = len(set(labels))
        print 'Estimated number of clusters: %d' % n_clusters
    
    def perform(self, train, test):
        #self.cluster_users(train)
        pass
    
def main():
    config = load_config()
    AbstractCF.config = config
    
    run_method = config['run.method'].lower()
    if run_method == 'cf':
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
