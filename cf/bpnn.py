# Back-Propagation Neural Networks
# 
# Written in Python.  See http://www.python.org/
# Placed in the public domain.
# Neil Schemenauer <nas@arctrix.com>
#
# Download from http://arctrix.com/nas/python/bpnn.py

import math
import random
import numpy as py
from sklearn import svm
import milk.supervised.adaboost as adaboost
import milk.supervised.svm as svm2
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model.logistic import LogisticRegression

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b - a) * random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill] * J)
    return m

# our sigmoid function, tanh is a little nicer than the standard 1/(1+e^-x)
def sigmoid(x):
    return math.tanh(x)

# derivative of our sigmoid function, in terms of the output (i.e. y)
def dsigmoid(y):
    return 1.0 - y ** 2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1  # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [1.0] * self.ni
        self.ah = [1.0] * self.nh
        self.ao = [1.0] * self.no
        
        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        
        init_range = 0.5
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-init_range, init_range)
                
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-init_range, init_range)

        # last change in weights for momentum   
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni - 1:
            raise ValueError('wrong number of inputs')

        # input activations
        for i in range(self.ni - 1):
            # self.ai[i] = sigmoid(inputs[i])
            self.ai[i] = inputs[i]

        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + self.ai[i] * self.wi[i][j]
            self.ah[j] = sigmoid(sum)

        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + self.ah[j] * self.wo[j][k]
            self.ao[k] = sigmoid(sum)

        return self.ao[:]


    def backPropagate(self, targets, N, M):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = targets[k] - self.ao[k]
            output_deltas[k] = dsigmoid(self.ao[k]) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k] * self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k] * self.ah[j]
                self.wo[j][k] = self.wo[j][k] + N * change + M * self.co[j][k]
                self.co[j][k] = change
                # print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j] * self.ai[i]
                self.wi[i][j] = self.wi[i][j] + N * change + M * self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + 0.5 * (targets[k] - self.ao[k]) ** 2
        return error


    def test(self, patterns):
        
        max_accuracy = 0
        max_theta = 0
        
        for j in range(10):
            theta = 0.5 + j * 0.05
            
            i = 0
            k = 0
            for p in patterns:
                # print(p[0], '->', self.update(p[0]))
                k += 1
                output = self.update(p[0])[0]
                
                # print k, output
                
                if output < theta:
                    label = 0.0
                else:
                    label = 1.0
                
                if label == p[1][0]: i += 1
            
            accuracy = float(i) / len(patterns)
            print 'accuracy = ', accuracy
            
            if max_accuracy < accuracy: 
                max_accuracy = accuracy
                max_theta = theta
        
        return max_accuracy, max_theta

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1):
        # N: learning rate
        # M: momentum factor
        min_error = py.inf
        min_error_cnt = 0
        for i in range(iterations):
            error = 0.0
            for p in patterns:
                inputs = p[0]
                targets = p[1]
                self.update(inputs)
                error = error + self.backPropagate(targets, N, M)
                
            if i % 20 == 0:
                print('iteration {0:-d} with error {1:-.5f}'.format(i, error))
            
            if min_error > error: 
                min_error = error
                min_error_cnt = 0
            else:
                min_error_cnt += 1
                
            if min_error_cnt > 100:break


def test_demo():
    # Teach network XOR function
    pat = [
        [[0, 0], [0]],
        [[0, 1], [1]],
        [[1, 0], [1]],
        [[1, 1], [0]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 2, 1)
    # train it with some patterns
    n.train(pat)
    # test it
    n.test(pat)

def prepare_data(expending=True):
    train_data = [] 
    train_targets = []
    
    test_data = []
    test_targets = []
    with open('flixster_features.txt') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line[0] == '#':
                print 'commenting line'
                continue
            
            features = []
            touples = line.split(',')
            
            '''Basic features'''
            # avg_weights
            avg_w1 = float(touples[12][1:])
            avg_w2 = float(touples[13][:-1])
            e_ws = avg_w1 - avg_w2
            # features.append(e_ws)  # => 0.51, 0.559
            
            # conf
            conf1 = float(touples[10][1:])
            conf2 = float(touples[11][:-1])
            e_conf = conf1 - conf2
            features.append(e_conf)  # => 0.543
            
            # total_cnt
            cnt1 = float(touples[2][1:])
            cnt2 = float(touples[3][:-1])
            e_cnt = cnt1 - cnt2
            #if cnt1 == 1.0 or cnt2 == 1.0:
            #    continue
            features.append(e_cnt)  # => 0.537, 0.555, (0.548)
            
            # sim_cnt
            sim_cnt1 = float(touples[6][1:])
            sim_cnt2 = float(touples[7][:-1])
            e_sim_cnt = sim_cnt1 - sim_cnt2
            # features.append(e_sim_cnt)  # =>0.5258, 0.549, (0.545)
            
            # trust_cnt
            trust_cnt1 = float(touples[8][1:])
            trust_cnt2 = float(touples[9][:-1])
            e_trust_cnt = trust_cnt1 - trust_cnt2
            # features.append(e_trust_cnt)  # =>0.533, 0.544, (0.553), (0.545)
            
            r1 = max([sim_cnt1, sim_cnt2])
            r2 = max([trust_cnt1, trust_cnt2])
            # features.append(r1 / (r1 + r2)) #=>0.524, 0.541, (0.518), (0.533)
            
            # preds
            pred1 = float(touples[4][1:])
            pred2 = float(touples[5][:-1])
            e_pred = pred1 - pred2
            features.append(e_pred)  # =>0.518, 0.552, (0.546, ), (0.549,)
            
            # std
            std1 = float(touples[0][1:])
            std2 = float(touples[1][:-1])
            e_std = std1 - std2
            features.append(e_std)  # =>0.512, 0.548, (0.543), (0.541)
            
            # targets
            label = float(touples[14])
            if label == 1.0:  # adopt greater prediction
                if pred1 < pred2:
                    target = 1
                else:
                    target = 0
            else:
                if pred1 < pred2:
                    target = 0
                else:
                    target = 1
            
            
            if line_num < 1000:
                test_data.append(features)
                test_targets.append(target)
            else:
                train_data.append(features)
                train_targets.append(target)
            
            line_num += 1
            if line_num >= 6000:
                break
    
    # normalization
    # step 1: find the maximum and minimum for each attribute
    max_norms = []
    min_norms = []
    
    for i in range(len(features)):
        max_norms.append(-py.inf)
        min_norms.append(py.Inf)
        
    for vec_features in train_data:
        for i in range(len(vec_features)):
            val = vec_features[i]
            if max_norms[i] < val: max_norms[i] = val
            if min_norms[i] > val: min_norms[i] = val
    
    # step 2: normalize the features values to [0, 1]
    for vec_features in train_data: 
        for i in range(len(vec_features)):
            val = vec_features[i]
            max_val = max_norms[i]
            min_val = min_norms[i]
            norm_val = (val - min_val) / (max_val - min_val)
            vec_features[i] = norm_val    
            
    for vec_features in test_data: 
        for i in range(len(vec_features)):
            val = vec_features[i]
            max_val = max_norms[i]
            min_val = min_norms[i]
            norm_val = (val - min_val) / (max_val - min_val)
            vec_features[i] = norm_val
    
    print 'number of features before expending:', len(features)
    
    if expending:
        # expending basic features
        ''' Expending basic features '''
        for features in train_data: 
            appending_features = []
            m = len(features)
            for x in range(m - 1):
                fx = features[x]
                for y in range(x + 1, m):
                    fy = features[y]
                    appending_features.append(fx + fy)
                    appending_features.append(abs(fx - fy))
                    # appending_features.append(fx - fy)
            
            features.extend(appending_features)
            
            
        for features in test_data: 
            appending_features = []
            m = len(features)
            for x in range(m - 1):
                fx = features[x]
                for y in range(x + 1, m):
                    fy = features[y]
                    appending_features.append(fx + fy)
                    appending_features.append(abs(fx - fy))
                    # appending_features.append(fx - fy)
            
            features.extend(appending_features)
        
        print 'number of features after expending:', len(features)
        # re-normalization: 
        max_norms = []
        min_norms = []
        
        for i in range(len(features)):
            max_norms.append(-py.inf)
            min_norms.append(py.Inf)
            
        for vec_features in train_data:
            for i in range(len(vec_features)):
                val = vec_features[i]
                if max_norms[i] < val: max_norms[i] = val
                if min_norms[i] > val: min_norms[i] = val
        
        # normalize the features values to [0, 1]
        for vec_features in train_data: 
            for i in range(len(vec_features)):
                val = vec_features[i]
                max_val = max_norms[i]
                min_val = min_norms[i]
                norm_val = (val - min_val) / (max_val - min_val)
                vec_features[i] = norm_val    
                
        for vec_features in test_data: 
            for i in range(len(vec_features)):
                val = vec_features[i]
                max_val = max_norms[i]
                min_val = min_norms[i]
                norm_val = (val - min_val) / (max_val - min_val)
                vec_features[i] = norm_val
            
    return  train_data, train_targets, test_data, test_targets
        
def prepare_nn_data(expending=True):
    train_data = [] 
    test_data = []
    
    with open('flixster_features.txt') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line[0] == '#':
                print 'commenting line'
                continue
            
            features = []
            touples = line.split(',')
            
            '''Basic features'''
            # avg_weights
            avg_w1 = float(touples[12][1:])
            avg_w2 = float(touples[13][:-1])
            e_ws = avg_w1 - avg_w2
            features.append(e_ws)  # => 0.51, 0.559
            
            # conf
            conf1 = float(touples[10][1:])
            conf2 = float(touples[11][:-1])
            e_conf = conf1 - conf2
            features.append(e_conf)  # => 0.543
            
            # total_cnt
            cnt1 = float(touples[2][1:])
            cnt2 = float(touples[3][:-1])
            e_cnt = cnt1 - cnt2
            # if cnt1 != 1.0 and cnt2 != 1.0:
            #    continue
            # features.append(e_cnt)  # => 0.537, 0.555, (0.548)
            
            # sim_cnt
            sim_cnt1 = float(touples[6][1:])
            sim_cnt2 = float(touples[7][:-1])
            e_sim_cnt = sim_cnt1 - sim_cnt2
            # features.append(e_sim_cnt)  # =>0.5258, 0.549, (0.545)
            
            # trust_cnt
            trust_cnt1 = float(touples[8][1:])
            trust_cnt2 = float(touples[9][:-1])
            e_trust_cnt = trust_cnt1 - trust_cnt2
            # features.append(e_trust_cnt)  # =>0.533, 0.544, (0.553), (0.545)
            
            r1 = max([sim_cnt1, sim_cnt2])
            r2 = max([trust_cnt1, trust_cnt2])
            # features.append(r1 / (r1 + r2)) #=>0.524, 0.541, (0.518), (0.533)
            
            # preds
            pred1 = float(touples[4][1:])
            pred2 = float(touples[5][:-1])
            e_pred = pred1 - pred2
            features.append(e_pred)  # =>0.518, 0.552, (0.546, ), (0.549,)
            
            # std
            std1 = float(touples[0][1:])
            std2 = float(touples[1][:-1])
            e_std = std1 - std2
            features.append(e_std)  # =>0.512, 0.548, (0.543), (0.541)
            
            # targets
            label = float(touples[14])
            if label == 1.0:  # adopt greater prediction
                if pred1 < pred2:
                    target = 1
                else:
                    target = 0
            else:
                if pred1 < pred2:
                    target = 0
                else:
                    target = 1
            
            
            if line_num < 1000:
                test_data.append([features, [target]])
            else:
                train_data.append([features, [target]])
            
            line_num += 1
            if line_num >= 6000:
                break
    
    # normalization
    # step 1: find the maximum and minimum for each attribute
    max_norms = []
    min_norms = []
    
    for i in range(len(features)):
        max_norms.append(-py.inf)
        min_norms.append(py.Inf)
        
    for vec_features in train_data:
        for i in range(len(vec_features[0])):
            val = vec_features[0][i]
            if max_norms[i] < val: max_norms[i] = val
            if min_norms[i] > val: min_norms[i] = val
    
    # step 2: normalize the features values to [0, 1]
    for vec_features in train_data: 
        for i in range(len(vec_features[0])):
            val = vec_features[0][i]
            max_val = max_norms[i]
            min_val = min_norms[i]
            norm_val = (val - min_val) / (max_val - min_val)
            vec_features[0][i] = norm_val    
            
    for vec_features in test_data: 
        for i in range(len(vec_features[0])):
            val = vec_features[0][i]
            max_val = max_norms[i]
            min_val = min_norms[i]
            norm_val = (val - min_val) / (max_val - min_val)
            vec_features[0][i] = norm_val
    
    print 'number of features before expending:', len(features)
    
    if expending:
        # expending basic features
        ''' Expending basic features '''
        for vec_features in train_data: 
            features = vec_features[0]
            appending_features = []
            m = len(features)
            for x in range(m - 1):
                fx = features[x]
                for y in range(x + 1, m):
                    fy = features[y]
                    appending_features.append(fx + fy)
                    appending_features.append(abs(fx - fy))
                    # appending_features.append(fx - fy)
            
            features.extend(appending_features)
            
            
        for vec_features in test_data: 
            features = vec_features[0]
            appending_features = []
            m = len(features)
            for x in range(m - 1):
                fx = features[x]
                for y in range(x + 1, m):
                    fy = features[y]
                    appending_features.append(fx + fy)
                    appending_features.append(abs(fx - fy))
                    # appending_features.append(fx - fy)
            
            features.extend(appending_features)
        
        print 'number of features after expending:', len(features)
        # re-normalization: 
        max_norms = []
        min_norms = []
        
        for i in range(len(features)):
            max_norms.append(-py.inf)
            min_norms.append(py.Inf)
            
        for vec_features in train_data:
            for i in range(len(vec_features[0])):
                val = vec_features[0][i]
                if max_norms[i] < val: max_norms[i] = val
                if min_norms[i] > val: min_norms[i] = val
        
        # normalize the features values to [0, 1]
        for vec_features in train_data: 
            for i in range(len(vec_features[0])):
                val = vec_features[0][i]
                max_val = max_norms[i]
                min_val = min_norms[i]
                norm_val = (val - min_val) / (max_val - min_val)
                vec_features[0][i] = norm_val    
                
        for vec_features in test_data: 
            for i in range(len(vec_features[0])):
                val = vec_features[0][i]
                max_val = max_norms[i]
                min_val = min_norms[i]
                norm_val = (val - min_val) / (max_val - min_val)
                vec_features[0][i] = norm_val
            
    return  train_data, test_data

def test_nn():
    train_data, test_data = prepare_nn_data()    
    
    num = len(train_data[0][0])
    n = NN(num, 2 * num, 1)
    n.train(train_data, iterations=100, N=0.1, M=0.02)
    n.test(test_data)
     
def test_svm():
    
    train_data, train_targets, test_data, test_targets = prepare_data()
    
    max_accuracy = 0
    max_accuracy_gamma = 0
    for i in range(11):
        g = i * 0.1
        clf = svm.NuSVC(kernel='rbf', gamma=g, probability=True)
        clf.fit(train_data, train_targets)
        pred_targets = clf.predict(test_data)
        pred_ps = clf.predict_proba(test_data)
        
        k = 0
        for i in range(len(pred_targets)):
            pred = pred_targets[i]
            truth = test_targets[i]
            
            if pred == truth:
                k += 1
        accuracy = float(k) / len(test_targets)
        
        print 'gamma =', g, ', accuracy =', accuracy
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            max_accuracy_gamma = g
    
    print '\nBest accuracy =', max_accuracy, ', best gamma =', max_accuracy_gamma
    
def test_adaboost():
    train_data, train_targets, test_data, test_targets = prepare_data(expending=False) 
    
    weak = svm2.svm_binary()
    learner = adaboost.boost_learner(weak, 100)
    # learner = multi.one_against_one(learner)
    model = learner.train(train_data, train_targets, normalisedlabels=True)
    
    correct = 0
    for i in range(len(test_data)):
        pred = model.apply(test_data[i])
        target = test_targets[i]
        
        if pred == target:
            correct += 1
    
    print 'Accuracy =', float(correct) / len(test_data)

def gen_threshold(train_targets):
    pos = 0
    neg = 0
    for label in train_targets:
        if label == 1:
            pos += 1
        elif label == 0:
            neg += 1
    
    return float(pos) / (pos + neg)

def test_logistic_regression():
    train_data, train_targets, test_data, test_targets = prepare_data(expending=False) 
    
    logit = LogisticRegression()
    logit.fit(train_data, train_targets)
    res = logit.predict_proba(test_data)
    
    theta = gen_threshold(train_targets)
    
    correct = 0
    for i in range(len(test_targets)):
        label = test_targets[i]
        prob = res[i][0]
        if prob > theta:
            pred = 0.0
        else:
            pred = 1.0
        
        if label == pred:
            correct += 1
    
    print 'Accuracy =', float(correct)/len(test_targets)
        
if __name__ == '__main__':
    # test_demo2()
    # test_svm()
    test_logistic_regression()
    # test_adaboost()
    # test_nn()
