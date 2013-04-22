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

def norm(val, max_range, min_range):
    
    return (val - min_range) / (max_range - min_range)   

def test_svm():
    train_data = [] 
    train_targets = []
    
    test_data = []
    test_targets = []
    
    test = True
    max_e_cnt = 22.0 if test else 0.0
    min_e_cnt = -21.0 if test else py.inf
    
    max_e_ws = 1.63052828876 if test else 0.0
    min_e_ws = -1.84894243415 if test else py.inf
    
    max_e_pred = 3.68294309876 if test else 0.0
    min_e_pred = -4.5 if test else py.inf
    
    max_e_conf = 0.586835439012 if test else 0.0
    min_e_conf = 3.68294309876 if test else py.inf
    
    with open('features.txt') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line: continue
            
            features = []
            
            touples = line.split(',')
            
            # targets
            target = float(touples[12])
            
            # avg_weights
            avg_w1 = float(touples[0][1:])
            avg_w2 = float(touples[1][:-1])
            e_ws = avg_w1 - avg_w2
            features.append(norm(e_ws, max_e_ws, min_e_ws) if test else e_ws)  # => when disabled, accuracy = 0.52
            
            if max_e_ws < e_ws: max_e_ws = e_ws
            if min_e_ws > e_ws: min_e_ws = e_ws
            
            # conf
            conf1 = float(touples[2][1:])
            conf2 = float(touples[3][:-1])
            e_conf = conf1 - conf2
            features.append(norm(e_conf, max_e_conf, min_e_conf) if test else e_conf)
            
            if max_e_conf < e_conf: max_e_conf = e_conf
            if min_e_conf > e_conf: min_e_conf = e_conf
            
            # total_cnt
            cnt1 = float(touples[8][1:])
            cnt2 = float(touples[9][:-1])
            e_cnt = cnt1 - cnt2
            features.append(norm(e_cnt, max_e_cnt, min_e_cnt) if test else e_cnt)
            
            if max_e_cnt < e_cnt: max_e_cnt = e_cnt
            if min_e_cnt > e_cnt: min_e_cnt = e_cnt
            
            # preds
            pred1 = float(touples[10][1:])
            pred2 = float(touples[11][:-1])
            e_pred = pred1 - pred2
            # features.append(norm(e_pred, max_e_pred, min_e_pred) if test else e_pred)
            
            if max_e_pred < e_pred: max_e_pred = e_pred
            if min_e_pred > e_pred: min_e_pred = e_pred
            
            if line_num < 100:
                test_data.append(features)
                test_targets.append(target)
            else:
                train_data.append(features)
                train_targets.append(target)
            
            line_num += 1
            if line_num >= 2000:
                break
    for i in range(10):
        g = i * 0.1
        clf = svm.NuSVC(kernel='poly', gamma=g, coef0=100)
        clf.fit(train_data, train_targets)
        pred_targets = clf.predict(test_data)
        
        k = 0
        for i in range(len(pred_targets)):
            pred = pred_targets[i]
            truth = test_targets[i]
            
            if pred == truth:
                k += 1
        
        print 'gamma =', g, ', accuracy =', float(k) / len(test_targets)
        
def test_svm2():
    train_data = [] 
    train_targets = []
    
    test_data = []
    test_targets = []
    
    test = True
    dataset = 'flixster'
    max_e_cnt = 34.0 if test else 0.0
    min_e_cnt = -31.0 if test else py.inf
    
    max_e_sim_cnt = 22.0 if test else 0.0
    min_e_sim_cnt = -30.0 if test else py.inf
    
    max_e_trust_cnt = 36.0 if test else 0.0
    min_e_trust_cnt = -37.0 if test else py.inf
    
    max_e_ws = 1.91766293548 if test else -py.inf
    min_e_ws = -1.94491118252 if test else py.inf
    
    max_e_pred = 4.5 if test else 0.0
    min_e_pred = -4.5 if test else py.inf
    
    # TODO: 
    max_e_cp = 3.00743449594 if test else -py.inf
    min_e_cp = -3.16453079404 if test else py.inf
    
    max_e_std = 2.25 if test else 0.0
    min_e_std = -2.25 if test else py.inf
    
    max_e_conf = 0.647055770484 if test else 0.0
    min_e_conf = -0.611959124809 if test else py.inf
    
    with open(dataset + '_features.txt') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line: continue
            if line[0] == '#':
                print 'commenting line'
                continue
            
            features = []
            touples = line.split(',')
            
            # targets
            target = float(touples[14])
            
            # avg_weights
            avg_w1 = float(touples[12][1:])
            avg_w2 = float(touples[13][:-1])
            e_ws = avg_w1 - avg_w2
            # features.append(norm(e_ws, max_e_ws, min_e_ws) if test else e_ws)  # => when disabled, accuracy = 0.52
            
            if max_e_ws < e_ws: max_e_ws = e_ws
            if min_e_ws > e_ws: min_e_ws = e_ws
            
            # conf
            conf1 = float(touples[10][1:])
            conf2 = float(touples[11][:-1])
            e_conf = conf1 - conf2
            features.append(norm(e_conf, max_e_conf, min_e_conf) if test else e_conf)
            
            if max_e_conf < e_conf: max_e_conf = e_conf
            if min_e_conf > e_conf: min_e_conf = e_conf
            
            # total_cnt
            cnt1 = float(touples[2][1:])
            cnt2 = float(touples[3][:-1])
            e_cnt = cnt1 - cnt2
            # features.append(norm(e_cnt, max_e_cnt, min_e_cnt) if test else e_cnt)
            
            if max_e_cnt < e_cnt: max_e_cnt = e_cnt
            if min_e_cnt > e_cnt: min_e_cnt = e_cnt
            
            # sim_cnt
            sim_cnt1 = float(touples[6][1:])
            sim_cnt2 = float(touples[7][:-1])
            e_sim_cnt = sim_cnt1 - sim_cnt2
            features.append(norm(e_sim_cnt, max_e_sim_cnt, min_e_sim_cnt) if test else e_cnt)
            
            if max_e_sim_cnt < e_sim_cnt: max_e_sim_cnt = e_sim_cnt
            if min_e_sim_cnt > e_sim_cnt: min_e_sim_cnt = e_sim_cnt
            
            # trust_cnt
            trust_cnt1 = float(touples[8][1:])
            trust_cnt2 = float(touples[9][:-1])
            e_trust_cnt = trust_cnt1 - trust_cnt2
            # features.append(norm(e_trust_cnt, max_e_trust_cnt, min_e_trust_cnt) if test else e_cnt)
            
            if max_e_trust_cnt < e_trust_cnt: max_e_trust_cnt = e_trust_cnt
            if min_e_trust_cnt > e_trust_cnt: min_e_trust_cnt = e_trust_cnt
            
            # preds
            pred1 = float(touples[4][1:])
            pred2 = float(touples[5][:-1])
            e_pred = pred1 - pred2
            features.append(norm(e_pred, max_e_pred, min_e_pred) if test else e_pred)
            
            if max_e_pred < e_pred: max_e_pred = e_pred
            if min_e_pred > e_pred: min_e_pred = e_pred
            
            # conf * pred
            cp1 = conf1 * pred1
            cp2 = conf2 * pred2
            e_cp = cp1 - cp2
            features.append(norm(e_cp, max_e_cp, min_e_cp) if test else e_cp)
            
            if max_e_cp < e_cp: max_e_cp = e_cp
            if min_e_cp > e_cp: min_e_cp = e_cp
            
            # std
            std1 = float(touples[0][1:])
            std2 = float(touples[1][:-1])
            e_std = std1 - std2
            features.append(norm(e_std, max_e_std, min_e_std) if test else e_std)
            
            if max_e_std < e_std: max_e_std = e_std
            if min_e_std > e_std: min_e_std = e_std
            
            if line_num < 100:
                test_data.append(features)
                test_targets.append(target)
            else:
                train_data.append(features)
                train_targets.append(target)
            
            line_num += 1
            # if line_num >= 5000:
            #    break
    max_accuracy = 0
    max_accuracy_gamma = 0
    for i in range(11):
        g = i * 0.1
        clf = svm.NuSVC(kernel='rbf', gamma=g)
        clf.fit(train_data, train_targets)
        pred_targets = clf.predict(test_data)
        
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
    
    print '\nbest accuracy =', max_accuracy, ', best gamma =', max_accuracy_gamma

def test_demo2():
    
    train_data = [] 
    test_data = []
    
    test = True
    max_e_cnt = 22.0 if test else 0.0
    min_e_cnt = -21.0 if test else py.inf
    
    max_e_ws = 1.63052828876 if test else 0.0
    min_e_ws = -1.84894243415 if test else py.inf
    
    max_e_pred = 3.68294309876 if test else 0.0
    min_e_pred = -4.5 if test else py.inf
    
    max_e_conf = 0.586835439012 if test else 0.0
    min_e_conf = 3.68294309876 if test else py.inf
    
    with open('features.txt') as f:
        line_num = 0
        for line in f:
            line = line.strip()
            if not line: continue
            
            features = []
            targets = []
            
            touples = line.split(',')
            
            # targets
            target = float(touples[12])
            targets.append(target)
            
            # avg_weights
            avg_w1 = float(touples[0][1:])
            avg_w2 = float(touples[1][:-1])
            e_ws = avg_w1 - avg_w2
            # features.append(norm(e_ws, max_e_ws, min_e_ws) if test else e_ws)  # => when disabled, accuracy = 0.52
            
            if max_e_ws < e_ws: max_e_ws = e_ws
            if min_e_ws > e_ws: min_e_ws = e_ws
            
            # conf
            conf1 = float(touples[2][1:])
            conf2 = float(touples[3][:-1])
            e_conf = conf1 - conf2
            features.append(norm(e_conf, max_e_conf, min_e_conf) if test else e_conf)
            
            if max_e_conf < e_conf: max_e_conf = e_conf
            if min_e_conf > e_conf: min_e_conf = e_conf
            
            # total_cnt
            cnt1 = float(touples[8][1:])
            cnt2 = float(touples[9][:-1])
            e_cnt = cnt1 - cnt2
            features.append(norm(e_cnt, max_e_cnt, min_e_cnt) if test else e_cnt)
            
            if max_e_cnt < e_cnt: max_e_cnt = e_cnt
            if min_e_cnt > e_cnt: min_e_cnt = e_cnt
            
            # preds
            pred1 = float(touples[10][1:])
            pred2 = float(touples[11][:-1])
            e_pred = pred1 - pred2
            features.append(norm(e_pred, max_e_pred, min_e_pred) if test else e_pred)
            
            if max_e_pred < e_pred: max_e_pred = e_pred
            if min_e_pred > e_pred: min_e_pred = e_pred
            
            data = []
            data.append(features)
            data.append(targets)
            
            if line_num < 100:
                test_data.append(data)
            else:
                train_data.append(data)
            
            line_num += 1
            if line_num >= 2000:
                break
    
    nodes = len(features)
    n = NN(nodes, 10, 1)
    n.train(train_data, iterations=10000, N=0.1, M=0.05)
    
    max_accuracy, max_theta = n.test(train_data)
    print 'best training accuracy =', max_accuracy, ', with threshold =', max_theta
    
    max_accuracy, max_theta = n.test(test_data)
    print 'best testing accuracy =', max_accuracy, ', with threshold =', max_theta

if __name__ == '__main__':
    # test_demo2()
    test_svm2()
