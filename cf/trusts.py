'''
Created on Feb 14, 2013

@author: Felix
'''
import sims
import numpy as py

class TrustAll(object):
    '''
    classdocs
    '''

    def __init__(self):
        '''
        Constructor
        '''
        pass
    
    def perform(self, ds):
        '''
        ds: Dataset class object
        '''
        train = ds.ratings
        test = ds.test
        
        errors = []
        count = 0
        for test_user in test.keys():
            count += 1
            if count % 10 == 0:
                print 'current progress =', count, ' out of', len(test)
            # predict test item's rating
            for test_item in test[test_user]:
                truth = float(test[test_user][test_item])
                a = {item: float(rate) for item, rate in train[test_user].items() if item != test_item}
                
                # find similar users, train, weights
                users = []
                rates = []
                weights = []
                for user in train.keys():
                    if user == test_user: 
                        continue
                    b = {item: float(rate) for item, rate in train[user].items() if (item != test_item) and (test_item in train[user])}
                    if len(b) == 0: 
                        continue
                    sim = 1.0
                    # cos = sims.cos(a, b)
                    # pcc = sims.pcc(a, b)
                    users.append(user)
                    rates.append(float(train[user][test_item]))
                    weights.append(sim)
                
                # prediction
                if len(rates) == 0:
                    continue
                pred = [rate * weight for rate, weight in zip(rates, weights)]
                pred = sum(pred) / len(rates)
                errors.append(abs(truth - pred))
                
        MAE = py.mean(errors)
        print 'MAE =', MAE

