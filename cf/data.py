'''
Created on Feb 13, 2013

@author: Felix
'''
class Dataset:
    def __init__(self):
        self.users = []
        self.items = []
        self.ratings = {}
        self.trust = {}
        self.test = {}
        self.count = 0
        
    def load_ratings(self, file_path):
        ''' {user, {item: rating}} '''
        data = {}
        with open(file_path, 'r') as r:
            for line in r:
                user, item, rating = line.split()
                item_ratings = data[user] if user in data else {};
                item_ratings[item] = rating
                data[user] = item_ratings
                if user not in self.users:
                    self.users.append(user)
                if item not in self.items:
                    self.items.append(item)
                self.count += 1
        self.ratings = data
    
    def load_trust(self, file_path):
        ''' {trustor, {trustee, trust}} '''
        data = {}
        with open(file_path, 'r') as r:
            for line in r:
                trustor, trustee, trust = line.split()
                trustee_trust = data[trustor] if trustor in data else {}
                trustee_trust[trustee] = trust
                data[trustor] = trustee_trust
        self.trust = data
    
    def prep_test(self, mode):
        if mode == 'cold':
            self.test = {user:item_ratings for user, item_ratings in self.ratings.items() if len(self.ratings[user]) < 5}
        elif mode == 'all':
            self.test = self.ratings.copy()

def test():
    dirs = "D:\Java\Workspace\CF-RS\Datasets\FilmTrust\\"
    ds = Dataset()
    ds.load_ratings(dirs + 'ratings.txt')
    ds.load_trust(dirs + 'trust.txt')
    ds.prep_test('all')
    print 'get here'

            
