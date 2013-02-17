'''
Created on Feb 13, 2013

@author: Felix
'''
class Dataset(object): 
    def __init__(self):
        self.users = []
        self.items = []
        self.count = 0  # number of ratings
        
    def load_ratings(self, file_path):
        ''' {user, {item: rating}} '''
        data = {}
        with open(file_path, 'r') as r:
            for line in r:
                line = line.strip()
                if not line: continue
                
                user, item, rating = line.split()
                item_ratings = data[user] if user in data else {};
                item_ratings[item] = float(rating)
                data[user] = item_ratings
                if user not in self.users:
                    self.users.append(user)
                if item not in self.items:
                    self.items.append(item)
                self.count += 1
        return data
    
    def load_trust(self, file_path):
        ''' {trustor, {trustee, trust}} '''
        data = {}
        with open(file_path, 'r') as r:
            for line in r:
                trustor, trustee, trust = line.strip().split()
                trustee_trust = data[trustor] if trustor in data else {}
                trustee_trust[trustee] = float(trust)
                data[trustor] = trustee_trust
        return data


            
