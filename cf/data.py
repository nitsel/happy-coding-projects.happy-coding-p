'''
Created on Feb 13, 2013

@author: Felix
'''
class Dataset(object): 
    def __init__(self):
        self.count = 0  # number of ratings
        
    def load_ratings(self, file_path):
        ''' {user, {item: rating}} '''
        user_data = {}
        item_data = {}
        with open(file_path, 'r') as r:
            for line in r:
                line = line.strip()
                if not line: continue
                
                user, item, rating = line.split()
                
                item_ratings = user_data[user] if user in user_data else {}
                item_ratings[item] = float(rating)
                user_data[user] = item_ratings
                
                user_ratings = item_data[item] if item in item_data else {}
                user_ratings[user] = float(rating)
                item_data[item] = user_ratings
                
                self.count += 1
        return user_data, item_data
    
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


            
