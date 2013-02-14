'''
Created on Feb 14, 2013

@author: Felix
'''
from cf.data import Dataset
from cf.trusts import TrustAll

def main():
    dirs = "D:\Java\Workspace\CF-RS\Datasets\FilmTrust\\"
    ds = Dataset()
    ds.load_ratings(dirs + 'ratings.txt')
    ds.load_trust(dirs + 'trust.txt')
    ds.prep_test('all')
    ta = TrustAll()
    ta.perform(ds)
    
if __name__ == '__main__':
    main()
