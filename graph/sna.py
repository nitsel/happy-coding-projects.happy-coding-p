import networkx as nx
import operator
import powerlaw

def read_graph(path):
    wiki = nx.read_gml(path)
    
    # powerlaw
    vals = wiki.degree().values()
    pl = powerlaw.Fit(vals, xmin=2007)
    print 'alpha =', pl.power_law.alpha
    print 'xmin =', pl.power_law.xmin
    R, p = pl.distribution_compare('power_law', 'exponential')
    print R, p
    
    pl2 = powerlaw.Fit(vals)
    print 'alpha-2 =', pl2.power_law.alpha
    print 'xmin-2 =', pl2.power_law.xmin
    R2, p2 = pl2.distribution_compare('power_law', 'exponential')
    print R2, p2
    
    # degree
    ids = wiki.in_degree()
    max_id_key, max_id_val = max(ids.iteritems(), key=operator.itemgetter(1))
    print 'id =', max_id_key, ', highest in-degree =', max_id_val, wiki.node[max_id_key]
    
    ods = wiki.out_degree()
    max_od_key, max_od_val = max(ods.iteritems(), key=operator.itemgetter(1))
    print 'id =', max_od_key, ', highest out-degree =', max_od_val, wiki.node[max_od_key]
    
    # page rank
    prs = nx.pagerank(wiki, alpha=0.85)
    pid, pval = max(prs.iteritems(), key=operator.itemgetter(1))
    print 'id =', pid, ', highest page rank =', pval, wiki.node[pid]
    
    # betweenness
    bws = nx.betweenness_centrality(wiki)
    bid, bval = max(bws.iteritems(), Key=operator.itemgetter(1))
    print 'id =', bid, ', highest betweenness =', bval, wiki.node[bid]
    
if __name__ == '__main__':
    path = "D:\Dropbox\PhD\My Work\Algorithms\@Machine Learning\Lectures\Social Network Analysis\Week 3_Centrality\wikipedia.gml"
    read_graph(path)
