'''
Created on Feb 19, 2013

@author: guoguibing
'''

class Vertex(object):
    def __init__(self, identity, weight=1.0):
        self.identity = str(identity)
        self.weight = weight
        
    def __str__(self, *args, **kwargs):
        return self.identity

class Edge(object):
    
    def __init__(self, source, target, weight=1.0):
        self.source = source
        self.target = target
        
        self.identity = '<' + str(self.source.identity) + ', ' + str(self.target.identity) + '>'
        self.weight = weight
    
    def __str__(self, *args, **kwargs):
        return self.identity

class Graph(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        # {source, {target, edge}}
        self.source_target_edges = {}
        # {target, {source, edges}}
        self.target_source_edges = {}
        
        self.edges = []
    
    def copy(self):
        g = Graph()
        g.edges = [e for e in self.edges]
        g.source_target_edges = {s:tes for s, tes in self.source_target_edges.items()}
        g.target_source_edges = {t:ses for t, ses in self.target_source_edges.items()}
    
    def print_graph(self):
        print sorted({edge.identity: edge.weight for edge in self.edges}.items(), key=lambda x:x[0])
    
    def add_edge(self, edge):
        if edge not in self.edges: 
            self.edges.append(edge)
        source = edge.source
        target = edge.target
        
        # self.source_target_edges
        target_edges = self.source_target_edges[source] if source in self.source_target_edges else {}
        
        if target in target_edges:
            e = target_edges[target]
            edge.weight += e.weight
        target_edges[target] = edge
            
        self.source_target_edges[source] = target_edges
        # for complete purpose
        if target not in self.source_target_edges: self.source_target_edges[target] = {}
        
        # self.target_source_edges
        source_edges = self.target_source_edges[target] if target in self.target_source_edges else {}
        source_edges[source] = edge
        self.target_source_edges[target] = source_edges
        
        if source not in self.target_source_edges: self.target_source_edges[source] = {} 
    
    def page_rank(self, d=0.85, normalized=False):
        
        # initial the weights of all vertices and edges
        for source, target_edges in self.source_target_edges.viewitems(): 
            if source.weight == 0.0: source.weight = 1.0 
            if not target_edges: continue
            
            sum_weights = sum([e.weight for e in target_edges.values()])
            for edge in target_edges.viewvalues(): edge.weight /= sum_weights
        
        return self.weighted_page_rank(d, normalized)
    
    def wpr(self, d=0.85, normalized=False):
        ''' Xing and Ghorbani, Weighted pagerank algorithm, 
            In Proceedings of the 2nd Annual Conference on Communication Networks and Services Research, 2004
        
            Implementation of weighted pagerank algorithm
        '''
        # initial nodes' weights
        for source in self.source_target_edges.viewkeys(): 
            if source.weight == 0.0: source.weight = 1.0 
            
        # compute link weights according to the link structure of graph
        for edge in self.edges:
            v = edge.source
            u = edge.target
            
            # calculate in weight
            Iu = len(self.target_source_edges[u] if u in self.target_source_edges else {})
            
            targets = self.source_target_edges[v] if v in self.source_target_edges else {}
            Ivs = sum([len(self.target_source_edges[v]) for v in targets])
            
            weight_in = float(Iu) / Ivs if Ivs > 0 else 0
            
            # calculate out weight
            Ou = len(self.source_target_edges[u] if u in self.source_target_edges else {})
            Ovs = sum([len(self.source_target_edges[v]) for v in targets])
            
            weight_out = float(Ou) / Ovs if Ovs > 0 else 0
            
            edge.weight = weight_in * weight_out
            
            print edge.identity, 'in-weight =', weight_in, 'out-weight =', weight_out
            
        return self.weighted_page_rank(d, normalized)
        
    def weighted_page_rank(self, d=0.85, normalized=False):
        ''' computed general weighted page ranks
        
        parameters
        ----------------
        d: a dampening factor, default value is 0.85
        '''
        error_threshold = 0.000001
        iteration = 100
        for i in range(iteration):
            errors = 0.0
            for target, source_edges in self.target_source_edges.viewitems():
                if not source_edges: continue
                temp = target.weight
                target.weight = (1 - d) + d * sum([source.weight * edge.weight for source, edge in source_edges.items()])
                
                error = target.weight - temp
                errors += error ** 2
            if errors < error_threshold:
                # print 'stop at iteration', i, 'out of', iteration
                break
        norm = max([node.weight for node in self.source_target_edges.viewkeys()]) if normalized else 1.0
        # norm = sum([node.weight for node in self.source_target_edges.viewkeys()]) if normalized else 1.0
        return {node.identity: node.weight / float(norm) for node in self.source_target_edges.viewkeys()}

def test_page_rank():
    A = Vertex('A')
    B = Vertex('B')
    C = Vertex('C')
    
    g = Graph()
    g.add_edge(Edge(A, B))
    g.add_edge(Edge(B, C))
    g.add_edge(Edge(A, C))
    g.add_edge(Edge(C, A))
    
    g.print_graph()
    
    prs = g.page_rank(d=0.5)
    print sorted(prs.items(), key=lambda x:x[0])

def test_page_rank_multiple_edges():
    A0 = Vertex('0')
    A1 = Vertex('1')
    A2 = Vertex('2')
    A3 = Vertex('3')
    A4 = Vertex('4')
    
    g = Graph()
    g.add_edge(Edge(A0, A1))
    A12 = Edge(A1, A2)
    g.add_edge(A12)
    g.add_edge(A12)
    A13 = Edge(A1, A3)
    g.add_edge(A13)
    g.add_edge(A13)
    g.add_edge(Edge(A1, A4))
    g.add_edge(Edge(A2, A3))
    g.add_edge(Edge(A3, A0))
    g.add_edge(Edge(A4, A0))
    g.add_edge(Edge(A4, A2))
    
    prs = g.page_rank(d=0.85)
    sum_value = sum(prs.values())
    for key, value in prs.items(): 
        value /= sum_value
        prs[key] = value
    
    print sorted(prs.items(), key=lambda x:x[0])

def test_wpr():
    g = Graph()
    
    A = Vertex('A')
    B = Vertex('B')
    P1 = Vertex('P1')
    P2 = Vertex('P2')
    P3 = Vertex('P3')
    C1 = Vertex('C1')
    C2 = Vertex('C2')
    C3 = Vertex('C3')
    C4 = Vertex('C4')
    C5 = Vertex('C5')
    
    g.add_edge(Edge(A, P1))
    g.add_edge(Edge(A, P2))
    g.add_edge(Edge(B, P1))
    g.add_edge(Edge(B, P3))
    g.add_edge(Edge(P1, C1))
    g.add_edge(Edge(P1, C2))
    g.add_edge(Edge(P2, C1))
    g.add_edge(Edge(P2, C3))
    g.add_edge(Edge(P2, C5))
    g.add_edge(Edge(P3, C4))
    g.add_edge(Edge(P3, C5))
    
    prs = g.wpr(d=0.85)
    
    print sorted(prs.items(), key=lambda x: x[0])
