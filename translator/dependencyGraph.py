# encoding=utf8
import re
import random
import networkx as nx

class dependencyGraph(object):

    def __init__(self, dep, head, tok, tgt):
        # transform graph from original conll format into our own data structure
        self.graph = nx.DiGraph()
        self.name2concept = tok
        self.root = None
        for i, x in enumerate(head):
            if x==0:
                assert self.root is None
                self.root = i
            self.graph.add_node(i)
        
        for src, (des1, rel) in enumerate(zip(head, dep)):
            des = des1 - 1
            if des < 0:
                continue
            self._add_edge(rel, src, des)
        self.target = tgt


    def __len__(self):
        return len(self.name2concept)**2 + len(self.target)

    def _add_edge(self, rel, src, des):
        self.graph.add_node(src)
        self.graph.add_node(des)
        self.graph.add_edge(src, des, label=rel)
        self.graph.add_edge(des, src, label=rel + '_r_')
    
    def bfs(self):
        g = self.graph
        queue = [self.root]
        depths = [0]
        visited = set(queue)
        step = 0
        while step < len(queue):
            u = queue[step]
            depth = depths[step]
            step += 1
            for v in g.neighbors(u):
                if v not in visited:
                    queue.append(v)
                    depths.append(depth+1)
                    visited.add(v)
        is_connected = (len(queue) == g.number_of_nodes())
        return queue, depths, is_connected

    def collect_concepts_and_relations(self):
        g = self.graph
        nodes, depths, is_connected = self.bfs()
        concepts = [self.name2concept[n] for n in nodes] 
        relations = dict()
        for i, src in enumerate(nodes):
            relations[i] = dict()
            paths = nx.single_source_shortest_path(g, src)
            for j, tgt in enumerate(nodes):
                relations[i][j] = list()
                assert tgt in paths
                path = paths[tgt]
                info = dict()
                #info['node'] = path[1:-1]
                info['edge'] = [g[path[i]][path[i+1]]['label'] for i in range(len(path)-1)]
                info['length'] = len(info['edge'])
                relations[i][j].append(info)

        ## TODO, we just use the sequential order
        depths = nodes
        return concepts, depths, relations, is_connected
