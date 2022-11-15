import numpy as np
import pandas as pd

class MotifMixin:
    def __init__(self):
        super().__init__()
        pass

    def motif(
            self, 
            timestamp:str=None, 
            step:int=None, 
            accumulate:int=None):

        src, dst = self._source, self._destination
        
        self._temporal = False
        if timestamp is not None:
            self._temporal = True
        else:
            self._temporal = False

        # appending a dummy timestamp col
        if not self._temporal:
            timestamp = 'time'
            accumulate = 1
            step = 1
            self._edges[timestamp] = 0

        partitions = MotifMixin.split(
                    self._edges, 
                    accumulate,
                    timestamp,
        )

        # fetching motifs
        motifs = {}
        if not self._temporal:
            motif_edge_table = [] 

        for i in range(len(partitions)-step+1):
            _nodes = partitions[i][src].tolist() + partitions[i][dst].tolist()
            _nodes = set(_nodes)
            for v in _nodes:
                m = MotifMixin.build_motifs(
                        partitions[i:i+step+1], 
                        v, 
                        src, dst)

                if m is not None:
                    motif = MotifMixin.get_motifs(m)
                    if motif in motifs:
                        motifs[motif] += 1
                    else:
                        motifs[motif] = 1

                    if not self._temporal:
                        _m = [motif+'_'+j.split('_')[0] for j in m[0]]
                        for _n in _m[1:]:
                            motif_edge_table.append([_m[0], _n])
        
        if self._temporal:
            return motifs
        else:
            return self.edges(pd.DataFrame(motif_edge_table, columns=[src, dst]))

    @staticmethod
    def neighbors(graph, v, src, dst):
        connections = dict()

        # TODO: remove iterrows
        for _, row in graph.iterrows():
            i, j = row[src], row[dst]          

            if i == v:
                if i not in connections:
                    connections[i] = [j]
                else:
                    connections[i] += [j]

            if j == v:
                if j not in connections:
                    connections[j] = [i]
                else:
                    connections[j] += [i]
        
        if v in connections:
            return connections[v]
        else:
            return []
    
    @staticmethod
    def get_node_encoding(ids_no_ego,nodes_no_ego,lenght_ETNS):
        node_encoding = dict()
        for n in ids_no_ego:
            enc = []
            for k in range(lenght_ETNS):
                if str(n)+"_"+str(k) in nodes_no_ego:
                    enc.append(1)
                else:
                    enc.append(0)
            node_encoding[n]=enc
        return node_encoding

    @staticmethod
    def build_motifs(partitions, v, src, dst):

        # checking if neighbourhood of v is not 0
        if len(MotifMixin.neighbors(partitions[0], v, src, dst)) > 0:

            en_list = []
            
            for i in partitions:
                en_list.append(
                        [str(v)+"*"]+list(MotifMixin.neighbors(i, v, src, dst))
                )

            for i in range(len(en_list)):
                for j in range(len(en_list[i])):
                    en_list[i][j] = str(en_list[i][j])+"_"+str(i)
            
            en_list_long = []
            for en in en_list:
                en_list_long.append([n.split("_")[0] for n in en])

            for k in range(len(en_list_long)-1):
                for n in en_list_long[k]:
                    for en in range(len(en_list_long[k+1:])):
                        add = False
                        if n in en_list_long[k+en+1]:
                            add = True 
                            t = k + 1 + en
                            break
                    if add:
                        u, v = str(n)+"_"+str(k), str(n)+"_"+str(t)
                        en_list.append([u, v])
            
            return [i for i in en_list if len(i) != 1]

        else:
            return None
    
    @staticmethod
    def get_motifs(ETN):
        nodes = set([j for i in ETN for j in i])

        nodes_no_ego, ids_no_ego = [], []
        lenght_ETNS = 0
        for n in nodes:
            if not ("*" in n):
                nodes_no_ego.append(n)
                if not(n.split("_")[0] in ids_no_ego):
                    ids_no_ego.append(n.split("_")[0])
            else:
                ego = int(n.split("*")[0])
                lenght_ETNS = lenght_ETNS + 1

        node_encoding = MotifMixin.get_node_encoding(
                ids_no_ego,
                nodes_no_ego,
                lenght_ETNS
        )
    
        for k in node_encoding.keys():
            node_encoding[k] = '0e'+''.join(str(e) for e in node_encoding[k])

        binary_node_encodings = list(node_encoding.values())
        binary_node_encodings.sort()
    
        return '0e'+''.join(e[2:] for e in binary_node_encodings)

    @staticmethod
    def split(x, accumulate, timestamp):
        times = x[timestamp]
        pivot = times[0]
        partitions = []
        
        for i in range(len(times)):
            if not times[i] <= pivot + accumulate:
                partitions += [i]
                pivot = times[i]

        return np.split(x, partitions)
