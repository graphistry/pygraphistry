from attr import attr
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from .networks import RGCNEmbed
import dgl
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F

import pandas as pd

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': self.TransE,
                'DistMult': self.DistMult,
                'RotatE': self.RotatE
        }

    def embed(self, src, dst, relation, proto='DistMult', d=32):
        
        if callable(proto):
            proto = proto
        else:
            proto = self.protocol[proto]

        nodes = list(set(self._edges[src].tolist() + self._edges[dst].tolist()))
        relations = self._edges[relation].tolist()
        
        # type2id 
        node2id = {n:idx for idx, n in enumerate(nodes)}
        relation2id = {r:idx for idx, r in enumerate(relations)}

        s, r, t = self._edges[src].tolist(), self._edges[relation].tolist(), self._edges[dst].tolist()
        triplets = [[node2id[_s], relation2id[_r], node2id[_t]] for _s, _r, _t in zip(s, r, t)]
        del s, r, t
        
        num_nodes, num_rels = len(nodes), len(relations)

        s, r, t = torch.tensor(triplets).T
        g_dgl = dgl.graph((s, t), num_nodes=num_nodes)
        g_dgl.edata[dgl.ETYPE] = r

        # TODO: bidirectional connection
        g_iter = SubgraphIterator(g_dgl, num_rels)
        g_dataloader = GraphDataLoader(g_iter, batch_size=1, collate_fn=lambda x: x[0])

        # init model and optimizer
        model = HeteroEmbed(num_nodes, num_rels, d, proto=proto)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        for data in g_dataloader:
            model.train()
            g, node_ids, edges, labels = data

            emb = model(g, node_ids)
            loss = model.loss(emb, edges, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            print(f"loss: {loss.item()}")

    def TransE(self, h, r, t):
        return (h + r - t).norm(p=1, dim=1)

    def DistMult(self, h, r, t):
        return (h * r * t).sum(dim=-1)

    def RotatE(self, h, r, t):
        return -(h * r - t).norm(p=1, dim=1)
        

class HeteroEmbed(nn.Module):
    def __init__(self, num_nodes, num_rels, d, proto, reg=0.01):
        super().__init__()

        self.reg = reg
        self.proto = proto
        self.rgcn = RGCNEmbed(d, num_nodes, num_rels)
        self.relational_embedding = nn.Parameter(torch.Tensor(num_rels, d))

        nn.init.xavier_uniform_(
                self.relational_embedding,
                gain=nn.init.calculate_gain('relu')
        )

    def __call__(self, g, node_ids):
        # returns node embeddings
        return self.rgcn(g, node_ids)

    def score(self, node_embedding, triplets):
        h, r, t = triplets.T
        h, r, t = node_embedding[h], self.relational_embedding[r], node_embedding[t]
        return self.proto(h, r, t)

    def loss(self, node_embedding, triplets, labels):
        score = self.score(node_embedding, triplets)

        # binary crossentropy loss
        l = F.binary_cross_entropy_with_logits(score, labels)

        # regularization loss
        ne_ = torch.mean(node_embedding.pow(2))
        re_ = torch.mean(self.relational_embedding.pow(2))
        rl = ne_ + re_
        
        return l + self.reg * rl
    

class SubgraphIterator:
    def __init__(self, g, num_rels, sample_size=30000, num_epochs=1000):
        self.num_epochs = num_epochs
        # TODO: raise exception -> sample size must be > 1
        self.sample_size = int(sample_size/2)
        self.eids = np.arange(g.num_edges())
        self.g = g

    def __len__(self):
        return self.num_epochs
    
    def __getitem__(self, i):
        eids = torch.from_numpy(
                np.random.choice(
                    self.eids, self.sample_size
                )
        )

        src, dst = self.g.find_edges(eids)
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        uniq_v, _ = torch.unique(torch.cat((src, dst)), return_inverse=True)
        num_nodes = len(uniq_v)

        triplets = np.stack((src, rel, dst)).T
        
        # negative sampling
        samples, labels = SubgraphIterator.sample_neg_(
                triplets, 
                num_nodes, 
                self.sample_size
        )

        src, rel, dst = samples.T

        # might need to add bidirectional edges
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = rel
        sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = uniq_v.view(-1).long()

        return sub_g, uniq_v, samples, labels

    @staticmethod
    def sample_neg_(triplets, num_nodes, sample_size):

        # TODO: remove all numpy operations

        triplets = torch.tensor(triplets)
        h, r, t = triplets.T
        h_o_t = torch.randint(high=2, size=h.size())

        random_h = torch.randint(high=num_nodes, size=h.size())
        random_t = torch.randint(high=num_nodes, size=h.size())

        neg_h = torch.where(h_o_t == 0, random_h, h)
        neg_t = torch.where(h_o_t == 1, random_t, t)
        neg_triplets = torch.stack((neg_h, r, neg_t), dim=1)

        all_triplets = torch.cat((triplets, neg_triplets), dim=0)
        labels = torch.zeros((all_triplets.size()[0]))
        labels[:triplets.shape[0]] = 1
        
        return all_triplets, labels
