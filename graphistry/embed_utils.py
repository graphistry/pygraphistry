from attr import attr
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange
from networks import RGCN
from dgl.dataloading import GraphDataLoader

import pandas as pd

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': self.TransE,
                'DistMult': self.DistMult,
                'RotatE': self.RotatE
        }

    def embed(self, src, dst, relation, proto='TransE', d=32):
        
        if callable(proto):
            proto = proto
        else:
            proto = self.protocol[proto]

        # steps
        # ---
        # 1. graphistry graph -> dgl.graph

        # pseudo code
        g_dgl = self.dgl_graph() 
        num_nodes, num_rels = g_dgl.num_nodes(), d_dgl.num_rels

        # TODO: bidirectional connection

        g_iter = SubgraphIterator(g_dgl, num_rels)
        g_dataloader = GraphDataLoader(g_iter, batch_size=1, collate_fn=lambda x: x[0])

        # init model and optimizer
        model = HeteroEmbed(num_nodes, num_rels, d, proto=proto)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

        epochs = trange(enumerate(g_dataloader), leave=True)
        for epoch, data in epochs:
            model.train()
            g, node_ids, edges, labels = data

            emb = model(g, node_ids)
            loss = model.loss(emb, edges, labels)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epochs.set_description(f"loss: {loss.item()}")
            epochs.refresh()

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
        self.rgcn = RGCN(num_nodes, d, num_rels)
        self.relational_embedding = nn.Parameter(torch.Tensor(num_rels, d))

        nn.init.xavier_uniform_(
                self.relational_embedding,
                gain=nn.init.calculate_gain('relu')
        )

    def __call__(self, g, node_ids):
        # returns node embeddings
        return self.rgcn(g, node_ids)

    def score(self, node_embedding, triplets)
        h, r, t = triplets.T
        h, r, t = node_embedding[h], self.relational_embedding[r], node_embedding[t]
        return self.proto(h, r, t)

    def loss(self, node_embedding, triplets, labels):
        score = self.score(node_embedding, triplets)

        # binary crossentropy loss
        l = F.binary_cross_entropy_with_logits(score, labels)

        # regularization loss
        ne_ = torch.mean(node_embedding.pow(2))
        re_ = torch.mean(relational_embedding.pow(2))
        rl = ne_ + re_
        
        return l + self.reg * rl
    

class SubGraphIterator:
    def __init__(self, g, num_rels, sample_size=30000, num_epochs=1000):
        self.num_epochs = num_epochs
        self.sample_size = sample_size
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
        src, dst = src.numpy(), dst.numpy()
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        uniq_v, edges = np.unique((src, dst), return_inverse=True)
        num_nodes = len(uniq_v)

        src, dst = edges.T
        triplets = np.stack((src, rel, dst)).T

        # negative sampling
        samples, labels = SubGraphIterator.sample_neg(
                triplets, 
                num_nodes, 
                self.sample_size
        )

        chosen_ids = np.random.choice(np.arange(self.sample_size),
                                      size=int(self.sample_size / 2),
                                      replace=False)
        src, dst, rel = samples.T

        # might need to add bidirectional edges
        sub_g = dgl.graph((src, dst), num_nodes=num_nodes)
        sub_g.edata[dgl.ETYPE] = torch.from_numpy(rel)
        sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)
        uniq_v = torch.from_numpy(uniq_v).view(-1).long()

        return sub_g, uniq_v, samples, labels

    @staticmethod
    def sample_neg_(triplets, num_nodes, sample_size):

        triplets = torch.tensor(triplets)
        h, r, t = triplets.T
        h_o_t = torch.randint(high=2, size=triplets.size()[0])

        random_h = torch.randint(high=len(event2id), size=h.size())
        random_t = torch.randint(high=len(attrib2id), size=h.size())

        neg_h = torch.where(h_o_t == 0, random_h, h)
        neg_t = torch.where(h_o_t == 1, random_t, t)
                
        neg_triplets = torch.stack((neg_h, r, neg_t), dim=1)

        all_triplets = torch.cat((triplets, neg_triplets), dim=1)
        labels = np.zeros((all_triplets.size()[0]))
        labels[:triplets.size()[0]] = 1
        
        flag = torch.randint(high=2, size=all_triplets.size()[0])
        all_triplets, labels = all_triplets[idx][flag==1], labels[idx][flag==1]

        all_triplets = all_triplets.numpy()

        return all_triplets, labels
