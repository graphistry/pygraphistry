import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

import pandas as pd

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': self.TransE,
                'DistMult': self.DistMult,
                'RotatE': self.RotatE
        }

    def embed(self, proto='TransE', d=32, batch_size=32, epoch=100):

        if callable(proto):
            proto = proto
        else:
            proto = self.protocol[proto]

        # initialize hparams (from arguments)
        self.EPS = 2.0
        self.d = d
        self.emrange = 12 + self.EPS / d
        
        # initializing the model
        
        event = self._edges['EventID'].tolist()
        edgetype = self._edges['edgeType'].tolist()
        attrib = self._edges['attribID'].tolist()
        print(attrib)
        
        em_mod = HeteroEmbed(
                len(set(event)), 
                len(set(edgetype)), 
                len(set(attrib)), 
                d, 
                self.emrange, 
                proto
        )

        # type2id 
        event2id = {s:idx for idx, s in enumerate(set(event))}
        edgetype2id = {r:idx for idx, r in enumerate(set(edgetype))}
        attrib2id = {d:idx for idx, d in enumerate(set(attrib))}
        
        triplets = [[s, r, d] for s, r, d in zip(event, edgetype, attrib)]

        # initialize the dataloader
        dataset = EmbedDataset(event2id, edgetype2id, attrib2id, triplets)
        train_generator = DataLoader(dataset, batch_size=batch_size) # need autoscale batch for gpu? [AUTOML]

        # training loop
        optim = torch.optim.Adam(em_mod.parameters(), lr=0.01)

        # TODO: define epoch [AUTOML]

        epochs = trange(epoch, leave=True)
        for _ in epochs:
            loss_acc, c = 0, 0

            #for _ in trange(1000):
            em_mod.train()
            for h, r, t in train_generator:

                optim.zero_grad()

                pos_triples = torch.stack((h, r, t), dim=1)

                # negative sample generation
                h_o_t = torch.randint(high=2, size=h.size())

                random_h = torch.randint(high=len(event2id), size=h.size())
                random_t = torch.randint(high=len(attrib2id), size=h.size())

                neg_h = torch.where(h_o_t == 0, random_h, h)
                neg_t = torch.where(h_o_t == 1, random_t, t)
                neg_triples = torch.stack((neg_h, r, neg_t), dim=1)

                loss = em_mod(pos_triples, neg_triples)
                loss.mean().backward()
                loss_acc += loss.mean().item()
                c += 1
                optim.step()

            epochs.set_description(f"loss: {loss_acc/c}")
            epochs.refresh()

        #self._relational_node_embedding = em_mod.event_em.weight.detach().numpy() 
        res = pd.DataFrame(
                    em_mod.event_em.weight.detach().numpy(),
                    index=range(self._relational_node_embedding.shape[0])
        )

        
        return res

    
    def TransE(self, h, r, t):
        return (h + r - t).norm(p=1, dim=1)

    def TransD(self, h, r, t):
        return

    def DistMult(self, h, r, t):
        return (h * r * t).sum(dim=-1)

    def RotatE(self, h, r, t):
        return -(h * r - t).norm(p=1, dim=1)
        


class HeteroEmbed(nn.Module):
    def __init__(self, num_events, num_edgetype, num_attrib, d, erange, proto):
        super().__init__()

        self.erange = erange
        
        self.event_em = nn.Embedding(num_events, d)
        self.edgetype_em = nn.Embedding(num_edgetype, d)
        self.attrib_em = nn.Embedding(num_attrib, d)

        self.uniform_(self.event_em)
        self.uniform_(self.edgetype_em)
        self.uniform_(self.attrib_em)

        self.criterion = nn.MarginRankingLoss(margin=1, reduction='none')
        self.proto = proto

    def uniform_(self, x):
        x.weight.data.uniform_(-self.erange, self.erange)
     
    def forward(self, pos_triplets, neg_triplets):

        h, r, t = pos_triplets.T
        h, r, t = self.event_em(h), self.edgetype_em(r), self.attrib_em(t)
        pos_dist = self.proto(h, r, t)

        h, r, t = neg_triplets.T
        h, r, t = self.event_em(h), self.edgetype_em(r), self.attrib_em(t)
        neg_dist = self.proto(h, r, t)

        return self.loss(pos_dist, neg_dist)

    def loss(self, pos_dist, neg_dist):
        target = torch.tensor([-1], dtype=torch.long)
        return self.criterion(pos_dist, neg_dist, target)
    

class EmbedDataset(Dataset):
    def __init__(self, event, edgetype, attrib, triplets):

        self.event2id = event
        self.edgetype2id = edgetype
        self.attrib2id = attrib
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        return self.event2id[h], self.edgetype2id[r], self.attrib2id[t]
