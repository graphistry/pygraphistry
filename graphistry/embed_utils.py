import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import trange

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': self.TransE,
                'DistMult': self.DistMult,
                'RotatE': self.RotatE
        }

    def embed(self, proto='TransE', d=128, gamma=12):

        # initialize hparams (from arguments)
        self.EPS = 2.0
        self.d = d
        self.emrange = gamma + self.EPS / d
        
        # initializing the model
        em_mod = HeteroEmbed(len(self._nodes), self.EPS, d, self.emrange, self.protocol[proto])

        # entity2id <- from graphistry graph
        entity2id = dict()
        src = set(self._edges[self._source]).union(set(self._edges[self._destination]))
        for idx, s in enumerate(src):
            entity2id[s] = idx

        # relation2id <- from graphistry graph
        relation2id = dict()
        rel = set(self._edges[self._relation])
        for idx, r in enumerate(rel):
            relation2id[r] = idx

        # build triplets (TODO: Efficiency)
        triplets = []
        src = list(self._edges[self._source])
        dst = list(self._edges[self._destination])
        relation = list(self._edges[self._relation])

        for s, r, d in zip(src, relation, dst):
            triplets.append([s, r, d])
        # initialize the dataloader
        dataset = EmbedDataset(entity2id, relation2id, triplets)
        train_generator = DataLoader(dataset, batch_size=32) # need autoscale batch for gpu? [AUTOML]

        # training loop
        optim = torch.optim.Adam(em_mod.parameters(), lr=0.01)

        # TODO: define epoch [AUTOML]

        epoch = trange(1000, leave=True)
        for _ in epoch:
            loss_acc, c = 0, 0

            #for _ in trange(1000):
            em_mod.train()
            for h, r, t in train_generator:

                optim.zero_grad()

                pos_triples = torch.stack((h, r, t), dim=1)

                # negative sample generation
                h_o_t = torch.randint(high=2, size=h.size())
                random_nodes = torch.randint(high=len(entity2id), size=h.size())
                neg_h = torch.where(h_o_t == 0, random_nodes, h)
                neg_t = torch.where(h_o_t == 1, random_nodes, t)
                neg_triples = torch.stack((neg_h, r, neg_t), dim=1)

                loss = em_mod(pos_triples, neg_triples)
                loss.mean().backward()
                loss_acc += loss.mean().item()
                c += 1
                optim.step()

            epoch.set_description(f"loss: {loss_acc/c}")
            epoch.refresh()

        self.node_em = em_mod.node_em.weight.detach().numpy()
        self.edge_em = em_mod.edge_em.weight.detach().numpy()
        return self.node_em, self.edge_em


    def TransE(self, h, r, t):
        return (h + r - t).norm(p=1, dim=1)

    def TransD(self, h, r, t):
        return

    def DistMult(self, h, r, t):
        return (h * r * t).sum(dim=-1)

    def RotatE(self, h, r, t):
        return -(h * r - t).norm(p=1, dim=1)
        


class HeteroEmbed(nn.Module):
    def __init__(self, num_nodes, eps, d, erange, proto):
        super().__init__()

        self.erange = erange
        
        self.node_em = nn.Embedding(num_nodes, d)
        self.edge_em = nn.Embedding(num_nodes, d)

        self.uniform_(self.node_em)
        self.uniform_(self.edge_em)

        self.criterion = nn.MarginRankingLoss(margin=1, reduction='none')
        self.proto = proto

    def uniform_(self, x):
        x.weight.data.uniform_(-self.erange, self.erange)
     
    def forward(self, pos_triplets, neg_triplets):

        h, r, t = pos_triplets.T
        h, r, t = self.node_em(h), self.edge_em(r), self.node_em(t)
        pos_dist = self.proto(h, r, t)

        h, r, t = neg_triplets.T
        h, r, t = self.node_em(h), self.edge_em(r), self.node_em(t)
        neg_dist = self.proto(h, r, t)

        return self.loss(pos_dist, neg_dist)

    def loss(self, pos_dist, neg_dist):
        target = torch.tensor([-1], dtype=torch.long)
        # target = torch.ones_like(pos_dist, dtype=torch.long) * -1
        return self.criterion(pos_dist, neg_dist, target)
    
    def trainer(self):
        return


class EmbedDataset(Dataset):
    def __init__(self, entity2id, relation2id, triplets):

        self.entity2id = entity2id
        self.relation2id = relation2id
        self.triplets = triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        h, r, t = self.triplets[idx]
        return self.entity2id[h], self.relation2id[r], self.entity2id[t]
