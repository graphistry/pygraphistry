import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': self.TransE
        }

    def embed(self, proto='TransE', d=128, gamma=12):

        # initialize hparams (from arguments)
        self.EPS = 2.0
        self.d = d
        self.gamma = gamma
        self.emrange = gamma + self.EPS / d

        # initializing the model
        em_mod = HeteroEmbed(eps, d, gamma, erange, self.protocol[proto])

        # initialize the dataloader
        # TODO: prepare entity2id <- from graphistry graph
        # TODO: prepare relation2id <- from graphistry graph
        dataset = EmbedDataset(entity2id, relation2id)
        train_generator = DataLoader(dataset, batch_size=32) # need autoscale batch for gpu?

        # training loop
        # optim (TODO: ADAM)
        optim = optim.SGD(em_mod.parameters(), lr=0.01)

        for _ in range(10):
        
            model.train()
            for h, r, t in train_generator:

                optim.zero_grad()

                pos_triples = torch.stack((h, r, t), dim=1)

                # negative sample generation
                h_o_t = torch.randint(high=2, size=h.size())
                random_nodes = torch.randint(high=len(entity2id), size=h.size())
                neg_h = torch.where(h_o_t == 0, random_nodes, h)
                neg_t = torch.where(h_o_t == 1, random_nodes, t)
                neg_triples = torch.stack((neg_h, r, neg_t), dim=1)

                loss = model(pos_triples, neg_triples)
                loss.mean().backward()

                optim.step()

        # out vectors
        # --------------------------------------
        # exception handling will be added later


    def TransE(self, h, r, t):
        o = h + r - t
        return self.gamma.item() - torch.norm(o, p=1, dim=2)

    def TransD(self, h, r, t):
        return


class HeteroEmbed(nn.Module):
    def __init__(self, eps, d, gamma, erange, proto):
        super().__init__()

        self.erange = erange
        
        self.node_em = nn.Embedding(_,d) # _ will be num_nodes
        self.edge_em = nn.Embedding(_,d) # _ will be num_edges

        self.uniform_(self.node_em)
        self.uniform_(self.edge_em)

        self.criterion = nn.MarginRankingLoss()

    def uniform_(self, x):
        x.weight.data.uniform_(-self.erange, -self.erange)
     
    def forward(self, pos_triplets, neg_triplets):

        h, r, t = pos_triplets
        pos_dist = proto(h, r, t)

        h, r, t = neg_triplets
        neg_dist = proto(h, r, t)

        return self.loss(pos_dist, neg_dist)

    def loss(self, pos_dist, neg_dist):
        target = torch.tensor([-1], dtype=torch.long)
        return self.criterion(pos_dist, neg_dist, target)
    
    def trainer(self):
        return


class EmbedDataset(Dataset):
    def __init__(self, entity2id, relation2id):

        self.entity2id = entity2id
        self.relation2id = relation2id

        self.relations = None # TODO: Need to get relations from graphistry graph

    def __len__(self):
        return len(self.relation)

    def __getitem__(self, idx):
        h, r, t = self.relation[idx]
        return h, r, t
