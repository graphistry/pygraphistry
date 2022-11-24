from matplotlib.font_manager import X11FontDirectories
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

import logging

logger = logging.getLogger(__name__)

class EmbedDistScore:

    @staticmethod
    def TransE(h, r, t):
        return (h + r - t).norm(p=1, dim=1)

    @staticmethod
    def DistMult(h, r, t):
        return (h * r * t).sum(dim=1)

    @staticmethod
    def RotatE(h, r, t):
        return -(h * r - t).norm(p=1, dim=1)
 
    

class HeterographEmbedModuleMixin(nn.Module):
    def __init__(self):
        super().__init__()

        self.protocol = {
                'TransE': EmbedDistScore.TransE,
                'DistMult': EmbedDistScore.DistMult,
                'RotatE':  EmbedDistScore.RotatE
        }

    def embed(self, relation, proto='DistMult', d=32, use_feat=True, X=None, epochs=2, 
              batch_size=32, train_split=0.8, *args, **kwargs):
        """Embed a graph using a relational graph convolutional network (RGCN), 
            and return a new graphistry graph with the embeddings as node attributes.

        Args:
            relation pd.column: column to use as relation between nodes
            proto (str, optional): which metric to use, ['TransE', 'RotateE', 'DistMult'] or provide your own. 
                Defaults to 'DistMult'.
            d (int, optional): relation embedding dimension. Defaults to 32.
            use_feat (bool, optional): whether to featurize nodes, if False will produce 
                random embeddings and shape them during training.
                Defaults to True.
            X (List or pd.DataFrame, optional): Which columns in the nodes dataframe to 
                featurize. Inherets args from graphistry.featurize(). 
                Defaults to None.
            epochs (int, optional): traing epoch. Defaults to 2.
            batch_size (int, optional): batch size. Defaults to 32.
            train_split (float, optional): train percentage, between 0, 1. Defaults to 0.8.

        Returns:
            self: graphistry instance
        """

        src, dst = self._source, self._destination
        self._relation = relation
        self._use_feat = use_feat

        if callable(proto):
            self.proto = proto
        else:
            self.proto = self.protocol[proto]
 
        if self._use_feat and self._nodes is not None:
            res = self.bind() #bind the node features to the graph
            # todo decouple self from res
            self = res = res.featurize(kind="nodes", X=X, *args, **kwargs)

        if self._node is not None and self._nodes is not None:
            nodes = self._nodes[self._node]
        elif self._node is None and self._nodes is not None:
            nodes = list(range(
                        self._nodes.index.start,
                        self._nodes.index.stop,
                        self._nodes.index.step
            ))

        else:
            nodes = list(set(
                self._edges[src].tolist() + self._edges[dst].tolist()
            ))
            
        edges = self._edges
        edges = edges[edges[src].isin(nodes) & edges[dst].isin(nodes)]
        relations = list(set(edges[relation].tolist()))
        
        # type2id 
        self._node2id = {n:idx for idx, n in enumerate(nodes)}
        self._relation2id = {r:idx for idx, r in enumerate(relations)}

        self._id2node = {idx:n for idx, n in enumerate(nodes)}
        self._id2relation = {idx:r for idx, r in enumerate(relations)}

        s, r, t = self._edges[src].tolist(), self._edges[relation].tolist(), self._edges[dst].tolist()
        triplets = [[self._node2id[_s], self._relation2id[_r], self._node2id[_t]] for _s, _r, _t in zip(s, r, t)]

        # split idx
        train_size = int(train_split * len(triplets))
        test_size = len(triplets) - train_size
        train_dataset, test_dataset = torch.utils.data.random_split(
                torch.tensor(triplets), 
                [train_size, test_size]
        )
        self.train_idx = train_dataset.indices
        self.test_idx = test_dataset.indices

        self.triplets_ = triplets

        del s, r, t
        
        num_nodes, num_rels = len(self._node2id), len(self._relation2id)
        
        s, r, t = torch.tensor(triplets).T
        g_dgl = dgl.graph(
                (s[self.train_idx], t[self.train_idx]), 
                num_nodes=num_nodes
        )
        g_dgl.edata[dgl.ETYPE] = r[self.train_idx]
        g_dgl.edata['norm'] = dgl.norm_by_dst(g_dgl).unsqueeze(-1)

        self.g_dgl = g_dgl

        # TODO: bidirectional connection
        g_iter = SubgraphIterator(g_dgl)
        g_dataloader = GraphDataLoader(
                g_iter, 
                batch_size=batch_size, 
                collate_fn=lambda x: x[0]
        )
        
        # init model and optimizer
        model = HeteroEmbed(num_nodes, num_rels, d, proto=self.proto, 
                    node_features=self._node_features)

            
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        
        pbar = trange(epochs, desc=None)
        
        score = 0
        for epoch in pbar:
            for data in g_dataloader:
                model.train()
                g, edges, labels = data

                emb = model(g)
                loss = model.loss(emb, edges, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                pbar.set_description(f"loss: {loss.item()}, score: {score}")

            model.eval()
            self._embed_model = model
            self._embeddings = model(g_dgl).detach().numpy()
            score = self._eval(threshold=0.95)
            pbar.set_description(f"epoch: {epoch}, loss: {loss.item()}, score:{score}")

        return self

    def calculate_prob(self, test_triplet, test_triplets, threshold, h_r, node_embeddings, infer=None):
        # TODO: simplify
        if infer == "all":
            s, r, o_ = test_triplet
        else:
            s, r = test_triplet

        subject_relation = test_triplet[:2]
        num_entity = len(node_embeddings)

        delete_idx = torch.sum(h_r == subject_relation, dim = 1)
        delete_idx = torch.nonzero(delete_idx == 2).squeeze()
    
        delete_entity_idx = test_triplets[delete_idx, 2].view(-1).numpy()
        perturb_entity_idx = np.array(list(set(np.arange(num_entity)) - set(delete_entity_idx)))
        perturb_entity_idx = torch.from_numpy(perturb_entity_idx).squeeze()

        if infer == "all":
            perturb_entity_idx = torch.cat((perturb_entity_idx, torch.unsqueeze(o_, 0)))

        o = self.proto(
                node_embeddings[s],
                self._embed_model.relational_embedding[r],
                node_embeddings[perturb_entity_idx])

        score = torch.sigmoid(o)
        return perturb_entity_idx[score > threshold]


    def _predict(self, test_triplets, threshold=0.95, directed=True, infer=None):
        
        if type(test_triplets) != torch.Tensor:
            test_triplets = torch.tensor(test_triplets)
        triplets = torch.tensor(self.triplets_)

        s, r, o = triplets.T
        #nodes = torch.tensor(list(set(s.tolist() + o.tolist())))
        edge_index = torch.stack([s, o])

        # make graph
        g = dgl.graph((s, o), num_nodes=edge_index.max()+1)
        g.edata[dgl.ETYPE] = r
        g.edata['norm'] = dgl.norm_by_dst(g).unsqueeze(-1)
        del s, r, o

        node_embeddings = self._embed_model(g)

        h_r = triplets[:, :2]
        t_r = torch.stack((triplets[:, 2], triplets[:, 1])).transpose(0, 1)

        visited, predicted_links = {}, []
        for test_triplet in test_triplets:
            s, r, o_ = test_triplet
            k = ''.join([str(s), "_", str(r)])
            kr = ''.join([str(r), "_", str(s)])

            # for [s, r] -> {d}
            if k not in visited:

                links = self.calculate_prob(
                        test_triplet, 
                        test_triplets,
                        threshold, 
                        h_r, 
                        node_embeddings,
                        infer
                ) 
                visited[k] = ''
                predicted_links += [[self._id2node[s.item()], self._id2relation[r.item()], 
                    self._id2node[i.item()]] for i in links]

            # for [d, r] -> {s}
            if kr not in visited and not directed:
                links = self.calculate_prob(
                        test_triplet,
                        test_triplets,
                        threshold,
                        t_r,
                        node_embeddings,
                        infer
                )
                visited[k] = ''
                predicted_links += [[self._id2node[s.item()], self._id2relation[r.item()], 
                    self._id2node[i.item()]] for i in links]
                
            
        predicted_links = pd.DataFrame(
            predicted_links, columns = [self._source, self._relation, self._destination]
        )
        
        return predicted_links, node_embeddings


    def predict_link(self, test_df: pd.DataFrame, src, rel, threshold: float=0.95):
        """predict links from a test dataframe given src/dst & rel columns

        Args:
            test_df (pd.DataFrame): dataframe of test data
            src : source column name
            rel : relation column name
            threshold (float, optional): probability threshold - confidence. 
                Defaults to 0.95.

        Returns:
            pd.DataFrame: dataframe of predicted links
        """

        nodes = [self._node2id[i] for i in test_df[src].tolist()]
        relations = [self._relation2id[i] for i in test_df[rel].tolist()]

        all_nodes = self._node2id.values()
        result = None
        for s, r in zip(nodes, relations):
            t_ = [[s, r, i] for i in all_nodes]
            o = self._score(t_)
            o = torch.tensor(t_)[o>=threshold]
            result = np.concatenate((result, o), axis=0) if result is not None else o

        result_df = []
        for i in result:
            s, r, d = i
            result_df += [[self._id2node[s], self._id2relation[r], self._id2node[d]]]
        result_df = pd.DataFrame(result_df, columns=[src, rel, "predicted_destination"])

        return result_df


    def predict_links(self, threshold=0.99, return_embeddings=True, retain_old_edges=False):
        """predict links over entire graph given a threshold

        Args:
            threshold (float, optional): Probability threshold. Defaults to 0.99.
            return_embeddings (bool, optional): will return DataFrame of predictions and node embeddings. 
                Defaults to True.
            retain_old_edges (bool, optional): will include old edges in predicted graph. 
                Defaults to False.

        Returns:
            graph: (graphistry graph) or 
                    (graphistry graph, DataFrame of predictions and node embeddings) when return_embeddings=True
        """

        predicted_links, node_embeddings = self._predict(
                    torch.tensor(self.triplets_),
                    threshold,
                    infer="all"
        )

        existing_links = self._edges[[self._source, self._relation, self._destination]]
    
        if retain_old_edges:
            all_links = pd.concat(
                [existing_links, predicted_links],
                ignore_index=True
            ).drop_duplicates()
        else:
            all_links = predicted_links

        g_new = self.nodes(self._nodes, self._node).edges(all_links, self._source, self._destination)

        #create a new graphistry graph
        if return_embeddings:
            return g_new, predicted_links, node_embeddings
        return g_new
    
    def _score(self, triplets):
        emb = torch.tensor(self._embeddings)
        triplets = torch.tensor(triplets)
        score =  self._embed_model.score(emb, triplets)
        prob = torch.sigmoid(score)
        return prob.detach().numpy()

    def _eval(self, threshold):
        if self.test_idx != []:
            triplets = torch.tensor(self.triplets_)[self.test_idx]
            score = self._score(triplets)
            return 100 * len(score[score >= threshold]) / len(score) 
        else:
            #raise exception -> "train_split must be < 1 for _eval()"
            logger.warning('train_split must be < 1 for _eval()')
       

class HeteroEmbed(nn.Module):
    def __init__(self, num_nodes, num_rels, d, proto, node_features=None, reg=0.01):
        super().__init__()

        self.reg = reg
        self.proto = proto
        self._node_features = node_features

        if self._node_features is not None:
            self._node_features = torch.tensor(self._node_features.values, dtype=torch.float32)
            print("node_features shape", node_features.shape)
        hidden = self._node_features.shape[-1] if node_features is not None else None
        self.rgcn = RGCNEmbed(d, num_nodes, num_rels, hidden)
        self.relational_embedding = nn.Parameter(torch.Tensor(num_rels, d))

        nn.init.xavier_uniform_(
                self.relational_embedding,
                gain=nn.init.calculate_gain('relu')
        )

    def __call__(self, g):
        # returns node embeddings
        return self.rgcn(g, node_features=self._node_features)
    
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
    def __init__(self, g, sample_size=30000, num_epochs=1000):
        self.num_epochs = num_epochs
        # TODO: raise exception -> sample size must be > 1
        self.sample_size = int(sample_size/2)
        self.eids = np.arange(g.num_edges())
        self.g = g
        self.num_nodes = g.num_nodes()

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

        triplets = np.stack((src, rel, dst)).T
        
        # negative sampling
        samples, labels = SubgraphIterator.sample_neg_(
                triplets, 
                self.num_nodes, 
                self.sample_size  # does nothing
        )

        src, rel, dst = samples.T

        # might need to add bidirectional edges
        sub_g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        sub_g.edata[dgl.ETYPE] = rel
        sub_g.edata['norm'] = dgl.norm_by_dst(sub_g).unsqueeze(-1)

        return sub_g, samples, labels

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
