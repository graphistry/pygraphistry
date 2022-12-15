from typing import TYPE_CHECKING, Any
from . import constants as config

import logging

logger = logging.getLogger(__name__)

def lazy_import_networks():  # noqa
    try:
        import dgl
        import dgl.nn as dglnn
        import dgl.function as fn
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        Module = nn.Module
        return nn, dgl, dglnn, fn, torch, F, Module
    except:
        return Any, Any, Any, Any, Any, Any, object

if TYPE_CHECKING:  # noqa
    _, dgl, dglnn, fn, torch, F, Module = lazy_import_networks()
else:
    nn = Any 
    dgl = Any
    dglnn = Any
    fn = Any
    torch = Any
    F = Any
    Module = object

try:
    import torch.nn as nn  # noqa
    Module = nn.Module  # noqa
except:
    Module = object



class GCN(Module):  # type: ignore
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        _, _, dglnn, _, _, _, _ = lazy_import_networks()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats)
        self.conv2 = dglnn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        _, _, _, _, _, F, _ = lazy_import_networks()
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class RGCN(Module):  # type: ignore
    """
        Heterograph where we gather message from neighbors along all edge types.
        You can use the module dgl.nn.pytorch.HeteroGraphConv (also available in MXNet and Tensorflow) to perform
        message passing on all edge types,
        then combining different graph convolution modules for each edge type.

    :returns
        torch model with forward pass methods useful for fitting model in standard way
    """

    def __init__(self, in_feats, hid_feats, out_feats, rel_names):
        super().__init__()
        _, _, dglnn, _, _, _, _ = lazy_import_networks()        
        
        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names},
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names},
            aggregate="sum",
        )

    def forward(self, graph, inputs):
        import torch.nn.functional as F
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(Module):  # type: ignore
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        nn, _, _, _, _, _, _ = lazy_import_networks()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        import dgl 
        h = g.ndata[config.FEATURE]
        h = self.rgcn(g, h)
        with g.local_scope():  # create a local scope to hold onto hidden layer 'h'
            g.ndata["h"] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, "h", ntype=ntype)
            return self.classify(hg)


class MLPPredictor(Module):  # type: ignore
    """One can also write a prediction function that predicts a vector for each edge with an MLP.
    Such vector can be used in further downstream tasks, e.g. as logits of a categorical distribution."""

    def __init__(self, in_features, out_classes):
        super().__init__()
        nn, _, _, _, _, _, _ = lazy_import_networks()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
        import torch
        h_u = edges.src["h"]
        h_v = edges.dst["h"]
        score = self.W(torch.cat([h_u, h_v], 1))
        return {"score": score}

    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(self.apply_edges)
            return graph.edata["score"]


class SAGE(Module):  # type: ignore
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        _, _, dglnn, _, _, _, _ = lazy_import_networks()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean"
        )
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type="mean"
        )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        import torch.nn.functional as F
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


class DotProductPredictor(Module):  # type: ignore
    def forward(self, graph, h):
        _, _, _, fn, _, _, _ = lazy_import_networks()

        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]


class LinkPredModel(Module):  # type: ignore
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        #nn, dgl, dglnn, fn, torch, F = lazy_import_networks()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


class LinkPredModelMultiOutput(Module):  # type: ignore
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        _, _, dglnn, _, _, _, _ = lazy_import_networks()
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_classes)
        self.embedding = dglnn.GraphConv(out_features, 2)

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)
    
    def embed(self, g, x):
        import dgl
        h = self.sage(g, x)
        g = dgl.add_self_loop(g)
        return self.embedding(g, h)


class RGCNEmbed(Module):  # type: ignore
    def __init__(self, d, num_nodes, num_rels, hidden=None, device='cpu'):
        super().__init__()

        nn, _, dglnn, _, torch, _, _ = lazy_import_networks()
        self.node_ids = torch.tensor(range(num_nodes))
        
        self.node_ids = self.node_ids.to(device)
        self.emb = nn.Embedding(num_nodes, d)
        hidden = d if not hidden else d + hidden

        # TODO: need to think something about the self loop
        self.rgc1 = dglnn.RelGraphConv(d, d, num_rels, regularizer='bdd', num_bases=d, self_loop=True)
        self.rgc2 = dglnn.RelGraphConv(hidden, d, num_rels, self_loop=True)

        self.dropout = nn.Dropout(0.2)

    def forward(self, g, node_features=None):

        _, dgl, _, _, torch, F, _ = lazy_import_networks()

        x = self.emb(self.node_ids)
        x = self.rgc1(g, x, g.edata[dgl.ETYPE], g.edata['norm'])
        if node_features is not None:
            x = F.relu(torch.cat([x, node_features], dim=1))
        else:
            x = F.relu(x)
        x = self.rgc2(g, self.dropout(x), g.edata[dgl.ETYPE], g.edata['norm'])
        return self.dropout(x)


class HeteroEmbed(Module):  # type: ignore
    def __init__(
        self,
        num_nodes: int,
        num_rels: int,
        d: int,
        proto,
        node_features = None,
        device = 'cpu',
        reg = 0.01
    ):
        super().__init__()
        nn, _, _, _, torch, _, _ = lazy_import_networks()
        self.reg = reg
        self.proto = proto
        self.node_features = node_features

        if self.node_features is not None:
            self.node_features = torch.tensor(
                self.node_features.values, dtype=torch.float32
            ).to(device)
            logger.info(f"--Using node features of shape {str(node_features.shape)}")  # type: ignore
        hidden = None
        if node_features is not None:
            hidden = self.node_features.shape[-1]  # type: ignore
        self.rgcn = RGCNEmbed(d, num_nodes, num_rels, hidden, device=device)
        self.relational_embedding = nn.Parameter(torch.Tensor(num_rels, d))

        nn.init.xavier_uniform_(
            self.relational_embedding, gain=nn.init.calculate_gain("relu")
        )

    def __call__(self, g):  # type: ignore
        # returns node embeddings
        return self.rgcn.forward(g, node_features=self.node_features)

    def score(self, node_embedding, triplets):
        h, r, t = triplets.T  # type: ignore
        h, r, t = (node_embedding[h], self.relational_embedding[r], node_embedding[t])  # type: ignore
        score = self.proto(h, r, t)
        return score

    def loss(self, node_embedding, triplets, labels):
        _, _, _, _, torch, F, _ = lazy_import_networks()
        score = self.score(node_embedding, triplets)

        # binary crossentropy loss
        ce_loss = F.binary_cross_entropy_with_logits(score, labels)

        # regularization loss
        ne_ = torch.mean(node_embedding.pow(2))  # type: ignore
        re_ = torch.mean(self.relational_embedding.pow(2))
        rl = ne_ + re_
        
        return ce_loss + self.reg * rl


############################################################################################

# training

#from sklearn import metrics 
#MAE = metrics.mean_absolute_error
#ACC = metrics.accuracy_score
   
def train_link_pred(model, G, epochs=100, use_cross_entropy_loss = False):
    _, _, _, _, torch, F, _ = lazy_import_networks()
    # take the node features out
    node_features = G.ndata["feature"].float()
    # we are predicting edges
    edge_label = G.edata["target"]

    n_targets = edge_label.shape[1]
    labels = edge_label.argmax(1)
    train_mask = G.edata["train_mask"]
    test_mask = G.edata["test_mask"]

    if edge_label.shape[1] > 1:
        print(f'Predicting {n_targets} target multiOutput')
    else:
        print(f'Predicting {n_targets} target')
    
    opt = torch.optim.Adam(model.parameters())

    # train the model
    for epoch in range(epochs):
        logits = model(G, node_features)

        if use_cross_entropy_loss:
            loss = F.cross_entropy(logits[train_mask], edge_label[train_mask])
        else:  # in regressive context
            loss = ((logits[train_mask] - edge_label[train_mask]) ** 2).mean()
            
        p = logits.argmax(1)
        acc = sum(p[test_mask] == labels[test_mask]) / len(p[test_mask])

        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 100 == 0:
            #evaluate(model, G, node_features, logits, test_mask)
            print(
                f"epoch: {epoch} --------\nloss: {loss.item():.4f}\n\t Accuracy: {acc:.4f} across {n_targets} targets"
            )
