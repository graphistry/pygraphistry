from typing import TYPE_CHECKING
import torch.nn as nn

if TYPE_CHECKING:
    import dgl
    import dgl.nn as dglnn
    import dgl.function as fn
    import torch
    import torch.nn.functional as F

from . import constants as config

def lazy_import_networks():
    import dgl
    import dgl.nn as dglnn
    import dgl.function as fn
    import torch
    import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(GCN, self).__init__()
        lazy_import_networks()
        self.conv1 = dglnn.GraphConv(in_feats, h_feats)
        self.conv2 = dglnn.GraphConv(h_feats, num_classes)

    def forward(self, g, in_feat):
        h = self.conv1(g, in_feat)
        h = F.relu(h)
        h = self.conv2(g, h)
        return h


class RGCN(nn.Module):
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
        lazy_import_networks()

        self.conv1 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(in_feats, hid_feats) for rel in rel_names},
            aggregate="sum",
        )
        self.conv2 = dglnn.HeteroGraphConv(
            {rel: dglnn.GraphConv(hid_feats, out_feats) for rel in rel_names},
            aggregate="sum",
        )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


class HeteroClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, rel_names):
        super().__init__()
        lazy_import_networks()
        self.rgcn = RGCN(in_dim, hidden_dim, hidden_dim, rel_names)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata[config.FEATURE]
        h = self.rgcn(g, h)
        with g.local_scope():  # create a local scope to hold onto hidden layer 'h'
            g.ndata["h"] = h
            # Calculate graph representation by average readout.
            hg = 0
            for ntype in g.ntypes:
                hg = hg + dgl.mean_nodes(g, "h", ntype=ntype)
            return self.classify(hg)


class MLPPredictor(nn.Module):
    """One can also write a prediction function that predicts a vector for each edge with an MLP.
    Such vector can be used in further downstream tasks, e.g. as logits of a categorical distribution."""

    def __init__(self, in_features, out_classes):
        super().__init__()
        lazy_import_networks()
        self.W = nn.Linear(in_features * 2, out_classes)

    def apply_edges(self, edges):
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


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        lazy_import_networks()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type="mean"
        )
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats, out_feats=out_feats, aggregator_type="mean"
        )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata["h"] = h
            graph.apply_edges(fn.u_dot_v("h", "h", "score"))
            return graph.edata["score"]


class LinkPredModel(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        lazy_import_networks()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


class LinkPredModelMultiOutput(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, out_classes):
        super().__init__()
        lazy_import_networks()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = MLPPredictor(out_features, out_classes)
        self.embedding = dglnn.GraphConv(out_features, 2)

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)
    
    def embed(self, g, x):
        h = self.sage(g, x)
        g = dgl.add_self_loop(g)
        return self.embedding(g, h)


############################################################################################

# training

#from sklearn import metrics 
#MAE = metrics.mean_absolute_error
#ACC = metrics.accuracy_score
   
def train_link_pred(model, G, epochs=10000, use_cross_entropy_loss = False):
    lazy_import_networks()
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
