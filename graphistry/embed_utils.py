import logging
import numpy as np
import pandas as pd
from typing import Optional, Union, Callable, List, TYPE_CHECKING, Any, Tuple

from .PlotterBase import Plottable
from .compute.ComputeMixin import ComputeMixin

def lazy_embed_import_dep():
    try:
        import torch
        import torch.nn as nn
        import dgl
        from dgl.dataloading import GraphDataLoader
        import torch.nn.functional as F
        from .networks import HeteroEmbed
        from tqdm import trange  # type: ignore
        return True, torch, nn, dgl, GraphDataLoader, HeteroEmbed, F, trange

    except:
        return False, None, None, None, None, None, None, None


if TYPE_CHECKING:
    _, torch, _, _, _, _, _, _ = lazy_embed_import_dep()
    TT = torch.Tensor
    MIXIN_BASE = ComputeMixin
else:
    TT = Any
    MIXIN_BASE = object

XSymbolic = Optional[Union[List[str], str, pd.DataFrame]]
ProtoSymbolic = Optional[Union[str, Callable[[TT, TT, TT], TT]]]  # type: ignore

logging.StreamHandler.terminator = ""
logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)


def log(msg:str) -> None:
    # setting every logs to WARNING level
    logger.log(msg=msg, level=30)


class EmbedDistScore:
    @staticmethod
    def TransE(h:TT, r:TT, t:TT) -> TT:  # type: ignore
        return (h + r - t).norm(p=1, dim=1)  # type: ignore

    @staticmethod
    def DistMult(h:TT, r:TT, t:TT) -> TT:  # type: ignore
        return (h * r * t).sum(dim=1)  # type: ignore

    @staticmethod
    def RotatE(h:TT, r:TT, t:TT) -> TT:  # type: ignore
        return -(h * r - t).norm(p=1, dim=1)  # type: ignore


class HeterographEmbedModuleMixin(MIXIN_BASE):
    def __init__(self):
        super().__init__()

        self.protocol = {
            "TransE": EmbedDistScore.TransE,
            "DistMult": EmbedDistScore.DistMult,
            "RotatE": EmbedDistScore.RotatE,
        }

        self._node2id = {}
        self._relation2id = {}
        self._id2node = {}
        self._id2relation = {}

        self._kg_embeddings = None
        self._embed_model = None

        self.train_idx = None
        self.test_idx = None
        

    def _preprocess_embedding_data(self, res, train_split:Union[float, int] = 0.8) -> Plottable:
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        log('Preprocessing embedding data')
        src, dst = res._source, res._destination
        relation = res._relation

        if res._node is not None and res._nodes is not None:
            nodes = res._nodes[self._node]
        elif res._node is None and res._nodes is not None:
            nodes = res._nodes.reset_index(drop=True).reset_index()["index"]
            #print('None but not None', nodes)
        else:
            res = res.materialize_nodes()
            nodes = res._nodes[res._node]
            #print('Materialize nodes')
        
        edges = res._edges
        edges = edges[edges[src].isin(nodes) & edges[dst].isin(nodes)]
        relations = edges[relation].unique()

        # type2id
        res._node2id = {n: idx for idx, n in enumerate(nodes)}
        res._relation2id = {r: idx for idx, r in enumerate(relations)}

        res._id2node = {idx: n for idx, n in enumerate(nodes)}
        res._id2relation = {idx: r for idx, r in enumerate(relations)}

        s, r, t = (
            edges[src].map(res._node2id),
            edges[relation].map(res._relation2id),
            edges[dst].map(res._node2id),
        )
        triplets = torch.from_numpy(pd.concat([s, r, t], axis=1).to_numpy())

        # split idx
        if not hasattr(res, "train_idx") or res._train_split != train_split:
            log(msg="--Splitting data")
            train_size = int(train_split * len(triplets))
            test_size = len(triplets) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(triplets, [train_size, test_size])  # type: ignore
            res.train_idx = train_dataset.indices
            res.test_idx = test_dataset.indices

        res.triplets = triplets
        res._num_nodes, res._num_rels = (len(res._node2id), len(res._relation2id))
        log(
            f"--num_nodes: {res._num_nodes}, num_relationships: {res._num_rels}")
        return res

    def _build_graph(self, res:Plottable) -> Plottable:
        _, torch, _, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        s, r, t = res.triplets.T  # type: ignore

        if res.train_idx is not None:  # type: ignore
            g_dgl = dgl.graph(
                (s[res.train_idx], t[res.train_idx]), num_nodes=self._num_nodes  # type: ignore
            )
            g_dgl.edata[dgl.ETYPE] = r[res.train_idx]  # type: ignore

        else:
            g_dgl = dgl.graph(
                (s, t), num_nodes=self._num_nodes  # type:ignore
            )
            g_dgl.edata[dgl.ETYPE] = r  # type: ignore

        g_dgl.edata["norm"] = dgl.norm_by_dst(g_dgl).unsqueeze(-1)
        res.g_dgl = g_dgl  # type: ignore
        return res

    def _init_model(self, res:Plottable, batch_size:int, sample_size:int, num_steps:int, device):
        _, torch, nn, dgl, GraphDataLoader, HeteroEmbed, F, _ = lazy_embed_import_dep()
        g_iter = SubgraphIterator(res.g_dgl, sample_size, num_steps)  # type: ignore
        g_dataloader = GraphDataLoader(
            g_iter, batch_size=batch_size, collate_fn=lambda x: x[0]
        )

        # init model
        model = HeteroEmbed(
            res._num_nodes,  # type: ignore
            res._num_rels,   # type: ignore
            res._kg_embed_dim,
            proto=res.proto,  # type: ignore
            node_features=res._node_features,
            device=device,
        )

        return model, g_dataloader

    def _train_embedding(self, res:Plottable, epochs:int, batch_size:int, lr:float, sample_size:int, num_steps:int, device) -> Plottable:
        _, torch, nn, dgl, GraphDataLoader, _, F, trange = lazy_embed_import_dep()
        log('Training embedding')
        model, g_dataloader = res._init_model(res, batch_size, sample_size, num_steps, device)  # type: ignore
        if hasattr(res, "_embed_model") and not res._build_new_embedding_model:  # type: ignore
            model = res._embed_model
            log("--Reusing previous model")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        pbar = trange(epochs, desc=None)
        model.to(device)

        score = 0
        for epoch in pbar:
            model.train()
            for data in g_dataloader:
                g, edges, labels = data

                g = g.to(device)
                edges = edges.to(device)
                labels = labels.to(device)

                emb = model(g)
                loss = model.loss(emb, edges, labels)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                pbar.set_description(
                    f"epoch: {epoch+1}, loss: {loss.item():.4f}, score: {score:.4f}%"
                )

            model.eval()
            res._kg_embeddings = model(res.g_dgl.to(device)).detach()  # type: ignore
            res._embed_model = model  # type: ignore
            if res._eval_flag:  # type: ignore
                score = res._eval(threshold=0.5)  # type: ignore
                pbar.set_description(
                    f"epoch: {epoch+1}, loss: {loss.item():.4f}, score: {score:.2f}%"
                )

        return res

    def embed(
        self,
        relation:str,
        proto:ProtoSymbolic = 'DistMult',
        embedding_dim:Optional[int] = 32,
        use_feat:Optional[bool] = False,
        X:XSymbolic = None,
        epochs:Optional[int] = 2,
        batch_size:Optional[int] = 32,
        train_split:Optional[Union[float, int]] = 0.8,
        sample_size:int = 1000, 
        num_steps:int = 50,
        lr:Optional[float] = 1e-2,
        inplace:Optional[bool] = False,
        device:Optional['str'] = "cpu",
        evaluate:Optional[bool] = True,
        *args,
        **kwargs,
    ) -> Plottable:
        """Embed a graph using a relational graph convolutional network (RGCN),
        and return a new graphistry graph with the embeddings as node
        attributes.


        Parameters
        ----------
        relation : str
            column to use as relation between nodes
        proto : ProtoSymbolic
            metric to use, ['TransE', 'RotateE', 'DistMult'] or provide your own. Defaults to 'DistMult'.
        embedding_dim : Optional[int]
            relation embedding dimension. defaults to 32
        use_feat : Optional[bool]
            wether to featurize nodes, if False will produce random embeddings and shape them during training.
            Defaults to True
        X : XSymbolic
            Which columns in the nodes dataframe to featurize. Inherets args from graphistry.featurize().
            Defaults to None.
        epochs : Optional[int]
            Number of training epochs. Defaults to 2
        batch_size : Optional[int]
            batch_size. Defaults to 32
        train_split : Optional[Union[float, int]]
            train percentage, between 0, 1. Defaults to 0.8.
        lr : Optional[float]
            learning rate. Defaults to 0.002
        inplace : Optional[bool]
            inplace
        device : Optional[str]
            accelarator. Defaults to "cpu"
        evaluate : Optional[bool]
            Whether to evaluate. Defaults to False.

        Returns
        -------
            self : graphistry instance
        """
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        if inplace:
            res = self
        else:
            res = self.bind()
        
        requires_new_model = False
        if res._relation != relation:
            requires_new_model = True
            res._relation = relation
        if res._use_feat != use_feat:
            requires_new_model = True
            res._use_feat = use_feat  # type: ignore
        if res._kg_embed_dim != embedding_dim:
            requires_new_model = True
            res._kg_embed_dim = embedding_dim  # type: ignore
        res._build_new_embedding_model = requires_new_model  # type: ignore
        res._train_split = train_split  # type: ignore
        res._eval_flag = evaluate  # type: ignore

        if callable(proto):
            res.proto = proto  # type: ignore
        else:
            res.proto = res.protocol[proto]  # type: ignore

        if res._use_feat and res._nodes is not None:
            # todo decouple self from res
            res = res.featurize(kind="nodes", X=X, *args, **kwargs)  # type: ignore

        if not hasattr(res, "triplets") or res._build_new_embedding_model:  # type: ignore
            res = res._preprocess_embedding_data(res, train_split=train_split)  # type: ignore
            res = res._build_graph(res)  # type: ignore

        return res._train_embedding(res, epochs, batch_size, lr=lr, sample_size=sample_size, num_steps=num_steps,device=device)  # type: ignore

    def calculate_prob(
        self, test_triplet, test_triplets, threshold, h_r, node_embeddings, infer=None
    ):
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        # TODO: simplify
        if infer == "all":
            s, r, o_ = test_triplet
        else:
            s, r = test_triplet

        subject_relation = test_triplet[:2]
        num_entity = len(node_embeddings)
        delete_idx = torch.sum(h_r == subject_relation, dim=1)
        delete_idx = torch.nonzero(delete_idx == 2).squeeze()
        delete_entity_idx = test_triplets[delete_idx, 2].view(-1).numpy()
        perturb_entity_idx = np.array(
            list(set(np.arange(num_entity)) - set(delete_entity_idx))
        )
        perturb_entity_idx = torch.from_numpy(perturb_entity_idx).squeeze()

        if infer == "all":
            perturb_entity_idx = torch.cat((perturb_entity_idx, torch.unsqueeze(o_, 0)))

        o = self.proto(
            node_embeddings[s],
            self._embed_model.relational_embedding[r],
            node_embeddings[perturb_entity_idx],
        )

        score = torch.sigmoid(o)
        return perturb_entity_idx[score > threshold]

    def _predict(self, test_triplets, threshold=0.95, directed=True, infer=None):
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()

        if type(test_triplets) != torch.Tensor:
            test_triplets = torch.tensor(test_triplets)
        triplets = self.triplets

        s, r, o = triplets.T
        edge_index = torch.stack([s, o])

        # make graph
        g = dgl.graph((s, o), num_nodes=edge_index.max() + 1)
        g.edata[dgl.ETYPE] = r
        g.edata["norm"] = dgl.norm_by_dst(g).unsqueeze(-1)
        del s, r, o

        node_embeddings = self._embed_model(g)

        h_r = triplets[:, :2]
        t_r = torch.stack((triplets[:, 2], triplets[:, 1])).transpose(0, 1)

        visited, predicted_links = {}, []
        for test_triplet in test_triplets:
            s, r, o_ = test_triplet
            k = "".join([str(s), "_", str(r)])
            kr = "".join([str(r), "_", str(s)])

            # for [s, r] -> {d}
            if k not in visited:

                links = self.calculate_prob(
                    test_triplet, test_triplets, threshold, h_r, node_embeddings, infer
                )
                visited[k] = ""
                predicted_links += [
                    [
                        self._id2node[s.item()],
                        self._id2relation[r.item()],
                        self._id2node[i.item()],
                    ]
                    for i in links
                ]

            # for [d, r] -> {s}
            if kr not in visited and not directed:
                links = self.calculate_prob(
                    test_triplet, test_triplets, threshold, t_r, node_embeddings, infer
                )
                visited[k] = ""
                predicted_links += [
                    [
                        self._id2node[s.item()],
                        self._id2relation[r.item()],
                        self._id2node[i.item()],
                    ]
                    for i in links
                ]

        predicted_links = pd.DataFrame(
            predicted_links, columns=[self._source, self._relation, self._destination]
        )
        return predicted_links, node_embeddings

    def predict_link(
        self,
        test_df: pd.DataFrame,
        src:str,
        rel:str,
        threshold:Optional[float] = 0.95
    ) -> pd.DataFrame:
        """predict links from a test dataframe given src/dst and rel columns

        Parameters
        ----------
        test_df : pd.DataFrame
            dataframe of test data
        src : str
            source column name
        rel : str
            relation column name
        threshold : Optional[float]
            Probability threshold/confidence. Defaults to 0.95.

        Returns
        -------
        pd.DataFrame
            dataframe containing predicted links

        """
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        pred = "predicted_destination"
        nodes = test_df[src].map(self._node2id)
        relations = test_df[rel].map(self._relation2id)

        all_nodes = self._node2id.values()
        test_df = pd.concat([nodes, relations], axis=1)
        test_df[pred] = [all_nodes] * len(test_df)
        test_df = test_df.explode(pred)
        test_df = test_df[test_df[src] != test_df[pred]]
        score = self._score(
            torch.from_numpy(test_df.to_numpy().astype(np.float32)).to(dtype=torch.long)
        )
        result_df = test_df.loc[pd.Series(score.detach().numpy()) >= threshold]  # type: ignore
        s, r, d = (
            test_df[src].map(self._id2node),
            test_df[rel].map(self._id2relation),
            test_df[pred].map(self._id2node),
        )
        result_df = pd.concat([s, r, d], axis=1)
        result_df.columns = [src, rel, pred]  # type: ignore
        return result_df

    def predict_links(
        self, 
        threshold:Optional[float] = 0.99,
        return_embeddings:Optional[bool] = True,
        retain_old_edges:Optional[bool] = False
    ) -> Union[Tuple[Plottable, pd.DataFrame, TT], Plottable]:  # type: ignore
        """predict_links over entire graph given a threshold

        Parameters
        ----------
        threshold : Optional[float]
            Probability threshold. Defaults to 0.99.
        return_embeddings : Optional[bool]
            will return DataFrame of predictions and node_embeddings. Defaults
            to True
        retain_old_edges : Optional[bool]
            will include old edges in predicted graph. Defaults to False.

        Returns
        -------
        Union[Tuple[Plottable, pd.DataFrame, TT], Plottable]
            graphistry graph or (graphistry graph, DataFrame of predicted
            links, node embeddings) when return_embeddings=True

        """
       
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        predicted_links, node_embeddings = self._predict(
            self.triplets, threshold, infer="all"
        )

        existing_links = self._edges[[self._source, self._relation, self._destination]]

        if retain_old_edges:
            all_links = pd.concat(
                [existing_links, predicted_links], ignore_index=True
            ).drop_duplicates()
        else:
            all_links = predicted_links

        g_new = self.nodes(self._nodes, self._node)
        g_new = g_new.edges(all_links, self._source, self._destination)

        if return_embeddings:
            return g_new, predicted_links, node_embeddings
        return g_new

    def _score(self, triplets:Union[np.ndarray, TT]) -> TT:  # type: ignore
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        emb = self._kg_embeddings.clone().detach()
        if type(triplets) != torch.Tensor:
            triplets = torch.tensor(triplets)
        score = self._embed_model.score(emb, triplets)
        prob = torch.sigmoid(score)
        return prob.detach()

    def _eval(self, threshold:float):
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        if self.test_idx != []:
            triplets = self.triplets[self.test_idx]  # type: ignore
            score = self._score(triplets)
            score = 100 * len(score[score >= threshold]) / len(score)  # type: ignore
            return score
        else:
            log("WARNING: train_split must be < 1 for _eval()")


class SubgraphIterator:
    def __init__(self, g, sample_size:int = 30000, num_steps:int = 1000):
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        self.num_steps = num_steps
        self.sample_size = int(sample_size / 2)
        self.eids = np.arange(g.num_edges())
        self.g = g
        self.num_nodes = g.num_nodes()

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, i:int):
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        eids = torch.from_numpy(np.random.choice(self.eids, self.sample_size))

        src, dst = self.g.find_edges(eids)
        rel = self.g.edata[dgl.ETYPE][eids].numpy()

        triplets = np.stack((src, rel, dst)).T
        samples, labels = SubgraphIterator._sample_neg(
            triplets,
            self.num_nodes,
        )

        src, rel, dst = samples.T  # type: ignore

        # might need to add bidirectional edges
        sub_g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        sub_g.edata[dgl.ETYPE] = rel
        sub_g.edata["norm"] = dgl.norm_by_dst(sub_g).unsqueeze(-1)

        return sub_g, samples, labels

    @staticmethod
    def _sample_neg(triplets:np.ndarray, num_nodes:int) -> Tuple[TT, TT]:  # type: ignore
        _, torch, nn, dgl, GraphDataLoader, _, F, _ = lazy_embed_import_dep()
        triplets = torch.tensor(triplets)  # type: ignore
        h, r, t = triplets.T
        h_o_t = torch.randint(high=2, size=h.size())

        random_h = torch.randint(high=num_nodes, size=h.size())
        random_t = torch.randint(high=num_nodes, size=h.size())

        neg_h = torch.where(h_o_t == 0, random_h, h)
        neg_t = torch.where(h_o_t == 1, random_t, t)
        neg_triplets = torch.stack((neg_h, r, neg_t), dim=1)

        all_triplets = torch.cat((triplets, neg_triplets), dim=0)  # type: ignore
        labels = torch.zeros((all_triplets.size()[0]))
        labels[: triplets.shape[0]] = 1
        return all_triplets, labels
