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
        from tqdm import trange
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
    torch = Any

XSymbolic = Optional[Union[List[str], str, pd.DataFrame]]
ProtoSymbolic = Optional[Union[str, Callable[[TT, TT, TT], TT]]]  # type: ignore

logging.StreamHandler.terminator = ""
logger = logging.getLogger(__name__)


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

        self._protocol = {
            "TransE": EmbedDistScore.TransE,
            "DistMult": EmbedDistScore.DistMult,
            "RotatE": EmbedDistScore.RotatE,
        }

        self._node2id = {}
        self._relation2id = {}
        self._id2node = {}
        self._id2relation = {}
        self._relation = None
        self._use_feat = False
        self._kg_embed_dim = None
        self._kg_embeddings = None
        
        self._embed_model = None

        self._train_idx = None
        self._test_idx = None

        self._num_nodes = None
        self._train_split = None
        self._eval_flag = None

        self._build_new_embedding_model = None
        self._proto = None
        self._device = "cpu"

    def _preprocess_embedding_data(self, res, train_split:Union[float, int] = 0.8) -> Plottable:
        _, torch, _, _, _, _, F, _ = lazy_embed_import_dep()
        log('Preprocessing embedding data')
        src, dst = res._source, res._destination
        relation = res._relation

        if res._node is not None and res._nodes is not None:
            nodes = res._nodes[self._node]
        elif res._node is None and res._nodes is not None:
            nodes = res._nodes.reset_index(drop=True).reset_index()["index"]
        else:
            res = res.materialize_nodes()
            nodes = res._nodes[res._node]
        
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
        if res._train_idx is None or res._train_split != train_split:
            log(msg="--Splitting data")
            train_size = int(train_split * len(triplets))
            test_size = len(triplets) - train_size
            train_dataset, test_dataset = torch.utils.data.random_split(triplets, [train_size, test_size])
            res._train_idx = train_dataset.indices
            res._test_idx = test_dataset.indices

        res._triplets = triplets
        res._num_nodes, res._num_rels = (len(res._node2id), len(res._relation2id))
        log(
            f"--num_nodes: {res._num_nodes}, num_relationships: {res._num_rels}")
        return res

    def _build_graph(self, res) -> Plottable:
        _, _, _, dgl, _, _, _, _ = lazy_embed_import_dep()
        s, r, t = res._triplets.T

        if res._train_idx is not None:
            g_dgl = dgl.graph(
                (s[res._train_idx], t[res._train_idx]), num_nodes=res._num_nodes  # type: ignore

            )
            g_dgl.edata[dgl.ETYPE] = r[res._train_idx]

        else:
            g_dgl = dgl.graph(
                (s, t), num_nodes=res._num_nodes  # type:ignore
            )
            g_dgl.edata[dgl.ETYPE] = r

        g_dgl.edata["norm"] = dgl.norm_by_dst(g_dgl).unsqueeze(-1)
        res.g_dgl = g_dgl
        return res


    def _init_model(self, res, batch_size:int, sample_size:int, num_steps:int, device):
        _, _, _, _, GraphDataLoader, HeteroEmbed, _, _ = lazy_embed_import_dep()
        g_iter = SubgraphIterator(res.g_dgl, sample_size, num_steps)
        g_dataloader = GraphDataLoader(
            g_iter, batch_size=batch_size, collate_fn=lambda x: x[0]
        )

        # init model
        model = HeteroEmbed(
            res._num_nodes,
            res._num_rels,
            res._kg_embed_dim,
            proto=res._proto,
            node_features=res._node_features,
            device=device,
        )

        return model, g_dataloader

    def _train_embedding(self, res, epochs:int, batch_size:int, lr:float, sample_size:int, num_steps:int, device) -> Plottable:
        _, torch, nn, _, _, _, _, trange = lazy_embed_import_dep()
        log('Training embedding')
        model, g_dataloader = res._init_model(res, batch_size, sample_size, num_steps, device)
        if hasattr(res, "_embed_model") and not res._build_new_embedding_model:
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
                    f"epoch: {epoch+1}, loss: {loss.item():.4f}, score: {100*score:.4f}%"
                )

            model.eval()
            res._kg_embeddings = model(res.g_dgl.to(device)).detach()
            res._embed_model = model
            if res._eval_flag and res._train_idx is not None:
                score = res._eval(threshold=0.5)
                pbar.set_description(
                    f"epoch: {epoch+1}, loss: {loss.item():.4f}, score: {100*score:.2f}%"
                )

        return res

    @property
    def _gcn_node_embeddings(self):
        _, torch, _, _, _, _, _, _ = lazy_embed_import_dep()
        g_dgl = self.g_dgl.to(self._device)
        em = self._embed_model(g_dgl).detach()
        torch.cuda.empty_cache()
        return em

    def embed(
        self,
        relation:str,
        proto: ProtoSymbolic = 'DistMult',
        embedding_dim: int = 32,
        use_feat: bool = False,
        X: XSymbolic = None,
        epochs: int = 2,
        batch_size: int = 32,
        train_split: Union[float, int] = 0.8,
        sample_size: int = 1000, 
        num_steps: int = 50,
        lr: float = 1e-2,
        inplace: Optional[bool] = False,
        device: Optional['str'] = "cpu",
        evaluate: bool = True,
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
        embedding_dim : int
            relation embedding dimension. defaults to 32
        use_feat : bool
            wether to featurize nodes, if False will produce random embeddings and shape them during training.
            Defaults to True
        X : XSymbolic
            Which columns in the nodes dataframe to featurize. Inherets args from graphistry.featurize().
            Defaults to None.
        epochs : int
            Number of training epochs. Defaults to 2
        batch_size : int
            batch_size. Defaults to 32
        train_split : Union[float, int]
            train percentage, between 0, 1. Defaults to 0.8.
        sample_size : int
            sample size. Defaults to 1000
        num_steps : int
            num_steps. Defaults to 50
        lr : float
            learning rate. Defaults to 0.002
        inplace : Optional[bool]
            inplace
        device : Optional[str]
            accelarator. Defaults to "cpu"
        evaluate : bool
            Whether to evaluate. Defaults to False.

        Returns
        -------
            self : graphistry instance
        """
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
            res._use_feat = use_feat
        if res._kg_embed_dim != embedding_dim:
            requires_new_model = True
            res._kg_embed_dim = embedding_dim
        res._build_new_embedding_model = requires_new_model
        res._train_split = train_split
        res._eval_flag = evaluate
        res._device = device

        if callable(proto):
            res._proto = proto
        else:
            res._proto = res._protocol[proto]

        if res._use_feat and res._nodes is not None:
            res = res.featurize(kind="nodes", X=X, *args, **kwargs)  # type: ignore

        if not hasattr(res, "_triplets") or res._build_new_embedding_model:
            res = res._preprocess_embedding_data(res, train_split=train_split)  # type: ignore
            res = res._build_graph(res)  # type: ignore

        return res._train_embedding(res, epochs, batch_size, lr=lr, sample_size=sample_size, num_steps=num_steps, device=device)  # type: ignore


    def _score_triplets(self, triplets, threshold, anomalous, retain_old_edges, return_dataframe):
        """Score triplets using the trained model."""
        
        log(f"{triplets.shape[0]} triplets for inference")
        ############################################################
        # the bees knees 
        scores = self._score(triplets)
        ############################################################
        if len(triplets) > 1:
            if anomalous:
                predicted_links = triplets[scores < threshold]  # type: ignore
                this_score = scores[scores < threshold]  # type: ignore
            else:
                predicted_links = triplets[scores > threshold]  # type: ignore
                this_score = scores[scores > threshold]  # type: ignore
        else:
            predicted_links = triplets
            this_score = scores
            
        predicted_links = pd.DataFrame(predicted_links, columns=[self._source, self._relation, self._destination])
        predicted_links[self._source] = predicted_links[self._source].map(self._id2node)
        predicted_links[self._relation] = predicted_links[self._relation].map(self._id2relation)
        predicted_links[self._destination] = predicted_links[self._destination].map(self._id2node)

        predicted_links['score'] = this_score.detach().numpy()
        predicted_links.sort_values(by='score', ascending=False, inplace=True)
        
        log(f"-- {predicted_links.shape[0]} triplets scored at threshold {threshold:.2f}")

        if retain_old_edges:
            existing_links = self._edges[[self._source, self._relation, self._destination]]
            all_links = pd.concat(
                [existing_links, predicted_links], ignore_index=True
            ).drop_duplicates()
        else:
            all_links = predicted_links
            
        
        if return_dataframe:
            return all_links
        else:
            g_new = self.nodes(self._nodes, self._node).edges(all_links, self._source, self._destination)
            return g_new
        
        
    def predict_links(
        self, 
        source: Union[list, None] = None,
        relation: Union[list, None] = None,
        destination: Union[list, None] = None,
        threshold: Optional[float] = 0.5,
        anomalous: Optional[bool] = False,
        retain_old_edges: Optional[bool] = False,
        return_dataframe: Optional[bool] = False
    ) -> Plottable:  # type: ignore
        """predict_links over all the combinations of given source, relation, destinations.

        Parameters
        ----------
        source: list
            Targeted source nodes. Defaults to None(all).
        relation: list
            Targeted relations. Defaults to None(all).
        destination: list
            Targeted destinations. Defaults to None(all).
        threshold : Optional[float]
            Probability threshold. Defaults to 0.5
        retain_old_edges : Optional[bool]
            will include old edges in predicted graph. Defaults to False.
        return_dataframe : Optional[bool]
            will return a dataframe instead of a graphistry instance. Defaults to False.
        anomalous : Optional[False]
            will return the edges < threshold or low confidence edges(anomaly).

        Returns
        -------
        Graphistry Instance
            containing the corresponding source, relation, destination and score column
            where score >= threshold if anamalous if False else score <= threshold, or a dataframe
            
        """

        all_nodes = self._node2id.values()
        all_relations = self._relation2id.values()

        if source is None:
            src = pd.Series(all_nodes)
        else:
            src = pd.Series(source)
            src = src.map(self._node2id)

        if relation is None:
            rel = pd.Series(all_relations)
        else:
            rel = pd.Series(relation)
            rel = rel.map(self._relation2id)

        if destination is None:
            dst = pd.Series(all_nodes)
        else:
            dst = pd.Series(destination)
            dst = dst.map(self._node2id)

        def fetch_triplets_for_inference(source, relation, destination):
            source = pd.DataFrame(source.unique(), columns=['source'])
            source['relation'] = [relation.unique()] * source.shape[0]

            source_with_relation = source.explode('relation')
            source_with_relation['destination'] = [destination.unique()] * source_with_relation.shape[0]

            triplets = source_with_relation.explode('destination')
            triplets = triplets[triplets['source'] != triplets['destination']]

            return triplets.drop_duplicates().reset_index(drop=True)

        triplets = fetch_triplets_for_inference(src, rel, dst)
        triplets = triplets.to_numpy().astype(np.int64)
        
        return self._score_triplets(triplets, threshold, anomalous, retain_old_edges, return_dataframe)
 

    def predict_links_all(
        self, 
        threshold: Optional[float] = 0.5,
        anomalous: Optional[bool] = False,
        retain_old_edges: Optional[bool] = False,
        return_dataframe: Optional[bool] = False
    ) -> Plottable:  # type: ignore
        """predict_links over entire graph given a threshold

        Parameters
        ----------
        threshold : Optional[float]
            Probability threshold. Defaults to 0.5
        anomalous : Optional[False]
            will return the edges < threshold or low confidence edges(anomaly).
        retain_old_edges : Optional[bool]
            will include old edges in predicted graph. Defaults to False.
        return_dataframe: Optional[bool]
            will return a dataframe instead of a graphistry instance. Defaults to False.

        Returns
        -------
        Plottable
            graphistry graph instance containing all predicted/anomalous links or dataframe

        """
        h_r = pd.DataFrame(self._triplets.numpy())  # type: ignore
        t_r = h_r.copy()
        t_r[[0,1,2]] = t_r[[2,1,0]]

        all_nodes = set(self._node2id.values())

        def fetch_triplets_for_inference(x_r):
            existing_collapsed = pd.DataFrame(
                x_r.groupby(by=[0, 1])[2].apply(set)
            ).reset_index()

            non_existing_collapsed = existing_collapsed[2].map(lambda x: set(all_nodes).difference(x))
            triplets_for_inference = pd.concat(
                [
                    existing_collapsed[[0, 1]], 
                    non_existing_collapsed
                ], axis=1
            ).explode(2)

            return triplets_for_inference

        triplets = pd.concat([fetch_triplets_for_inference(h_r), fetch_triplets_for_inference(t_r)], axis=0)
        triplets = triplets[triplets[0] < triplets[2]]
        triplets = triplets.drop_duplicates().to_numpy().astype(np.int64)

        return self._score_triplets(triplets, threshold, anomalous, retain_old_edges, return_dataframe)
        

    def _score(self, triplets: Union[np.ndarray, TT]) -> TT:  # type: ignore
        _, torch, _, _, _, _, _, _ = lazy_embed_import_dep()
        emb = self._kg_embeddings.clone().detach()
        if type(triplets) != torch.Tensor:
            triplets = torch.tensor(triplets)
        score = self._embed_model.score(emb, triplets)
        prob = torch.sigmoid(score)
        return prob.detach()


    def _eval(self, threshold: float):
        if self._test_idx is not None:
            triplets = self._triplets[self._test_idx]  # type: ignore
            score = self._score(triplets)
            score = len(score[score >= threshold]) / len(score)  # type: ignore
            return score
        else:
            log("WARNING: train_split must be < 1 for _eval()")


class SubgraphIterator:
    def __init__(self, g, sample_size:int = 3000, num_steps:int = 1000):
        self.num_steps = num_steps
        self.sample_size = sample_size
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

        sub_g = dgl.graph((src, dst), num_nodes=self.num_nodes)
        sub_g.edata[dgl.ETYPE] = rel
        sub_g.edata["norm"] = dgl.norm_by_dst(sub_g).unsqueeze(-1)

        return sub_g, samples, labels

    @staticmethod
    def _sample_neg(triplets:np.ndarray, num_nodes:int) -> Tuple[TT, TT]:  # type: ignore
        _, torch, _, _, _, _, _, _ = lazy_embed_import_dep()
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
        labels[: triplets.shape[0]] = 1
        return all_triplets, labels
