from typing import TYPE_CHECKING, List, Dict, Any, Optional, Union, Tuple
from graphistry.text_utils import SearchToGraphMixin
from graphistry.compute.ai.utils import AIBaseMixin, encode_df, encode_metadata
import os
import pandas as pd

import logging


try:
    from langchain import OpenAI
    from langchain.prompts import PromptTemplate
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.embeddings.cohere import CohereEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
    from langchain.vectorstores.faiss import FAISS
    from langchain.chains import VectorDBQAWithSourcesChain
    from langchain.chains.qa_with_sources import load_qa_with_sources_chain

    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain.embeddings.cohere import CohereEmbeddings
    from langchain.text_splitter import CharacterTextSplitter
    from langchain.vectorstores.elastic_vector_search import ElasticVectorSearch
    from langchain.vectorstores.faiss import FAISS
    from langchain.docstore.document import Document
    from langchain.prompts import PromptTemplate

except ImportError as e:
    print("langchain not installed. Please install with pip install langchain")


if TYPE_CHECKING:
    MIXIN_BASE = SearchToGraphMixin
else:
    MIXIN_BASE = object


logger = logging.getLogger(__name__)


template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.

QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

DEFAULT_PROMPT = PromptTemplate(
    template=template, input_variables=["question", "summaries"]
)

refine_template = (
    "The original question is as follows: {question}\n"
    "We have provided an existing answer, including sources: {existing_answer}\n"
    "We have the opportunity to refine the existing answer"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{context_str}\n"
    "------------\n"
    "Given the new context, refine the original answer to better "
    "answer the question"
    "If you do update it, please update the sources as well. "
    "If the context isn't useful, return the original answer."
)
refine_prompt = PromptTemplate(
    input_variables=["question", "existing_answer", "context_str"],
    template=refine_template,
)

question_template = (
    "Context information is below. \n"
    "---------------------\n"
    "{context_str}"
    "\n---------------------\n"
    "Given the context information and not prior knowledge, "
    "answer the question: {question}\n"
)
question_prompt = PromptTemplate(
    input_variables=["context_str", "question"], template=question_template
)

# ######################################################################################################


def split_text(text, splitter, chunk_size=1000, chunk_overlap=0):
    """Split text into chunks of size chunk_size"""
    logger.info(f"Splitting text into chunks of size {chunk_size}")
    text_splitter = splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_text(text)
    return texts


def add_metadata_to_search_index(search_index, metadata):
    """Add metadata to a search index"""
    logger.info(f"Adding metadata to search index")
    if metadata:
        for i, d in enumerate(search_index.docstore._dict.values()):
            d.metadata = metadata[i]
    else:
        # Add in a chuck fiducial source information for chunk
        for i, d in enumerate(search_index.docstore._dict.values()):
            d.metadata = {"source": f"{i}"}
    return search_index


def set_search_index(texts, metadata):
    """Create a search index from a list of texts and embeddings"""
    logger.info(f"Creating FAISS search index with {len(texts)}")
    embeddings = (
        OpenAIEmbeddings()
    )  # TODO make this configurable, and pass in handrolled embeddings for speed
    search_index = FAISS.from_texts(texts, embeddings)  # , metadatas=metadata)
    return add_metadata_to_search_index(search_index, metadata)


def create_vectorDB_chain_at_tempurature(
    texts,
    metadata=None,
    chain_type="map_reduce",
    splitter=CharacterTextSplitter,
    prompt=DEFAULT_PROMPT,
    temperature=0.0,
    chunk_size=1000,
    chunk_overlap=0,
    save_path=None,
):
    """Create a chain at a given temperature

    example:
        chain = create_chain_at_tempurature(state_of_the_union, chain_type="map_reduce", temperature=0.0)
        chain({"question": "What did the president say about Justice Breyer"}, return_only_outputs=True)
    """
    if isinstance(texts, pd.Series):
        texts = list(texts.values)

    texts = split_text(
        texts, splitter=splitter, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    if os.path.exists(f"{save_path}/search_index"):
        logger.info(f"Loading search index from search_index")
        search_index = FAISS.load_local("search_index")
    else:
        search_index = set_search_index(texts, metadata=metadata)
        search_index.save_local(f"{save_path}/search_index")

    chain = VectorDBQAWithSourcesChain.from_chain_type(
        OpenAI(temperature=temperature), chain_type=chain_type, vectorstore=search_index
    )

    logger.info(f"Created chain {chain}")
    if save_path:
        chain.save(save_path)
        logger.info(f"Saved chain to {save_path}")
    return chain, search_index, texts


def fast_chain(
    query,
    docs,
    prompt=DEFAULT_PROMPT,
    question_prompt=None,
    combine_prompt=None,
    refine_prompt=None,
    chain_type="map_reduce",
    temperature=0.0,
):
    """fits chain on supplied docs and query, returns answer

    docs come from query -> similar docs -> docs
    """
    logger.info(f"Creating {chain_type}-chain with {len(docs)} docs for query {query}")

    if chain_type is ["refine", "map_reduce"]:
        return_intermediate_steps = True
    else:
        return_intermediate_steps = False

    chain = load_qa_with_sources_chain(
        OpenAI(temperature=temperature),
        chain_type=chain_type,
        return_intermediate_steps=return_intermediate_steps,
        prompt=prompt,
        question_prompt=question_prompt,
        combine_prompt=combine_prompt,
        refine_prompt=refine_prompt,
    )
    return chain({"input_documents": docs, "question": query}, return_only_outputs=True)


class LangChainMixin(MIXIN_BASE):
    def __init__(self, temperature=0.0, *args, **kwargs) -> None:
        self._temperature = temperature
        self._cols = kwargs.get("cols", None)
        self._prompt = kwargs.get("prompt", DEFAULT_PROMPT)

    def _unwrap_graphistry(self, g):
        ndf = g._nodes
        node = g._node

        edf = g._edges
        src = g._source
        dst = g._destination

        return ndf, edf, node, src, dst

    def _encode_graph_chain(
        self, node_cols, node_metadata, edge_cols, edge_metadata, temperature=0.0
    ):
        ndf, edf, node, src, dst = self._unwrap_graphistry(self)

        node_text = encode_df(ndf, node_cols)
        # node_text = ' '.join(node_text)
        node_metadata = encode_metadata(ndf, node_metadata)

        edge_text = encode_df(edf, edge_cols)
        # edge_text = ' '.join(edge_text)
        edge_metadata = encode_metadata(edf, edge_metadata)

        (
            nodes_chain,
            nodes_search_index,
            nodes_texts,
        ) = create_vectorDB_chain_at_tempurature(
            node_text, metadata=node_metadata, temperature=temperature
        )
        (
            edges_chain,
            edges_search_index,
            edges_texts,
        ) = create_vectorDB_chain_at_tempurature(
            edge_text, metadata=edge_metadata, temperature=temperature
        )

        self._texts = {"nodes": nodes_texts, "edges": edges_texts}
        self._indices = {"nodes": nodes_search_index, "edges": edges_search_index}
        self._chains = {"nodes": nodes_chain, "edges": edges_chain, "fast": fast_chain}
        print("Created node and edge qa-chains")

    def _get_similar_texts(self, query, kind="nodes"):
        docs = self._indices[kind].similarity_search(query, k=10, return_metadata=True)
        return docs

    def _query(
        self,
        query: str,
        docs: Optional[List[str]] = None,
        recall: Optional[str] = None,
        kind: str = "nodes",
        fast=True,
        *args,
        **kwargs,
    ):
        if recall is None:
            recall = query
        if fast:
            # reduce the corpus to similar docs to run chain on
            if docs is None:
                docs = self._get_similar_texts(recall, kind=kind)
            kind = "fast"
            chain = self._chains[kind]
            return chain(query, docs, *args, **kwargs)
        # otherwise use the full chain, searching over entire corpus
        chain = self._chains[kind]
        return chain({"question": query}, return_only_outputs=True)

    def _query_nodes(self, query: str, recall: Optional[str] = None, *args, **kwargs):
        return self._query(
            query, docs=None, recall=recall, kind="nodes", *args, **kwargs
        )

    def _query_edges(self, query: str, recall: Optional[str] = None, *args, **kwargs):
        return self._query(
            query, docs=None, recall=recall, kind="edges", *args, **kwargs
        )

    def _query_nodes_refine_edges(
        self, query: str, recall: Optional[str] = None, *args, **kwargs
    ):
        if recall is None:
            recall = query
        docs = self._get_similar_texts(recall, kind="nodes")
        nodes_chain = self._query(
            query, docs=docs, recall=recall, kind="nodes", fast=True, *args, **kwargs
        )

        docs = self._get_similar_texts(recall, kind="edges")
        answer_chain = self._query(
            nodes_chain,
            docs=docs,
            recall=recall,
            kind="edges",
            fast=False,
            *args,
            **kwargs,
        )
        return answer_chain
