import os
import tempfile
from typing import Optional, List
from IPython.display import Markdown, display

import pandas as pd
from gpt_index import GPTSimpleVectorIndex, SimpleDirectoryReader
from gpt_index.composability import ComposableGraph
from gpt_index.constants import MAX_CHUNK_SIZE
from gpt_index import GPTListIndex, Document, SimpleWebPageReader


def get_urls_index(urls):
    documents = SimpleWebPageReader(html_to_text=True).load_data(urls)
    index = GPTListIndex(documents)
    return index


def get_search_internet_index(keywords):
    urls = []
    for keyword in keywords:
        url = f"https://www.google.com/search?q={keyword}"
        urls.append(url)
    return get_urls_index(urls)


def encode_record(row: pd.Series, cols, concat=True):
    """Encode a record into a string"""
    if concat:
        res = " ".join([str(row[col]) for col in cols])
    else:
        res = str({k: v for k, v in row.to_dict().items() if k in cols})
    return res


# this seems to not be able to search ...
def process_list(encode_df, concat=False):
    """Process a dataframe into a GPTListIndex"""
    documents = []
    cols = encode_df.columns
    for _, row in encode_df.iterrows():
        res = encode_record(row, cols, concat=concat)
        documents.append(Document(res[:MAX_CHUNK_SIZE]))
    index = GPTListIndex(documents)
    return index


def process_directory(encode_df, concat=False):
    """Process a dataframe into a GPTSimpleVectorIndex"""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"created temp directory: {temp_dir}")
        temp_dir = Path(temp_dir)

        cols = encode_df.columns
        for index, row in encode_df.iterrows():
            with open(temp_dir / (f"{index}.txt"), "w") as f:
                res = encode_record(row, cols, concat=concat)
                f.write(res[:MAX_CHUNK_SIZE])

        documents = SimpleDirectoryReader(temp_dir).load_data()
        index = GPTSimpleVectorIndex(documents)
    return index


def factory_index(df, cols, processor, loader, load, concat=False, pathfile=None):
    """Base function to create an index from a dataframe or load it from disk"""
    if df is None:
        load = True

    if not load:
        if cols:
            encode_df = df[cols]
        else:
            encode_df = df
        print(f"Processing dataframe of size {encode_df.shape}")
        #################################################
        index = processor(encode_df, concat=concat)
        #################################################
        print(f"Encoded dataframe as GPTListIndex")
        if pathfile:
            index.save_to_disk(pathfile)
            print(f"and saved it to {pathfile}")
    else:
        print(f"Opening {pathfile} - {loader}")
        index = loader.load_from_disk(pathfile)
    return index


def list_df_index_processor(
    df: Optional[pd.DataFrame],
    pathfile: Optional[str],
    cols: Optional[List[str]] = None,
    concat: bool = False,
    load: bool = False,
):
    # seems very slow when i query it...
    """_summary_

    :param: df (Optional[pd.DataFrame]): _description_
    :param: pathfile (Optional[str]): _description_
    :param: cols (Optional[List[str]], optional): _description_. Defaults to None.
    :param: concat (bool, optional): _description_. Defaults to False.
    :param: load (bool, optional): _description_. Defaults to False.

    :returns: index
    """
    index = factory_index(
        df,
        cols,
        process_list,
        GPTListIndex,
        concat=concat,
        pathfile=pathfile,
        load=load,
    )
    return index


def vector_df_index_processor(
    df: Optional[pd.DataFrame],
    pathfile: Optional[str],
    cols: Optional[List[str]] = None,
    concat: bool = False,
    load: bool = False,
):
    """Get a vector index from a dataframe or load from disk

    :param df (Optional): dataframe to encode, if None, then load from disk
    :param pathfile: path to save the index to, if None, then don't save
    :param cols: columns to encode, if None, then encode all columns
    :param load: whether to load the index from disk
    :param concat: whether to concatenate all columns into one string or make a string from a dictionary the dataframe
    """
    index = factory_index(
        df,
        cols,
        process_directory,
        GPTSimpleVectorIndex,
        concat=concat,
        pathfile=pathfile,
        load=load,
    )
    return index


class GPTIndex:
    """Wrapper class for GPTSimpleVectorIndex"""

    def __init__(
        self,
        df: pd.DataFrame,
        pathfile: str,
        cols,
        load: bool = False,
        concat: bool = True,
        index_func=vector_df_index_processor,
    ):
        self.df = df
        self.pathfile = pathfile
        self.concat = concat
        self.index = index_func(
            df=df, pathfile=pathfile, cols=cols, load=load, concat=concat
        )
        # self._set_summary()

    def _set_summary(self):
        """Set the summary of the dataframe as it allows composition into a graph over many different indices"""
        summary = self.index.query("Give a Summary of the document", similarity_top_k=4)
        self.index.set_text(str(summary))
        self._summary = summary

    def return_index(self):
        return self.index

    def search(
        self, query: str, similarity_top_k: int = 3, verbose: bool = False, **kwargs
    ):
        """Search the dataframe for a query"""
        results = self.index.query(
            query, similarity_top_k=similarity_top_k, verbose=verbose, **kwargs
        )
        if verbose:
            display(Markdown(f"<b> {results} </b>"))
        return results

    def get_df(self, results):
        """Get the dataframe from the results"""
        return self.df.iloc[[result.index for result in results]]

    def get_df_from_query(self, query: str, verbose: bool = False, **kwargs):
        """Get the dataframe from the query"""
        results = self.search(query, verbose=verbose, **kwargs)
        return self.get_df(results)

    def get_df_from_index(self, index: int):
        """Get the dataframe from the index"""
        return self.df.iloc[index]

    def get_df_from_indexes(self, indexes: list):
        """Get the dataframe from the indexes"""
        return self.df.iloc[indexes]

    def get_df_from_results(self, results):
        """Get the dataframe from the results"""
        return self.get_df_from_indexes([result.index for result in results])

    def get_results_from_query(self, query: str, top_k: int = 5, verbose: bool = False):
        """Get the results from the query"""
        return self.search(query, top_k=top_k, verbose=verbose)

    def get_results_from_index(self, index: int):
        """Get the results from the index"""
        return self.index.get_document(index)

    def get_results_from_indexes(self, indexes: list):
        """Get the results from the indexes"""
        return [self.index.get_document(index) for index in indexes]

    def get_results_from_df(self, df: pd.DataFrame):
        """Get the results from the dataframe"""
        return self.get_results_from_indexes(df.index)

    def merge_indexes(self, list_index: list, save_path: str = None):
        """Merge the indexes"""
        graph = ComposableGraph.build_from_index(list_index)

        # [Optional] save to disk
        if save_path:
            graph.save_to_disk(save_path)
        # update
        self.index = graph

    def load_index_graph(self, save_path: str):
        """Load the index graph"""
        graph = ComposableGraph.load_from_disk(save_path)
        self.index = graph

    # def get_formatted_sources(self):
    #     response.get_formatted_sources()
