from typing import List, Dict, Any, Union, Optional, Tuple
from abc import ABC, abstractmethod
import pandas as pd

import symai as ai


# def encode_record(row: pd.Series, cols, concat=True):
#     """Encode a record into a string
#         expensive and slow at scale"""
#     if concat:
#         res = ' '.join([str(row[col]) for col in cols])
#     else:
#         res = str({k:v for k, v in row.to_dict().items() if k in cols})
#     return res


def encode_df(
    df: pd.DataFrame, cols: Optional[List[str]], sep: str = " | ", na_rep: str = "<na>"
) -> pd.Series:
    """Encode a dataframe into a pd.Series of strings
    vectorized version of encode_record, much faster
    """
    if cols is None:
        cols = df.columns
    # coherce to string
    df = df.astype(str)
    encoded = df[cols[0]]
    for col in cols[1:]:
        encoded = encoded.str.cat(df[col], sep=sep, na_rep=na_rep)
    return encoded


def encode_metadata(
    df: pd.DataFrame, cols: Union[Optional[List[str]], bool]
) -> List[Dict]:
    """Encode a dataframe into a list of dicts"""
    if cols is False:
        return [{}] * len(df)
    if cols is not None:
        cols = df.columns
    df2 = df[cols]
    return df2.to_dict("records")


# #######################################################################################################
#
#  SymbolicAI helpers
#
# #######################################################################################################
def process_df_to_syms(ndf, as_records):
    if as_records:
        syms = [ai.Symbol(row) for row in ndf.to_dict("records")]
    else:
        syms = [ai.Symbol(str(row)) for row in ndf.values]
    return syms


def process_df(ndf, as_records, max_doc_length=200):
    syms = process_df_to_syms(ndf, as_records)
    if max_doc_length:
        syms = ai.Symbol(
            [ai.Symbol(str(sym.value).split()[:max_doc_length]) for sym in syms]
        )
    return syms


def process_df_to_sym(df, as_records):
    if as_records:
        syms = ai.Symbol(df.to_dict("records"))
    else:
        syms = ai.Symbol(list(df.values))
    return syms


# ######################################################################################################
class AIBaseMixin(ABC):
    def _unwrap_graphistry(self, g):
        ndf = g._nodes
        node = g._node

        edf = g._edges
        src = g._source
        dst = g._destination

        return ndf, edf, node, src, dst

    def _encode_text_for_gpt(self, text):
        """Encode text to be used with GPT-2"""
        raise NotImplementedError
