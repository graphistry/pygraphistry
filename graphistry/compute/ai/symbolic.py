# import sympy
from typing import TYPE_CHECKING, List, Dict, Any, Union, Optional, Tuple
from collections import Counter
import numpy as np

from graphistry.text_utils import SearchToGraphMixin
from graphistry.plugins.splunkconn import GraphistryAdminSplunk, SplunkConnector

import pandas as pd
from time import time

try:
    import symai as ai
    from symai import *
    from symai.post_processors import StripPostProcessor
    from symai.pre_processors import PreProcessor
except ImportError:
    ai = None

__doc__ = """graphistry.compute symbolic
                providing symbolic graph operations
                (e.g., computing the characteristic polynomial of a graph's adjacency matrix)
                or symbolic graph operations using the graphistry.ai chatbot"""


if TYPE_CHECKING:
    MIXIN_BASE = SearchToGraphMixin
else:
    MIXIN_BASE = object


def compute_characteristic_polynomial(g, as_expression=True, bound=42):
    """Compute the characteristic polynomial of a graph's adjacency matrix"""
    import sympy

    if hasattr(g, "_adjacency"):
        A = g._adjacency
    elif hasattr(g, "_weighted_adjacency"):
        A = g._weighted_adjacency

    if A.shape[0] > bound:
        Warning(
            f"Computing the characteristic polynomial of a graph with {A.shape[0]} nodes may take a long time"
        )

    M = sympy.SparseMatrix(A.todense())
    x = sympy.Symbol("x")
    if as_expression:
        return M.charpoly(x).as_expr()
    return x, M.charpoly(x)


#######################################################################################################
#
#  Symbolic AI helpers
#
#######################################################################################################


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


class LambdaHelper(ai.Expression):
    def __init__(self, callable: ai.Callable):
        super().__init__()

        def _callable(*args, **kwargs):
            kw = {
                "args": args,
                "kwargs": kwargs,
            }
            return callable(kw)

        self.callable: ai.Callable = _callable

    def forward(self, *args, **kwargs) -> ai.Symbol:
        return self.callable(*args, **kwargs)


CONTEXTS = [
    None,
    "graph",
    "edges",
    "summary",
    "search",
    "entities",
    "relationships",
    "money",
    "time",
    "threats",
]


def extract_edges(reports: List[ai.Symbol]) -> pd.DataFrame:
    edf = []
    for i, sym in enumerate(reports):
        try:
            print("worked", i)
            edges = [
                row for row in sym.ast()
            ]  # sym.extract('extract edges list by splitting on "," and then splitting on "|", resulting in a list of lists, dont forget to make it json serializable')#.extract('List of lists')
            edf.append(pd.DataFrame(edges, columns=["src", "rel", "dst"]))
            # edf2.append(edges)
        except Exception as e:
            print(e)
            print("failure", i)
            # print(edges)
            # fix mixed length lists, repair
            sym2 = sym.extract(
                "repair edge list to a json serializable lists of length 3 eg ([['Asset Funders Network', 'Organization'], ['State Street Corporation', 'Business, Public Company']] -> [['Asset Funders Network', 'Organization', ''], ['State Street Corporation', 'Business', 'Public Company']])"
            )
            try:
                print("second worked", i)
                edges = [
                    row for row in sym2.ast()
                ]  # sym.extract('extract edges list by splitting on "," and then splitting on "|", resulting in a list of lists, dont forget to make it json serializable')#.extract('List of lists')
                edf.append(pd.DataFrame(edges, columns=["src", "rel", "dst"]))
            except:
                print("second failure", i)
                pass

    edf = pd.concat(edf)
    if not edf.empty:
        edf = edf[edf["src"] != ""]
        edf = edf[edf["dst"] != ""]
        edf = edf[edf["rel"] != ""]
    return edf


def safe_encode_df(df, max_doc_length=1800, individual_doc_length=1000):
    # encodes df to utf-8, truncates to max_doc_length, since df is rank ordered by semantic similarity to query under sentence transformer.
    pd.set_option("display.max_colwidth", individual_doc_length)
    ll = []
    docs = []
    index = []
    for i, row in df.iterrows():
        doc = row.to_string(header=False, index=False)
        doc = " ".join([k for k in doc.split() if k not in ["NaN", " "]])
        docs.append(doc)
        ll.append(len(doc.split()))
        index.append(i)
        if sum(ll) >= max_doc_length:
            break
    new_df = pd.DataFrame(docs, index=index, columns=["text"])
    print("truncated to", len(new_df), "docs")
    return new_df


ONLY_CODE = "ONLY CODE, no explanation."
ONLY_EXPLANATION = "ONLY EXPLANATION, no code."
EXPLANATION_AND_CODE = "EXPLANATION AND CODE."


def process(data, *args, **kwargs):
    """Process data with consistent pipeline of functions."""
    pipeline = kwargs.get("pipeline", LambdaHelper(lambda x: x["args"][0]))
    cluster = kwargs.get("cluster", False)
    stream = ai.Stream(ai.Sequence(pipeline))
    sym = ai.Symbol(data)
    res = ai.Symbol(list(stream(sym)))
    if cluster:
        expr = ai.Cluster()
        return expr(res)
    return res


WEB_DESCRIPTION = """Design a web app with HTML, CSS and inline JavaScript. 
Use dark theme and best practices for colors, text font, etc. 
Use Bootstrap for styling.
Do NOT remove the {{placeholder}} tag and do NOT add new tags into the body!"""


SPLUNK_CONTEXT = """[Description]
The following statements provide an overview of Splunk and writing Splunk queries using natural language inputs:
Splunk is a software platform that allows you to search, monitor, and analyze machine-generated data in real-time. \
    Splunk queries are used to search and analyze data in Splunk, and can be written using Splunk's search processing language (SPL).\
    This makes it easy for users with little or no programming experience to write effective Splunk queries.
[Examples]
# methods to list all fields for indexes. 
search index=yourindex| fieldsummary | table field 
or 
search index=yourindex | stats values(*) AS * | transpose | table column | rename column AS Fieldnames
or 
search index=yourindex | stats dc() as * | transpose 
or 
search index=yourindex | table *

# List the size of lookup files with an SPL search.
| rest splunk_server=local /services/data/lookup-table-files/
| rename eai:acl.app as app 
| table app title 
| search NOT title IN (*.kmz) 
| map maxsearches=990 search="| inputlookup $title$ 
| eval size=0
| foreach * [ eval size=size+coalesce(len('<<FIELD>>'),0), app=\"$app$\", title=$title$ | fields app title size]" 
| stats sum(size) by app title
| sort - sum(size)

# Search for disabled AD accounts that have been re-enabled: This query searches for any AD accounts that have been disabled in the past 90 days and then re-enabled in the past 24 hours. 
| search index=YOURINDEX EventCode IN (4725, 4722) earliest=-90d | eval account=mvindex(Account_Name,1) 

# Search Common EventCodes (EventID’s) for Suspicious Behavior: This query searches for suspicious behavior in a Windows environment by looking at common EventCodes (EventID’s).
| source index=YOURINDEX (EventCode IN (4688 OR 4624 OR 4625) AND SubjectLogonId!=0x3e7) earliest=-1d | eval account=mvindex(Account_Name,1)
[Last Example]
--------------
"""


class SplunkPreProcessor(PreProcessor):
    def __call__(self, wrp_self, wrp_params, *args, **kwds):
        super().override_reserved_signature_keys(wrp_params, *args, **kwds)
        return "| {} SPL:".format(str(wrp_self))


NEXT = " =>"


class SplunkPrompts(ai.Prompt):
    # https://docs.splunk.com/Documentation/SCS/current/SearchReference/SearchCommandExamples
    def __init__(self) -> ai.Prompt:
        super().__init__(
            [
                "search the main index where sourcetype is access_combined limit 10"
                + NEXT
                + "search index=main sourcetype=access_combined | head 10",
                "Modify the following Splunk query that is useful for identifying malware in FireEye logs to instead use Palo Alto Networks logs:"
                + NEXT
                + "index=palo_alto sourcetype=pan:logs (action=allow OR action=deny) | eval src_ip=src_ip | eval dest_ip=dest_ip | where (src_ip=* OR dest_ip=*) | stats count by src_ip, dest_ip, action | sort - count",
                "Lets try to find out how many errors have occurred on the Buttercup Games website"
                + NEXT
                + "buttercupgames (error OR fail* OR severe)",
                "write a splunk query for the index `redteam_50k` that uses the src and dst information to output a table for events where RED=1. you can use closest matching fields from [src_computer, other, dst_computer, time]"
                + NEXT
                + '| search index="redteam_50k" RED=1 | Table src_computer, dst_computer',
            ]
        )


class Splunk(ai.Expression):
    @property
    def static_context(self):
        return SPLUNK_CONTEXT

    def forward(self, sym: ai.Symbol, *args, **kwargs):
        @ai.few_shot(
            prompt="Generate queries based on the Splunk SPL domain specific language description\n",
            examples=SplunkPrompts(),
            pre_processor=[SplunkPreProcessor()],
            post_processor=[StripPostProcessor()],
            # stop=[''],
            **kwargs,
        )
        def _func(_) -> str:
            pass

        return self._sym_return_type(_func(Splunk(sym)))

    @property
    def _sym_return_type(self):
        return Splunk

    def as_splunk(self):
        return self.value


class AIGraph(Splunk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mem = {}
        # self.process = self._process

    def process(self, data, *args, **kwargs):
        return process(data, *args, **kwargs)

    def cache_sym(self, *args, **kwargs):
        key = args[0]
        value = args[1]
        self.mem[key] = value

    # @cache_sym
    def _get_likely_src_dst(self, df):
        srcdst = self.query(
            f"what are the likely src and dst columns of this dataframe? {df.columns}\n return as [src, dst]"
        ).list("[]", constraint=lambda x: len(x) == 2)
        return srcdst

    def _qa(self, df, *args, **kwargs):
        sym = self.process(df, *args, **kwargs)
        res = sym.query("What are interesting questions to ask of this dataset?")
        res2 = sym.query("ask and answer these questions", attach=res)
        # res2.query('continue')
        return res2

    def _kg(self, df, *args, **kwargs):
        sym = self.process(df, *args, **kwargs)
        srcdst = self._get_likely_src_dst(df)
        res = sym.query(
            f"Summarize the relationships between {srcdst[0]} and {srcdst[1]}?"
        )
        res2 = sym.query("ask and answer these questions", attach=res)
        return res2

    def learn_from_url(self, url):
        res = ai.Expression()
        res = res.fetch(url)
        # print(f'DOC LENGHT {res.length()}')
        res = res[:3000]
        sym = self.process(res)
        summary = sym.query("Provide a concise summary of this dataset.")
        self.mem[summary] = sym
        return summary


def get_splunk_condition(res, splunk):
    context = "No Context"
    if not pd.DataFrame(res).empty:
        condition = False
        context = "*Found Data:"
    if isinstance(res, Exception):
        condition = True
        context = f"The following SPL query returned an error: {res}"
    elif pd.DataFrame(res).empty:
        condition = pd.DataFrame(res).empty
        context = f"The following SPL query returned no results: {splunk}"
    print("context", context)
    print(res.head()) if isinstance(res, pd.DataFrame) else None
    return condition, context


class SplunkAIGraph(AIGraph):
    def __init__(self, index, all_indexes=False, verbose=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.index = index
        self.verbose = verbose
        self.mem = {}
        self.antimem = {}
        #self.conn = GraphistryAdminSplunk()
        #self.get_context(index, all_indexes=all_indexes)
        self.splunk = Splunk()

        self.PREFIX = f"make a splunk query that returns a table of events using some or all of the following fields: {self.fields}"
        self.SUFFIX = "\n\nRemember that this is a splunk search and to prepend `search` to your result. GO!"
        self.SPLUNK_HINT = "hint: |search index=* | Table src, rel, dst, **,"

    def connect(self, username, password, host, *args, **kwargs):
        self.conn = SplunkConnector(username, password, host, *args, **kwargs)
        #self.get_context(index, all_indexes=all_indexes)


    def get_context(self, index, all_indexes=False):
        self.get_fields(index)
        context = f"You are working with the index `{index}` and the following fields: {self.fields}"
        if all_indexes:
            self.get_indexes()
            context = f"You are working with the index `{index}` and the following fields: {self.fields}. You can also use the following indexes: {self.indexes.keys()}"
        print("-" * 80) if self.verbose else None
        print("context:", context) if self.verbose else None
        self._splunk_context = context
        return context

    def _search(self, query: str, *args, **kwargs) -> pd.DataFrame:
        try:
            res = self.conn.query(query, *args, **kwargs)
        except Exception as e:
            return e
        return pd.DataFrame(res)

    def _query_to_splunk(
        self, query: str, attach=None, verbose=False, *args, **kwargs
    ) -> pd.DataFrame:
        splunk = self.splunk(query, attach=attach, *args, **kwargs)
        res = self._search(splunk)
        print("-" * 60) if verbose else None
        print("****splunk:", splunk) if verbose else None
        print(
            "**result:", (res.shape if isinstance(res, pd.DataFrame) else res)
        ) if verbose else None
        return res, splunk

    def splunk_search(self, query: str, timeout: int = 1, *args, **kwargs):
        is_spl = is_splunk_query(query, self, verbose=True)
        if is_spl in ["yes", True, "True"]:
            res = self._search(query)
            splunk = query

        is_asking_for_splunk = is_asking_for_a_splunk_query(query, self, verbose=True)
        if is_asking_for_splunk in ["yes", True, "True"]:
            # try to convert to splunk
            res, splunk = self._query_to_splunk(
                query, attach=self._splunk_context, verbose=True
            )
        else:
            return self.query(query, *args, **kwargs)

        # check if we got data back
        condition, context = get_splunk_condition(res, splunk)

        i = 0
        while condition:
            if i > timeout:
                context = "\n\nWe timed out on Neural Cycles, maybe next time?"
                print(context)
                break
            old_splunk = splunk
            res, splunk = self._query_to_splunk(
                f"The previous query: {query}\n produced last incorrect SPL: {splunk} \
                \n producing error: {context}\nYou have ONE CHANCE TO FIX/regenerate a new SPL splunk search given the query.\
                    You can add or remove fields or rename them if needed. Remember {self._splunk_context}, if needed, to formulate the query into SPL. \
                        The new SPL should be different than the last.\
                        \n{self.SUFFIX}"
            )

            if old_splunk == splunk:
                print("!!same splunk, what?\n\t", splunk)

            print("new splunk, who dis?\n\t", splunk, "\n")
            condition, context = get_splunk_condition(res, splunk)
            i += 1

        if isinstance(res, pd.DataFrame) and not res.empty:
            # get good example pairs of query and splunk
            self.mem[query] = splunk
            print("-" * 30)
            print("--Added a successful memory:", query)
            condition = False
        else:
            self.antimem[query] = splunk
            print("-" * 30)
            print("--Added a failed memory:", query)
        return res

    def get_indexes(self):
        print("getting indexes") if self.verbose else None
        indices = self.conn.get_indexes()
        self.indexes = indices
        return indices

    def get_fields(self, index):
        print("getting fields") if self.verbose else None
        fields = self.conn.get_fields(index)
        fields = [k["field"] for k in fields]
        self.fields = fields
        return fields

    def _likely_fields(self):
        fields = self.fields
        good_cols = self.query(
            f"what are the likely fields of interest for a graph data science model: \n`{fields}`"
        ).list("item")
        print("good_cols:", good_cols)
        return good_cols

    def _likely_edges(self, *args, **kwargs):
        return get_likely_edges(self, *args, **kwargs)

    def qa(self, query):
        df = self.splunk_search(query)
        res = self._qa(df)
        return res


# Splunk specific functions


def is_splunk_query(query, sym: SplunkAIGraph, verbose=False, *args, **kwargs) -> bool:
    issplunk = sym.query(
        f"is this: `{query}` a SPL (splunk) query? Return yes or no",
        constraint=lambda x: x.lower() in ["yes", "no"],
        default="no",
    )
    print("-" * 60) if verbose else None
    if issplunk == "no":
        print(f"`{query[:400]}` \nis not a splunk query") if verbose else None
        return False
    print(f"This `{query[:400]}` is a splunk query") if verbose else None
    return True


def is_asking_for_a_splunk_query(
    query, sym: SplunkAIGraph, verbose=False, *args, **kwargs
) -> bool:
    issplunk = sym.query(
        f"is this asking you to generate a SPL (splunk) query? `{query}`, return a yes or no",
        constraint=lambda x: x.lower() in ["yes", "no"],
        default="no",
    )
    print("-" * 60) if verbose else None
    if issplunk == "no":
        print(
            f"`{query}` is not asking to generate a splunk query"
        ) if verbose else None
        return False

    print(f"`{query}` is asking to generate a splunk query") if verbose else None
    return True


def correct_splunk_query(
    query, context, sym: SplunkAIGraph, reason=False, verbose=False, *args, **kwargs
):
    query2 = sym.query(
        ai.Symbol(
            f"query: {query}\ncontext: {context} \nYou have ONE CHANCE TO FIX IT. index `{sym.index}` can use fields in {sym.fields} to formulate the query.\n{sym.SUFFIX}"
        )
    )
    if reason:
        reason = query2.query(f"why did you change the query?", attach=query)
        print(reason) if verbose else None
    return query


def find_splunk_index(query, sym: SplunkAIGraph, verbose, *args, **kwargs):
    index = sym.query(
        f"query: {query} uses the following index:",
        constraint=lambda x: len(x.split()) == 1,
        attach=sym.indices.keys(),
    )
    print(index) if verbose else None
    return index


def get_likely_edges(query, sym: SplunkAIGraph, verbose=False, *args, **kwargs):
    edges = sym.query(
        f"query: {query} uses the following columns as edges:",
        attach=sym.fields,
        constraint=lambda x: len(x.split()) == 2,
    )
    print(edges) if verbose else None
    return edges


###########################################################################################################
### AI Graph Class using Symbolic AI


class SymbolicMixin(MIXIN_BASE):
    def __init__(self, *args, **kwargs):
        self._sym = None
        self.splunk = SplunkAIGraph("redteam_50k")

    def ai(self, query, context=None, *args, **kwargs):
        if getattr(self, "_sym", None) is None:
            self._sym = ai.Expression()
        sym = self._sym  # add iteration to the sym

        res = self.splunk.splunk_search(query, previous=sym)

        if isinstance(res, pd.DataFrame) and not res.empty:
            g = self.edges(res, "src_computer", "dst_computer")
            return g

        self._sym = res
        return res

    def _reset_sym(self):
        self._sym = None

    def _encode_df_as_sym(
        self, context_df, as_records, max_doc_length=1800, cluster=False
    ):
        context_df = safe_encode_df(context_df, max_doc_length=max_doc_length)
        sym = process_df_to_sym(context_df, as_records)
        rr = process(sym, cluster=cluster)
        return rr

    def _add_context_and_query(self, sym, query, context="summary"):
        """adds context to the query

        Args:
            sym (_type_): symbolicAI
            query (_type_): the query
            context (str, optional): either key words or what ever other context. Defaults to 'summary'.

        Returns:
            _type_: _description_
        """
        if context == "summary":
            res = sym.query(f"Make a concise summary of: {query}")
        elif context == "entities":
            res = sym.query(
                f"extract relevant entities with a short discription of each in bullet format"
            )
        elif context == "relationships":
            res = sym.query(
                f'express concise relationships found between entities extracted in this document in bullet form (e.g., "entity1 is related (how?) to entity2")'
            )
        elif context == "edges":
            # res = sym.query(f'extract relevant edges and relationships only as a comma separated list, eg ["entity1,relationship,entity2"|"entity1,relationship2,entity4"| ..], be consistent and concise with the relationship names, and only return single best relationship (or short sentence, eg "Federal Reserve System,Central Bank of,United States") between two entities, or duplicate them across multiple rows, one for each relationship')
            res = sym.extract(
                'extract relevant entities and relationships (eg [["entity1", "relationship", "entity2"], [..]]) \
                              in a json serializable list of list. \
                If entitys are empty (eg "") do not include them. Be consistent and concise with the relationship names, \
                    simplifying similar relationships into a single name or short sentence, \
                        (eg [["Federal Reserve System","Central Bank of", "United States"], [..]]), or duplicate them across multiple rows, \
                            one for each relationship found between two entities.'
            )
        elif context == "money":
            res = sym.query(
                f'extract money related entities and relationships, eg "entity1 contributed a total of $100 to entity2" or "entity1 acquired entity2 for 100 million euros"'
            )
        elif context == "time":
            res = sym.query(
                f'extract time related entities and relationships, eg "3 days ago, *" or "Last week entity1 bought entity2 '
            )
        elif context == "threats":
            res = sym.query(
                f'extract threats and alerts related entities and relationships, eg "entity1 infiltrated entity2 via *" or "alert1 is a serious CVE * alert compromising entity1'
            )
        elif context == "search":
            res = ai.Expression(sym).query(query)
        elif context is None:
            res = sym.query(query)
        else:
            res = sym.query(context)
        res = res.query(
            "From the context: {context} \nand query: {query}\nreturn only relevant data. Lastly, make bold any entities found (eg **BlackRock**), and make italics any relationships found (eg *acquired*)."
        )
        return res

    def forward(
        self,
        query,
        context="summary",
        cols=None,
        top_n=21,
        as_records=True,
        edge_cols=None,
        cluster=False,
        fuzzy=True,
    ):
        t = time()
        context_df = self.search(query, top_n=top_n, fuzzy=fuzzy, cols=cols)[0]
        # print(context_df.head())
        if cols is not None:
            # encode only relevant columns
            context_df = context_df[cols]

        edges = ""
        if edge_cols is not None:
            # encode edges into context_df
            _, edf, node, src, dst = self._unwrap_graphistry(self)
            tdf = edf[edf[src].isin(context_df[node]) | edf[dst].isin(context_df[node])]
            a = tdf.groupby(src).agg(set)
            b = tdf.groupby(dst).agg(set)
            r = pd.concat([a, b])
            context_df = context_df.merge(
                r.reset_index(), left_on=node, right_on="index", how="inner"
            )

            edges = "edge"
            if cols is not None:
                edge_cols = cols + edge_cols
            context_df = context_df[edge_cols]
            # print(f'{edges} context_df', context_df)

        print(f"{edges} context_df.shape", context_df.shape)
        sym = self._encode_df_as_sym(context_df, as_records, cluster=cluster)
        sym = self._add_context_and_query(sym, query, context)
        print(f"forward time: {time() - t}")
        return sym, context_df

    def _make_graph(self, query, cols=None, top_n=10, as_records=True):
        ndf, edf, node, src, dst = self._unwrap_graphistry(self)
        # edges = self.forward(query, cols=cols, top_n=top_n, as_records=as_records, context='edges', find='edges', fuzzy=True)
        edges = self._analyze(
            query=query,
            context="edges",
            cols=cols,
            cluster_col="_dbscan",
            sample=4,
            max_clusters=10,
            as_records=True,
            verbose=False,
        )
        extract_edges

        lndf = pd.concat(
            [ndf[ndf.Node.isin(edges.src)], ndf[ndf.Node.isin(edges.dst)]]
        ).drop_duplicates()

        # get the edges
        return lndf

    def _make_graph_from_df(self, df):
        """IMPLEMENT ME"""
        return df

    def _analyze(
        self,
        g,
        query=None,
        context="summary",
        cols=None,
        cluster_col="_dbscan",
        sample=4,
        max_clusters=10,
        as_records=True,
        cluster=False,
        verbose=False,
        *args,
        **kwargs,
    ):
        """Make a summary graph of the top `max_clusters` clusters from `cluster_col`"""
        reports = []
        ndf = g._nodes

        # if cluster_col is None:
        #     sym = ai.Symbol(ndf)
        #     labels = sym.query(f'extract the top {max_clusters} clusters from the graph').list('[*]')
        #     cluster_col = sym.query(f'extract the cluster column name from the graph', attach=labels)
        #     report = self._add_context_and_query(sym, query, context)

        # else:
        label_cnts = Counter(ndf[cluster_col]).most_common()
        labels = [label for label, cnt in label_cnts]

        if max_clusters is None:
            max_clusters = len(label_cnts)

        if verbose:
            print(f"Found {len(label_cnts)} clusters from `{cluster_col}`")
            print(f" will analyze the top {max_clusters} clusters")

        these_labels = []
        for label in labels[:max_clusters]:
            context_df = ndf[ndf[cluster_col] == label]
            if context_df.empty:
                continue
            if sample is not None and len(context_df) >= sample:
                context_df = context_df.sample(sample)
            if cols is not None:
                context_df = context_df[cols]
            sym = self._encode_df_as_sym(context_df, as_records, cluster=cluster)
            report = self._add_context_and_query(sym, query, context)
            # inspect if this cluster answers the query or not

            reports.append(report)
            these_labels.append(label)
            if verbose:
                print("-" * 80)
                print(f"cluster {label+1}:")
                print(report)
                print()
        return reports, these_labels

    def query_graph(
        self,
        query,
        context="summary",
        cols=None,
        cluster_col="_dbscan",
        top_n=10,
        sample=4,
        as_records=True,
        verbose=True,
    ):
        gg = self.search_graph(query, top_n=top_n)  # .dbscan()
        # these are the reports for each cluster
        reports, labels = self._analyze(
            gg,
            query,
            context,
            cols,
            cluster_col=cluster_col,
            as_records=as_records,
            verbose=verbose,
            sample=sample,
        )

        ndf, edf, node, src, dst = self._unwrap_graphistry(gg)
        print(node, src, dst)
        df = []
        df2 = []

        for i, label in enumerate(labels):
            df.append([label, context, query, reports[i].value])
            # find edges that connect to this cluster
            for j, label2 in enumerate(labels):
                if label2 == label:
                    continue
                srcdst = edf[src].isin(ndf[ndf._dbscan == label][node]) & edf[dst].isin(
                    ndf[ndf._dbscan == label2][node]
                )
                dstsrc = edf[src].isin(ndf[ndf._dbscan == label2][node]) & edf[
                    dst
                ].isin(ndf[ndf._dbscan == label][node])
                edges = edf[srcdst | dstsrc]
                if not edges.empty:
                    print("New Edge")
                    print([label, label2, len(edges)])
                    df2.append([label, label2, len(edges)])

        df = pd.DataFrame(df, columns=[cluster_col, "context", "query", "report"])
        df2 = pd.DataFrame(df2, columns=["src", "dst", "weight"])

        res = self.bind()
        g_cluster = (
            res.nodes(df, cluster_col)
            .edges(df2, "src", "dst")
            .bind(edge_weight="weight")
        )
        # should aggregate vectors etc ...
        return g_cluster

    def on_select(self, query, context, nodeIDs, cluster=False):
        context_df = self._nodes.iloc[nodeIDs]
        sym = self._encode_df_as_sym(context_df, as_records=True, cluster=cluster)
        return self._add_context_and_query(sym, query, context)
