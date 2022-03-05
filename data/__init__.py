import pandas as pd
from graphistry.ai_utils import setup_logger

logger = setup_logger(__name__)

def get_reddit_dataframe(nrows=100, min_doc_length=100):
    logger.info(f'Loading Reddit Data and pruning to documents with at least {min_doc_length} words, and returning random sample of {nrows} rows')
    df = pd.read_csv('data/reddit.csv', index_col=0)
    tdf = df[(df.doc_length > min_doc_length)]  # & (df.user!='None')]# & (df['type']=='summary')] # get good textual features
    return tdf.sample(nrows)


def get_blackrock_dataframes():
    logger.info(f'Loading Blackrock Data')
    edf = pd.read_csv('data/edges_blackrock.csv', index_col=0)
    ndf = pd.read_csv('data/nodes_blackrock.csv', index_col=0)
    return ndf, edf

def get_amazon_dataframe():
    logger.info(f'Loading Amazon Data')
    ndf = pd.read_csv('data/final_g.nodes.csv', index_col=0)
    edf = pd.read_csv('data/final_g.edges.csv', index_col=0)
    return ndf, edf


# Get sample cyber security data
def get_host_dataframe(nrows=10000):
    logger.info(f'Loading Host Events Data')
    df = pd.read_csv("data/host_events.csv", nrows=nrows, index_col=0)
    return df


def get_netflow_dataframe(nrows=10000):
    logger.info(f'Loading Netflow Events Data')
    df = pd.read_csv("data/netflow_events.csv", nrows=nrows, index_col=0)
    return df


def get_botnet_dataframe(nrows=10000):
    logger.info(f'Loading Botnet Capture Data with Targets')

    df = pd.read_csv(
        "data/malware_capture_bot.csv", nrows=nrows, index_col=0
    )
    # cast things as strings or floats
    as_string = [
        "StartTime",
        "Proto",
        "SrcAddr",
        "Sport",
        "DstAddr",
        "Dir",
        "Dport",
        "State",
    ]
    as_float = ["Dur", "sTos", "dTos", "TotPkts", "TotBytes", "SrcBytes"]
    df[as_string] = df[as_string].astype(str)
    df[as_float] = df[as_float].astype(float)
    return df


def get_stocks_dataframe():
    logger.info(f'Loading Stock Metadata and Prices Data')

    ndf = pd.read_csv(
        "data/stocks_metadata.csv", index_col=0
    )
    prices_df = pd.read_csv(
        "data/stocks_all.csv", index_col=0
    )
    return ndf, prices_df


