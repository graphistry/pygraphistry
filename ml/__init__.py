import pandas as pd

class DefaulMLConfig():
    API_KEY = 'etc'
    
# Get sample data
def get_host_dataframe(nrows=100000):
    df = pd.read_csv('graphistry/data/host_events.csv', nrows=nrows, index_col=0)
    return df


def get_netflow_dataframe(nrows=100000):
    df = pd.read_csv('graphistry/data/netflow_events.csv', nrows=nrows, index_col=0)
    return df

def get_botnet_dataframe(nrows=100000):
    df = pd.read_csv('graphistry/data/malware_capture_bot.csv', nrows=nrows, index_col=0)
    # cast things as strings or floats
    as_string = ['StartTime', 'Proto', 'SrcAddr', 'Sport', 'DstAddr', 'Dir',
     'Dport', 'State']
    as_float = ['Dur', 'sTos', 'dTos', 'TotPkts', 'TotBytes', 'SrcBytes']
    df[as_string] = df[as_string].astype(str)
    df[as_float] = df[as_float].astype(float)
    return df

