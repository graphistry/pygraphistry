import splunklib.client as client
import splunklib.results as results
import sys
from urllib.parse import quote
import pandas as pd

class SplunkConnector:
    def __init__(self, username, password, host):
        self.username = username
        self.password = password
        self.host = host
        self.connect()
        
    def connect(self):
        try:
            self.service = client.connect(host=self.host, username=self.username, password=self.password)
            print("Splunk connection established")
        except Exception as e:
            print(e)   
            
    def get_indexes(self):
        """Returns a dictionary of index names and their fields
            This is used to provide context for the symbolic AI
        """
        indexes = {}
        print('retrieving index information')
        for index_name in self.service.indexes:
            index = self.service.indexes[index_name.name]
            fields = index.fields
            indexes[index_name.name] = fields
        return indexes
    
    def get_fields(self, index):
        """Returns a list of fields for a given index
            This is used to provide context for the symbolic AI
        """
        print(f'Returning fields from {index}')
        query = f"search index={index} | fieldsummary | table field"
        return self.query(query)

    
    def query(self, search_string, earliest_time=None, latest_time=None, maxEvents=30000000):
        try:
            kwargs = {
                "exec_mode": "normal",
                "count": 0
            }
            if earliest_time:
                kwargs["earliest_time"] = earliest_time
            if latest_time:
                kwargs["latest_time"] = latest_time
            
            job = self.service.jobs.create(search_string, maxEvents=maxEvents, **kwargs)
            while True:
                while not job.is_ready():
                    pass

                stats = {"isDone": job["isDone"],
                        "doneProgress": float(job["doneProgress"])*100,
                        "scanCount": int(job["scanCount"]),
                        "eventCount": int(job["eventCount"]),
                        "resultCount": int(job["resultCount"])}

                status = ("\r%(doneProgress)03.1f%%   %(scanCount)d scanned   "
                  "%(eventCount)d matched   %(resultCount)d results") % stats

                sys.stdout.write(status)
                sys.stdout.flush()

                if stats["isDone"] == "1":
                    sys.stdout.write("\nDone!\n")
                    break

            result_count = stats["resultCount"]
            offset = 0
            count = 5000000
            results_list = []

            while len(results_list) < int(result_count):
                r = results.JSONResultsReader(job.results(output_mode='json', count=count, offset=offset))
                offset += count
                results_list.extend(r)
                print('len results', len(results_list))
            return results_list
        except Exception as e:
            print(e)
            return []
        
    def to_dataframe(self, search_query, earliest_time=None, latest_time=None):
        data = list(self.query(search_query, earliest_time, latest_time))
        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df.columns = [field for field in data[0].keys()]

        return df
    

class GraphistryAdminSplunk(SplunkConnector):
    
    def __init__(self):
        username = 'alex'
        password = 'graph1234!'
        host="splunk.graphistry.com"
        # username = ''
        # password = ''
        # host="splunk.graphistry.com"
        super().__init__(username, password, host)
        
        
        