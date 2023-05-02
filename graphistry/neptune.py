import awswrangler as wr
import pandas as pd

class NeptuneConnector:
    def __init__(self, url, port=8182, iam_enabled=False):
        self.url = url
        self.port = port
        self.iam_enabled = iam_enabled
        self.client = wr.neptune.connect(url, port, iam_enabled=iam_enabled)

    def execute_opencypher(self, query):
        return wr.neptune.execute_opencypher(self.client, query)

    def status(self):
        return self.client.status()

    def get_graph_summary(self):
        # Construct the summary endpoint URL
        summary_url = f"{self.url}:{self.port}/propertygraph/statistics/summary"

        # Send an HTTP GET request to the summary endpoint
        response = requests.get(summary_url)

        # Parse the response as JSON
        summary = response.json()

        return summary
