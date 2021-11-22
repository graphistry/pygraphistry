# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC # Databricks <> Graphistry Tutorial: Notebooks & Dashboards on IoT data
# MAGIC 
# MAGIC This tutorial visualizes a set of sensors by clustering them based on lattitude/longitude and overlaying summary statistics
# MAGIC 
# MAGIC We show how to load the interactive plots both with Databricks notebook and dashboard modes
# MAGIC 
# MAGIC Steps:
# MAGIC 
# MAGIC * Install Graphistry
# MAGIC * Prepare IoT data
# MAGIC * Plot in a notebook
# MAGIC * Plot in a dashboard

# COMMAND ----------

# MAGIC  %md
# MAGIC ## Install & connect

# COMMAND ----------

# Uncomment and run first time
! pip install graphistry

# Can sometimes help:
# dbutils.library.restartPython()

# COMMAND ----------

#Optional: Uncomment - We find this speeds up calls 10%+ on some datasets
#spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# COMMAND ----------

import graphistry  # if not yet available, install and/or restart Python kernel using the above

graphistry.register(
    api=3, username='MY_USERNAME', password='MY_PASSWRD',

    server='hub.graphistry.com'  # or your private server
    protocol='https',  # if http-only, browsers may prevent embedding plots: switch to ".plot(render=False)"
)    

graphistry.__version__

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare IoT data
# MAGIC Sample data provided by Databricks
# MAGIC 
# MAGIC We create tables for different plots:
# MAGIC 
# MAGIC * Raw table of device sensor reads
# MAGIC * Summarized table:
# MAGIC   - rounded latitude/longitude
# MAGIC   - summarize min/max/avg for battery_level, c02_level, humidity, timestamp

# COMMAND ----------

# Load the data from its source.
devices = spark.read \
  .format('json') \
  .load('/databricks-datasets/iot/iot_devices.json')

# Show the results.
print('type: ', str(type(devices)))
display(devices.take(10))

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.functions import concat_ws, col, round

devices_with_rounded_locations = (
    devices
    .withColumn(
        'location_rounded1',
        concat_ws(
            '_',
            round(col('latitude'), 0).cast('integer'),
            round(col('longitude'), 0).cast('integer')))
    .withColumn(
        'location_rounded2',
        concat_ws(
            '_',
            round(col('latitude'), -1).cast('integer'),
            round(col('longitude'), -1).cast('integer')))
)

cols = ['battery_level', 'c02_level', 'humidity', 'timestamp']
id_cols = ['cca2', 'cca3', 'cn', 'device_name', 'ip', 'location_rounded1', 'location_rounded2']
devices_summarized = (
    devices_with_rounded_locations.groupby('device_id').agg(
        *[F.min(col) for col in cols],
        *[F.max(col) for col in cols],
        *[F.avg(col) for col in cols],
        *[F.first(col) for col in id_cols]
    )
)

# [(from1, to1), ...]
renames = (
    [('device_id', 'device_id')]
    + [(f'first({col})', f'{col}') for col in id_cols]
    + [(f'min({col})', f'{col}_min') for col in cols] 
    + [(f'max({col})', f'{col}_max') for col in cols]
    + [(f'avg({col})', f'{col}_avg') for col in cols]
 )
devices_summarized = devices_summarized.select(list(
       map(lambda old,new:F.col(old).alias(new),*zip(*renames))
       ))

display(devices_summarized.take(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Notebook plot
# MAGIC 
# MAGIC * Simple: Graph connections between `device_name` and `cca3` (country code)
# MAGIC * Advanced: Graph multiple connections, like `ip -> device_name` and `locaation_rounded1 -> ip`

# COMMAND ----------

displayHTML(graphistry.edges(devices.sample(fraction=0.1), 'device_name', 'cca3').plot())

# COMMAND ----------

hg = graphistry.hypergraph(
    devices_with_rounded_locations.sample(fraction=0.1).toPandas(),
    ['ip', 'device_name', 'location_rounded1', 'location_rounded2', 'cca3'],
    direct=True,
    opts={
        'EDGES': {
            'ip': ['device_name'],
            'location_rounded1': ['ip'],
            'location_rounded2': ['ip'],
            'cca3': ['location_rounded2']
        }
    })
displayHTML(hg['graph'].plot())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Dashboard plot
# MAGIC 
# MAGIC * Make a `graphistry` object as usual...
# MAGIC * ... Then disable the splash screen and optionally set custom dimensions
# MAGIC 
# MAGIC The visualization will now load without needing to interact in the dashboard (`view` -> `+ New Dashboard`)

# COMMAND ----------

g = graphistry.edges(devices.sample(fraction=0.1), 'device_name', 'cca3')

displayHTML(
    g
        .settings(url_params={'splashAfter': 'false'})
        .plot(override_html_style="""
            border: 1px #DDD dotted;
            width: 50em; height: 50em;
        """)
)

# COMMAND ----------


