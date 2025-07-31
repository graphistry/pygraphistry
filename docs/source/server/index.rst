Login and Share
=================

PyGraphistry streamlines working with optional Graphistry server capabilities such as GPU-accelerated visual analytics, sharing visualizations, simplifying graph pipelines, GFQL compute endpoints, and sharing GPU resources.

Server interactions are typically by first logging in (`graphistry.register()`) and then sending data, such as via `g.plot()`. For multi-tenant applications or concurrent processing, use isolated client instances (`graphistry.client()`) to safely serve multiple users. 

You can set access control policies on all of your uploaded data via `graphistry.privacy()`. Read on for more on both.

.. toctree::
   :maxdepth: 2

   register
   concurrency
   privacy

