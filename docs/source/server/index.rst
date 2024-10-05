Login and Sharing
=================

PyGraphistry contains the official bindings to Graphistry servers. It is a convenient way to benefit from server capabilities such as GPU-accelerated visual analytics, sharing visualizations, simplifying graph pipelines, GFQL compute endpoints, and sharing GPU resources.

You can self-host Graphistry on your own GPU instances, or get an account on `Graphistry Hub <https://www.graphistry.com/get-started>`_, including the free GPU tier.

Server interactions are typically by first logging in (`graphistry.register()`) and then sending data, such as via `g.plot()`. You can set access control policies on all of your uploaded data via `graphistry.privacy()`. Read on for more on both.

.. toctree::
   :maxdepth: 2

   register
   privacy

