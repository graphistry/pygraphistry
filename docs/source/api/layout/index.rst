.. _pyg-layout-api:

Layouts
=======

Native layout engines within Graphistry.

We recommend using the various plugins for additional layouts, such as for tree and hierarchical data diagramming.


.. toctree::
   :maxdepth: 3
   :caption: Contents:

   circle
   fa2
   gib
   mercator
   modularity_weighted
   ring
   sugiyama
   legacy_utils


Layout plugins: igraph, graphviz, and more
-------------------------------------------

Several plugins provide a large variety of additional layouts:

* :ref:`cugraph` : GPU-accelerated FA2, a naive version of Graphistry's layout engine
* :ref:`graphviz`: Especially strong at tree and hierarchical data diagramming such as the dot engine
* :ref:`igraph` : A variety of layouts, including Sugiyama, Fruchterman-Reingold, and Kamada-Kawai
* NetworkX: A variety of layouts


LayoutsMixin
------------

.. automodule:: graphistry.layouts.LayoutsMixin
   :members:
   :undoc-members:
   :show-inheritance: