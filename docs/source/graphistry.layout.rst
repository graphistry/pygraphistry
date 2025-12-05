graphistry.layout package
=========================

.. graphviz::

   digraph layout_toy {
       rankdir=LR;
       data [label="Data (nodes/edges)"];
       layout [label="Layout (x,y)"];
       render [label="Render (Graphviz/Graphistry)"];
       data -> layout -> render;
   }

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   graphistry.layout.gib
   graphistry.layout.graph
   graphistry.layout.modularity_weighted
   graphistry.layout.ring
   graphistry.layout.sugiyama
   graphistry.layout.utils

Submodules
----------

graphistry.layout.circle module
-------------------------------

.. automodule:: graphistry.layout.circle
   :members:
   :undoc-members:
   :show-inheritance:

graphistry.layout.fa2 module
----------------------------

.. automodule:: graphistry.layout.fa2
   :members:
   :undoc-members:
   :show-inheritance:

Module contents
---------------

.. automodule:: graphistry.layout
   :members:
   :undoc-members:
   :show-inheritance:
