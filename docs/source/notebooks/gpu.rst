GPU
==========================

GFQL has two NVIDIA GPU engines: ``engine='cudf'`` (RAPIDS, eager) and
``engine='polars-gpu'`` (the fused lazy Polars plan on GPU). See
:doc:`Choosing a GFQL Engine </gfql/engines>` for which to use and how they compare to the
CPU ``pandas`` / ``polars`` engines.

.. toctree::
   :maxdepth: 2
   :caption: GPU compute with Nvidia RAPIDS
   :titlesonly:

   GPU I: CPU Pandas <../demos/demos_databases_apis/gpu_rapids/part_i_cpu_pandas.ipynb>
   GPU II: cuDF <../demos/demos_databases_apis/gpu_rapids/part_ii_gpu_cudf.ipynb>
   GPU IV: cuML UMAP <../demos/demos_databases_apis/gpu_rapids/part_iv_gpu_cuml.ipynb>
   GPU V: cuGraph <../demos/demos_databases_apis/gpu_rapids/cugraph.ipynb>
   GPU Memory Planning <../demos/gfql/GPU_memory_consumption_tutorial.ipynb>
