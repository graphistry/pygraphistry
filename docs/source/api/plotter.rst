Plotter API Reference
=====================

The below Python API reference documentation is for three views of the core graph abstraction, `Plottable`:

* The :py:class:`graphistry.plotter.Plotter` class that mixes in all layers such as plugins
* The :py:class:`graphistry.Plottable` abstract interface for the core Graphistry graph object
* The :py:class:`graphistry.PlotterBase` class implementing it

Arrow Conversion Validation
---------------------------

``plot()``, ``upload()``, and ``to_arrow()`` use ``validate='autofix'`` by default
for pandas/cuDF to Arrow conversion. When Arrow rejects mixed-type object columns,
autofix probes object columns, coerces only the failing columns to strings, and
emits a warning naming those columns when ``warn=True``. Use
``validate='strict'`` or ``validate='strict-fast'`` to raise instead.

For pandas inputs, autofix prefers pandas' nullable string dtype when available
so missing values remain Arrow nulls after coercion. Older pandas versions fall
back to standard Python string coercion.

When a plotter has an experimental ``graphistry.schema.GraphSchema`` bound via
``g.bind(schema=schema)``, pass ``schema_validate='strict'`` to ``plot()``,
``upload()``, or ``to_arrow()`` to require declared columns and Arrow types at
the boundary. Use ``schema_validate='autofix'`` to cast compatible columns to
declared Arrow types after normal Arrow conversion. The default
``schema_validate=False`` preserves existing behavior.

Use the experimental read-only ``g.schema`` accessor to inspect the bound
``GraphSchema`` object. Check ``g.schema is not None`` when only a predicate is
needed. This reports only the local declaration attached through
``bind(schema=...)``: it does not infer a schema from data, fetch a remote
dataset schema, or serialize the schema into ``gfql_remote()`` requests.

.. toctree::
   :maxdepth: 3

Plotter Class
----------------------
.. automodule:: graphistry.plotter.Plotter
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


PlotterBase Class
----------------------
.. automodule:: graphistry.PlotterBase
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:


Plottable Interface
----------------------
.. automodule:: graphistry.Plottable
   :members:
   :undoc-members:
   :inherited-members:
   :show-inheritance:
