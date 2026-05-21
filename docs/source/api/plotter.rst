Plotter API Reference
=====================

The below Python API reference documentation is for three views of the core graph abstraction, `Plottable`:

* The :py:class:`graphistry.plotter.Plotter` class that mixes in all layers such as plugins
* The :py:class:`graphistry.Plottable` abstract interface for the core Graphistry graph object
* The :py:class:`graphistry.PlotterBase` class implementing it

Arrow Conversion Coercion
-------------------------

``plot()``, ``upload()``, and ``to_arrow()`` accept ``auto_coerce=False`` by
default. With the default strict behavior, mixed-type pandas/cuDF object columns
that PyArrow cannot convert still fail with ``ArrowConversionError`` and list the
problem columns when Graphistry can identify them.

Set ``auto_coerce=True`` to opt in to a narrow fallback for dirty real-world
columns. Graphistry first tries normal Arrow conversion. On failure, it probes
object columns one at a time and coerces only columns that fail Arrow conversion
to strings before retrying. Coerced column names are emitted through the
``graphistry.PlotterBase`` logger at info level, for example:
``Auto-coerced mixed-type columns to string for Arrow conversion: ['amount']``.

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
