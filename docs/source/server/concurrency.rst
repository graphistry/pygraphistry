Multi-Tenancy & Concurrency
==========================

Graphistry `register()` and `plot()` are stateful. Methods are the concurreny boundary.

This is because on `plot`, the `ArrowUploader` manages user auth token refresh on session
state. 

`Plottable._pygraphistry.session` = `Graphistry.session`

A `Plottable`'s state is tied to the client used to create it.

Example
-------

.. code-block:: python

   import graphistry
   # Independent clients
   alice = graphistry.client().register(api=3, username='alice', password='pw')
   bob   = graphistry.client().register(api=3, username='bob',   password='pw')

   alice.privacy(mode='public')
   g = alice.bind(...)
   # g Plottable has reference to alice's client
   # The upload may trigger a token refresh
   url_a = g.plot(render=False)

   # Bob can take ownership of the Plottable
   g_b = bob.set_client_for(g)
   
   # Privacy can be changed on the Plottable, independent of the client
   g_b.privacy(mode='org')
   url_b = g_b.plot(render=False)

   


Performance notes
-----------------

Creating a client costs an auth round trip; unless you set the jwt manually.
Each :py:meth:`plot()` may refresh tokens. Watch connection limits and memory if many clients stay resident.
