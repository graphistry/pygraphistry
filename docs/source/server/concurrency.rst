Concurrency & Multi-tenancy
===========================

To safely use pygraphistry in concurrent and multitenant settings, use client objects. Use of top-level calls like ``register()`` and ``plot()`` are unsafe in these settings as they use global variables.

Client objects automatically isolate session state like graphistry server and database tokens. Each client and derived plottable objects are safe for use within a single concurrency context like a thread or event loop, and you can have multiple in different threads and event loops.

Creating Client Objects
-----------------------

.. code-block:: python

   import graphistry
   
   # Create independent client instances
   alice_g = graphistry.client()
   alice_g.register(api=3, username='alice', password='pw')
   
   bob_g = graphistry.client()
   bob_g.register(api=3, username='bob', password='pw')

Multi-tenant Example
--------------------

Different users can work with their own isolated sessions:

.. code-block:: python

   import graphistry
   
   # Alice's client with her credentials and settings
   alice_g = graphistry.client()
   alice_g.register(api=3, username='alice', password='alice_pw')
   alice_g.privacy(mode='public')
   
   # Bob's client with his credentials and settings
   bob_g = graphistry.client()
   bob_g.register(api=3, username='bob', password='bob_pw')
   bob_g.privacy(mode='org')
   
   # Each client creates isolated plottables
   alice_plot = alice_g.edges(alice_data).plot(render=False)
   bob_plot = bob_g.edges(bob_data).plot(render=False)

Multiple Servers Example
------------------------

Connect to different Graphistry servers (e.g., staging vs production):

.. code-block:: python

   import graphistry
   
   # Production server client
   prod_g = graphistry.client()
   prod_g.register(
       api=3,
       server='prod.graphistry.com',
       username='user',
       password='pw'
   )
   
   # Staging server client
   staging_g = graphistry.client()
   staging_g.register(
       api=3,
       server='staging.graphistry.com',
       username='user',
       password='pw'
   )
   
   # Use different servers for different purposes
   prod_url = prod_g.edges(production_data).plot(render=False)
   staging_url = staging_g.edges(test_data).plot(render=False)

Multi-threaded Example
----------------------

Each thread should use its own client instance:

.. code-block:: python

   import graphistry
   import threading
   
   def process_user_data(username, password, data):
       # Each thread creates its own client
       g = graphistry.client()
       g.register(api=3, username=username, password=password)
       
       # Process and plot data
       url = g.edges(data).plot(render=False)
       return url
   
   # Launch threads with separate clients
   threads = []
   for user_info in users:
       t = threading.Thread(
           target=process_user_data,
           args=(user_info['username'], user_info['password'], user_info['data'])
       )
       threads.append(t)
       t.start()
   
   # Wait for all threads to complete
   for t in threads:
       t.join()

Transferring Plottables Between Clients
---------------------------------------

You can transfer ownership of a plottable from one client to another:

.. code-block:: python

   import graphistry
   
   # Create plottable with Alice's client
   alice_g = graphistry.client()
   alice_g.register(api=3, username='alice', password='alice_pw')
   g = alice_g.edges(data)
   
   # Transfer to Bob's client
   bob_g = graphistry.client()
   bob_g.register(api=3, username='bob', password='bob_pw')
   g_bob = bob_g.set_client_for(g)
   
   # Now the plottable uses Bob's credentials and settings
   url = g_bob.plot(render=False)

Performance Considerations
--------------------------

- Creating a client requires an authentication round trip, unless you set the JWT token manually
- Each ``plot()`` call may refresh authentication tokens
