GFQL Policy Hooks
=================

Policy hooks provide external control over GFQL query execution, enabling security, resource management, and usage tracking.

Quick Start
-----------

.. code-block:: python

    from graphistry.compute.gfql.policy import PolicyException

    def my_policy(context):
        # Deny remote data loading
        if context.get('is_remote'):
            dataset_id = context['current_ast'].dataset_id
            if dataset_id == 'forbidden':
                raise PolicyException('preload', 'Access denied', code=403)

    # Apply policy to query
    g.gfql(query, policy={'preload': my_policy})


Policy Phases
-------------

Policies are invoked at three distinct phases:

**preload**
    Before data is loaded (local or remote). Can prevent data access.

**postload**
    After data is loaded. Can check size/content and deny further processing.

**call**
    Before method execution (hop, filter, etc.). Can control operations.


Context Fields
--------------

The context dictionary passed to policy functions contains:

**Always present:**

- ``phase``: Current phase ('preload', 'postload', 'call')
- ``hook``: Hook name (same as phase, useful for shared handlers)
- ``query``: Global/original query AST
- ``current_ast``: Current sub-AST being executed
- ``query_type``: Type of query ('chain', 'dag', 'single')
- ``_policy_depth``: Internal recursion counter

**Phase-specific:**

- ``plottable``: Graph instance (postload/call phases)
- ``graph_stats``: Data statistics (postload phase)
- ``call_op``: Operation name (call phase)
- ``call_params``: Operation parameters (call phase)

**Remote operations:**

- ``is_remote``: True for network operations
- ``engine``: Current engine ('pandas', 'cudf', etc.)


Examples
--------

**Limit Data Size**

.. code-block:: python

    def size_limit_policy(context):
        if context['phase'] == 'postload':
            stats = context.get('graph_stats', {})
            if stats.get('nodes', 0) > 10000:
                raise PolicyException(
                    'postload',
                    f"Too many nodes: {stats['nodes']}",
                    code=413
                )

    g.gfql(query, policy={'postload': size_limit_policy})


**Control Remote Access**

.. code-block:: python

    def remote_access_policy(context):
        if context.get('is_remote'):
            # Check JWT token for remote operations
            ast = context['current_ast']
            if hasattr(ast, 'token') and not ast.token:
                raise PolicyException(
                    'preload',
                    'Authentication required',
                    code=401
                )

    g.gfql(query, policy={'preload': remote_access_policy})


**Track Usage**

.. code-block:: python

    def create_usage_tracker():
        stats = {'calls': 0, 'data_loaded': 0}

        def track(context):
            if context['phase'] == 'call':
                stats['calls'] += 1
            elif context['phase'] == 'postload':
                data = context.get('graph_stats', {})
                stats['data_loaded'] += data.get('nodes', 0)

        return track, stats

    tracker, stats = create_usage_tracker()
    g.gfql(query, policy={
        'postload': tracker,
        'call': tracker
    })
    print(f"Usage: {stats}")


**Shared Handler**

.. code-block:: python

    def universal_policy(context):
        hook = context['hook']  # Which hook fired

        if hook == 'preload':
            # Pre-execution checks
            pass
        elif hook == 'postload':
            # Data validation
            pass
        elif hook == 'call':
            # Operation control
            pass

    # Use same handler for all phases
    g.gfql(query, policy={
        'preload': universal_policy,
        'postload': universal_policy,
        'call': universal_policy
    })


PolicyException
---------------

Deny operations by raising ``PolicyException``:

.. code-block:: python

    from graphistry.compute.gfql.policy import PolicyException

    raise PolicyException(
        phase='preload',      # Which phase denied
        reason='Forbidden',   # Human-readable reason
        code=403,            # HTTP-like status code
        **kwargs             # Additional context
    )

The exception can be enriched with additional fields for logging/debugging.


Thread Safety
-------------

Policy execution is thread-safe with built-in recursion prevention. Policies are not invoked recursively when operations trigger internal queries.


Integration with Hub
--------------------

The policy system is designed for Graphistry Hub integration:

1. Hub creates policies based on user tier/permissions
2. Policies enforce resource limits and feature access
3. Usage tracking for billing/analytics
4. JWT token validation for remote operations

.. code-block:: python

    # Hub example
    def create_tier_policy(tier='free'):
        limits = {
            'free': {'max_nodes': 1000},
            'pro': {'max_nodes': 100000}
        }

        def policy(context):
            if context['phase'] == 'postload':
                stats = context.get('graph_stats', {})
                if stats.get('nodes', 0) > limits[tier]['max_nodes']:
                    raise PolicyException(
                        'postload',
                        f'{tier} tier limit exceeded',
                        code=403
                    )

        return policy