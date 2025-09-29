GFQL Policy Hooks
=================

Policy hooks provide external control over GFQL query execution, enabling security, resource management, and usage tracking.

Quick Start
-----------

.. code-block:: python

    from graphistry.compute.gfql.policy import PolicyException

    def my_policy(context):
        # Deny remote data loading for specific datasets
        if context.get('is_remote'):
            # For remote operations, current_ast is ASTRemoteGraph
            ast = context.get('current_ast')
            if hasattr(ast, 'dataset_id') and ast.dataset_id == 'forbidden':
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
- ``_policy_depth``: Internal recursion counter

**Usually present:**

- ``query``: Global/original query AST (None in call context)
- ``current_ast``: Current sub-AST being executed (None in call context for method calls)
- ``query_type``: Type of query ('chain', 'dag', 'single', 'call')

**Phase-specific:**

- ``plottable``: Graph instance (postload/call phases)
- ``graph_stats``: Data statistics as GraphStats TypedDict (postload/call phases)
- ``call_op``: Operation name (call phase only)
- ``call_params``: Operation parameters (call phase only)

**Context-specific:**

- ``is_remote``: True for remote data operations (ASTRemoteGraph)
- ``engine``: Current engine value when available


GraphStats Type
---------------

The ``graph_stats`` field provides typed statistics:

.. code-block:: python

    from graphistry.compute.gfql.policy import GraphStats

    # GraphStats is a TypedDict with:
    # - nodes: int (number of nodes)
    # - edges: int (number of edges)
    # - node_bytes: int (memory usage)
    # - edge_bytes: int (memory usage)


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

Policy execution is thread-safe with built-in recursion prevention. Policies are not invoked recursively when operations trigger internal queries (depth limit of 1).


Remote Data Loading
-------------------

Policies can control remote data operations (``ASTRemoteGraph``). When ``is_remote`` is True in the context, the operation involves loading data from a remote source:

.. code-block:: python

    def remote_data_policy(context):
        # Check remote operations in preload phase
        if context['phase'] == 'preload' and context.get('is_remote'):
            ast = context.get('current_ast')

            # For ASTRemoteGraph, check dataset_id
            if hasattr(ast, 'dataset_id'):
                if ast.dataset_id in banned_datasets:
                    raise PolicyException('preload', 'Dataset blocked')

                # Check for JWT token
                if hasattr(ast, 'token') and not validate_jwt(ast.token):
                    raise PolicyException('preload', 'Invalid token', code=401)

        # Check size after remote data loads
        elif context['phase'] == 'postload' and context.get('is_remote'):
            stats = context.get('graph_stats', {})
            if stats.get('nodes', 0) > remote_limit:
                raise PolicyException('postload', 'Remote data too large')

Remote operations trigger both preload and postload hooks, allowing control before and after data transfer.


Query Types
-----------

Policies work with different GFQL query patterns:

**Chain queries** - Sequential operations:

.. code-block:: python

    # query_type will be 'chain'
    g.gfql([n(), e(), n()], policy=policy_dict)

**DAG queries** - Named bindings with dependencies:

.. code-block:: python

    # query_type will be 'dag'
    g.gfql({'persons': n({'type': 'person'})}, policy=policy_dict)

**Call operations** - Method invocations:

.. code-block:: python

    # query_type will be 'call', only 'call' phase triggered
    from graphistry.compute.ast import call
    g.gfql(call('hop', {'hops': 2}), policy={'call': my_policy})

Each query type provides appropriate context to the policy for decision making.


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


Advanced Topics
---------------

**Policy Composition**

Combine multiple policies using composition patterns:

.. code-block:: python

    def compose_policies(*policies):
        """Compose multiple policies into one."""
        def composed(context):
            for policy in policies:
                policy(context)  # Each can raise PolicyException
        return composed

    # Use composed policy
    combined = compose_policies(
        size_limit_policy,
        rate_limit_policy,
        tier_policy
    )
    g.gfql(query, policy={'postload': combined})


**Stateful Policies with Closures**

Track state across multiple queries:

.. code-block:: python

    def create_rate_limiter(max_per_minute=60):
        from collections import deque
        from time import time

        calls = deque()

        def policy(context):
            if context['phase'] == 'preload':
                now = time()
                # Remove calls older than 1 minute
                while calls and calls[0] < now - 60:
                    calls.popleft()

                if len(calls) >= max_per_minute:
                    raise PolicyException(
                        'preload',
                        'Rate limit exceeded',
                        code=429
                    )
                calls.append(now)

        return policy


**Testing Policies**

Test policies in isolation:

.. code-block:: python

    def test_policy():
        # Create mock context
        context = {
            'phase': 'postload',
            'graph_stats': {'nodes': 5000},
            '_policy_depth': 0
        }

        # Test acceptance
        my_policy(context)  # Should not raise

        # Test denial
        context['graph_stats']['nodes'] = 50000
        with pytest.raises(PolicyException) as exc:
            my_policy(context)
        assert exc.value.code == 413


**Performance Considerations**

- Policies execute synchronously - keep them lightweight
- Use caching for expensive validations
- Consider async patterns for external calls (future enhancement)
- Recursion prevention adds minimal overhead (depth limit of 1)


**Debugging Policies**

Use logging to debug policy decisions:

.. code-block:: python

    import logging
    logger = logging.getLogger(__name__)

    def debug_policy(context):
        phase = context['phase']
        logger.debug(f"Policy called: phase={phase}")

        if phase == 'postload':
            stats = context.get('graph_stats', {})
            logger.debug(f"Graph stats: {stats}")

            if stats.get('nodes', 0) > limit:
                logger.warning(f"Denying: {stats['nodes']} > {limit}")
                raise PolicyException(...)

        logger.debug(f"Policy accepted in {phase}")


API Reference
-------------

**Main Interface**

.. code-block:: python

    g.gfql(query, policy={
        'preload': preload_function,   # Optional
        'postload': postload_function, # Optional
        'call': call_function          # Optional
    })

**Imports**

.. code-block:: python

    from graphistry.compute.gfql.policy import (
        PolicyException,  # Exception class for denying operations
        PolicyContext,   # TypedDict for context parameter
        GraphStats,      # TypedDict for graph statistics
        PolicyFunction,  # Type alias for policy functions
        PolicyDict       # Type alias for policy dictionary
    )

**PolicyException Parameters**

- ``phase`` (str): Phase where denial occurred ('preload', 'postload', 'call')
- ``reason`` (str): Human-readable explanation
- ``code`` (int): HTTP-like status code (default: 403)
- ``query_type`` (str, optional): Type of query being executed
- ``data_size`` (dict, optional): Graph statistics at time of denial

**Common HTTP Status Codes**

- ``401``: Unauthorized (authentication required)
- ``403``: Forbidden (authenticated but not allowed)
- ``413``: Payload too large (data size limit exceeded)
- ``429``: Too many requests (rate limit exceeded)
- ``503``: Service unavailable (resource constraints)