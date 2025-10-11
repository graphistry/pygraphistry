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

Policies are invoked at ten distinct phases:

**preload**
    Before data is loaded (local or remote). Can prevent data access.

**postload**
    After data is loaded. Can check size/content and deny further processing.

**prelet**
    Before ``let()`` DAG execution starts. Can control entire DAG execution and validate DAG structure.

**postlet**
    After ``let()`` DAG execution completes (even on error). Can track DAG-level performance and enforce DAG-level policies.

**prechain**
    Before chain operations execute. Can control entire chain execution and validate chain structure.

**postchain**
    After chain operations complete (even on error). Can track chain-level performance and enforce chain-level policies.

**preletbinding**
    Before each binding execution in ``let()`` DAGs. Can control per-binding execution and validate dependencies.

**postletbinding**
    After each binding execution (even on error). Can track binding performance and enforce per-binding policies.

**precall**
    Before method execution (hop, filter, etc.). Can control operations and validate parameters.

**postcall**
    After method execution. Can validate result size, track execution time, and log performance.


Context Fields
--------------

The context dictionary passed to policy functions contains:

**Always present:**

- ``phase``: Current phase ('preload', 'postload', 'prelet', 'postlet', 'prechain', 'postchain', 'precall', 'postcall', 'preletbinding', 'postletbinding')
- ``hook``: Hook name (same as phase, useful for shared handlers)
- ``_policy_depth``: Internal recursion counter

**Usually present:**

- ``query``: Global/original query AST (None in call context)
- ``current_ast``: Current sub-AST being executed (None in call context for method calls)
- ``query_type``: Type of query ('chain', 'dag', 'single', 'call')

**Phase-specific:**

- ``plottable``: Graph instance (postload/precall/postcall phases)
- ``graph_stats``: Data statistics as GraphStats TypedDict (postload/precall/postcall phases)
- ``call_op``: Operation name (precall/postcall phases only)
- ``call_params``: Operation parameters (precall/postcall phases only)
- ``execution_time``: Method execution duration in seconds (postcall phase only)
- ``success``: Execution success flag (postcall/postlet/postchain/postletbinding phases)
- ``error``: Error message string (post* phases when success=False)
- ``error_type``: Error type name (post* phases when success=False)

**Binding-specific** (preletbinding/postletbinding phases only):

- ``binding_name``: Name of the current binding being executed
- ``binding_index``: Execution order of this binding (0-indexed)
- ``total_bindings``: Total number of bindings in the let expression
- ``binding_dependencies``: List of binding names this binding depends on
- ``binding_ast``: The AST object being bound (the value in let({name: ast}))

**Hierarchy/Tracing fields** (all phases):

- ``execution_depth``: Nesting depth (0=query, 1=let/chain, 2=binding/op, 3=call)
- ``operation_path``: Unique operation identifier like "query.dag.binding:hg.call:hypergraph"
- ``parent_operation``: Parent operation path (for OpenTelemetry span relationships)

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


**Control Operation Execution and Performance**

.. code-block:: python

    def operation_control_policy(context):
        if context['phase'] == 'precall':
            # Validate operation parameters before execution
            op = context.get('call_op', '')
            params = context.get('call_params', {})

            # Deny expensive operations
            if op == 'hop' and params.get('hops', 0) > 3:
                raise PolicyException(
                    'precall',
                    f"Too many hops: {params['hops']} > 3",
                    code=413
                )

        elif context['phase'] == 'postcall':
            # Track execution performance
            exec_time = context.get('execution_time', 0)
            success = context.get('success', False)

            if not success:
                raise PolicyException(
                    'postcall',
                    'Operation failed',
                    code=500
                )

            # Log slow operations
            if exec_time > 5.0:  # 5 seconds
                print(f"Slow operation detected: {exec_time:.2f}s")

            # Validate result size
            stats = context.get('graph_stats', {})
            if stats.get('nodes', 0) > 50000:
                raise PolicyException(
                    'postcall',
                    f"Result too large: {stats['nodes']} nodes",
                    code=413
                )

    g.gfql(query, policy={
        'precall': operation_control_policy,
        'postcall': operation_control_policy
    })


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


**Per-Binding Control**

.. code-block:: python

    def binding_policy(context):
        # Control execution of specific bindings
        if context['phase'] == 'preletbinding':
            binding_name = context.get('binding_name')
            deps = context.get('binding_dependencies', [])

            # Deny bindings with too many dependencies
            if len(deps) > 5:
                raise PolicyException(
                    'preletbinding',
                    f"Binding '{binding_name}' has too many dependencies: {len(deps)}",
                    code=413
                )

        elif context['phase'] == 'postletbinding':
            # Track binding performance
            binding_name = context.get('binding_name')
            success = context.get('success', False)

            if not success:
                error = context.get('error', 'Unknown error')
                print(f"Binding '{binding_name}' failed: {error}")

    from graphistry.compute.ast import ASTLet, n, call

    dag = ASTLet({
        'people': n({'type': 'person'}),
        'orgs': n({'type': 'org'}),
        'connections': call('hypergraph', {})
    })

    g.gfql(dag, policy={
        'preletbinding': binding_policy,
        'postletbinding': binding_policy
    })


**Track Usage**

.. code-block:: python

    def create_usage_tracker():
        stats = {'calls': 0, 'data_loaded': 0, 'execution_times': []}

        def track(context):
            if context['phase'] == 'precall':
                stats['calls'] += 1
            elif context['phase'] == 'postcall':
                # Track execution performance
                exec_time = context.get('execution_time', 0)
                stats['execution_times'].append(exec_time)
            elif context['phase'] == 'postload':
                data = context.get('graph_stats', {})
                stats['data_loaded'] += data.get('nodes', 0)

        return track, stats

    tracker, stats = create_usage_tracker()
    g.gfql(query, policy={
        'postload': tracker,
        'precall': tracker,
        'postcall': tracker
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
        elif hook == 'precall':
            # Operation control and parameter validation
            pass
        elif hook == 'postcall':
            # Performance tracking and result validation
            pass

    # Use same handler for all phases
    g.gfql(query, policy={
        'preload': universal_policy,
        'postload': universal_policy,
        'precall': universal_policy,
        'postcall': universal_policy
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

    # query_type will be 'call', precall and postcall phases triggered
    from graphistry.compute.ast import call
    g.gfql(call('hop', {'hops': 2}), policy={
        'precall': my_precall_policy,
        'postcall': my_postcall_policy
    })

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
        'preload': preload_function,              # Optional
        'postload': postload_function,            # Optional
        'prelet': prelet_function,                # Optional
        'postlet': postlet_function,              # Optional
        'prechain': prechain_function,            # Optional
        'postchain': postchain_function,          # Optional
        'preletbinding': preletbinding_function,  # Optional
        'postletbinding': postletbinding_function,# Optional
        'precall': precall_function,              # Optional
        'postcall': postcall_function             # Optional
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

- ``phase`` (str): Phase where denial occurred ('preload', 'postload', 'prelet', 'postlet', 'prechain', 'postchain', 'preletbinding', 'postletbinding', 'precall', 'postcall')
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