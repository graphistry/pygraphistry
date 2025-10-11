OpenTelemetry Integration
=========================

GFQL policy hooks provide built-in support for OpenTelemetry span tracing with proper parent-child relationships, enabling complete observability of query execution.

Quick Start
-----------

.. code-block:: python

    from opentelemetry import trace
    from opentelemetry.trace import Status, StatusCode

    tracer = trace.get_tracer(__name__)
    span_map = {}  # operation_path → span

    def create_span(ctx):
        """Start span in pre* hooks"""
        # Get parent span using parent_operation
        parent_span = span_map.get(ctx.get('parent_operation'))

        # Create span with unique operation_path as name
        span = tracer.start_span(
            ctx['operation_path'],
            parent=parent_span
        )

        # Add span attributes from context
        span.set_attribute('execution_depth', ctx['execution_depth'])
        span.set_attribute('query_type', ctx.get('query_type', 'unknown'))

        # Add phase-specific attributes
        if ctx.get('binding_name'):
            span.set_attribute('binding_name', ctx['binding_name'])
        if ctx.get('call_op'):
            span.set_attribute('call_op', ctx['call_op'])

        # Store span for children and post hook
        span_map[ctx['operation_path']] = span

    def end_span(ctx):
        """End span in post* hooks"""
        span = span_map.pop(ctx['operation_path'], None)
        if not span:
            return

        # Add result attributes
        if ctx.get('graph_stats'):
            stats = ctx['graph_stats']
            span.set_attribute('nodes', stats.get('nodes', 0))
            span.set_attribute('edges', stats.get('edges', 0))

        # Add execution time if available
        if ctx.get('execution_time'):
            span.set_attribute('execution_time_sec', ctx['execution_time'])

        # Handle errors
        if not ctx.get('success', True):
            span.set_status(
                Status(StatusCode.ERROR, ctx.get('error', 'Unknown error'))
            )
            span.set_attribute('error_type', ctx.get('error_type', 'Unknown'))
            if ctx.get('error'):
                span.set_attribute('error_message', ctx['error'])

        span.end()

    # Apply to all hook phases
    policy = {
        'preload': create_span,
        'postload': end_span,
        'preletbinding': create_span,
        'postletbinding': end_span,
        'precall': create_span,
        'postcall': end_span
    }

    # Execute query with tracing
    result = g.gfql(my_query, policy=policy)


Hierarchy Fields
----------------

Three key fields enable proper span tracing:

**execution_depth**
    Nesting depth of execution (0=query, 1=let/chain, 2=binding/op, 3=call)

**operation_path**
    Unique identifier for each operation in the execution tree

    Examples:

    - ``"query"`` - Entry point
    - ``"query.dag"`` - DAG execution
    - ``"query.dag.binding:people"`` - Binding named "people"
    - ``"query.dag.binding:hg.call:hypergraph"`` - Hypergraph call within "hg" binding

**parent_operation**
    Path to parent operation, enabling correct span parent-child relationships

These fields are populated in all hook phases (preload, postload, preletbinding, postletbinding, precall, postcall).


Span Hierarchy Example
-----------------------

For this query:

.. code-block:: python

    from graphistry.compute.ast import ASTLet, n, call

    dag = ASTLet({
        'people': n({'type': 'person'}),
        'hg': call('hypergraph', {}),
        'filtered': ASTRef('hg', [n()])
    })

    result = g.gfql(dag, policy=policy)

You get this span tree:

.. code-block:: text

    query  (depth=0)
    └── query.dag  (depth=1)
        ├── query.dag.binding:people  (depth=1)
        │   └── query.dag.binding:people.chain  (depth=2)
        ├── query.dag.binding:hg  (depth=1)
        │   └── query.dag.binding:hg.call:hypergraph  (depth=2)
        └── query.dag.binding:filtered  (depth=1)
            └── query.dag.binding:filtered.chain  (depth=2)

Each operation has a unique path and correct parent relationship.


Complete Example
----------------

.. code-block:: python

    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        ConsoleSpanExporter,
        BatchSpanProcessor
    )
    from opentelemetry.trace import Status, StatusCode

    # Setup OpenTelemetry
    provider = TracerProvider()
    processor = BatchSpanProcessor(ConsoleSpanExporter())
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)

    tracer = trace.get_tracer(__name__)


    class OpenTelemetryPolicy:
        """Reusable OpenTelemetry policy for GFQL"""

        def __init__(self, tracer=None):
            self.tracer = tracer or trace.get_tracer(__name__)
            self.span_map = {}

        def pre_hook(self, ctx):
            """Start span in pre* hooks"""
            # Get parent span
            parent_path = ctx.get('parent_operation')
            parent_span = self.span_map.get(parent_path)

            # Create span
            span = self.tracer.start_span(
                ctx['operation_path'],
                parent=parent_span
            )

            # Add common attributes
            span.set_attribute('execution_depth', ctx['execution_depth'])
            span.set_attribute('query_type', ctx.get('query_type', 'unknown'))
            span.set_attribute('phase', ctx['phase'])

            # Add phase-specific attributes
            if ctx.get('binding_name'):
                span.set_attribute('binding_name', ctx['binding_name'])
                span.set_attribute('binding_index', ctx.get('binding_index', -1))
                span.set_attribute('total_bindings', ctx.get('total_bindings', -1))
                deps = ctx.get('binding_dependencies', [])
                if deps:
                    span.set_attribute('binding_dependencies', ','.join(deps))

            if ctx.get('call_op'):
                span.set_attribute('call_op', ctx['call_op'])

            if ctx.get('is_remote'):
                span.set_attribute('is_remote', True)

            # Store for children and post hook
            self.span_map[ctx['operation_path']] = span

        def post_hook(self, ctx):
            """End span in post* hooks"""
            span = self.span_map.pop(ctx['operation_path'], None)
            if not span:
                return

            # Add result attributes
            if ctx.get('graph_stats'):
                stats = ctx['graph_stats']
                span.set_attribute('result_nodes', stats.get('nodes', 0))
                span.set_attribute('result_edges', stats.get('edges', 0))
                if stats.get('node_bytes'):
                    span.set_attribute('result_node_bytes', stats['node_bytes'])
                if stats.get('edge_bytes'):
                    span.set_attribute('result_edge_bytes', stats['edge_bytes'])

            # Add execution time
            if ctx.get('execution_time'):
                span.set_attribute('execution_time_sec', ctx['execution_time'])

            # Handle errors
            success = ctx.get('success', True)
            span.set_attribute('success', success)

            if not success:
                error_msg = ctx.get('error', 'Unknown error')
                span.set_status(Status(StatusCode.ERROR, error_msg))
                span.set_attribute('error_type', ctx.get('error_type', 'Unknown'))
                span.set_attribute('error_message', error_msg)

            span.end()

        def get_policy_dict(self):
            """Get policy dictionary for gfql()"""
            return {
                'preload': self.pre_hook,
                'postload': self.post_hook,
                'preletbinding': self.pre_hook,
                'postletbinding': self.post_hook,
                'precall': self.pre_hook,
                'postcall': self.post_hook
            }


    # Use the policy
    otel_policy = OpenTelemetryPolicy(tracer)

    from graphistry.compute.ast import ASTLet, n, call

    dag = ASTLet({
        'people': n({'type': 'person'}),
        'orgs': n({'type': 'org'}),
        'hg': call('hypergraph', {}),
    })

    result = g.gfql(dag, policy=otel_policy.get_policy_dict())

    # Spans are automatically exported to console


Integration with Other Exporters
---------------------------------

**Jaeger**

.. code-block:: python

    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    jaeger_exporter = JaegerExporter(
        agent_host_name='localhost',
        agent_port=6831,
    )
    provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))


**OTLP (OpenTelemetry Protocol)**

.. code-block:: python

    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    otlp_exporter = OTLPSpanExporter(
        endpoint="http://localhost:4317"
    )
    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))


**Custom Exporter**

.. code-block:: python

    from opentelemetry.sdk.trace.export import SpanExporter

    class CustomExporter(SpanExporter):
        def export(self, spans):
            for span in spans:
                # Send to your backend
                print(f"Span: {span.name}, Duration: {span.end_time - span.start_time}")
            return SpanExportResult.SUCCESS

    provider.add_span_processor(BatchSpanProcessor(CustomExporter()))


Attributes Reference
--------------------

The OpenTelemetry policy adds these span attributes:

**Common** (all spans):

- ``execution_depth``: Nesting level (int)
- ``query_type``: Type of query (str)
- ``phase``: Hook phase (str)
- ``success``: Whether operation succeeded (bool)

**Binding-specific**:

- ``binding_name``: Name of binding (str)
- ``binding_index``: Execution order (int)
- ``total_bindings``: Total bindings in DAG (int)
- ``binding_dependencies``: Comma-separated dep list (str)

**Call-specific**:

- ``call_op``: Operation name (str)

**Result metrics**:

- ``result_nodes``: Number of nodes (int)
- ``result_edges``: Number of edges (int)
- ``result_node_bytes``: Node memory (int)
- ``result_edge_bytes``: Edge memory (int)
- ``execution_time_sec``: Duration (float)

**Error attributes** (when success=False):

- ``error_type``: Exception class name (str)
- ``error_message``: Error message (str)

**Context flags**:

- ``is_remote``: Remote data operation (bool)


Best Practices
--------------

**1. Reuse Policy Instances**

Create one policy instance and reuse it across queries:

.. code-block:: python

    otel_policy = OpenTelemetryPolicy()
    policy_dict = otel_policy.get_policy_dict()

    # Use for multiple queries
    result1 = g.gfql(query1, policy=policy_dict)
    result2 = g.gfql(query2, policy=policy_dict)


**2. Add Custom Attributes**

Extend the policy with domain-specific attributes:

.. code-block:: python

    class CustomPolicy(OpenTelemetryPolicy):
        def __init__(self, user_id, session_id):
            super().__init__()
            self.user_id = user_id
            self.session_id = session_id

        def pre_hook(self, ctx):
            super().pre_hook(ctx)
            span = self.span_map[ctx['operation_path']]
            span.set_attribute('user_id', self.user_id)
            span.set_attribute('session_id', self.session_id)


**3. Filter Spans by Depth**

Only trace top-level operations:

.. code-block:: python

    def create_span_filtered(ctx):
        # Only trace depth 0 and 1
        if ctx['execution_depth'] <= 1:
            create_span(ctx)

    policy = {
        'preload': create_span_filtered,
        'postload': end_span,
        # ...
    }


**4. Sampling**

Use OpenTelemetry's built-in sampling:

.. code-block:: python

    from opentelemetry.sdk.trace.sampling import TraceIdRatioBased

    # Sample 10% of traces
    sampler = TraceIdRatioBased(0.1)
    provider = TracerProvider(sampler=sampler)


**5. Error Handling**

Always wrap span operations in try/except:

.. code-block:: python

    def safe_create_span(ctx):
        try:
            create_span(ctx)
        except Exception as e:
            logger.error(f"Failed to create span: {e}")

    def safe_end_span(ctx):
        try:
            end_span(ctx)
        except Exception as e:
            logger.error(f"Failed to end span: {e}")


Performance Considerations
--------------------------

- **Span overhead**: Creating spans adds ~100-500μs per operation
- **Memory**: Each active span uses ~1-2KB of memory
- **Network**: Batch exporting amortizes network cost
- **Sampling**: Use sampling for high-throughput workloads

For production use:

1. Use batch span processors (not simple processors)
2. Configure appropriate batch sizes (default: 512)
3. Enable sampling for high-volume queries
4. Monitor exporter performance


Troubleshooting
---------------

**Spans not appearing**

Check that:

1. Tracer provider is properly initialized
2. Span processor is added to provider
3. Provider is set as global: ``trace.set_tracer_provider(provider)``
4. Exporter is configured correctly

**Missing parent-child relationships**

Verify:

1. Spans are stored in span_map before children are created
2. parent_operation correctly references parent's operation_path
3. Parent span exists when child is created

**Performance issues**

Try:

1. Use BatchSpanProcessor instead of SimpleSpanProcessor
2. Enable sampling: ``TraceIdRatioBased(0.1)``
3. Filter spans by depth: only trace depth 0-1
4. Use async exporters if available


.. _policy-otel-see-also:

See Also
--------

- :doc:`policy` - Full policy hooks documentation
- `OpenTelemetry Python Docs <https://opentelemetry.io/docs/languages/python/>`_
- `OpenTelemetry Specification <https://opentelemetry.io/docs/specs/otel/>`_
