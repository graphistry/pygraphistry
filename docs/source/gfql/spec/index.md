(gfql-specifications)=

# GFQL Specifications

This section contains formal specifications for GFQL (Graph Frame Query Language), designed to support both human understanding and automated tooling, including LLM-based code synthesis.

```{toctree}
:maxdepth: 1

language
python_embedding
wire_protocol
cypher_mapping
llm_guide
```

## Overview

- {ref}`gfql-spec-language` - Complete formal grammar, operations, predicates, and type system
- {ref}`gfql-spec-python-embedding` - Python-specific implementation with pandas/cuDF
- {ref}`gfql-spec-wire-protocol` - JSON serialization format for client-server communication
- {ref}`gfql-spec-cypher-mapping` - Cypher to GFQL translations with both Python and wire protocol
- {ref}`gfql-spec-llm-guide` - LLM-optimized guide for generating valid GFQL JSON (Claude, GPT, etc.)

These specifications are optimized for text-to-GFQL synthesis, Cypher-to-GFQL pipelines, query validation, and schema-aware code generation.

### Tiny diagrams

```{graphviz}
digraph gfql_spec_toy {
  rankdir=LR;
  a -> b -> c -> d;
  a [shape=box, style=filled, fillcolor=lightgray];
}
```

```{code-block} mermaid
graph LR
  a --> b --> c --> d
  %% positions optional; see plot_static(engine="mermaid-code") for generated DSL
```
