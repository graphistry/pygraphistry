(gfql-specifications)=

# GFQL Specifications

This section contains formal specifications for GFQL (Graph Frame Query Language), designed to support both human understanding and LLM-based code synthesis.

## Available Specifications

```{toctree}
:maxdepth: 1

language
wire_protocol
cypher_mapping
synthesis_examples
```

## Overview

These specifications provide:

- **Language Specification**: Complete formal grammar, operations, predicates, and type system
- **Wire Protocol**: JSON serialization format for client-server communication
- **Cypher Mapping**: Translation rules between Cypher and GFQL
- **Synthesis Examples**: Comprehensive examples for LLM training and code generation

## For LLM Integration

These specifications are optimized for:

- Text-to-GFQL synthesis
- Text-to-Cypher-to-GFQL pipelines
- Query validation and error correction
- Schema-aware code generation

## Quick Links

- {ref}`gfql-spec-language` - Formal language specification
- {ref}`gfql-spec-wire-protocol` - JSON wire protocol
- {ref}`gfql-spec-cypher-mapping` - Cypher translation guide
- {ref}`gfql-spec-synthesis-examples` - Code synthesis examples