# GFQL Programs User Personas

## Overview
These personas represent key user types who would benefit from GFQL Programs features based on our Phase 1 analysis.

## Persona 1: Alex - The Security Analyst
**Role**: Security Operations Center (SOC) Analyst  
**Experience**: Intermediate Python, expert in security tools  
**Primary Need**: Investigate security incidents by correlating data across multiple sources

**Key GFQL Features Used**:
- Remote graph loading (pulling from different security datasets)
- Graph combinators (merging alerts with network data)
- DAG composition (multi-step investigations)

**Pain Points**:
- Currently has to manually combine data from different systems
- Python scripts become complex when doing multi-hop analysis
- Hard to share investigation workflows with team

## Persona 2: Sam - The Data Scientist
**Role**: Senior Data Scientist at a financial institution  
**Experience**: Expert Python/pandas, some graph experience  
**Primary Need**: Build reusable graph analysis pipelines for fraud detection

**Key GFQL Features Used**:
- Call operations (UMAP, clustering algorithms)
- DAG composition (complex multi-stage pipelines)
- Graph combinators (enriching transaction graphs)

**Pain Points**:
- Wants declarative workflows instead of imperative code
- Needs to version and share analysis pipelines
- Resource limits when processing large transaction graphs

## Persona 3: Jordan - The Business Analyst
**Role**: Business Intelligence Analyst  
**Experience**: SQL expert, basic Python  
**Primary Need**: Create graph reports combining multiple data sources without deep programming

**Key GFQL Features Used**:
- Remote graph loading (pre-built department graphs)
- Simple combinators (union, intersection)
- Basic call operations (layout, simple filters)

**Pain Points**:
- Limited programming skills but needs complex analysis
- Wants SQL-like declarative syntax for graphs
- Needs clear error messages when things go wrong

## Persona 4: Morgan - The DevOps Engineer
**Role**: Platform Engineer supporting data teams  
**Experience**: Expert in infrastructure, basic data analysis  
**Primary Need**: Monitor and analyze infrastructure dependencies and service meshes

**Key GFQL Features Used**:
- DAG composition (service dependency tracking)
- Dotted references (navigating nested infrastructure)
- Call operations (topology analysis, layout)

**Pain Points**:
- Complex service dependencies need multi-level analysis
- Performance concerns with large infrastructure graphs
- Need to set resource limits for different teams

## Persona 5: Casey - The Compliance Officer
**Role**: Regulatory Compliance Analyst  
**Experience**: Domain expert, minimal programming  
**Primary Need**: Run standard compliance checks across entity relationship graphs

**Key GFQL Features Used**:
- Remote graphs (loading standard compliance datasets)
- Graph combinators (comparing against watchlists)
- Pre-built call operations (compliance-specific algorithms)

**Pain Points**:
- Needs audit trails for all operations
- Must handle sensitive data appropriately
- Requires reproducible, documented workflows

## Persona 6: Riley - The Research Scientist
**Role**: Computational Biologist  
**Experience**: R/Python expert, graph algorithm knowledge  
**Primary Need**: Analyze biological networks with custom algorithms

**Key GFQL Features Used**:
- Call operations (custom scientific algorithms)
- Complex DAGs (multi-stage analysis pipelines)
- Graph combinators (merging different biological datasets)

**Pain Points**:
- Needs to integrate custom algorithms
- Large graphs hit memory limits
- Wants to parallelize complex analyses