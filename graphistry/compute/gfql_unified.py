"""GFQL unified entrypoint for chains and DAGs"""

from typing import List, Union, Optional
from graphistry.Plottable import Plottable
from graphistry.Engine import EngineAbstract
from graphistry.util import setup_logger
from .ast import ASTObject, ASTLet, ASTNode, ASTEdge
from .chain import Chain, chain as chain_impl
from .chain_let import chain_let as chain_let_impl

logger = setup_logger(__name__)


def gfql(self: Plottable,
         query: Union[ASTObject, List[ASTObject], ASTLet, Chain, dict],
         engine: Union[EngineAbstract, str] = EngineAbstract.AUTO,
         output: Optional[str] = None) -> Plottable:
    """
    Execute a GFQL query - either a chain or a DAG
    
    Unified entrypoint that automatically detects query type and
    dispatches to the appropriate execution engine.
    
    :param query: GFQL query - ASTObject, List[ASTObject], Chain, ASTLet, or dict
    :param engine: Execution engine (auto, pandas, cudf)
    :param output: For DAGs, name of binding to return (default: last executed)
    :returns: Resulting Plottable
    :rtype: Plottable
    
    **Example: Chain query**
    
    ::
    
        from graphistry.compute.ast import n, e
        
        # As list
        result = g.gfql([n({'type': 'person'}), e(), n()])
        
        # As Chain object
        from graphistry.compute.chain import Chain
        result = g.gfql(Chain([n({'type': 'person'}), e(), n()]))
    
    **Example: DAG query**
    
    ::
    
        from graphistry.compute.ast import let, ref, n, e
        
        result = g.gfql(let({
            'people': n({'type': 'person'}),
            'friends': ref('people', [e({'rel': 'knows'}), n()])
        }))
        
        # Select specific output
        friends = g.gfql(result, output='friends')
    
    **Example: Transformations (e.g., hypergraph)**

    ::

        from graphistry.compute import hypergraph

        # Simple transformation
        hg = g.gfql(hypergraph(entity_types=['user', 'product']))

        # Or using call()
        from graphistry.compute.ast import call
        hg = g.gfql(call('hypergraph', {'entity_types': ['user', 'product']}))

        # In a DAG with other operations
        result = g.gfql(let({
            'hg': hypergraph(entity_types=['user', 'product']),
            'filtered': ref('hg', [n({'type': 'user'})])
        }))

    **Example: Auto-detection**

    ::

        # List → chain execution
        g.gfql([n(), e(), n()])

        # Single ASTObject → chain execution
        g.gfql(n({'type': 'person'}))

        # Dict → DAG execution (convenience)
        g.gfql({'people': n({'type': 'person'})})
    """
    # Handle dict convenience first (convert to ASTLet)
    if isinstance(query, dict):
        # Auto-wrap ASTNode and ASTEdge values in Chain for GraphOperation compatibility
        wrapped_dict = {}
        for key, value in query.items():
            if isinstance(value, (ASTNode, ASTEdge)):
                logger.debug(f'Auto-wrapping {type(value).__name__} in Chain for dict key "{key}"')
                wrapped_dict[key] = Chain([value])
            else:
                wrapped_dict[key] = value
        query = ASTLet(wrapped_dict)  # type: ignore
    
    # Dispatch based on type - check specific types before generic
    if isinstance(query, ASTLet):
        logger.debug('GFQL executing as DAG')
        return chain_let_impl(self, query, engine, output)
    elif isinstance(query, Chain):
        logger.debug('GFQL executing as Chain')
        if output is not None:
            logger.warning('output parameter ignored for chain queries')
        return chain_impl(self, query.chain, engine)
    elif isinstance(query, ASTObject):
        # Single ASTObject -> execute as single-item chain
        logger.debug('GFQL executing single ASTObject as chain')
        if output is not None:
            logger.warning('output parameter ignored for chain queries')
        return chain_impl(self, [query], engine)
    elif isinstance(query, list):
        logger.debug('GFQL executing list as chain')
        if output is not None:
            logger.warning('output parameter ignored for chain queries')
        return chain_impl(self, query, engine)
    else:
        raise TypeError(
            f"Query must be ASTObject, List[ASTObject], Chain, ASTLet, or dict. "
            f"Got {type(query).__name__}"
        )
