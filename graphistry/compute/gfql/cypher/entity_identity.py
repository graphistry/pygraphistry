from __future__ import annotations

from typing import Mapping

from graphistry.compute.ast import ASTEdge, ASTNode, ASTObject
from graphistry.compute.gfql.expr_parser import (
    BinaryOp,
    CaseWhen,
    ExprNode,
    FunctionCall,
    Identifier,
    IsNullOp,
    ListComprehension,
    ListLiteral,
    Literal as ExprLiteral,
    MapLiteral,
    QuantifierExpr,
    UnaryOp,
    Wildcard,
    _rebuild_expr_node,
)


def _entity_wrapper_call(alias_name: str, *, alias_targets: Mapping[str, ASTObject]) -> ExprNode:
    target = alias_targets.get(alias_name)
    if isinstance(target, ASTNode):
        fn = "__node_entity__"
    elif isinstance(target, ASTEdge):
        fn = "__edge_entity__"
    else:
        return Identifier(alias_name)
    extra_aliases = tuple(
        ExprLiteral(str(name))
        for name in alias_targets.keys()
        if str(name) != alias_name
    )
    return FunctionCall(fn, (Identifier(alias_name),) + extra_aliases)


def _rewrite_collection_alias_entities(
    node: ExprNode,
    *,
    alias_targets: Mapping[str, ASTObject],
    inside_collection: bool = False,
) -> ExprNode:
    def _node_value_expr(value: ExprNode) -> bool:
        if isinstance(value, Identifier):
            return isinstance(alias_targets.get(value.name), ASTNode)
        if isinstance(value, FunctionCall) and value.name.lower() == "head" and len(value.args) == 1:
            source = value.args[0]
            return isinstance(source, ListLiteral) and bool(source.items) and _node_value_expr(source.items[0])
        return False

    def _rewrite_node_value_expr(value: ExprNode) -> ExprNode:
        if isinstance(value, Identifier) and isinstance(alias_targets.get(value.name), ASTNode):
            return FunctionCall("__node_identity__", (Identifier(value.name),))
        if isinstance(value, FunctionCall) and value.name.lower() == "head" and len(value.args) == 1:
            source = value.args[0]
            if isinstance(source, ListLiteral) and source.items:
                return _rewrite_node_value_expr(source.items[0])
        if isinstance(value, FunctionCall):
            return FunctionCall(
                value.name,
                tuple(_rewrite_node_value_expr(arg) for arg in value.args),
                distinct=value.distinct,
            )
        if isinstance(value, ListLiteral):
            return ListLiteral(tuple(_rewrite_node_value_expr(item) for item in value.items))
        return value

    if isinstance(node, Identifier):
        if inside_collection and "." not in node.name and node.name in alias_targets:
            return _entity_wrapper_call(node.name, alias_targets=alias_targets)
        return node
    if isinstance(node, ExprLiteral):
        return node
    if isinstance(node, UnaryOp):
        return UnaryOp(node.op, _rewrite_collection_alias_entities(node.operand, alias_targets=alias_targets))
    if isinstance(node, BinaryOp):
        if (
            node.op == "in"
            and isinstance(node.left, Identifier)
            and isinstance(alias_targets.get(node.left.name), ASTNode)
            and isinstance(node.right, ListLiteral)
        ):
            identity_items: list[ExprNode] = []
            for item in node.right.items:
                if isinstance(item, Identifier) and isinstance(alias_targets.get(item.name), ASTNode):
                    identity_items.append(FunctionCall("__node_identity__", (Identifier(item.name),)))
                elif isinstance(item, ExprLiteral) and item.value is None:
                    identity_items.append(item)
                elif _node_value_expr(item):
                    identity_items.append(FunctionCall(
                        "__cypher_node_identity_value__",
                        (_rewrite_node_value_expr(item),),
                    ))
                else:
                    identity_items.append(FunctionCall("__cypher_non_entity_null__", (item,)))
            return FunctionCall(
                "__cypher_node_in__",
                (
                    FunctionCall("__node_identity__", (Identifier(node.left.name),)),
                    ListLiteral(tuple(identity_items)),
                ),
            )
        return BinaryOp(
            node.op,
            _rewrite_collection_alias_entities(node.left, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.right, alias_targets=alias_targets),
        )
    if isinstance(node, IsNullOp):
        return IsNullOp(_rewrite_collection_alias_entities(node.value, alias_targets=alias_targets), negated=node.negated)
    if isinstance(node, FunctionCall):
        return FunctionCall(
            node.name,
            tuple(_rewrite_collection_alias_entities(arg, alias_targets=alias_targets) for arg in node.args),
            distinct=node.distinct,
        )
    if isinstance(node, Wildcard):
        return node
    if isinstance(node, CaseWhen):
        return CaseWhen(
            _rewrite_collection_alias_entities(node.condition, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.when_true, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.when_false, alias_targets=alias_targets),
        )
    if isinstance(node, QuantifierExpr):
        return QuantifierExpr(
            node.fn,
            node.var,
            _rewrite_collection_alias_entities(node.source, alias_targets=alias_targets),
            _rewrite_collection_alias_entities(node.predicate, alias_targets=alias_targets),
        )
    if isinstance(node, ListComprehension):
        return ListComprehension(
            node.var,
            _rewrite_collection_alias_entities(node.source, alias_targets=alias_targets),
            predicate=None
            if node.predicate is None
            else _rewrite_collection_alias_entities(node.predicate, alias_targets=alias_targets),
            projection=None
            if node.projection is None
            else _rewrite_collection_alias_entities(node.projection, alias_targets=alias_targets),
        )
    if isinstance(node, ListLiteral):
        return ListLiteral(
            tuple(
                _rewrite_collection_alias_entities(item, alias_targets=alias_targets, inside_collection=True)
                for item in node.items
            )
        )
    if isinstance(node, MapLiteral):
        return MapLiteral(
            tuple(
                (
                    key,
                    _rewrite_collection_alias_entities(value, alias_targets=alias_targets, inside_collection=True),
                )
                for key, value in node.items
            )
        )
    return _rebuild_expr_node(
        node,
        rewrite=lambda child: _rewrite_collection_alias_entities(child, alias_targets=alias_targets),
        error_context="collection alias rewrite",
    )
