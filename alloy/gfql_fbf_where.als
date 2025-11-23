module gfql_fbf_where

// Simplified Alloy model for GFQL linear patterns under set semantics.
// Scopes (checks): up to 8 Nodes, 8 Edges, 4 Steps, 4 Values.

abstract sig Value {}
sig Val extends Value {}

// Total order over values for inequalities
sig Ord { lt: Value -> Value } {
  lt in Value -> Value
  all v: Value | v not in v.^(lt) // irreflexive, acyclic
}
fact TotalOrder {
  one Ord
  all v1, v2: Value | v1 != v2 implies (v1 -> v2 in Ord.lt or v2 -> v1 in Ord.lt)
}

sig Node { vals: set Value }
sig Edge { src: one Node, dst: one Node, vals: set Value }

abstract sig Step {}
sig NodeStep extends Step { aliasN: lone Alias, nFilter: set Value }
sig EdgeStep extends Step { aliasE: lone Alias, eFilter: set Value }
sig Alias {}

// WHERE refs point to node aliases and a required value
sig WhereRef { a: one Alias, v: one Value }
sig WhereClause { lhs: one WhereRef, rhs: one WhereRef, op: one Op }
abstract sig Op {}
one sig Eq, Neq, Lt, Lte, Gt, Gte extends Op {}

sig Chain { steps: seq Step, where: set WhereClause }

// Binding = sequence of nodes/edges aligned with steps
pred BindingFor(c: Chain, bn: seq Node, be: seq Edge) {
  // shape
  #bn = (#(c.steps) + 1) / 2
  #be = #(c.steps) / 2
  all i: c.steps.inds |
    (i % 2 = 0 => c.steps[i] in NodeStep and bn[i/2] in Node and nFilterOK[c.steps[i], bn[i/2]]) and
    (i % 2 = 1 => c.steps[i] in EdgeStep and be[i/2] in Edge and eFilterOK[c.steps[i], be[i/2]] and be[i/2].src = bn[(i-1)/2] and be[i/2].dst = bn[(i+1)/2])
  // where clauses satisfied
  all w: c.where | whereHolds[w, c, bn]
}

pred nFilterOK[s: NodeStep, n: Node] { no s.nFilter or s.nFilter in n.vals }
pred eFilterOK[s: EdgeStep, e: Edge] { no s.eFilter or s.eFilter in e.vals }

// resolve alias to node in binding
fun aliasNode(c: Chain, bn: seq Node, a: Alias): set Node {
  { n: Node | some i: c.steps.inds | i%2=0 and c.steps[i].aliasN = a and n = bn[i/2] }
}

pred whereHolds(w: WhereClause, c: Chain, bn: seq Node) {
  some ln: aliasNode(c, bn, w.lhs.a)
  some rn: aliasNode(c, bn, w.rhs.a)
  let lvals = aliasNode(c, bn, w.lhs.a).vals, rvals = aliasNode(c, bn, w.rhs.a).vals |
    (w.op = Eq => some v: lvals & rvals | v = w.lhs.v and v = w.rhs.v)
    or (w.op = Neq => no (lvals & rvals))
    or (w.op = Lt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and lv -> rv in Ord.lt)
    or (w.op = Lte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or lv -> rv in Ord.lt))
    or (w.op = Gt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and rv -> lv in Ord.lt)
    or (w.op = Gte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or rv -> lv in Ord.lt))
}

// Spec: collect nodes/edges that participate in SOME satisfying binding
fun SpecNodes(c: Chain): set Node { { n: Node | some bn: seq Node, be: seq Edge | BindingFor(c,bn,be) and n in bn.elems[] } }
fun SpecEdges(c: Chain): set Edge { { e: Edge | some bn: seq Node, be: seq Edge | BindingFor(c,bn,be) and e in be.elems[] } }

// Algo: forward/backward/forward under set semantics
pred Algo(c: Chain, outN: set Node, outE: set Edge) {
  // forward filter
  let fn = { n: Node | some i: c.steps.inds | i%2=0 and nFilterOK[c.steps[i], n] },
      fe = { e: Edge | some i: c.steps.inds | i%2=1 and eFilterOK[c.steps[i], e] } |
    // backward prune: edge endpoints must be allowed nodes
    outE = { e: fe | e.src in fn and e.dst in fn }
    outN = fn
    // where enforcement: nodes in outN must admit some binding satisfying where
    all n: outN | some bn: seq Node, be: seq Edge | BindingFor(c,bn,be) and n in bn.elems[]
}

assert SpecNoWhereEqAlgoNoWhere {
  all c: Chain | no c.where implies (SpecNodes[c] = AlgoNodes[c] and SpecEdges[c] = AlgoEdges[c])
}

fun AlgoNodes(c: Chain): set Node { { n: Node | some outN: set Node, outE: set Edge | Algo(c, outN, outE) and n in outN } }
fun AlgoEdges(c: Chain): set Edge { { e: Edge | some outN: set Node, outE: set Edge | Algo(c, outN, outE) and e in outE } }

assert SpecWhereEqAlgoLowered {
  all c: Chain | SpecNodes[c] = AlgoNodes[c] and SpecEdges[c] = AlgoEdges[c]
}

check SpecNoWhereEqAlgoNoWhere for 8 but 4 Step, 4 Value
check SpecWhereEqAlgoLowered for 8 but 4 Step, 4 Value
