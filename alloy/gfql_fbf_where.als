module gfql_fbf_where
open util/ordering[Value] as ord
open util/integer

// Alloy model to compare Python hop/chain (path semantics) vs executor (set semantics with F/B/F lowerings).
// Path semantics: bindings are sequences aligned to seqSteps with WHERE applied per binding.
// Set semantics: forward/backward/forward collects per-alias node/edge sets, then checks WHERE via summaries.
// Scopes (checks): up to 8 Nodes, 8 Edges, 4 Steps, 4 Values. Nulls/hashing omitted; bounded values only.
// Mapping to Python hop/chain:
// - seqSteps alternates NodeStep/EdgeStep like graphistry.compute.GSQL chain builder.
// - aliasN/aliasE mirror user aliases; WHERE binds to NodeStep aliases only.
// - nFilter/eFilter correspond to per-step filter columns; WHERE models cross-step predicates.
// - Spec uses path bindings (sequence) like hop composition; Algo uses set semantics like executor.
// - Null/NaN not modeled; hashing treated as prefilter and omitted here.
// - Hop ranges/output slicing (min/max/output bounds) are not explicitly modeled; approximate via unrolled fixed-length chains.

abstract sig Value {}
sig Val extends Value {}

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

// Chain mirrors Python chain construction: alternating NodeStep/EdgeStep with alias + filters.
sig Chain { seqSteps: seq Step, wheres: set WhereClause }
sig Binding {
  owner: one Chain,
  bn: Int -> lone Node,
  be: Int -> lone Edge
}

// Well-formed chains: non-empty, odd length (N,E,N,...), typed positions
fact WellFormedChains {
  all c: Chain |
    #seq/inds[c.seqSteps] > 0 and rem[#seq/inds[c.seqSteps], 2] = 1 and
    all i: seq/inds[c.seqSteps] |
      (rem[i, 2] = 0 => c.seqSteps[i] in NodeStep) and
      (rem[i, 2] = 1 => c.seqSteps[i] in EdgeStep)
}

// Ensure we analyze non-empty chains; allow multiple chains/bindings within scope.
fact NonEmptyChains { some Chain }
fact OneBindingPerChain { all c: Chain | some b: Binding | b.owner = c }

// All bindings must satisfy their owner's shape and WHERE clauses
fact BindingsRespectOwners {
  all c: Chain | some b: Binding | BindingFor[c, b]
}

// Project binding sequences into sets (path semantics)
fun bindNodes[b: Binding]: set Node { b.bn[Int] }
fun bindEdges[b: Binding]: set Edge { b.be[Int] }

// Binding = sequence of nodes/edges aligned with steps (path-based semantics)
pred BindingFor[c: Chain, b: Binding] {
  b.owner = c and
  let bnSeq = b.bn, beSeq = b.be |
  isSeq[bnSeq] and isSeq[beSeq] and
  // shape
  #bnSeq = div[#(c.seqSteps) + 1, 2] and
  #beSeq = div[#(c.seqSteps), 2] and
  all i: seq/inds[c.seqSteps] |
    (rem[i, 2] = 0 => c.seqSteps[i] in NodeStep and nFilterOK[c.seqSteps[i], bnSeq[div[i, 2]]]) and
    (rem[i, 2] = 1 => c.seqSteps[i] in EdgeStep and eFilterOK[c.seqSteps[i], beSeq[div[i, 2]]] and beSeq[div[i, 2]].src = bnSeq[div[i - 1, 2]] and beSeq[div[i, 2]].dst = bnSeq[div[i + 1, 2]]) and
  // where clauses satisfied
  all w: c.wheres | whereHolds[w, c, bnSeq]
}

// Binding shape without WHERE (used by set-based algo path connectivity)
pred BindingShape[c: Chain, b: Binding] {
  b.owner = c and
  let bnSeq = b.bn, beSeq = b.be |
  isSeq[bnSeq] and isSeq[beSeq] and
  #bnSeq = div[#(c.seqSteps) + 1, 2] and
  #beSeq = div[#(c.seqSteps), 2] and
  all i: seq/inds[c.seqSteps] |
    (rem[i, 2] = 0 => c.seqSteps[i] in NodeStep and nFilterOK[c.seqSteps[i], bnSeq[div[i, 2]]]) and
    (rem[i, 2] = 1 => c.seqSteps[i] in EdgeStep and eFilterOK[c.seqSteps[i], beSeq[div[i, 2]]] and beSeq[div[i, 2]].src = bnSeq[div[i - 1, 2]] and beSeq[div[i, 2]].dst = bnSeq[div[i + 1, 2]])
}

pred nFilterOK[s: NodeStep, n: Node] { no s.nFilter or s.nFilter in n.vals }
pred eFilterOK[s: EdgeStep, e: Edge] { no s.eFilter or s.eFilter in e.vals }

// resolve alias to node in binding
fun aliasNode[c: Chain, bn: Int -> lone Node, a: Alias]: set Node {
  { n: Node | some i: seq/inds[c.seqSteps] | rem[i, 2] = 0 and c.seqSteps[i].aliasN = a and n = bn[div[i, 2]] }
}

pred whereHolds[w: WhereClause, c: Chain, bn: Int -> lone Node] {
  let ln = aliasNode[c, bn, w.lhs.a], rn = aliasNode[c, bn, w.rhs.a] |
    some ln and some rn and
    let lvals = ln.vals, rvals = rn.vals |
      (w.op = Eq => some vv: lvals & rvals | vv = w.lhs.v and vv = w.rhs.v)
      or (w.op = Neq => no (lvals & rvals))
      or (w.op = Lt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and lv in ord/prevs[rv])
      or (w.op = Lte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or lv in ord/prevs[rv]))
      or (w.op = Gt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and rv in ord/prevs[lv])
      or (w.op = Gte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or rv in ord/prevs[lv]))
}

// Spec (path semantics): nodes/edges that appear in some satisfying binding
pred SpecNode[c: Chain, n: Node] { some b: Binding | BindingFor[c, b] and n in bindNodes[b] }
pred SpecEdge[c: Chain, e: Edge] { some b: Binding | BindingFor[c, b] and e in bindEdges[b] }

pred SpecAlgoEq[c: Chain] {
  all n: Node | SpecNode[c, n] <=> n in AlgoOutN[c]
  all e: Edge | SpecEdge[c, e] <=> e in AlgoOutE[c]
}

// Algo: forward/backward/forward under set semantics with simple lowerings:
// - Inequalities lowered to min/max summaries per alias/value
// - Equalities lowered to exact value sets per alias
fun AlgoOutN[c: Chain]: set Node { { n: Node | some b: Binding | BindingShape[c, b] and n in bindNodes[b] } }
fun AlgoOutE[c: Chain]: set Edge { { e: Edge | some b: Binding | BindingShape[c, b] and e in bindEdges[b] } }

pred Algo[c: Chain] {
  let outN = AlgoOutN[c], outE = AlgoOutE[c] |
    all w: c.wheres | lowerWhere[w, c, outN, outE]
}

pred lowerWhere[w: WhereClause, c: Chain, outN: set Node, outE: set Edge] {
  // compute per-alias value sets
  let ln = aliasNodes[outN, c, w.lhs.a], rn = aliasNodes[outN, c, w.rhs.a] |
    some ln and some rn and
    let lvals = ln.vals, rvals = rn.vals |
      (w.op = Eq => some vv: lvals & rvals | vv = w.lhs.v and vv = w.rhs.v)
      or (w.op = Neq => no (lvals & rvals))
      or (w.op = Lt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and ord/lt[lv, rv])
      or (w.op = Lte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or ord/lt[lv, rv]))
      or (w.op = Gt => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and ord/lt[rv, lv])
      or (w.op = Gte => some lv: lvals | some rv: rvals | lv = w.lhs.v and rv = w.rhs.v and (lv = rv or ord/lt[rv, lv]))
}

fun aliasNodes[ns: set Node, c: Chain, a: Alias]: set Node {
  { n: ns | some i: seq/inds[c.seqSteps] | rem[i, 2] = 0 and c.seqSteps[i].aliasN = a }
}

assert SpecNoWhereEqAlgoNoWhere {
  all c: Chain |
    Algo[c] and
    (no c.wheres implies SpecAlgoEq[c])
}

assert SpecWhereEqAlgoLowered {
  all c: Chain |
    Algo[c] and SpecAlgoEq[c]
}

// Derived assertions for alternate scopes (multi-chain)
assert SpecNoWhereEqAlgoNoWhereMultiChain {
  all c: Chain |
    Algo[c] and (no c.wheres implies SpecAlgoEq[c])
}

assert SpecWhereEqAlgoLoweredMultiChain {
  all c: Chain |
    Algo[c] and SpecAlgoEq[c]
}

assert SpecNoWhereEqAlgoNoWhereMultiChainFull {
  all c: Chain |
    Algo[c] and (no c.wheres implies SpecAlgoEq[c])
}

assert SpecWhereEqAlgoLoweredMultiChainFull {
  all c: Chain |
    Algo[c] and SpecAlgoEq[c]
}

// Convenience aliases for alternate scopes
assert SpecNoWhereEqAlgoNoWhereSmall {
  all c: Chain |
    Algo[c] and
    (no c.wheres implies SpecAlgoEq[c])
}
assert SpecWhereEqAlgoLoweredSmall {
  all c: Chain | Algo[c] and SpecAlgoEq[c]
}

// Scenario coverage: topologies and query shapes that tend to surface path/set differences.
pred FanOutGraph { some n: Node | some disj e1, e2: Edge | e1.src = n and e2.src = n and e1.dst != e2.dst }
pred FanInGraph { some n: Node | some disj e1, e2: Edge | e1.dst = n and e2.dst = n and e1.src != e2.src }
pred CycleGraph { some e: Edge | e.src = e.dst or some disj e1, e2: Edge | e1.src = e2.dst and e2.src = e1.dst }
pred ParallelEdgesGraph { some disj e1, e2: Edge | e1.src = e2.src and e1.dst = e2.dst }
pred DisconnectedGraph { some n: Node | no e: Edge | e.src = n or e.dst = n }

pred ChainAliasReuse[c: Chain] {
  #seq/inds[c.seqSteps] >= 3 and
  c.seqSteps[0] in NodeStep and c.seqSteps[2] in NodeStep and
  some al: Alias | c.seqSteps[0].aliasN = al and c.seqSteps[2].aliasN = al and
  some w: c.wheres | (w.lhs.a = al or w.rhs.a = al)
}

pred ChainMixedWhere[c: Chain] {
  some wEq: c.wheres | wEq.op = Eq and
  some wCmp: c.wheres | wCmp.op != Eq
}

pred ChainFilterMix[c: Chain] {
  some ns: NodeStep | ns in c.seqSteps.elems and some ns.nFilter and
  some es: EdgeStep | es in c.seqSteps.elems and some es.eFilter
}

pred FanCounterexample {
  FanOutGraph and FanInGraph and
  some c: Chain | Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredFan { not FanCounterexample }

pred CycleCounterexample {
  CycleGraph and
  some c: Chain | Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredCycle { not CycleCounterexample }

pred ParallelCounterexample {
  ParallelEdgesGraph and
  some c: Chain | Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredParallel { not ParallelCounterexample }

pred DisconnectedCounterexample {
  DisconnectedGraph and
  some c: Chain | Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredDisconnected { not DisconnectedCounterexample }

pred AliasCounterexample {
  some c: Chain | ChainAliasReuse[c] and Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredAliasWhere { not AliasCounterexample }

pred MixedWhereCounterexample {
  some c: Chain | ChainMixedWhere[c] and Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredMixedWhere { not MixedWhereCounterexample }

pred FilterMixCounterexample {
  some c: Chain | ChainFilterMix[c] and Algo[c] and not SpecAlgoEq[c]
}
assert SpecWhereEqAlgoLoweredFilterMix { not FilterMixCounterexample }

// Contradictory WHERE: clauses that cannot be simultaneously satisfied
// E.g., a.v < c.v AND a.v > c.v, or a.v == c.v AND a.v != c.v
pred ContradictoryWhere[c: Chain] {
  some disj w1, w2: c.wheres |
    w1.lhs.a = w2.lhs.a and w1.rhs.a = w2.rhs.a and
    w1.lhs.v = w2.lhs.v and w1.rhs.v = w2.rhs.v and
    ((w1.op = Lt and w2.op = Gt) or (w1.op = Lt and w2.op = Gte) or
     (w1.op = Gt and w2.op = Lt) or (w1.op = Gt and w2.op = Lte) or
     (w1.op = Eq and w2.op = Neq) or (w1.op = Neq and w2.op = Eq))
}

// When WHERE is contradictory, no paths can satisfy both, so output should be empty
pred ContradictoryCounterexample {
  some c: Chain | ContradictoryWhere[c] and (some AlgoOutN[c] or some AlgoOutE[c])
}
assert ContradictoryWhereEmpty { not ContradictoryCounterexample }

check SpecNoWhereEqAlgoNoWhere for 8 but 4 Step, 4 Value, 4 Binding, 1 Chain
check SpecWhereEqAlgoLowered for 8 but 4 Step, 4 Value, 4 Binding, 1 Chain

// Debug-friendly smaller scopes
check SpecNoWhereEqAlgoNoWhereSmall for 4 but 3 Step, 3 Value, 3 Binding, 4 Node, 4 Edge, 1 Chain
check SpecWhereEqAlgoLoweredSmall for 4 but 3 Step, 3 Value, 3 Binding, 4 Node, 4 Edge, 1 Chain

// Multi-chain sanity (small scope to keep solve time low)
check SpecNoWhereEqAlgoNoWhereMultiChain for 4 but 3 Step, 3 Value, 2 Binding, 4 Node, 4 Edge, 2 Chain
check SpecWhereEqAlgoLoweredMultiChain for 4 but 3 Step, 3 Value, 2 Binding, 4 Node, 4 Edge, 2 Chain

// Multi-chain fuller scope (optional; gated via script env to keep runtime predictable)
check SpecNoWhereEqAlgoNoWhereMultiChainFull for 8 but 4 Step, 4 Value, 4 Binding, 2 Chain
check SpecWhereEqAlgoLoweredMultiChainFull for 8 but 4 Step, 4 Value, 4 Binding, 2 Chain

// Scenario-specific coverage (smaller scopes to keep solving fast)
check SpecWhereEqAlgoLoweredFan for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredCycle for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredParallel for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredDisconnected for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredAliasWhere for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredMixedWhere for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check SpecWhereEqAlgoLoweredFilterMix for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
check ContradictoryWhereEmpty for 6 but 3 Step, 3 Value, 3 Binding, 6 Node, 6 Edge, 1 Chain
