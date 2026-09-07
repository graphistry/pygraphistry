---
name: docs-review
description: Editorial policy for user-facing documentation (docs/source/**). Use when writing or reviewing prose in docs pages, docstrings that render in docs, or release notes. Plain, direct, forward-readable text; no internal process talk; no AI-writing tells.
---

# Docs editorial policy

Audience: a pandas, Polars, or Cypher user who has never seen this codebase. They read once,
forwards, and stop at the first sentence that does not pay off.

## Rules (ASD-STE100 in spirit)

- One idea per sentence, about 20 words, active voice, present tense.
- Verbs over nouns: "the index reads only those neighborhoods", not "the index enables
  neighborhood-scoped reads".
- Say the benefit to the reader, then the mechanism, then the evidence link. Never the
  mechanism alone.
- Define or replace jargon on first use. Prefer the plain phrase (right column):

  | avoid | write |
  |---|---|
  | shape (of a query) | kind of query, query pattern, workload |
  | seeded / seed set / seeded lookup | a query that starts from a few known nodes |
  | lane, route, fast path, hot path | (omit; say what runs faster and when) |
  | parity, oracle | the same result on every engine |
  | decline, typed decline | raises an error before the query runs |
  | materialize (intermediate) | build an intermediate result |
  | frontier | the nodes reached at this hop |
  | point lookup | a query for one node by id |
  | receipt, artifact, committed artifact | (omit; link the provenance section once) |
  | attenuation, engagement | (omit) |
  | release gate, CI, sweep, lever, step N | (omit; internal process) |

- No parentheticals inside sentences. Split them into sentences or delete them.
- No unnecessary contrast flourishes: "not X but Y", "no GPU, same results", "X — and Y".
  State Y.
- No mannerist titles ("The one-line speedup"). Title = what the reader gets to do
  ("Switch engines with one keyword").
- No footnote-style citations in body text ("[F1] Polars leads"). Put one fact inline with
  its link, or move the block to a provenance section.
- No competitor ammunition. Say where GFQL is good and link the full board with the
  losses shown. Do not editorialize a competitor's strengths.
- No internal process in user docs: release gates, CI lanes, what we refuse to publish,
  how numbers were audited. One provenance line with a link is enough.
- Every number comes from a vendored benchmark cell (`:bench-*:` roles). Never type a
  measured number as a literal.
- Keep examples and tables; cut prose. A rewrite that shortens prose by a third with no
  loss of meaning is the normal outcome of a review.

## AI-writing tells to remove

- Triads for rhythm ("fast, safe, and simple"), stacked em-dash asides, "not only … but".
- Sentences that restate the previous one with more adjectives.
- "Deliberately", "carefully", "seamlessly", "robust", "powerful", "leverage".
- Rhetorical questions and "so what does this mean?" transitions.
- Claims about the writing itself ("this page is honest about losses").

## Review procedure

1. Read the page forwards once as the target reader. Mark every stop.
2. For each section ask: what is the one message that is powerful to deliver? If none,
   delete the section. If one, rewrite the section to deliver it in the fewest sentences.
3. Apply the jargon table, then the tells list.
4. Check every link and every `:bench-*:` role still resolves; run
   `python -m pytest docs/test_bench_numbers.py`.
5. Post the ReadTheDocs preview link per changed page with the sections to check.
