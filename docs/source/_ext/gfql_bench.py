"""Sphinx extension: publish GFQL benchmark numbers from the source-of-truth.

Docs never restate a measured number. They *reference* it:

.. code-block:: rst

   * - 1-hop from 10K seeds
     - :bench:`orkut.hop1_10k.pandas`
     - :bench:`orkut.hop1_10k.polars`

and every page that references a cell must also carry that cell's provenance and
disclosures:

.. code-block:: rst

   .. bench-provenance:: orkut-4engine-20260703
   .. bench-disclosures::

Failure modes are build failures, never silent text:

- a key the source-of-truth does not contain       -> build fails
- a run older than ``policy.max_age_days``         -> build fails
- a non-board-quotable cell used as a bare number  -> build fails
- a page that drops a cell's provenance/disclosure -> build fails

``bin/check_bench_numbers.py`` runs the same data checks without Sphinx, plus a
commit-drift check against the current tree and a hand-typed-literal guard over
the managed pages, so CI can gate on staleness a pure docs build cannot see.
"""

from __future__ import annotations

import datetime
import os
from typing import Dict, List, Optional, Tuple, Union

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.errors import SphinxError
from sphinx.util import logging as sphinx_logging

from gfql_bench_data import (
    BenchCell,
    BenchData,
    BenchDataError,
    BenchRun,
    load_bench_data,
)

logger = sphinx_logging.getLogger(__name__)

RoleResult = Tuple[List[nodes.Node], List[nodes.system_message]]


class BenchNumberError(SphinxError):
    category = 'GFQL benchmark number check failed'


class _State:
    """Loaded once per build: the data, the clock, and the problems found."""

    def __init__(self, data: BenchData, today: datetime.date) -> None:
        self.data = data
        self.today = today
        self.problems: List[str] = []
        #: docname -> benchmark keys referenced by that page, in source order
        self.refs: Dict[str, List[str]] = {}
        #: docname -> run ids whose provenance the page renders
        self.provenance: Dict[str, List[str]] = {}
        #: docnames carrying a ``bench-disclosures`` block
        self.disclosed: List[str] = []

    def fail(self, message: str) -> None:
        self.problems.append(message)
        logger.warning('[gfql-bench] %s', message)

    def forget(self, docname: str) -> None:
        self.refs.pop(docname, None)
        self.provenance.pop(docname, None)
        if docname in self.disclosed:
            self.disclosed.remove(docname)


_STATE: Optional[_State] = None


def _state() -> _State:
    if _STATE is None:
        raise BenchNumberError('gfql_bench extension used before builder-inited')
    return _STATE


def _bench_role_impl(key: str, rawtext: str, lineno: int, inliner: Inliner, diagnostic: bool) -> RoleResult:
    state = _state()
    env: BuildEnvironment = inliner.document.settings.env
    docname = env.docname
    state.refs.setdefault(docname, []).append(key)

    try:
        cell: Optional[BenchCell] = state.data.cell(key)
    except BenchDataError as exc:
        state.fail('{}:{}: {}'.format(docname, lineno, exc))
        cell = None
    if cell is None:
        return [nodes.strong(rawtext, '[MISSING BENCHMARK NUMBER: {}]'.format(key))], []

    if diagnostic and cell.board_quotable:
        state.fail(
            '{}:{}: {!r} is board-quotable; use :bench: rather than :bench-diag:'.format(docname, lineno, key)
        )
    if not diagnostic and not cell.board_quotable:
        state.fail(
            '{}:{}: {!r} is NOT board-quotable (status={}, comparison_allowed={}); it may only be '
            'published via :bench-diag:, which labels it diagnostic-only'.format(
                docname, lineno, key, cell.status, cell.comparison_allowed)
        )

    text = cell.render()
    rendered: nodes.Node = nodes.strong(rawtext, text) if cell.board_quotable else nodes.Text(text)
    result: List[nodes.Node] = [rendered]
    if diagnostic:
        result.append(nodes.Text(' (diagnostic only — not a board result)'))
    return result, []


def bench_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Optional[Dict[str, str]] = None,
    content: Optional[List[str]] = None,
) -> RoleResult:
    return _bench_role_impl(text.strip(), rawtext, lineno, inliner, diagnostic=False)


def bench_diag_role(
    name: str,
    rawtext: str,
    text: str,
    lineno: int,
    inliner: Inliner,
    options: Optional[Dict[str, str]] = None,
    content: Optional[List[str]] = None,
) -> RoleResult:
    return _bench_role_impl(text.strip(), rawtext, lineno, inliner, diagnostic=True)


def _field(label: str, value: str) -> nodes.definition_list_item:
    item = nodes.definition_list_item()
    item += nodes.term('', label)
    definition = nodes.definition()
    definition += nodes.paragraph('', value)
    item += definition
    return item


def _provenance_block(run: BenchRun) -> nodes.Element:
    lock = 'perf lock held for the whole session' if run.perf_lock_held else 'PERF LOCK NOT HELD'
    quiet = 'quiet host' if run.quiet_host else 'host contention not established'
    listing = nodes.definition_list()
    listing += _field('Measured', '{} on {} ({}, {})'.format(
        run.measured_at.isoformat(), run.host, quiet, lock))
    listing += _field('pygraphistry', run.pygraphistry_commit)
    listing += _field('Benchmark harness', 'graphistry/pyg-bench {}'.format(run.pyg_bench_commit))
    listing += _field('Runtime', run.runtime)
    listing += _field('Dataset', run.dataset)
    listing += _field('Protocol', run.reps)
    listing += _field('Result validation', run.row_validation)
    if run.competitor is not None:
        listing += _field('Competitor', '{} {}'.format(
            run.competitor, run.competitor_version or 'version unrecorded'))
    listing += _field('Raw artifacts', 'graphistry/pyg-bench {}'.format(run.artifact))

    admonition = nodes.admonition()
    admonition['classes'] = ['note', 'gfql-bench-provenance']
    admonition += nodes.title('', 'Provenance: {}'.format(run.run_id))
    admonition += listing
    return admonition


class BenchProvenanceDirective(Directive):
    """``.. bench-provenance:: <run_id>`` — render a run's full provenance."""

    has_content = False
    required_arguments = 1
    optional_arguments = 0

    def run(self) -> List[nodes.Node]:
        state = _state()
        env: BuildEnvironment = self.state.document.settings.env
        docname = env.docname
        run_id = self.arguments[0].strip()
        state.provenance.setdefault(docname, []).append(run_id)
        run = state.data.runs.get(run_id)
        if run is None:
            state.fail('{}: unknown benchmark run {!r}'.format(docname, run_id))
            return [nodes.strong('', '[UNKNOWN BENCHMARK RUN: {}]'.format(run_id))]
        return [_provenance_block(run)]


class BenchDisclosuresDirective(Directive):
    """``.. bench-disclosures::`` — render every disclosure this page owes.

    The body is generated from the source-of-truth, so a caveat cannot be
    paraphrased away or silently dropped when a number is refreshed.
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0

    def run(self) -> List[nodes.Node]:
        state = _state()
        env: BuildEnvironment = self.state.document.settings.env
        docname = env.docname
        if docname not in state.disclosed:
            state.disclosed.append(docname)
        placeholder = nodes.container()
        placeholder['classes'] = ['gfql-bench-disclosures-placeholder']
        return [placeholder]


def _disclosure_lines(state: _State, docname: str) -> List[str]:
    lines: List[str] = []
    for key in state.refs.get(docname, []):
        cell = state.data.cells.get(key)
        if cell is None:
            continue
        for disclosure in cell.disclosures:
            line = '{}: {}'.format(cell.workload, disclosure)
            if line not in lines:
                lines.append(line)
    return lines


def _fill_disclosures(app: Sphinx, doctree: nodes.document, docname: str) -> None:
    state = _state()
    for container in list(doctree.findall(nodes.container)):
        if 'gfql-bench-disclosures-placeholder' not in container['classes']:
            continue
        admonition = nodes.admonition()
        admonition['classes'] = ['important', 'gfql-bench-disclosures']
        admonition += nodes.title('', 'Disclosures that travel with these numbers')
        lines = _disclosure_lines(state, docname)
        if lines:
            bullet = nodes.bullet_list()
            for line in lines:
                item = nodes.list_item()
                item += nodes.paragraph('', line)
                bullet += item
            admonition += bullet
        else:
            admonition += nodes.paragraph(
                '', 'Every number on this page is a clean, row-validated result with no caveat.')
        container.replace_self(admonition)


def _check_consistency(app: Sphinx, env: BuildEnvironment) -> None:
    state = _state()
    for docname in sorted(state.refs):
        keys = state.refs[docname]
        if not keys:
            continue
        needed_runs: List[str] = []
        needs_disclosure = False
        for key in keys:
            cell = state.data.cells.get(key)
            if cell is None:
                continue
            if cell.run_id not in needed_runs:
                needed_runs.append(cell.run_id)
            if cell.disclosures:
                needs_disclosure = True
        have_runs = state.provenance.get(docname, [])
        for run_id in needed_runs:
            if run_id not in have_runs:
                state.fail('{}: publishes numbers from run {!r} without a '
                           '`.. bench-provenance:: {}` block'.format(docname, run_id, run_id))
            run = state.data.runs[run_id]
            age = run.age_days(state.today)
            if age > state.data.policy.max_age_days:
                state.fail('{}: run {!r} was measured {} days ago (policy max {}); '
                           're-measure it or remove the claim'.format(
                               docname, run_id, age, state.data.policy.max_age_days))
        if needs_disclosure and docname not in state.disclosed:
            state.fail('{}: publishes a number that carries a disclosure but has no '
                       '`.. bench-disclosures::` block — a bare ratio without its asterisk '
                       'launders the caveat'.format(docname))

    if state.problems:
        raise BenchNumberError(
            'benchmark numbers failed validation ({} problem(s)):\n  - {}'.format(
                len(state.problems), '\n  - '.join(state.problems)))


def _builder_inited(app: Sphinx) -> None:
    global _STATE
    configured = str(app.config.gfql_bench_data_path)
    path: Optional[str] = configured or None
    if path is not None and not os.path.isabs(path):
        path = os.path.join(str(app.srcdir), path)
    data = load_bench_data(path)
    today_setting = str(app.config.gfql_bench_today)
    if today_setting:
        year, month, day = today_setting.split('-')
        today = datetime.date(int(year), int(month), int(day))
    else:
        today = datetime.date.today()
    _STATE = _State(data, today)

    # Freshness is checked for EVERY run in the file, not just the ones this build
    # happens to re-read: an incremental build can serve a cached page whose numbers
    # have aged out, and that page must not be publishable either.
    stale = data.stale_runs(today)
    if stale:
        raise BenchNumberError(
            'benchmark run(s) past policy.max_age_days={}:\n  - {}'.format(
                data.policy.max_age_days,
                '\n  - '.join(
                    '{} measured {} ({} days ago)'.format(
                        run.run_id, run.measured_at.isoformat(), age)
                    for run, age in stale)))


def _purge_doc(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    if _STATE is not None:
        _STATE.forget(docname)


def setup(app: Sphinx) -> Dict[str, Union[str, bool]]:
    app.add_config_value('gfql_bench_data_path', '', 'env')
    app.add_config_value('gfql_bench_today', '', 'env')
    app.add_role('bench', bench_role)
    app.add_role('bench-diag', bench_diag_role)
    app.add_directive('bench-provenance', BenchProvenanceDirective)
    app.add_directive('bench-disclosures', BenchDisclosuresDirective)
    app.connect('builder-inited', _builder_inited)
    app.connect('env-purge-doc', _purge_doc)
    app.connect('doctree-resolved', _fill_disclosures)
    app.connect('env-check-consistency', _check_consistency)
    # Bookkeeping lives in module state, not the pickled env: single-process reads only.
    return {'version': '1', 'parallel_read_safe': False, 'parallel_write_safe': True}
