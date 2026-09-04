"""Sphinx extension: pretty-print the benchmark numbers pyg-bench publishes.

Measurement, provenance and publishability all live in `graphistry/pyg-bench`, which
owns the runs. This repository renders them, and nothing else.

Docs never restate a measured number, they reference one::

    * - Twitter, GPU
      - :bench:`pagerank.twitter.gfql_gpu`

and the page that references a cell must also render that cell's provenance and its
disclosures::

    .. bench-provenance:: filter-pagerank-20260728
    .. bench-disclosures::

Every rule is in ``gfql_bench_data``, which is stdlib-only so the ordinary test lane
can run it without Sphinx installed. This module is the docutils half: roles,
directives, and turning a recorded problem into a failed build.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional, Tuple

from docutils import nodes
from docutils.parsers.rst import Directive, directives
from docutils.parsers.rst.states import Inliner
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.errors import SphinxError
from sphinx.util import logging as sphinx_logging

from gfql_bench_data import (
    BenchDataError,
    JSONObject,
    State,
    audit_pages,
    check_reference,
    format_cell,
    load_state,
)

logger = sphinx_logging.getLogger(__name__)

RoleResult = Tuple[List[nodes.Node], List[nodes.system_message]]

_STATE = None  # type: Optional[State]


class BenchNumberError(SphinxError):
    category = 'GFQL benchmark number check failed'


def _state() -> State:
    if _STATE is None:
        raise BenchNumberError('gfql_bench used before builder-inited')
    return _STATE


def _bench_role(diagnostic: bool):
    def role(name: str, rawtext: str, key: str, lineno: int, inliner: Inliner,
             options=None, content=None) -> RoleResult:
        state = _state()
        docname = inliner.document.settings.env.docname
        before = len(state.problems)
        cell = check_reference(state, key, docname, lineno, diagnostic)
        for message in state.problems[before:]:
            logger.warning('[gfql-bench] %s', message)
        if cell is None:
            return [nodes.strong(rawtext, '[MISSING BENCHMARK NUMBER: {}]'.format(key))], []
        text = format_cell(cell)
        if diagnostic:
            text = '{} (diagnostic)'.format(text)
        return [nodes.literal(rawtext, text)], []

    return role


class BenchProvenance(Directive):
    """Render the run records behind the numbers on this page as one block.

    Several run ids may be given; the block then keeps only the fields a reader compares
    across runs. A field whose value is the same in every run is shown once, and a field
    that differs is shown per run, keyed by that run's measurement date. The
    ``:disclosures:`` flag folds every disclosure attached to a number this page prints
    into the same block, in place of a separate ``bench-disclosures``.
    """

    required_arguments = 1
    optional_arguments = 8
    final_argument_whitespace = False
    has_content = False
    option_spec = {'disclosures': directives.flag}

    FIELDS = [
        ('measured_at', 'Measured'),
        ('host', 'Host'),
        ('reps', 'Repetitions'),
        ('runtime', 'Runtime'),
        ('dataset', 'Dataset'),
        ('pygraphistry_commit', 'PyGraphistry commit'),
        ('pyg_bench_commit', 'Benchmark commit'),
        ('artifact', 'Raw artifacts'),
        ('row_validation', 'Result validation'),
        ('competitor_version', 'Competitor version'),
    ]
    MERGED_FIELDS = [
        ('measured_at', 'Measured'),
        ('host', 'Host'),
        ('reps', 'Repetitions'),
        ('runtime', 'Runtime'),
        ('dataset', 'Dataset'),
        ('row_validation', 'Result validation'),
    ]

    def run(self) -> List[nodes.Node]:
        state = _state()
        docname = self.state.document.settings.env.docname
        runs = []  # type: List[JSONObject]
        for argument in self.arguments:
            run_id = argument.strip()
            state.provenance.setdefault(docname, []).append(run_id)
            run = state.run(run_id)
            if run is None:
                message = '{}: no run {!r} in the published artifact'.format(docname, run_id)
                state.fail(message)
                logger.warning('[gfql-bench] %s', message)
                return []
            runs.append(run)
        spec = self.FIELDS if len(runs) == 1 else self.MERGED_FIELDS
        field_list = _merged_fields(runs, spec)
        if 'disclosures' in self.options:
            state.disclosed.append(docname)
            disclosures = _disclosures(state, docname)
            if disclosures:
                field_list += _field('Caveats', _bullets(disclosures))
        return [_admonition('Measurement', field_list)]


class BenchDisclosures(Directive):
    """Render every disclosure attached to a number this page prints."""

    required_arguments = 0
    optional_arguments = 0
    has_content = False

    def run(self) -> List[nodes.Node]:
        state = _state()
        docname = self.state.document.settings.env.docname
        state.disclosed.append(docname)
        seen = _disclosures(state, docname)
        if not seen:
            return []
        return [_admonition('About these measurements', _bullets(seen))]


def _disclosures(state: State, docname: str) -> List[str]:
    seen = []  # type: List[str]
    for key in state.refs.get(docname, []):
        cell = state.cell(key)
        if cell is None:
            continue
        raw = cell.get('disclosures')
        if not isinstance(raw, list):
            continue
        for item in raw:
            if isinstance(item, str) and item and item not in seen:
                seen.append(item)
    return seen


def _bullets(items: List[str]) -> nodes.bullet_list:
    bullets = nodes.bullet_list()
    for item in items:
        entry = nodes.list_item()
        entry += nodes.paragraph(text=item)
        bullets += entry
    return bullets


def _field(label: str, body_content: nodes.Element) -> nodes.field:
    field = nodes.field()
    field += nodes.field_name(text=label)
    body = nodes.field_body()
    body += body_content
    field += body
    return field


def _merged_fields(runs: List[JSONObject], spec: List[Tuple[str, str]]) -> nodes.field_list:
    field_list = nodes.field_list()
    for key, label in spec:
        values = []  # type: List[Tuple[str, str]]
        for run in runs:
            value = run.get(key)
            if isinstance(value, str) and value:
                values.append((str(run.get('measured_at', '')), value))
        if not values:
            continue
        distinct = []  # type: List[str]
        for _, value in values:
            if value not in distinct:
                distinct.append(value)
        if len(distinct) == 1:
            field_list += _field(label, nodes.paragraph(text=distinct[0]))
        elif key == 'measured_at':
            field_list += _field(label, nodes.paragraph(text=' and '.join(distinct)))
        else:
            field_list += _field(label, _bullets(
                ['{}: {}'.format(date, value) for date, value in values]))
    return field_list


def _fields(run: JSONObject, spec: List[Tuple[str, str]]) -> nodes.field_list:
    field_list = nodes.field_list()
    for key, label in spec:
        value = run.get(key)
        if not isinstance(value, str) or not value:
            continue
        field = nodes.field()
        field += nodes.field_name(text=label)
        body = nodes.field_body()
        body += nodes.paragraph(text=value)
        field += body
        field_list += field
    return field_list


def _admonition(title: str, content: nodes.Element) -> nodes.Element:
    container = nodes.admonition()
    container += nodes.title(text=title)
    container += content
    container['classes'].append('note')
    return container


def _on_builder_inited(app: Sphinx) -> None:
    global _STATE
    try:
        _STATE = load_state(datetime.date.today())
    except BenchDataError as exc:
        raise BenchNumberError(str(exc)) from exc


def _on_purge(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    if _STATE is not None:
        _STATE.forget(docname)


def _on_build_finished(app: Sphinx, exception: Optional[Exception]) -> None:
    if exception is not None or _STATE is None:
        return
    before = len(_STATE.problems)
    audit_pages(_STATE)
    for message in _STATE.problems[before:]:
        logger.warning('[gfql-bench] %s', message)
    if _STATE.problems:
        raise BenchNumberError(
            'the published benchmark numbers were used incorrectly:\n  {}'.format(
                '\n  '.join(_STATE.problems)))


def setup(app: Sphinx) -> Dict[str, object]:
    app.add_role('bench', _bench_role(diagnostic=False))
    app.add_role('bench-diag', _bench_role(diagnostic=True))
    app.add_directive('bench-provenance', BenchProvenance)
    app.add_directive('bench-disclosures', BenchDisclosures)
    app.connect('builder-inited', _on_builder_inited)
    app.connect('env-purge-doc', _on_purge)
    app.connect('build-finished', _on_build_finished)
    return {'version': '1', 'parallel_read_safe': False, 'parallel_write_safe': True}
