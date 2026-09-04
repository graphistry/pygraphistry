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
    format_tally,
    tally,
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


def _tally_role(name: str, rawtext: str, text: str, lineno: int, inliner: Inliner,
                options=None, content=None) -> RoleResult:
    """``:bench-tally:`<prefix>|<engine>|<competitor>``` -> "N of M" from published cells."""
    state = _state()
    docname = inliner.document.settings.env.docname
    parts = [part.strip() for part in text.split('|')]
    before = len(state.problems)
    result = None
    if len(parts) != 3 or not all(parts):
        state.fail('{}:{}: bench-tally expects <prefix>|<engine>|<competitor>, got {!r}'.format(
            docname, lineno, text))
    else:
        result = tally(state, parts[0], parts[1], parts[2], docname, lineno)
    for message in state.problems[before:]:
        logger.warning('[gfql-bench] %s', message)
    if result is None:
        return [nodes.strong(rawtext, '[MISSING BENCHMARK TALLY: {}]'.format(text))], []
    return [nodes.Text(format_tally(*result))], []


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


class BenchBoard(Directive):
    """A results table from published cells with the fastest cell per row in bold.

    ::

        .. bench-board:: graphbench.100k
           :rows: q1,q2,q3
           :columns: kuzu=Kuzu, polars=GFQL polars
           :diagnostic: gfql_polars_059
           :units: ms

    Every cell is ``<prefix>.<row>.<column-key>`` looked up through the same gate as the
    ``:bench:`` role (a column named under ``:diagnostic:`` goes through ``:bench-diag:``).
    A missing cell renders as a dash, is not a candidate for fastest, and is not an error:
    a database that cannot run a query has no number. The last column names the fastest
    system on the row, so a win or a loss is read off without comparing digits.
    """

    required_arguments = 1
    optional_arguments = 0
    has_content = False
    option_spec = {
        'rows': directives.unchanged_required,
        'columns': directives.unchanged_required,
        'diagnostic': directives.unchanged,
        'row-labels': directives.unchanged,
    }

    def run(self) -> List[nodes.Node]:
        state = _state()
        docname = self.state.document.settings.env.docname
        prefix = self.arguments[0].strip()
        rows = [r.strip() for r in self.options['rows'].split(',') if r.strip()]
        columns = []  # type: List[Tuple[str, str]]
        for item in self.options['columns'].split(','):
            key, _, label = item.partition('=')
            columns.append((key.strip(), (label or key).strip()))
        diagnostic = {c.strip() for c in self.options.get('diagnostic', '').split(',') if c.strip()}
        labels = {}  # type: Dict[str, str]
        for item in self.options.get('row-labels', '').split(';'):
            key, _, label = item.partition('=')
            if key.strip():
                labels[key.strip()] = label.strip()
        before = len(state.problems)
        table = nodes.table()
        tgroup = nodes.tgroup(cols=len(columns) + 2)
        table += tgroup
        for _ in range(len(columns) + 2):
            tgroup += nodes.colspec(colwidth=1)
        thead = nodes.thead()
        tgroup += thead
        thead += _row([nodes.paragraph(text=t) for t in ['Query'] + [c[1] for c in columns] + ['Fastest']])
        tbody = nodes.tbody()
        tgroup += tbody
        for row in rows:
            cells = []  # type: List[Tuple[str, Optional[JSONObject]]]
            for key, _ in columns:
                cell_key = '{}.{}.{}'.format(prefix, row, key)
                cell = state.cell(cell_key)
                if cell is None:
                    cells.append((key, None))
                    continue
                cells.append((key, check_reference(state, cell_key, docname, self.lineno, key in diagnostic)))
            values = [(key, cell['value']) for key, cell in cells
                      if cell is not None and isinstance(cell['value'], (int, float)) and key not in diagnostic]
            fastest = min(values, key=lambda kv: kv[1])[0] if values else None
            entries = [nodes.paragraph(text=labels.get(row, row))]
            for key, cell in cells:
                if cell is None:
                    entries.append(nodes.paragraph(text='\u2014'))
                    continue
                text = format_cell(cell)
                if key in diagnostic:
                    text += ' (diagnostic)'
                literal = nodes.literal(text, text)
                para = nodes.paragraph()
                if key == fastest and len(values) > 1:
                    strong = nodes.strong()
                    strong += literal
                    para += strong
                else:
                    para += literal
                entries.append(para)
            fastest_label = dict(columns).get(fastest, '') if fastest and len(values) > 1 else '\u2014'
            entries.append(nodes.paragraph(text=fastest_label))
            tbody += _row(entries)
        for message in state.problems[before:]:
            logger.warning('[gfql-bench] %s', message)
        return [table]


def _row(entries: List[nodes.Node]) -> nodes.row:
    row = nodes.row()
    for entry in entries:
        cell = nodes.entry()
        cell += entry
        row += cell
    return row


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
    app.add_role('bench-tally', _tally_role)
    app.add_directive('bench-provenance', BenchProvenance)
    app.add_directive('bench-disclosures', BenchDisclosures)
    app.add_directive('bench-board', BenchBoard)
    app.connect('builder-inited', _on_builder_inited)
    app.connect('env-purge-doc', _on_purge)
    app.connect('build-finished', _on_build_finished)
    return {'version': '1', 'parallel_read_safe': False, 'parallel_write_safe': True}
