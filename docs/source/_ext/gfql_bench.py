"""Sphinx extension: pretty-print the benchmark numbers pyg-bench publishes.

Measurement, provenance and publishability all live in `graphistry/pyg-bench`, which
owns the runs. This repository does one thing with them: it renders them. The two files
under ``docs/source/_data`` are vendored copies of that repository's
``published/docs-numbers.json`` and ``manifests/docs-numbers.contract.json``.

Docs never restate a measured number, they reference one::

    * - Twitter, GPU
      - :bench:`pagerank.twitter.gfql_gpu`

and the page that references a cell must also render that cell's provenance and its
disclosures::

    .. bench-provenance:: filter-pagerank-20260728
    .. bench-disclosures::

The contract names five obligations for a consumer. They are build failures here, never
silent text:

- a key the artifact does not contain            -> build fails
- a run measured longer ago than ``max_age_days`` -> build fails
- a non-quotable cell printed as a bare number    -> build fails (``:bench-diag:`` labels it)
- a page that drops a referenced cell's provenance or disclosures -> build fails
- a payload that breaks the contract at all       -> build fails, before anything renders

The re-verification below is written against the contract *document*, independently of
pyg-bench's own implementation. A shared library would fail identically on both sides of
the boundary and prove nothing.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
from typing import Dict, List, Optional, Tuple, Union

from docutils import nodes
from docutils.parsers.rst import Directive
from docutils.parsers.rst.states import Inliner
from sphinx.application import Sphinx
from sphinx.environment import BuildEnvironment
from sphinx.errors import SphinxError
from sphinx.util import logging as sphinx_logging

logger = sphinx_logging.getLogger(__name__)

RoleResult = Tuple[List[nodes.Node], List[nodes.system_message]]

#: A decoded JSON document.
JSONValue = Union[
    None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]
]
JSONObject = Dict[str, JSONValue]

#: ``_ext`` and ``_data`` are siblings under ``docs/source``.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_data')
BENCHMARKS_JSON = os.path.join(DATA_DIR, 'gfql_benchmarks.json')
CONTRACT_JSON = os.path.join(DATA_DIR, 'gfql_benchmarks.contract.json')


class BenchNumberError(SphinxError):
    category = 'GFQL benchmark number check failed'


def _obj(value: JSONValue, where: str) -> JSONObject:
    if not isinstance(value, dict):
        raise BenchNumberError('{}: expected an object'.format(where))
    return value


def _strings(value: JSONValue, where: str) -> List[str]:
    if not isinstance(value, list):
        raise BenchNumberError('{}: expected an array of strings'.format(where))
    out = []  # type: List[str]
    for item in value:
        if not isinstance(item, str) or not item:
            raise BenchNumberError('{}: entries must be non-empty strings'.format(where))
        out.append(item)
    return out


def _load(path: str) -> JSONObject:
    if not os.path.exists(path):
        raise BenchNumberError(
            '{} is missing. It is a vendored copy of pyg-bench '
            'published/docs-numbers.json.'.format(path))
    with open(path, encoding='utf-8') as handle:
        return _obj(json.load(handle), path)


def _reverify(payload: JSONObject, contract: JSONObject) -> None:
    """Check the artifact against the contract document, before anything renders."""
    problems = []  # type: List[str]

    expected_version = contract.get('contract_version')
    if payload.get('contract_version') != expected_version:
        problems.append(
            'contract_version {!r} but this repository vendors contract {!r}'.format(
                payload.get('contract_version'), expected_version))

    for field in _strings(contract.get('top_level_required'), 'contract.top_level_required'):
        if field not in payload:
            problems.append('artifact is missing {!r}'.format(field))

    policy = payload.get('policy')
    if not isinstance(policy, dict):
        problems.append('policy: expected an object')
        policy = {}
    for field in _strings(contract.get('policy_required'), 'contract.policy_required'):
        if field not in policy:
            problems.append('policy is missing {!r}'.format(field))

    run_required = _strings(contract.get('run_required'), 'contract.run_required')
    cell_required = _strings(contract.get('cell_required'), 'contract.cell_required')
    statuses = _strings(contract.get('cell_status'), 'contract.cell_status')
    units = _strings(contract.get('cell_unit'), 'contract.cell_unit')
    key_re = re.compile(str(contract.get('key_pattern')))
    date_re = re.compile(str(contract.get('date_pattern')))
    max_decimals = contract.get('max_decimals')
    if not isinstance(max_decimals, int) or isinstance(max_decimals, bool):
        raise BenchNumberError('contract.max_decimals must be an integer')

    runs = payload.get('runs')
    if not isinstance(runs, dict):
        problems.append('runs: expected an object')
        runs = {}
    for run_id in sorted(runs):
        run = runs[run_id]
        if not isinstance(run, dict):
            problems.append('runs.{}: expected an object'.format(run_id))
            continue
        for field in run_required:
            if field not in run:
                problems.append('runs.{}: missing provenance {!r}'.format(run_id, field))
        measured_at = run.get('measured_at')
        if not isinstance(measured_at, str) or not date_re.match(measured_at):
            problems.append('runs.{}: measured_at is not a date'.format(run_id))

    cells = payload.get('cells')
    if not isinstance(cells, dict):
        problems.append('cells: expected an object')
        cells = {}
    if not cells:
        problems.append('cells: the artifact publishes nothing')
    for key in sorted(cells):
        cell = cells[key]
        if not isinstance(cell, dict):
            problems.append('cells.{}: expected an object'.format(key))
            continue
        if not key_re.match(key):
            problems.append('cells.{}: key does not match the published pattern'.format(key))
        missing = [field for field in cell_required if field not in cell]
        if missing:
            problems.append('cells.{}: missing {}'.format(key, ', '.join(sorted(missing))))
            continue
        if cell['run'] not in runs:
            problems.append('cells.{}: run {!r} has no provenance'.format(key, cell['run']))
        if cell['status'] not in statuses:
            problems.append('cells.{}: unknown status {!r}'.format(key, cell['status']))
        if cell['unit'] not in units:
            problems.append('cells.{}: unknown unit {!r}'.format(key, cell['unit']))
        if not isinstance(cell['engine'], str) or not cell['engine']:
            problems.append('cells.{}: engine is not named'.format(key))
        quotable = cell['board_quotable']
        comparable = cell['comparison_allowed']
        if not isinstance(quotable, bool) or not isinstance(comparable, bool):
            problems.append('cells.{}: quotability flags must be booleans'.format(key))
            continue
        disclosures = cell['disclosures']
        if not isinstance(disclosures, list):
            problems.append('cells.{}: disclosures must be an array'.format(key))
            disclosures = []
        if quotable and not comparable:
            problems.append('cells.{}: board_quotable but not comparison_allowed'.format(key))
        if quotable and cell['status'] != 'ok':
            problems.append('cells.{}: board_quotable with status {!r}'.format(key, cell['status']))
        if (cell['status'] != 'ok' or not comparable) and not disclosures:
            problems.append('cells.{}: caveated but carries no disclosure'.format(key))
        if cell['unit'] == 'x' and not comparable:
            problems.append('cells.{}: a ratio over figures never established as comparable'.format(key))
        value = cell['value']
        decimals = cell['decimals']
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            problems.append('cells.{}: value is not a finite number'.format(key))
            continue
        if not isinstance(decimals, int) or isinstance(decimals, bool) or not 0 <= decimals <= max_decimals:
            problems.append('cells.{}: decimals out of range'.format(key))
            continue
        if round(float(value), decimals) != float(value):
            problems.append('cells.{}: value is not rounded to its own decimals'.format(key))

    if problems:
        raise BenchNumberError(
            '{} breaks the data contract vendored at {}:\n  {}'.format(
                BENCHMARKS_JSON, CONTRACT_JSON, '\n  '.join(problems)))


class _State:
    """Loaded once per build: the data, the clock, and the problems found."""

    def __init__(self, payload: JSONObject, today: datetime.date) -> None:
        self.payload = payload
        self.today = today
        self.problems = []  # type: List[str]
        self.refs = {}  # type: Dict[str, List[str]]
        self.provenance = {}  # type: Dict[str, List[str]]
        self.disclosed = []  # type: List[str]

        policy = _obj(payload.get('policy'), 'policy')
        max_age = policy.get('max_age_days')
        self.max_age_days = max_age if isinstance(max_age, int) and not isinstance(max_age, bool) else 0
        self.cells = _obj(payload.get('cells'), 'cells')
        self.runs = _obj(payload.get('runs'), 'runs')

    def fail(self, message: str) -> None:
        self.problems.append(message)
        logger.warning('[gfql-bench] %s', message)

    def forget(self, docname: str) -> None:
        self.refs.pop(docname, None)
        self.provenance.pop(docname, None)
        if docname in self.disclosed:
            self.disclosed.remove(docname)

    def cell(self, key: str) -> Optional[JSONObject]:
        raw = self.cells.get(key)
        return raw if isinstance(raw, dict) else None

    def run(self, run_id: str) -> Optional[JSONObject]:
        raw = self.runs.get(run_id)
        return raw if isinstance(raw, dict) else None

    def age_days(self, run_id: str) -> Optional[int]:
        run = self.run(run_id)
        if run is None:
            return None
        measured_at = run.get('measured_at')
        if not isinstance(measured_at, str):
            return None
        measured = datetime.datetime.strptime(measured_at, '%Y-%m-%d').date()
        return (self.today - measured).days


_STATE = None  # type: Optional[_State]


def _state() -> _State:
    if _STATE is None:
        raise BenchNumberError('gfql_bench used before builder-inited')
    return _STATE


def _format(cell: JSONObject) -> str:
    value = cell['value']
    decimals = cell['decimals']
    unit = cell['unit']
    assert isinstance(value, (int, float)) and isinstance(decimals, int) and isinstance(unit, str)
    text = '{:.{}f}'.format(float(value), decimals)
    return '{}{}'.format(text, unit) if unit == 'x' else '{} {}'.format(text, unit)


def _bench_role(diagnostic: bool):
    def role(name: str, rawtext: str, key: str, lineno: int, inliner: Inliner,
             options=None, content=None) -> RoleResult:
        state = _state()
        env = inliner.document.settings.env  # type: BenchmarkEnv
        docname = env.docname
        state.refs.setdefault(docname, []).append(key)

        cell = state.cell(key)
        if cell is None:
            state.fail('{}:{}: no published benchmark number {!r}. pyg-bench publishes it or '
                       'the docs do not print it.'.format(docname, lineno, key))
            return [nodes.strong(rawtext, '[MISSING BENCHMARK NUMBER: {}]'.format(key))], []

        run_id = cell['run']
        assert isinstance(run_id, str)
        age = state.age_days(run_id)
        if age is None:
            state.fail('{}:{}: {!r} names run {!r}, which has no usable measurement date'.format(
                docname, lineno, key, run_id))
        elif age > state.max_age_days:
            state.fail('{}:{}: {!r} was measured {} days ago; policy.max_age_days is {}. '
                       'Re-measure in pyg-bench and republish.'.format(
                           docname, lineno, key, age, state.max_age_days))

        quotable = cell['board_quotable'] is True
        if diagnostic and quotable:
            state.fail('{}:{}: {!r} is a published result; use :bench: not :bench-diag:'.format(
                docname, lineno, key))
        if not diagnostic and not quotable:
            state.fail(
                '{}:{}: {!r} is not board-quotable (status={!r}, comparison_allowed={!r}); it may '
                'only appear through :bench-diag:, which labels it diagnostic-only'.format(
                    docname, lineno, key, cell['status'], cell['comparison_allowed']))

        text = _format(cell)
        if diagnostic:
            text = '{} (diagnostic)'.format(text)
        return [nodes.literal(rawtext, text)], []

    return role


BenchmarkEnv = BuildEnvironment


class BenchProvenance(Directive):
    """Render the run record behind the numbers on this page."""

    required_arguments = 1
    optional_arguments = 0
    has_content = False

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

    def run(self) -> List[nodes.Node]:
        state = _state()
        env = self.state.document.settings.env
        run_id = self.arguments[0].strip()
        state.provenance.setdefault(env.docname, []).append(run_id)
        run = state.run(run_id)
        if run is None:
            state.fail('{}: no run {!r} in the published artifact'.format(env.docname, run_id))
            return []

        field_list = nodes.field_list()
        for key, label in self.FIELDS:
            value = run.get(key)
            if not isinstance(value, str) or not value:
                continue
            field = nodes.field()
            field += nodes.field_name(text=label)
            body = nodes.field_body()
            body += nodes.paragraph(text=value)
            field += body
            field_list += field
        container = nodes.admonition()
        container += nodes.title(text='Measurement')
        container += field_list
        container['classes'].append('note')
        return [container]


class BenchDisclosures(Directive):
    """Render every disclosure attached to a number this page prints."""

    required_arguments = 0
    optional_arguments = 0
    has_content = False

    def run(self) -> List[nodes.Node]:
        state = _state()
        env = self.state.document.settings.env
        docname = env.docname
        state.disclosed.append(docname)

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
        if not seen:
            return []
        bullets = nodes.bullet_list()
        for item in seen:
            entry = nodes.list_item()
            entry += nodes.paragraph(text=item)
            bullets += entry
        container = nodes.admonition()
        container += nodes.title(text='About these measurements')
        container += bullets
        container['classes'].append('note')
        return [container]


def _on_builder_inited(app: Sphinx) -> None:
    global _STATE
    payload = _load(BENCHMARKS_JSON)
    contract = _load(CONTRACT_JSON)
    _reverify(payload, contract)
    _STATE = _State(payload, datetime.date.today())


def _on_purge(app: Sphinx, env: BuildEnvironment, docname: str) -> None:
    if _STATE is not None:
        _STATE.forget(docname)


def _on_build_finished(app: Sphinx, exception: Optional[Exception]) -> None:
    if exception is not None or _STATE is None:
        return
    state = _STATE
    for docname, keys in sorted(state.refs.items()):
        runs = set()
        needs_disclosure = False
        for key in keys:
            cell = state.cell(key)
            if cell is None:
                continue
            run = cell['run']
            if isinstance(run, str):
                runs.add(run)
            raw = cell.get('disclosures')
            if isinstance(raw, list) and raw:
                needs_disclosure = True
        rendered = set(state.provenance.get(docname, []))
        for run_id in sorted(runs - rendered):
            state.fail('{}: prints a number from run {!r} without rendering its '
                       'provenance (add ".. bench-provenance:: {}")'.format(
                           docname, run_id, run_id))
        if needs_disclosure and docname not in state.disclosed:
            state.fail('{}: prints a number that carries a disclosure but has no '
                       '".. bench-disclosures::" block'.format(docname))
    if state.problems:
        raise BenchNumberError(
            'the published benchmark numbers were used incorrectly:\n  {}'.format(
                '\n  '.join(state.problems)))


def setup(app: Sphinx) -> Dict[str, object]:
    app.add_role('bench', _bench_role(diagnostic=False))
    app.add_role('bench-diag', _bench_role(diagnostic=True))
    app.add_directive('bench-provenance', BenchProvenance)
    app.add_directive('bench-disclosures', BenchDisclosures)
    app.connect('builder-inited', _on_builder_inited)
    app.connect('env-purge-doc', _on_purge)
    app.connect('build-finished', _on_build_finished)
    return {'version': '1', 'parallel_read_safe': False, 'parallel_write_safe': True}
