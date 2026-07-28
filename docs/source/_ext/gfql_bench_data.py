"""Load and re-verify the benchmark artifact pyg-bench publishes.

Deliberately stdlib-only. The Sphinx extension in ``gfql_bench.py`` needs docutils;
the contract re-verification, the staleness rule and the point-of-use decision must
not, or they would run in exactly one CI lane and be invisible everywhere else.

``docs/source/_data/gfql_benchmarks.json`` is a vendored copy of pyg-bench's
``published/docs-numbers.json``; ``gfql_benchmarks.contract.json`` is a vendored copy
of the contract it promises to satisfy. pyg-bench checks those promises before it
publishes. This checks them again, from the contract *document*, with an
implementation that does not import that repository - a shared library would fail
identically on both sides of the boundary and prove nothing.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import re
from typing import Dict, List, Optional, Union

#: A decoded JSON document.
JSONValue = Union[
    None, bool, int, float, str, List["JSONValue"], Dict[str, "JSONValue"]
]
JSONObject = Dict[str, JSONValue]

#: ``_ext`` and ``_data`` are siblings under ``docs/source``.
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '_data')
BENCHMARKS_JSON = os.path.join(DATA_DIR, 'gfql_benchmarks.json')
CONTRACT_JSON = os.path.join(DATA_DIR, 'gfql_benchmarks.contract.json')


class BenchDataError(Exception):
    """The vendored artifact cannot be trusted; nothing may render from it."""


def _obj(value: JSONValue, where: str) -> JSONObject:
    if not isinstance(value, dict):
        raise BenchDataError('{}: expected an object'.format(where))
    return value


def _strings(value: JSONValue, where: str) -> List[str]:
    if not isinstance(value, list):
        raise BenchDataError('{}: expected an array of strings'.format(where))
    out = []  # type: List[str]
    for item in value:
        if not isinstance(item, str) or not item:
            raise BenchDataError('{}: entries must be non-empty strings'.format(where))
        out.append(item)
    return out


def load(path: str) -> JSONObject:
    if not os.path.exists(path):
        raise BenchDataError(
            '{} is missing. It is a vendored copy of pyg-bench '
            'published/docs-numbers.json.'.format(path))
    with open(path, encoding='utf-8') as handle:
        return _obj(json.load(handle), path)


def reverify(payload: JSONObject, contract: JSONObject) -> None:
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
        raise BenchDataError('contract.max_decimals must be an integer')

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
        raise BenchDataError(
            '{} breaks the data contract vendored at {}:\n  {}'.format(
                BENCHMARKS_JSON, CONTRACT_JSON, '\n  '.join(problems)))


class State:
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




def format_cell(cell: JSONObject) -> str:
    """Print the number exactly as published; rounding never happens at render time."""
    value = cell['value']
    decimals = cell['decimals']
    unit = cell['unit']
    assert isinstance(value, (int, float)) and isinstance(decimals, int) and isinstance(unit, str)
    text = '{:.{}f}'.format(float(value), decimals)
    return '{}{}'.format(text, unit) if unit == 'x' else '{} {}'.format(text, unit)


def check_reference(state: State, key: str, docname: str, lineno: int,
                    diagnostic: bool) -> Optional[JSONObject]:
    """Decide whether this page may print this number, AT THE POINT OF USE.

    Returns the cell to render, or None when there is nothing publishable. Every
    refusal is recorded on ``state.problems``, which the build turns into a failure.
    """
    state.refs.setdefault(docname, []).append(key)

    cell = state.cell(key)
    if cell is None:
        state.fail(
            '{}:{}: no published benchmark number {!r}. pyg-bench publishes it or the '
            'docs do not print it.'.format(docname, lineno, key))
        return None

    run_id = cell['run']
    if isinstance(run_id, str):
        age = state.age_days(run_id)
        if age is None:
            state.fail('{}:{}: {!r} names run {!r}, which has no usable measurement '
                       'date'.format(docname, lineno, key, run_id))
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
            '{}:{}: {!r} is not board-quotable (status={!r}, comparison_allowed={!r}); it '
            'may only appear through :bench-diag:, which labels it diagnostic-only'.format(
                docname, lineno, key, cell['status'], cell['comparison_allowed']))
    return cell


def audit_pages(state: State) -> None:
    """Every page that prints a number must also carry its provenance and caveats."""
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
            state.fail('{}: prints a number from run {!r} without rendering its provenance '
                       '(add ".. bench-provenance:: {}")'.format(docname, run_id, run_id))
        if needs_disclosure and docname not in state.disclosed:
            state.fail('{}: prints a number that carries a disclosure but has no '
                       '".. bench-disclosures::" block'.format(docname))


def load_state(today: Optional[datetime.date] = None) -> State:
    """Load the vendored artifact, re-verify it, and return the render-time state."""
    payload = load(BENCHMARKS_JSON)
    contract = load(CONTRACT_JSON)
    reverify(payload, contract)
    return State(payload, today or datetime.date.today())
