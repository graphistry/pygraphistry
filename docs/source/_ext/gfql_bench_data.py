"""Typed loader + validator for the GFQL benchmark source-of-truth.

Single machine-readable source of every benchmark number published in the docs
(``docs/source/_data/gfql_benchmarks.json``), produced by the pyg-bench exporter
``scripts/export_docs_numbers.py``.

This module is deliberately Sphinx-free so that ``bin/check_bench_numbers.py``
can validate the data (and the docs that reference it) without building the
docs, and without re-running any benchmark.

Design rules enforced here:

- A cell may only be published if it is attached to a run with full provenance.
- A cell that is not board-quotable (``comparison_allowed`` false, or a
  non-``ok`` status that has not been dispositioned) cannot be rendered by the
  plain reference form; the caller must ask for the diagnostic form explicitly.
- A cell's disclosures travel with it: the consumer is told which disclosures a
  page owes, and it is an error to publish a disclosed cell without them.
"""

from __future__ import annotations

import datetime
import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    'BenchDataError',
    'BenchCell',
    'BenchRun',
    'BenchPolicy',
    'BenchData',
    'load_bench_data',
    'default_data_path',
]

JsonScalar = Union[str, int, float, bool, None]
JsonValue = Union[JsonScalar, List['JsonValue'], Dict[str, 'JsonValue']]
JsonObject = Dict[str, 'JsonValue']

SCHEMA_VERSION = 1

#: Statuses a cell may carry. Anything other than ``ok`` is a disclosure-bearing
#: result and may never be published as a bare number.
VALID_STATUSES = (
    'ok',                    # clean result, values verified
    'partial',               # answered, but with a documented workaround
    'adapter_workaround',    # the harness, not GFQL, shaped the query
    'result_mismatch',       # rows disagree with the reference => unquotable
    'unsupported',           # the engine declined the query
)

#: Statuses that are still publishable on a board, provided their disclosures
#: are rendered alongside.
BOARD_STATUSES = ('ok', 'partial', 'adapter_workaround')

VALID_UNITS = ('ms', 's', 'x', 'rows')

_KEY_RE = re.compile(r'^[a-z0-9]+(?:[._-][a-z0-9]+)*$')
_DATE_RE = re.compile(r'^\d{4}-\d{2}-\d{2}$')
_SHA_RE = re.compile(r'^[0-9a-f]{7,40}$')


class BenchDataError(Exception):
    """Raised when the benchmark source-of-truth is malformed or unpublishable."""


def _obj(value: JsonValue, where: str) -> JsonObject:
    if not isinstance(value, dict):
        raise BenchDataError('{}: expected an object, got {}'.format(where, type(value).__name__))
    return value


def _req_str(obj: JsonObject, key: str, where: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise BenchDataError('{}: missing/invalid string field {!r}'.format(where, key))
    return value


def _opt_str(obj: JsonObject, key: str, where: str) -> Optional[str]:
    value = obj.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value:
        raise BenchDataError('{}: field {!r} must be a non-empty string or absent'.format(where, key))
    return value


def _req_bool(obj: JsonObject, key: str, where: str) -> bool:
    value = obj.get(key)
    if not isinstance(value, bool):
        raise BenchDataError('{}: missing/invalid boolean field {!r}'.format(where, key))
    return value


def _req_int(obj: JsonObject, key: str, where: str) -> int:
    value = obj.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise BenchDataError('{}: missing/invalid integer field {!r}'.format(where, key))
    return value


def _req_number(obj: JsonObject, key: str, where: str) -> float:
    value = obj.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BenchDataError('{}: missing/invalid numeric field {!r}'.format(where, key))
    return float(value)


def _str_list(value: JsonValue, where: str) -> List[str]:
    if not isinstance(value, list):
        raise BenchDataError('{}: expected a list of strings'.format(where))
    out: List[str] = []
    for i, item in enumerate(value):
        if not isinstance(item, str) or not item:
            raise BenchDataError('{}[{}] must be a non-empty string'.format(where, i))
        out.append(item)
    return out


def _req_str_list(obj: JsonObject, key: str, where: str) -> List[str]:
    if key not in obj:
        raise BenchDataError('{}: missing list field {!r}'.format(where, key))
    return _str_list(obj[key], '{}.{}'.format(where, key))


class BenchRun:
    """Provenance for one measurement session. Every field is mandatory.

    A run without a commit, a host, a perf-lock disposition and a rep count is
    not a measurement anyone can defend, so the loader refuses to construct one.
    """

    def __init__(
        self,
        run_id: str,
        measured_at: datetime.date,
        host: str,
        perf_lock_held: bool,
        quiet_host: bool,
        reps: str,
        pygraphistry_commit: str,
        pyg_bench_commit: str,
        runtime: str,
        dataset: str,
        artifact: str,
        row_validation: str,
        competitor: Optional[str],
        competitor_version: Optional[str],
    ) -> None:
        self.run_id = run_id
        self.measured_at = measured_at
        self.host = host
        self.perf_lock_held = perf_lock_held
        self.quiet_host = quiet_host
        self.reps = reps
        self.pygraphistry_commit = pygraphistry_commit
        self.pyg_bench_commit = pyg_bench_commit
        self.runtime = runtime
        self.dataset = dataset
        self.artifact = artifact
        self.row_validation = row_validation
        self.competitor = competitor
        self.competitor_version = competitor_version

    def age_days(self, today: datetime.date) -> int:
        return (today - self.measured_at).days

    @staticmethod
    def from_json(run_id: str, raw: JsonValue) -> 'BenchRun':
        where = 'runs.{}'.format(run_id)
        obj = _obj(raw, where)
        measured_at_str = _req_str(obj, 'measured_at', where)
        if not _DATE_RE.match(measured_at_str):
            raise BenchDataError('{}: measured_at must be YYYY-MM-DD, got {!r}'.format(where, measured_at_str))
        year, month, day = measured_at_str.split('-')
        commit = _req_str(obj, 'pygraphistry_commit', where)
        if not _SHA_RE.match(commit):
            raise BenchDataError('{}: pygraphistry_commit must be a git sha, got {!r}'.format(where, commit))
        bench_commit = _req_str(obj, 'pyg_bench_commit', where)
        if not _SHA_RE.match(bench_commit):
            raise BenchDataError('{}: pyg_bench_commit must be a git sha, got {!r}'.format(where, bench_commit))
        return BenchRun(
            run_id=run_id,
            measured_at=datetime.date(int(year), int(month), int(day)),
            host=_req_str(obj, 'host', where),
            perf_lock_held=_req_bool(obj, 'perf_lock_held', where),
            quiet_host=_req_bool(obj, 'quiet_host', where),
            reps=_req_str(obj, 'reps', where),
            pygraphistry_commit=commit,
            pyg_bench_commit=bench_commit,
            runtime=_req_str(obj, 'runtime', where),
            dataset=_req_str(obj, 'dataset', where),
            artifact=_req_str(obj, 'artifact', where),
            row_validation=_req_str(obj, 'row_validation', where),
            competitor=_opt_str(obj, 'competitor', where),
            competitor_version=_opt_str(obj, 'competitor_version', where),
        )


class BenchCell:
    """One published number, with the disclosures that must travel with it."""

    def __init__(
        self,
        key: str,
        run_id: str,
        workload: str,
        engine: str,
        value: float,
        unit: str,
        decimals: int,
        status: str,
        comparison_allowed: bool,
        board_quotable: bool,
        disclosures: List[str],
        competitor: Optional[str],
    ) -> None:
        self.key = key
        self.run_id = run_id
        self.workload = workload
        self.engine = engine
        self.value = value
        self.unit = unit
        self.decimals = decimals
        self.status = status
        self.comparison_allowed = comparison_allowed
        self.board_quotable = board_quotable
        self.disclosures = disclosures
        self.competitor = competitor

    def render(self) -> str:
        """The published text for this cell, including its unit."""
        number = '{:.{}f}'.format(self.value, self.decimals)
        if self.unit == 'x':
            return number + '×'
        if self.unit == 'rows':
            return '{:,} rows'.format(int(self.value))
        return '{} {}'.format(number, self.unit)

    @staticmethod
    def from_json(key: str, raw: JsonValue) -> 'BenchCell':
        where = 'cells.{}'.format(key)
        if not _KEY_RE.match(key):
            raise BenchDataError('{}: key must be lowercase dotted/underscored ascii'.format(where))
        obj = _obj(raw, where)
        unit = _req_str(obj, 'unit', where)
        if unit not in VALID_UNITS:
            raise BenchDataError('{}: unit {!r} not in {}'.format(where, unit, VALID_UNITS))
        status = _req_str(obj, 'status', where)
        if status not in VALID_STATUSES:
            raise BenchDataError('{}: status {!r} not in {}'.format(where, status, VALID_STATUSES))
        comparison_allowed = _req_bool(obj, 'comparison_allowed', where)
        board_quotable = _req_bool(obj, 'board_quotable', where)
        disclosures = _req_str_list(obj, 'disclosures', where)

        if board_quotable and not comparison_allowed:
            raise BenchDataError(
                '{}: board_quotable=true is impossible with comparison_allowed=false '
                '(a diagnostic-only pairing is never board-quotable)'.format(where)
            )
        if board_quotable and status not in BOARD_STATUSES:
            raise BenchDataError(
                '{}: board_quotable=true is impossible with status={!r}'.format(where, status)
            )
        if board_quotable and status != 'ok' and not disclosures:
            raise BenchDataError(
                '{}: status={!r} must carry at least one disclosure'.format(where, status)
            )
        return BenchCell(
            key=key,
            run_id=_req_str(obj, 'run', where),
            workload=_req_str(obj, 'workload', where),
            engine=_req_str(obj, 'engine', where),
            value=_req_number(obj, 'value', where),
            unit=unit,
            decimals=_req_int(obj, 'decimals', where),
            status=status,
            comparison_allowed=comparison_allowed,
            board_quotable=board_quotable,
            disclosures=disclosures,
            competitor=_opt_str(obj, 'competitor', where),
        )


class BenchPolicy:
    """Build-breaking thresholds, versioned with the data itself."""

    def __init__(
        self,
        max_age_days: int,
        max_compute_commit_drift: int,
        managed_docs: List[str],
        literal_allowlist: Dict[str, List[str]],
    ) -> None:
        self.max_age_days = max_age_days
        self.max_compute_commit_drift = max_compute_commit_drift
        self.managed_docs = managed_docs
        #: doc path -> literals that are structural, not measured claims.
        self.literal_allowlist = literal_allowlist

    def allowed_literals(self, doc: str) -> List[str]:
        return self.literal_allowlist.get(doc, [])

    @staticmethod
    def from_json(raw: JsonValue) -> 'BenchPolicy':
        where = 'policy'
        obj = _obj(raw, where)
        allowlist: Dict[str, List[str]] = {}
        raw_allowlist = obj.get('literal_allowlist')
        if raw_allowlist is not None:
            allow_obj = _obj(raw_allowlist, where + '.literal_allowlist')
            for doc in sorted(allow_obj):
                allowlist[doc] = _str_list(
                    allow_obj[doc], '{}.literal_allowlist.{}'.format(where, doc))
        return BenchPolicy(
            max_age_days=_req_int(obj, 'max_age_days', where),
            max_compute_commit_drift=_req_int(obj, 'max_compute_commit_drift', where),
            managed_docs=_req_str_list(obj, 'managed_docs', where),
            literal_allowlist=allowlist,
        )


class BenchData:
    def __init__(
        self,
        source_path: str,
        generated_by: str,
        policy: BenchPolicy,
        runs: Dict[str, BenchRun],
        cells: Dict[str, BenchCell],
    ) -> None:
        self.source_path = source_path
        self.generated_by = generated_by
        self.policy = policy
        self.runs = runs
        self.cells = cells

    def cell(self, key: str) -> BenchCell:
        try:
            return self.cells[key]
        except KeyError:
            raise BenchDataError(
                'unknown benchmark key {!r}; it is not in {} '
                '(a number the source-of-truth does not contain must not be published)'.format(
                    key, os.path.basename(self.source_path)
                )
            )

    def run_for(self, cell: BenchCell) -> BenchRun:
        return self.runs[cell.run_id]

    def stale_runs(self, today: datetime.date) -> List[Tuple[BenchRun, int]]:
        """Runs older than the policy threshold, newest-first."""
        out: List[Tuple[BenchRun, int]] = []
        for run in self.runs.values():
            age = run.age_days(today)
            if age > self.policy.max_age_days:
                out.append((run, age))
        out.sort(key=lambda pair: pair[1])
        return out

    def referenced_runs(self, keys: Sequence[str]) -> List[BenchRun]:
        seen: List[str] = []
        for key in keys:
            run_id = self.cell(key).run_id
            if run_id not in seen:
                seen.append(run_id)
        return [self.runs[run_id] for run_id in seen]


def default_data_path() -> str:
    """``docs/source/_data/gfql_benchmarks.json``, resolved from this file."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(here), '_data', 'gfql_benchmarks.json')


def load_bench_data(path: Optional[str] = None) -> BenchData:
    data_path = path if path is not None else default_data_path()
    if not os.path.isfile(data_path):
        raise BenchDataError('benchmark source-of-truth not found: {}'.format(data_path))
    with open(data_path, 'r', encoding='utf-8') as handle:
        raw_value: JsonValue = json.load(handle)
    raw = _obj(raw_value, 'root')

    schema_version = _req_int(raw, 'schema_version', 'root')
    if schema_version != SCHEMA_VERSION:
        raise BenchDataError(
            'schema_version {} is not supported by this checkout (expected {})'.format(
                schema_version, SCHEMA_VERSION
            )
        )

    policy = BenchPolicy.from_json(raw.get('policy'))

    raw_runs = _obj(raw.get('runs'), 'runs')
    runs: Dict[str, BenchRun] = {}
    for run_id in sorted(raw_runs):
        runs[run_id] = BenchRun.from_json(run_id, raw_runs[run_id])

    raw_cells = _obj(raw.get('cells'), 'cells')
    cells: Dict[str, BenchCell] = {}
    for key in sorted(raw_cells):
        cell = BenchCell.from_json(key, raw_cells[key])
        if cell.run_id not in runs:
            raise BenchDataError(
                'cells.{}: references run {!r}, which has no provenance record'.format(key, cell.run_id)
            )
        cells[key] = cell

    return BenchData(
        source_path=data_path,
        generated_by=_req_str(raw, 'generated_by', 'root'),
        policy=policy,
        runs=runs,
        cells=cells,
    )
