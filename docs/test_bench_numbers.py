"""The benchmark-number gate must FAIL on the ways a published figure goes wrong.

These are negative tests on purpose: a gate that has never been shown to reject
anything is indistinguishable from no gate. Each case here corresponds to a real
failure this repository has had — a number that outlived its measurement, a
diagnostic-only cell quoted as a competitor result, a caveat dropped in transit,
and a figure typed in by hand.
"""

from __future__ import annotations

import datetime
import json
import os
import sys
from typing import Dict, List

import pytest

_DOCS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_DOCS, 'source', '_ext'))
sys.path.insert(0, os.path.join(os.path.dirname(_DOCS), 'bin'))

from gfql_bench_data import (  # noqa: E402
    BenchDataError,
    load_bench_data,
    default_data_path,
)

import check_bench_numbers as gate  # noqa: E402


def _write(tmp_path: str, payload: Dict[str, object]) -> str:
    path = os.path.join(tmp_path, 'bench.json')
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(payload, handle)
    return path


def _base_payload() -> Dict[str, object]:
    return {
        'schema_version': 1,
        'generated_by': 'test',
        'policy': {
            'max_age_days': 60,
            'max_compute_commit_drift': 12,
            'managed_docs': [],
        },
        'runs': {
            'r1': {
                'measured_at': '2026-07-26',
                'host': 'dgx-spark',
                'perf_lock_held': True,
                'quiet_host': True,
                'reps': '3 warmups + 7 timed',
                'pygraphistry_commit': '84be35fb',
                'pyg_bench_commit': '47f94ba',
                'runtime': 'rapids 26.02',
                'dataset': 'graph-benchmark 20k',
                'artifact': 'results/x',
                'row_validation': 'rows equal the competitor on every slot',
            },
        },
        'cells': {
            'a.b.polars': {
                'run': 'r1', 'workload': 'q1', 'engine': 'polars', 'value': 13.13,
                'unit': 'ms', 'decimals': 2, 'status': 'ok',
                'comparison_allowed': True, 'board_quotable': True, 'disclosures': [],
            },
        },
    }


def test_the_shipped_source_of_truth_loads() -> None:
    data = load_bench_data(default_data_path())
    assert data.runs, 'the shipped source-of-truth must contain at least one run'
    for cell in data.cells.values():
        assert cell.run_id in data.runs


def test_unknown_key_is_refused(tmp_path: object) -> None:
    data = load_bench_data(_write(str(tmp_path), _base_payload()))
    with pytest.raises(BenchDataError) as excinfo:
        data.cell('a.b.polars_new')
    assert 'must not be published' in str(excinfo.value)


def test_a_diagnostic_cell_cannot_claim_to_be_board_quotable(tmp_path: object) -> None:
    payload = _base_payload()
    cells: Dict[str, Dict[str, object]] = payload['cells']  # type: ignore[assignment]
    cells['a.b.polars']['comparison_allowed'] = False
    with pytest.raises(BenchDataError) as excinfo:
        load_bench_data(_write(str(tmp_path), payload))
    assert 'diagnostic-only pairing is never board-quotable' in str(excinfo.value)


def test_a_non_ok_status_must_carry_a_disclosure(tmp_path: object) -> None:
    payload = _base_payload()
    cells: Dict[str, Dict[str, object]] = payload['cells']  # type: ignore[assignment]
    cells['a.b.polars']['status'] = 'partial'
    with pytest.raises(BenchDataError) as excinfo:
        load_bench_data(_write(str(tmp_path), payload))
    assert 'must carry at least one disclosure' in str(excinfo.value)


def test_a_run_without_provenance_is_not_loadable(tmp_path: object) -> None:
    payload = _base_payload()
    runs: Dict[str, Dict[str, object]] = payload['runs']  # type: ignore[assignment]
    del runs['r1']['perf_lock_held']
    with pytest.raises(BenchDataError) as excinfo:
        load_bench_data(_write(str(tmp_path), payload))
    assert 'perf_lock_held' in str(excinfo.value)


def test_freshness_gate_fires_once_a_run_ages_out(tmp_path: object) -> None:
    data = load_bench_data(_write(str(tmp_path), _base_payload()))
    fresh = gate.check_freshness(data, datetime.date(2026, 8, 1))
    assert fresh == []
    stale = gate.check_freshness(data, datetime.date(2026, 12, 1))
    assert len(stale) == 1
    assert 'Re-measure or drop the claim' in stale[0].render()


def test_hand_typed_literal_is_rejected_in_a_managed_doc(tmp_path: object) -> None:
    source = os.path.join(str(tmp_path), 'source')
    os.makedirs(source)
    with open(os.path.join(source, 'managed.rst'), 'w', encoding='utf-8') as handle:
        handle.write('Seeded lookups run in 0.124 ms and are 9.4x faster.\n')
    payload = _base_payload()
    policy: Dict[str, object] = payload['policy']  # type: ignore[assignment]
    policy['managed_docs'] = ['managed.rst']
    data = load_bench_data(_write(str(tmp_path), payload))

    previous = gate.DOCS_SOURCE
    try:
        gate.DOCS_SOURCE = source
        findings: List[gate.Finding] = gate.check_hand_typed_literals(data)
    finally:
        gate.DOCS_SOURCE = previous
    rendered = ' '.join(finding.render() for finding in findings)
    assert '0.124 ms' in rendered
    assert '9.4x' in rendered


def test_allowlisted_literal_is_permitted(tmp_path: object) -> None:
    source = os.path.join(str(tmp_path), 'source')
    os.makedirs(source)
    with open(os.path.join(source, 'managed.rst'), 'w', encoding='utf-8') as handle:
        handle.write('The graph grows 10x between the two columns.\n')
    payload = _base_payload()
    policy: Dict[str, object] = payload['policy']  # type: ignore[assignment]
    policy['managed_docs'] = ['managed.rst']
    policy['literal_allowlist'] = {'managed.rst': ['10x']}
    data = load_bench_data(_write(str(tmp_path), payload))

    previous = gate.DOCS_SOURCE
    try:
        gate.DOCS_SOURCE = source
        assert gate.check_hand_typed_literals(data) == []
    finally:
        gate.DOCS_SOURCE = previous
