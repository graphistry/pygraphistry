"""Consumer-side re-verification of the benchmark numbers pyg-bench publishes.

`docs/source/_data/gfql_benchmarks.json` is a vendored copy of that repository's
`published/docs-numbers.json`, and `gfql_benchmarks.contract.json` is a vendored copy of
the contract it promises to satisfy. pyg-bench checks those promises before it publishes;
this checks them again before we print anything, because a boundary only holds if both
sides check it.

These run in the ordinary test lane, not only in the docs build, so a number going stale
or a page referencing a key that no longer exists fails CI rather than a nightly. That is
why every rule lives in `gfql_bench_data`, which imports nothing but the standard library;
the docutils half is a renderer. A gate that needs Sphinx to run is a gate that runs in one
job out of forty.
"""

import datetime
import json
import os
import re
import sys

import pytest

DOCS_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(DOCS_DIR, 'source')
sys.path.insert(0, os.path.join(SOURCE_DIR, '_ext'))

import gfql_bench_charts as charts  # noqa: E402
import gfql_bench_data as bench  # noqa: E402

#: ``:bench:`key``` / ``:bench-diag:`key``` as written in the .rst sources.
BENCH_REF = re.compile(r':(bench|bench-diag):`([^`]+)`')


@pytest.fixture(scope='module')
def payload():
    return bench.load(bench.BENCHMARKS_JSON)


@pytest.fixture(scope='module')
def contract():
    return bench.load(bench.CONTRACT_JSON)


def _rst_sources():
    for root, _dirs, files in os.walk(SOURCE_DIR):
        for name in files:
            if name.endswith('.rst') or name.endswith('.md'):
                yield os.path.join(root, name)


def _references():
    """(path, key, is_diagnostic) for every benchmark reference in the docs."""
    for path in sorted(_rst_sources()):
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        for role, key in BENCH_REF.findall(text):
            yield path, key, role == 'bench-diag'


def test_the_vendored_artifact_satisfies_the_vendored_contract(payload, contract):
    bench.reverify(payload, contract)


def test_the_vendored_contract_is_the_one_the_artifact_was_built_against(payload, contract):
    assert payload['contract_version'] == contract['contract_version']


def test_no_published_number_is_stale(payload):
    """The staleness rule is the whole reason this pipeline exists: a number nobody
    re-measured must fail loudly rather than keep looking authoritative."""
    max_age = payload['policy']['max_age_days']
    today = datetime.date.today()
    overdue = []
    for run_id, run in sorted(payload['runs'].items()):
        measured = datetime.datetime.strptime(run['measured_at'], '%Y-%m-%d').date()
        age = (today - measured).days
        if age > max_age:
            overdue.append('{} measured {} days ago (limit {})'.format(run_id, age, max_age))
    assert not overdue, (
        'Re-measure in pyg-bench and republish published/docs-numbers.json: '
        + '; '.join(overdue))


def test_every_number_the_docs_reference_is_published(payload):
    cells = payload['cells']
    missing = ['{}: {}'.format(os.path.relpath(path, DOCS_DIR), key)
               for path, key, _ in _references() if key not in cells]
    assert not missing, 'the docs reference numbers pyg-bench does not publish: ' + '; '.join(missing)


def test_every_reference_uses_the_role_its_quotability_allows(payload):
    cells = payload['cells']
    wrong = []
    for path, key, diagnostic in _references():
        cell = cells.get(key)
        if cell is None:
            continue
        quotable = cell['board_quotable'] is True
        if quotable and diagnostic:
            wrong.append('{}: {} is a published result, use :bench:'.format(path, key))
        if not quotable and not diagnostic:
            wrong.append('{}: {} is diagnostic-only, use :bench-diag:'.format(path, key))
    assert not wrong, '; '.join(wrong)


def test_a_board_quotable_cell_that_is_not_comparable_is_rejected(payload, contract):
    broken = json.loads(json.dumps(payload))
    key = sorted(broken['cells'])[0]
    broken['cells'][key]['board_quotable'] = True
    broken['cells'][key]['comparison_allowed'] = False
    with pytest.raises(bench.BenchDataError) as excinfo:
        bench.reverify(broken, contract)
    assert 'not comparison_allowed' in str(excinfo.value)


def test_a_run_missing_provenance_is_rejected(payload, contract):
    broken = json.loads(json.dumps(payload))
    run_id = sorted(broken['runs'])[0]
    del broken['runs'][run_id]['host']
    with pytest.raises(bench.BenchDataError) as excinfo:
        bench.reverify(broken, contract)
    assert "missing provenance 'host'" in str(excinfo.value)


def test_a_caveated_number_without_its_caveat_is_rejected(payload, contract):
    broken = json.loads(json.dumps(payload))
    key = sorted(broken['cells'])[0]
    broken['cells'][key]['status'] = 'partial'
    broken['cells'][key]['board_quotable'] = False
    broken['cells'][key]['comparison_allowed'] = False
    broken['cells'][key]['disclosures'] = []
    with pytest.raises(bench.BenchDataError) as excinfo:
        bench.reverify(broken, contract)
    assert 'carries no disclosure' in str(excinfo.value)


def _use(key, diagnostic=False, today=None):
    """Reference a benchmark key the way a page does, and report what broke.

    Deliberately does not go through Sphinx: the decision lives in the stdlib-only
    module precisely so it is checked in every lane, not just the docs build.
    """
    state = bench.State(bench.load(bench.BENCHMARKS_JSON), today or datetime.date.today())
    bench.check_reference(state, key, 'gfql/example', 1, diagnostic)
    return state.problems


def test_a_key_that_is_not_published_fails_the_build(payload):
    problems = _use('pagerank.twitter.this_was_never_measured')
    assert problems and 'no published benchmark number' in problems[0]


def test_a_published_key_renders_cleanly(payload):
    assert _use(sorted(payload['cells'])[0]) == []


def test_a_number_older_than_the_policy_fails_the_build(payload):
    """The failure the withdrawn figures needed and did not have."""
    oldest = min(
        datetime.datetime.strptime(run['measured_at'], '%Y-%m-%d').date()
        for run in payload['runs'].values())
    much_later = oldest + datetime.timedelta(days=payload['policy']['max_age_days'] + 400)
    problems = _use(sorted(payload['cells'])[0], today=much_later)
    assert problems and 'max_age_days' in problems[0]


def test_a_diagnostic_only_number_cannot_be_printed_as_a_result(payload):
    diagnostic = [key for key, cell in sorted(payload['cells'].items())
                  if cell['board_quotable'] is not True]
    if not diagnostic:
        pytest.skip('the artifact currently publishes no diagnostic-only cell')
    problems = _use(diagnostic[0])
    assert problems and 'not board-quotable' in problems[0]
    assert _use(diagnostic[0], diagnostic=True) == []


def test_an_artifact_from_another_contract_version_is_rejected(payload, contract):
    broken = json.loads(json.dumps(payload))
    broken['contract_version'] = int(contract['contract_version']) + 1
    with pytest.raises(bench.BenchDataError) as excinfo:
        bench.reverify(broken, contract)
    assert 'contract_version' in str(excinfo.value)


def test_every_chart_matches_the_published_numbers():
    """A chart is a number too.

    The withdrawn figures outlived their withdrawal on this page because they were
    baked into an SVG as glyph paths, where no check could see them. The charts are
    now rendered from the artifact, so this re-renders them and fails on any drift.
    Regenerate with ``python3 docs/source/_ext/gfql_bench_charts.py --write``.
    """
    stale = []
    for name, svg in charts.rendered().items():
        path = os.path.join(charts.CHART_DIR, name)
        if not os.path.exists(path):
            stale.append('{} is missing'.format(name))
            continue
        with open(path, encoding='utf-8') as handle:
            if handle.read() != svg:
                stale.append('{} no longer matches the published numbers'.format(name))
    assert not stale, (
        'Regenerate: python3 docs/source/_ext/gfql_bench_charts.py --write — '
        + '; '.join(stale))


def test_every_chart_draws_only_published_cells(payload):
    """The charts obey the same rule the prose does: published cells, or nothing."""
    referenced = sorted({
        key
        for chart in charts.CHARTS.values()
        for bar in chart.bars
        for key in (bar.value, bar.ratio)
        if key is not None})
    assert referenced, 'the charts reference no benchmark cell at all'
    missing = [key for key in referenced if key not in payload['cells']]
    assert not missing, 'charts draw numbers pyg-bench does not publish: ' + '; '.join(missing)


def test_a_chart_over_an_unpublished_cell_fails(payload):
    """The failure the SVGs needed and did not have."""
    broken = json.loads(json.dumps(payload))
    name = sorted(charts.CHARTS)[0]
    drawn = next(bar.value for bar in charts.CHARTS[name].bars if bar.value)
    del broken['cells'][drawn]
    with pytest.raises(charts.ChartError) as excinfo:
        charts.render(name, broken)
    assert 'does not publish' in str(excinfo.value)


def test_the_chart_renderer_stays_importable_without_sphinx():
    with open(os.path.join(SOURCE_DIR, '_ext', 'gfql_bench_charts.py'), encoding='utf-8') as f:
        source = f.read()
    for forbidden in ('docutils', 'sphinx', 'matplotlib'):
        assert 'import {}'.format(forbidden) not in source


def test_the_rules_module_stays_importable_without_sphinx():
    """CI caught the first draft: the rules lived in the docutils module, so the
    minimal lane could not import them and the whole gate ran in one job."""
    with open(os.path.join(SOURCE_DIR, '_ext', 'gfql_bench_data.py'), encoding='utf-8') as f:
        source = f.read()
    for forbidden in ('docutils', 'sphinx'):
        assert 'import {}'.format(forbidden) not in source
        assert 'from {}'.format(forbidden) not in source
