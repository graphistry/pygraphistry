"""Draw the benchmark charts from the numbers pyg-bench publishes.

The charts on ``gfql/benchmark_filter_pagerank.rst`` are static SVGs, but they are not
hand-drawn: every bar length and every figure printed on them comes from a cell of
``docs/source/_data/gfql_benchmarks.json``, through the same ``format_cell`` the
``:bench:`` role uses. ``docs/test_bench_numbers.py`` re-renders them and fails if a
committed file differs, so a chart cannot go on asserting a number the artifact no longer
publishes - which is exactly how withdrawn figures survived on this page as glyph paths.

Regenerate after vendoring a new artifact::

    python3 docs/source/_ext/gfql_bench_charts.py --write

Deliberately stdlib-only, for the same reason ``gfql_bench_data`` is: the sync check has
to run in the ordinary test lane, which has neither Sphinx nor matplotlib. Hand-written
SVG is also byte-reproducible, which a plotting library's output is not, and it can carry
a ``prefers-color-scheme`` block so the charts read in dark mode too.
"""

from __future__ import annotations

import argparse
import os
from collections import OrderedDict
from collections.abc import Sequence
from typing import NamedTuple

from gfql_bench_data import BENCHMARKS_JSON, JSONObject, format_cell, load

#: Where the committed charts live.
CHART_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'gfql', '_static', 'filter_pagerank')

WIDTH = 720
PAD = 16
HEADER_H = 74
ROW_H = 62
BAR_H = 24
BAR_MAX = 424
FOOT_H = 30
MIN_BAR = 5.0
RADIUS = 4.0

#: Categorical slots 1/2/3 of the validated default palette, light and dark steps. Three
#: slots clear every all-pairs colour gate in both modes; each bar also carries its own
#: name and its own number, which is what the aqua slot's sub-3:1 light contrast requires.
STYLE = (
    '.c{--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--ink3:#78776f;'
    '--rule:#dcdbd6;--track:#f0efec;--neo:#eb6834;--cpu:#2a78d6;--gpu:#1baf7a}'
    '@media (prefers-color-scheme:dark){'
    '.c{--surface:#1a1a19;--ink:#ffffff;--ink2:#c3c2b7;--ink3:#908f86;'
    '--rule:#3b3a37;--track:#262523;--neo:#d95926;--cpu:#3987e5;--gpu:#199e70}}'
    '.c text{font-family:system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,'
    'sans-serif}'
    '.t{font-size:17px;font-weight:600;fill:var(--ink)}'
    '.s{font-size:12.5px;fill:var(--ink2)}'
    '.n{font-size:13px;font-weight:600;fill:var(--ink)}'
    '.a{font-size:12.5px;fill:var(--ink2)}'
    '.m{font-size:12.5px;fill:var(--ink3)}'
    '.neo{fill:var(--neo)}.cpu{fill:var(--cpu)}.gpu{fill:var(--gpu)}'
)


class Bar(NamedTuple):
    """One row of a chart. ``value`` and ``ratio`` name published cells, never numbers."""

    label: str
    tone: str
    value: str | None
    ratio: str | None = None
    note: str = ''


class Chart(NamedTuple):
    title: str
    subtitle: str
    bars: Sequence[Bar]
    foot: str


CHARTS: dict[str, Chart] = OrderedDict((
    ('twitter_pipeline.svg', Chart(
        title='Twitter: 81,306 nodes / 2.4M edges',
        subtitle='Warm pipeline \u2014 search, PageRank, search \u2014 on the resident '
                 'graph. Lower is better.',
        bars=(
            Bar('Neo4j + GDS', 'neo', 'pagerank.twitter.neo4j_gds'),
            Bar('GFQL Cypher on CPU (pandas + igraph)', 'cpu',
                'pagerank.twitter.gfql_cpu'),
            Bar('GFQL Cypher on GPU (cuDF + cuGraph)', 'gpu',
                'pagerank.twitter.gfql_gpu', 'pagerank.twitter.gfql_gpu_vs_gfql_cpu',
                'faster than the GFQL CPU path'),
        ),
        foot='Direct timings use different profiles; only the GFQL GPU/CPU ratio is valid.',
    )),
    ('gplus_pipeline.svg', Chart(
        title='GPlus: 107,614 nodes / 30M edges',
        subtitle='Warm pipeline \u2014 search, PageRank, search \u2014 on the resident '
                 'graph. Lower is better.',
        bars=(
            Bar('Neo4j + GDS', 'neo', None, None, 'did not complete \u2014 no timing exists'),
            Bar('GFQL Cypher on CPU (pandas + igraph)', 'cpu', 'pagerank.gplus.gfql_cpu'),
            Bar('GFQL Cypher on GPU (cuDF + cuGraph)', 'gpu', 'pagerank.gplus.gfql_gpu',
                'pagerank.gplus.gfql_gpu_vs_gfql_cpu', 'faster than the CPU path'),
        ),
        foot='Neo4j lost its Bolt connection mid-transaction on the first warmup '
             'iteration; no GPlus figure exists.',
    )),
))


class ChartError(Exception):
    """A chart asked for a number the artifact does not publish."""


def _esc(text: str) -> str:
    return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def _num(value: float) -> str:
    """Round to a tenth of a pixel, so the SVG is byte-identical everywhere."""
    text = '{:.1f}'.format(value)
    return text[:-2] if text.endswith('.0') else text


def _cell(payload: JSONObject, key: str) -> JSONObject:
    cells = payload.get('cells')
    cell = cells.get(key) if isinstance(cells, dict) else None
    if not isinstance(cell, dict):
        raise ChartError(
            '{} draws {!r}, which the artifact does not publish. A chart renders published '
            'cells and nothing else.'.format(os.path.basename(BENCHMARKS_JSON), key))
    return cell


def _seconds(payload: JSONObject, key: str) -> float:
    value = _cell(payload, key)['value']
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError('{!r} is not a number'.format(key))
    return float(value)


def _bar_path(x: float, y: float, width: float, height: float) -> str:
    """A bar rounded only at its data end, square against the baseline."""
    radius = min(RADIUS, width)
    return 'M{x} {y}H{a}a{r} {r} 0 0 1 {r} {r}V{b}a{r} {r} 0 0 1 -{r} {r}H{x}Z'.format(
        x=_num(x), y=_num(y), a=_num(x + width - radius), r=_num(radius),
        b=_num(y + height - radius))


def render(name: str, payload: JSONObject) -> str:
    """Render one chart to SVG text."""
    chart = CHARTS[name]
    lengths = [_seconds(payload, bar.value) for bar in chart.bars if bar.value]
    widest = max(lengths) if lengths else 1.0

    height = HEADER_H + ROW_H * len(chart.bars) + FOOT_H
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"'
        ' role="img" aria-label="{}">'.format(
            WIDTH, height, WIDTH, height, _esc(chart.title)),
        '<title>{}</title>'.format(_esc(chart.title)),
        '<style>{}</style>'.format(STYLE),
        '<g class="c">',
        '<rect width="{}" height="{}" fill="var(--surface)"/>'.format(WIDTH, height),
        '<text class="t" x="{}" y="28">{}</text>'.format(PAD, _esc(chart.title)),
        '<text class="s" x="{}" y="49">{}</text>'.format(PAD, _esc(chart.subtitle)),
        '<rect x="{}" y="{}" width="1" height="{}" fill="var(--rule)"/>'.format(
            PAD, HEADER_H, ROW_H * len(chart.bars) - 14),
    ]

    for index, bar in enumerate(chart.bars):
        row_top = HEADER_H + index * ROW_H
        bar_top = row_top + 20
        out.append('<text class="a" x="{}" y="{}">{}</text>'.format(
            PAD + 1, row_top + 11, _esc(bar.label)))

        if bar.value is None:
            out.append(
                '<rect x="{}" y="{}" width="{}" height="{}" rx="{}" fill="var(--track)"'
                ' stroke="var(--rule)" stroke-dasharray="5 4"/>'.format(
                    PAD + 0.5, bar_top + 0.5, BAR_MAX, BAR_H, _num(RADIUS)))
            out.append('<text class="m" x="{}" y="{}">{}</text>'.format(
                PAD + 12, bar_top + 17, _esc(bar.note)))
            continue

        width = max(MIN_BAR, BAR_MAX * _seconds(payload, bar.value) / widest)
        out.append('<path class="{}" d="{}"/>'.format(
            bar.tone, _bar_path(PAD, bar_top, width, BAR_H)))

        spans = ['<tspan class="n">{}</tspan>'.format(
            _esc(format_cell(_cell(payload, bar.value))))]
        if bar.ratio is not None:
            spans.append('<tspan class="a" dx="11">{}</tspan>'.format(
                _esc('{} {}'.format(format_cell(_cell(payload, bar.ratio)), bar.note))))
        out.append('<text x="{}" y="{}">{}</text>'.format(
            _num(PAD + width + 10), bar_top + 17, ''.join(spans)))

    out.append('<text class="m" x="{}" y="{}">{}</text>'.format(
        PAD, height - 11, _esc(chart.foot)))
    out.append('</g></svg>')
    return '\n'.join(out) + '\n'


def rendered(payload: JSONObject | None = None) -> dict[str, str]:
    """Every chart, keyed by file name, rendered from the vendored artifact."""
    data = payload if payload is not None else load(BENCHMARKS_JSON)
    return OrderedDict((name, render(name, data)) for name in CHARTS)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out-dir', default=CHART_DIR)
    parser.add_argument('--write', action='store_true',
                        help='rewrite the charts; without it, only report what is stale')
    args = parser.parse_args(argv)

    if args.write:
        os.makedirs(args.out_dir, exist_ok=True)

    stale: list[str] = []
    for name, svg in rendered().items():
        path = os.path.join(args.out_dir, name)
        current: str | None = None
        if os.path.exists(path):
            with open(path, encoding='utf-8') as handle:
                current = handle.read()
        if current == svg:
            continue
        stale.append(name)
        if args.write:
            with open(path, 'w', encoding='utf-8') as handle:
                handle.write(svg)
            print('[wrote] {}'.format(path))
    if args.write:
        return 0
    for name in stale:
        print('[stale] {}'.format(name))
    print('{} of {} charts differ from the published numbers'.format(len(stale), len(CHARTS)))
    return 1 if stale else 0


if __name__ == '__main__':
    raise SystemExit(main())
