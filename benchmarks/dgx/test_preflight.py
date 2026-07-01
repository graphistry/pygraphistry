"""Regression test for the dgx safety preflight estimator.

Guards the exact failure that crashed dgx-spark (2026-07-01): a 1.8B-edge
cudf/cugraph run whose ~87 GB peak exceeds the ~80 GB unified-memory budget must
be REFUSED, while the #1658 handoff's 80M-edge run must be allowed.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight


def test_friendster_1p8b_is_refused():
    # 1.8B edges -> ~87 GB peak > budget -> must refuse (this is what crashed the box).
    assert preflight.peak_gb(1_806_067_135) > (preflight.HOST_GB - preflight.DEFAULT_FLOOR_GB)
    assert preflight.is_safe(1_806_067_135) is False


def test_handoff_80m_is_allowed():
    # 80M edges -> ~3.8 GB peak -> safely allowed (index_takeover_bench largest N).
    assert preflight.is_safe(80_000_000) is True


def test_peak_scales_with_edges_and_cugraph_factor():
    assert preflight.peak_gb(1_000_000_000) > preflight.peak_gb(100_000_000)
    assert preflight.peak_gb(100_000_000, cugraph_factor=3.0) > preflight.peak_gb(100_000_000, cugraph_factor=1.0)
