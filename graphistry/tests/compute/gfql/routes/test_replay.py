"""The existing-suite replay is a failing gate and includes every route switch."""
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

from graphistry.tests.compute.gfql.routes.switch import ROUTES


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("workers", [None, "2"])
def test_replay_broadcasts_existing_tests_and_propagates_failures(tmp_path, failure, workers):
    root = Path(__file__).resolve().parents[5]
    fake = tmp_path / "python"
    # Use the real registry import, replacing only pytest execution with a recorder.
    fake.write_text(
        "#!/bin/bash\n"
        f'if [ "$1" = "-c" ]; then exec "{sys.executable}" "$@"; fi\n'
        'echo "$GFQL_ROUTES_OFF|$*" >> "$REPLAY_CALLS"\n'
        'if [ "$REPLAY_FAIL" = 1 ] && [ "$GFQL_ROUTES_OFF" = point-rows ]; then exit 2; fi\n'
        'echo "1 passed"\n'
    )
    fake.chmod(0o755)
    calls = tmp_path / "calls"
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "OUT": str(tmp_path / "logs"), "REPLAY_CALLS": str(calls),
           "REPLAY_FAIL": str(int(failure)),
           "SUITES": "graphistry/tests/compute/test_chain.py"}
    env.pop("MODES", None)
    env.pop("ROUTES_XDIST", None)
    if workers is not None:
        env["ROUTES_XDIST"] = workers
    result = subprocess.run(["bash", str(root / "bin/test-routes-off.sh")], env=env,
                            cwd=root, text=True, capture_output=True)
    assert result.returncode == int(failure), result.stderr
    records = [line.split("|", 1) for line in calls.read_text().splitlines()]
    assert [record[0] for record in records] == [*ROUTES, ",".join(ROUTES)]
    assert all("graphistry/tests/compute/test_chain.py" in record[1] for record in records)
    assert all(("-n 2" in record[1]) == (workers is not None) for record in records)
    assert all((tmp_path / "logs" / f"{mode}.log").is_file() for mode in (*ROUTES, "all-off"))


def _hosted_replay_job() -> str:
    root = Path(__file__).resolve().parents[5]
    workflow = (root / ".github/workflows/ci.yml").read_text()
    job = re.search(r"^  gfql-routes-off:\n(.*?)(?=^  [\w-]+:|\Z)", workflow, re.M | re.S)
    assert job is not None, "The hosted existing-suite replay job is missing"
    return job.group(1)


def test_hosted_replay_matrix_covers_every_registered_route():
    job = _hosted_replay_job()
    matrix = re.search(r"^        mode: \[(.*?)\]$", job, re.M)
    assert matrix is not None, "Update this guard if the workflow matrix representation changes"
    modes = [mode.strip() for mode in matrix.group(1).split(",")]
    assert sorted(modes) == sorted([*ROUTES, "all-off"])


def test_hosted_replay_failures_fail_the_workflow():
    # The shell runner's nonzero exit must not be suppressed at job or step scope.
    allowances = re.findall(r"^\s+continue-on-error:\s*(.*?)\s*$", _hosted_replay_job(), re.M)
    assert all(value == "false" for value in allowances), allowances


def test_route_recorder_preserves_chain_introspection():
    from graphistry.compute.chain import chain, _try_chain_fast_path
    assert hasattr(chain, "__wrapped__")
    assert not hasattr(_try_chain_fast_path, "__wrapped__")


def test_route_engagement_manifest_selects_affected_tests_and_rejects_stale_collection(tmp_path):
    root = Path(__file__).resolve().parents[5]
    affected = "graphistry/tests/compute/test_gfql.py::TestGFQL::test_gfql_deprecation_and_migration"
    unaffected = "graphistry/tests/compute/test_gfql.py::TestGFQLAPI::test_public_api_methods"
    introspection = (
        "graphistry/tests/compute/gfql/routes/test_replay.py"
        "::test_route_recorder_preserves_chain_introspection"
    )
    tests = [affected, unaffected, introspection]
    plugin = "graphistry.tests.compute.gfql.routes.engagement_manifest"
    args = [sys.executable, "-m", "pytest", *tests, "-q", "-p", plugin,
            "-p", "no:cacheprovider", "-o", "addopts="]
    env = {key: value for key, value in os.environ.items()
           if key not in {"GFQL_ROUTES_OFF", "GFQL_ROUTE_SELECT_MANIFEST", "GFQL_ROUTE_RECORD_DIR"}}
    record_dir = tmp_path / "records"
    baseline = subprocess.run(args, cwd=root, env={**env, "GFQL_ROUTE_RECORD_DIR": str(record_dir)},
                              text=True, capture_output=True)
    assert baseline.returncode == 0, baseline.stdout + baseline.stderr
    manifest = tmp_path / "manifest.json"
    build = subprocess.run([sys.executable, "-m", "bin.ci.build_route_manifest",
                            str(record_dir), str(manifest), "--workers", "1"],
                           cwd=root, env=env, text=True, capture_output=True)
    assert build.returncode == 0, build.stdout + build.stderr
    selected = subprocess.run(args, cwd=root,
                              env={**env, "GFQL_ROUTES_OFF": "point-rows",
                                   "GFQL_ROUTE_SELECT_MANIFEST": str(manifest)},
                              text=True, capture_output=True)
    assert selected.returncode == 0, selected.stdout + selected.stderr
    assert "1 passed, 2 deselected" in selected.stdout
    stale = subprocess.run([*args[:3], affected, *args[5:]], cwd=root,
                           env={**env, "GFQL_ROUTES_OFF": "point-rows",
                                "GFQL_ROUTE_SELECT_MANIFEST": str(manifest)},
                           text=True, capture_output=True)
    assert stale.returncode != 0
    assert "does not match this test collection" in stale.stdout + stale.stderr
