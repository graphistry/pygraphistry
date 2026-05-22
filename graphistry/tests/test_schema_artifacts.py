import json
import runpy
import sys
from pathlib import Path

from graphistry.devschemas import export as schema_export
from graphistry.devschemas.export import SCHEMA_FILENAMES, build_schemas, export_schemas
from graphistry.io.types import PLOTTABLE_SIMPLE_ENCODING_BIND_KEYS
from graphistry.viz_settings import (
    APPLY_ENCODINGS_REACT_KEYS,
    REACT_SETTING_NAMES,
    URL_PARAM_NAMES,
)


SCHEMA_DIR = Path(__file__).resolve().parents[2] / "schemas"


def _load(name: str):
    with (SCHEMA_DIR / SCHEMA_FILENAMES[name]).open(encoding="utf-8") as f:
        return json.load(f)


def test_committed_schema_artifacts_match_exporter():
    assert _load("url_params") == build_schemas()["url_params"]
    assert _load("react_settings") == build_schemas()["react_settings"]
    assert _load("encodings") == build_schemas()["encodings"]


def test_settings_schema_keys_match_public_contracts():
    assert set(_load("url_params")["properties"]) == set(URL_PARAM_NAMES)
    assert set(_load("react_settings")["properties"]) == set(REACT_SETTING_NAMES)
    assert (
        _load("react_settings")["x-graphistry"]["apply_encodings_react_keys"]
        == list(APPLY_ENCODINGS_REACT_KEYS)
    )


def test_encodings_schema_simple_keys_match_io_contract():
    simple_props = _load("encodings")["$defs"]["metadataEncodings"]["properties"]
    assert {k for k in simple_props if k != "complex_encodings"} == set(PLOTTABLE_SIMPLE_ENCODING_BIND_KEYS)
    assert "node_encodings" in _load("encodings")["$defs"]["nodeEdgeEncodingsPayload"]["properties"]


def test_schema_exporter_writes_and_checks_artifacts(tmp_path, capsys):
    assert export_schemas(tmp_path) is True
    assert export_schemas(tmp_path, check=True) is True

    (tmp_path / SCHEMA_FILENAMES["url_params"]).write_text("{}\n", encoding="utf-8")

    assert export_schemas(tmp_path, check=True) is False
    assert "schema drift:" in capsys.readouterr().out


def test_schema_exporter_cli_main(tmp_path, monkeypatch):
    export_schemas(tmp_path)
    monkeypatch.setattr(sys, "argv", ["export.py", "--output-dir", str(tmp_path), "--check"])

    assert schema_export.main() == 0


def test_schema_exporter_module_entrypoint(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["export.py", "--output-dir", str(tmp_path)])

    try:
        runpy.run_path(str(Path(schema_export.__file__)), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
    else:
        raise AssertionError("module entrypoint did not exit")
