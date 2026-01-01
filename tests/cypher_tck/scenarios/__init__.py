from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_SCENARIO_ROOT = Path(__file__).resolve().parent / "tck" / "features"
SCENARIOS = []

for path in sorted(_SCENARIO_ROOT.rglob("*.py"), key=lambda p: p.as_posix()):
    module_name = "tests.cypher_tck.scenarios." + path.relative_to(Path(__file__).resolve().parent).with_suffix("").as_posix().replace("/", ".")
    spec = spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        continue
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    SCENARIOS.extend(getattr(module, "SCENARIOS", []))
