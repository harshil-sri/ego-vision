#!/usr/bin/env bash
# CI-style check: everything must pass before a package is "done".
# Usage: bash scripts/check_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== unit tests =="
python3 -m pytest tests/ -q --ignore=tests/equivalence

echo "== equivalence (reference) =="
python3 -m pytest tests/equivalence -q

echo "== registry switches cleanly =="
EGO_IMPL=core python3 -c "
import src.registry as r
try:
    r.get('metrics')
    print('ERROR: core/ should not exist yet')
    exit(1)
except ModuleNotFoundError as e:
    assert 'core.metrics' in str(e), e
    print('ok: core/ correctly reported missing')
"

echo "== recipe validation fails loudly on bad recipes =="
python3 - <<'EOF'
import tempfile, pathlib
from src.pipeline.recipe import load_recipe
bad = pathlib.Path(tempfile.mkdtemp()) / "bad.yaml"
bad.write_text("name: x\ningredients: [a]\nverbs: [take]\nsteps: [{verb: chop, ingredient: a}]\n")
try:
    load_recipe(bad)
    raise SystemExit("ERROR: bad recipe loaded")
except ValueError as e:
    assert "chop" in str(e)
    print("ok: bad recipe rejected:", str(e)[:60])
EOF

echo "== app boots headless (synthetic source) =="
python3 -m scripts.run_app --mode keyboard --synthetic --port 8901 &
APP=$!
sleep 6
curl -sf http://127.0.0.1:8901/api/state > /dev/null && echo "ok: /api/state"
curl -sf --max-time 2 -o /dev/null http://127.0.0.1:8901/video/stream && echo "ok: /video/stream"
kill $APP 2>/dev/null || true
wait $APP 2>/dev/null || true

echo
echo "ALL CHECKS PASSED"
