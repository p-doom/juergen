#!/usr/bin/env bash
# Rebuild the sign-of-life SGLang serving environment.
#
#   ./rebuild-sign_of_life_eval_v2_venv.sh <target-dir> <juergen-repository>

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:?usage: $0 <target-dir> <juergen-repository>}"
SOURCE="${2:?usage: $0 <target-dir> <juergen-repository>}"
SOURCE_COMMIT=119b2e252b9a91ee4e15124b720daccfd1c9789b

if [ -e "$TARGET" ]; then
  echo "target already exists: $TARGET" >&2
  exit 2
fi
if ! git -C "$SOURCE" cat-file -e "$SOURCE_COMMIT^{commit}" 2>/dev/null; then
  echo "source repository does not contain $SOURCE_COMMIT" >&2
  exit 2
fi

snapshot="$(mktemp -d -t sign-of-life-venv-source-XXXXXX)"
trap 'rm -rf "$snapshot"' EXIT
git -C "$SOURCE" archive "$SOURCE_COMMIT" | tar -x -C "$snapshot"

uv venv --no-config --python 3.12.11 "$TARGET"
# The live set overrides sglang's OpenAI pin; every dependency is explicit here.
# The PyTorch index mirrors PyPI names, so uv must consider both for exact pins.
uv pip install --no-config --python "$TARGET/bin/python" \
  --index https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match --no-deps \
  -r "$HERE/sign_of_life_eval_v2_venv.requirements.txt"
uv pip install --no-config --python "$TARGET/bin/python" \
  --no-build-isolation --no-deps \
  "$snapshot" "$snapshot/data_pipeline" "$snapshot/eval"

"$TARGET/bin/python" - <<'PY'
import hashlib
from importlib.metadata import distributions

installed = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in distributions())
digest = hashlib.sha256("\n".join(installed).encode()).hexdigest()
expected = "d68a25f22ce9baaaf1dbcb2d0c13c47af4187726a8fcad2cc543c95103077cd9"
if len(installed) != 344 or digest != expected:
    raise SystemExit(f"rebuilt set mismatch: {len(installed)} distributions, sha256 {digest}")
print(f"rebuilt set: {len(installed)} distributions, sha256 {digest}")
PY
"$TARGET/bin/python" -VV
