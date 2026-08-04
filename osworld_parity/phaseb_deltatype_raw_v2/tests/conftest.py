from __future__ import annotations

import os
import sys
from pathlib import Path


# The five hash-pinned implementation modules import each other as flat siblings
# (``from action_v2 import ...``). That is how the sealed build executed them
# (``python .../build.py`` puts their directory on ``sys.path[0]``) and their
# bytes are frozen, so the package layout reproduces that import root instead of
# rewriting the imports. See ../README-scope note in osworld_parity/README.md.
ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def repo_relative(*parts: str) -> Path:
    return REPO.joinpath(*parts)


def external_root(env_var: str, default: str) -> Path:
    """Cluster-local dataset/rollout root: env override, documented default."""
    return Path(os.environ.get(env_var, default))
