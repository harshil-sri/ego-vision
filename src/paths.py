"""Data-path resolution across the two machines this repo has lived on.

Configs were written on WSL, where the data SSD was mounted at /mnt/ssd.
On native Windows the same tree lives at C:/ssd-data. Configs keep the WSL
spelling (one source of truth); this module translates at load time.

Order of precedence for a path that starts with /mnt/ssd/:
    1. EGO_DATA_ROOT environment variable, if set (replaces /mnt/ssd)
    2. on Windows: C:/ssd-data
    3. otherwise: unchanged
Any other path is returned unchanged.
"""

from __future__ import annotations

import os

WSL_PREFIX = "/mnt/ssd/"
WINDOWS_ROOT = "C:/ssd-data"


def resolve_data_path(path) -> str:
    """Translate a config path to this machine.

    Input: path (str or Path), e.g. "/mnt/ssd/epic-kitchens/EPIC-256".
    Output: str, e.g. "C:/ssd-data/epic-kitchens/EPIC-256" on Windows.
    """
    s = str(path).replace("\\", "/")
    if not s.startswith(WSL_PREFIX):
        return str(path)
    root = os.environ.get("EGO_DATA_ROOT")
    if root is None and os.name == "nt":
        root = WINDOWS_ROOT
    if root is None:
        return str(path)
    return root.rstrip("/\\") + "/" + s[len(WSL_PREFIX):]
