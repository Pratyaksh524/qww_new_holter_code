"""Startup hook that routes Full Disclosure through the interval-aware wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
_src = _root / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

try:
    from ecg.holter import holter_full_disclosure_patch as _patch
    sys.modules["ecg.holter.holter_full_disclosure"] = _patch
except Exception:
    pass
