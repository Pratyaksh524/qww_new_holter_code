"""ECG package bootstrap hooks."""

from __future__ import annotations

import sys

try:
    from .holter import holter_full_disclosure_patch as _holter_full_disclosure_patch
    sys.modules["ecg.holter.holter_full_disclosure"] = _holter_full_disclosure_patch
except Exception:
    pass

