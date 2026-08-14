"""Import hook that patches Full Disclosure with interval measurement."""

from __future__ import annotations

import importlib.abc
import importlib.machinery
import sys
from pathlib import Path

_src = Path(__file__).resolve().parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))


def _patch_full_disclosure(module):
    try:
        def _patched_handle_full_disclosure(parent):
            if hasattr(parent, "_replay_engine") and parent._replay_engine:
                from ecg.holter.holter_full_disclosure_patch import HolterFullDisclosureDialog
                dialog = HolterFullDisclosureDialog(parent._replay_engine, parent)
                dialog.exec_()
            else:
                print("[Holter] Full Disclosure unavailable: no replay engine.")

        module.HolterToolHandlers.handle_full_disclosure = staticmethod(_patched_handle_full_disclosure)
    except Exception:
        pass


class _FullDisclosurePatchLoader(importlib.abc.Loader):
    def __init__(self, wrapped_loader):
        self._wrapped_loader = wrapped_loader

    def create_module(self, spec):
        if hasattr(self._wrapped_loader, "create_module"):
            return self._wrapped_loader.create_module(spec)
        return None

    def exec_module(self, module):
        self._wrapped_loader.exec_module(module)
        _patch_full_disclosure(module)


class _FullDisclosurePatchFinder(importlib.abc.MetaPathFinder):
    TARGET = "ecg.holter.holter_full_disclosure"

    def find_spec(self, fullname, path, target=None):
        if fullname != self.TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _FullDisclosurePatchLoader(spec.loader)
        return spec


if not any(isinstance(finder, _FullDisclosurePatchFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _FullDisclosurePatchFinder())
