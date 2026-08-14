"""Full Disclosure wrapper that adds Cardiox-style interval measurement."""

from __future__ import annotations

import numpy as np

try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy
except ImportError:
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QFrame, QLabel, QSizePolicy

try:
    from .theme import COL_GREEN, COL_GREEN_DRK
except ImportError:
    from ecg.holter.theme import COL_GREEN, COL_GREEN_DRK

try:
    from .ecg_calculations import calculate_all_ecg_metrics
except ImportError:
    from ecg.ecg_calculations import calculate_all_ecg_metrics

try:
    from .holter_full_disclosure import (
        HolterFullDisclosureDialog as _BaseHolterFullDisclosureDialog,
        HolterToolHandlers as _BaseHolterToolHandlers,
    )
except ImportError:
    from ecg.holter.holter_full_disclosure import (
        HolterFullDisclosureDialog as _BaseHolterFullDisclosureDialog,
        HolterToolHandlers as _BaseHolterToolHandlers,
    )


class HolterFullDisclosureDialog(_BaseHolterFullDisclosureDialog):
    """Wrap the stock Full Disclosure dialog and expose live interval metrics."""

    def __init__(self, replay_engine, parent=None):
        self._interval_metric_instance_id = f"full_disclosure:{id(self)}"
        self.lbl_intervals = None
        super().__init__(replay_engine, parent)

    def _build_ui(self):
        super()._build_ui()
        self._install_interval_label()

    def _install_interval_label(self):
        if getattr(self, "lbl_intervals", None) is not None:
            return

        undo_btn = getattr(self, "btn_undo", None)
        if undo_btn is None:
            return

        top_bar = undo_btn.parentWidget()
        top_layout = top_bar.layout() if top_bar is not None else None
        if top_layout is None:
            return

        self.lbl_intervals = QLabel("PR: -- ms | QRS: -- ms | QT: -- ms | QTc: -- ms")
        self.lbl_intervals.setStyleSheet(
            f"color: {COL_GREEN}; font-weight: bold; font-size: 13px;"
        )
        self.lbl_intervals.setAlignment(Qt.AlignVCenter | Qt.AlignLeft)
        self.lbl_intervals.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        self.lbl_intervals.setMinimumWidth(340)

        separator = QFrame(top_bar)
        separator.setFrameShape(QFrame.VLine)
        separator.setStyleSheet(f"color: {COL_GREEN_DRK};")

        undo_index = top_layout.indexOf(undo_btn)
        if undo_index < 0:
            undo_index = top_layout.count()

        top_layout.insertWidget(undo_index, self.lbl_intervals)
        top_layout.insertWidget(undo_index + 1, separator)

    def _update_canvases(self, start_sec: float, update_extras: bool = True):
        super()._update_canvases(start_sec, update_extras=update_extras)
        if update_extras:
            self._update_interval_labels_from_window()

    def _update_interval_labels_from_window(self):
        label = "PR: -- ms | QRS: -- ms | QT: -- ms | QTc: -- ms"

        try:
            if not hasattr(self, "_reader") or not hasattr(self, "_engine"):
                self.lbl_intervals.setText(label)
                return

            fs = float(getattr(self._engine, "fs", 500.0) or 500.0)
            end_sec = min(self._current_start + self._window_sec, float(self._engine.duration_sec))
            data = self._reader.read_range(self._current_start, end_sec)
            if data is None or not isinstance(data, np.ndarray) or data.size == 0:
                self.lbl_intervals.setText(label)
                return

            if data.ndim == 1:
                lead_ii = np.asarray(data, dtype=float)
            else:
                lead_idx = 1 if data.shape[0] > 1 else 0
                lead_ii = np.asarray(data[lead_idx], dtype=float)

            if lead_ii.size < max(200, int(fs * 0.5)):
                self.lbl_intervals.setText(label)
                return

            metrics = calculate_all_ecg_metrics(
                lead_ii,
                fs=fs,
                instance_id=self._interval_metric_instance_id,
            )

            def fmt(value):
                try:
                    if value is None:
                        return "-- ms"
                    ivalue = int(round(float(value)))
                    return f"{ivalue} ms" if ivalue > 0 else "-- ms"
                except Exception:
                    return "-- ms"

            label = (
                f"PR: {fmt(metrics.get('pr_interval'))} | "
                f"QRS: {fmt(metrics.get('qrs_duration'))} | "
                f"QT: {fmt(metrics.get('qt_interval'))} | "
                f"QTc: {fmt(metrics.get('qtc_interval'))}"
            )
        except Exception as exc:
            print(f"[Full Disclosure] Interval metric error: {exc}")

        if getattr(self, "lbl_intervals", None) is not None:
            self.lbl_intervals.setText(label)


class HolterToolHandlers(_BaseHolterToolHandlers):
    """Tool handlers that route Full Disclosure through the interval wrapper."""

    @staticmethod
    def handle_full_disclosure(parent):
        """Open Full Disclosure dialog with live interval calculations."""
        if hasattr(parent, "_replay_engine") and parent._replay_engine:
            dialog = HolterFullDisclosureDialog(parent._replay_engine, parent)
            dialog.exec_()
        else:
            print("[Holter] Full Disclosure unavailable: no replay engine.")
