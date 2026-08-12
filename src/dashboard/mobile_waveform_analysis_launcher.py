"""Convenience launcher for CardioX waveform analysis by mobile number.

The existing `ECGAnalysisWindow` already supports:
 - loading public reports by mobile number
 - selecting one of the matching reports
 - plotting the waveform
 - showing PR, QRS, QT, QTc, and QTcF metrics

This module adds a small, dedicated entrypoint so the feature can be opened
directly from CardioX or run as a standalone script.
"""

from __future__ import annotations

import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMessageBox

try:
    from dashboard.analysis_window import ECGAnalysisWindow
except Exception:  # pragma: no cover - fallback for alternate package layouts
    from src.dashboard.analysis_window import ECGAnalysisWindow  # type: ignore


def launch(mobile_no: str = ""):
    """Open the standard waveform analysis window and optionally pre-load a mobile number."""
    app = QApplication.instance() or QApplication(sys.argv)
    window = ECGAnalysisWindow()
    window.show()

    if mobile_no:
        def _prefill_and_load():
            try:
                if hasattr(window, "mobile_no_input"):
                    window.mobile_no_input.setText(mobile_no)
                if hasattr(window, "load_mobile_reports"):
                    window.load_mobile_reports()
                elif hasattr(window, "load_reports"):
                    window.load_reports()
            except Exception as exc:
                QMessageBox.warning(window, "Waveform Analysis", f"Could not preload mobile reports:\n{exc}")

        QTimer.singleShot(0, _prefill_and_load)

    return app, window


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="CardioX mobile waveform analysis launcher")
    parser.add_argument(
        "--mobile",
        default="",
        help="Optional mobile number to auto-load when the window opens.",
    )
    args = parser.parse_args(argv)

    app, _window = launch(args.mobile.strip())
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
