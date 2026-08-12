from __future__ import annotations

import gc
import json
import math
import os
import shutil
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np

from PyQt5.QtCore import QEvent, QPoint, QPointF, QRect, QThread, QTimer, Qt, QObject, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter, QPainterPath, QPalette, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QAbstractItemView, QButtonGroup, QComboBox, QDialog, QDoubleSpinBox, QFileDialog,
    QFrame, QGridLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox,
    QProgressBar, QPushButton, QScrollArea, QScrollBar, QSlider, QSizePolicy, QSpinBox, QSplitter,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QVBoxLayout,
    QWidget, QInputDialog
)

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False

from ..theme import (
    ADC_TO_MV, COL_BEAT_S, COL_BG, COL_BLACK, COL_BTN_ACTIVE_BG, COL_BTN_ACTIVE_TEXT, COL_DARK,
    COL_GRAY, COL_GREEN, COL_GREEN_DRK, COL_GREEN_MID, COL_GRID_MAJOR, COL_GRID_MINOR, COL_RED,
    COL_TEXT, COL_TIMESTAMP, COL_WAVE_ORANGE, COL_WAVE_RED, COL_WHITE, COL_YELLOW, GAINS,
    PAPER_SPEEDS, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT,
)
from ..holter_helpers import (
    UI_ACCENT, UI_ACCENT_HOVER, UI_BG, UI_BORDER, UI_CARD, UI_MUTED, UI_PANEL, UI_PANEL_ALT,
    UI_SUCCESS, UI_TEXT, UI_WARNING, _class_matches_filter, _find_latest_completed_session,
    _format_system_time, _get_recording_start_end_times, _metrics_duration_sec,
    _normalize_beat_class, _normalize_patient_info, _resolve_recordings_dir, _sec_to_hms,
    _style_active_btn, _style_btn, _table_style, _template_filter_key,
)

from .holter_widgets import ECGStripCanvas, HistogramCanvas, LorenzCanvas, MagnifierOverlay, STCanvas, STTMarkerCanvas

# 12. HOLTER AF ANALYSIS PANEL
class HolterAFPanel(QWidget):
    def __init__(self, parent=None, session_dir: str = ""):
        super().__init__(parent)
        self.session_dir = session_dir
        self.setStyleSheet(f"background:{COL_BG};")
        self._build_ui()

    def _find_template_host(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "_show_template_card_menu"):
                return parent
            parent = parent.parentWidget()
        window = self.window()
        if window is not None and hasattr(window, "_show_template_card_menu"):
            return window
        return None
    def _build_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Left: AF event list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        title = QLabel("AF Analysis")
        title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:6px 10px;border-radius:6px;")
        left_layout.addWidget(title)

        cols = ["Start time", "Duration", "Type"]
        self._af_table = QTableWidget(0, len(cols))
        self._af_table.setHorizontalHeaderLabels(cols)
        self._af_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._af_table.setStyleSheet(_table_style())
        self._af_table.verticalHeader().setVisible(False)
        self._af_table.setEditTriggers(QTableWidget.NoEditTriggers)
        left_layout.addWidget(self._af_table, 1)

        no_items = QLabel("There are no items to show.")
        no_items.setStyleSheet(f"color:{COL_GREEN_DRK};font-style:italic;padding:8px;border:none;")
        self._no_items_lbl = no_items
        left_layout.addWidget(no_items)

        nav_row = QHBoxLayout()
        for lbl in ["AF Analysis", "Parameters", "Prev Event", "Next Event", "Remove All", "Remove"]:

            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        layout.addWidget(left, 2)

        # Right: ECG strip + Lorenz
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background:transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(4)
        
        self._thumb_strips = []
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            lbl = QLabel(f"Lead {lead_names[i]}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            scroll_layout.addWidget(lbl)
            
            # Disable vertical lines and labels for AF Analysis panel
            strip = ECGStripCanvas(height=120, show_vertical_lines=False)
            strip.lead_name = lead_names[i]
            scroll_layout.addWidget(strip)
            self._thumb_strips.append(strip)
            
        scroll_area.setWidget(scroll_content)
        right_layout.addWidget(scroll_area, 1)

        # Disable vertical lines for bottom ECG strip too
        self._af_ecg_strip = ECGStripCanvas(height=70, show_vertical_lines=False)
        right_layout.addWidget(self._af_ecg_strip)

        self._af_lorenz = LorenzCanvas()
        self._af_lorenz.setFixedHeight(160)
        right_layout.addWidget(self._af_lorenz)
        layout.addWidget(right, 3)

    def update_from_metrics(self, metrics_list: list, duration_sec: float = 0):
        af_events = [(m['t'], m.get('arrhythmias', [])) for m in metrics_list
                     if any('AF' in a or 'Fibrill' in a for a in m.get('arrhythmias', []))]
        self._af_table.setRowCount(len(af_events))
        
        # Get the actual recording start time from session
        recording_start_timestamp = None
        if self.session_dir:
            try:
                ecgh_path = os.path.join(self.session_dir, 'recording.ecgh')
                if os.path.exists(ecgh_path):
                    # Get file modification time as the recording start time
                    recording_start_timestamp = os.path.getmtime(ecgh_path) - duration_sec
                else:
                    # Fallback: use current time minus duration
                    import time
                    recording_start_timestamp = time.time() - duration_sec
            except Exception as e:
                print(f"[HolterAFPanel] Error getting recording start time: {e}")
                import time
                recording_start_timestamp = time.time() - duration_sec
        
        if af_events:
            self._no_items_lbl.hide()
            for i, (t, arrhy) in enumerate(af_events):
                # Calculate actual system time if we have the recording start time
                if recording_start_timestamp:
                    from datetime import datetime
                    event_timestamp = recording_start_timestamp + t
                    actual_time_str = datetime.fromtimestamp(event_timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    actual_time_str = _sec_to_hms(t)
                
                for j, val in enumerate([actual_time_str, "30s", "AF/Af"]):
                    item = QTableWidgetItem(val)
                    item.setForeground(QColor(COL_WHITE))
                    self._af_table.setItem(i, j, item)
        else:
            self._no_items_lbl.show()

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        fs = 500.0
        n_samples = min(N, int(10 * fs))
        x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
        if n_samples > 0:
            lead2_idx = 1 if data.shape[0] > 1 else 0
            self._af_ecg_strip.set_data(x, data[lead2_idx, :n_samples].copy())
            for i, ts in enumerate(self._thumb_strips):
                if i < data.shape[0]:
                    ts.set_data(x, data[i, :n_samples].copy())


