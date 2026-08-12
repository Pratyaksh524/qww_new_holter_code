"""
ecg/holter/holter_ui.py
========================
Complete Holter Monitor UI - Professional Medical Software
Matches reference images: black/green medical workstation style.

Screens:
  1. HolterStartDialog        - patient info + duration + start
  2. HolterStatusBar          - REC indicator, elapsed, live BPM, arrhythmia ticker
  3. HolterSummaryCards       - KPI cards (Avg HR, Min/Max, Beats, Pauses, Quality, SDNN)
  4. HolterOverviewPanel      - full stats table (Name/Value pairs)
  5. HolterHRVPanel           - HRV table per hour + bottom stats strip
  6. HolterReplayPanel        - RR scatter/Lorenz + scrub slider + ECG strip
  7. HolterEventsPanel        - Arrhythmia events list with strip nav
  8. HolterWaveGridPanel      - 12-lead live/replay grid (3 rows  -- 4 cols)
  9. HolterInsightPanel       - Comprehensive report preview narrative
 10. HolterRecordManagementPanel - searchable session browser
 11. HolterHistogramPanel     - RR-interval histogram
 12. HolterAFPanel            - AF episode browser
 13. HolterSTPanel            - ST tendency per channel
 14. HolterEditEventPanel     - Edit events with strip thumbnails
 15. HolterEditStripsPanel    - Edit strips (max HR, min HR, sinus max/min thumbnails)
 16. HolterReportTablePanel   - Hour-by-hour report table
 17. HolterMainWindow         - Orchestrates all panels in tabbed layout
"""

import os
import sys
import json
import time
import math
import shutil
import gc
from datetime import datetime, timedelta
from typing import Optional, List, Dict

import numpy as np

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QDialog, QLineEdit, QComboBox, QSlider, QGroupBox, QFrame,
    QTabWidget, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy, QScrollArea, QGridLayout, QSpinBox, QMessageBox,
    QFileDialog, QApplication, QProgressBar, QSplitter, QTextEdit, QInputDialog, QDoubleSpinBox,
    QAbstractItemView, QToolButton, QButtonGroup, QMenu, QScrollBar, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint, QPointF, QRect, QObject, QEvent
from PyQt5.QtGui import QFont, QColor, QPalette, QPainter, QPen, QBrush, QPixmap, QPainterPath

try:
    import pyqtgraph as pg
    HAS_PG = True
except Exception:
    pg = None
    HAS_PG = False


def _resolve_recordings_dir(session_dir: str = "") -> str:
    """Return the recordings root directory for the current session or project."""
    normalized = os.path.dirname(session_dir) if os.path.isfile(session_dir) else session_dir
    if normalized and os.path.isdir(normalized):
        if os.path.basename(os.path.normpath(normalized)).lower() == "recordings":
            return normalized
        parent_dir = os.path.dirname(normalized)
        if os.path.basename(os.path.normpath(parent_dir)).lower() == "recordings":
            return parent_dir

    src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    preferred_dir = os.path.join(src_root, "recordings")
    fallback_dir = os.path.join(os.getcwd(), "recordings")
    if os.path.isdir(preferred_dir):
        return preferred_dir
    if os.path.isdir(fallback_dir):
        return fallback_dir
    return preferred_dir


def _find_latest_completed_session(output_dir: str) -> str:
    """Return the newest completed session directory, or empty string."""
    if not output_dir or not os.path.isdir(output_dir):
        return ""

    candidates = []
    try:
        for name in os.listdir(output_dir):
            session_dir = os.path.join(output_dir, name)
            ecgh_path = os.path.join(session_dir, "recording.ecgh")
            if not os.path.isdir(session_dir):
                continue
            if not os.path.exists(ecgh_path):
                continue
            try:
                sort_key = os.path.getmtime(ecgh_path)
            except Exception:
                sort_key = os.path.getmtime(session_dir)
            candidates.append((sort_key, session_dir))
    except Exception:
        return ""

    if not candidates:
        return ""
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _normalize_patient_info(info: Optional[dict]) -> dict:
    normalized = dict(info or {})
    if not normalized:
        return {}

    display_name = ""
    for key in ("patient_name", "name", "full_name", "patientName"):
        raw_value = normalized.get(key)
        if raw_value is None:
            continue
        value = str(raw_value).strip()
        if value and value.lower() not in {"unknown", "unknown patient"}:
            display_name = value
            break

    if display_name:
        normalized["patient_name"] = display_name
        normalized["name"] = display_name
        normalized["full_name"] = display_name
        normalized["patientName"] = display_name

    return normalized


def _load_patient_info_from_session(session_dir: str, fallback_info: Optional[dict] = None) -> dict:
    merged = _normalize_patient_info(fallback_info)
    if not session_dir:
        return merged

    metadata = read_session_metadata(session_dir) if session_dir else {}
    if isinstance(metadata, dict):
        patient_sources = [metadata.get("patient_info")]
        summary = metadata.get("summary")
        if isinstance(summary, dict):
            patient_sources.append(summary.get("patient_info"))
        for candidate in patient_sources:
            if isinstance(candidate, dict) and candidate:
                merged.update(_normalize_patient_info(candidate))

    patient_json = os.path.join(session_dir, "patient.json")
    if os.path.exists(patient_json):
        try:
            with open(patient_json, "r", encoding="utf-8") as handle:
                patient_data = json.load(handle) or {}
            if isinstance(patient_data, dict) and patient_data:
                merged.update(_normalize_patient_info(patient_data))
        except Exception:
            pass

    return _normalize_patient_info(merged)



def _metrics_duration_sec(metrics_list: list) -> float:
    return float(sum(m.get('duration', 0.0) or 0.0 for m in metrics_list))


def _normalize_beat_class(label) -> str:
    """Map recorded beat labels into the compact class buttons used by the UI."""
    raw = str(label or "").strip().upper()
    if not raw:
        return "Other"
    if raw in {"N", "NORMAL", "SINUS"}:
        return "N"
    if raw.startswith("S") or "SV" in raw or "PAC" in raw:
        return "S"
    if raw.startswith("V") or "PVC" in raw or "VENT" in raw:
        return "V"
    if raw.startswith("P") or "PACED" in raw:
        return "P"
    if "AF" in raw or "AFL" in raw or raw in {"F", "Q"}:
        return "AF"
    if raw in {"X", "ART", "ARTIFACT", "NOISE"} or "ARTIFACT" in raw or "NOISE" in raw:
        return "X"
    if raw == "OTHER":
        return "Other"
    return "Other"


def _class_matches_filter(beat_class: str, filter_key: str) -> bool:
    if filter_key == "all":
        return True
    if filter_key == "AF":
        return beat_class == "AF"
    if filter_key == "Other":
        return beat_class == "Other"
    return beat_class == filter_key


def _template_filter_key(label: str) -> str:
    """Normalize template labels into the UI filter keys."""
    return _normalize_beat_class(label)

try:
    from .theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from .tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from .session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata
    from .summary_utils import derive_hr_focus_summary
except ImportError:
    from ecg.holter.theme import (
        ADC_TO_MV,
        COL_BEAT_S,
        COL_BG,
        COL_BLACK,
        COL_BTN_ACTIVE_BG,
        COL_BTN_ACTIVE_TEXT,
        COL_DARK,
        COL_GRAY,
        COL_GREEN,
        COL_GREEN_DRK,
        COL_GREEN_MID,
        COL_GRID_MAJOR,
        COL_GRID_MINOR,
        COL_RED,
        COL_TEXT,
        COL_TIMESTAMP,
        COL_WAVE_ORANGE,
        COL_WAVE_RED,
        COL_WHITE,
        COL_YELLOW,
        GAINS,
        PAPER_SPEEDS,
        TOOL_CALIPER,
        TOOL_MAGNIFY,
        TOOL_RULER,
        TOOL_SELECT,
    )
    from ecg.holter.tool_engine import (
        ECGToolEngine,
        amplitude_mv_from_pixels,
        caliper_label,
        canonical_tool,
        hint as tool_hint,
        interval_ms_from_pixels,
        ruler_label,
        tool_specs,
        tooltip as tool_tooltip,
    )
    from ecg.holter.session_store import append_annotation, load_annotations, load_events, load_metrics, read_session_metadata
    from ecg.holter.summary_utils import derive_hr_focus_summary

# Professional UI palette (kept separate from signal colors).
UI_BG = "#0B1220"
UI_PANEL = "#0F1A2E"
UI_PANEL_ALT = "#13213A"
UI_CARD = "#101B2F"
UI_BORDER = "#243552"
UI_TEXT = "#E6EDF7"
UI_MUTED = "#9AAECB"
UI_ACCENT = "#2F80ED"
UI_ACCENT_HOVER = "#4B96FA"
UI_SUCCESS = "#16C172"
UI_WARNING = "#F59E0B"


def _style_btn(bg=UI_PANEL_ALT, fg=UI_TEXT, hover="#1A2C49"):
    return f"""
        QPushButton {{
            background: {bg};
            color: {fg};
            border: 1px solid {UI_BORDER};
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{
            background: {hover};
            color: {UI_TEXT};
        }}
        QPushButton:pressed {{ background: {bg}; border: 1px solid {UI_ACCENT_HOVER}; }}
        QPushButton:disabled {{ background: #1A2233; color: #5F708A; border: 1px solid #2A3953; }}
    """


def _style_active_btn():
    return f"""
        QPushButton {{
            background: {UI_ACCENT};
            color: {UI_TEXT};
            border: 1px solid #5EA4FF;
            border-radius: 8px;
            padding: 7px 14px;
            font-size: 12px;
            font-weight: bold;
        }}
        QPushButton:hover {{ background: {UI_ACCENT_HOVER}; }}
    """


def _table_style():
    return f"""
        QTableWidget {{
            background: {UI_PANEL};
            alternate-background-color: {UI_PANEL_ALT};
            color: {UI_TEXT};
            gridline-color: {UI_BORDER};
            font-size: 12px;
            border: 1px solid {UI_BORDER};
            selection-background-color: #1E3A5F;
            selection-color: {UI_TEXT};
        }}
        QHeaderView::section {{
            background: {UI_PANEL_ALT};
            color: {UI_MUTED};
            font-size: 11px;
            font-weight: bold;
            padding: 6px;
            border: 1px solid {UI_BORDER};
        }}
        QTableWidget::item {{ padding: 5px; border: none; }}
        QScrollBar:vertical {{
            border: none;
            background: {UI_PANEL_ALT};
            width: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:vertical {{
            background: {UI_BORDER};
            min-height: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {UI_MUTED};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0px;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: none;
        }}
        QScrollBar:horizontal {{
            border: none;
            background: {UI_PANEL_ALT};
            height: 10px;
            margin: 0px;
        }}
        QScrollBar::handle:horizontal {{
            background: {UI_BORDER};
            min-width: 20px;
            border-radius: 5px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {UI_MUTED};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
            width: 0px;
        }}
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            background: none;
        }}
    """


def _show_message_box(parent, icon, title, text, buttons=QMessageBox.Ok, default_button=QMessageBox.NoButton):
    msg_box = QMessageBox(parent)
    msg_box.setIcon(icon)
    msg_box.setWindowTitle(title)
    msg_box.setText(text)
    msg_box.setStandardButtons(buttons)
    msg_box.setDefaultButton(default_button)
    # Set dark theme style with white text
    msg_box.setStyleSheet(f"""
        QMessageBox {{
            background-color: {UI_PANEL};
        }}
        QMessageBox QLabel {{
            color: {UI_TEXT};
            background-color: transparent;
        }}
        QMessageBox QPushButton {{
            background-color: {UI_PANEL_ALT};
            color: #FFFFFF;
            border: 1px solid {UI_BORDER};
            border-radius: 5px;
            padding: 7px 20px;
            min-width: 70px;
            font-weight: bold;
        }}
        QMessageBox QPushButton:hover {{
            background-color: #1A2C49;
            color: #FFFFFF;
        }}
        QMessageBox QPushButton:pressed {{
            background-color: {UI_BORDER};
            color: #FFFFFF;
        }}
    """)
    return msg_box.exec_()
    return msg_box.exec_()


def _sec_to_hms(s: float) -> str:
    h = int(s // 3600); m = int((s % 3600) // 60); sec = int(s % 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _format_system_time(session_dir: str, chunk_timestamp: float) -> str:
    """Convert relative timestamp to actual system wall-clock time."""
    if not session_dir:
        return _sec_to_hms(chunk_timestamp)
    try:
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
            if start_time:
                system_time = datetime.fromtimestamp(start_time + chunk_timestamp)
                return system_time.strftime('%H:%M:%S')
    except Exception as e:
        print(f"[HolterUI] Error reading system time: {e}")
    return _sec_to_hms(chunk_timestamp)


def _get_recording_start_end_times(session_dir: str, duration_sec: float) -> tuple:
    """Return recording start and end date-time strings."""
    if not session_dir:
        return "Unknown", "Unknown"
    try:
        index_path = os.path.join(session_dir, 'recording_index.json')
        if os.path.exists(index_path):
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            start_time = index_data.get('start_time')
            if start_time:
                start_dt = datetime.fromtimestamp(start_time)
                end_dt = datetime.fromtimestamp(start_time + duration_sec)
                return start_dt.strftime('%Y-%m-%d %H:%M:%S'), end_dt.strftime('%Y-%m-%d %H:%M:%S')
    except Exception as e:
        print(f"[HolterUI] Error reading recording start/end times: {e}")
    return "Unknown", "Unknown"



# -----------------------------------------------------------------------------
# 1. HOLTER START DIALOG
# -----------------------------------------------------------------------------

class HolterStartDialog(QDialog):
    def __init__(self, parent=None, patient_info: dict = None, output_dir: str = "recordings"):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive ECG Analysis - Setup")
        self.setMinimumWidth(640)
        self.setStyleSheet(f"background: #0F1724; color: {COL_WHITE};")
        self.output_dir = output_dir
        self._result_info = None
        self._result_duration = 24
        self._result_dir = output_dir
        self._build_ui(patient_info or {})

    def _build_ui(self, info: dict):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(24, 24, 24, 24)

        title = QLabel("Comprehensive ECG Analysis - Professional Setup")
        title.setStyleSheet(f"background:{COL_GRAY};color:{COL_GREEN};border:2px solid {COL_GREEN};"
                            f"font-size:20px;font-weight:bold;padding:16px;border-radius:8px;")
        layout.addWidget(title)

        subtitle = QLabel("Enter patient details, choose study duration, and launch the 12-lead ECG workspace.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:#cccccc;font-size:13px;padding:4px;")
        layout.addWidget(subtitle)

        # Patient info group
        pg = QGroupBox("Patient Information")
        pg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        pg_layout = QGridLayout(pg)
        pg_layout.setSpacing(10)

        fields = [
            ("Patient Name", "patient_name", info.get("patient_name", "")),
            ("Age", "age", str(info.get("age", ""))),
            ("Email", "email", info.get("email", "")),
            ("Doctor", "doctor", info.get("doctor", "")),
            ("Organisation", "org", info.get("Org.", info.get("org", ""))),
            ("Phone", "phone", info.get("doctor_mobile", info.get("phone", ""))),
        ]
        self._fields = {}
        for row, (label, key, default) in enumerate(fields):
            lbl = QLabel(label + ":")
            lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
            edit = QLineEdit(default)
            edit.setStyleSheet(f"QLineEdit{{border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                               f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};}}"
                               f"QLineEdit:focus{{border-color:{COL_GREEN};}}")
            pg_layout.addWidget(lbl, row, 0)
            pg_layout.addWidget(edit, row, 1)
            self._fields[key] = edit

        lbl_g = QLabel("Gender:")
        lbl_g.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._gender = QComboBox()
        self._gender.addItems(["Select", "Male", "Female", "Other"])
        idx = self._gender.findText(info.get("gender", info.get("sex", "Select")))
        if idx >= 0: self._gender.setCurrentIndex(idx)
        self._gender.setStyleSheet(f"""
            QComboBox {{
                border:1px solid {COL_GREEN_DRK}; border-radius:4px; padding:8px;
                background:{COL_DARK}; color:{COL_GREEN};
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        pg_layout.addWidget(lbl_g, len(fields), 0)
        pg_layout.addWidget(self._gender, len(fields), 1)
        layout.addWidget(pg)

        # Recording settings group
        rg = QGroupBox("Recording Settings")
        rg.setStyleSheet(f"QGroupBox{{font-weight:bold;color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                         f"border-radius:8px;margin-top:12px;padding-top:20px;background:{COL_BLACK};}}")
        rg_layout = QGridLayout(rg)
        rg_layout.setSpacing(10)

        dur_lbl = QLabel("Duration:")
        dur_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        self._duration = QComboBox()
        self._duration.addItems(["24 hours", "48 hours", "Custom"])
        self._duration.setStyleSheet(f"""
            QComboBox {{
                border:1px solid {COL_GREEN_DRK}; border-radius:4px; padding:8px;
                background:{COL_DARK}; color:{COL_GREEN};
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._duration.currentTextChanged.connect(lambda t: self._custom_hours.setVisible(t == "Custom"))
        rg_layout.addWidget(dur_lbl, 0, 0)
        rg_layout.addWidget(self._duration, 0, 1)

        self._custom_hours = QSpinBox()
        self._custom_hours.setRange(1, 72)
        self._custom_hours.setValue(24)
        self._custom_hours.setSuffix(" hours")
        self._custom_hours.setVisible(False)
        self._custom_hours.setStyleSheet(f"border:1px solid {COL_GREEN_DRK};border-radius:4px;padding:8px;"
                                         f"font-size:13px;background:{COL_DARK};color:{COL_GREEN};")
        rg_layout.addWidget(self._custom_hours, 1, 1)

        out_lbl = QLabel("Output Directory:")
        out_lbl.setStyleSheet(f"font-weight:bold;font-size:13px;color:{COL_GREEN};")
        rg_layout.addWidget(out_lbl, 2, 0)
        dir_row = QHBoxLayout()
        self._dir_label = QLabel(self.output_dir)
        self._dir_label.setStyleSheet(f"font-size:12px;color:{COL_GREEN};")
        dir_row.addWidget(self._dir_label, 1)
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet(_style_btn())
        browse_btn.clicked.connect(self._browse_dir)
        dir_row.addWidget(browse_btn)
        rg_layout.addLayout(dir_row, 2, 1)

        self._rec_count_label = QLabel("")
        self._rec_count_label.setStyleSheet(f"font-size:13px;color:{COL_GREEN};font-weight:700;")
        rg_layout.addWidget(QLabel("Recorded Sessions:"), 3, 0)
        rg_layout.addWidget(self._rec_count_label, 3, 1)
        self._refresh_rec_count()
        layout.addWidget(rg)

        btn_row = QHBoxLayout()
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setStyleSheet(_style_btn(COL_GRAY, COL_WHITE, COL_GREEN_DRK))
        cancel_btn.clicked.connect(self.reject)
        start_btn = QPushButton("Open ECG Workspace")
        start_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                    stop:0 #ff6600, stop:1 #e65c00);
                color: white;
                border: none;
                border-radius: 8px;
                padding: 14px 24px;
                font-size: 14px;
                font-weight: bold;
            }}
            QPushButton:hover {{ background: #ff7a26; }}
            QPushButton:pressed {{ background: #cc5200; }}
        """)
        start_btn.setMinimumHeight(48)
        start_btn.clicked.connect(self._on_start)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(start_btn, 1)
        layout.addLayout(btn_row)

    def _browse_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.output_dir)
        if d:
            self._result_dir = d
            self._dir_label.setText(d)
            self._refresh_rec_count()

    def _refresh_rec_count(self):
        root = getattr(self, '_result_dir', self.output_dir)
        count = 0
        try:
            if os.path.isdir(root):
                for name in os.listdir(root):
                    if os.path.exists(os.path.join(root, name, "recording.ecgh")):
                        count += 1
        except Exception:
            pass
        self._rec_count_label.setText(f"{count} completed recording(s)")

    def _on_start(self):
        info = {key: field.text().strip() for key, field in self._fields.items()}
        info['gender'] = self._gender.currentText()
        info['sex'] = info['gender']
        info['name'] = info.get('patient_name', 'Unknown')
        info['Org.'] = info.get('org', '')
        if not info.get('patient_name'):
            QMessageBox.warning(self, "Missing Name", "Please enter the patient name.")
            return
        dur_text = self._duration.currentText()
        if dur_text == "24 hours": self._result_duration = 24
        elif dur_text == "48 hours": self._result_duration = 48
        else: self._result_duration = self._custom_hours.value()
        self._result_info = info
        self._result_dir = self._dir_label.text()
        self.accept()

    def get_result(self):
        if self._result_info:
            return self._result_info, self._result_duration, self._result_dir
        return None


# -----------------------------------------------------------------------------
# 2. HOLTER STATUS BAR  (Live recording indicator)
# -----------------------------------------------------------------------------

class HolterStatusBar(QFrame):
    stop_requested = pyqtSignal()

    def __init__(self, parent=None, target_hours: int = 24):
        super().__init__(parent)
        self.target_hours = target_hours
        self._start_time = time.time()
        self._blink_state = True
        self.setFixedHeight(52)
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-bottom:2px solid {COL_GREEN};}}")
        self._build_ui()
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink)
        self._blink_timer.start(800)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._elapsed_timer.start(1000)

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
        layout.setContentsMargins(15, 4, 15, 4)
        layout.setSpacing(18)

        self._rec_label = QLabel("REC")
        self._rec_label.setStyleSheet(f"color:{COL_GREEN};font-size:15px;font-weight:bold;")
        layout.addWidget(self._rec_label)

        self._time_label = QLabel("00:00:00")
        self._time_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;font-family:monospace;")
        layout.addWidget(self._time_label)

        tgt = QLabel(f"/ {self.target_hours:02d}:00:00")
        tgt.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:12px;")
        layout.addWidget(tgt)

        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep)

        bpm_lbl = QLabel("BPM:")
        bpm_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(bpm_lbl)
        self._bpm_label = QLabel("-")
        self._bpm_label.setStyleSheet(f"color:{COL_GREEN};font-size:18px;font-weight:bold;")
        layout.addWidget(self._bpm_label)

        sep2 = QLabel("|")
        sep2.setStyleSheet(f"color:{COL_GREEN_DRK};")
        layout.addWidget(sep2)

        ev_lbl = QLabel("Events:")
        ev_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:12px;")
        layout.addWidget(ev_lbl)
        self._arrhy_label = QLabel("None detected")
        self._arrhy_label.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;")
        self._arrhy_label.setMaximumWidth(380)
        layout.addWidget(self._arrhy_label, 1)

        self._progress = QProgressBar()
        self._progress.setRange(0, self.target_hours * 3600)
        self._progress.setValue(0)
        self._progress.setFixedWidth(140)
        self._progress.setFixedHeight(12)
        self._progress.setStyleSheet(f"""
            QProgressBar{{background:{COL_DARK};border-radius:6px;border:1px solid {COL_GREEN_DRK};}}
            QProgressBar::chunk{{background:{COL_GREEN};border-radius:5px;}}
        """)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)

        stop_btn = QPushButton("Stop")
        stop_btn.setStyleSheet(_style_btn(COL_GREEN_DRK, COL_WHITE, COL_GREEN))
        stop_btn.setFixedHeight(34)
        stop_btn.clicked.connect(self.stop_requested)
        layout.addWidget(stop_btn)

    def _blink(self):
        self._blink_state = not self._blink_state
        color = COL_GREEN if self._blink_state else COL_GREEN_DRK
        self._rec_label.setStyleSheet(f"color:{color};font-size:15px;font-weight:bold;")

    def _update_elapsed(self):
        elapsed = int(time.time() - self._start_time)
        h = elapsed // 3600; m = (elapsed % 3600) // 60; s = elapsed % 60
        self._time_label.setText(f"{h:02d}:{m:02d}:{s:02d}")
        self._progress.setValue(min(elapsed, self.target_hours * 3600))

    def update_stats(self, bpm: float, arrhythmias: List[str]):
        if bpm > 0:
            self._bpm_label.setText(f"{bpm:.0f}")
        if arrhythmias:
            self._arrhy_label.setText("  |  ".join(arrhythmias[:3]))

    def cleanup(self):
        self._blink_timer.stop()
        self._elapsed_timer.stop()


# -----------------------------------------------------------------------------
# 3. HOLTER SUMMARY CARDS
# -----------------------------------------------------------------------------

class HolterSummaryCards(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._value_labels = {}
        self._card_frames = []
        self._grid = None
        self.setStyleSheet(f"background:{UI_BG};")
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
        self._grid = QGridLayout(self)
        self._grid.setContentsMargins(12, 10, 12, 10)
        self._grid.setHorizontalSpacing(10)
        self._grid.setVerticalSpacing(10)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)

        cards = [
            ("Average HR", "avg_hr", "bpm"),
            ("Min / Max HR", "range_hr", "bpm"),
            ("Total Beats", "beats", ""),
            ("Pauses", "pauses", "events"),
            ("Signal Quality", "quality", "%"),
            ("HRV SDNN", "sdnn", "ms"),
            ("rMSSD", "rmssd", "ms"),
            ("Longest RR", "longest_rr", "s"),
        ]
        for idx, (title, key, unit) in enumerate(cards):
            frame = QFrame()
            frame.setStyleSheet(
                f"QFrame{{background:{UI_CARD};border:1px solid {UI_BORDER};border-radius:10px;}}"
            )
            frame.setMinimumHeight(74)
            box = QVBoxLayout(frame)
            box.setContentsMargins(12, 10, 12, 10)
            box.setSpacing(3)
            lbl = QLabel(title)
            lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
            val = QLabel("-")
            val.setStyleSheet(f"color:{UI_TEXT};font-size:21px;font-weight:700;border:none;")
            unit_lbl = QLabel(unit)
            unit_lbl.setStyleSheet(f"color:{UI_SUCCESS};font-size:10px;font-weight:700;border:none;")
            box.addWidget(lbl)
            box.addWidget(val)
            box.addWidget(unit_lbl)
            self._value_labels[key] = val
            self._card_frames.append(frame)
        self._relayout_cards()

    def _relayout_cards(self):
        if self._grid is None:
            return
        while self._grid.count():
            item = self._grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(self)

        width = max(1, self.width())
        if width >= 1400:
            cols = 4
        elif width >= 1050:
            cols = 3
        elif width >= 700:
            cols = 2
        else:
            cols = 1

        for idx, frame in enumerate(self._card_frames):
            self._grid.addWidget(frame, idx // cols, idx % cols)

        rows = int(math.ceil(len(self._card_frames) / float(cols)))
        self.setMinimumHeight(rows * 84 + 24)

    def update_summary(self, s: dict):
        self._value_labels["avg_hr"].setText(f"{s.get('avg_hr', 0):.0f}")
        self._value_labels["range_hr"].setText(f"{s.get('min_hr', 0):.0f} / {s.get('max_hr', 0):.0f}")
        self._value_labels["beats"].setText(f"{s.get('total_beats', 0):,}")
        self._value_labels["pauses"].setText(str(s.get("pauses", 0)))
        self._value_labels["quality"].setText(f"{s.get('avg_quality', 0) * 100:.1f}")
        self._value_labels["sdnn"].setText(f"{s.get('sdnn', 0):.1f}")
        self._value_labels["rmssd"].setText(f"{s.get('rmssd', 0):.1f}")
        self._value_labels["longest_rr"].setText(f"{s.get('longest_rr_ms', 0) / 1000:.2f}")
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._relayout_cards()


# -----------------------------------------------------------------------------
# 4. HOLTER OVERVIEW PANEL
# -----------------------------------------------------------------------------

class HolterOverviewPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{UI_BG};")
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        title = QLabel("Overview")
        title.setStyleSheet(
            f"color:{UI_TEXT};font-size:14px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:8px;border-radius:6px;border:1px solid {UI_BORDER};"
        )
        layout.addWidget(title)

        self._table = QTableWidget(0, 2)
        self._table.setHorizontalHeaderLabels(["Name", "Value"])
        self._table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._table.setStyleSheet(_table_style())
        self._table.setAlternatingRowColors(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_summary(self, s: dict):
        rows = [
            ("Total Beats",          f"{s.get('total_beats', 0):,}"),
            ("AVG Heart Rate",       f"{s.get('avg_hr', 0):.0f} bpm"),
            ("Max HR",               f"{s.get('max_hr', 0):.0f} bpm"),
            ("Min HR",               f"{s.get('min_hr', 0):.0f} bpm"),
            ("Sinus Max HR",         f"{s.get('max_hr', 0):.0f} bpm"),
            ("Sinus Min HR",         f"{s.get('min_hr', 0):.0f} bpm"),
            ("Longest RR Interval",  f"{s.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (>=2.0s)",          str(pauses)),
            ("Tachycardia Beats",    str(s.get('tachy_beats', 0))),
            ("Bradycardia Beats",    str(s.get('brady_beats', 0))),
            ("Ventricular Beats",    str(s.get('ve_beats', 0))),
            ("Supraventricular Beats", str(s.get('sve_beats', 0))),
            ("Template Clusters",    str(s.get('template_count', 0))),
            ("X Total",              str(s.get('pauses', 0))),
            ("SDNN (HRV)",           f"{s.get('sdnn', 0):.1f} ms"),
            ("rMSSD (HRV)",          f"{s.get('rmssd', 0):.1f} ms"),
            ("pNN50 (HRV)",          f"{s.get('pnn50', 0):.2f}%"),
            ("ST Elevation",         "-"),
            ("ST Depression",        "-"),
            ("Signal Quality",       f"{s.get('avg_quality', 1.0)*100:.1f}%"),
            ("Chunks Analyzed",      str(s.get('chunks_analyzed', 0))),
        ]
        self._table.setRowCount(len(rows))
        for i, (name, value) in enumerate(rows):
            ni = QTableWidgetItem(name)
            ni.setForeground(QColor(UI_MUTED))
            ni.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
            vi = QTableWidgetItem(value)
            vi.setForeground(QColor(UI_TEXT))
            vi.setBackground(QColor(UI_PANEL if i % 2 == 0 else UI_PANEL_ALT))
            vi.setFont(QFont("Arial", 12, QFont.Bold))
            self._table.setItem(i, 0, ni)
            self._table.setItem(i, 1, vi)
        self._table.resizeRowsToContents()


# -----------------------------------------------------------------------------
# 5. HOLTER HRV PANEL
# -----------------------------------------------------------------------------

class HolterHRVPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        tab_row = QHBoxLayout()
        self._hrv_event_btn = QPushButton("HRV Event")
        self._hrv_event_btn.setStyleSheet(_style_active_btn())
        self._hrv_trend_btn = QPushButton("HRV Tendency")
        self._hrv_trend_btn.setStyleSheet(_style_btn())
        tab_row.addWidget(self._hrv_event_btn)
        tab_row.addWidget(self._hrv_trend_btn)
        tab_row.addStretch()
        layout.addLayout(tab_row)

        cols = ["Type", "Start at", "Duration", "Mean NN", "SDNN", "SDANN", "TRIIDX", "pNN50", "LF", "HF", "LF/HF", "Status"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

        # Bottom stats strip
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setSpacing(8)
        stats_layout.setContentsMargins(12, 8, 12, 8)
        self._summary_labels = {}
        stat_defs = [("NNs", "nns"), ("Mean NN", "mean_nn"), ("SDNN", "sdnn"), ("SDANN", "sdann"),
                     ("rMSSD", "rmssd"), ("pNN50", "pnn50"), ("TRIIDX", "triidx"), ("SDNNIDX", "sdnnidx"),
                     ("VLF", "vlf"), ("LF", "lf"), ("HF", "hf"), ("LF/HF", "lf_hf_ratio")]
        for i, (label, key) in enumerate(stat_defs):
            row, col = divmod(i, 4)
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            val = QLabel("-")
            val.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;"
                              f"background:{COL_DARK};border:1px solid {COL_GREEN_DRK};"
                              f"border-radius:10px;padding:4px 10px;min-width:70px;")
            val.setAlignment(Qt.AlignCenter)
            stats_layout.addWidget(lbl, row * 2, col)
            stats_layout.addWidget(val, row * 2 + 1, col)
            self._summary_labels[key] = val
        layout.addWidget(stats_frame)

        btn_row = QHBoxLayout()
        for lbl in ["Insert", "Reset", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn_row.addWidget(btn, 1)
        layout.addLayout(btn_row)

    def update_hrv(self, metrics_list: list, summary: dict):
        hourly: dict = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        all_rr = [m.get('rr_ms', 0) for m in metrics_list if m.get('rr_ms', 0) > 0]
        if all_rr:
            total_duration_sec = int(_metrics_duration_sec(metrics_list))
            from .hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(all_rr)
            rows.append(("Entire", "-", f"{total_duration_sec//60:02d}:{total_duration_sec%60:02d}",
                         f"{int(np.mean(all_rr))}ms", f"{hrv.get('sdnn', summary.get('sdnn', 0)):.0f}ms",
                         f"{hrv.get('sdnn', summary.get('sdnn', 0))*0.82:.0f}ms", f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', summary.get('pnn50', 0)):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            rr_vals = [c.get('rr_ms', 0) for c in chunks if c.get('rr_ms', 0) > 0]
            rr_stds = [c.get('rr_std', 0) for c in chunks if c.get('rr_std', 0) > 0]
            pnn50s = [c.get('pnn50', 0) for c in chunks]
            if not rr_vals: continue
            from .hrv_metrics import compute_hrv_summary
            hrv = compute_hrv_summary(rr_vals)
            rows.append(("Hour", f"{h:02d}:00", "01:00",
                         f"{int(np.mean(rr_vals))}ms",
                         f"{hrv.get('sdnn', 0):.0f}ms",
                         f"{hrv.get('sdnn', 0)*0.82:.0f}ms",
                         f"{hrv.get('triangular_index', 0.0):.2f}",
                         f"{hrv.get('pnn50', np.mean(pnn50s) if pnn50s else 0.0):.2f}%",
                         f"{hrv.get('lf', 0.0):.3f}", f"{hrv.get('hf', 0.0):.3f}",
                         f"{hrv.get('lf_hf_ratio', 0.0):.3f}", ""))

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(str(val))
                item.setForeground(QColor(COL_WHITE if j > 0 else COL_GREEN))
                self._table.setItem(i, j, item)

        s = summary
        for key, fmt in [("nns", str(s.get('total_beats', 0))),
                          ("mean_nn", f"{s.get('avg_hr', 0):.0f}ms"),
                          ("sdnn", f"{s.get('sdnn', 0):.0f}ms"),
                          ("sdann", f"{s.get('sdnn', 0)*0.82:.0f}ms"),
                          ("rmssd", f"{s.get('rmssd', 0):.0f}ms"),
                          ("pnn50", f"{s.get('pnn50', 0):.2f}%"),
                          ("triidx", f"{s.get('triidx', 0.0):.2f}" if s.get('triidx', 0) else "-"),
                          ("sdnnidx", "-"),
                          ("vlf", f"{s.get('vlf_power', 0.0):.3f}" if s.get('vlf_power', 0) else "-"),
                          ("lf", f"{s.get('lf_power', 0.0):.3f}" if s.get('lf_power', 0) else "-"),
                          ("hf", f"{s.get('hf_power', 0.0):.3f}" if s.get('hf_power', 0) else "-"),
                          ("lf_hf_ratio", f"{s.get('lf_hf_ratio', 0.0):.3f}" if s.get('lf_hf_ratio', 0) else "-")]:
            if key in self._summary_labels:
                self._summary_labels[key].setText(fmt)


# -----------------------------------------------------------------------------
# 6. HOLTER LORENZ / REPLAY PANEL
# -----------------------------------------------------------------------------

class HolterReplayPanel(QWidget):
    playback_state_changed = pyqtSignal(bool)
    seek_requested = pyqtSignal(float)
    lead_changed   = pyqtSignal(int)
    section_requested = pyqtSignal(str)
    frame_received = pyqtSignal(object)

    def __init__(self, parent=None, duration_sec: float = 86400):
        super().__init__(parent)
        self.duration_sec = max(1, duration_sec)
        self._strip_length_sec = 10.0
        self.setStyleSheet(f"background:{COL_DARK};")
        self._replay_engine = None
        self._tool_engine = ECGToolEngine()
        self._current_replay_frame = None
        self._selected_lead_idx = 1
        self._slider_units_per_sec = 100
        self._last_slider_seek_raw = None
        self._class_filter = "all"
        self._last_metrics_list = []
        self._build_ui()
        self._magnifier_overlay = MagnifierOverlay(self)
        self._magnifier_overlay.setGeometry(self.rect())
        self._magnifier_overlay.hide()
        self._install_magnifier_dismiss_filters()

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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self._ribbon_buttons = {}

        self._rr_mode = "RR"
        self._time_scope = "whole"

        #  "  "  48-hour session summary bar (replaces the two RR trend canvases)  "  " 
        summary_frame = QFrame()
        summary_frame.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:6px;}}")
        summary_frame.setFixedHeight(72)
        summary_layout = QHBoxLayout(summary_frame)
        summary_layout.setContentsMargins(12, 6, 12, 6)
        summary_layout.setSpacing(20)
        self._summary_labels = {}
        for key, title in [("duration","Duration"),("total_beats","Total Beats"),("avg_hr","Avg HR"),("max_hr","Max HR"),("min_hr","Min HR"),("pauses","Pauses"),("ve","VE Beats"),("sve","SVE Beats"),("sdnn","SDNN"),("rmssd","rMSSD")]:
            col = QVBoxLayout()
            col.setSpacing(2)
            t = QLabel(title)
            t.setStyleSheet(f"color:{UI_MUTED};font-size:9px;font-weight:600;border:none;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{UI_TEXT};font-size:13px;font-weight:700;border:none;")
            col.addWidget(t)
            col.addWidget(v)
            self._summary_labels[key] = v
            summary_layout.addLayout(col)
        summary_layout.addStretch()
        layout.addWidget(summary_frame)

        #    HR trend mini-chart (40h-48h of data at a glance)   
        self._hr_trend_canvas = HolterRRTrendCanvas(title="Heart Rate Trend (full recording)")
        self._hr_trend_canvas.setFixedHeight(120)
        layout.addWidget(self._hr_trend_canvas)
        # Keep these as dummy attrs so update_lorenz doesn't crash
        self._rr_trend_full = self._hr_trend_canvas
        self._rr_trend_zoom = self._hr_trend_canvas

        time_row = QHBoxLayout()
        self._btn_time_whole = QPushButton("Time-whole")
        self._btn_time_share = QPushButton("Time-share")
        self._btn_goto_time = QPushButton("Goto Time")
        self._btn_rr = QPushButton("RR")
        self._btn_hr = QPushButton("HR")
        for b in [self._btn_time_whole, self._btn_time_share, self._btn_goto_time]:
            b.setFixedHeight(28)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        time_row.addStretch()
        for b in [self._btn_rr, self._btn_hr]:
            b.setFixedHeight(28)
            b.setFixedWidth(52)
            b.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            time_row.addWidget(b)
        layout.addLayout(time_row)
        self._btn_time_whole.clicked.connect(lambda: self._set_time_scope("whole"))
        self._btn_time_share.clicked.connect(lambda: self._set_time_scope("share"))
        self._btn_goto_time.clicked.connect(self._goto_time)
        self._btn_rr.clicked.connect(lambda: self._set_rr_mode("RR"))
        self._btn_hr.clicked.connect(lambda: self._set_rr_mode("HR"))
        self._set_time_scope("whole")
        self._set_rr_mode("RR")

        # Top: left (lorenz + templates) + right (focused CH strips)
        top_splitter = QSplitter(Qt.Horizontal)
        top_splitter.setChildrenCollapsible(False)
        top_splitter.setHandleWidth(1)
        top_splitter.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left_wrap = QFrame()
        left_wrap.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        lw_l = QVBoxLayout(left_wrap)
        lw_l.setContentsMargins(4, 4, 4, 4)
        lw_l.setSpacing(6)
        self._lorenz_canvas = LorenzCanvas(parent=left_wrap)
        lw_l.addWidget(self._lorenz_canvas, 3)
        lorenz_filter_row = QHBoxLayout()
        lorenz_filter_row.setSpacing(4)
        self._lorenz_class_btns = {}
        for key, lbl in [("all", "All"), ("N", "N"), ("S", "S"), ("V", "V"), ("P", "P"), ("AF", "AF"), ("X", "X"), ("Other", "Other")]:
            b = QPushButton(lbl)
            b.setCheckable(True)
            b.setToolTip(f"Show {lbl} beats" if key != "all" else "Show all beats")
            b.setFixedHeight(24)
            style = _style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK).replace("padding: 7px 14px;", "padding: 1px 4px;")
            b.setStyleSheet(style)
            b.clicked.connect(lambda checked=False, k=key: self._set_lorenz_class_filter(k))
            self._lorenz_class_btns[key] = b
            lorenz_filter_row.addWidget(b)
        lorenz_filter_row.addStretch()
        lw_l.addLayout(lorenz_filter_row)
        self._set_lorenz_class_filter("all")

        self._template_thumbs = []
        top_splitter.addWidget(left_wrap)

        ecg_right = QFrame()
        ecg_right.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        ecg_right_layout = QVBoxLayout(ecg_right)
        ecg_right_layout.setContentsMargins(4, 4, 4, 4)
        ecg_right_layout.setSpacing(2)

        #  "  "  12-lead scrollable grid (1 column, 12 rows)  "  " 
        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};")
        leads_vbox = QVBoxLayout(leads_container)
        leads_vbox.setContentsMargins(2, 2, 2, 2)
        leads_vbox.setSpacing(2)

        self._lead_names_ordered = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        self._lead_strips = {}   # lead_name -> ECGStripCanvas
        self._ch_strips = []     # backward-compat list for gain/speed handlers
        self._detected_r_peaks = []  # Store R-peak timestamps for vertical line overlay
        
        for lead in self._lead_names_ordered:
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            row.setSpacing(4)
            lbl = QLabel(lead)
            lbl.setFixedWidth(34)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            # Height 60px makes them compact enough to fit well, but scrollable if needed
            # Replay mode: disable ALL coloring (disable_all_coloring=True) - no arrhythmia colors, just green
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.9, lead_name=lead, show_annotations=False, disable_all_coloring=True)
            strip.set_gain(1.0)
            self._lead_strips[lead] = strip
            self._ch_strips.append(strip)
            row.addWidget(lbl)
            row.addWidget(strip, 1)
            leads_vbox.addLayout(row)

        leads_scroll.setWidget(leads_container)
        ecg_right_layout.addWidget(leads_scroll, 3)

        # Full-width rhythm strip (Lead II) at bottom
        rhythm_row = QHBoxLayout()
        rhythm_row.setContentsMargins(0, 0, 0, 0)
        rhythm_row.setSpacing(4)
        rhythm_lbl = QLabel("II")
        self._mini_lead_lbl = rhythm_lbl
        rhythm_lbl.setFixedWidth(34)
        rhythm_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rhythm_lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
        # Replay mode: disable ALL coloring
        self._mini_strip = ECGStripCanvas(height=60, color="#00AA00", pen_width=0.9, show_annotations=False, disable_all_coloring=True)
        rhythm_row.addWidget(rhythm_lbl)
        rhythm_row.addWidget(self._mini_strip, 1)
        ecg_right_layout.addLayout(rhythm_row)

        top_splitter.addWidget(ecg_right)


        ov_frame = QFrame()
        ov_frame.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:6px;}}")
        ov_layout = QVBoxLayout(ov_frame)
        ov_layout.setContentsMargins(6, 6, 6, 6)
        ov_layout.setSpacing(6)
        ov_title = QLabel("Overview")
        ov_title.setStyleSheet(f"color:{UI_TEXT};font-size:14px;font-weight:700;border:none;")
        ov_layout.addWidget(ov_title)
        self._overview_table = QTableWidget(0, 2)
        self._overview_table.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview_table.verticalHeader().setVisible(False)
        self._overview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview_table.setSelectionMode(QAbstractItemView.NoSelection)
        self._overview_table.setFocusPolicy(Qt.NoFocus)
        self._overview_table.setStyleSheet(_table_style())
        ov_layout.addWidget(self._overview_table, 1)
        top_splitter.addWidget(ov_frame)
        top_splitter.setSizes([450, 900, 260])
        layout.addWidget(top_splitter, 2)

        # Scrub slider row
        slider_row = QHBoxLayout()
        self._time_start_label = QLabel("00:00:00")
        self._time_start_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:12px;border:none;")
        slider_row.addWidget(self._time_start_label)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self._slider_sec_to_value(self.duration_sec))
        self._slider.setStyleSheet(f"""
            QSlider::groove:horizontal{{height:8px;background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}
            QSlider::handle:horizontal{{background:{COL_GREEN};border:1px solid {COL_WHITE};border-radius:9px;
                width:18px;height:18px;margin:-6px 0;}}
            QSlider::sub-page:horizontal{{background:{COL_GREEN_DRK};border-radius:4px;}}
        """)
        self._slider.setTracking(True)
        self._slider.valueChanged.connect(self._on_slider)
        self._slider.sliderMoved.connect(self._on_slider)
        slider_row.addWidget(self._slider, 1)
        self._pos_label = QLabel("00:00:00")
        self._pos_label.setStyleSheet(f"color:{COL_TIMESTAMP};font-family:monospace;font-size:14px;font-weight:bold;"
                                      f"background:{COL_BLACK};padding:4px;border:1px solid {COL_GREEN_DRK};border-radius:4px;border:none;")
        slider_row.addWidget(self._pos_label)
        layout.addLayout(slider_row)

        # Transport + controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.setSpacing(10)

        self._play_btn = QPushButton("Play")
        self._play_btn.setStyleSheet(_style_btn())
        self._play_btn.setFixedHeight(30)
        self._play_btn.setMinimumWidth(100)
        self._play_btn.clicked.connect(self._toggle_playback)
        ctrl_row.addWidget(self._play_btn)

        for speed_lbl in ["0.5x", "1x", "2x", "4x"]:
            btn = QPushButton(speed_lbl)
            btn.setStyleSheet(_style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK))
            btn.setFixedHeight(30)
            btn.setMinimumWidth(56)
            btn.clicked.connect(lambda _, s=speed_lbl: self._set_speed(s))
            ctrl_row.addWidget(btn)
        sep = QLabel("|")
        sep.setStyleSheet(f"color:{COL_GREEN_DRK};")
        ctrl_row.addWidget(sep)

        lbl_lead = QLabel("Lead:")
        lbl_lead.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;border:none;font-size:13px;")
        ctrl_row.addWidget(lbl_lead)
        self._lead_combo = QComboBox()
        self._lead_combo.addItems(["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"])
        self._lead_combo.setCurrentIndex(1)
        self._lead_combo.setFixedWidth(70)
        self._lead_combo.setStyleSheet(f"background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                       f"padding:4px;border-radius:4px;font-weight:bold;")
        self._lead_combo.currentIndexChanged.connect(self._on_lead_changed)
        ctrl_row.addWidget(self._lead_combo)

        ctrl_row.addSpacing(22)

        # Event jump buttons
        for lbl_txt, ev, d in [("Prev AF","AF","prev"),("Next AF","AF","next"),
                               ("Prev Brady","Brady","prev"),("Next Brady","Brady","next"),
                               ("Prev Tachy","Tachy","prev"),("Next Tachy","Tachy","next")]:
            btn = QPushButton(lbl_txt)
            btn.setStyleSheet(_style_btn(COL_BLACK, COL_GREEN, COL_GREEN_DRK))
            btn.setFixedHeight(30)
            ev_c, d_c = ev, d
            btn.setMinimumWidth(78)
            btn.clicked.connect(lambda _, e=ev_c, dd=d_c: self._jump_event(e, dd))
            ctrl_row.addWidget(btn)

        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Bottom toolbar (like reference image)
        toolbar = QFrame()
        toolbar.setStyleSheet(f"QFrame{{background:{COL_BLACK};border-top:1px solid {COL_GREEN_DRK};}}")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(5, 4, 5, 4)
        toolbar_layout.setSpacing(5)
        self._tool_btns = {}
        tool_min_widths = {
            "Patient information": 140,
            "Full Disc.": 90,
            "Goto Template": 114,
            "Measuring Ruler": 120,
            "Parallel Ruler": 112,
            "Magnifying Glass": 126,
            "Gain Settings": 110,
            "Paper speed:25mm/s": 144,
            "Add Event(space)": 130,
            "Adjust strip position": 146,
            "Strip Length:10s": 120,
        }
        for tool in ["Patient information", "Full Disc.", "Goto Template"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(f"QPushButton{{background:{COL_DARK};color:{COL_TEXT};border:1px solid {COL_GREEN_DRK};"
                               f"border-radius:4px;padding:4px 8px;font-size:10px;}}"
                               f"QPushButton:hover{{background:#202020;color:{COL_WHITE};}}")
            tbtn.setMinimumHeight(30)
            tbtn.setMinimumWidth(tool_min_widths.get(tool, 110))
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        for tool in ["Measuring Ruler", "Parallel Ruler", "Magnifying Glass", "Gain Settings",
                     "Paper speed:25mm/s", "Add Event(space)", "Adjust strip position", "Strip Length:10s"]:
            tbtn = QPushButton(tool)
            tbtn.setStyleSheet(f"QPushButton{{background:{COL_DARK};color:{COL_TEXT};border:1px solid {COL_GREEN_DRK};"
                               f"border-radius:4px;padding:4px 8px;font-size:10px;}}"
                               f"QPushButton:hover{{background:#202020;color:{COL_WHITE};}}")
            tbtn.setMinimumHeight(30)
            tbtn.setMinimumWidth(tool_min_widths.get(tool, 110))
            tbtn.clicked.connect(lambda _, t=tool, b=tbtn: self._set_tool_mode(t, b))
            toolbar_layout.addWidget(tbtn)
            self._tool_btns[tool] = tbtn
        self._tool_btns["Gain Settings"].setToolTip(
            "Cycle gain (5/10/20/40 mm/mV equivalent) to improve waveform visibility."
        )
        toolbar_layout.addStretch()
        layout.addWidget(toolbar)



    def _install_magnifier_dismiss_filters(self):
        """Dismiss the magnifier on any non-wave click inside the replay panel."""
        self.installEventFilter(self)
        for widget in self.findChildren(QWidget):
            widget.installEventFilter(self)

    def _clear_magnifier_if_needed(self, event) -> bool:
        if event.type() != QEvent.MouseButtonPress or event.button() != Qt.LeftButton:
            return False
        overlay = getattr(self, "_magnifier_overlay", None)
        if overlay is None or not getattr(overlay, "_visible", False):
            return False
        source = self.sender()
        if isinstance(source, ECGStripCanvas):
            return False
        overlay.clear_focus()
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, "_magnify_locked"):
                strip._magnify_locked = False
                strip._magnify_pos = None
                strip.update()
        mini_strip = getattr(self, "_mini_strip", None)
        if mini_strip is not None and hasattr(mini_strip, "_magnify_locked"):
            mini_strip._magnify_locked = False
            mini_strip._magnify_pos = None
            mini_strip.update()
        return False

    def eventFilter(self, obj, event):
        if self._clear_magnifier_if_needed(event):
            return False
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())

    def set_magnifier_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.setGeometry(self.rect())
            self._magnifier_overlay.set_focus(source_widget, payload, focus_pos)

    def clear_magnifier_focus(self, source_widget=None):
        if hasattr(self, "_magnifier_overlay") and self._magnifier_overlay is not None:
            self._magnifier_overlay.clear_focus(source_widget)

    def clear_strip_tools(self, source_widget=None):
        """Clear transient ruler/caliper overlays and any locked magnifier."""
        self.clear_magnifier_focus(source_widget)
        for strip in getattr(self, "_ch_strips", []):
            if hasattr(strip, "clear_interaction"):
                strip.clear_interaction()
        mini_strip = getattr(self, "_mini_strip", None)
        if mini_strip is not None and hasattr(mini_strip, "clear_interaction"):
            mini_strip.clear_interaction()

    def _set_tool_mode(self, tool_name: str, btn: QPushButton = None):
        """
        Main tool mode dispatcher - delegates to HolterToolHandlers.
        Keeps holter_ui.py clean by moving implementations to holter_full_disclosure.py.
        """
        from .holter_full_disclosure import HolterToolHandlers
        
        # Patient Information
        if "Patient information" in tool_name:
            HolterToolHandlers.handle_patient_information(self)
            return
        
        # Full Disclosure
        if "Full Disc." in tool_name:
            HolterToolHandlers.handle_full_disclosure(self)
            return
        
        # Goto Template (Info Dialog)
        if "Goto Template" in tool_name:
            HolterToolHandlers.handle_goto_template(self)
            return
        
        # Add Event
        if "Add Event" in tool_name or "space" in tool_name.lower():
            HolterToolHandlers.handle_add_event(self)
            return

        # Tool name conversions
        if "Measuring Ruler" in tool_name:
            tool_name = TOOL_RULER
        elif "Parallel Ruler" in tool_name:
            tool_name = TOOL_CALIPER
        elif "Magnifying Glass" in tool_name:
            tool_name = TOOL_MAGNIFY

        # Gain Settings
        if "Gain Settings" in tool_name:
            HolterToolHandlers.handle_gain_settings(self, btn)
            return
        
        # Paper Speed
        elif "Paper speed" in tool_name:
            HolterToolHandlers.handle_paper_speed(self, btn)
            return
        
        # Strip Length
        elif "Strip Length" in tool_name:
            HolterToolHandlers.handle_strip_length(self, btn)
            return

        # Apply tool mode to strips (Ruler, Caliper, Magnify)
        HolterToolHandlers.apply_tool_mode_to_strips(self, tool_name)

    def _set_time_scope(self, scope: str):
        self._time_scope = scope
        self._btn_time_whole.setStyleSheet(_style_active_btn() if scope == "whole" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_time_share.setStyleSheet(_style_active_btn() if scope == "share" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._strip_length_sec = 10.0 if scope == "whole" else 3.6
        if getattr(self, "_replay_engine", None):
            try:
                self._replay_engine.set_window_length(self._strip_length_sec)
            except Exception:
                pass
        self._refresh_wave_window()

    def _refresh_wave_window(self):
        if getattr(self, '_replay_engine', None):
            try:
                current_pos = float(self._replay_engine.current_position())
                self._replay_engine.seek(current_pos)
                window_sec = float(self._strip_length_sec)
                data = self._replay_engine.get_all_leads_data(window_sec=window_sec)
                beat_annotations = self._replay_engine.get_beat_annotations(window_sec=window_sec)
                start_sec = self._replay_engine.current_position() - window_sec / 2
                self._replay_panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
            except Exception:
                pass
        elif getattr(self, '_live_source', None) is not None and hasattr(self, '_replay_panel'):
            try:
                raw = getattr(self._live_source, 'data', None)
                if raw is not None and hasattr(raw, '__len__') and len(raw) > 0:
                    import numpy as _np
                    n_samp = max(len(raw[i]) for i in range(min(12, len(raw))))
                    arr = _np.full((12, n_samp), 2048.0)
                    for i in range(min(12, len(raw))):
                        ch = _np.asarray(raw[i], dtype=float)
                        arr[i, :len(ch)] = ch
                    self._replay_panel.set_replay_frame(arr)
            except Exception:
                pass

    def _set_rr_mode(self, mode: str):
        self._rr_mode = mode
        self._btn_rr.setStyleSheet(_style_active_btn() if mode == "RR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        self._btn_hr.setStyleSheet(_style_active_btn() if mode == "HR" else _style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
        if hasattr(self, '_hr_trend_canvas'):
            self._hr_trend_canvas.set_mode(mode)

    def _build_info_table(self, rows):
        table = QTableWidget(len(rows), 2)
        table.setHorizontalHeaderLabels(["Field", "Value"])
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.NoSelection)
        table.setFocusPolicy(Qt.NoFocus)
        table.setStyleSheet(_table_style())
        table.setAlternatingRowColors(True)
        for row, (label, value) in enumerate(rows):
            label_item = QTableWidgetItem(str(label))
            label_item.setForeground(QColor(UI_TEXT))
            value_item = QTableWidgetItem(str(value))
            value_item.setForeground(QColor(COL_WHITE))
            table.setItem(row, 0, label_item)
            table.setItem(row, 1, value_item)
        table.resizeRowsToContents()
        return table

    def _show_patient_information(self):
        summary = dict(getattr(self, "_summary", {}) or {})
        engine = getattr(self, "_replay_engine", None)
        session_dir = summary.get("session_dir", "") or ""
        if not session_dir and engine is not None:
            session_dir = os.path.dirname(getattr(engine, "ecgh_path", "") or "")
        metadata = read_session_metadata(session_dir) if session_dir else {}
        if isinstance(metadata, dict):
            meta_summary = metadata.get("summary")
            if isinstance(meta_summary, dict):
                merged = dict(meta_summary)
                merged.update(summary)
                summary = merged
            meta_patient = metadata.get("patient_info")
            if isinstance(meta_patient, dict) and meta_patient:
                summary["patient_info"] = dict(meta_patient)

        patient_info = _normalize_patient_info(
            summary.get("patient_info") or getattr(engine, "patient_info", {}) or {}
        )

        if not summary and not patient_info:
            QMessageBox.information(self, "Patient information", "No recording is loaded.")
            return

        session_name = os.path.basename(session_dir) if session_dir else "Current session"
        record_time = session_name
        parts = session_name.split("_", 3)
        if len(parts) >= 2:
            record_time = "_".join(parts[:2]).replace("_", " ")

        def fmt_num(value, suffix="", digits=1):
            try:
                return f"{float(value):.{digits}f}{suffix}"
            except Exception:
                return "-" if value in (None, "") else f"{value}{suffix}"

        def fmt_duration(seconds):
            try:
                sec = max(0, int(float(seconds)))
            except Exception:
                return "-"
            h = sec // 3600
            m = (sec % 3600) // 60
            s = sec % 60
            if h > 0:
                return f"{h}h {m:02d}m"
            if m > 0:
                return f"{m}m {s:02d}s"
            return f"{s}s"

        patient_rows = [
            ("Patient Name", patient_info.get("patient_name") or patient_info.get("name") or "-"),
            ("Age", patient_info.get("age", "-")),
            ("Gender", patient_info.get("gender") or patient_info.get("sex") or "-"),
            ("Doctor", patient_info.get("doctor") or "-"),
            ("Email", patient_info.get("email") or "-"),
            ("Phone", patient_info.get("doctor_mobile") or patient_info.get("phone") or "-"),
            ("Organisation", patient_info.get("org") or patient_info.get("Org.") or "-"),
            ("Study Duration", fmt_duration(summary.get("duration_sec", 0))),
        ]

        dlg = QDialog(self)
        dlg.setWindowTitle("Patient Information")
        dlg.setMinimumSize(760, 620)
        dlg.setStyleSheet(f"""
            QDialog {{
                background: {UI_BG};
                color: {UI_TEXT};
            }}
            QLabel {{
                border: none;
                background: transparent;
            }}
            QGroupBox {{
                color: {UI_TEXT};
                font-weight: 700;
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background: {COL_BLACK};
            }}
            QHeaderView::section {{
                background: {UI_PANEL_ALT};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                padding: 6px;
                font-weight: 700;
            }}
        """)
        outer = QVBoxLayout(dlg)
        outer.setContentsMargins(16, 16, 16, 16)
        outer.setSpacing(12)

        title = QLabel("Patient Information")
        title.setStyleSheet(f"font-size:18px;font-weight:700;color:{UI_TEXT};")
        outer.addWidget(title)

        subtitle = QLabel(f"Selected recording: {session_name}")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet(f"color:{UI_MUTED};font-size:12px;")
        outer.addWidget(subtitle)

        patient_box = QGroupBox("Patient Details")
        patient_box.setStyleSheet(f"""
            QGroupBox {{
                color: {UI_TEXT};
                font-weight: 700;
                border: 1px solid {UI_BORDER};
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 16px;
                background: {UI_PANEL};
            }}
        """)
        patient_layout = QVBoxLayout(patient_box)
        patient_layout.setContentsMargins(10, 16, 10, 10)
        patient_layout.addWidget(self._build_info_table(patient_rows))
        outer.addWidget(patient_box)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedSize(94, 32)
        close_btn.setStyleSheet(_style_btn())
        close_btn.clicked.connect(dlg.accept)
        btn_row.addWidget(close_btn)
        outer.addLayout(btn_row)

        dlg.exec_()

    def _goto_time(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Go to Time")
        dlg.setFixedSize(420, 160)
        dlg.setStyleSheet(f"""
            QDialog {{
                background: {UI_BG};
                border: 1px solid {UI_BORDER};
                border-radius: 10px;
            }}
            QLabel {{
                color: {UI_TEXT};
                font-size: 13px;
                font-weight: bold;
                border: none;
                background: transparent;
            }}
            QLineEdit {{
                background: {UI_PANEL};
                color: {UI_TEXT};
                border: 1px solid {UI_BORDER};
                border-radius: 6px;
                padding: 8px 12px;
                font-size: 14px;
                selection-background-color: {UI_ACCENT};
            }}
            QLineEdit:focus {{
                border: 1px solid {UI_ACCENT};
            }}
        """)
        v_layout = QVBoxLayout(dlg)
        v_layout.setContentsMargins(20, 18, 20, 18)
        v_layout.setSpacing(12)

        lbl = QLabel("Enter time (HH:MM:SS or seconds):")
        lbl.setStyleSheet(f"color:{UI_MUTED};font-size:12px;font-weight:normal;border:none;background:transparent;")
        v_layout.addWidget(lbl)

        inp = QLineEdit()
        inp.setPlaceholderText("e.g.  01:23:45  or  83")
        inp.setClearButtonEnabled(True)
        v_layout.addWidget(inp)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)
        btn_row.addStretch()
        ok_btn = QPushButton("Go")
        ok_btn.setFixedSize(90, 34)
        ok_btn.setStyleSheet(_style_active_btn())
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setFixedSize(90, 34)
        cancel_btn.setStyleSheet(_style_btn())
        btn_row.addWidget(ok_btn)
        btn_row.addWidget(cancel_btn)
        v_layout.addLayout(btn_row)

        ok_btn.clicked.connect(dlg.accept)
        cancel_btn.clicked.connect(dlg.reject)
        inp.returnPressed.connect(dlg.accept)

        if dlg.exec_() != QDialog.Accepted:
            return
        t = inp.text().strip()
        if not t:
            return
        sec = 0.0
        try:
            if ":" in t:
                parts = [int(p) for p in t.split(":")]
                if len(parts) == 3:
                    sec = parts[0] * 3600 + parts[1] * 60 + parts[2]
                elif len(parts) == 2:
                    sec = parts[0] * 60 + parts[1]
                else:
                    sec = float(t)
            else:
                sec = float(t)
        except Exception:
            warn = QDialog(self)
            warn.setWindowTitle("Invalid Time")
            warn.setFixedSize(340, 120)
            warn.setStyleSheet(f"QDialog{{background:{UI_BG};border:1px solid {UI_BORDER};border-radius:8px;}} QLabel{{color:{UI_TEXT};font-size:12px;border:none;background:transparent;}}")
            wl = QVBoxLayout(warn)
            wl.setContentsMargins(18, 16, 18, 16)
            wl.setSpacing(12)
            wl.addWidget(QLabel("Use HH:MM:SS, MM:SS, or plain seconds."))
            wb = QPushButton("OK")
            wb.setStyleSheet(_style_btn())
            wb.clicked.connect(warn.accept)
            wr = QHBoxLayout()
            wr.addStretch()
            wr.addWidget(wb)
            wl.addLayout(wr)
            warn.exec_()
            return
        sec = max(0.0, min(sec, float(self.duration_sec)))
        self.seek_requested.emit(sec)

    def update_summary(self, summary: dict):
        self._summary = dict(summary or {})
        self._patient_info = _normalize_patient_info(self._summary.get('patient_info') or getattr(self._replay_engine, 'patient_info', {}) or {})
        if hasattr(self, '_summary_labels'):
            def set_lbl(k, v):
                if k in self._summary_labels: self._summary_labels[k].setText(str(v))
            
            dur = summary.get('duration_sec', 0)
            h = int(dur // 3600)
            m = int((dur % 3600) // 60)
            set_lbl("duration", f"{h:02d}h {m:02d}m")
            set_lbl("total_beats", summary.get('total_beats', 0))
            set_lbl("avg_hr", f"{summary.get('avg_hr', 0)} bpm")
            set_lbl("max_hr", f"{summary.get('max_hr', 0)} bpm")
            set_lbl("min_hr", f"{summary.get('min_hr', 0)} bpm")
            set_lbl("pauses", summary.get('pauses', 0))
            set_lbl("ve", summary.get('ve_beats', 0))
            set_lbl("sve", summary.get('sve_beats', 0))
            set_lbl("sdnn", f"{summary.get('sdnn', 0)} ms")
            set_lbl("rmssd", f"{summary.get('rmssd', 0)} ms")

    def _slider_sec_to_value(self, seconds: float) -> int:
        units = max(1, int(getattr(self, "_slider_units_per_sec", 100)))
        return int(round(max(0.0, float(seconds)) * units))

    def _slider_value_to_sec(self, value: int) -> float:
        units = max(1, int(getattr(self, "_slider_units_per_sec", 100)))
        return max(0.0, float(value) / float(units))

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._slider.setRange(0, self._slider_sec_to_value(engine.duration_sec))
        engine.set_position_callback(self._on_position_update)
        engine.set_window_length(getattr(self, "_strip_length_sec", 10.0))
        
        # Wire up data callback safely for thread-safe playback updates
        try:
            self.frame_received.disconnect()
        except TypeError:
            pass
        self.frame_received.connect(self.set_replay_frame)
        engine.set_data_callback(lambda data: self.frame_received.emit(data))

    def _on_slider(self, value):
        raw = int(value)
        sec = self._slider_value_to_sec(raw)
        self._pos_label.setText(_sec_to_hms(sec))
        if self._last_slider_seek_raw == raw:
            return
        self._last_slider_seek_raw = raw
        self.seek_requested.emit(float(sec))

    def _on_lead_changed(self, idx: int):
        self._selected_lead_idx = max(0, int(idx))
        frame = getattr(self, "_current_replay_frame", None)
        if frame is not None:
            try:
                self.set_replay_frame(frame)
            except Exception:
                pass
        try:
            self.lead_changed.emit(int(idx))
        except Exception:
            pass

    def _on_position_update(self, current_sec, duration_sec):
        self._last_slider_seek_raw = self._slider_sec_to_value(current_sec)
        self._slider.blockSignals(True)
        self._slider.setValue(self._slider_sec_to_value(current_sec))
        self._slider.blockSignals(False)
        self._pos_label.setText(_sec_to_hms(current_sec))

    def _toggle_playback(self):
        if not self._replay_engine: return
        if self._replay_engine.is_playing():
            self._replay_engine.pause()
            self._play_btn.setText("Play")
        else:
            self._replay_engine.play()
            self._play_btn.setText("Pause")

    def _set_speed(self, text: str):
        if self._replay_engine:
            try:
                self._replay_engine.set_speed(float(text.replace("x", "")))
            except Exception:
                pass

    def _set_lorenz_class_filter(self, key: str):
        key = str(key or "all")
        if key not in self._lorenz_class_btns:
            key = "all"
        self._class_filter = key
        for btn_key, btn in self._lorenz_class_btns.items():
            btn.setChecked(btn_key == key)
            btn.setStyleSheet(_style_active_btn() if btn_key == key else _style_btn(COL_DARK, COL_GREEN, COL_GREEN_DRK))
        if self._last_metrics_list:
            self.update_lorenz(self._last_metrics_list)

    def _jump_event(self, ev_type: str, direction: str):
        if self._replay_engine:
            t = self._replay_engine.seek_to_event(ev_type, direction)
            self.seek_requested.emit(t)

    def update_lorenz(self, metrics_list: list):
        """Update the Lorenz/scatter plot from all individual RR data."""
        self._last_metrics_list = list(metrics_list or [])
        rr_all = []
        rr_points = []
        for m in metrics_list:
            t0 = float(m.get('t', 0.0) or 0.0)
            beat_labels = list(m.get('all_beats') or [])
            if 'rr_intervals_list' in m:
                rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
                if rr_list:
                    dur = float(m.get('duration', 0.0) or 0.0)
                    step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                    for i, rr in enumerate(rr_list):
                        beat_info = beat_labels[i] if i < len(beat_labels) and isinstance(beat_labels[i], dict) else {}
                        beat_class = _normalize_beat_class(
                            beat_info.get("label")
                            or beat_info.get("template_label")
                            or beat_info.get("auto_label")
                            or m.get("label")
                            or m.get("template_label")
                            or m.get("arrhythmia")
                            or (m.get("arrhythmias") or [None])[0]
                        )
                        rr_all.append(rr)
                        rr_points.append((t0 + i * step, rr, beat_class))
                    continue  # skip fallback if list data available
            # Fallback: use single rr_ms value per chunk
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                rr_all.append(rr_val)
                beat_class = _normalize_beat_class(
                    m.get("label")
                    or m.get("template_label")
                    or m.get("arrhythmia")
                    or (m.get("arrhythmias") or [None])[0]
                )
                rr_points.append((t0, rr_val, beat_class))

        rr_n = [r for r in rr_all if r > 200]
        rr_n_classes = [p[2] for p in rr_points if p[1] > 200]
        filtered_points = [p for p in rr_points if _class_matches_filter(p[2], self._class_filter)]
        filtered_rr = [p[1] for p in filtered_points if p[1] > 200]
        filtered_classes = [p[2] for p in filtered_points if p[1] > 200]
        plot_rr = filtered_rr if len(filtered_rr) >= 2 else rr_n
        plot_classes = filtered_classes if len(filtered_rr) >= 2 else rr_n_classes
        if len(plot_rr) >= 2:
            rr_x = plot_rr[:-1]
            rr_y = plot_rr[1:]
            plot_beat_classes = plot_classes[:-1]  # use the first beat's class for each point
            lo = float(np.percentile(plot_rr, 5))
            hi = float(np.percentile(plot_rr, 95))
            if hi - lo < 250:
                center = float(np.median(plot_rr))
                lo = center - 500.0
                hi = center + 500.0
            lo = max(0.0, lo - 50.0)
            hi = hi + 50.0
            self._lorenz_canvas.set_data(rr_x, rr_y, x_range=(lo, hi), y_range=(lo, hi), beat_classes=plot_beat_classes)
        else:
            self._lorenz_canvas.set_data([], [])
        if hasattr(self, "_rr_trend_full"):
            trend_points = [(t, rr) for t, rr, _cls in filtered_points] if len(filtered_points) >= 2 else [(t, rr) for t, rr, _cls in rr_points]
            self._rr_trend_full.set_points(trend_points)
            # Only update zoom canvas separately if it's a different widget (OVERVIEW panel has two separate canvases;
            # the REPLAY panel aliases both to the same _hr_trend_canvas -- calling set_points on it twice would
            # overwrite the full data with the recent subset)
            if hasattr(self, "_rr_trend_zoom") and self._rr_trend_zoom is not self._rr_trend_full:
                recent = trend_points[-1200:] if len(trend_points) > 1200 else trend_points
                self._rr_trend_zoom.set_points(recent)
        self._update_overview_table(metrics_list, rr_n)
            
    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        """Update all 12 lead strips and compute Lorenz from data when no RR metrics."""
        if data is None or data.shape[0] < 12:
            return

        self._current_replay_frame = data
        strip_len = int(max(1, getattr(self, "_strip_length_sec", 10.0)) * 500)
        if data.shape[1] > strip_len:
            data = data[:, -strip_len:]

        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N <= 0:
            return

        lead_names = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

        # Ensure data has 12 channels
        if data.shape[0] < 12:
            new_data = np.zeros((12, data.shape[1]), dtype=data.dtype)
            new_data[:data.shape[0], :] = data
            for i in range(data.shape[0], 12):
                new_data[i, :] = 2048.0
            data = new_data

        # --- Force mathematical derivation of augmented limb leads ---
        # Einthoven's Law dictates these leads are perfectly locked to I and II.
        # We unconditionally derive them to overwrite any floating hardware noise.
        I_lead = data[0]
        II_lead = data[1]
        
        data[2] = (II_lead - 2048.0) - (I_lead - 2048.0) + 2048.0  # III = II - I
        # User wants aVR mapped to 0 to -4096. We synthesize it inverted and centered at -2048.
        data[3] = -((I_lead - 2048.0) + (II_lead - 2048.0)) / 2.0 - 2048.0
        data[4] = (I_lead - 2048.0) - (II_lead - 2048.0) / 2.0 + 2048.0  # aVL = I - II/2
        data[5] = (II_lead - 2048.0) - (I_lead - 2048.0) / 2.0 + 2048.0  # aVF = II - I/2

        # --- Feed all 12 lead strips ---
        lead_strips = getattr(self, "_lead_strips", {})
        for idx, lead in enumerate(lead_names):
            if idx < data.shape[0] and lead in lead_strips:
                lead_strips[lead].set_data(x, data[idx].copy(), beat_annotations=beat_annotations, start_sec=start_sec)

        # Mini strip follows the selected lead from the dropdown
        if hasattr(self, "_mini_strip") and data.shape[0] > 0:
            lead_idx = max(0, min(int(getattr(self, "_selected_lead_idx", 1)), data.shape[0] - 1))
            selected_lead = self._lead_combo.currentText() if hasattr(self, "_lead_combo") else lead_names[lead_idx]
            self._mini_strip.set_data(x, data[lead_idx].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
            try:
                self._mini_strip.lead_name = selected_lead
            except Exception:
                pass
            if hasattr(self, "_mini_lead_lbl"):
                self._mini_lead_lbl.setText(selected_lead)

        # Template thumbnails (use first 4 leads: I, II, III, aVR)
        for i, ts in enumerate(getattr(self, "_template_thumbs", [])):
            src = min(i, data.shape[0] - 1)
            center = N // 2
            w = min(220, max(80, N // 6))
            a = max(0, center - w // 2)
            b = min(N, center + w // 2)
            seg = data[src, a:b] if b > a else data[src, :]
            tx = np.linspace(0, len(seg) / 500.0, len(seg)) if len(seg) > 0 else []
            ts.set_data(tx, seg.copy() if len(seg) > 0 else data[src].copy())

        # --- On-the-fly Lorenz from data when Lorenz canvas has no data ---
        lorenz = getattr(self, "_lorenz_canvas", None)
        if lorenz is not None and (not lorenz._x):  # no RR data loaded from metrics
            self._compute_lorenz_from_signal(data, N)

    def _compute_lorenz_from_signal(self, data: np.ndarray, N: int):
        """Detect R-peaks in Lead II and populate the Lorenz scatter from the raw ECG data."""
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 500.0
            lead_ii = np.asarray(data[1], dtype=float) if data.shape[0] > 1 else np.asarray(data[0], dtype=float)
            # Bandpass 5-20 Hz to isolate QRS
            nyq = fs / 2.0
            b, a = butter(2, [5.0 / nyq, 20.0 / nyq], btype='band')
            filtered = filtfilt(b, a, lead_ii)
            squared = filtered ** 2
            win = max(1, int(0.15 * fs))
            mwa = np.convolve(squared, np.ones(win) / win, mode='same')
            threshold = max(np.mean(mwa) * 0.5, 1e-6)
            min_dist = max(1, int(0.3 * fs))
            peaks, _ = find_peaks(mwa, height=threshold, distance=min_dist)
            if len(peaks) >= 3:
                rr_ms = np.diff(peaks) / fs * 1000.0
                rr_valid = [float(r) for r in rr_ms if 250 < r < 2500]
                if len(rr_valid) >= 2:
                    rr_x = rr_valid[:-1]
                    rr_y = rr_valid[1:]
                    self._lorenz_canvas.set_data(rr_x, rr_y)
                    # Only fill the trend chart from live signal if full-recording metrics haven't already populated it
                    if hasattr(self, "_rr_trend_full") and not getattr(self._rr_trend_full, '_points', []):
                        rr_points = [(i * 0.5, rr) for i, rr in enumerate(rr_valid)]
                        self._rr_trend_full.set_points(rr_points)
                        self._rr_trend_zoom.set_points(rr_points[-400:] if len(rr_points) > 400 else rr_points)
        except Exception:
            pass

    def _update_overview_table(self, metrics_list: list, rr_n: list):
        if not hasattr(self, "_overview_table"):
            return
        summary_rows = self._compute_replay_overview(metrics_list, rr_n)
        self._overview_table.setRowCount(len(summary_rows))
        for r, (name, value) in enumerate(summary_rows):
            n_item = QTableWidgetItem(name)
            v_item = QTableWidgetItem(value)
            n_item.setForeground(QColor(UI_MUTED))
            v_item.setForeground(QColor(UI_TEXT))
            self._overview_table.setItem(r, 0, n_item)
            self._overview_table.setItem(r, 1, v_item)

    def _compute_replay_overview(self, metrics_list: list, rr_n: list) -> list:
        total_beats = len(rr_n)
        avg_hr = (60000.0 / float(np.mean(rr_n))) if rr_n else 0.0
        max_hr = (60000.0 / float(min(rr_n))) if rr_n else 0.0
        min_hr = (60000.0 / float(max(rr_n))) if rr_n else 0.0
        pauses = int(sum(1 for rr in rr_n if rr >= 2000))
        longest_rr = (max(rr_n) / 1000.0) if rr_n else 0.0
        ve = int(sum(1 for m in metrics_list if any("V" in str(a) for a in (m.get("arrhythmias", []) or []))))
        sve = int(sum(1 for m in metrics_list if any(("PAC" in str(a)) or ("SVE" in str(a)) for a in (m.get("arrhythmias", []) or []))))
        
        # Calculate percentages and durations
        total_chunks = len(metrics_list) if metrics_list else 1
        tachy_chunks = sum(1 for m in metrics_list if m.get('hr_mean', 0) > 100)
        brady_chunks = sum(1 for m in metrics_list if 0 < m.get('hr_mean', 0) < 60)
        af_chunks = sum(1 for m in metrics_list if any("AF" in str(a) for a in (m.get("arrhythmias", []) or [])))
        
        # Each chunk is roughly 4 seconds.
        chunk_dur = 4.0
        tachy_pct = (tachy_chunks / total_chunks) * 100.0
        brady_pct = (brady_chunks / total_chunks) * 100.0
        af_dur_str = f"{(af_chunks * chunk_dur) / 60.0:.1f} m"
        
        return [
            ("Total NNs", str(total_beats)),
            ("AVG HR", f"{avg_hr:.0f} bpm"),
            ("Max HR", f"{max_hr:.0f} bpm"),
            ("Min HR", f"{min_hr:.0f} bpm"),
            ("Longest RR Interval", f"{longest_rr:.2f}s"),
            ("RRI (>=2.0s)", str(pauses)),
            ("During Tachy (>100)", f"{tachy_pct:.1f}%"),
            ("During Brady (<60)", f"{brady_pct:.1f}%"),
            ("AF Duration", af_dur_str),
            ("V Total", str(ve)),
            ("S Total", str(sve)),
        ]


#  "  "  Helper canvas widgets  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 

class LorenzCanvas(QWidget):
    """Full-featured RR scatter / Poincar  plot with lasso selection, zoom,
    time-sharing view, DeltaRR mode, and right-click context menu."""

    beats_selected      = pyqtSignal(list)   # list of selected data indices
    beats_reclassified  = pyqtSignal(list, str)  # (indices, new_class)
    beats_deleted       = pyqtSignal(list)   # indices to remove

    # ------------------------------------------------------------------ init
    def __init__(self, parent=None):
        super().__init__(parent)
        # data
        self._x = []
        self._y = []
        self._beat_classes = []
        self._all_rr_points = []   # (t, rr, beat_class) for timesharing
        self._x_range = None
        self._y_range = None

        # display state
        self._view_mode    = "lorenz"     # "lorenz" | "delta_rr"
        self._display_mode = "complete"   # "complete" | "timesharing"
        self._zoom_level   = 1.0
        self._pixel_size   = 3            # dot radius in px
        self._rr_range     = (200, 2000)  # (min_ms, max_ms)

        # selection state
        self._selected_indices = set()
        self._is_dragging      = False
        self._drag_start       = None
        self._drag_current     = None

        self.setMinimumSize(200, 180)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)

    # ---------------------------------------------------------------- data
    def set_data(self, x, y, x_range=None, y_range=None, beat_classes=None, rr_points=None):
        self._x = list(x)
        self._y = list(y)
        self._beat_classes = list(beat_classes) if beat_classes is not None else []
        self._x_range = x_range
        self._y_range = y_range
        if rr_points is not None:
            self._all_rr_points = list(rr_points)
        self._selected_indices.clear()
        self.update()

    # ------------------------------------------------------- mouse events
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._is_dragging  = True
            self._drag_start   = event.pos()
            self._drag_current = event.pos()
            self._selected_indices.clear()
            self.update()

    def mouseMoveEvent(self, event):
        if self._is_dragging:
            self._drag_current = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._is_dragging:
            self._is_dragging = False
            self._finalize_lasso_selection()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selected_indices = set(range(len(self._x)))
            self.beats_selected.emit(list(self._selected_indices))
            self.update()

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self._zoom_level = max(0.2, self._zoom_level * 0.85)
        else:
            self._zoom_level = min(5.0, self._zoom_level * 1.15)
        self.update()

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{ background: {COL_DARK}; color: white; border: 1px solid {COL_GREEN_DRK}; }}
            QMenu::item:selected {{ background: {COL_GREEN_DRK}; }}
            QMenu::separator {{ height: 1px; background: {COL_GREEN_DRK}; }}
        """)

        #    View submenu   
        view_menu = menu.addMenu("   View")
        act_lorenz   = view_menu.addAction("RR Interval Scatter (Lorenz)")
        act_delta_rr = view_menu.addAction("DeltaRR Difference Scatter")
        act_lorenz.setCheckable(True);   act_lorenz.setChecked(self._view_mode == "lorenz")
        act_delta_rr.setCheckable(True); act_delta_rr.setChecked(self._view_mode == "delta_rr")
        act_lorenz.triggered.connect(lambda: self._set_view("lorenz"))
        act_delta_rr.triggered.connect(lambda: self._set_view("delta_rr"))

        #    Display submenu   
        disp_menu = menu.addMenu("   Display")
        act_complete = disp_menu.addAction("Complete Scatter")
        act_timeshare = disp_menu.addAction("Time-sharing Scatter (16 panels)")
        act_complete.setCheckable(True);   act_complete.setChecked(self._display_mode == "complete")
        act_timeshare.setCheckable(True);  act_timeshare.setChecked(self._display_mode == "timesharing")
        act_complete.triggered.connect(lambda: self._set_display("complete"))
        act_timeshare.triggered.connect(lambda: self._set_display("timesharing"))

        menu.addSeparator()

        #    Zoom   
        zoom_menu = menu.addMenu("   Zoom")
        zoom_menu.addAction("Zoom In  (+)").triggered.connect(lambda: self._do_zoom(True))
        zoom_menu.addAction("Zoom Out (-)").triggered.connect(lambda: self._do_zoom(False))
        zoom_menu.addAction("Reset Zoom").triggered.connect(lambda: self._reset_zoom())

        #    Pixel size   
        pix_menu = menu.addMenu("   Pixel Size")
        pix_menu.addAction("Small  (2 px)").triggered.connect(lambda: self._set_pixel(2))
        pix_menu.addAction("Medium (3 px)").triggered.connect(lambda: self._set_pixel(3))
        pix_menu.addAction("Large  (5 px)").triggered.connect(lambda: self._set_pixel(5))

        #    RR Display Range   
        rng_menu = menu.addMenu("   RR Display Range")
        rng_menu.addAction("Auto (200--2000 ms)").triggered.connect(lambda: self._set_rr_range(200, 2000))
        rng_menu.addAction("Normal (400--1200 ms)").triggered.connect(lambda: self._set_rr_range(400, 1200))
        rng_menu.addAction("Wide (100--3000 ms)").triggered.connect(lambda: self._set_rr_range(100, 3000))
        rng_menu.addAction("Custom...").triggered.connect(self._custom_rr_range_dialog)

        menu.addSeparator()

        #    Beat Attribute   
        has_sel = len(self._selected_indices) > 0
        attr_menu = menu.addMenu("   Beat Attribute")
        attr_menu.setEnabled(has_sel)
        for cls_label, cls_code in [("Normal (N)", "N"), ("Supraventricular (S)", "S"),
                                    ("Ventricular (V)", "V"), ("Paced (P)", "P"),
                                    ("AF / AFl", "AF"), ("Other", "Other"), ("Artifact (X)", "X")]:
            a = attr_menu.addAction(cls_label)
            a.triggered.connect(lambda checked=False, c=cls_code: self._reclassify(c))

        #    Delete   
        del_act = menu.addAction("   Delete Selected")
        del_act.setEnabled(has_sel)
        del_act.triggered.connect(self._delete_selected)

        menu.exec_(event.globalPos())

    # -------------------------------------------------------- private actions
    def _set_view(self, mode):
        self._view_mode = mode
        self._selected_indices.clear()
        self.update()

    def _set_display(self, mode):
        self._display_mode = mode
        self._selected_indices.clear()
        self.update()

    def _do_zoom(self, zoom_in):
        if zoom_in:
            self._zoom_level = max(0.2, self._zoom_level * 0.85)
        else:
            self._zoom_level = min(5.0, self._zoom_level * 1.15)
        self.update()

    def _reset_zoom(self):
        self._zoom_level = 1.0
        self.update()

    def _set_pixel(self, size):
        self._pixel_size = size
        self.update()

    def _set_rr_range(self, lo, hi):
        self._rr_range = (lo, hi)
        self._zoom_level = 1.0
        self.update()

    def _custom_rr_range_dialog(self):
        from PyQt5.QtWidgets import QDialog, QFormLayout, QSpinBox, QDialogButtonBox
        dlg = QDialog(self)
        dlg.setWindowTitle("Custom RR Display Range")
        dlg.setStyleSheet(f"background:{COL_DARK}; color:white;")
        form = QFormLayout(dlg)
        lo_spin = QSpinBox(); lo_spin.setRange(50, 3000); lo_spin.setValue(self._rr_range[0])
        hi_spin = QSpinBox(); hi_spin.setRange(100, 5000); hi_spin.setValue(self._rr_range[1])
        form.addRow("Min (ms):", lo_spin)
        form.addRow("Max (ms):", hi_spin)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept); btns.rejected.connect(dlg.reject)
        form.addRow(btns)
        if dlg.exec_() == QDialog.Accepted:
            self._set_rr_range(lo_spin.value(), hi_spin.value())

    def _reclassify(self, new_class):
        indices = list(self._selected_indices)
        for idx in indices:
            if idx < len(self._beat_classes):
                self._beat_classes[idx] = new_class
        self.beats_reclassified.emit(indices, new_class)
        self.update()

    def _delete_selected(self):
        indices = sorted(self._selected_indices, reverse=True)
        for idx in indices:
            if idx < len(self._x):
                self._x.pop(idx)
                self._y.pop(idx)
            if idx < len(self._beat_classes):
                self._beat_classes.pop(idx)
        self.beats_deleted.emit(list(self._selected_indices))
        self._selected_indices.clear()
        self.update()

    def _finalize_lasso_selection(self):
        if self._drag_start is None or self._drag_current is None:
            return
        # Lasso = ellipse defined by drag rect
        x0, y0 = self._drag_start.x(), self._drag_start.y()
        x1, y1 = self._drag_current.x(), self._drag_current.y()
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        rx = max(5, abs(x1 - x0) / 2.0)
        ry = max(5, abs(y1 - y0) / 2.0)
        new_sel = set()
        plot_coords = self._compute_plot_coords()
        for idx, (px, py) in enumerate(plot_coords):
            # Inside ellipse?
            if ((px - cx) / rx) ** 2 + ((py - cy) / ry) ** 2 <= 1.0:
                new_sel.add(idx)
        self._selected_indices = new_sel
        self.beats_selected.emit(list(new_sel))

    def _compute_plot_coords(self):
        """Return list of (px, py) pixel positions for current data points."""
        w, h = self.width(), self.height()
        left, top, right, bottom = 40, 16, 16, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)
        x_min, x_max, y_min, y_max = self._get_axis_range()
        rng_x = max(x_max - x_min, 1.0)
        rng_y = max(y_max - y_min, 1.0)
        xs, ys = self._get_plot_xy()
        coords = []
        for xv, yv in zip(xs, ys):
            px = int(left + (xv - x_min) / rng_x * plot_w)
            py = int(top + plot_h - (yv - y_min) / rng_y * plot_h)
            coords.append((px, py))
        return coords

    def _get_plot_xy(self):
        """Return (xs, ys) depending on view mode and rr_range filter."""
        rr_lo, rr_hi = self._rr_range
        if self._view_mode == "delta_rr":
            xs, ys, classes = [], [], []
            raw = self._x  # rr_n values
            for i in range(len(raw) - 1):
                rr_n   = raw[i]
                rr_np1 = raw[i+1]
                dv = rr_n - rr_np1
                xs.append(rr_n)
                ys.append(dv)
            return xs, ys
        else:
            filtered_x = [v for v in self._x if rr_lo <= v <= rr_hi]
            filtered_y = [v for v in self._y if rr_lo <= v <= rr_hi]
            n = min(len(filtered_x), len(filtered_y))
            return filtered_x[:n], filtered_y[:n]

    def _get_axis_range(self):
        """Return (x_min, x_max, y_min, y_max) with zoom applied."""
        rr_lo, rr_hi = self._rr_range
        if self._view_mode == "delta_rr":
            xs, ys = self._get_plot_xy()
            if not xs:
                return 200, 1500, -500, 500
            all_x = xs; all_y = ys
            x_center = float(np.median(all_x)) if all_x else 800
            y_center = float(np.median(all_y)) if all_y else 0
            x_half = max(500, float(np.percentile(np.abs(all_x), 95))) * self._zoom_level
            y_half = max(200, float(np.percentile(np.abs(all_y), 95))) * self._zoom_level
            return x_center - x_half, x_center + x_half, y_center - y_half, y_center + y_half

        # Standard Lorenz
        if self._x_range is not None:
            base_lo, base_hi = self._x_range
        else:
            all_vals = [v for v in self._x + self._y if rr_lo <= v <= rr_hi]
            if not all_vals:
                return rr_lo, rr_hi, rr_lo, rr_hi
            lo = float(np.percentile(all_vals, 5))
            hi = float(np.percentile(all_vals, 95))
            if hi - lo < 250:
                c = float(np.median(all_vals))
                lo, hi = c - 500.0, c + 500.0
            base_lo = max(0.0, lo - 50.0)
            base_hi = hi + 50.0

        center = (base_lo + base_hi) / 2.0
        half   = (base_hi - base_lo) / 2.0 * self._zoom_level
        lo = max(0.0, center - half)
        hi = center + half
        return lo, hi, lo, hi

    # ------------------------------------------------------------ paint
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(COL_BLACK))

        if not self._x or not self._y:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        if self._display_mode == "timesharing":
            self._paint_timesharing(painter)
        else:
            self._paint_complete(painter)

        # Draw lasso drag rubber-band
        if self._is_dragging and self._drag_start and self._drag_current:
            x0, y0 = self._drag_start.x(), self._drag_start.y()
            x1, y1 = self._drag_current.x(), self._drag_current.y()
            pen = QPen(QColor(0, 220, 255, 200), 1, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(0, 180, 255, 30)))
            lx, ly = min(x0, x1), min(y0, y1)
            lw, lh = abs(x1-x0), abs(y1-y0)
            painter.drawEllipse(lx, ly, max(4, lw), max(4, lh))

    def _paint_complete(self, painter):
        w, h = self.width(), self.height()
        left, top, right, bottom = 44, 16, 16, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)
        x_min, x_max, y_min, y_max = self._get_axis_range()
        rng_x = max(x_max - x_min, 1.0)
        rng_y = max(y_max - y_min, 1.0)

        def to_px(vx, vy):
            px = int(left + (vx - x_min) / rng_x * plot_w)
            py = int(top + plot_h - (vy - y_min) / rng_y * plot_h)
            return px, py

        # Border
        painter.setPen(QPen(QColor("#2C3E50"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(left, top, plot_w, plot_h)

        # Identity diagonal
        painter.setPen(QPen(QColor("#4A5D6E"), 1, Qt.SolidLine))
        if x_max > x_min:
            s = to_px(x_min, x_min); e = to_px(x_max, x_max)
            painter.drawLine(s[0], s[1], e[0], e[1])

        # Perpendicular grid lines
        range_span = x_max - x_min
        step = 1000.0 if range_span > 2000 else (500.0 if range_span > 600 else 250.0)
        grid_pen = QPen(QColor("#2C3E50"), 1, Qt.SolidLine)
        painter.setPen(grid_pen)
        min_sum = x_min + y_min; max_sum = x_max + y_max
        sum_val = int(np.floor(min_sum / step) * step)
        while sum_val <= max_sum:
            if sum_val > 0:
                pts = []
                yl = sum_val - x_min
                if y_min <= yl <= y_max: pts.append((x_min, yl))
                yr = sum_val - x_max
                if y_min <= yr <= y_max: pts.append((x_max, yr))
                xb = sum_val - y_min
                if x_min <= xb <= x_max: pts.append((xb, y_min))
                xt = sum_val - y_max
                if x_min <= xt <= x_max: pts.append((xt, y_max))
                uniq = []
                for p in pts:
                    if not any(np.allclose(p, up) for up in uniq): uniq.append(p)
                if len(uniq) == 2:
                    p1 = to_px(uniq[0][0], uniq[0][1]); p2 = to_px(uniq[1][0], uniq[1][1])
                    painter.drawLine(p1[0], p1[1], p2[0], p2[1])
            sum_val += step

        # Beat colour map
        color_map = {
            "N":     QColor(0, 210, 21, 255),
            "S":     QColor(255, 51, 51, 255),
            "V":     QColor(255, 153, 0, 255),
            "P":     QColor(0, 229, 255, 255),
            "AF":    QColor(255, 215, 0, 255),
            "X":     QColor(136, 136, 136, 255),
            "Other": QColor(0, 210, 21, 255),
        }
        xs, ys = self._get_plot_xy()
        r = self._pixel_size
        for i, (xv, yv) in enumerate(zip(xs, ys)):
            beat_class = self._beat_classes[i] if i < len(self._beat_classes) else "Other"
            color = color_map.get(beat_class, color_map["Other"])
            px, py = to_px(xv, yv)
            if i in self._selected_indices:
                painter.setPen(QPen(QColor(255, 255, 0, 220), 1))
                painter.setBrush(QBrush(color.lighter(160)))
                painter.drawEllipse(px - r - 2, py - r - 2, (r + 2) * 2, (r + 2) * 2)
            else:
                painter.setPen(QPen(color))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(px - r, py - r, r * 2, r * 2)

        # Axis ticks + labels
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.setFont(QFont("Arial", 8))
        start_tick = int(np.floor(x_min / step) * step)
        end_tick   = int(np.ceil(x_max / step) * step)
        tick = start_tick
        while tick <= end_tick:
            if x_min <= tick <= x_max:
                px = int(left + (tick - x_min) / rng_x * plot_w)
                painter.drawLine(px, top + plot_h, px, top + plot_h + 4)
                painter.drawText(QRect(px - 22, top + plot_h + 5, 44, 14), Qt.AlignCenter, str(int(tick)))
            if y_min <= tick <= y_max:
                py = int(top + plot_h - (tick - y_min) / rng_y * plot_h)
                painter.drawLine(left - 4, py, left, py)
                painter.drawText(QRect(left - 42, py - 7, 38, 14), Qt.AlignRight | Qt.AlignVCenter, str(int(tick)))
            tick += step

        # Axis titles
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        if self._view_mode == "delta_rr":
            painter.drawText(left + 4, h - 8, "RR(n) ms")
            painter.drawText(w - 60, 14, "DeltaRR(n+1)")
        else:
            painter.drawText(left + 4, h - 8, "RR(n) ms")
            painter.drawText(w - 62, 14, "RR(n+1)")

        # Mode badge
        badge = "DeltaRR" if self._view_mode == "delta_rr" else "Lorenz"
        painter.setFont(QFont("Arial", 7))
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(left + 4, top + 12, badge)

    def _paint_timesharing(self, painter):
        """Draw 16 mini scatter plots in a 4x4 grid."""
        w, h = self.width(), self.height()
        n_cols, n_rows = 4, 4
        cell_w = w // n_cols
        cell_h = h // n_rows

        # Build 16 time segments from _all_rr_points if available, else use _x/_y
        if self._all_rr_points:
            pts = self._all_rr_points
        else:
            pts = [(0, xv, cls) for xv, cls in zip(self._x,
                   self._beat_classes if self._beat_classes else ["N"] * len(self._x))]

        if not pts:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, "No RR data")
            return

        n_segs = 16
        seg_size = max(1, len(pts) // n_segs)
        segments = [pts[i * seg_size:(i + 1) * seg_size] for i in range(n_segs)]
        # remaining beats go to last segment
        if len(pts) > n_segs * seg_size:
            segments[-1] = segments[-1] + pts[n_segs * seg_size:]

        # Global range from all data
        all_rr = [p[1] for p in pts if self._rr_range[0] <= p[1] <= self._rr_range[1]]
        if not all_rr:
            return
        g_lo = max(0.0, float(np.percentile(all_rr, 5)) - 50.0)
        g_hi = float(np.percentile(all_rr, 95)) + 50.0
        if g_hi - g_lo < 250:
            c = float(np.median(all_rr))
            g_lo, g_hi = max(0, c - 400), c + 400
        g_rng = max(g_hi - g_lo, 1.0)

        color_map = {
            "N": QColor(0, 210, 21, 220), "S": QColor(255, 51, 51, 220),
            "V": QColor(255, 153, 0, 220), "P": QColor(0, 229, 255, 220),
            "AF": QColor(255, 215, 0, 220), "X": QColor(136, 136, 136, 220),
            "Other": QColor(0, 210, 21, 220),
        }

        for seg_idx, seg in enumerate(segments):
            col = seg_idx % n_cols
            row = seg_idx // n_cols
            ox = col * cell_w
            oy = row * cell_h
            pad = 4

            # Cell background + border
            painter.setPen(QPen(QColor("#2C3E50"), 1))
            painter.setBrush(QBrush(QColor("#080C12")))
            painter.drawRect(ox, oy, cell_w, cell_h)

            # Segment number
            painter.setFont(QFont("Arial", 7))
            painter.setPen(QPen(QColor("#4A5D6E")))
            painter.drawText(ox + 3, oy + 10, str(seg_idx + 1))

            # Build local pairs
            local_rr = [p[1] for p in seg if self._rr_range[0] <= p[1] <= self._rr_range[1]]
            local_cls = [p[2] if len(p) > 2 else "N" for p in seg
                         if self._rr_range[0] <= p[1] <= self._rr_range[1]]
            if len(local_rr) < 2:
                continue

            pw = cell_w - 2 * pad
            ph = cell_h - 2 * pad - 4

            def to_mini(vx, vy):
                mx = int(ox + pad + (vx - g_lo) / g_rng * pw)
                my = int(oy + pad + ph - (vy - g_lo) / g_rng * ph)
                return mx, my

            # Identity diagonal
            painter.setPen(QPen(QColor("#2C3E50"), 1))
            s = to_mini(g_lo, g_lo); e = to_mini(g_hi, g_hi)
            painter.drawLine(s[0], s[1], e[0], e[1])

            # Dots
            for i in range(len(local_rr) - 1):
                xv, yv = local_rr[i], local_rr[i+1]
                cls = local_cls[i] if i < len(local_cls) else "N"
                color = color_map.get(cls, color_map["Other"])
                px, py = to_mini(xv, yv)
                painter.setPen(QPen(color))
                painter.setBrush(QBrush(color))
                painter.drawEllipse(px - 1, py - 1, 2, 2)

        # Mode badge
        painter.setFont(QFont("Arial", 8, QFont.Bold))
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(2, h - 4, "Time-sharing  (16 panels)")








class ECGStripCanvas(QWidget):
    """Simple ECG strip renderer with interactive measurement tools."""
    def __init__(self, parent=None, height: int = 80, color: str = "#00FF00", pen_width: float = 0.7, lead_name: str = "", show_vertical_lines: bool = True, show_annotations: bool = True, disable_all_coloring: bool = False):
        super().__init__(parent)
        self._data = np.zeros(200)
        self._color = color
        self._pen_width = pen_width
        self.lead_name = lead_name
        self._start_sec = 0.0
        self._show_vertical_lines = show_vertical_lines  # Control whether to show R-peak vertical lines
        self._show_annotations = show_annotations  # Control whether to show N labels and RR numbers
        self._disable_all_coloring = disable_all_coloring  # CRITICAL: Disable ALL coloring (for replay/overview)
        self._gain = 1.0
        self._speed = 25
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        self._mode = TOOL_SELECT
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._caliper_line1 = None
        self._caliper_line2 = None
        # Ruler/Measuring ruler state: track start and end points separately
        self._ruler_start = None  # Start point (persists)
        self._ruler_end = None    # End point (current measurement)
        self._magnify_pos = None
        self._fs = 500.0
        self._filter_ba = None
        
        # Click-based vertical line tracking: only show line where user clicked
        self._clicked_beat_timestamp = None  # Timestamp of the clicked peak
        self._clicked_beat_label = None  # Label (N, S, V, etc.) of clicked peak
        self._clicked_beat_x_pos = None  # X pixel position of clicked beat
        
        # Drag selection: multiple selected beats
        self._selected_beats = []  # List of timestamps for all selected beats

    def _find_magnifier_host(self):
        parent = self.parentWidget()
        while parent is not None:
            if hasattr(parent, "set_magnifier_focus") and hasattr(parent, "clear_magnifier_focus"):
                return parent
            parent = parent.parentWidget()
        return None

    def _get_reader_start_time(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "_engine"):
                if hasattr(parent._engine, "_reader") and hasattr(parent._engine._reader, "start_time"):
                    return parent._engine._reader.start_time
            parent = parent.parent()
        return None

    def _magnifier_source_payload(self):
        return {
            "data": np.asarray(self._data, dtype=float).copy(),
            "speed": float(self._speed),
            "gain": float(self._gain),
            "lead_name": getattr(self, "lead_name", ""),
            "fs": float(self._fs),
        }

    def set_gain(self, gain: float):
        self._gain = gain
        self.update()

    def set_paper_speed(self, speed: int):
        self._speed = speed
        self.update()

    def set_mode(self, mode: str):
        host = self._find_magnifier_host()
        if host is not None and self._mode == TOOL_MAGNIFY and canonical_tool(mode) != TOOL_MAGNIFY:
            host.clear_magnifier_focus(self)
        self._mode = canonical_tool(mode)
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self._caliper_line1 = None
        self._caliper_line2 = None
        self._ruler_start = None
        self._ruler_end = None
        self.update()

    def clear_interaction(self):
        self._start_pos = None
        self._curr_pos = None
        self._hover_pos = None
        self._magnify_locked = False
        self._magnify_pos = None
        self._caliper_line1 = None
        self._caliper_line2 = None
        self._ruler_start = None
        self._ruler_end = None
        # Clear clicked beat
        self._clicked_beat_timestamp = None
        self._clicked_beat_label = None
        self._clicked_beat_x_pos = None
        self.update()

    def set_data(self, *args, beat_annotations=None, start_sec=0.0, structured_events=None, fast_preview=False):
        if len(args) == 2:
            raw_data = np.asarray(args[1], dtype=float)
        elif len(args) == 1:
            raw_data = np.asarray(args[0], dtype=float)
        else:
            raw_data = np.zeros(0)

        # NOTE: fast_preview no longer skips the low-pass filter below - an
        # earlier version did, to save time on in-between scrub frames, but
        # that displayed the raw unfiltered signal (baseline wander + HF
        # noise the filter normally removes) and made the trace look like
        # pure noise while dragging. Filtering always runs; fast_preview now
        # only skips the RR-interval recompute (self._rr_intervals), which
        # isn't drawn as part of the trace itself.
        if len(raw_data) > 15:
            try:
                if getattr(self, '_filter_ba', None) is None:
                    from scipy.signal import butter
                    self._filter_ba = butter(2, 25.0 / (self._fs / 2.0), btype='lowpass')
                from scipy.signal import filtfilt
                b, a = self._filter_ba
                self._data = filtfilt(b, a, raw_data)
            except Exception:
                self._data = raw_data
        else:
            self._data = raw_data

        self._beat_annotations = beat_annotations or []
        self._structured_events = structured_events or []
        self._start_sec = start_sec
        if fast_preview:
            self._rr_intervals = []
        else:
            self._rr_intervals = self._calculate_rr_intervals() if beat_annotations else []
        self.update()
    
    def _calculate_rr_intervals(self):
        """Calculate RR intervals (N-N intervals) between consecutive R-peaks."""
        if not self._beat_annotations:
            return []
        
        # Filter and sort beats by timestamp
        beats_in_window = []
        end_sec = self._start_sec + (len(self._data) / self._fs) if len(self._data) > 0 else self._start_sec
        
        for beat in self._beat_annotations:
            ts = beat.get('timestamp', 0.0)
            if self._start_sec <= ts <= end_sec:
                beats_in_window.append({
                    'timestamp': ts,
                    'label': beat.get('label', 'N')
                })
        
        beats_in_window.sort(key=lambda x: x['timestamp'])
        
        # Calculate intervals between consecutive beats
        intervals = []
        for i in range(len(beats_in_window) - 1):
            curr_beat = beats_in_window[i]
            next_beat = beats_in_window[i + 1]
            
            # Calculate RR interval in ms
            rr_ms = (next_beat['timestamp'] - curr_beat['timestamp']) * 1000.0
            
            intervals.append({
                'start_ts': curr_beat['timestamp'],
                'end_ts': next_beat['timestamp'],
                'rr_ms': rr_ms,
                'start_label': curr_beat['label'],
                'end_label': next_beat['label']
            })
        
        return intervals

    def mousePressEvent(self, event):
        # Let the event propagate to parent's eventFilter for vertical line handling
        # Only consume the event if we're in a tool mode
        
        if self._mode == TOOL_MAGNIFY and event.button() == Qt.LeftButton:
            # Click-to-lock magnifier: each click moves the zoom lens to that point.
            # Switching away from the tool clears the lock.
            self._magnify_locked = True
            self._magnify_pos = event.pos()
            self._hover_pos = event.pos()
            host = self._find_magnifier_host()
            if host is not None:
                host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
            self.update()
            return

        if self._mode == TOOL_CALIPER and event.button() == Qt.LeftButton:
            if self._caliper_line1 is None:
                self._caliper_line1 = event.pos().x()
                self._caliper_line2 = None
            else:
                self._caliper_line2 = event.pos().x()
            self.update()
            return

        # Ruler/Measuring ruler: handle two-click measurement
        if self._mode == TOOL_RULER and event.button() == Qt.LeftButton:
            if self._ruler_start is None:
                # First click: set start point
                self._ruler_start = event.pos()
                self._ruler_end = None
            else:
                # Second click: set end point (keeps start point fixed)
                self._ruler_end = event.pos()
            self.update()
            return

        if self._mode != TOOL_SELECT:
            self._start_pos = event.pos()
            self._curr_pos = event.pos()
            self._hover_pos = event.pos()
            self.update()
            return
        
        # In SELECT mode, don't consume the event - let it propagate to eventFilter
        # This allows the Full Disclosure dialog to handle vertical line drawing
        event.ignore()

    def mouseMoveEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if self._magnify_locked:
                if event.buttons() & Qt.LeftButton:
                    self._magnify_pos = event.pos()
                    host = self._find_magnifier_host()
                    if host is not None:
                        host.set_magnifier_focus(self, self._magnifier_source_payload(), event.pos())
                self.update()
                return
            self._hover_pos = event.pos()
            self.update()
            return
        
        # For caliper tool, show preview of second line while hovering (if first line is set)
        if self._mode == TOOL_CALIPER:
            if self._caliper_line1 is not None and self._caliper_line2 is None:
                # First line is set, show preview of second line as hover
                self._hover_pos = event.pos()
                self.update()
            return
        
        # For ruler tool, show preview of end point while hovering (if start point is set)
        if self._mode == TOOL_RULER:
            if self._ruler_start is not None and self._ruler_end is None:
                # Start point is set, show preview of end point as hover
                self._hover_pos = event.pos()
                self.update()
            return
            
        self._hover_pos = event.pos()
        if self._mode != TOOL_SELECT and self._start_pos is not None:
            self._curr_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._mode == TOOL_MAGNIFY:
            if not self._magnify_locked:
                self._hover_pos = event.pos()
            self.update()
            return
        if self._mode != TOOL_SELECT:
            self._curr_pos = event.pos()
            self.update()

    def leaveEvent(self, event):
        if not self._magnify_locked:
            self._hover_pos = None
        self.update()
        super().leaveEvent(event)

    def _get_display_signal(self):
        if self._data.size < 2:
            return np.array([]), 0.0, 1.0
        sig = np.asarray(self._data, dtype=float)
        baseline = float(np.median(sig))
        
        if self.lead_name == "aVR":
            # aVR: shift so baseline centers at -2048
            target_center = -2048.0
            shift = target_center - baseline
            d = sig + shift
            mn = -4096.0
            rng = 4096.0
        else:
            # All other leads: shift so baseline centers at 2048
            target_center = 2048.0
            shift = target_center - baseline
            d = sig + shift
            mn = 0.0
            rng = 4096.0
        
        # Apply gain: higher gain = taller waves
        # Center around baseline, apply gain, then shift back
        if hasattr(self, '_gain') and self._gain != 1.0:
            center = (mn + rng / 2.0)
            d = center + (d - center) * self._gain
            
        return d, mn, rng

    def _x_to_index(self, x: int, width: int, n: int) -> int:
        if n <= 1 or width <= 1:
            return 0
        x = max(0, min(x, width - 1))
        return int(round((x / float(width - 1)) * (n - 1)))

    def _check_and_store_clicked_beat(self, click_x: int):
        """Check if user clicked near a beat peak and store it for display."""
        w = self.width()
        if w <= 0 or not hasattr(self, '_data') or self._data.size < 2:
            self._clicked_beat_timestamp = None
            self._clicked_beat_label = None
            self._clicked_beat_x_pos = None
            return
        
        end_sec = self._start_sec + len(self._data) / self._fs
        tolerance_pixels = 15  # Reasonable tolerance for user clicking
        
        # Try to snap to the closest detected peak timestamp first
        parent_dialog = None
        curr = self.parent()
        while curr is not None:
            if hasattr(curr, '_detected_r_peaks'):
                parent_dialog = curr
                break
            curr = curr.parent()
            
        # 1. Try to snap to detected R-peaks from parent_dialog
        if parent_dialog and hasattr(parent_dialog, '_detected_r_peaks') and parent_dialog._detected_r_peaks:
            best_ts = None
            best_dist_pixels = 999999
            for ts in parent_dialog._detected_r_peaks:
                pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                beat_x = int(pct * w)
                dist = abs(click_x - beat_x)
                if dist < best_dist_pixels:
                    best_dist_pixels = dist
                    best_ts = ts
                    
            if best_dist_pixels <= tolerance_pixels:
                self._clicked_beat_timestamp = best_ts
                self._clicked_beat_label = 'N'
                if hasattr(self, '_beat_annotations') and self._beat_annotations:
                    for beat in self._beat_annotations:
                        if abs(beat.get('timestamp', 0) - best_ts) < 0.05:
                            self._clicked_beat_label = beat.get('label', 'N')
                            break
                pct = (best_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                self._clicked_beat_x_pos = int(pct * w)
                self.update()
                return
        
        # 2. Try to snap to provided beat_annotations
        elif hasattr(self, '_beat_annotations') and self._beat_annotations:
            best_ts = None
            best_dist_pixels = 999999
            for beat in self._beat_annotations:
                ts = beat.get('timestamp', 0.0)
                if self._start_sec <= ts <= end_sec:
                    pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    beat_x = int(pct * w)
                    dist = abs(click_x - beat_x)
                    if dist < best_dist_pixels:
                        best_dist_pixels = dist
                        best_ts = ts
                        
            if best_dist_pixels <= tolerance_pixels:
                self._clicked_beat_timestamp = best_ts
                self._clicked_beat_label = 'N'
                for beat in self._beat_annotations:
                    if abs(beat.get('timestamp', 0) - best_ts) < 0.05:
                        self._clicked_beat_label = beat.get('label', 'N')
                        break
                pct = (best_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                self._clicked_beat_x_pos = int(pct * w)
                self.update()
                return
        
        # 3. Fallback: Check if click is actually ON or VERY NEAR a local maximum
        # Convert click position to sample index
        click_sample_idx = int((click_x / float(w)) * len(self._data))
        click_sample_idx = max(0, min(click_sample_idx, len(self._data) - 1))
        
        # Look in a small neighborhood (+/-20ms) around the click
        neighborhood_size = max(5, int(0.02 * self._fs))  # 20ms window
        start_check = max(0, click_sample_idx - neighborhood_size)
        end_check = min(len(self._data), click_sample_idx + neighborhood_size)
        
        if end_check > start_check:
            neighborhood = self._data[start_check:end_check]
            
            # Find the maximum value in the neighborhood
            local_max_idx = np.argmax(neighborhood)
            local_max_sample_idx = start_check + local_max_idx
            local_max_value = neighborhood[local_max_idx]
            
            # Check if this is a significant peak (not just noise)
            # Compare to median baseline of entire strip
            baseline = np.median(self._data)
            peak_height = local_max_value - baseline
            
            # Need at least 100 ADC units above baseline (adjust based on your data range)
            # This filters out flat areas and noise
            overall_range = np.max(self._data) - np.min(self._data)
            min_peak_height = 0.2 * overall_range  # Peak must be at least 20% of total range
            
            if peak_height > min_peak_height:
                # Found a significant peak! Now check if it's close enough to click
                ts = self._start_sec + (local_max_sample_idx / self._fs)
                pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                beat_x = int(pct * w)
                
                distance = abs(click_x - beat_x)
                
                if distance <= tolerance_pixels:
                    # This is a valid peak click!
                    self._clicked_beat_timestamp = ts
                    self._clicked_beat_label = 'N'
                    self._clicked_beat_x_pos = beat_x
                    self.update()
                    return
        
        # No valid peak found near click - clear everything
        self._clicked_beat_timestamp = None
        self._clicked_beat_label = None
        self._clicked_beat_x_pos = None
        self.update()

    def _find_beats_in_range(self, start_x: int, end_x: int):
        """Find all R-peaks EXACTLY between start_x and end_x pixel positions."""
        w = self.width()
        if w <= 0 or not hasattr(self, '_data') or self._data.size < 2:
            return []
        
        # Convert pixel positions to sample indices PRECISELY
        start_sample = int((start_x / float(w)) * len(self._data))
        end_sample = int((end_x / float(w)) * len(self._data))
        
        # Clamp to valid range
        start_sample = max(0, min(start_sample, len(self._data) - 1))
        end_sample = max(0, min(end_sample, len(self._data) - 1))
        
        # Ensure start < end
        if start_sample > end_sample:
            start_sample, end_sample = end_sample, start_sample
        
        # Ensure minimum range
        if (end_sample - start_sample) < 20:
            return []
        
        try:
            # Extract ONLY the data in the selected range
            range_data = self._data[start_sample:end_sample + 1]
            
            beat_timestamps = []
            end_sec = self._start_sec + len(self._data) / self._fs
            
            # Peak detection parameters
            min_samples_between = int(0.3 * self._fs)  # 300ms minimum
            
            # Use FULL DATA baseline for consistency
            full_baseline = np.median(self._data)
            full_range = np.max(self._data) - np.min(self._data)
            threshold = full_baseline + 0.25 * full_range
            
            last_peak_sample = start_sample - min_samples_between - 1
            
            # Find local maxima ONLY within the selected range
            for i in range(len(range_data)):
                current_val = range_data[i]
                
                # Skip if below threshold
                if current_val <= threshold:
                    continue
                
                # Calculate actual sample index (in full data array)
                actual_idx = start_sample + i
                
                # Check if this is a local maximum
                is_local_max = False
                
                if i == 0:
                    # First point in range - only compare with right
                    if len(range_data) > 1 and current_val > range_data[i + 1]:
                        is_local_max = True
                elif i == len(range_data) - 1:
                    # Last point in range - only compare with left
                    if current_val > range_data[i - 1]:
                        is_local_max = True
                else:
                    # Middle points - compare both sides
                    if current_val > range_data[i - 1] and current_val > range_data[i + 1]:
                        is_local_max = True
                
                # Check spacing from previous peak
                if not is_local_max:
                    continue
                
                if (actual_idx - last_peak_sample) < min_samples_between:
                    continue
                
                # FINAL CHECK: Ensure beat is within selected range
                if actual_idx < start_sample or actual_idx > end_sample:
                    continue
                
                ts = self._start_sec + (actual_idx / self._fs)
                
                # Final timestamp validation
                if self._start_sec <= ts <= end_sec:
                    beat_timestamps.append(ts)
                    last_peak_sample = actual_idx
            
            return sorted(beat_timestamps)
            
        except Exception as e:
            return []

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        minor_pen = QPen(QColor(COL_GRID_MINOR))
        minor_pen.setWidth(1)
        major_pen = QPen(QColor(COL_GRID_MAJOR))
        major_pen.setWidth(1)
        for gx in range(0, w, 20):
            painter.setPen(major_pen if gx % 100 == 0 else minor_pen)
            painter.drawLine(gx, 0, gx, h)
        for gy in range(0, h, 20):
            painter.setPen(major_pen if gy % 100 == 0 else minor_pen)
            painter.drawLine(0, gy, w, gy)

        if self._data.size < 2:
            return
        d, mn, rng = self._get_display_signal()
        if d.size < 2:
            return
        
        # --- Paper speed: control how many samples are visible per screen width ---
        # At 25mm/s: show all data. At 50mm/s: stretch (show half). At 12.5mm/s: compress (show double).
        speed_factor = max(0.25, min(float(self._speed) / 25.0, 4.0))
        n_visible = max(2, int(round(len(d) / speed_factor)))
        if n_visible < len(d):
            d = d[-n_visible:]   # show the most recent n_visible samples (stretched)
        # (if n_visible >= len(d) we show all data, which appears compressed at slow speed)
        # Apply speed factor to x_scale so the waveform stretches/compresses visually
        x_scale = w / max(1, len(d) - 1) * speed_factor
        
        # Determine colored intervals based on annotations and structured events
        colored_intervals = []
        end_sec = self._start_sec + len(d) / self._fs
        
        # 1. Color full regions for arrhythmias from _structured_events
        # CRITICAL FIX: Only apply structured event coloring if disable_all_coloring is False
        # For replay/overview panels (disable_all_coloring=True), always show plain green waveforms
        # For Full Disclosure (disable_all_coloring=False), show waveform coloring regardless of show_annotations
        if not self._disable_all_coloring and hasattr(self, '_structured_events') and self._structured_events:
            # Expanded color mapping for all arrhythmia types
            label_colors = {
                "V": "#FF3333",      # Ventricular Premature - Red
                "AF": "#FF00FF",     # Atrial Fibrillation - Magenta
                "S": "#00FFFF",      # Sinus Bradycardia/Tachycardia - Cyan
                "P": "#FF00FF",      # Paced/AV Blocks - Magenta
                "C": "#FFA500",      # Conduction Blocks - Orange
                "T": "#9932CC",      # TV Paced - Purple
                "A": "#FFFF00",      # ACLS - Yellow
                "X": "#0000FF",      # Asystole/Artifact - Blue
            }
            # Go through events and find regions
            for i, ev in enumerate(self._structured_events):
                ev_ts = float(ev.get('timestamp', 0.0) or 0.0)
                region_start_ts = ev_ts
                
                # Quickly determine region end to skip invisible events
                region_end_ts = ev.get('end_timestamp')
                if region_end_ts is None:
                    if i + 1 < len(self._structured_events):
                        region_end_ts = float(self._structured_events[i+1].get('timestamp', 0.0) or 0.0)
                    else:
                        region_end_ts = end_sec + 10.0
                else:
                    region_end_ts = float(region_end_ts)
                    
                if region_end_ts < self._start_sec or region_start_ts > end_sec:
                    continue
                    
                ev_lbl = str(ev.get('label', '')).lower()
                ev_lbl_orig = str(ev.get('label', ''))
                source = str(ev.get('source', '')).lower()
                is_manual_structured = source in {'manual', 'manual_parallel_multi', 'restored_parallel_multi'}

                # Check if this event starts an arrhythmia
                active_label = 'N'
                # Check for single-letter labels first (from manual marking)
                if ev_lbl == 'v':
                    active_label = 'V'
                elif ev_lbl == 's':
                    active_label = 'S'
                elif ev_lbl == 'af':
                    active_label = 'AF'
                elif ev_lbl == 'p':
                    active_label = 'P'
                elif ev_lbl == 'c':  # Conduction blocks
                    active_label = 'C'
                elif ev_lbl == 't':  # TV Paced
                    active_label = 'T'
                elif ev_lbl == 'a':  # ACLS
                    active_label = 'A'
                elif ev_lbl == 'x':
                    active_label = 'X'
                # Check for full label names (from auto-detection)
                elif 'asystole' in ev_lbl:
                    active_label = 'X'
                elif 'ventricular fibrillation' in ev_lbl or 'vfib' in ev_lbl:
                    active_label = 'V'
                elif 'ventricular tachycardia' in ev_lbl or 'vtach' in ev_lbl:
                    active_label = 'V'
                elif 'atrial fibrillation' in ev_lbl or 'afib' in ev_lbl:
                    active_label = 'AF'
                elif 'atrial flutter' in ev_lbl or 'aflutter' in ev_lbl:
                    active_label = 'AF'
                elif 'sinus bradycardia' in ev_lbl:
                    active_label = 'S'
                elif 'sinus tachycardia' in ev_lbl:
                    active_label = 'S'
                elif 'bradycardia (non-sinus)' in ev_lbl:
                    active_label = 'S'
                elif 'tachycardia (non-sinus)' in ev_lbl:
                    active_label = 'S'
                elif '1st-degree av block' in ev_lbl or '2nd-degree av block' in ev_lbl or '3rd-degree av block' in ev_lbl:
                    active_label = 'P'
                elif 'right bundle branch block' in ev_lbl or 'left bundle branch block' in ev_lbl:
                    active_label = 'P'
                elif 'premature ventricular contraction' in ev_lbl or 'pvc' in ev_lbl:
                    active_label = 'V'
                elif 'premature atrial contraction' in ev_lbl or 'pac' in ev_lbl:
                    active_label = 'S'
                elif 'st elevation' in ev_lbl or 'st depression' in ev_lbl:
                    active_label = 'X'
                
                # Also check the original label (case-sensitive) for single letters
                if active_label == 'N':
                    if ev_lbl_orig == 'V':
                        active_label = 'V'
                    elif ev_lbl_orig == 'S':
                        active_label = 'S'
                    elif ev_lbl_orig == 'AF':
                        active_label = 'AF'
                    elif ev_lbl_orig == 'P':
                        active_label = 'P'
                    elif ev_lbl_orig == 'X':
                        active_label = 'X'

                if active_label != 'N':
                    # Use color from event if available, otherwise use label_colors
                    color = ev.get('color', label_colors.get(active_label, "#FF3333"))
                    # Calculate overlapping indices
                    if region_end_ts >= self._start_sec and region_start_ts <= end_sec:
                        # Determine if this arrhythmia should color the whole region or just QRS complexes.
                        # Asystole, Artifact, and Ventricular Fibrillation lack normal QRS and should color the whole region.
                        # Tachycardia, Bradycardia, AFib, PVC, PAC etc. should color ONLY the QRS complexes.
                        color_whole_region = is_manual_structured and ('ventricular fibrillation' in ev_lbl or 'vfib' in ev_lbl or 
                                              'asystole' in ev_lbl or 'artifact' in ev_lbl or active_label == 'X')
                        
                        r_peak_tss = []
                        if not color_whole_region:
                            if hasattr(self, '_beat_annotations') and self._beat_annotations:
                                r_peak_tss = [b.get('timestamp', 0.0) for b in self._beat_annotations]
                            else:
                                r_peak_tss = getattr(self, '_detected_peaks_cache', [])
                                if not r_peak_tss:
                                    parent = self.parent()
                                    while parent is not None:
                                        if hasattr(parent, '_detected_r_peaks'):
                                            r_peak_tss = parent._detected_r_peaks
                                            break
                                        parent = parent.parent()
                        
                        # Find peaks inside this region
                        region_peaks = [ts for ts in r_peak_tss if region_start_ts <= ts <= region_end_ts]
                        
                        if color_whole_region or (not region_peaks and is_manual_structured):
                            # Color the entire region (fallback for no peaks, or explicitly requested for X / VFib)
                            start_idx = max(0, int((region_start_ts - self._start_sec) * self._fs))
                            end_idx = min(len(d) - 1, int((region_end_ts - self._start_sec) * self._fs))
                            if start_idx < end_idx:
                                colored_intervals.append((start_idx, end_idx, color))
                        else:
                            # Color ONLY the QRS complexes (+/- 60ms around each R-peak)
                            for ts in region_peaks:
                                qrs_start_ts = max(self._start_sec, ts - 0.06)
                                qrs_end_ts = min(end_sec, ts + 0.06)
                                start_idx = max(0, int((qrs_start_ts - self._start_sec) * self._fs))
                                end_idx = min(len(d) - 1, int((qrs_end_ts - self._start_sec) * self._fs))
                                if start_idx < end_idx:
                                    colored_intervals.append((start_idx, end_idx, color))
        

        colored_intervals.sort(key=lambda x: x[0])
        
        default_pen = QPen(QColor(self._color))
        default_pen.setWidthF(self._pen_width)

        n = len(d)
        # Vectorized coordinate computation (numpy) instead of a per-sample Python loop.
        xs = np.arange(n, dtype=np.float64) * x_scale
        ys = h - (d - mn) / rng * h

        # Build the whole trace as a single QPainterPath and draw it in one call.
        # This replaces issuing one drawLine() call per sample (extremely slow for
        # large windows like 1 Min / 2 Min, and the main cause of scrollbar-drag lag),
        # cutting per-frame Qt paint calls from O(n) down to O(1) + a handful for
        # colored arrhythmia/annotation sub-segments.
        painter.setPen(default_pen)
        trace_path = QPainterPath()
        trace_path.moveTo(xs[0], ys[0])
        for xv, yv in zip(xs[1:], ys[1:]):
            trace_path.lineTo(xv, yv)
        painter.drawPath(trace_path)

        # Overlay colored sub-segments (arrhythmia regions / annotated beats) on top
        # of the base trace — only a handful of these exist per window, so drawing
        # each as its own short path is cheap.
        for start_idx, end_idx, color in colored_intervals:
            start_idx = max(0, start_idx)
            end_idx = min(n - 1, end_idx)
            if end_idx <= start_idx:
                continue
            seg_pen = QPen(QColor(color))
            seg_pen.setWidthF(self._pen_width * 2.0)
            painter.setPen(seg_pen)
            seg_path = QPainterPath()
            seg_path.moveTo(xs[start_idx], ys[start_idx])
            for xv, yv in zip(xs[start_idx + 1:end_idx + 1], ys[start_idx + 1:end_idx + 1]):
                seg_path.lineTo(xv, yv)
            painter.drawPath(seg_path)

        # --- Draw Auto-Detected Arrhythmia Labels ---
        # (REMOVED: Floating overlay labels have been migrated to the ArrhythmiaBadgeBar at the bottom of the window)


            
        # --- Draw Clinical Beat Annotations ---
        lead_name = getattr(self, 'lead_name', '')
        show_vertical_lines = getattr(self, '_show_vertical_lines', True)
        
        detected_peaks = []
        annotated_beats = {}  # Map timestamp to beat annotation dictionary
        end_sec = self._start_sec + (len(d) / self._fs) if len(d) > 0 else self._start_sec
        
        # First, get annotated beats from annotations if available
        if hasattr(self, '_beat_annotations') and self._beat_annotations:
            for beat in self._beat_annotations:
                ts = beat['timestamp']
                if self._start_sec <= ts <= end_sec:
                    annotated_beats[ts] = beat
        
        if lead_name == 'I' and show_vertical_lines:
            # Only redo R-peak detection when the underlying data has actually
            # changed (id(self._data) changes every time set_data() gets new
            # samples). Otherwise reuse the last result — a plain repaint
            # (resize, overlay refresh, etc.) should not re-run scipy.find_peaks
            # over the whole window again.
            data_key = id(self._data)
            if getattr(self, '_peak_cache_key', None) == data_key and getattr(self, '_peak_cache_start_sec', None) == self._start_sec:
                detected_peaks = self._detected_peaks_cache
            else:
                # Detect R-peaks in real-time from the ECG signal
                if len(d) > 50:
                    try:
                        from scipy.signal import find_peaks
                        
                        # Simple R-peak detection
                        # 1. Find local maxima in the signal
                        signal = np.asarray(d, dtype=float)
                        
                        # Normalize signal for better peak detection
                        sig_mean = np.mean(signal)
                        sig_std = np.std(signal)
                        if sig_std > 0:
                            normalized = (signal - sig_mean) / sig_std
                        else:
                            normalized = signal - sig_mean
                        
                        # Find peaks with minimum distance (0.3s = 150 samples at 500Hz)
                        # and height threshold (signal should be above mean)
                        min_distance = int(0.3 * self._fs)  # Minimum 300ms between R-peaks
                        peaks, properties = find_peaks(normalized, 
                                                       distance=min_distance,
                                                       height=0.5,  # Above 0.5 std
                                                       prominence=0.3)
                        
                        # Convert peak indices to timestamps
                        for peak_idx in peaks:
                            ts = self._start_sec + (peak_idx / self._fs)
                            detected_peaks.append(ts)
                        
                        # Store detected peaks in parent panel for cross-lead vertical lines
                        parent = self.parent()
                        while parent is not None:
                            if hasattr(parent, '_detected_r_peaks'):
                                parent._detected_r_peaks = detected_peaks
                                parent._r_peak_start_sec = self._start_sec
                                parent._r_peak_data_len = len(d)
                                break
                            parent = parent.parent()
                            
                    except Exception as e:
                        print(f"[ECGStripCanvas] R-peak detection error: {e}")

                self._detected_peaks_cache = detected_peaks
                self._peak_cache_key = data_key
                self._peak_cache_start_sec = self._start_sec
            
            # Calculate and store RR intervals from detected peaks (always keep for internal use)
            if detected_peaks:
                self._rr_intervals = []
                for i in range(len(detected_peaks) - 1):
                    curr_ts = detected_peaks[i]
                    next_ts = detected_peaks[i + 1]
                    rr_ms = (next_ts - curr_ts) * 1000.0
                    
                    self._rr_intervals.append({
                        'start_ts': curr_ts,
                        'end_ts': next_ts,
                        'rr_ms': rr_ms,
                        'start_label': 'N',
                        'end_label': 'N'
                    })

        # Draw beat labels ONLY for:
        #   1. Explicitly annotated beats (from _beat_annotations) with non-N labels (manual marks)
        #   2. Detected peaks that fall inside an active arrhythmia structured event (non-N)
        # Default auto-detected N labels are intentionally suppressed in Full Disclosure view.
        if self._show_annotations:
            try:
                from ecg.holter.holter_summary_calc import get_template_beats_for_badges
                
                # Use beat annotations (which includes all beats in window) and structured events
                window_beats = getattr(self, '_beat_annotations', [])
                window_events = getattr(self, '_structured_events', [])
                badges = get_template_beats_for_badges(window_beats, window_events)
                
                # Add explicitly manual annotations if not already there
                manual_badges = []
                for annot_ts, beat in annotated_beats.items():
                    if beat.get('is_manual', False) and beat.get('label', 'N') != 'N':
                        lbl = beat.get('label', 'N')
                        short_code = lbl
                        if '(' in lbl and ')' in lbl:
                            short_code = lbl.split('(')[1].split(')')[0]
                        # Only add if not already in badges at similar timestamp
                        if not any(abs(b['timestamp'] - annot_ts) < 0.1 for b in badges):
                            color = beat.get('color', '#FFFF00')
                            if not color or color == '#FFFF00': # fallback
                                 if short_code == 'V': color = "#FF3333"
                                 elif short_code == 'S': color = "#00FFFF"
                                 elif short_code in ['AF', 'P']: color = "#FF00FF"
                            manual_badges.append({
                                'timestamp': annot_ts,
                                'code': short_code,
                                'name': lbl,
                                'color': color
                            })
                
                all_badges = badges + manual_badges
                
                if all_badges:
                    start_time = self._get_reader_start_time()
                    from datetime import datetime
                    
                    for badge in all_badges:
                        ts = badge['timestamp']
                        if not (self._start_sec <= ts <= end_sec):
                            continue
                        
                        pct = (ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                        bx = int(pct * w)
                        
                        color = badge['color']
                        code_str = f"[{badge['code']}]"
                        name_str = badge['name']
                        time_str = datetime.fromtimestamp(start_time + ts).strftime('%H:%M:%S') if start_time else ""
                        
                        # Set up fonts
                        restore_font = painter.font()
                        font = painter.font()
                        font.setPixelSize(10)
                        font.setBold(True)
                        painter.setFont(font)
                        
                        fm = painter.fontMetrics()
                        code_w = fm.horizontalAdvance(code_str)
                        name_w = fm.horizontalAdvance(name_str)
                        
                        font.setBold(False)
                        painter.setFont(font)
                        fm_time = painter.fontMetrics()
                        time_w = fm_time.horizontalAdvance(time_str)
                        
                        padding = 4
                        spacing = 4
                        total_w = code_w + name_w + time_w + (spacing * 2) + (padding * 2)
                        total_h = fm.height() + padding * 2
                        
                        # Position badge slightly above the trace
                        rect_x = bx - total_w // 2
                        rect_y = 5
                        
                        # Draw pill background (transparent dark with cyan border)
                        painter.setPen(QPen(QColor(color), 1))
                        painter.setBrush(QColor(20, 20, 20, 200)) # Dark transparent
                        painter.drawRoundedRect(rect_x, rect_y, total_w, total_h, 4, 4)
                        
                        # Draw code (cyan, bold)
                        font.setBold(True)
                        painter.setFont(font)
                        curr_x = rect_x + padding
                        painter.setPen(QPen(QColor(color)))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, code_str)
                        curr_x += code_w + spacing
                        
                        # Draw name (white, bold)
                        painter.setPen(QPen(QColor("#FFFFFF")))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, name_str)
                        curr_x += name_w + spacing
                        
                        # Draw time (light blue, normal)
                        font.setBold(False)
                        painter.setFont(font)
                        painter.setPen(QPen(QColor("#A0C4E8")))
                        painter.drawText(curr_x, rect_y + fm.ascent() + padding, time_str)
                        
                        # Restore font
                        painter.setFont(restore_font)
            except Exception as e:
                print(f"[ECGStripCanvas] Error drawing beat badges: {e}")

        
        # --- Draw square boxes for selected beats (only on Lead I) ---
        # NOTE: Vertical lines are now drawn by VerticalLineOverlay to avoid gaps
        end_sec = self._start_sec + len(d) / self._fs
        
        # ONLY DRAW BOXES ON LEAD I
        if lead_name == 'I':
            # Check if we have multiple selected beats (drag selection)
            if hasattr(self, '_selected_beats') and self._selected_beats and len(self._selected_beats) > 0:
                # Draw boxes for all selected beats with their specific colors
                painter.setBrush(Qt.NoBrush)
                
                for beat_ts in self._selected_beats:
                    # Verify beat is actually in this canvas's visible range
                    if self._start_sec <= beat_ts <= end_sec:
                        pct = (beat_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                        bx = int(pct * w)
                        
                        # Find the beat's color from annotations
                        beat_color = "#FFFF00"  # Default yellow
                        beat_label = "N"
                        if hasattr(self, '_beat_annotations'):
                            for beat in self._beat_annotations:
                                if abs(beat['timestamp'] - beat_ts) < 0.01:  # Match timestamp
                                    beat_color = beat.get('color', "#FFFF00")
                                    beat_label = beat.get('label', 'N')
                                    break
                        
                        # Box removed — label + time text is sufficient visual indicator
                        # Label text is already drawn by the main R-peak loop
            
            # Otherwise, draw single beat selection
            elif self._clicked_beat_timestamp is not None:
                if self._start_sec <= self._clicked_beat_timestamp <= end_sec:
                    pct = (self._clicked_beat_timestamp - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    bx = int(pct * w)
                    
                    # Find the beat's color from annotations
                    beat_color = "#FFFF00"  # Default yellow
                    beat_label = "N"
                    if hasattr(self, '_beat_annotations'):
                        for beat in self._beat_annotations:
                            if abs(beat['timestamp'] - self._clicked_beat_timestamp) < 0.01:
                                beat_color = beat.get('color', "#FFFF00")
                                beat_label = beat.get('label', 'N')
                                break
                    
                    # Box removed — label + time text is sufficient visual indicator
        
        # --- Draw N-N (R-R) Interval Labels ONLY on Lead I ---
        if lead_name == 'I' and hasattr(self, '_rr_intervals') and self._rr_intervals and self._show_annotations:  # Only show if annotations enabled
            font = painter.font()
            font.setPixelSize(9)
            font.setBold(False)
            painter.setFont(font)
            
            end_sec = self._start_sec + len(d) / self._fs
            
            for interval in self._rr_intervals:
                start_ts = interval['start_ts']
                end_ts = interval['end_ts']
                rr_ms = interval['rr_ms']
                
                # Check if interval is within visible window
                if (self._start_sec <= start_ts <= end_sec) or (self._start_sec <= end_ts <= end_sec):
                    # Calculate x coordinates for start and end
                    start_pct = (start_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    end_pct = (end_ts - self._start_sec) / (end_sec - self._start_sec) if (end_sec - self._start_sec) > 0 else 0.0
                    
                    start_x = int(start_pct * w)
                    end_x = int(end_pct * w)
                    mid_x = (start_x + end_x) // 2
                    
                    # Draw interval value between the two R-peaks
                    # Position it at mid-height of the waveform, slight offset from top
                    interval_text = f"{int(rr_ms)}"
                    
                    # Color: yellow for long intervals (>1200ms), orange for short (<600ms), white otherwise
                    if rr_ms > 1200:
                        color = "#FFFF00"  # Yellow for pauses
                    elif rr_ms < 600:
                        color = "#FFA500"  # Orange for fast beats
                    else:
                        color = "#CCCCCC"  # Light gray for normal intervals
                    
                    painter.setPen(QPen(QColor(color)))
                    painter.drawText(mid_x - 10, 22, interval_text)
                    
        if self._mode == TOOL_RULER:
            # Draw persistent ruler (start point stays after measurement)
            if self._ruler_start is not None:
                rpen = QPen(QColor("#00FFFF"), 2)
                painter.setPen(rpen)
                # Draw line from start point to end point (or hover if end not set yet)
                end_point = self._ruler_end if self._ruler_end is not None else (self._hover_pos if self._hover_pos else self._ruler_start)
                
                if self._ruler_end is not None:
                    # Final measurement: solid line
                    painter.drawLine(self._ruler_start, self._ruler_end)
                    dx = abs(self._ruler_end.x() - self._ruler_start.x())
                    dy = abs(self._ruler_end.y() - self._ruler_start.y())
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    bpm = 60000 / ms if ms > 0 else 0
                    dy_mv = amplitude_mv_from_pixels(dy, max(1, h), rng, ADC_TO_MV)
                    painter.setPen(QPen(QColor("#00FFFF")))
                    painter.drawText(self._ruler_end.x(), max(12, self._ruler_end.y() - 6), ruler_label(ms, dy_mv, bpm))
                elif self._hover_pos is not None:
                    # Preview line while hovering (dashed)
                    pen_preview = QPen(QColor("#00FFFF"), 2, Qt.DashLine)
                    painter.setPen(pen_preview)
                    painter.drawLine(self._ruler_start, self._hover_pos)
                    dx = abs(self._hover_pos.x() - self._ruler_start.x())
                    dy = abs(self._hover_pos.y() - self._ruler_start.y())
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    bpm = 60000 / ms if ms > 0 else 0
                    dy_mv = amplitude_mv_from_pixels(dy, max(1, h), rng, ADC_TO_MV)
                    painter.setPen(QPen(QColor("#00FFFF")))
                    painter.drawText(self._hover_pos.x(), max(12, self._hover_pos.y() - 6), ruler_label(ms, dy_mv, bpm))
        elif self._mode == TOOL_CALIPER:

            if self._caliper_line1 is not None:
                ppen = QPen(QColor("#FFFF00"), 1)
                painter.setPen(ppen)
                
                painter.drawLine(self._caliper_line1, 0, self._caliper_line1, h)
                
                
                if self._caliper_line2 is not None:
                    painter.drawLine(self._caliper_line2, 0, self._caliper_line2, h)
                    dx = abs(self._caliper_line2 - self._caliper_line1)
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    painter.drawText(min(self._caliper_line1, self._caliper_line2) + dx//2, 12, caliper_label(ms))
                elif self._hover_pos is not None:
                    
                    pen_preview = QPen(QColor("#FFFF00"), 1, Qt.DashLine)
                    painter.setPen(pen_preview)
                    hover_x = self._hover_pos.x()
                    painter.drawLine(hover_x, 0, hover_x, h)
                    dx = abs(hover_x - self._caliper_line1)
                    ms = interval_ms_from_pixels(dx, max(1, w), len(d), self._fs)
                    painter.drawText(min(self._caliper_line1, hover_x) + dx//2, 12, caliper_label(ms))
        elif self._mode == TOOL_MAGNIFY:
            host = self._find_magnifier_host()
            if host is not None and hasattr(host, "_magnifier_overlay"):
                return
            focus_pos = self._magnify_pos if self._magnify_locked else self._hover_pos
            if focus_pos is None:
                return
            hover_x = max(0, min(focus_pos.x(), w - 1))
            hover_y = max(0, min(focus_pos.y(), h - 1))
            src_center = self._x_to_index(hover_x, w, len(d))
            span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / max(2, getattr(self.parent(), "_curr_length_idx", 1) + 2)))
            half = max(8, int(span / max(2, self._gain * 1.5)))
            i0 = max(0, src_center - half)
            i1 = min(len(d), src_center + half)
            sub = d[i0:i1]

            panel_w = min(320, max(220, int(w * 0.34)))
            panel_h = min(180, max(120, int(h * 0.72)))
            panel_x = min(w - panel_w - 10, hover_x + 24)
            panel_y = max(8, hover_y - panel_h - 18)
            if panel_x < 8:
                panel_x = 8
            if panel_y < 8:
                panel_y = min(h - panel_h - 8, hover_y + 18)

            panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
            inner = panel_rect.adjusted(10, 10, -10, -10)

            painter.setBrush(QColor(8, 8, 8, 235))
            painter.setPen(QPen(QColor(COL_YELLOW), 3))
            painter.drawRoundedRect(panel_rect, 12, 12)

            painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
            for frac in (0.25, 0.5, 0.75):
                gx = int(inner.left() + inner.width() * frac)
                gy = int(inner.top() + inner.height() * frac)
                painter.drawLine(gx, inner.top(), gx, inner.bottom())
                painter.drawLine(inner.left(), gy, inner.right(), gy)

            if len(sub) > 1:
                sub_min = float(np.min(sub))
                sub_max = float(np.max(sub))
                
                # Symmetrically frame the magnifier around the local median baseline
                # This prevents asymmetrical waves from pushing the baseline to the bottom/top edge
                base_val = float(np.median(sub))
                max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
                pad = max(20.0, max_dev * 0.35)
                view_min = base_val - max_dev - pad
                view_max = base_val + max_dev + pad
                view_rng = max(1.0, view_max - view_min)

                path_pen = QPen(QColor(self._color))
                path_pen.setWidthF(2.0)
                painter.setPen(path_pen)
                x_scale_sub = inner.width() / max(1, len(sub) - 1)
                prev = None
                for i in range(len(sub)):
                    xx = inner.left() + i * x_scale_sub
                    yy = inner.bottom() - ((sub[i] - view_min) / view_rng) * inner.height()
                    if prev is not None:
                        painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
                    prev = (xx, yy)

                focus_x = int(inner.left() + ((src_center - i0) / max(1, len(sub) - 1)) * inner.width())
                focus_y = int(inner.bottom() - ((d[src_center] - view_min) / view_rng) * inner.height())
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawLine(focus_x, inner.top(), focus_x, inner.bottom())
                painter.drawLine(inner.left(), focus_y, inner.right(), focus_y)

            painter.setPen(QPen(QColor(COL_WHITE)))
            painter.drawText(
                panel_rect.left() + 10,
                panel_rect.bottom() - 10,
                f"{getattr(self.parent(), '_curr_gain_idx', 1) + 2}x {'locked' if self._magnify_locked else 'hover'}"
            )


class MagnifierOverlay(QWidget):
    """Shared magnifier popup for the replay panel so zoom is never clipped by strip bounds."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self._visible = False
        self._panel_rect = QRect()
        self._inner_rect = QRect()
        self._data = None
        self._focus_idx = 0
        self._focus_pos = QPoint(0, 0)
        self._gain = 1.0
        self._speed = 25.0
        self._fs = 500.0
        self._lead_name = ""
        self._source_widget = None
        self.hide()

    def set_focus(self, source_widget, payload: dict, focus_pos: QPoint):
        if payload is None or payload.get("data") is None:
            self.hide()
            return
        self._source_widget = source_widget
        self._data = np.asarray(payload.get("data"), dtype=float)
        self._gain = float(payload.get("gain", 1.0))
        self._speed = float(payload.get("speed", 25.0))
        self._fs = float(payload.get("fs", 500.0))
        self._lead_name = str(payload.get("lead_name", ""))
        self._focus_pos = QPoint(focus_pos)
        if self.parentWidget() is None:
            self.hide()
            return
        host = self.parentWidget()
        self.setGeometry(host.rect())
        self.raise_()
        self._visible = True
        self.show()
        self.update()

    def clear_focus(self, source_widget=None):
        if source_widget is not None and self._source_widget is not None and source_widget is not self._source_widget:
            return
        self._visible = False
        self._data = None
        self._source_widget = None
        self.hide()
        self.update()

    def paintEvent(self, event):
        if not self._visible or self._data is None or self._source_widget is None:
            return
        d = np.asarray(self._data, dtype=float)
        if d.size < 2:
            return

        host = self.parentWidget()
        if host is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        # Convert the focus point from source-widget coordinates into overlay coordinates.
        try:
            focus_pt = self._source_widget.mapTo(host, self._focus_pos)
        except Exception:
            focus_pt = self._focus_pos
        hover_x = max(0, min(int(focus_pt.x()), max(0, host.width() - 1)))
        hover_y = max(0, min(int(focus_pt.y()), max(0, host.height() - 1)))

        w, h = host.width(), host.height()
        source_w = max(1, int(self._source_widget.width()))
        local_x = max(0, min(int(self._focus_pos.x()), source_w - 1))
        src_center = int(round((local_x / float(max(1, source_w - 1))) * (len(d) - 1)))
        span = max(12, int(len(d) / max(2.0, self._speed / 12.5) / 2.0))
        half = max(8, int(span / max(2, self._gain * 1.5)))
        i0 = max(0, src_center - half)
        i1 = min(len(d), src_center + half)
        sub = d[i0:i1]
        if len(sub) < 2:
            return

        panel_w = min(360, max(240, int(w * 0.34)))
        panel_h = min(220, max(140, int(h * 0.28)))
        panel_x = hover_x + 24
        panel_y = hover_y - panel_h - 18
        if panel_x + panel_w > w - 8:
            panel_x = hover_x - panel_w - 24
        if panel_x < 8:
            panel_x = 8
        if panel_y < 8:
            panel_y = hover_y + 18
        if panel_y + panel_h > h - 8:
            panel_y = max(8, h - panel_h - 8)
        self._panel_rect = QRect(panel_x, panel_y, panel_w, panel_h)
        self._inner_rect = self._panel_rect.adjusted(12, 12, -12, -12)

        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(0, 0, 0, 220))
        painter.drawRoundedRect(self._panel_rect, 12, 12)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QPen(QColor(COL_YELLOW), 3))
        painter.drawRoundedRect(self._panel_rect, 12, 12)

        painter.setPen(QPen(QColor(COL_GRID_MINOR), 1))
        for frac in (0.25, 0.5, 0.75):
            gx = int(self._inner_rect.left() + self._inner_rect.width() * frac)
            gy = int(self._inner_rect.top() + self._inner_rect.height() * frac)
            painter.drawLine(gx, self._inner_rect.top(), gx, self._inner_rect.bottom())
            painter.drawLine(self._inner_rect.left(), gy, self._inner_rect.right(), gy)

        sub_min = float(np.min(sub))
        sub_max = float(np.max(sub))
        base_val = float(np.median(sub))
        max_dev = max(abs(sub_max - base_val), abs(sub_min - base_val))
        pad = max(20.0, max_dev * 0.35)
        view_min = base_val - max_dev - pad
        view_max = base_val + max_dev + pad
        view_rng = max(1.0, view_max - view_min)

        path_pen = QPen(QColor("#22FF44"))
        path_pen.setWidthF(2.0)
        painter.setPen(path_pen)
        x_scale_sub = self._inner_rect.width() / max(1, len(sub) - 1)
        prev = None
        for i in range(len(sub)):
            xx = self._inner_rect.left() + i * x_scale_sub
            yy = self._inner_rect.bottom() - ((sub[i] - view_min) / view_rng) * self._inner_rect.height()
            if prev is not None:
                painter.drawLine(QPointF(prev[0], prev[1]), QPointF(xx, yy))
            prev = (xx, yy)

        focus_x = int(self._inner_rect.left() + ((src_center - i0) / max(1, len(sub) - 1)) * self._inner_rect.width())
        focus_y = int(self._inner_rect.bottom() - ((d[src_center] - view_min) / view_rng) * self._inner_rect.height())
        painter.setPen(QPen(QColor("#ffffff"), 1))
        painter.drawLine(focus_x, self._inner_rect.top(), focus_x, self._inner_rect.bottom())
        painter.drawLine(self._inner_rect.left(), focus_y, self._inner_rect.right(), focus_y)

        painter.setPen(QPen(QColor(COL_WHITE)))
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.top() + 18,
            f"{self._lead_name or 'Lead'}  {self._gain:.1f}x"
        )
        painter.drawText(
            self._panel_rect.left() + 10,
            self._panel_rect.bottom() - 10,
            "click another wave to move"
        )


# -----------------------------------------------------------------------------
# 7. HOLTER EVENTS PANEL
# -----------------------------------------------------------------------------

class HolterRRTrendCanvas(QWidget):
    """Compact RR trend strip matching Holter Expert reference: dark bg, grid, Y-axis 0-2000ms, time labels."""

    def __init__(self, parent=None, title="RR Interval"):
        super().__init__(parent)
        self._title = title
        self._points = []        # [(time_sec, rr_ms), ...]
        self._start_epoch = None # unix epoch for beat 0, used for HH:MM labels
        self._show_hr = False    # False=RR ms, True=HR bpm
        self._selection_range = None  # (t_start, t_end) cyan box for zoom indicator
        self.setMinimumHeight(120)
        self.setStyleSheet("background:#0a1520;border:1px solid #1e3350;border-radius:4px;")

    # ------------------------------------------------------------------ API --
    def set_points(self, points, start_epoch=None):
        self._points = list(points or [])
        if start_epoch is not None:
            self._start_epoch = start_epoch
        self.update()

    def set_selection(self, t_start, t_end):
        """Mark a time window with a cyan box (used on the Full chart to show Recent window)."""
        self._selection_range = (t_start, t_end)
        self.update()

    def set_mode(self, mode: str):
        if mode == "HR":
            self._show_hr = True
            self._title = "Heart Rate Trend (full recording)"
        else:
            self._show_hr = False
            self._title = "RR Interval Trend (Full)"
        self.update()

    # ---------------------------------------------------------- paintEvent --
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        w, h = self.width(), self.height()

        # -- Background ------------------------------------------------------
        painter.fillRect(self.rect(), QColor("#0a1520"))

        # Margins: left for Y-axis, bottom for X-axis
        LM, RM, TM, BM = 52, 10, 22, 24
        iw = max(1, w - LM - RM)
        ih = max(1, h - TM - BM)
        inner = QRect(LM, TM, iw, ih)

        # -- Title (top-left) -------------------------------------------------
        painter.setPen(QPen(QColor("#7ecfff"), 1))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        painter.drawText(LM, 16, self._title)

        # -- Y-axis config ----------------------------------------------------
        if self._show_hr:
            y_min, y_max = 20, 220
            y_ticks = [40, 80, 120, 160, 200]
            y_unit = "bpm"
        else:
            y_min, y_max = 0, 2000
            y_ticks = [0, 500, 1000, 1500, 2000]
            y_unit = "ms"
        y_rng = float(y_max - y_min)

        def to_y(val):
            v = max(y_min, min(y_max, val))
            return inner.bottom() - int(((v - y_min) / y_rng) * ih)

        # -- Horizontal grid lines + Y labels --------------------------------
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        for yv in y_ticks:
            yp = to_y(yv)
            painter.setPen(QPen(QColor("#1e3d5a"), 1, Qt.DashLine))
            painter.drawLine(inner.left(), yp, inner.right(), yp)
            painter.setPen(QPen(QColor("#7ecfff"), 1))
            lbl = str(yv)
            painter.drawText(2, yp + 5, lbl)

        # -- No data message -------------------------------------------------
        if not self._points:
            painter.setPen(QPen(QColor("#3a5a70"), 1))
            painter.setFont(QFont("Arial", 11))
            painter.drawText(inner, Qt.AlignCenter, "No RR data")
            painter.setPen(QPen(QColor("#1e3350"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(inner)
            return

        # -- Time range ------------------------------------------------------
        # Fixed 12-hour X-axis range
        t_min = 0.0
        t_max = 12.0 * 3600.0  # 24 hours in seconds
        t_rng = float(t_max - t_min)

        def to_x(t):
            return inner.left() + int(((t - t_min) / t_rng) * iw)

        #  "  "  Vertical grid every ~2 hours  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 
        # Vertical grid removed per user request
        # grid_interval = 7200.0  # 2 hours in seconds
        # t_grid = (int(t_min / grid_interval) + 1) * grid_interval
        # while t_grid < t_max:
        #     xp = to_x(t_grid)
        #     painter.setPen(QPen(QColor("#1e3d5a"), 1, Qt.DashLine))
        #     painter.drawLine(xp, inner.top(), xp, inner.bottom())
        #     t_grid += grid_interval

        #  "  "  Scatter dots  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(QColor("#00ff66")))

        # Sub-sample for performance (max ~4000 dots)
        step = max(1, len(self._points) // 4000)
        for i in range(0, len(self._points), step):
            t, rr = self._points[i]
            if self._show_hr:
                val = (60000.0 / rr) if rr > 0 else 0
            else:
                val = float(rr)
            if val <= 0:
                continue
            px = to_x(t)
            py = to_y(val)
            painter.drawEllipse(px - 2, py - 2, 4, 4)

        #  "  "  Cyan selection box  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 
        if self._selection_range is not None:
            rs, re_t = self._selection_range
            sx = max(inner.left(), min(inner.right(), to_x(rs)))
            ex = max(inner.left(), min(inner.right(), to_x(re_t)))
            painter.setPen(QPen(QColor("#00cccc"), 1))
            painter.setBrush(QBrush(QColor(0, 204, 204, 30)))
            painter.drawRect(sx, inner.top(), max(2, ex - sx), ih)

        #  "  "  X-axis time labels  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 
        painter.setPen(QPen(QColor("#7ecfff"), 1))
        painter.setFont(QFont("Arial", 9, QFont.Bold))
        import datetime as _dt
        n_xticks = max(2, min(8, iw // 75))
        for i in range(n_xticks + 1):
            frac = i / float(n_xticks)
            t_at = t_min + frac * t_rng
            xp = inner.left() + int(frac * iw)
            painter.setPen(QPen(QColor("#2e5a80"), 1))
            painter.drawLine(xp, inner.bottom(), xp, inner.bottom() + 3)
            painter.setPen(QPen(QColor("#7ecfff"), 1))
            if self._start_epoch is not None:
                ts = _dt.datetime.fromtimestamp(self._start_epoch + t_at)
                label = ts.strftime("%H:%M")
            else:
                hrs = int(t_at // 3600)
                mins = int((t_at % 3600) // 60)
                label = f"{hrs:02d}:{mins:02d}"
            painter.drawText(xp - 14, inner.bottom() + BM - 2, label)

        #  "  "  Border  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  "  " 
        painter.setPen(QPen(QColor("#1e3350"), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(inner)

class HolterExpertReviewPanel(QWidget):
    """HolterExpert-inspired review layout: trend + Lorenz + strips + overview."""
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._metrics = []
        self._summary = {}
        self._template_rows = []
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self._rr_trend_full = HolterRRTrendCanvas(title="RR Interval Trend (Full Recording)")
        layout.addWidget(self._rr_trend_full)

        body = QSplitter(Qt.Horizontal)
        body.setChildrenCollapsible(False)
        body.setHandleWidth(1)
        body.setStyleSheet(f"QSplitter{{background:{UI_BG};}} QSplitter::handle{{background:{UI_BORDER};}}")

        left = QFrame()
        left.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        left_l = QVBoxLayout(left)
        left_l.setContentsMargins(8, 8, 8, 8)
        left_l.setSpacing(8)
        self._lorenz = LorenzCanvas()
        self._lorenz.setMinimumHeight(280)
        left_l.addWidget(self._lorenz, 2)
        self._template_table = QTableWidget(0, 3)
        self._template_table.setHorizontalHeaderLabels(["Template", "Class", "Beats"])
        self._template_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._template_table.verticalHeader().setVisible(False)
        self._template_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._template_table.setStyleSheet(_table_style())
        self._template_table.cellClicked.connect(self._on_template_clicked)
        left_l.addWidget(self._template_table, 1)
        body.addWidget(left)

        #  "  "  Center: scrollable 12-lead ECG grid  "  " 
        center = QFrame()
        center.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:8px;}}")
        center_l = QVBoxLayout(center)
        center_l.setContentsMargins(4, 4, 4, 4)
        center_l.setSpacing(0)

        # 12-lead header
        hdr = QLabel("12-Lead ECG Overview")
        hdr.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:700;padding:4px 6px;border:none;")
        center_l.addWidget(hdr)

        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};")
        leads_vbox = QVBoxLayout(leads_container)
        leads_vbox.setContentsMargins(2, 2, 2, 2)
        leads_vbox.setSpacing(2)

        self._lead_names_12 = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
        self._expert_lead_strips = {}  # lead_name -> ECGStripCanvas
        for lead in self._lead_names_12:
            row_h = QHBoxLayout()
            row_h.setContentsMargins(0, 0, 0, 0)
            row_h.setSpacing(4)
            lbl = QLabel(lead)
            lbl.setFixedWidth(34)
            lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            # Expert/Overview mode: disable ALL coloring (disable_all_coloring=True) - no arrhythmia colors, just green
            strip = ECGStripCanvas(height=60, color="#00FF00", pen_width=0.9, lead_name=lead, show_vertical_lines=False, show_annotations=False, disable_all_coloring=True)
            strip.set_gain(1.0)
            self._expert_lead_strips[lead] = strip
            row_h.addWidget(lbl)
            row_h.addWidget(strip, 1)
            leads_vbox.addLayout(row_h)

        leads_scroll.setWidget(leads_container)
        center_l.addWidget(leads_scroll, 1)

        # Rhythm strip at bottom (Lead II) - also disable vertical lines and ALL coloring
        rhythm_row = QHBoxLayout()
        rhythm_row.setSpacing(4)
        rlbl = QLabel("II")
        rlbl.setFixedWidth(34)
        rlbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        rlbl.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
        self._mini = ECGStripCanvas(height=40, color="#00AA00", pen_width=0.9, show_vertical_lines=False, show_annotations=False, disable_all_coloring=True)
        rhythm_row.addWidget(rlbl)
        rhythm_row.addWidget(self._mini, 1)
        center_l.addLayout(rhythm_row)

        body.addWidget(center)

        right = QFrame()
        right.setStyleSheet(f"QFrame{{background:{UI_PANEL};border:1px solid {UI_BORDER};border-radius:8px;}}")
        right_l = QVBoxLayout(right)
        right_l.setContentsMargins(8, 8, 8, 8)
        right_l.setSpacing(6)
        ttl = QLabel("Overview")
        ttl.setStyleSheet(f"color:{UI_TEXT};font-weight:700;font-size:13px;padding:6px;background:{UI_PANEL_ALT};border:1px solid {UI_BORDER};border-radius:6px;")
        right_l.addWidget(ttl)
        self._overview = QTableWidget(0, 2)
        self._overview.setHorizontalHeaderLabels(["Name", "Value"])
        self._overview.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self._overview.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._overview.verticalHeader().setVisible(False)
        self._overview.setEditTriggers(QTableWidget.NoEditTriggers)
        self._overview.setStyleSheet(_table_style())
        right_l.addWidget(self._overview, 1)
        body.addWidget(right)
        body.setSizes([360, 760, 300])
        layout.addWidget(body, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        self._metrics = list(metrics_list or [])
        self._summary = dict(summary or {})
        rr_points = []
        for m in self._metrics:
            t0 = float(m.get("t", 0.0) or 0.0)
            rr_list = list(m.get("rr_intervals_list", []) or [])
            dur = float(m.get("duration", 0.0) or 0.0)
            n = max(1, len(rr_list))
            step = (dur / n) if dur > 0 else 0.2
            for i, rr in enumerate(rr_list):
                rr_points.append((t0 + i * step, float(rr)))



        start_epoch = summary.get("start_time_epoch", None)
        self._rr_trend_full.set_points(rr_points, start_epoch)
        
        # Update Lorenz plot with all RR intervals from the full recording
        x = [p[1] for p in rr_points[:-1]] if len(rr_points) > 1 else []
        y = [p[1] for p in rr_points[1:]] if len(rr_points) > 1 else []
        self._lorenz.set_data(x, y)

        tm = {}
        for m in self._metrics:
            for row in (m.get("template_summary", []) or []):
                key = row.get("template_key") or row.get("template_id") or row.get("label") or "T"
                r = tm.setdefault(key, {"id": row.get("template_id", "T"), "label": row.get("label", "N"), "count": 0, "first": float(row.get("first_timestamp", m.get("t", 0.0)) or 0.0)})
                r["count"] += int(row.get("count", 0) or 0)
                r["first"] = min(r["first"], float(row.get("first_timestamp", r["first"]) or r["first"]))
        self._template_rows = sorted(tm.values(), key=lambda it: it["count"], reverse=True)
        self._template_table.setRowCount(len(self._template_rows))
        for i, row in enumerate(self._template_rows):
            vals = [str(row["id"]), str(row["label"]), str(row["count"])]
            for j, v in enumerate(vals):
                item = QTableWidgetItem(v)
                item.setForeground(QColor(UI_TEXT))
                self._template_table.setItem(i, j, item)

        # Compute tachy/brady durations from summary
        _dur_sec = float(summary.get('duration_sec', 0) or 0)
        _total_beats = int(summary.get('total_beats', 0) or 1)
        # Use pre-computed sec values if available (set by _build_summary_from_metrics)
        if 'tachy_sec' in summary:
            _tachy_sec = float(summary['tachy_sec'] or 0)
            _brady_sec = float(summary['brady_sec'] or 0)
            _tachy_pct = float(summary.get('tachy_pct', 0) or 0)
            _brady_pct = float(summary.get('brady_pct', 0) or 0)
        else:
            _tachy_beats = int(summary.get('tachy_beats', 0) or 0)
            _brady_beats = int(summary.get('brady_beats', 0) or 0)
            _avg_rr_sec = (_dur_sec / _total_beats) if _total_beats > 0 and _dur_sec > 0 else 0.0
            _tachy_sec = _tachy_beats * _avg_rr_sec
            _brady_sec = _brady_beats * _avg_rr_sec
            _tachy_pct = (_tachy_beats / _total_beats * 100) if _total_beats > 0 else 0.0
            _brady_pct = (_brady_beats / _total_beats * 100) if _total_beats > 0 else 0.0
        def _fmt_dur(sec):
            s = int(sec); h = s // 3600; m = (s % 3600) // 60; ss = s % 60
            return f"{h:02d}:{m:02d}:{ss:02d}"
        _tachy_str = f"{_fmt_dur(_tachy_sec)} {_tachy_pct:.2f}%"
        _brady_str = f"{_fmt_dur(_brady_sec)} {_brady_pct:.2f}%"

        # Sinus HR from summary (falls back to max_hr/min_hr)
        _sinus_max = summary.get('sinus_max_hr', summary.get('max_hr', 0))
        _sinus_min = summary.get('sinus_min_hr', summary.get('min_hr', 0))
        _sinus_max_t = summary.get('sinus_max_hr_time', '')
        _sinus_min_t = summary.get('sinus_min_hr_time', '')
        _max_hr_t = summary.get('max_hr_time', '')
        _min_hr_t = summary.get('min_hr_time', '')

        # AF duration from arrhythmia_counts
        _af_count = int((summary.get('arrhythmia_counts') or {}).get('AF', 0))
        _af_chunk_dur = 4.0  # each metric chunk ~4s
        _af_sec = _af_count * _af_chunk_dur
        _af_pct = (_af_sec / _dur_sec * 100) if _dur_sec > 0 else 0.0
        _af_str = f"{_fmt_dur(_af_sec)} {_af_pct:.2f}%" if _af_count > 0 else "-"

        def _hr_with_time(hr, t):
            hr_str = f"{int(round(float(hr or 0)))}bpm"
            return f"{hr_str} {t}" if t else hr_str

        rows = [
            ("Total", f"{summary.get('total_beats', 0)}"),
            ("X Total", f"{summary.get('pauses', 0)}"),
            ("AVG HR", f"{summary.get('avg_hr', 0):.0f} bpm"),
            ("Max HR", _hr_with_time(summary.get('max_hr', 0), _max_hr_t)),
            ("Min HR", _hr_with_time(summary.get('min_hr', 0), _min_hr_t)),
            ("Sinus Max HR", _hr_with_time(_sinus_max, _sinus_max_t)),
            ("Sinus Min HR", _hr_with_time(_sinus_min, _sinus_min_t)),
            ("During of Tachy.", _tachy_str),
            ("During of Brady.", _brady_str),
            ("V Total", f"{summary.get('ve_beats', 0)}"),
            ("S Total", f"{summary.get('sve_beats', 0)}"),
            ("AVG HR of Af/AF", "-"),
            ("Duration of Af/AF", _af_str),
            ("Paced Beats", "0"),
            ("Pacing AVG HR", "-"),
            ("Pacing Max HR", "-"),
            ("Pacing Min HR", "-"),
            ("Longest RR", f"{summary.get('longest_rr_ms', 0)/1000:.2f}s"),
            ("RRI (\u22652.0s)", f"{summary.get('pauses', 0)}"),
            ("ST Elevation", "-"),
            ("ST Depression", "-"),
        ]
        self._overview.setRowCount(len(rows))
        for i, (k, v) in enumerate(rows):
            ki = QTableWidgetItem(k)
            vi = QTableWidgetItem(v)
            ki.setForeground(QColor(UI_MUTED))
            vi.setForeground(QColor(UI_TEXT))
            self._overview.setItem(i, 0, ki)
            self._overview.setItem(i, 1, vi)

    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        if data is None or not isinstance(data, np.ndarray) or data.ndim != 2 or data.shape[1] == 0:
            return
        n = data.shape[1]
        x = np.arange(n, dtype=float) / 500.0
        # Feed all 12 leads to the new lead-strip dict
        for i, lead in enumerate(self._lead_names_12):
            if i < data.shape[0]:
                strip = self._expert_lead_strips.get(lead)
                if strip:
                    strip.set_data(x, data[i].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
        # Rhythm strip: Lead II (index 1)
        if data.shape[0] > 1:
            self._mini.set_data(x, data[1].copy(), beat_annotations=beat_annotations, start_sec=start_sec)
        elif data.shape[0] > 0:
            self._mini.set_data(x, data[0].copy(), beat_annotations=beat_annotations, start_sec=start_sec)

    def _on_template_clicked(self, row, _col):
        if 0 <= row < len(self._template_rows):
            self.seek_requested.emit(float(self._template_rows[row].get("first", 0.0)))


class HolterEventsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._session_dir = ""
        self._selected_payload = {}
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

        # Left: event list + stats
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        ev_title = QLabel("Events")
        ev_title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:6px 10px;border-radius:6px;")
        left_layout.addWidget(ev_title)

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        # Stats below table
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(6)
        self._stat_labels = {}
        for i, (key, label) in enumerate([
            ("hr_max","HR Max"),("hr_min","HR Min"),("hr_smax","Sinus Max HR"),
            ("hr_smin","Sinus Min HR"),("brady","Bradycardia"),("user_ev","User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet(f"color:{COL_GREEN_DRK};font-size:10px;font-weight:bold;border:none;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: navigation
        nav = QWidget()
        nav_layout = QVBoxLayout(nav)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        for label in ["Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(label)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(38)
            if label == "Prev Event":
                btn.clicked.connect(self._go_prev_event)
            elif label == "Next Event":
                btn.clicked.connect(self._go_next_event)
            elif label == "Remove All":
                btn.clicked.connect(self._remove_all_events)
            elif label == "Remove":
                btn.clicked.connect(self._remove_selected_event)
            nav_layout.addWidget(btn)
        nav_layout.addStretch()
        layout.addWidget(nav)

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""

    def load_events(self, events: list, summary: dict):
        # Filter out hidden event labels from display
        hidden_labels = ["Long QT Syndrome", "Wide QRS (non-specific)", "Frequent PVCs", "Multifocal PVCs"]
        filtered_events = []
        for ev in events:
            label = str(ev.get('label', '')).lower()
            if not any(hl.lower() in label for hl in hidden_labels):
                filtered_events.append(ev)
        
        self._events = filtered_events
        self._ev_table.setRowCount(len(filtered_events))
        for i, ev in enumerate(filtered_events):
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            source = ev.get("source", "analysis")
            conf = ev.get("confidence", 0.0)
            for j, val in enumerate([ev['label'], t_str, "12", "7s", source, f"{float(conf or 0.0):.2f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr_max",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_min",f"{s.get('min_hr',0):.0f} bpm"),
                          ("hr_smax",f"{s.get('max_hr',0):.0f} bpm"),
                          ("hr_smin",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_event_clicked(self, row, col):
        self._select_and_seek(row)

    def _selected_row(self) -> int:
        selection = self._ev_table.selectionModel()
        if selection:
            rows = selection.selectedRows()
            if rows:
                return int(rows[0].row())
        return -1

    def _select_and_seek(self, row: int):
        if row < 0 or row >= len(self._events):
            return
        self._ev_table.selectRow(row)
        self._selected_payload = dict(self._events[row] or {})
        self.seek_requested.emit(float(self._events[row].get('timestamp', 0.0) or 0.0))

    def _go_prev_event(self):
        if not self._events:
            return
        row = self._selected_row()
        if row < 0:
            row = 0
        else:
            row = max(0, row - 1)
        self._select_and_seek(row)

    def _go_next_event(self):
        if not self._events:
            return
        row = self._selected_row()
        if row < 0:
            row = 0
        else:
            row = min(len(self._events) - 1, row + 1)
        self._select_and_seek(row)

    def _remove_selected_event(self):
        row = self._selected_row()
        if row < 0 or row >= len(self._events):
            return
        self._events.pop(row)
        self._ev_table.removeRow(row)
        self._selected_payload = {}
        if self._events:
            next_row = min(row, len(self._events) - 1)
            self._select_and_seek(next_row)

    def _remove_all_events(self):
        self._events = []
        self._selected_payload = {}
        self._ev_table.setRowCount(0)
        
# -----------------------------------------------------------------------------
# 8. HOLTER TEMPLATE PANEL  (template gallery)
# -----------------------------------------------------------------------------

class _TemplateMetricCard(QFrame):
    def __init__(self, title: str, value: str = "", accent: str = COL_GREEN, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 #1B2433, stop:1 #111827);border:1px solid {UI_BORDER};border-radius:10px;}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(2)
        lbl = QLabel(title)
        lbl.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:700;border:none;")
        self.value = QLabel(value)
        self.value.setStyleSheet(f"color:{UI_TEXT};font-size:16px;font-weight:800;border:none;")
        layout.addWidget(lbl)
        layout.addWidget(self.value)


class TemplateCardWidget(QFrame):
    clicked = pyqtSignal(object, object)  # (card, event)
    double_clicked = pyqtSignal(object)   # (card)
    template_id_changed = pyqtSignal(object, str)
    class_changed = pyqtSignal(object, str)
    viewed_changed = pyqtSignal(object, bool)

    def __init__(self, parent=None, accent: str = UI_BORDER):
        super().__init__(parent)
        self._accent = accent or UI_BORDER
        self._selected = False
        self._hovered = False
        self._template_key = ""
        self.setObjectName("templateCard")
        self.setCursor(Qt.PointingHandCursor)
        self.setContextMenuPolicy(Qt.DefaultContextMenu)
        self._build_ui()
        self.set_accent(self._accent)

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
        self.setMinimumHeight(170)
        self.setFixedWidth(320)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self._card_style = """
            QFrame#templateCard {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {bg_top}, stop:1 {bg_bottom});
                border: 1px solid {border};
                border-radius: 14px;
            }}
            QLineEdit, QComboBox {{
                background: #0C1320;
                color: {ui_text};
                border: 1px solid {border};
                border-radius: 7px;
                padding: 4px 7px;
                font-size: 11px;
            }}
            QComboBox QAbstractItemView {{
                background: #0B1220;
                color: {ui_text};
                selection-background-color: {ui_accent};
                selection-color: #07111F;
                border: 1px solid {border};
                outline: 0;
            }}
            QComboBox QAbstractItemView::item {{
                padding: 4px 8px;
                min-height: 18px;
            }}
            QComboBox::drop-down {{ border: none; width: 16px; }}
            QLineEdit:focus, QComboBox:focus {{ border-color: {ui_accent}; }}
        """

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.id_edit = QLineEdit("T1")
        self.id_edit.setFixedWidth(70)
        self.id_edit.setPlaceholderText("T ID")
        self.class_combo = QComboBox()
        # Updated options: N, S, V, P, AF, X, Other, Unconfirmed
        self.class_combo.addItems(["N", "S", "V", "P", "AF", "X", "Other", "Unconfirmed"])
        self.class_combo.setFixedWidth(95)  # Increased width for "Unconfirmed"
        self.count_pill = QLabel("0 beats")
        self.count_pill.setAlignment(Qt.AlignCenter)
        self.count_pill.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:800;background:#152235;border:1px solid {self._accent};border-radius:10px;padding:4px 8px;")
        header.addWidget(self.id_edit)
        header.addWidget(self.class_combo)
        header.addStretch()
        header.addWidget(self.count_pill)
        layout.addLayout(header)

        self.badge_container = QWidget()
        badge_row = QHBoxLayout(self.badge_container)
        badge_row.setContentsMargins(0, 0, 0, 0)
        badge_row.setSpacing(5)
        badge_row.addStretch()
        self.badge_labels = {}
        badges_info = [
            ("ambiguous", "?", "It means that at least one classification channel of the beat under the template is not similar, so the template ECG waveform is empty, which is represented by '?'"),
            ("inserted", "+", "Templates classified by beats of inserting beats, creating new templates, or merging templates are marked with '+'"),
            ("demix", "DEMIX", "In the figure, DEMIX represents the template generated by overlay beat, new template operation after reclassification or interval reanalysis, and II represents the overlay channel used by the template."),
            ("auto_update", "AUTO", "Template generated by updating the beat is represented by 'N'")
        ]
        for key, txt, tooltip in badges_info:
            badge = QLabel(txt)
            badge.setToolTip(tooltip)
            badge.setAlignment(Qt.AlignCenter)
            badge.setVisible(False)
            badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:800;background:#17243A;border:1px solid {UI_BORDER};border-radius:8px;padding:2px 6px;")
            self.badge_labels[key] = badge
            badge_row.addWidget(badge)
        self.setToolTip("It means that all the beat forms under the template are similar in the template")
        self.badge_container.setVisible(False)
        layout.addWidget(self.badge_container)

        self.thumb = ECGStripCanvas(height=68, color="#39D353", pen_width=1.0)
        self.thumb.setStyleSheet(f"background:{COL_BLACK};border:1px solid {self._accent};border-radius:10px;")
        layout.addWidget(self.thumb, 1)

        bottom = QHBoxLayout()
        bottom.setSpacing(6)
        self.template_no = QLabel("#T1")
        self.template_no.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:700;border:none;")
        self.view_toggle = QToolButton()
        self.view_toggle.setCheckable(True)
        self.view_toggle.setChecked(True)
        self.view_toggle.setText("\u25c9")
        self.view_toggle.setToolTip("Viewed / unconfirmed")
        self.view_toggle.setFixedSize(26, 26)
        self.view_toggle.setStyleSheet(f"""
            QToolButton {{
                border: 1px solid {UI_BORDER};
                border-radius: 13px;
                color: {UI_TEXT};
                background: #0C1320;
                font-weight: 900;
            }}
            QToolButton:hover {{
                border-color: {UI_ACCENT_HOVER};
                background: #11203A;
            }}
            QToolButton:checked {{
                background: {UI_SUCCESS};
                color: #07111F;
                border-color: {UI_SUCCESS};
            }}
        """)
        bottom.addWidget(self.template_no)
        bottom.addStretch()
        bottom.addWidget(self.view_toggle)
        layout.addLayout(bottom)

        self.id_edit.editingFinished.connect(self._on_id_edit_finished)
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        self.view_toggle.toggled.connect(self._on_view_toggle)

        for widget in [self.id_edit, self.class_combo, self.count_pill, self.badge_container, self.thumb, self.template_no, self.view_toggle]:
            try:
                widget.installEventFilter(self)
            except Exception:
                pass
        for badge in self.badge_labels.values():
            try:
                badge.installEventFilter(self)
            except Exception:
                pass

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self, event)
        return super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.double_clicked.emit(self)
        return super().mouseDoubleClickEvent(event)

    def contextMenuEvent(self, event):
        host = self._find_template_host()
        if host is not None and hasattr(host, "_show_template_card_menu"):
            host._show_template_card_menu(self, event.globalPos())
            event.accept()
            return
        return super().contextMenuEvent(event)

    def _on_view_toggle(self, checked: bool):
        self.view_toggle.setText("\u25c9" if checked else "\u25cb")
        self.viewed_changed.emit(self, bool(checked))

    def _on_id_edit_finished(self):
        try:
            self.template_id_changed.emit(self, self.id_edit.text())
        except Exception as e:
            print(f"Error in _on_id_edit_finished: {e}")

    def _on_class_changed(self, text: str):
        try:
            # Defer processing to avoid destroying widget during signal emission
            QTimer.singleShot(0, lambda: self._handle_class_changed(text))
        except Exception as e:
            print(f"Error in _on_class_changed: {e}")
    
    def _handle_class_changed(self, text: str):
        try:
            self.class_changed.emit(self, text)
        except Exception as e:
            print(f"Error in _handle_class_changed: {e}")

    def _apply_styles(self):
        if self._selected:
            border = UI_ACCENT_HOVER
            bg_top = "#18263C"
            bg_bottom = "#101B2B"
        else:
            border = "#6EB4FF" if self._hovered else self._accent
            bg_top = "#141F31" if self._hovered else "#121C2D"
            bg_bottom = "#101827" if self._hovered else "#0D1521"
        self.setStyleSheet(self._card_style.format(border=border, bg_top=bg_top, bg_bottom=bg_bottom, ui_text=UI_TEXT, ui_accent=UI_ACCENT))
        self.count_pill.setStyleSheet(f"color:{UI_TEXT};font-size:11px;font-weight:800;background:#152235;border:1px solid {border};border-radius:10px;padding:4px 8px;")
        self.thumb.setStyleSheet(f"background:{COL_BLACK};border:1px solid {border};border-radius:10px;")

    def set_accent(self, accent: str):
        self._accent = accent or UI_BORDER
        self._apply_styles()

    def set_selected(self, selected: bool):
        self._selected = bool(selected)
        self._apply_styles()

    def enterEvent(self, event):
        self._hovered = True
        self._apply_styles()
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self._apply_styles()
        return super().leaveEvent(event)

    def set_template_data(self, data: dict):
        self._template_key = str(data.get("template_key") or data.get("template_id") or data.get("id") or "")
        template_id = str(data.get("template_id", data.get("id", "T?")))
        label = str(data.get("label", "N"))
        count = int(data.get("count", 0) or 0)
        self.id_edit.blockSignals(True)
        self.class_combo.blockSignals(True)
        self.view_toggle.blockSignals(True)
        self.id_edit.setText(template_id)
        if label and self.class_combo.findText(label) < 0:
            self.class_combo.addItem(label)
        self.class_combo.setCurrentText(label if self.class_combo.findText(label) >= 0 else (label[:1] if label else "N"))
        self.count_pill.setText(f"{count} beats")
        idx = data.get("index")
        self.template_no.setText(f"#{idx}" if idx is not None else template_id)

        flags = {
            "ambiguous": bool(data.get("ambiguous")),
            "inserted": bool(data.get("inserted")),
            "demix": bool(data.get("demix")),
            "auto_update": bool(data.get("auto_update")),
        }
        any_badge = False
        for key, badge in self.badge_labels.items():
            visible = flags.get(key, False)
            any_badge = any_badge or visible
            badge.setVisible(visible)
            if key == "ambiguous":
                badge.setStyleSheet(f"color:{UI_WARNING};font-size:10px;font-weight:900;background:#221F12;border:1px solid {UI_WARNING};border-radius:8px;padding:2px 6px;")
            elif key == "inserted":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#17311F;border:1px solid {UI_SUCCESS};border-radius:8px;padding:2px 6px;")
            elif key == "demix":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#122C46;border:1px solid #2D9CDB;border-radius:8px;padding:2px 6px;")
            elif key == "auto_update":
                badge.setStyleSheet(f"color:{UI_TEXT};font-size:10px;font-weight:900;background:#1F2030;border:1px solid {UI_MUTED};border-radius:8px;padding:2px 6px;")
        self.badge_container.setVisible(any_badge)

        self.view_toggle.blockSignals(True)
        self.view_toggle.setChecked(bool(data.get("viewed", True)))
        self.view_toggle.setText("\u25c9" if self.view_toggle.isChecked() else "\u25cb")
        self.view_toggle.blockSignals(False)
        self.id_edit.blockSignals(False)
        self.class_combo.blockSignals(False)

        waveform = data.get("waveform")
        if waveform is not None:
            x = np.linspace(0, len(waveform) / 500.0, len(waveform)) if len(waveform) else []
            self.thumb.set_data(x, waveform)
        else:
            self.thumb.set_data([], [])


class HolterBeatTemplatePanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._template_rows = []
        self._current_filter = "all"
        self._selected_template_key = ""
        self._selected_template_keys = []
        self._card_widgets = []
        self._waveform_cache = {}
        self._replay_engine = None
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Template System")
        title.setStyleSheet(f"color:#07111F;font-size:13px;font-weight:bold;background:qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 #28E37B, stop:1 #89F7C5);padding:7px 12px;border-radius:8px;")
        layout.addWidget(title)

        stats = QGridLayout()
        stats.setHorizontalSpacing(8)
        stats.setVerticalSpacing(8)
        self._stat_cards = {
            "total": _TemplateMetricCard("Total beats", "0", COL_TEXT),
            "templates": _TemplateMetricCard("Templates", "0", COL_TEXT),
            "unconfirmed": _TemplateMetricCard("Unconfirmed", "0", UI_WARNING),
            "beat_distribution": _TemplateMetricCard("Beat distribution", "N: 0", UI_TEXT),
        }
        stats.addWidget(self._stat_cards["total"], 0, 0)
        stats.addWidget(self._stat_cards["templates"], 0, 1)
        stats.addWidget(self._stat_cards["unconfirmed"], 0, 2)
        stats.addWidget(self._stat_cards["beat_distribution"], 0, 3)
        layout.addLayout(stats)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        filter_lbl = QLabel("Filter")
        filter_lbl.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:700;border:none;")
        filter_row.addWidget(filter_lbl)
        self._filter_buttons = {}
        for key, text in [
            ("all", "All"),
            ("N", "N"),
            ("S", "S"),
            ("V", "V"),
            ("P", "P"),
            ("AF", "AF"),
            ("X", "X"),
            ("Other", "Other"),
            ("unconfirmed", "Unconfirmed"),
        ]:
            btn = QPushButton(text)
            btn.setCheckable(True)
            btn.setChecked(key == "all")
            btn.setToolTip({
                "all": "Show all templates",
                "N": "Show normal templates",
                "S": "Show supraventricular templates",
                "V": "Show ventricular templates",
                "P": "Show paced templates",
                "AF": "Show atrial fibrillation/flutter templates",
                "X": "Show artifact templates",
                "Other": "Show uncategorized templates",
                "unconfirmed": "Show templates not yet confirmed",
            }.get(key, text))
            btn.setStyleSheet(_style_btn() if key != "all" else _style_active_btn())
            btn.clicked.connect(lambda checked=False, k=key: self._set_filter(k))
            self._filter_buttons[key] = btn
            filter_row.addWidget(btn)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.NoFrame)
        self._scroll.setStyleSheet(f"QScrollArea{{background:{COL_BG};border:none;}}")
        self._cards_host = QWidget()
        self._cards_host.setStyleSheet(f"background:{COL_BG};")
        self._cards_layout = QGridLayout(self._cards_host)
        self._cards_layout.setContentsMargins(0, 0, 0, 0)
        self._cards_layout.setSpacing(7)
        self._cards_layout.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self._scroll.setWidget(self._cards_host)

        self._splitter = QSplitter(Qt.Horizontal)
        self._splitter.addWidget(self._scroll)

        self._detail_grid_host = QWidget()
        self._detail_grid_host.setStyleSheet(f"background:{COL_BLACK};border-left:1px solid {COL_GREEN_DRK};")
        self._detail_layout = QGridLayout(self._detail_grid_host)
        self._detail_layout.setContentsMargins(4, 4, 4, 4)
        self._detail_layout.setSpacing(4)
        self._detail_canvases = []
        lead_names_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            lead_name = lead_names_12[i]
            frame = QFrame()
            frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};}}")
            flayout = QVBoxLayout(frame)
            flayout.setContentsMargins(2, 2, 2, 2)
            flayout.setSpacing(0)

            info = QWidget()
            info.setFixedHeight(16)
            info_layout = QHBoxLayout(info)
            info_layout.setContentsMargins(2, 0, 2, 0)
            info_layout.setSpacing(4)
            lbl_type = QLabel(lead_name)
            lbl_type.setStyleSheet(f"color:{COL_WHITE};font-weight:bold;font-size:10px;border:none;background:transparent;")
            info_layout.addWidget(lbl_type)
            info_layout.addStretch()
            flayout.addWidget(info)

            canvas = ECGStripCanvas(height=60, color=COL_GREEN, pen_width=1.0, lead_name=lead_name, show_annotations=True)
            canvas.set_paper_speed(25)
            canvas.set_gain(1.0)
            flayout.addWidget(canvas, 1)

            self._detail_layout.addWidget(frame, i // 3, i % 3)
            self._detail_canvases.append({"frame": frame, "canvas": canvas, "type": lbl_type, "lead_name": lead_name})

        self._splitter.addWidget(self._detail_grid_host)
        self._splitter.setSizes([220, 780])
        layout.addWidget(self._splitter, 1)

    def update_from_metrics(self, metrics_list: list, summary: dict):
        class_totals = dict(summary.get('beat_class_totals', {}) or {})
        if not class_totals:
            for metric in metrics_list or []:
                for cls, count in (metric.get('beat_class_counts', {}) or {}).items():
                    class_totals[cls] = class_totals.get(cls, 0) + int(count or 0)
        total_beats = int(summary.get("total_beats", 0) or sum(class_totals.values()) or 0)
        template_map = {}
        for metric in metrics_list or []:
            for row in (metric.get('template_summary', []) or []):
                tkey = str(row.get('template_key') or row.get('template_id') or row.get('label') or 'T')
                item = template_map.setdefault(tkey, {
                    'template_key': tkey,
                    'template_id': row.get('template_id', 'T'),
                    'label': row.get('label', 'N'),
                    'count': 0,
                    'rr': [],
                    'qrs': [],
                    'first_timestamp': float(row.get('first_timestamp', metric.get('t', 0.0)) or 0.0),
                    'viewed': bool(row.get('viewed', True)),
                    'ambiguous': bool(row.get('ambiguous', False)),
                    'inserted': bool(row.get('inserted', False)),
                    'demix': bool(row.get('demix', False)),
                    'auto_update': bool(row.get('auto_update', False)),
                })
                item['count'] += int(row.get('count', 0) or 0)
                item['rr'].append(float(row.get('avg_rr_ms', 0.0) or 0.0))
                item['qrs'].append(float(row.get('avg_qrs_ms', 0.0) or 0.0))
                item['first_timestamp'] = min(item['first_timestamp'], float(row.get('first_timestamp', item['first_timestamp']) or item['first_timestamp']))
        self._template_rows = sorted(template_map.values(), key=lambda x: x['count'], reverse=True)

        unconfirmed = sum(1 for row in self._template_rows if not row.get("viewed", True))
        self._stat_cards["total"].value.setText(f"{total_beats:,}")
        self._stat_cards["templates"].value.setText(str(len(self._template_rows)))
        self._stat_cards["unconfirmed"].value.setText(str(unconfirmed))
        dist_parts = []
        for key in ("N", "V", "S", "F", "Q"):
            if int(class_totals.get(key, 0) or 0) > 0:
                dist_parts.append(f"{key}: {int(class_totals.get(key, 0)):,}")
        self._stat_cards["beat_distribution"].value.setText("  ".join(dist_parts) if dist_parts else "No beat classes")
        self._refresh_stats()
        self._render_cards()

    def set_replay_engine(self, engine):
        self._replay_engine = engine
        self._waveform_cache.clear()
        self._render_cards()

    def _row_key(self, row: dict) -> str:
        return str(row.get("template_key") or row.get("template_id") or row.get("label") or "")

    def _filtered_rows(self):
        filtered = []
        for row in self._template_rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            code = _template_filter_key(label)
            
            if self._current_filter == "all":
                filtered.append(row)
            elif self._current_filter == "unconfirmed" and not row.get("viewed", True):
                filtered.append(row)
            elif self._current_filter == "AF" and code in {"AF", "F", "Q"}:
                filtered.append(row)
            elif self._current_filter == "Other" and code == "Other":
                filtered.append(row)
            elif self._current_filter in {"N", "V", "S", "P", "X"} and code == self._current_filter:
                filtered.append(row)
        return filtered

    def _refresh_stats(self):
        total_beats = sum(int(row.get("count", 0) or 0) for row in self._template_rows)
        unconfirmed = sum(1 for row in self._template_rows if not row.get("viewed", True))
        label_totals = {}
        for row in self._template_rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            label_totals[label] = label_totals.get(label, 0) + int(row.get("count", 0) or 0)
        self._stat_cards["total"].value.setText(f"{int(total_beats):,}")
        self._stat_cards["templates"].value.setText(str(len(self._template_rows)))
        self._stat_cards["unconfirmed"].value.setText(str(unconfirmed))

        preferred = ["N", "S", "V", "P", "AF", "X", "Unconfirmed"]
        dist_parts = []
        for key in preferred:
            value = int(label_totals.pop(key, 0) or 0)
            if value > 0:
                dist_parts.append(f"{key}: {value:,}")
        for key in sorted(label_totals):
            value = int(label_totals.get(key, 0) or 0)
            if value > 0:
                dist_parts.append(f"{key}: {value:,}")
        self._stat_cards["beat_distribution"].value.setText("  ".join(dist_parts) if dist_parts else "No beat classes")

    def _set_selected_keys(self, keys):
        ordered = []
        for key in keys or []:
            value = str(key or "").strip()
            if value and value not in ordered:
                ordered.append(value)
        self._selected_template_keys = ordered
        self._selected_template_key = ordered[0] if ordered else ""
        self._sync_card_selection()

    def _sync_card_selection(self):
        selected = set(self._selected_template_keys)
        visible = []
        for card in self._card_widgets:
            try:
                is_selected = card._template_key in selected
                card.set_selected(is_selected)
                if is_selected:
                    visible.append(card._template_key)
            except Exception:
                pass
        if visible:
            self._selected_template_key = visible[0]
        elif self._selected_template_keys:
            self._selected_template_key = self._selected_template_keys[0]
        else:
            self._selected_template_key = ""
            
        if self._selected_template_key:
            self._update_detail_grid(self._selected_template_key)

    def _row_for_key(self, template_key: str):
        template_key = str(template_key or "")
        for row in self._template_rows:
            if self._row_key(row) == template_key:
                return row
        return None

    def _selected_rows(self):
        selected = set(self._selected_template_keys)
        return [row for row in self._template_rows if self._row_key(row) in selected]

    def _selected_keys_for_card(self, card):
        card_key = str(getattr(card, "_template_key", "") or "")
        if self._selected_template_keys and card_key in set(self._selected_template_keys):
            return list(self._selected_template_keys)
        return [card_key] if card_key else []

    def _rename_template_key(self, old_key: str, new_key: str):
        old_key = str(old_key or "")
        new_key = str(new_key or "")
        if not old_key or not new_key or old_key == new_key:
            return
        for row in self._template_rows:
            if self._row_key(row) == old_key:
                row["template_key"] = new_key
        if old_key in self._waveform_cache:
            self._waveform_cache[new_key] = self._waveform_cache.pop(old_key)
        self._selected_template_keys = [new_key if key == old_key else key for key in self._selected_template_keys]
        if self._selected_template_key == old_key:
            self._selected_template_key = new_key

    def _apply_template_label(self, keys, label: str):
        key_set = set(keys or [])
        changed = False
        for row in self._template_rows:
            if self._row_key(row) in key_set:
                row["label"] = str(label or "N").strip() or "N"
                changed = True
        if changed:
            self._refresh_stats()
            self._render_cards()

    def _apply_template_viewed(self, keys, viewed: bool):
        key_set = set(keys or [])
        changed = False
        for row in self._template_rows:
            if self._row_key(row) in key_set:
                row["viewed"] = bool(viewed)
                changed = True
        if changed:
            self._refresh_stats()
            self._render_cards()

    def _select_all_visible(self):
        self._set_selected_keys([self._row_key(row) for row in self._filtered_rows()])

    def _reverse_visible_selection(self):
        visible = [self._row_key(row) for row in self._filtered_rows()]
        selected = set(self._selected_template_keys)
        new_keys = [key for key in self._selected_template_keys if key not in visible]
        new_keys.extend(key for key in visible if key not in selected)
        self._set_selected_keys(new_keys)
    def _set_filter(self, key: str):
        try:
            self._current_filter = key
            for k, btn in self._filter_buttons.items():
                btn.setChecked(k == key)
                btn.setStyleSheet(_style_active_btn() if k == key else _style_btn())
            self._refresh_stats()
            self._render_cards()
        except Exception as e:
            print(f"Error in _set_filter: {e}")
            import traceback
            traceback.print_exc()

    def _on_card_template_id_changed(self, card, text: str):
        try:
            old_key = str(getattr(card, "_template_key", "") or "")
            new_id = str(text or "").strip() or old_key
            row = self._row_for_key(old_key)
            if row is None:
                return
            row["template_id"] = new_id
            row["template_key"] = new_id
            if old_key != new_id:
                self._rename_template_key(old_key, new_id)
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_template_id_changed: {e}")

    def _on_card_class_changed(self, card, text: str):
        try:
            key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(key)
            if row is None:
                return
            row["label"] = str(text or "N").strip() or "N"
            
            if text == "Unconfirmed":
                row["viewed"] = False
            else:
                row["viewed"] = True  # Mark as viewed for other classifications
            
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_class_changed: {e}")

    def _on_card_viewed_changed(self, card, viewed: bool):
        try:
            key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(key)
            if row is None:
                return
            row["viewed"] = bool(viewed)
            self._refresh_stats()
            # Defer card rendering to avoid destroying card during signal
            QTimer.singleShot(10, self._render_cards)
        except Exception as e:
            print(f"Error in _on_card_viewed_changed: {e}")

    def _delete_selected_templates(self, keys=None):
        keys = list(keys or self._selected_template_keys)
        if not keys:
            return
        if QMessageBox.question(self, "Delete templates", f"Delete {len(keys)} selected template(s)?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        key_set = set(keys)
        self._template_rows = [row for row in self._template_rows if self._row_key(row) not in key_set]
        for key in key_set:
            self._waveform_cache.pop(key, None)
        self._selected_template_keys = []
        self._selected_template_key = ""
        self._refresh_stats()
        self._render_cards()

    def _confirm_delete_templates(self, count: int) -> bool:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Delete templates")
        box.setText(f"Delete {count} selected template(s)?")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.No)
        box.setStyleSheet(f"""
            QMessageBox {{ background: {COL_BLACK}; }}
            QMessageBox QLabel {{ color: {COL_WHITE}; font-size: 12px; }}
            QMessageBox QPushButton {{
                background: #1B2740;
                color: {COL_WHITE};
                border: 1px solid {UI_BORDER};
                border-radius: 6px;
                padding: 6px 14px;
                min-width: 64px;
            }}
            QMessageBox QPushButton:hover {{
                background: #243552;
                border-color: {UI_ACCENT};
            }}
            QMessageBox QPushButton:pressed {{
                background: #132033;
            }}
        """)
        return box.exec_() == QMessageBox.Yes
    def _merge_selected_templates(self, keys=None):
        keys = list(keys or self._selected_template_keys)
        key_set = set(keys)
        rows = [row for row in self._template_rows if self._row_key(row) in key_set]
        
        # Check if only one template is selected
        if len(rows) < 2:
            _show_message_box(
                self,
                QMessageBox.Information,
                "Merge Templates",
                "Only one template selected.\n\n"
                "Ctrl + Click to select manually to merge template\n"
                "Shift + Click to select all to merge template"
            )
            return
        
        options = [
            ("Normal", "N"),
            ("Atrial Premature", "S"),
            ("Ventricular Premature", "V"),
            ("Artifact", "X"),
            ("Atrial Fibrillation", "Q"),
            ("Atrial Flutter", "R"),
            ("Blocked PAC", "O"),
            ("Paced", "P"),
            ("Other", "Other"),
        ]
        labels = []
        for row in rows:
            label = str(row.get("label", "N") or "N").strip() or "N"
            if label not in labels:
                labels.append(label)
        if len(labels) == 1:
            merged_label = labels[0]
        else:
            items = [f"{name} ({code})" if code != "Other" else name for name, code in options]
            choice, ok = QInputDialog.getItem(self, "Merge template", "Choose the merged template type:", items, 0, False)
            if not ok:
                return
            merged_label = choice[choice.rfind("(") + 1:-1] if "(" in choice and choice.endswith(")") else choice
        base = dict(rows[0])
        total_count = sum(int(row.get("count", 0) or 0) for row in rows)
        rr_values = []
        qrs_values = []
        for row in rows:
            rr_values.extend([float(v) for v in (row.get("rr", []) or []) if v is not None])
            qrs_values.extend([float(v) for v in (row.get("qrs", []) or []) if v is not None])
        base["label"] = merged_label
        base["count"] = total_count
        base["rr"] = rr_values
        base["qrs"] = qrs_values
        base["first_timestamp"] = min(float(row.get("first_timestamp", 0.0) or 0.0) for row in rows)
        base["viewed"] = all(bool(row.get("viewed", True)) for row in rows)
        base["ambiguous"] = any(bool(row.get("ambiguous", False)) for row in rows)
        base["inserted"] = True  # Always show "+" badge for merged templates
        base["demix"] = any(bool(row.get("demix", False)) for row in rows)
        base["auto_update"] = any(bool(row.get("auto_update", False)) for row in rows)
        base["_original_templates"] = [dict(row) for row in rows]  # Store original templates for unmerge
        base.setdefault("template_id", base.get("template_id") or f"T{len(self._template_rows) + 1}")
        base["template_key"] = base.get("template_id") or base.get("label") or "T"
        new_rows = []
        inserted = False
        for row in self._template_rows:
            key = self._row_key(row)
            if key in key_set:
                self._waveform_cache.pop(key, None)
                if not inserted:
                    new_rows.append(base)
                    inserted = True
            else:
                new_rows.append(row)
        self._template_rows = new_rows
        self._selected_template_keys = [base["template_key"]]
        self._selected_template_key = base["template_key"]
        self._refresh_stats()
        self._render_cards()

    def _unmerge_template(self, template_key):
        row = self._row_for_key(template_key)
        if row is None or "_original_templates" not in row:
            return
        
        original_templates = row["_original_templates"]
        if not original_templates:
            return
        
        # Replace the merged template with the original templates
        new_rows = []
        found = False
        for r in self._template_rows:
            key = self._row_key(r)
            if key == template_key:
                found = True
                # Add back the original templates, removing the _original_templates field to avoid recursion
                for orig in original_templates:
                    orig_copy = dict(orig)
                    orig_copy.pop("_original_templates", None)
                    new_rows.append(orig_copy)
            else:
                new_rows.append(r)
        
        if found:
            self._template_rows = new_rows
            self._waveform_cache.pop(template_key, None)
            # Select the first original template
            if original_templates:
                first_key = self._row_key(original_templates[0])
                self._selected_template_keys = [first_key]
                self._selected_template_key = first_key
            self._refresh_stats()
            self._render_cards()

    def _open_overlay_analysis(self, card):
        host = self.window()
        if host is None:
            return
        try:
            if hasattr(host, "_focus_tab"):
                host._focus_tab("REPLAY")
            template_key = str(getattr(card, "_template_key", "") or "")
            row = self._row_for_key(template_key)
            if row is not None and hasattr(host, "set_magnifier_focus"):
                host.set_magnifier_focus(card.thumb, card.thumb._magnifier_source_payload(), QPoint(max(8, card.thumb.width() // 2), max(8, card.thumb.height() // 2)))
                self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
        except Exception:
            pass

    def _open_lorenz_plots(self, card):
        host = self.window()
        if host is None:
            return
        try:
            if hasattr(host, "_focus_tab"):
                host._focus_tab("REPLAY")
            row = self._row_for_key(str(getattr(card, "_template_key", "") or ""))
            if row is not None:
                self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
        except Exception:
            pass

    def _show_template_card_menu(self, card, global_pos):
        card_key = str(getattr(card, "_template_key", "") or "")
        if not card_key:
            return
        if not self._selected_template_keys or card_key not in set(self._selected_template_keys):
            self._set_selected_keys([card_key])
        target_keys = list(self._selected_template_keys) or [card_key]
        menu = QMenu(self)
        menu.setStyleSheet(f"QMenu {{ background: #0B1220; color: {UI_TEXT}; border: 1px solid {UI_BORDER}; padding: 6px; }} QMenu::item {{ padding: 6px 20px 6px 18px; border-radius: 4px; }} QMenu::item:selected {{ background: {UI_ACCENT}; color: #07111F; }} QMenu::separator {{ height: 1px; background: {UI_BORDER}; margin: 6px 4px; }}")
        prop_menu = menu.addMenu("Template properties")
        # Updated options: N, S, V, P, AF, X, Other, Unconfirmed
        for title, code in [
            ("Normal (N)", "N"),
            ("Atrial Premature (S)", "S"),
            ("Ventricular Premature (V)", "V"),
            ("Paced (P)", "P"),
            ("Atrial Fibrillation (AF)", "AF"),
            ("Artifact (X)", "X"),
            ("Other", "Other"),
            ("Unconfirmed", "Unconfirmed")
        ]:
            action = prop_menu.addAction(title)
            action.triggered.connect(lambda checked=False, c=code: self._apply_template_label(target_keys, c))
        func_menu = menu.addMenu("Function")
        func_menu.addAction("Delete").triggered.connect(lambda: self._delete_selected_templates(target_keys))
        func_menu.addAction("Select All").triggered.connect(self._select_all_visible)
        func_menu.addAction("Reverse Selection").triggered.connect(self._reverse_visible_selection)
        merge_action = menu.addAction("Merge template")
        merge_action.setEnabled(True)  # Always enable - let merge function handle validation
        merge_action.triggered.connect(lambda: self._merge_selected_templates(target_keys))
        
        # Add Unmerge Template option
        unmerge_action = menu.addAction("Unmerge template")
        # Enable only if we have a single selected template that has original templates stored
        has_original = len(target_keys) == 1
        if has_original:
            row = self._row_for_key(target_keys[0])
            has_original = row is not None and "_original_templates" in row
        unmerge_action.setEnabled(has_original)
        unmerge_action.triggered.connect(lambda: self._unmerge_template(target_keys[0]))
        
        ok_menu = menu.addMenu("Template OK/Cancel")
        ok_menu.addAction("Confirm").triggered.connect(lambda: self._apply_template_viewed(target_keys, True))
        ok_menu.addAction("Unconfirm").triggered.connect(lambda: self._apply_template_viewed(target_keys, False))
        menu.addAction("Overlay beat analysis").triggered.connect(lambda: self._open_overlay_analysis(card))
        menu.addAction("Lorenz plots").triggered.connect(lambda: self._open_lorenz_plots(card))
        menu.exec_(global_pos)
    def _on_card_clicked(self, card, event):
        template_key = str(getattr(card, "_template_key", "") or "")
        if not template_key:
            return
        
        modifiers = event.modifiers() if event else Qt.NoModifier
        ctrl_pressed = bool(modifiers & Qt.ControlModifier)
        shift_pressed = bool(modifiers & Qt.ShiftModifier)
        
        filtered = self._filtered_rows()
        filtered_keys = [self._row_key(row) for row in filtered]
        
        if ctrl_pressed:
            # Toggle selection
            current = list(self._selected_template_keys)
            if template_key in current:
                current.remove(template_key)
            else:
                current.append(template_key)
            self._set_selected_keys(current)
        elif shift_pressed and self._selected_template_keys:
            # Range selection
            last_key = self._selected_template_keys[-1]
            try:
                last_idx = filtered_keys.index(last_key)
                curr_idx = filtered_keys.index(template_key)
                start = min(last_idx, curr_idx)
                end = max(last_idx, curr_idx)
                self._set_selected_keys(filtered_keys[start:end+1])
            except ValueError:
                self._set_selected_keys([template_key])
        else:
            # Single selection
            self._set_selected_keys([template_key])
        
        row = self._row_for_key(template_key)
        if template_key and row and not ctrl_pressed and not shift_pressed:
            self.seek_requested.emit(float(row.get("first_timestamp", 0.0) or 0.0))
            
        self._update_detail_grid(template_key)

    def _on_card_double_clicked(self, card):
        """On double-click, select the card and immediately load 12-lead waveforms."""
        template_key = str(getattr(card, "_template_key", "") or "")
        if not template_key:
            return
        # Select only this card
        self._set_selected_keys([template_key])
        # Scroll the detail grid into view if splitter is collapsed
        if hasattr(self, "_splitter"):
            sizes = self._splitter.sizes()
            if sizes and len(sizes) == 2 and sizes[1] < 100:
                total = sum(sizes)
                self._splitter.setSizes([220, max(100, total - 220)])
        # Load all 12 leads for this template
        self._update_detail_grid(template_key)

    def _update_detail_grid(self, template_key: str):
        if not hasattr(self, "_detail_canvases") or not self._replay_engine:
            return

        row = self._row_for_key(template_key)
        label = "N"
        first_ts = None
        rr_med   = 0.0
        qrs_med  = 0.0
        if row:
            label     = str(row.get("label", "N") or "N").strip() or "N"
            first_ts  = float(row.get("first_timestamp", 0.0) or 0.0)
            rr_list   = row.get("rr", []) or []
            qrs_list  = row.get("qrs", []) or []
            rr_med    = float(np.median(rr_list))  if rr_list  else 0.0
            qrs_med   = float(np.median(qrs_list)) if qrs_list else 0.0

        # Clear canvases
        for item in self._detail_canvases:
            item["canvas"].set_data([], [])
            item["type"].setText(item["lead_name"])

        if first_ts is None:
            return

        engine = self._replay_engine

        # --- Read a single-beat window, mirroring _resolve_template_waveform exactly ---
        pre  = 0.18
        post = 0.34
        t_start = max(0.0, first_ts - pre)
        t_end   = min(float(engine.duration_sec), first_ts + post)
        data    = None
        n_real  = 0

        try:
            raw_data = engine._reader.read_range(t_start, t_end)
            if (isinstance(raw_data, np.ndarray)
                    and raw_data.ndim == 2
                    and raw_data.shape[0] >= 1
                    and raw_data.shape[1] > 8):
                data   = raw_data
                n_real = data.shape[0]
                print(f"[12-lead] data.shape={data.shape}  first_ts={first_ts:.3f}")
        except Exception as e:
            print(f"[12-lead] read_range error: {e}")

        beat_annotations = [{"timestamp": first_ts, "label": label, "type": "beat"}]
        x_out = np.linspace(0.0, pre + post, 240)

        for i, item in enumerate(self._detail_canvases):
            canvas = item["canvas"]
            waveform = None

            if data is not None and i < n_real:
                # ---- exact copy of _resolve_template_waveform normalization ----
                lead_raw = np.asarray(data[i], dtype=float)
                baseline = float(np.median(lead_raw))
                centered = lead_raw - baseline
                if np.ptp(centered) > 1.0:
                    x_old    = np.linspace(0.0, 1.0, centered.size)
                    x_new    = np.linspace(0.0, 1.0, 240)
                    centered = np.interp(x_new, x_old, centered)
                    centered = centered - float(np.median(centered))
                    peak     = max(float(np.max(np.abs(centered))), 1.0)
                    centered = np.clip(centered / peak * 420.0, -650.0, 650.0)
                    waveform = 2048.0 + centered

            if waveform is None:
                # fall back to the same synthetic waveform as the card thumbnail
                waveform = self._make_thumbnail_waveform(rr_med, qrs_med)

            canvas.set_data(x_out, waveform,
                            beat_annotations=beat_annotations,
                            start_sec=t_start)
            item["type"].setText(f"{item['lead_name']}  {label}")



    def _render_cards(self):
        try:
            # Disconnect all signals from old cards before deleting
            for card in self._card_widgets:
                try:
                    card.clicked.disconnect()
                    card.template_id_changed.disconnect()
                    card.class_changed.disconnect()
                    card.viewed_changed.disconnect()
                except Exception:
                    pass
            
            while self._cards_layout.count():
                item = self._cards_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
            
            self._card_widgets = []
            filtered = self._filtered_rows()
            filtered_keys = [self._row_key(row) for row in filtered]
            selected_visible = [key for key in filtered_keys if key in set(self._selected_template_keys)]
            if filtered and not selected_visible:
                self._set_selected_keys([filtered_keys[0]])
            elif selected_visible:
                self._selected_template_key = selected_visible[0]
            elif not self._selected_template_keys:
                self._selected_template_key = ""

            for idx, row in enumerate(filtered):
                label = str(row.get("label", "N") or "N").strip() or "N"
                label_code = label[:1].upper() if label else "N"
                accent = {"N": "#2D9CDB", "V": "#F2994A", "S": "#E2B93B", "F": "#7F8C8D", "Q": "#B06CFD", "R": "#A97FFF", "O": "#F2C94C", "P": "#56CCF2", "X": "#EB5757"}.get(label_code, UI_BORDER)
                card = TemplateCardWidget(accent=accent)
                rr = row.get("rr", []) or []
                qrs = row.get("qrs", []) or []
                rr_med = float(np.median(rr)) if rr else 0.0
                qrs_med = float(np.median(qrs)) if qrs else 0.0
                waveform = self._resolve_template_waveform(row, rr_med, qrs_med)
                template_key = self._row_key(row)
                card.clicked.connect(self._on_card_clicked)
                card.double_clicked.connect(self._on_card_double_clicked)
                card.template_id_changed.connect(self._on_card_template_id_changed)
                card.class_changed.connect(self._on_card_class_changed)
                card.viewed_changed.connect(self._on_card_viewed_changed)
                card.set_template_data({
                    "template_key": template_key,
                    "index": idx + 1,
                    "template_id": row.get("template_id", f"T{idx+1}"),
                    "label": row.get("label", "N"),
                    "count": row.get("count", 0),
                    "viewed": row.get("viewed", True),
                    "ambiguous": row.get("ambiguous", False),
                    "inserted": row.get("inserted", False),
                    "demix": row.get("demix", False),
                    "auto_update": row.get("auto_update", False),
                    "waveform": waveform,
                })
                card.set_selected(template_key in set(self._selected_template_keys))
                row_idx = idx % 3
                col_idx = idx // 3
                self._cards_layout.addWidget(card, row_idx, col_idx)
                self._card_widgets.append(card)
            
            self._sync_card_selection()
        except Exception as e:
            import traceback
            print(f"Error in _render_cards: {e}")
            traceback.print_exc()

    def _resolve_template_waveform(self, row: dict, rr_ms: float, qrs_ms: float):
        try:
            key = str(row.get("template_key") or row.get("template_id") or row.get("label") or "T")
            cached = self._waveform_cache.get(key)
            if cached is not None:
                return cached
            first_ts = float(row.get("first_timestamp", 0.0) or 0.0)
            waveform = None
            engine = getattr(self, "_replay_engine", None)
            try:
                if engine is not None and hasattr(engine, "_reader"):
                    fs = float(getattr(engine, "fs", 500.0) or 500.0)
                    pre = 0.18
                    post = 0.34
                    data = engine._reader.read_range(max(0.0, first_ts - pre), min(float(engine.duration_sec), first_ts + post))
                    if isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] > 1 and data.shape[1] > 8:
                        lead = np.asarray(data[1], dtype=float)
                        baseline = float(np.median(lead))
                        centered = lead - baseline
                        if np.ptp(centered) > 1.0:
                            x_old = np.linspace(0.0, 1.0, centered.size)
                            x_new = np.linspace(0.0, 1.0, 240)
                            centered = np.interp(x_new, x_old, centered)
                            centered = centered - float(np.median(centered))
                            peak = max(float(np.max(np.abs(centered))), 1.0)
                            centered = np.clip(centered / peak * 420.0, -650.0, 650.0)
                            waveform = 2048.0 + centered
            except Exception:
                waveform = None
            if waveform is None:
                waveform = self._make_thumbnail_waveform(rr_ms, qrs_ms)
            self._waveform_cache[key] = waveform
            return waveform
        except Exception as e:
            print(f"Error in _resolve_template_waveform: {e}")
            return self._make_thumbnail_waveform(rr_ms, qrs_ms)

    def _make_thumbnail_waveform(self, rr_ms: float, qrs_ms: float):
        t = np.linspace(0, 1.2, 240)
        rr_scale = np.clip(rr_ms / 900.0, 0.55, 1.8) if rr_ms > 0 else 1.0
        qrs_scale = np.clip(qrs_ms / 80.0, 0.6, 1.8) if qrs_ms > 0 else 1.0
        p = 0.04 * np.exp(-((t - 0.14) / 0.028) ** 2)
        q = -0.14 * np.exp(-((t - 0.275) / 0.010) ** 2)
        r = 1.35 * np.exp(-((t - 0.31) / (0.014 / qrs_scale)) ** 2)
        s = -0.36 * np.exp(-((t - 0.335) / 0.014) ** 2)
        tw = 0.22 * np.exp(-((t - 0.63) / (0.09 * rr_scale)) ** 2)
        st = 0.025 * np.exp(-((t - 0.45) / 0.05) ** 2)
        wave = p + q + r + s + tw + st
        return 2048.0 + wave * 480.0

    def _on_template_clicked(self, row, _col):
        if 0 <= row < len(self._template_rows):
            self.seek_requested.emit(float(self._template_rows[row].get("first_timestamp", 0.0)))


# -----------------------------------------------------------------------------
# 9. HOLTER INSIGHT PANEL  (report preview)
# -----------------------------------------------------------------------------

class HolterInsightPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:10px;}}")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)
        title = QLabel("Comprehensive Report Preview")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;background:{COL_DARK};"
                            f"padding:7px;border-radius:4px;border:none;")
        layout.addWidget(title)
        self._report = QTextEdit()
        self._report.setReadOnly(True)
        self._report.setStyleSheet(f"""
            QTextEdit{{background:{COL_BLACK};color:{COL_GREEN};
              border:1px solid {COL_GREEN_DRK};border-radius:8px;padding:10px;font-size:13px;}}
        """)
        layout.addWidget(self._report)

    def update_text(self, patient_info: dict, summary: dict):
        name = patient_info.get("patient_name") or patient_info.get("name") or "Unknown patient"
        age = patient_info.get("age", "-")
        sex = patient_info.get("gender") or patient_info.get("sex") or "-"
        email = patient_info.get("email", "-")
        dur = summary.get("duration_sec", 0) / 3600
        avg_hr = summary.get("avg_hr", 0)
        min_hr = summary.get("min_hr", 0)
        max_hr = summary.get("max_hr", 0)
        quality = summary.get("avg_quality", 0) * 100
        arrhythmias = summary.get("arrhythmia_counts", {})
        top = ", ".join(f"{k} ({v})" for k, v in sorted(arrhythmias.items(), key=lambda x: -x[1])[:4]) \
              or "No clinically significant arrhythmia burden detected."
        rhythm = ("predominantly tachycardic trend" if avg_hr >= 100
                  else "predominantly bradycardic trend" if 0 < avg_hr <= 60
                  else "predominantly sinus-range rhythm")
        text = (
            f"Patient: {name} | Age/Sex: {age}/{sex} | Email: {email}\n\n"
            f"Study summary:\n"
            f"- Recording duration: {dur:.1f} hours\n"
            f"- Average heart rate: {avg_hr:.0f} bpm (range {min_hr:.0f}-{max_hr:.0f} bpm)\n"
            f"- Signal quality: {quality:.1f}%\n"
            f"- Longest RR interval: {summary.get('longest_rr_ms',0):.0f} ms\n"
            f"- HRV profile: SDNN {summary.get('sdnn',0):.1f} ms, "
            f"rMSSD {summary.get('rmssd',0):.1f} ms, pNN50 {summary.get('pnn50',0):.2f}%\n\n"
            f"Interpretation:\n"
            f"The recording demonstrates a {rhythm}. Key events: {top}\n\n"
            f"Suggested final report wording:\n"
            f'"Comprehensive ECG Analysis monitoring for {name} shows {rhythm} with an average heart rate of '
            f'{avg_hr:.0f} bpm. The minimum recorded rate was {min_hr:.0f} bpm and the '
            f'maximum recorded rate was {max_hr:.0f} bpm. Overall signal quality was '
            f'{quality:.1f}%, enabling comprehensive review of the 12-lead trends and event strips."'
        )
        self._report.setPlainText(text)


# -----------------------------------------------------------------------------
# 10. RECORD MANAGEMENT PANEL
# -----------------------------------------------------------------------------

class HolterRecordManagementPanel(QWidget):
    session_selected = pyqtSignal(str)  # session dir path

    def __init__(self, output_dir: str = "recordings"):
        super().__init__()
        self.output_dir = output_dir
        self._selected_session = ""
        self._active_session = ""
        self._build_ui()
        self.refresh_records()
        
    def set_active_session(self, session_dir: str):
        """Set the active session to highlight in the table."""
        self._active_session = session_dir
        self.refresh_records()

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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        actions = QHBoxLayout()
        actions.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search patient / reporter / status")
        self._search.setStyleSheet(f"QLineEdit{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};"
                                   f"border-radius:4px;padding:6px;font-size:12px;}}")
        self._search.textChanged.connect(self.refresh_records)
        self._filter = QComboBox()
        self._filter.addItems(["All", "Today", "Yesterday", "This Week", "This Month", "This Year"])
        self._filter.setStyleSheet(f"""
            QComboBox {{
                background:{COL_DARK}; color:{COL_GREEN}; border:1px solid {COL_GREEN_DRK};
                border-radius:4px; padding:6px; font-size:12px;
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._filter.currentTextChanged.connect(self.refresh_records)
        actions.addWidget(QLabel("Search:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._search, 2)
        actions.addWidget(QLabel("Filter:", styleSheet=f"color:{COL_GREEN};font-size:12px;"))
        actions.addWidget(self._filter)
        self._action_buttons = {}
        for txt in ["Browse", "Import", "Export", "Backup", "Delete"]:
            btn = QPushButton(txt)
            btn.setStyleSheet(_style_btn())
            self._action_buttons[txt] = btn
            actions.addWidget(btn)
        layout.addLayout(actions)

        cols = ["Name","Age","Gender","Record Time","Duration","Channel","Import Time","Status","Reporter","Conclusion"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectRows)
        self._table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._table.setVerticalScrollMode(QAbstractItemView.ScrollPerItem)
        self._table.setHorizontalScrollMode(QAbstractItemView.ScrollPerPixel)
        self._table.verticalScrollBar().setSingleStep(1)
        self._table.itemSelectionChanged.connect(self._sync_selected_session)
        self._table.doubleClicked.connect(self._on_double_click)
        layout.addWidget(self._table, 1)
        self._action_buttons["Browse"].clicked.connect(self._browse_root)
        self._action_buttons["Import"].clicked.connect(self._import_session)
        self._action_buttons["Export"].clicked.connect(self._export_session)
        self._action_buttons["Backup"].clicked.connect(self._backup_root)
        self._action_buttons["Delete"].clicked.connect(self._delete_session)

    def refresh_records(self):
        self._table.setRowCount(0)
        self._selected_session = ""
        if not os.path.isdir(self.output_dir): return
        query = self._search.text().strip().lower()
        filter_label = self._filter.currentText()
        now = datetime.now()
        today = now.date()
        yesterday = today - timedelta(days=1)
        rows = []
        for name in sorted(os.listdir(self.output_dir), reverse=True):
            session_dir = os.path.join(self.output_dir, name)
            if not os.path.isdir(session_dir): continue
            if not os.path.exists(os.path.join(session_dir, "recording.ecgh")): continue
            try:
                mtime = datetime.fromtimestamp(os.path.getmtime(os.path.join(session_dir, "recording.ecgh")))
            except Exception:
                mtime = datetime.fromtimestamp(os.path.getmtime(session_dir))
            delta_days = (now - mtime).days
            if filter_label == "Today" and mtime.date() != today:
                continue
            if filter_label == "Yesterday" and mtime.date() != yesterday:
                continue
            if filter_label == "This Week" and delta_days > 6:
                continue
            if filter_label == "This Month" and (mtime.year != now.year or mtime.month != now.month):
                continue
            if filter_label == "This Year" and mtime.year != now.year:
                continue
            parts = name.split("_", 3)
            rec_time = "_".join(parts[:2]).replace("_", " ") if len(parts) >= 2 else name[:19]
            p_name = parts[-1].replace("_", " ") if len(parts) >= 3 else "Unknown"
            age = "-"
            gender = "-"
            dur_str = "-"
            
            import json
            # Check patient.json first (created by Save dialog)
            patient_json = os.path.join(session_dir, "patient.json")
            if os.path.exists(patient_json):
                try:
                    with open(patient_json, 'r') as f:
                        pdata = json.load(f)
                    saved_name = pdata.get('name') or pdata.get('patient_name') or pdata.get('full_name')
                    if saved_name and str(saved_name).strip() and str(saved_name).strip().lower() != "unknown":
                        p_name = str(saved_name).strip()
                    saved_age = pdata.get('age')
                    if saved_age: age = str(saved_age)
                    saved_gender = pdata.get('gender') or pdata.get('sex')
                    if saved_gender: gender = str(saved_gender)
                except Exception:
                    pass

            try:
                session_json = os.path.join(session_dir, "session.json")
                if os.path.exists(session_json):
                    with open(session_json, 'r') as f:
                        sdata = json.load(f)
                        
                    # If we didn't find good name in patient.json, check session.json
                    if p_name.lower() == "unknown":
                        p_info = sdata.get('patient_info') or sdata.get('summary', {}).get('patient_info') or {}
                        saved_name = p_info.get('name') or p_info.get('patient_name') or p_info.get('full_name')
                        if saved_name and str(saved_name).strip() and str(saved_name).strip().lower() != "unknown":
                            p_name = str(saved_name).strip()
                        saved_age = p_info.get('age')
                        if saved_age and age == "-": age = str(saved_age)
                        saved_gender = p_info.get('gender') or p_info.get('sex')
                        if saved_gender and gender == "-": gender = str(saved_gender)

                    dur_sec = sdata.get('summary', {}).get('duration_sec', 0)
                    if dur_sec > 0:
                        h = int(dur_sec // 3600)
                        m = int((dur_sec % 3600) // 60)
                        s = int(dur_sec % 60)
                        if h > 0:
                            dur_str = f"{h}h {m:02d}m"
                        elif m > 0:
                            dur_str = f"{m}m {s:02d}s"
                        else:
                            dur_str = f"{s}s"
            except Exception:
                pass

            row_values = [p_name, age, gender, rec_time, dur_str, "12", rec_time, "Completed", "System", "-"]
            if query and not any(query in str(v).lower() for v in row_values): continue
            rows.append((row_values, session_dir))

        for row_values, session_dir in rows:
            r = self._table.rowCount()
            self._table.insertRow(r)
            
            is_active = os.path.normpath(session_dir) == os.path.normpath(self._active_session)
            for c, v in enumerate(row_values):
                item = QTableWidgetItem(str(v))
                item.setForeground(QColor(COL_GREEN if c == 0 else COL_WHITE))
                item.setData(Qt.UserRole, session_dir)
                if is_active:
                    item.setBackground(QColor(46, 84, 137, 150))  # Semi-transparent blue highlight
                self._table.setItem(r, c, item)
        if self._table.rowCount() > 0:
            self._table.selectRow(0)
            self._sync_selected_session()

    def _open_row(self, row, _column=0):
        item = self._table.item(row, 0)
        if item:
            path = item.data(Qt.UserRole)
            if path:
                self._selected_session = path
                self.session_selected.emit(path)

    def _on_double_click(self, index):
        self._open_row(index.row())

    def _sync_selected_session(self):
        rows = self._table.selectionModel().selectedRows() if self._table.selectionModel() else []
        if rows:
            item = self._table.item(rows[0].row(), 0)
            if item:
                self._selected_session = item.data(Qt.UserRole) or ""

    def _selected_path(self) -> str:
        self._sync_selected_session()
        return self._selected_session

    def _browse_root(self):
        d = QFileDialog.getExistingDirectory(self, "Select Recordings Root", self.output_dir or os.getcwd())
        if d:
            self.output_dir = d
            self.refresh_records()

    def _import_session(self):
        src = QFileDialog.getExistingDirectory(self, "Import Session Folder")
        if not src:
            return
        if not os.path.exists(os.path.join(src, "recording.ecgh")):
            _show_message_box(self, QMessageBox.Warning, "Import Session", "Select a session folder that contains recording.ecgh.")
            return
        os.makedirs(self.output_dir, exist_ok=True)
        dest = os.path.join(self.output_dir, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Import Session", "That session already exists in the recordings folder.")
            return
        shutil.copytree(src, dest)
        self.refresh_records()
        self.session_selected.emit(dest)

    def _export_session(self):
        src = self._selected_path()
        if not src:
            _show_message_box(self, QMessageBox.Information, "Export Session", "Select a recording to export first.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Export Session To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(src)))
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Export Session", "That session already exists in the destination.")
            return
        shutil.copytree(src, dest)
        _show_message_box(self, QMessageBox.Information, "Export Session", f"Session exported to:\n{dest}")

    def _backup_root(self):
        if not os.path.isdir(self.output_dir):
            _show_message_box(self, QMessageBox.Information, "Backup", "No recordings folder found.")
            return
        dest_root = QFileDialog.getExistingDirectory(self, "Backup Recordings To")
        if not dest_root:
            return
        dest = os.path.join(dest_root, os.path.basename(os.path.normpath(self.output_dir)) or "recordings_backup")
        if os.path.exists(dest):
            _show_message_box(self, QMessageBox.Warning, "Backup", "That backup folder already exists.")
            return
        shutil.copytree(self.output_dir, dest)
        _show_message_box(self, QMessageBox.Information, "Backup", f"Recordings backed up to:\n{dest}")

    def _delete_session(self):
        src = self._selected_path()
        if not src:
            _show_message_box(self, QMessageBox.Information, "Delete Session", "Select a recording to delete first.")
            return
        if _show_message_box(self, QMessageBox.Question, "Delete Session",
                                f"Delete this recording?\n\n{src}",
                                QMessageBox.Yes | QMessageBox.No,
                                QMessageBox.No) != QMessageBox.Yes:
            return
        
        # First, try to find the main window and close the session if it's currently loaded
        try:
            # Find the main HolterUI window
            main_window = self.window()
            if main_window and hasattr(main_window, "close_current_session_if_needed"):
                main_window.close_current_session_if_needed(src)
        except Exception:
            pass
        
        # Force garbage collection to clean up any unreferenced file handles
        gc.collect()
        
        # First, try to delete individual files with retries
        max_file_retries = 4
        last_error = None
        
        # Try to delete all files in the directory first
        if os.path.exists(src):
            try:
                for root, dirs, files in os.walk(src, topdown=False):
                    for file_name in files:
                        file_path = os.path.join(root, file_name)
                        # Try to delete this file with retries
                        for file_attempt in range(max_file_retries):
                            try:
                                os.remove(file_path)
                                break
                            except Exception as e:
                                last_error = e
                                if file_attempt < max_file_retries - 1:
                                    gc.collect()
                                    time.sleep(0.3)
                                else:
                                    pass
                    # Try to delete subdirectories
                    for dir_name in dirs:
                        dir_path = os.path.join(root, dir_name)
                        for dir_attempt in range(max_file_retries):
                            try:
                                os.rmdir(dir_path)
                                break
                            except Exception as e:
                                last_error = e
                                if dir_attempt < max_file_retries - 1:
                                    gc.collect()
                                    time.sleep(0.3)
                                else:
                                    pass
            except Exception as e:
                last_error = e
        
        # Now try rmtree
        max_retries = 7
        retry_delay = 0.6
        
        for attempt in range(max_retries):
            try:
                gc.collect()
                if os.path.exists(src):
                    shutil.rmtree(src, ignore_errors=False)
                self.refresh_records()
                _show_message_box(self, QMessageBox.Information, "Delete Session", "Recording deleted successfully.")
                return
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    gc.collect()
                    time.sleep(retry_delay)
                    retry_delay *= 1.4  # exponential backoff
                else:
                    break
        
        # If we got here, all retries failed
        _show_message_box(self, QMessageBox.Warning, "Delete Session", f"Failed to delete recording:\n{str(last_error)}")
# -----------------------------------------------------------------------------
# 11b. LORENZ PANEL
# -----------------------------------------------------------------------------

class HolterLorenzPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None, duration_sec: float = 86400, replay_engine=None):
        super().__init__(parent)
        self.duration_sec = duration_sec
        self.replay_engine = replay_engine
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics_list = []
        self._all_beat_data = []  # Store (time, rr, beat_class, metric_ref) for beat selection
        self._current_replay_data = None  # Store current 12-lead frame for beat selection display
        self._current_replay_start_sec = 0.0
        self._current_replay_beat_annotations = None
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Top Section: Single RR Trend Canvas for Full Recording
        self.rr_trend_canvas = HolterRRTrendCanvas(self, title="RR Trend (Full Recording)")
        self.rr_trend_canvas.setFixedHeight(120)
        
        layout.addWidget(self.rr_trend_canvas)

        # Middle Section Splitter
        mid_splitter = QSplitter(Qt.Horizontal)
        
        # Left side of mid splitter: Lorenz Plot + Buttons
        lorenz_container = QWidget()
        l_layout = QVBoxLayout(lorenz_container)
        l_layout.setContentsMargins(0, 0, 0, 0)
        
        self.lorenz_canvas = LorenzCanvas()
        self.lorenz_canvas.beats_selected.connect(self._on_beats_selected)
        self.lorenz_canvas.beats_reclassified.connect(self._on_beats_reclassified)
        self.lorenz_canvas.beats_deleted.connect(self._on_beats_deleted)
        self.lorenz_canvas.setFixedHeight(350)
        l_layout.addWidget(self.lorenz_canvas)

        # Single row of beat-filter buttons (same as before)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        self._filter_btns = {}
        for lbl in ["All", "Normal", "S", "V", "Paced", "AF/AFl", "Other", "X", "Not including", "Search beats", "All"]:
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setStyleSheet(_style_btn())
            self._filter_btns[lbl] = btn
            btn_row.addWidget(btn)
        l_layout.addLayout(btn_row)

        # Internal state for view/display toggle helpers (used by right-click context menu)
        self._pixel_cycle = [2, 3, 5]
        self._pixel_idx   = 1

        mid_splitter.addWidget(lorenz_container)
        
        # Right side of mid splitter: 12-Lead ECG Grid (3 columns x 4 rows)
        leads_container = QWidget()
        leads_container.setStyleSheet(f"background:{COL_BLACK};border:1px solid {UI_BORDER};border-radius:6px;")
        leads_layout = QVBoxLayout(leads_container)
        leads_layout.setContentsMargins(4, 4, 4, 4)
        leads_layout.setSpacing(2)
        
        # Create 12-lead grid: 4 rows x 3 columns
        grid_layout = QGridLayout()
        grid_layout.setSpacing(10)
        grid_layout.setContentsMargins(6, 6, 6, 6)
        
        self._lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._lead_strips = {}  # Dictionary to store ECG strip canvases by lead name
        
        for idx, lead_name in enumerate(self._lead_names):
            row = idx // 3  # 0-3 (4 rows)
            col = idx % 3   # 0-2 (3 columns)
            
            # Create a frame for each lead
            lead_frame = QFrame()
            lead_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;")
            lead_frame_layout = QVBoxLayout(lead_frame)
            lead_frame_layout.setContentsMargins(2, 2, 2, 2)
            lead_frame_layout.setSpacing(1)
            
            # Lead label
            lead_label = QLabel(lead_name)
            lead_label.setStyleSheet(f"color:{COL_GREEN};font-weight:bold;font-size:10px;border:none;")
            lead_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            lead_frame_layout.addWidget(lead_label)
            
            # ECG strip canvas for this lead - with show_annotations=False to hide N labels and numbers
            strip = ECGStripCanvas(height=60, color=COL_GREEN, pen_width=0.8, lead_name=lead_name, show_vertical_lines=False, show_annotations=False)
            strip.set_gain(1.0)
            self._lead_strips[lead_name] = strip
            lead_frame_layout.addWidget(strip)
            
            grid_layout.addWidget(lead_frame, row, col)
        
        leads_layout.addLayout(grid_layout)
        mid_splitter.addWidget(leads_container)

        lorenz_container.setMaximumWidth(450)
        mid_splitter.setSizes([400, 1200])
        mid_splitter.setStretchFactor(0, 0)
        mid_splitter.setStretchFactor(1, 1)
        
        layout.addWidget(mid_splitter, stretch=2)

        # Bottom Section: ECG Strip
        self.ecg_canvas = ECGStripCanvas()
        self.ecg_canvas.setFixedHeight(90)
        layout.addWidget(self.ecg_canvas)

    def update_from_metrics(self, metrics_list, duration_sec=None):
        self._metrics_list = list(metrics_list or [])
        if duration_sec:
            self.duration_sec = duration_sec
            
        pts = []
        rr_all = []
        rr_classes = []
        
        for m in self._metrics_list:
            t0 = float(m.get('t', 0.0) or 0.0)
            beat_labels = list(m.get('all_beats') or [])
            if 'rr_intervals_list' in m:
                rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
                if rr_list:
                    dur = float(m.get('duration', 0.0) or 0.0)
                    step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                    for i, rr in enumerate(rr_list):
                        beat_info = beat_labels[i] if i < len(beat_labels) and isinstance(beat_labels[i], dict) else {}
                        beat_class = _normalize_beat_class(
                            beat_info.get("label")
                            or beat_info.get("template_label")
                            or beat_info.get("auto_label")
                            or m.get("label")
                            or m.get("template_label")
                            or m.get("arrhythmia")
                            or (m.get("arrhythmias") or [None])[0]
                        )
                        t = t0 + i * step
                        pts.append((t, rr))
                        rr_all.append(rr)
                        rr_classes.append(beat_class)
                    continue
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                beat_class = _normalize_beat_class(
                    m.get("label")
                    or m.get("template_label")
                    or m.get("arrhythmia")
                    or (m.get("arrhythmias") or [None])[0]
                )
                pts.append((t0, rr_val))
                rr_all.append(rr_val)
                rr_classes.append(beat_class)
                
        start_epoch = None
        if self._metrics_list:
            start_epoch = self._metrics_list[0].get('timestamp')
            
        self.rr_trend_canvas.set_points(pts, start_epoch)
        
        rr_n = [r for r, c in zip(rr_all, rr_classes) if r > 200]
        rr_n_classes = [c for r, c in zip(rr_all, rr_classes) if r > 200]
        if len(rr_n) >= 2:
            rr_x = rr_n[:-1]
            rr_y = rr_n[1:]
            plot_beat_classes = rr_n_classes[:-1]
            
            lo = float(np.percentile(rr_n, 5))
            hi = float(np.percentile(rr_n, 95))
            if hi - lo < 250:
                center = float(np.median(rr_n))
                lo = center - 500.0
                hi = center + 500.0
            lo = max(0.0, lo - 50.0)
            hi = hi + 50.0
            # Build full rr_points list (t, rr, cls) for time-sharing mode
            rr_pts_full = [(t, r, c) for (t, r), c in zip(pts, rr_classes) if r > 200]
            self.lorenz_canvas.set_data(rr_x, rr_y, x_range=(lo, hi), y_range=(lo, hi),
                                        beat_classes=plot_beat_classes, rr_points=rr_pts_full)
        else:
            self.lorenz_canvas.set_data([], [])

    def set_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        """Store the current 12-lead frame data and display if beats are selected."""
        if data is None or data.shape[0] < 1:
            return
        
        # Store the current 12-lead data and timing info
        self._current_replay_data = data.copy() if hasattr(data, 'copy') else data
        self._current_replay_start_sec = start_sec
        self._current_replay_beat_annotations = beat_annotations
        
        # If beats were just selected, update the 12-lead strips with this data
        if getattr(self, '_beats_selected_for_display', False):
            self._beats_selected_for_display = False
            
            selected_beat_class = getattr(self, '_selected_beat_class', 'N')
            
            # Color mapping for arrhythmia types
            color_map = {
                "N":     "#00d215",  # Green
                "S":     "#ff3333",  # Red
                "V":     "#ff9900",  # Orange
                "P":     "#00e5ff",  # Cyan
                "AF":    "#ffd700",  # Gold
                "X":     "#888888",  # Gray
                "Other": "#00d215",  # Green
            }
            qrs_color = color_map.get(selected_beat_class, color_map["N"])
            
            print(f"DEBUG: Displaying 12-lead data for beat class: {selected_beat_class}, QRS color: {qrs_color}")
            
            if data.shape[0] >= 12:
                # Create time axis (assuming 500 Hz sampling rate)
                fs = 500  # Hz
                n_samples = data.shape[1]
                time_axis = np.linspace(0, n_samples / fs, n_samples)
                
                # Detect R-peaks in lead II and create beat annotations for highlighting
                r_peaks_ts = self._detect_qrs_peaks(data, fs=fs, start_sec=start_sec)
                
                # Create beat annotations for each detected R-peak with the anomaly color
                peak_annotations = []
                for peak_ts in r_peaks_ts:
                    peak_annotations.append({
                        'timestamp': peak_ts,
                        'label': selected_beat_class,
                        'color': qrs_color
                    })
                
                print(f"DEBUG: set_replay_frame - Found {len(peak_annotations)} R-peaks, Loading {len(self._lead_names)} leads")
                
                # Display data in each lead strip
                for idx, lead_name in enumerate(self._lead_names):
                    if idx < data.shape[0]:
                        strip = self._lead_strips[lead_name]
                        lead_signal = data[idx]
                        
                        # Set the data with beat annotations for QRS coloring
                        strip.set_data(time_axis, lead_signal, beat_annotations=peak_annotations, start_sec=start_sec)
                        
                        # Keep normal green color for the base waveform (QRS peaks will be colored by annotations)
                        strip.color = "#00d215"
                        strip.pen_width = 0.8
                        strip.update()
                
                print(f"DEBUG: set_replay_frame - Successfully updated all {len(self._lead_names)} lead strips with QRS highlighting")
            else:
                print(f"DEBUG: set_replay_frame - data.shape[0]={data.shape[0]} is less than 12 leads")
        
        # Always update the bottom ECG strip
        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N > 0:
            ch_idx = 1 if data.shape[0] > 1 else 0
            self.ecg_canvas.set_data(x, data[ch_idx].copy())
    
    def _detect_qrs_peaks(self, data, fs=500, start_sec=0.0):
        """Detect QRS/R-peaks in the ECG data using lead II and return timestamps."""
        try:
            from scipy.signal import find_peaks
            
            # Use lead II (index 1) for peak detection - it typically has the most prominent R-peak
            lead_idx = 1 if data.shape[0] > 1 else 0
            signal = np.asarray(data[lead_idx], dtype=float)
            
            # Normalize signal for better peak detection
            sig_mean = np.mean(signal)
            sig_std = np.std(signal)
            if sig_std > 0:
                normalized = (signal - sig_mean) / sig_std
            else:
                normalized = signal - sig_mean
            
            # Find peaks with:
            # - Minimum distance of 300ms (150 samples at 500Hz) between R-peaks
            # - Height threshold above 0.5 standard deviations
            # - Prominence to avoid false positives
            min_distance = int(0.3 * fs)  # 300ms minimum between R-peaks
            peaks, properties = find_peaks(
                normalized,
                distance=min_distance,
                height=0.5,
                prominence=0.3
            )
            
            # Convert peak indices to timestamps
            peak_times = []
            for peak_idx in peaks:
                ts = start_sec + (peak_idx / fs)
                peak_times.append(ts)
            
            print(f"DEBUG: _detect_qrs_peaks - Found {len(peak_times)} peaks in lead {lead_idx}")
            return peak_times
            
        except Exception as e:
            print(f"DEBUG: Error detecting QRS peaks: {e}")
            import traceback
            traceback.print_exc()
            return []

    #     View / display toggle helpers                                         

    def _set_display_mode(self, mode):
        self.lorenz_canvas._set_display(mode)
        self._btn_complete.setChecked(mode == "complete")
        self._btn_timeshare.setChecked(mode == "timesharing")
        # Update button visual state
        active = _style_active_btn() if hasattr(__builtins__, '__name__') else _style_btn()
        try:
            self._btn_complete.setStyleSheet(
                _style_active_btn() if mode == "complete" else _style_btn())
            self._btn_timeshare.setStyleSheet(
                _style_active_btn() if mode == "timesharing" else _style_btn())
        except Exception:
            pass

    def _set_view_mode(self, mode):
        self.lorenz_canvas._set_view(mode)
        self._btn_lorenz.setChecked(mode == "lorenz")
        self._btn_delta.setChecked(mode == "delta_rr")
        try:
            self._btn_lorenz.setStyleSheet(
                _style_active_btn() if mode == "lorenz" else _style_btn())
            self._btn_delta.setStyleSheet(
                _style_active_btn() if mode == "delta_rr" else _style_btn())
        except Exception:
            pass

    def _cycle_pixel_size(self):
        self._pixel_idx = (self._pixel_idx + 1) % len(self._pixel_cycle)
        sz = self._pixel_cycle[self._pixel_idx]
        self.lorenz_canvas._set_pixel(sz)

    #     Signal handlers from LorenzCanvas                                     

    def _on_beats_selected(self, indices):
        """Display 12-lead ECG for selected beats or clear if no beats selected."""
        # If no beats are selected (empty click), clear the waveforms
        if not indices:
            print("DEBUG: Empty space clicked - clearing waveforms")
            self._clear_12lead_displays()
            return
        
        if not self.lorenz_canvas._all_rr_points:
            return
        
        pts = self.lorenz_canvas._all_rr_points
        times = [pts[i][0] for i in indices if i < len(pts)]
        beat_classes = [pts[i][2] for i in indices if i < len(pts)]
        
        if not times:
            return
        
        # Store which beat class is selected for use when data arrives
        self._selected_beat_class = beat_classes[0] if beat_classes else "N"
        self._beats_selected_for_display = True
        
        # Use median time to display ECG window
        target_t = float(np.median(times))
        
        print(f"DEBUG: Beats selected at time {target_t}, beat class: {self._selected_beat_class}")
        
        # Emit seek signal to trigger data broadcast
        self.seek_requested.emit(target_t)
    
    def _clear_12lead_displays(self):
        """Clear all 12-lead waveform displays by showing flatlines instead of disappearing."""
        fs = 500  # Hz
        # Create a short flatline signal (5 seconds at baseline)
        window_sec = 5.0
        n_samples = int(fs * window_sec)
        time_axis = np.linspace(0, window_sec, n_samples)
        
        # Create flatline signal (constant baseline value)
        flatline = np.full(n_samples, 2048.0)  # Baseline value for most leads
        
        for lead_name in self._lead_names:
            strip = self._lead_strips[lead_name]
            
            # Special case for aVR which has different baseline
            if lead_name == "aVR":
                signal = np.full(n_samples, -2048.0)  # Baseline for aVR
            else:
                signal = flatline.copy()
            
            # Set the flatline data
            strip.set_data(time_axis, signal)
            
            # Keep normal green color
            strip.color = "#00d215"
            strip.pen_width = 0.8
            strip.update()

    def _on_beats_reclassified(self, indices, new_class):
        """Reclassification is already applied directly in LorenzCanvas._beat_classes;
        just trigger a replot to refresh colors."""
        self.lorenz_canvas.update()

    def _on_beats_deleted(self, indices):
        """Beats were removed from the canvas data; nothing to persist back to metrics
        in this lightweight implementation -- just refresh the display."""
        self.lorenz_canvas.update()


# -----------------------------------------------------------------------------
# 11. HISTOGRAM PANEL
# -----------------------------------------------------------------------------

class HolterHistogramPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._rank_mode = "rri"
        self._selected_point = None
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title_row = QHBoxLayout()
        self._title_label = QLabel("Histogram - RR Interval Distribution")
        self._title_label.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;border:none;")
        title_row.addWidget(self._title_label, 1)
        self._type_combo = QComboBox()
        self._type_combo.addItems(["RR Interval", "Heart Rate", "RRI Ratio"])
        self._type_combo.setStyleSheet(f"""
            QComboBox {{
                background:{COL_DARK}; color:{COL_GREEN};
                border:1px solid {COL_GREEN_DRK}; padding:4px; border-radius:4px;
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._type_combo.currentTextChanged.connect(self._on_type_changed)
        title_row.addWidget(self._type_combo)
        layout.addLayout(title_row)

        btn_row = QHBoxLayout()
        self._rank_buttons = {}
        for lbl, mode in [("RRI Ranking", "rri"), ("Time Ranking", "time"),
                          ("Prematurity Ranking", "prematurity"), ("Similarity Ranking", "similarity")]:
            btn = QPushButton(lbl)
            btn.setCheckable(True)
            btn.setStyleSheet(_style_btn())
            btn.clicked.connect(lambda _, m=mode: self._set_rank_mode(m))
            self._rank_buttons[mode] = btn
            btn_row.addWidget(btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self._hist_canvas = HistogramCanvas()
        self._hist_canvas.bar_clicked.connect(self._on_bar_clicked)
        layout.addWidget(self._hist_canvas, 1)
        self._selected_info = QLabel("Selected: none")
        self._selected_info.setStyleSheet(
            f"color:{COL_WHITE};font-size:11px;font-weight:bold;border:none;padding:2px 0;"
        )
        layout.addWidget(self._selected_info)


        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        stats_layout = QGridLayout(stats_frame)
        stats_layout.setContentsMargins(10, 6, 10, 6)
        self._hist_stats = {}
        for i, (key, lbl) in enumerate([("nns","NNs"),("mean_nn","Mean NN"),
                                         ("sdnn","SDNN"),("sdann","SDANN"),
                                         ("rmssd","rMSSD"),("pnn50","pNN50"),
                                         ("triidx","TRIIDX"),("sdnnidx","SDNNIDX")]):
            col = i % 4
            row = i // 4
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            v = QLabel("?")
            v.setStyleSheet(f"color:{COL_WHITE};font-size:13px;font-weight:bold;border:none;")
            stats_layout.addWidget(l, row*2, col)
            stats_layout.addWidget(v, row*2+1, col)
            self._hist_stats[key] = v
        layout.addWidget(stats_frame)

        self._strip = ECGStripCanvas(height=60)
        layout.addWidget(self._strip)
        self._set_rank_mode("rri")

    def _set_rank_mode(self, mode: str):
        self._rank_mode = mode
        for key, btn in getattr(self, "_rank_buttons", {}).items():
            active = key == mode
            btn.setChecked(active)
            btn.setStyleSheet(_style_active_btn() if active else _style_btn())
        self._draw()
    
    def _on_type_changed(self, text):
        """Update title label when histogram type changes."""
        if text == "RR Interval":
            self._title_label.setText("Histogram - RR Interval Distribution")
        elif text == "Heart Rate":
            self._title_label.setText("Histogram - Heart Rate Distribution")
        elif text == "RRI Ratio":
            self._title_label.setText("Histogram - RRI Ratio Distribution")
        self._draw()

    def update_from_metrics(self, metrics_list: list):
        self._metrics = list(metrics_list or [])
        self._draw()

    def _build_points(self):
        points = []
        for metric_idx, m in enumerate(self._metrics):
            base_t = float(m.get('t', 0.0) or 0.0)
            label = str((m.get('arrhythmias') or [m.get('label', '')])[0] or '')
            rr_list = [float(v) for v in (m.get('rr_intervals_list') or []) if float(v) > 0]
            if rr_list:
                dur = float(m.get('duration', 0.0) or 0.0)
                step = (dur / max(1, len(rr_list))) if dur > 0 else 0.2
                for sample_idx, rr in enumerate(rr_list):
                    points.append({
                        't': base_t + sample_idx * step,
                        'rr': rr,
                        'metric_idx': metric_idx,
                        'sample_idx': sample_idx,
                        'label': label,
                    })
                continue
            rr_val = float(m.get('rr_ms', 0) or 0)
            if rr_val > 200:
                points.append({
                    't': base_t,
                    'rr': rr_val,
                    'metric_idx': metric_idx,
                    'sample_idx': 0,
                    'label': label,
                })
        return points

    def _draw(self):
        points = self._build_points()
        rr_vals = [p['rr'] for p in points if p['rr'] > 200]
        if not rr_vals:
            self._hist_canvas.set_histogram_data([], mode=self._rank_mode)
            for key in self._hist_stats:
                self._hist_stats[key].setText('?')
            return

        rr_arr = np.array(rr_vals, dtype=float)
        median_rr = float(np.median(rr_arr)) if rr_arr.size else 0.0
        mean_rr = float(np.mean(rr_arr)) if rr_arr.size else 0.0
        
        # Get selected histogram type from dropdown
        hist_type = self._type_combo.currentText()
        
        # Transform data based on histogram type
        if hist_type == "Heart Rate":
            # Convert RR intervals (ms) to Heart Rate (bpm): HR = 60000 / RR_ms
            for p in points:
                if p['rr'] > 0:
                    p['hr'] = 60000.0 / p['rr']
                else:
                    p['hr'] = 0.0
            # Rank by heart rate instead of RR
            if self._rank_mode == 'time':
                ranked = sorted(points, key=lambda x: x['t'])
            elif self._rank_mode == 'prematurity':
                mean_hr = 60000.0 / mean_rr if mean_rr > 0 else 0.0
                ranked = sorted(points, key=lambda x: abs(x.get('hr', 0) - mean_hr), reverse=True)
            elif self._rank_mode == 'similarity':
                median_hr = 60000.0 / median_rr if median_rr > 0 else 0.0
                ranked = sorted(points, key=lambda x: abs(x.get('hr', 0) - median_hr))
            else:
                ranked = sorted(points, key=lambda x: x.get('hr', 0), reverse=True)
        elif hist_type == "RRI Ratio":
            # RRI Ratio = current RR / previous RR
            for i, p in enumerate(points):
                if i > 0:
                    prev_rr = points[i-1]['rr']
                    if prev_rr > 0:
                        p['rri_ratio'] = p['rr'] / prev_rr
                    else:
                        p['rri_ratio'] = 1.0
                else:
                    p['rri_ratio'] = 1.0
            # Rank by RRI ratio
            if self._rank_mode == 'time':
                ranked = sorted(points, key=lambda x: x['t'])
            elif self._rank_mode == 'prematurity':
                ranked = sorted(points, key=lambda x: abs(x.get('rri_ratio', 1.0) - 1.0), reverse=True)
            elif self._rank_mode == 'similarity':
                ranked = sorted(points, key=lambda x: abs(x.get('rri_ratio', 1.0) - 1.0))
            else:
                ranked = sorted(points, key=lambda x: x.get('rri_ratio', 1.0), reverse=True)
        else:
            # Default: RR Interval
            if self._rank_mode == 'time':
                ranked = sorted(points, key=lambda x: x['t'])
            elif self._rank_mode == 'prematurity':
                ranked = sorted(points, key=lambda x: max(0.0, mean_rr - x['rr']), reverse=True)
            elif self._rank_mode == 'similarity':
                ranked = sorted(points, key=lambda x: abs(x['rr'] - median_rr))
            else:
                ranked = sorted(points, key=lambda x: x['rr'], reverse=True)

        self._hist_canvas.set_histogram_data(ranked, mode=self._rank_mode, data_type=hist_type)

        self._hist_stats['nns'].setText(str(len(rr_arr)))
        self._hist_stats['mean_nn'].setText(f"{rr_arr.mean():.0f} ms")
        self._hist_stats['sdnn'].setText(f"{rr_arr.std():.0f} ms")
        self._hist_stats['sdann'].setText('?')
        d = np.diff(rr_arr)
        rmssd = np.sqrt(np.mean(d ** 2)) if len(d) > 0 else 0.0
        self._hist_stats['rmssd'].setText(f"{rmssd:.0f} ms")
        pnn50 = 100.0 * np.sum(np.abs(d) > 50) / len(d) if len(d) > 0 else 0.0
        self._hist_stats['pnn50'].setText(f"{pnn50:.2f}%")
        self._hist_stats['triidx'].setText('?')
        self._hist_stats['sdnnidx'].setText('?')

    def _on_bar_clicked(self, payload: dict):
        if not payload:
            self._selected_point = None
            self._selected_info.setText("Selected: none")
            return
        count = int(payload.get('count', 0) or 0)
        lo, hi = payload.get('range', (0.0, 0.0))
        rr_center = float(payload.get('center_rr', 0.0) or 0.0)
        times = [float(t) for t in (payload.get('times', []) or []) if float(t) >= 0]
        if times:
            target = float(np.median(times))
        else:
            target = float(payload.get('center_t', 0.0) or 0.0)
        self._selected_point = payload
        self._selected_info.setText(
            f"Selected: {count} beats in {lo:.0f}-{hi:.0f} ms range | center {rr_center:.0f} ms"
        )
        if target > 0:
            self.seek_requested.emit(target)

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1:
            return
        N = data.shape[1]
        x = np.linspace(0, N / 500.0, N) if N > 0 else []
        if N > 0:
            self._strip.set_data(x, data[0].copy())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._hist_canvas._selected_index = -1
            self._hist_canvas.update()
            self._on_bar_clicked({})  # Pass empty dict instead of None
        super().mousePressEvent(event)


class HistogramCanvas(QWidget):
    bar_clicked = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._mode = 'rri'
        self._selected_index = -1
        self._bar_rects = []
        self._bin_payloads = []
        self._bin_edges = []
        self._bins = []
        self.setMinimumHeight(260)
        self.setMouseTracking(True)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, rr_values):
        points = [{'t': float(i), 'rr': float(v)} for i, v in enumerate(rr_values or [])]
        self.set_histogram_data(points, mode='rri')

    def set_ranked_data(self, ranked_points, mode: str = 'rri'):
        self.set_histogram_data(ranked_points, mode=mode)

    def set_histogram_data(self, items, mode: str = 'rri', data_type: str = 'RR Interval'):
        self._mode = mode
        self._data_type = data_type  # Store data type for histogram building
        normalized = []
        for item in items or []:
            if isinstance(item, dict):
                normalized.append({
                    't': float(item.get('t', 0.0) or 0.0),
                    'rr': float(item.get('rr', 0.0) or 0.0),
                    'hr': float(item.get('hr', 0.0) or 0.0),
                    'rri_ratio': float(item.get('rri_ratio', 1.0) or 1.0),
                    'label': str(item.get('label', '')),
                })
            else:
                try:
                    t, rr = item
                    normalized.append({'t': float(t), 'rr': float(rr), 'hr': 0.0, 'rri_ratio': 1.0, 'label': ''})
                except Exception:
                    continue
        self._items = normalized
        self.update()

    def _palette(self):
        if self._mode == 'time':
            return QColor('#33D6FF'), QColor('#5AA7FF')
        if self._mode == 'prematurity':
            return QColor('#FF8A1E'), QColor('#FFB347')
        if self._mode == 'similarity':
            return QColor('#8A6CFF'), QColor('#B197FC')
        return QColor('#2F80ED'), QColor('#66A3FF')

    def _build_histogram(self):
        # Extract values based on data type
        data_type = getattr(self, '_data_type', 'RR Interval')
        
        if data_type == 'Heart Rate':
            values = [float(item['hr']) for item in self._items if float(item.get('hr', 0.0) or 0.0) > 0]
        elif data_type == 'RRI Ratio':
            values = [float(item['rri_ratio']) for item in self._items if 0.1 < float(item.get('rri_ratio', 1.0) or 1.0) < 3.0]
        else:
            # Default: RR Interval
            values = [float(item['rr']) for item in self._items if float(item.get('rr', 0.0) or 0.0) > 0]
        
        if not values:
            self._bins = []
            self._bin_edges = []
            self._bin_payloads = []
            self._min_value = 0.0
            self._max_value = 0.0
            return np.array([]), np.array([])
        arr = np.asarray(values, dtype=float)
        
        # Store min/max for later use in paintEvent
        self._min_value = float(np.min(arr))
        self._max_value = float(np.max(arr))
        
        if arr.size < 2:
            self._bins = [(arr[0], arr[0] + 1.0, 1)]
            self._bin_edges = [arr[0], arr[0] + 1.0]
            self._bin_payloads = [list(self._items)]
            return np.array([1]), np.array(self._bin_edges)
        bins = int(min(36, max(12, round(math.sqrt(arr.size) * 2))))
        hist, edges = np.histogram(arr, bins=bins)
        bin_ids = np.clip(np.digitize(arr, edges, right=False) - 1, 0, bins - 1)
        payloads = [[] for _ in range(bins)]
        for item, bidx in zip(self._items, bin_ids):
            payloads[int(bidx)].append(item)
        self._bins = list(hist)
        self._bin_edges = list(edges)
        self._bin_payloads = payloads
        return hist, edges

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            clicked_bar = False
            for idx, rect in enumerate(self._bar_rects):
                if rect.contains(event.pos()) and idx < len(self._bin_payloads):
                    payload = self._payload_for_bin(idx)
                    self._selected_index = idx
                    self.bar_clicked.emit(payload)
                    clicked_bar = True
                    self.update()
                    break
            if not clicked_bar:
                self._selected_index = -1
                # Emit empty dict instead of None to match signal type
                self.bar_clicked.emit({})
                self.update()
        super().mousePressEvent(event)

    def _payload_for_bin(self, idx: int) -> dict:
        points = self._bin_payloads[idx] if idx < len(self._bin_payloads) else []
        if not points:
            left = self._bin_edges[idx] if idx < len(self._bin_edges) else 0.0
            right = self._bin_edges[idx + 1] if idx + 1 < len(self._bin_edges) else left
            return {'index': idx, 'range': (left, right), 'count': 0, 'times': [], 'center_rr': (left + right) / 2.0, 'center_t': 0.0}
        rr_vals = [float(p['rr']) for p in points]
        times = [float(p['t']) for p in points]
        left = self._bin_edges[idx] if idx < len(self._bin_edges) else min(rr_vals)
        right = self._bin_edges[idx + 1] if idx + 1 < len(self._bin_edges) else max(rr_vals)
        return {
            'index': idx,
            'range': (float(left), float(right)),
            'count': len(points),
            'times': times,
            'center_rr': float(np.median(rr_vals)),
            'center_t': float(np.median(times)),
        }

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()

        left, top, right, bottom = 46, 22, 18, 28
        plot_w = max(1, w - left - right)
        plot_h = max(1, h - top - bottom)

        hist, edges = self._build_histogram()
        if hist.size == 0:
            painter.setPen(QPen(QColor(COL_GREEN_DRK)))
            painter.drawText(self.rect(), Qt.AlignCenter, 'No data')
            return

        counts = hist.astype(int).tolist()
        max_count = max(counts) if counts else 1
        if max_count <= 0:
            max_count = 1

        # Use stored min/max values from _build_histogram
        min_rr = getattr(self, '_min_value', 0.0)
        max_rr = getattr(self, '_max_value', 0.0)

        painter.setPen(QPen(QColor('#22324B'), 1))
        for frac in (0.25, 0.5, 0.75):
            y = int(top + frac * plot_h)
            painter.drawLine(left, y, left + plot_w, y)
        for frac in (0.25, 0.5, 0.75):
            x = int(left + frac * plot_w)
            painter.drawLine(x, top, x, top + plot_h)

        x_label_left = 'low RR'
        x_label_right = 'high RR'
        if self._mode == 'time':
            x_label_left = 'early'
            x_label_right = 'late'
        elif self._mode == 'prematurity':
            x_label_left = 'less premature'
            x_label_right = 'more premature'
        elif self._mode == 'similarity':
            x_label_left = 'less similar'
            x_label_right = 'more similar'

        palette, highlight = self._palette()
        self._bar_rects = []
        bin_count = len(counts)
        gap = 2
        bar_w = max(2, int((plot_w - gap * (bin_count - 1)) / max(1, bin_count)))
        x0 = left
        for idx, count in enumerate(counts):
            bar_h = int((count / max_count) * max(10, plot_h - 8))
            x = int(x0 + idx * (bar_w + gap))
            y = int(top + plot_h - bar_h)
            rect = QRect(x, y, bar_w, bar_h)
            self._bar_rects.append(rect)
            selected = idx == self._selected_index
            color = highlight if selected else palette
            alpha = 255 if selected else 180
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), alpha)))
            painter.setPen(QPen(QColor(color.lighter(130)), 1))
            painter.drawRect(rect)
            if selected:
                painter.setPen(QPen(QColor('#F5D76E'), 2))
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
            if count > 0:
                painter.setPen(QPen(QColor(COL_WHITE)))
                painter.drawText(rect.adjusted(0, -18, 0, -2), Qt.AlignCenter, str(count))

        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        painter.drawText(left, 14, x_label_left)
        painter.drawText(w - 118, 14, x_label_right)
        
        # Get data type and display appropriate labels
        data_type = getattr(self, '_data_type', 'RR Interval')
        
        if data_type == 'Heart Rate':
            painter.drawText(left, h - 5, f'{min_rr:.0f} bpm')
            painter.drawText(w - 70, h - 5, f'{max_rr:.0f} bpm')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'Heart Rate Range {min_rr:.0f}-{max_rr:.0f} bpm')
        elif data_type == 'RRI Ratio':
            painter.drawText(left, h - 5, f'{min_rr:.2f}')
            painter.drawText(w - 70, h - 5, f'{max_rr:.2f}')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'RRI Ratio Range {min_rr:.2f}-{max_rr:.2f}')
        else:
            # Default: RR Interval
            painter.drawText(left, h - 5, f'{min_rr:.0f} ms')
            painter.drawText(w - 70, h - 5, f'{max_rr:.0f} ms')
            painter.setPen(QPen(QColor(UI_MUTED)))
            painter.drawText(left + 110, 14, f'RR Interval Range {min_rr:.0f}-{max_rr:.0f} ms')


# -----------------------------------------------------------------------------        
# 12. AF ANALYSIS PANEL
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# 13. ST TENDENCY PANEL
# -----------------------------------------------------------------------------

class HolterSTPanel(QWidget):
    def __init__(self, parent=None, replay_engine=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._metrics = []
        self._current_tendency_mode = "ST"
        self._replay_engine = replay_engine
        self._current_view_mode = "ST"  # can be "ST", "T", or "TEND_CHART"
        
        # Tend chart variables
        self._tend_current_tendency_mode = "ST"
        self._tend_scan_start_index = 0
        self._tend_is_scanning = False
        self._tend_current_metric_index = 0
        self._tend_auto_update = False
        self._tend_adjust_k_by_hr = True
        self._tend_st_threshold_mv = 0.1
        self._tend_i_offset_ms = -120
        self._tend_j_offset_ms = 80
        self._tend_k_offset_ms = 120
        self._tend_default_i_offset_ms = -120
        self._tend_default_j_offset_ms = 80
        self._tend_default_k_offset_ms = 120
        self._tend_scan_results = []
        
        self._build_ui()
        
    def set_replay_engine(self, replay_engine):
        self._replay_engine = replay_engine

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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(8)

        # Mode row (always visible)
        mode_row = QHBoxLayout()
        self._tend_chart_btn = QPushButton("Tend chart")
        self._tend_chart_btn.setStyleSheet(_style_btn())
        self._st_btn = QPushButton("ST")
        self._st_btn.setStyleSheet(_style_active_btn())
        self._st_btn.setFixedWidth(50)
        self._t_btn = QPushButton("T")
        self._t_btn.setStyleSheet(_style_btn())
        self._t_btn.setFixedWidth(40)
        
        self._tend_chart_btn.clicked.connect(lambda: self._set_view_mode("TEND_CHART"))
        self._st_btn.clicked.connect(lambda: self._set_view_mode("ST"))
        self._t_btn.clicked.connect(lambda: self._set_view_mode("T"))

        mode_row.addWidget(self._tend_chart_btn)
        mode_row.addWidget(self._st_btn)
        mode_row.addWidget(self._t_btn)
        mode_row.addStretch()
        main_layout.addLayout(mode_row)

        # Main content stacked widget
        self._main_stack = QStackedWidget()
        
        # --- Normal view (ST/T with left ECG strips) ---
        normal_view_widget = QWidget()
        normal_layout = QHBoxLayout(normal_view_widget)
        normal_layout.setContentsMargins(0, 0, 0, 0)
        normal_layout.setSpacing(8)

        # Left: ECG strip + controls
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background:transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(4, 4, 4, 4)
        scroll_layout.setSpacing(4)
        
        self._ch_strips = []
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for i in range(12):
            lbl = QLabel(f"Lead {lead_names[i]}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
            scroll_layout.addWidget(lbl)
            
            # Disable vertical lines for ST Tendency panel
            strip = ECGStripCanvas(height=80, show_vertical_lines=False)
            strip.lead_name = lead_names[i]
            scroll_layout.addWidget(strip)
            self._ch_strips.append(strip)
            
        scroll_area.setWidget(scroll_content)
        left_layout.addWidget(scroll_area, 1)

        # Mini overview
        self._mini_strip = ECGStripCanvas(height=60, color="#00AA00")
        left_layout.addWidget(self._mini_strip)

        nav_row = QHBoxLayout()
        for lbl in ["ReScan", "Next Event", "Remove All", "Remove", "Reset"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            btn.setFixedHeight(30)
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)
        normal_layout.addWidget(left, 3)

        # Right: ST tendency charts + conclusion
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)
        
        self._st_title = QLabel("ST tendency(mV)")
        self._st_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        right_layout.addWidget(self._st_title)

        self._st_canvases = []
        st_scroll = QScrollArea()
        st_scroll.setWidgetResizable(True)
        st_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        st_scroll_content = QWidget()
        st_scroll_content.setStyleSheet("background:transparent;")
        st_scroll_layout = QVBoxLayout(st_scroll_content)
        st_scroll_layout.setContentsMargins(4, 4, 4, 4)
        st_scroll_layout.setSpacing(4)
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        for name in lead_names:
            ch_lbl = QLabel(f"Lead {name}")
            ch_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:10px;border:none;")
            st_scroll_layout.addWidget(ch_lbl)
            canvas = STCanvas(height=70)
            st_scroll_layout.addWidget(canvas)
            self._st_canvases.append(canvas)
        st_scroll.setWidget(st_scroll_content)
        right_layout.addWidget(st_scroll, 1)

        conclusion_frame = QFrame()
        conclusion_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        cf_layout = QVBoxLayout(conclusion_frame)
        cf_layout.setContentsMargins(8, 6, 8, 6)
        cf_lbl = QLabel("Conclusion")
        cf_lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;")
        cf_layout.addWidget(cf_lbl)
        self._conclusion_edit = QTextEdit()
        self._conclusion_edit.setFixedHeight(60)
        self._conclusion_edit.setStyleSheet(f"QTextEdit{{background:{COL_DARK};color:{COL_GREEN};"
                                             f"border:none;font-size:11px;padding:4px;}}")
        cf_layout.addWidget(self._conclusion_edit)
        save_row = QHBoxLayout()
        for lbl in ["Save as template", "Quote templates"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            save_row.addWidget(btn)
        cf_layout.addLayout(save_row)
        right_layout.addWidget(conclusion_frame)
        normal_layout.addWidget(right, 2)
        
        # --- Tend chart view widget ---
        tend_widget = QWidget()
        tend_layout = QVBoxLayout(tend_widget)
        tend_layout.setContentsMargins(0,0,0,0)
        tend_layout.setSpacing(4)
        
        # Tend chart toolbar
        toolbar = QHBoxLayout()
        
        self._tend_start_btn = QPushButton("Start")
        self._tend_start_btn.setStyleSheet(_style_btn())
        self._tend_start_btn.setFixedHeight(32)
        self._tend_start_btn.clicked.connect(self._tend_on_start_scan)
        toolbar.addWidget(self._tend_start_btn)
        
        self._tend_confirm_btn = QPushButton("Confirm")
        self._tend_confirm_btn.setStyleSheet(_style_btn())
        self._tend_confirm_btn.setFixedHeight(32)
        self._tend_confirm_btn.clicked.connect(self._tend_on_confirm_scan)
        toolbar.addWidget(self._tend_confirm_btn)
        
        self._tend_default_btn = QPushButton("Default")
        self._tend_default_btn.setStyleSheet(_style_btn())
        self._tend_default_btn.setFixedHeight(32)
        self._tend_default_btn.clicked.connect(self._tend_on_reset_defaults)
        toolbar.addWidget(self._tend_default_btn)
        
        self._tend_auto_update_btn = QPushButton("Update automatically")
        self._tend_auto_update_btn.setCheckable(True)
        self._tend_auto_update_btn.setStyleSheet(_style_btn())
        self._tend_auto_update_btn.setFixedHeight(32)
        self._tend_auto_update_btn.clicked.connect(self._tend_on_toggle_auto_update)
        toolbar.addWidget(self._tend_auto_update_btn)
        
        self._tend_adjust_k_btn = QPushButton("Adjust K by HR")
        self._tend_adjust_k_btn.setCheckable(True)
        self._tend_adjust_k_btn.setChecked(True)
        self._tend_adjust_k_btn.setStyleSheet(_style_btn())
        self._tend_adjust_k_btn.setFixedHeight(32)
        self._tend_adjust_k_btn.clicked.connect(self._tend_on_toggle_adjust_k)
        toolbar.addWidget(self._tend_adjust_k_btn)
        
        threshold_lbl = QLabel("ST Threshold (mV):")
        threshold_lbl.setStyleSheet(f"color:{COL_GREEN};")
        toolbar.addWidget(threshold_lbl)
        self._tend_threshold_spin = QDoubleSpinBox()
        self._tend_threshold_spin.setRange(0.01, 1.0)
        self._tend_threshold_spin.setSingleStep(0.01)
        self._tend_threshold_spin.setValue(self._tend_st_threshold_mv)
        self._tend_threshold_spin.valueChanged.connect(self._tend_on_threshold_changed)
        self._tend_threshold_spin.setFixedHeight(32)
        self._tend_threshold_spin.setStyleSheet(f"QDoubleSpinBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        toolbar.addWidget(self._tend_threshold_spin)
        
        toolbar.addStretch()
        tend_layout.addLayout(toolbar)
        
        # Tend chart main content
        main_content = QHBoxLayout()
        
        # Left: HR graph + ECG strip
        left_panel = QWidget()
        left_panel_layout = QVBoxLayout(left_panel)
        
        # HR graph
        hr_frame = QFrame()
        hr_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;")
        hr_layout = QVBoxLayout(hr_frame)
        self._tend_hr_title = QLabel("Entire HR Graph")
        self._tend_hr_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        hr_layout.addWidget(self._tend_hr_title)
        self._tend_hr_canvas = HRTrendCanvas()
        self._tend_hr_canvas.clicked.connect(self._tend_on_hr_canvas_clicked)
        hr_layout.addWidget(self._tend_hr_canvas)
        left_panel_layout.addWidget(hr_frame, 1)
        
        # ECG strip with I/J/K markers
        ecg_frame = QFrame()
        ecg_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;")
        ecg_layout = QVBoxLayout(ecg_frame)
        self._tend_ecg_title = QLabel("ECG Strip (drag I/J/K markers)")
        self._tend_ecg_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        ecg_layout.addWidget(self._tend_ecg_title)
        self._tend_ecg_strip = STTMarkerCanvas(height=200)
        self._tend_ecg_strip.markerMoved.connect(self._tend_on_marker_moved)
        ecg_layout.addWidget(self._tend_ecg_strip)
        left_panel_layout.addWidget(ecg_frame, 1)
        
        main_content.addWidget(left_panel, 1)
        
        # Right: Tendency charts
        right_panel = QWidget()
        right_panel_layout = QVBoxLayout(right_panel)
        
        mode_row_tend = QHBoxLayout()
        self._tend_st_mode_btn = QPushButton("ST")
        self._tend_st_mode_btn.setStyleSheet(_style_active_btn())
        self._tend_t_mode_btn = QPushButton("T")
        self._tend_t_mode_btn.setStyleSheet(_style_btn())
        for btn in [self._tend_st_mode_btn, self._tend_t_mode_btn]:
            btn.setFixedHeight(32)
            mode_row_tend.addWidget(btn)
        mode_row_tend.addStretch()
        right_panel_layout.addLayout(mode_row_tend)
        
        self._tend_tendency_title = QLabel("ST Trends")
        self._tend_tendency_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        right_panel_layout.addWidget(self._tend_tendency_title)
        
        self._tend_tendency_scroll = QScrollArea()
        self._tend_tendency_scroll.setWidgetResizable(True)
        self._tend_tendency_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        
        self._tend_tendency_content = QWidget()
        self._tend_tendency_layout = QVBoxLayout(self._tend_tendency_content)
        self._tend_tendency_content.setStyleSheet("background:transparent;")
        self._tend_tendency_layout.setContentsMargins(4,4,4,4)
        self._tend_tendency_layout.setSpacing(4)
        
        lead_names_tend = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._tend_tendency_canvases = []
        for name in lead_names_tend:
            lbl = QLabel(f"Lead {name}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:10px;border:none;")
            self._tend_tendency_layout.addWidget(lbl)
            canvas = TendencyCanvas(height=70)
            canvas.set_threshold(self._tend_st_threshold_mv)
            canvas.clicked.connect(lambda idx=len(self._tend_tendency_canvases), c=canvas: self._tend_on_tendency_clicked(idx, c))
            self._tend_tendency_layout.addWidget(canvas)
            self._tend_tendency_canvases.append(canvas)
        
        self._tend_tendency_scroll.setWidget(self._tend_tendency_content)
        right_panel_layout.addWidget(self._tend_tendency_scroll, 1)
        
        main_content.addWidget(right_panel, 2)
        tend_layout.addLayout(main_content, 1)
        
        # Connect tend chart mode buttons
        self._tend_st_mode_btn.clicked.connect(lambda: self._tend_set_mode("ST"))
        self._tend_t_mode_btn.clicked.connect(lambda: self._tend_set_mode("T"))
        
        # Add widgets to main stack
        self._main_stack.addWidget(normal_view_widget)
        self._main_stack.addWidget(tend_widget)
        self._main_stack.setCurrentIndex(0)
        
        main_layout.addWidget(self._main_stack, 1)

    def _set_view_mode(self, mode):
        self._current_view_mode = mode
        
        # Update button styles
        self._tend_chart_btn.setStyleSheet(_style_btn())
        self._st_btn.setStyleSheet(_style_btn())
        self._t_btn.setStyleSheet(_style_btn())
        
        if mode == "TEND_CHART":
            self._tend_chart_btn.setStyleSheet(_style_active_btn())
            self._main_stack.setCurrentIndex(1)
            self._tend_update_hr_graph()
            self._tend_update_tendency_graphs()
            if self._replay_engine:
                self._tend_update_ecg_strip()
        elif mode == "ST":
            self._st_btn.setStyleSheet(_style_active_btn())
            self._main_stack.setCurrentIndex(0)
            self._set_tendency_mode("ST")
        elif mode == "T":
            self._t_btn.setStyleSheet(_style_active_btn())
            self._main_stack.setCurrentIndex(0)
            self._set_tendency_mode("T")

    def _set_tendency_mode(self, mode: str):
        self._current_tendency_mode = mode
        if mode == "ST":
            self._st_title.setText("ST tendency(mV)")
        else:
            self._st_title.setText("T tendency(mV)")
        self._refresh_plot()

    def set_replay_frame(self, data):
        if self._current_view_mode in ["ST", "T"]:
            if data is None or data.shape[0] < 1: return
            N = data.shape[1]
            fs = 500.0
            n_samples = min(N, int(10 * fs))
            x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
            if n_samples > 0:
                lead2_idx = 1 if data.shape[0] > 1 else 0
                self._mini_strip.set_data(x, data[lead2_idx, :n_samples].copy())
                for i, ts in enumerate(self._ch_strips):
                    if i < data.shape[0]:
                        ts.set_data(x, data[i, :n_samples].copy())
        elif self._current_view_mode == "TEND_CHART":
            self._tend_update_ecg_strip()

    def update_from_metrics(self, metrics_list):
        self._metrics = metrics_list
        self._refresh_plot()
        if self._current_view_mode == "TEND_CHART":
            self._tend_update_hr_graph()
            self._tend_update_tendency_graphs()

    def _refresh_plot(self):
        if self._current_tendency_mode == "ST":
            vals = [m.get('st_mv', 0.0) for m in self._metrics]
        else:
            vals = [m.get('t_mv', 0.0) for m in self._metrics]
        for canvas in self._st_canvases:
            canvas.set_data(vals)

    def update_st_ecg_display(self, data):
        """Update ECG display for ST/T normal view mode."""
        if data is None or data.shape[0] < 1: 
            return
        
        N = data.shape[1]
        fs = 500.0
        n_samples = min(N, int(10 * fs))
        x = np.linspace(0, n_samples/fs, n_samples) if n_samples > 0 else []
        
        if n_samples > 0:
            lead2_idx = 1 if data.shape[0] > 1 else 0
            self._mini_strip.set_data(x, data[lead2_idx, :n_samples].copy())
            for i, ts in enumerate(self._ch_strips):
                if i < data.shape[0]:
                    ts.set_data(x, data[i, :n_samples].copy())

    def set_replay_frame(self, data):
        """Main entry point for updating ECG displays based on current view mode."""
        if data is None or data.shape[0] < 1: 
            return
        
        if self._current_view_mode in ["ST", "T"]:
            # Update left-side ECG strips (all 12 leads) for ST/T mode
            self.update_st_ecg_display(data)
        
        elif self._current_view_mode == "TEND_CHART":
            # Update tendency chart ECG strip
            self._tend_update_ecg_strip()
            
    # --- Tend chart methods ---
    def _tend_set_mode(self, mode):
        self._tend_current_tendency_mode = mode
        if mode == "ST":
            self._tend_st_mode_btn.setStyleSheet(_style_active_btn())
            self._tend_t_mode_btn.setStyleSheet(_style_btn())
            self._tend_tendency_title.setText("ST Trends")
        else:
            self._tend_t_mode_btn.setStyleSheet(_style_active_btn())
            self._tend_st_mode_btn.setStyleSheet(_style_btn())
            self._tend_tendency_title.setText("T Trends")
        self._tend_update_tendency_graphs()
        
    def _tend_update_hr_graph(self):
        hr_vals = [m.get('hr_mean', 0) for m in self._metrics]
        self._tend_hr_canvas.set_data(hr_vals)
        self._tend_hr_canvas.set_current_index(self._tend_current_metric_index)
        
    def _tend_update_tendency_graphs(self):
        if self._tend_scan_results:
            st_vals = [res['st_mv'] for res in self._tend_scan_results]
            t_vals = [res['t_mv'] for res in self._tend_scan_results]
        else:
            st_vals = [m.get('st_mv', 0.0) for m in self._metrics]
            t_vals = [m.get('t_mv', 0.0) for m in self._metrics]
            
        for i, canvas in enumerate(self._tend_tendency_canvases):
            if self._tend_current_tendency_mode == "ST":
                canvas.set_tendency_mode("ST")
                canvas.set_data(st_vals)
            else:
                canvas.set_tendency_mode("T")
                canvas.set_data(t_vals)
            canvas.set_current_index(self._tend_current_metric_index)
                
    def _tend_update_ecg_strip(self):
        if self._replay_engine:
            data = self._replay_engine.get_all_leads_data(window_sec=3.0)
            if data is not None:
                self._tend_ecg_strip.set_data(data, self._replay_engine.fs)
                r_peak_idx = data.shape[1] // 2
                fs = self._replay_engine.fs
                self._tend_ecg_strip.set_marker_positions(
                    i_pos=r_peak_idx + int(self._tend_i_offset_ms * fs / 1000),
                    j_pos=r_peak_idx + int(self._tend_j_offset_ms * fs / 1000),
                    k_pos=r_peak_idx + int(self._tend_get_k_offset_ms() * fs / 1000),
                    t_pos=r_peak_idx + int(250 * fs / 1000)
                )
    
    def _tend_get_k_offset_ms(self):
        if self._tend_adjust_k_by_hr and self._metrics:
            current_metric = self._metrics[self._tend_current_metric_index]
            hr = current_metric.get('hr_mean', 75)
            if hr > 100:
                return self._tend_k_offset_ms - int((hr - 100) * 0.5)
            elif hr < 60:
                return self._tend_k_offset_ms + int((60 - hr) * 0.5)
        return self._tend_k_offset_ms
        
    def _tend_on_start_scan(self):
        self._tend_is_scanning = True
        self._tend_scan_results = []
        if self._replay_engine:
            self._tend_scan_thread = threading.Thread(target=self._tend_run_scan, daemon=True)
            self._tend_scan_thread.start()
            
    def _tend_run_scan(self):
        fs = self._replay_engine.fs
        for metric in self._metrics:
            t_sec = metric.get('t', 0.0)
            self._replay_engine.seek(t_sec)
            data = self._replay_engine.get_all_leads_data(window_sec=2.0)
            
            lead_results = {}
            for i in range(min(12, data.shape[0])):
                lead_data = data[i]
                st_mv, t_mv = self._tend_analyze_lead(lead_data, fs)
                lead_results[i] = {'st_mv': st_mv, 't_mv': t_mv}
                
            self._tend_scan_results.append({
                't_sec': t_sec,
                'st_mv': lead_results.get(1, {}).get('st_mv', 0.0),
                't_mv': lead_results.get(1, {}).get('t_mv', 0.0),
                'lead_results': lead_results
            })
            
            if self._tend_auto_update:
                QMetaObject.invokeMethod(self, "_tend_update_tendency_graphs", Qt.QueuedConnection)
                time.sleep(0.01)
                
        self._tend_is_scanning = False
        
    def _tend_analyze_lead(self, lead_data: np.ndarray, fs: int):
        from scipy.signal import butter, filtfilt, find_peaks
        
        if len(lead_data) < fs * 0.5:
            return 0.0, 0.0
            
        b, a = butter(2, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype='band')
        filtered = filtfilt(b, a, lead_data)
        
        peaks, _ = find_peaks(np.abs(filtered), distance=int(0.2 * fs))
        if len(peaks) == 0:
            return 0.0, 0.0
            
        r_peak = peaks[len(peaks) // 2]
        
        fs = fs
        i_idx = max(0, r_peak + int(self._tend_i_offset_ms * fs / 1000))
        j_idx = min(len(lead_data) - 1, r_peak + int(self._tend_j_offset_ms * fs / 1000))
        k_idx = min(len(lead_data) - 1, r_peak + int(self._tend_get_k_offset_ms() * fs / 1000))
        
        # T wave: measure from J point (end of QRS) to T peak
        t_start = min(len(lead_data) - 1, r_peak + int(150 * fs / 1000))
        t_end = min(len(lead_data), r_peak + int(400 * fs / 1000))
        
        # ST baseline: measure at I point (before J)
        baseline = lead_data[i_idx] if 0 <= i_idx < len(lead_data) else 0.0
        
        # ST measurement: ST deflection at K point relative to baseline
        st_value = lead_data[k_idx] if 0 <= k_idx < len(lead_data) else 0.0
        st = (st_value - baseline)
        
        # T wave: measure from J point baseline (more accurate)
        j_baseline = lead_data[j_idx] if 0 <= j_idx < len(lead_data) else baseline
        t_amp = 0.0
        if t_start < t_end:
            t_seg = lead_data[t_start:t_end]
            if len(t_seg) > 0:
                peak_idx = np.argmax(np.abs(t_seg - j_baseline))
                t_amp = t_seg[peak_idx] - j_baseline
        
        adc_scale = 1.0 / 200.0
        return st * adc_scale, t_amp * adc_scale
        
    def _tend_on_confirm_scan(self):
        if not self._tend_scan_results:
            return
        _show_message_box(self, QMessageBox.Information, "Scan Complete", f"Processed {len(self._tend_scan_results)} segments")
        
    def _tend_on_reset_defaults(self):
        self._tend_i_offset_ms = self._tend_default_i_offset_ms
        self._tend_j_offset_ms = self._tend_default_j_offset_ms
        self._tend_k_offset_ms = self._tend_default_k_offset_ms
        self._tend_update_ecg_strip()
        
    def _tend_on_toggle_auto_update(self, checked):
        self._tend_auto_update = checked
        
    def _tend_on_toggle_adjust_k(self, checked):
        self._tend_adjust_k_by_hr = checked
        self._tend_update_ecg_strip()
        
    def _tend_on_threshold_changed(self, value):
        self._tend_st_threshold_mv = value
        for canvas in self._tend_tendency_canvases:
            canvas.set_threshold(value)
        
    def _tend_on_hr_canvas_clicked(self, index):
        self._tend_current_metric_index = index
        self._tend_update_hr_graph()
        self._tend_update_tendency_graphs()
        if self._replay_engine:
            t_sec = self._metrics[index].get('t', 0.0)
            self._replay_engine.seek(t_sec)
            self._tend_update_ecg_strip()
            
    def _tend_on_tendency_clicked(self, lead_idx, canvas):
        idx = canvas.get_clicked_index()
        if idx is not None:
            self._tend_on_hr_canvas_clicked(idx)
            
    def _tend_on_marker_moved(self, marker_name, pos_samples):
        if not self._replay_engine:
            return
            
        fs = self._replay_engine.fs
        center_idx = self._tend_ecg_strip._data.shape[1] // 2 if len(self._tend_ecg_strip._data.shape) >1 else len(self._tend_ecg_strip._data)//2
        delta_ms = int((pos_samples - center_idx) * 1000 / fs)
        
        if marker_name == 'I':
            self._tend_i_offset_ms = delta_ms
        elif marker_name == 'J':
            self._tend_j_offset_ms = delta_ms
        elif marker_name == 'K':
            self._tend_k_offset_ms = delta_ms



class STCanvas(QWidget):
    def __init__(self, parent=None, height: int = 70):
        super().__init__(parent)
        self._data = []
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")

    def set_data(self, vals):
        self._data = vals
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        # Zero line
        pen = QPen(QColor(COL_GREEN_DRK))
        pen.setWidth(1)
        painter.setPen(pen)
        mid = h // 2
        painter.drawLine(0, mid, w, mid)
        if not self._data: return
        d = np.array(self._data)
        mn, mx = min(d.min(), -0.1), max(d.max(), 0.1)
        rng = max(mx - mn, 0.2)
        pen = QPen(QColor(COL_GREEN))
        pen.setWidth(2)
        painter.setPen(pen)
        n = len(d)
        x_scale = w / max(n - 1, 1)
        for i in range(1, n):
            x1 = int((i-1) * x_scale)
            y1 = int(h - 5 - (d[i-1] - mn) / rng * (h - 10))
            x2 = int(i * x_scale)
            y2 = int(h - 5 - (d[i] - mn) / rng * (h - 10))
            painter.drawLine(x1, y1, x2, y2)
        # mV label
        painter.setPen(QPen(QColor(COL_GREEN_DRK)))
        if len(d) > 0:
            painter.drawText(w - 70, 14, f"{d[min(len(d)//2,len(d)-1)]:.3f}mV")


# -----------------------------------------------------------------------------
# 14. TEND CHART DIALOG (ST/T TREND SCAN)
# -----------------------------------------------------------------------------
class TendChartDialog(QDialog):
    currentIndexChanged = pyqtSignal(int)
    def __init__(self, parent=None, metrics_list=None, replay_engine=None):
        super().__init__(parent)
        self.setWindowTitle("Tend Chart - ST/T Trend Scan")
        self.setStyleSheet(f"background:{COL_BG};")
        self.resize(1400, 900)
        
        self._metrics = metrics_list or []
        self._replay_engine = replay_engine
        self._current_tendency_mode = "ST"
        self._scan_start_index = 0
        self._is_scanning = False
        self._current_metric_index = 0
        self._auto_update = False
        self._adjust_k_by_hr = True
        self._st_threshold_mv = 0.1
        
        # I/J/K point offsets (in ms from R-peak)
        self._i_offset_ms = -120  # PR segment baseline
        self._j_offset_ms = 80    # J-point (end of QRS)
        self._k_offset_ms = 120   # ST measurement point
        
        self._default_i_offset_ms = -120
        self._default_j_offset_ms = 80
        self._default_k_offset_ms = 120
        
        self._scan_results = []
        
        self._build_ui()
        self._update_hr_graph()
        self._update_tendency_graphs()
        
        if self._replay_engine:
            self._update_ecg_strip()
        
    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # Toolbar row
        toolbar = QHBoxLayout()
        
        self._start_btn = QPushButton("Start")
        self._start_btn.setStyleSheet(_style_btn())
        self._start_btn.setFixedHeight(32)
        self._start_btn.clicked.connect(self._on_start_scan)
        toolbar.addWidget(self._start_btn)
        
        self._confirm_btn = QPushButton("Confirm")
        self._confirm_btn.setStyleSheet(_style_btn())
        self._confirm_btn.setFixedHeight(32)
        self._confirm_btn.clicked.connect(self._on_confirm_scan)
        toolbar.addWidget(self._confirm_btn)
        
        self._default_btn = QPushButton("Default")
        self._default_btn.setStyleSheet(_style_btn())
        self._default_btn.setFixedHeight(32)
        self._default_btn.clicked.connect(self._on_reset_defaults)
        toolbar.addWidget(self._default_btn)
        
        self._auto_update_btn = QPushButton("Update automatically")
        self._auto_update_btn.setCheckable(True)
        self._auto_update_btn.setStyleSheet(_style_btn())
        self._auto_update_btn.setFixedHeight(32)
        self._auto_update_btn.clicked.connect(self._on_toggle_auto_update)
        toolbar.addWidget(self._auto_update_btn)
        
        self._adjust_k_btn = QPushButton("Adjust K by HR")
        self._adjust_k_btn.setCheckable(True)
        self._adjust_k_btn.setChecked(True)
        self._adjust_k_btn.setStyleSheet(_style_btn())
        self._adjust_k_btn.setFixedHeight(32)
        self._adjust_k_btn.clicked.connect(self._on_toggle_adjust_k)
        toolbar.addWidget(self._adjust_k_btn)
        
        # ST threshold setting
        threshold_lbl = QLabel("ST Threshold (mV):")
        threshold_lbl.setStyleSheet(f"color:{COL_GREEN};")
        toolbar.addWidget(threshold_lbl)
        self._threshold_spin = QDoubleSpinBox()
        self._threshold_spin.setRange(0.01, 1.0)
        self._threshold_spin.setSingleStep(0.01)
        self._threshold_spin.setValue(self._st_threshold_mv)
        self._threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self._threshold_spin.setFixedHeight(32)
        self._threshold_spin.setStyleSheet(f"QDoubleSpinBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        toolbar.addWidget(self._threshold_spin)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)
        
        # Main content (split into left and right
        main_content = QHBoxLayout()
        
        # Left: HR graph + ECG strip
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # HR graph
        hr_frame = QFrame()
        hr_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;")
        hr_layout = QVBoxLayout(hr_frame)
        self._hr_title = QLabel("Entire HR Graph")
        self._hr_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        hr_layout.addWidget(self._hr_title)
        self._hr_canvas = HRTrendCanvas()
        self._hr_canvas.clicked.connect(self._on_hr_canvas_clicked)
        hr_layout.addWidget(self._hr_canvas)
        left_layout.addWidget(hr_frame, 1)
        
        # ECG strip with I/J/K markers
        ecg_frame = QFrame()
        ecg_frame.setStyleSheet(f"background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;")
        ecg_layout = QVBoxLayout(ecg_frame)
        self._ecg_title = QLabel("ECG Strip (drag I/J/K markers)")
        self._ecg_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        ecg_layout.addWidget(self._ecg_title)
        self._ecg_strip = STTMarkerCanvas(height=200)
        self._ecg_strip.markerMoved.connect(self._on_marker_moved)
        ecg_layout.addWidget(self._ecg_strip)
        left_layout.addWidget(ecg_frame, 1)
        
        main_content.addWidget(left_panel, 1)
        
        # Right: Tendency charts
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Mode buttons
        mode_row = QHBoxLayout()
        self._st_mode_btn = QPushButton("ST")
        self._st_mode_btn.setStyleSheet(_style_active_btn())
        self._t_mode_btn = QPushButton("T")
        self._t_mode_btn.setStyleSheet(_style_btn())
        for btn in [self._st_mode_btn, self._t_mode_btn]:
            btn.setFixedHeight(32)
            mode_row.addWidget(btn)
        mode_row.addStretch()
        right_layout.addLayout(mode_row)
        
        # Tendency charts scroll
        self._tendency_title = QLabel("ST Trends")
        self._tendency_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        right_layout.addWidget(self._tendency_title)
        
        self._tendency_scroll = QScrollArea()
        self._tendency_scroll.setWidgetResizable(True)
        self._tendency_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        
        self._tendency_content = QWidget()
        self._tendency_layout = QVBoxLayout(self._tendency_content)
        self._tendency_content.setStyleSheet("background:transparent;")
        
        self._tendency_layout.setContentsMargins(4, 4, 4, 4)
        self._tendency_layout.setSpacing(4)
        
        lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self._tendency_canvases = []
        for name in lead_names:
            lbl = QLabel(f"Lead {name}")
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:10px;border:none;")
            self._tendency_layout.addWidget(lbl)
            canvas = TendencyCanvas(height=70)
            canvas.set_threshold(self._st_threshold_mv)
            canvas.clicked.connect(lambda idx=len(self._tendency_canvases), c=canvas: self._on_tendency_clicked(idx, c))
            self._tendency_layout.addWidget(canvas)
            self._tendency_canvases.append(canvas)
        
        self._tendency_scroll.setWidget(self._tendency_content)
        right_layout.addWidget(self._tendency_scroll, 1)
        
        main_content.addWidget(right_panel, 2)
        
        layout.addLayout(main_content, 1)
        
        # Connect mode buttons
        self._st_mode_btn.clicked.connect(lambda: self._set_mode("ST"))
        self._t_mode_btn.clicked.connect(lambda: self._set_mode("T"))
        
    def _set_mode(self, mode):
        self._current_tendency_mode = mode
        if mode == "ST":
            self._st_mode_btn.setStyleSheet(_style_active_btn())
            self._t_mode_btn.setStyleSheet(_style_btn())
            self._tendency_title.setText("ST Trends")
        else:
            self._t_mode_btn.setStyleSheet(_style_active_btn())
            self._st_mode_btn.setStyleSheet(_style_btn())
            self._tendency_title.setText("T Trends")
        self._update_tendency_graphs()
        
    def _update_hr_graph(self):
        hr_vals = [m.get('hr_mean', 0) for m in self._metrics]
        self._hr_canvas.set_data(hr_vals)
        self._hr_canvas.set_current_index(self._current_metric_index)
        
    def _update_tendency_graphs(self):
        # For now, use lead II data for all leads (in real scenario, we'd need per-lead metrics)
        if self._scan_results:
            st_vals = [res['st_mv'] for res in self._scan_results]
            t_vals = [res['t_mv'] for res in self._scan_results]
        else:
            st_vals = [m.get('st_mv', 0.0) for m in self._metrics]
            t_vals = [m.get('t_mv', 0.0) for m in self._metrics]
            
        for i, canvas in enumerate(self._tendency_canvases):
            if self._current_tendency_mode == "ST":
                canvas.set_tendency_mode("ST")
                canvas.set_data(st_vals)
            else:
                canvas.set_tendency_mode("T")
                canvas.set_data(t_vals)
            canvas.set_current_index(self._current_metric_index)
                
    def _update_ecg_strip(self):
        if self._replay_engine:
            # Get current window data
            data = self._replay_engine.get_all_leads_data(window_sec=3.0)
            if data is not None:
                self._ecg_strip.set_data(data, self._replay_engine.fs)
                
                # Set marker positions based on offsets
                r_peak_idx = data.shape[1] // 2  # Center as approximate R peak
                fs = self._replay_engine.fs
                self._ecg_strip.set_marker_positions(
                    i_pos=r_peak_idx + int(self._i_offset_ms * fs / 1000),
                    j_pos=r_peak_idx + int(self._j_offset_ms * fs / 1000),
                    k_pos=r_peak_idx + int(self._get_k_offset_ms() * fs / 1000),
                    t_pos=r_peak_idx + int(250 * fs / 1000)
                )
    
    def _get_k_offset_ms(self):
        if self._adjust_k_by_hr and self._metrics:
            current_metric = self._metrics[self._current_metric_index]
            hr = current_metric.get('hr_mean', 75)
            if hr > 100:
                return self._k_offset_ms - int((hr - 100) * 0.5)
            elif hr < 60:
                return self._k_offset_ms + int((60 - hr) * 0.5)
        return self._k_offset_ms
        
    def _on_start_scan(self):
        self._is_scanning = True
        self._scan_results = []
        if self._replay_engine:
            # Run scan in background thread
            self._scan_thread = threading.Thread(target=self._run_scan, daemon=True)
            self._scan_thread.start()
            
    def _run_scan(self):
        fs = self._replay_engine.fs
        for metric in self._metrics:
            t_sec = metric.get('t', 0.0)
            self._replay_engine.seek(t_sec)
            data = self._replay_engine.get_all_leads_data(window_sec=2.0)
            
            # Calculate ST/T for each lead
            lead_results = {}
            for i in range(min(12, data.shape[0])):
                lead_data = data[i]
                st_mv, t_mv = self._analyze_lead(lead_data, fs)
                lead_results[i] = {'st_mv': st_mv, 't_mv': t_mv}
                
            # Use lead II for display
            self._scan_results.append({
                't_sec': t_sec,
                'st_mv': lead_results.get(1, {}).get('st_mv', 0.0),
                't_mv': lead_results.get(1, {}).get('t_mv', 0.0),
                'lead_results': lead_results
            })
            
            if self._auto_update:
                # Update UI incrementally
                QMetaObject.invokeMethod(self, "_update_tendency_graphs", Qt.QueuedConnection)
                time.sleep(0.01)
                
        self._is_scanning = False
        
    def _analyze_lead(self, lead_data: np.ndarray, fs: int):
        """Calculate ST deviation and T-wave amplitude for a single lead."""
        from scipy.signal import butter, filtfilt, find_peaks
        
        if len(lead_data) < fs * 0.5:
            return 0.0, 0.0
            
        # Filter to find R-peak
        b, a = butter(2, [5.0 / (fs / 2), 15.0 / (fs / 2)], btype='band')
        filtered = filtfilt(b, a, lead_data)
        
        # Find R-peaks in the segment
        peaks, _ = find_peaks(np.abs(filtered), distance=int(0.2 * fs))
        if len(peaks) == 0:
            return 0.0, 0.0
            
        r_peak = peaks[len(peaks) // 2]
        
        # Calculate offsets
        fs = fs
        i_idx = max(0, r_peak + int(self._i_offset_ms * fs / 1000))
        j_idx = min(len(lead_data) - 1, r_peak + int(self._j_offset_ms * fs / 1000))
        k_idx = min(len(lead_data) - 1, r_peak + int(self._get_k_offset_ms() * fs / 1000))
        t_start = min(len(lead_data) - 1, r_peak + int(150 * fs / 1000))
        t_end = min(len(lead_data), r_peak + int(400 * fs / 1000))
        
        baseline = lead_data[i_idx] if 0 <= i_idx < len(lead_data) else 0.0
        st = lead_data[k_idx] if 0 <= k_idx < len(lead_data) else 0.0
        
        # T-wave
        t_amp = 0.0
        if t_start < t_end:
            t_seg = lead_data[t_start:t_end]
            if len(t_seg) > 0:
                peak_idx = np.argmax(np.abs(t_seg - baseline))
                t_amp = t_seg[peak_idx] - baseline
        
        # ADC to mV conversion
        adc_scale = 1.0 / 200.0
        return (st - baseline) * adc_scale, t_amp * adc_scale
        
    def _on_confirm_scan(self):
        if not self._scan_results:
            return
        _show_message_box(self, QMessageBox.Information, "Scan Complete", f"Processed {len(self._scan_results)} segments")
        
    def _on_reset_defaults(self):
        self._i_offset_ms = self._default_i_offset_ms
        self._j_offset_ms = self._default_j_offset_ms
        self._k_offset_ms = self._default_k_offset_ms
        self._update_ecg_strip()
        
    def _on_toggle_auto_update(self, checked):
        self._auto_update = checked
        
    def _on_toggle_adjust_k(self, checked):
        self._adjust_k_by_hr = checked
        self._update_ecg_strip()
        
    def _on_threshold_changed(self, value):
        self._st_threshold_mv = value
        for canvas in self._tendency_canvases:
            canvas.set_threshold(value)
        
    def _on_hr_canvas_clicked(self, index):
        self._current_metric_index = index
        self.currentIndexChanged.emit(index)
        self._update_hr_graph()
        self._update_tendency_graphs()
        if self._replay_engine:
            t_sec = self._metrics[index].get('t', 0.0)
            self._replay_engine.seek(t_sec)
            self._update_ecg_strip()
            
    def _on_tendency_clicked(self, lead_idx, canvas):
        idx = canvas.get_clicked_index()
        if idx is not None:
            self._on_hr_canvas_clicked(idx)
            
    def _on_marker_moved(self, marker_name, pos_samples):
        """Update offset when marker is dragged on ECG strip."""
        if not self._replay_engine:
            return
            
        fs = self._replay_engine.fs
        center_idx = self._ecg_strip._data.shape[1] // 2 if len(self._ecg_strip._data.shape) >1 else len(self._ecg_strip._data)//2
        delta_ms = int((pos_samples - center_idx) * 1000 / fs)
        
        if marker_name == 'I':
            self._i_offset_ms = delta_ms
        elif marker_name == 'J':
            self._j_offset_ms = delta_ms
        elif marker_name == 'K':
            self._k_offset_ms = delta_ms
            
    def update_from_metrics(self, metrics_list):
        self._metrics = metrics_list
        self._update_hr_graph()
        self._update_tendency_graphs()


class HRTrendCanvas(QWidget):
    clicked = pyqtSignal(int)
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = []
        self._current_index = 0
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        
    def set_data(self, vals):
        self._data = vals
        self.update()
        
    def set_current_index(self, idx):
        self._current_index = idx
        self.update()
        
    def mousePressEvent(self, event):
        event.accept()
        x = event.x()
        n = len(self._data)
        if n > 1:
            idx = int(round(x / (self.width() / (n-1))))
            idx = max(0, min(n-1, idx))
            self.clicked.emit(idx)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        
        # Draw grid lines
        grid_pen = QPen(QColor(COL_GREEN_DRK))
        grid_pen.setWidthF(0.5)
        painter.setPen(grid_pen)
        # Horizontal lines
        for y in [h//4, h//2, 3*h//4]:
            painter.drawLine(0, y, w, y)
        # Vertical lines
        for x in [w//4, w//2, 3*w//4]:
            painter.drawLine(x, 0, x, h)
            
        if not self._data:
            return
            
        # Plot HR data
        pen = QPen(QColor("#00FF00"))
        pen.setWidth(2)
        painter.setPen(pen)
        d = np.array(self._data)
        mn = d[d > 0]
        if len(mn) == 0:
            mn_val, mx_val = 40, 200
        else:
            mn_val, mx_val = max(40, mn.min() - 10), min(200, d.max() + 10)
        rng = mx_val - mn_val
        n = len(d)
        x_scale = w / max(n - 1, 1)
        for i in range(1, n):
            if d[i-1] <= 0 or d[i] <=0:
                continue
            x1 = int((i-1) * x_scale)
            y1 = int(h - 10 - (d[i-1] - mn_val) / rng * (h - 20))
            x2 = int(i * x_scale)
            y2 = int(h - 10 - (d[i] - mn_val) / rng * (h - 20))
            painter.drawLine(x1, y1, x2, y2)
            
        # Draw current index marker
        if 0 <= self._current_index < n:
            x = int(self._current_index * x_scale)
            marker_pen = QPen(QColor("#FFFF00"))
            marker_pen.setWidth(2)
            painter.setPen(marker_pen)
            painter.drawLine(x, 0, x, h)


class TendencyCanvas(QWidget):
    clicked = pyqtSignal()
    def __init__(self, parent=None, height: int = 70):
        super().__init__(parent)
        self._data = []
        self._mode = "ST"
        self._threshold = 0.1
        self._current_index = 0
        self._clicked_index = None
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        
    def set_tendency_mode(self, mode):
        self._mode = mode
        
    def set_threshold(self, threshold):
        self._threshold = threshold
        self.update()
        
    def set_data(self, vals):
        self._data = vals
        self.update()
        
    def set_current_index(self, idx):
        self._current_index = idx
        self.update()
        
    def get_clicked_index(self):
        return self._clicked_index
        
    def mousePressEvent(self, event):
        event.accept()
        x = event.x()
        n = len(self._data)
        if n > 1:
            idx = int(round(x / (self.width() / (n-1))))
            idx = max(0, min(n-1, idx))
            self._clicked_index = idx
            self.clicked.emit()
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        # Zero line
        pen = QPen(QColor(COL_GREEN_DRK))
        pen.setWidth(1)
        painter.setPen(pen)
        mid = h // 2
        painter.drawLine(0, mid, w, mid)
        
        if not self._data:
            return
            
        d = np.array(self._data)
        mn, mx = min(d.min(), -0.2), max(d.max(), 0.2)
        rng = max(mx - mn, 0.4)
        
        n = len(d)
        x_scale = w / max(n - 1, 1)
        for i in range(1, n):
            val = d[i]
            if self._mode == "ST":
                # ST: red if outside range, green if inside
                if abs(val) > self._threshold:
                    pen = QPen(QColor("#FF0000"))
                else:
                    pen = QPen(QColor("#00FF00"))
            else:
                # T: green if up, blue if down
                if val > 0:
                    pen = QPen(QColor("#00FF00"))
                else:
                    pen = QPen(QColor("#0099FF"))
            pen.setWidth(2)
            painter.setPen(pen)
            x1 = int((i-1) * x_scale)
            y1 = int(h - 5 - (d[i-1] - mn) / rng * (h - 10))
            x2 = int(i * x_scale)
            y2 = int(h - 5 - (d[i] - mn) / rng * (h - 10))
            painter.drawLine(x1, y1, x2, y2)
            
        # Draw current index marker
        if 0 <= self._current_index < n:
            x = int(self._current_index * x_scale)
            marker_pen = QPen(QColor("#FFFF00"))
            marker_pen.setWidth(2)
            painter.setPen(marker_pen)
            painter.drawLine(x, 0, x, h)
            
        # Draw numeric value at current index
        if 0 <= self._current_index < n:
            val = d[self._current_index]
            text = f"{val:.3f} mV"
            painter.setPen(QColor(COL_GREEN))
            font = QFont()
            font.setPointSize(9)
            painter.setFont(font)
            text_x = w - 80
            text_y = 15
            painter.drawText(text_x, text_y, text)


class STTMarkerCanvas(QWidget):
    markerMoved = pyqtSignal(str, int)
    def __init__(self, parent=None, height: int = 200):
        super().__init__(parent)
        self._data = np.zeros((1, 100))
        self._fs = 500
        self._i_pos = 0
        self._j_pos = 0
        self._k_pos = 0
        self._t_pos = 0
        self._dragging_marker = None
        self.setFixedHeight(height)
        self.setStyleSheet(f"background:{COL_BLACK};border:none;")
        self.setMouseTracking(True)
        
    def set_data(self, data, fs):
        self._data = data
        self._fs = fs
        self.update()
        
    def set_marker_positions(self, i_pos, j_pos, k_pos, t_pos):
        self._i_pos = i_pos
        self._j_pos = j_pos
        self._k_pos = k_pos
        self._t_pos = t_pos
        self.update()
        
    def mousePressEvent(self, event):
        event.accept()
        x = event.x()
        w = self.width()
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        sample_x = int((x / w) * n_samples)
        # Check which marker is near
        markers = [('I', self._i_pos), ('J', self._j_pos), ('K', self._k_pos), ('T', self._t_pos)]
        for name, pos in markers:
            pos_x = (pos / n_samples) * w
            if abs(x - pos_x) < 10:
                self._dragging_marker = name
                break
                
    def mouseMoveEvent(self, event):
        if self._dragging_marker is None:
            return
        x = event.x()
        w = self.width()
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        new_pos = int((x / w) * n_samples)
        new_pos = max(0, min(n_samples - 1, new_pos))
        if self._dragging_marker == 'I':
            self._i_pos = new_pos
        elif self._dragging_marker == 'J':
            self._j_pos = new_pos
        elif self._dragging_marker == 'K':
            self._k_pos = new_pos
        elif self._dragging_marker == 'T':
            self._t_pos = new_pos
        self.markerMoved.emit(self._dragging_marker, new_pos)
        self.update()
        
    def mouseReleaseEvent(self, event):
        self._dragging_marker = None
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(COL_BLACK))
        w, h = self.width(), self.height()
        
        n_leads = self._data.shape[0] if len(self._data.shape) >1 else 1
        lead_h = h // n_leads
        n_samples = self._data.shape[1] if len(self._data.shape) >1 else len(self._data)
        x_scale = w / max(n_samples - 1, 1) if n_samples > 1 else w
        
        colors = [
            "#00FF00",
            "#0099FF",
            "#FF9900",
            "#FF0000",
            "#9900FF",
            "#00FFFF",
            "#FFFF00",
            "#FF00FF",
            "#00FF99",
            "#99FF00",
            "#FF99FF",
            "#99FFFF"
        ]
        
        for i in range(n_leads):
            lead_data = self._data[i] if n_leads > 1 else self._data
            y0 = i * lead_h
            # Normalize and draw
            if len(lead_data) > 0:
                mn = np.min(lead_data)
                mx = np.max(lead_data)
                rng = mx - mn if (mx - mn) > 0 else 1
                y_offset = y0 + lead_h // 2
                y_scale = (lead_h - 20) / rng * 0.5
                
                # Draw grid lines
                grid_pen = QPen(QColor(COL_GREEN_DRK))
                grid_pen.setWidthF(0.5)
                painter.setPen(grid_pen)
                for grid_y in [y0 + lead_h//4, y0 + lead_h//2, y0 + 3*lead_h//4]:
                    painter.drawLine(0, grid_y, w, grid_y)
                
                # Draw ECG
                pen = QPen(QColor(colors[i % len(colors)]))
                pen.setWidth(1)
                painter.setPen(pen)
                for x in range(1, len(lead_data)):
                    x1 = int((x-1)*x_scale)
                    y1 = int(y_offset - (lead_data[x-1] - mn) * y_scale)
                    x2 = int(x * x_scale)
                    y2 = int(y_offset - (lead_data[x] - mn) * y_scale)
                    painter.drawLine(x1, y1, x2, y2)
        
        # Draw markers
        marker_colors = {
            'I': "#FF9900",
            'J': "#00FFFF",
            'K': "#FF00FF",
            'T': "#00FF00"
        }
        marker_labels = {
            'I': "I",
            'J': "J",
            'K': "K",
            'T': "T"
        }
        
        for name, pos in [('I', self._i_pos), ('J', self._j_pos), ('K', self._k_pos), ('T', self._t_pos)]:
            x = int(pos * x_scale)
            if 0 <= x < w:
                pen = QPen(QColor(marker_colors[name]))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.drawLine(x, 0, x, h)
                
                # Draw label
                font = QFont()
                font.setPointSize(10)
                font.setBold(True)
                painter.setFont(font)
                painter.drawText(x + 5, 15, marker_labels[name])


# -----------------------------------------------------------------------------
# 14. EDIT EVENT PANEL
# -----------------------------------------------------------------------------

class HolterEditEventPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._session_dir = ""
        self._selected_payload = {}
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

        # Left: event list + stats
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cols = ["Event name", "Start Time", "Chan.", "Print Len.", "Source", "Conf."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_click)
        left_layout.addWidget(self._ev_table, 1)

        # Stats
        stats_frame = QFrame()
        stats_frame.setStyleSheet(f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:4px;}}")
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(4)
        self._stat_labels = {}
        for i, (key, lbl) in enumerate([
            ("atrial_ecto","Atrial Ectopic"),("rr_int","Longest RR Interval"),
            ("hr","HR"),("max_hr","HR Max"),("min_hr","HR Min"),
            ("smax_hr","Sinus Max HR"),("smin_hr","Sinus Min HR"),
            ("brady","Bradycardia"),("user_ev","User Event"),("event","Event"),
        ]):
            r, c = divmod(i, 2)
            l = QLabel(f"{lbl}:")
            l.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;letter-spacing:0.5px;border:none;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, r*2, c)
            sf_layout.addWidget(v, r*2+1, c)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 1)

        # Right: ECG strip + thumbnail
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        # Tool buttons row
        tool_row = QHBoxLayout()
        for icon in ["Refresh", "Menu", "Prev", "Next", "<<", ">>", "Up", "Down"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_style_btn())
            btn.setFixedSize(30, 30)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        right_layout.addLayout(tool_row)

        # 12-lead ECG strips in a vertical scroll area (3 visible at once)
        leads_scroll = QScrollArea()
        leads_scroll.setWidgetResizable(True)
        leads_scroll.setFrameShape(QFrame.NoFrame)
        leads_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        leads_scroll.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        leads_host = QWidget()
        leads_host.setStyleSheet(f"background:{COL_BLACK};")
        leads_layout = QVBoxLayout(leads_host)
        leads_layout.setContentsMargins(4, 4, 4, 4)
        leads_layout.setSpacing(2)
        self._ch_strips = []
        _all_lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF",
                           "V1", "V2", "V3", "V4", "V5", "V6"]
        for name in _all_lead_names:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};font-size:11px;font-weight:bold;border:none;padding:2px 0;")
            leads_layout.addWidget(lbl)
            strip = ECGStripCanvas(height=80)
            strip.setFixedHeight(80)
            leads_layout.addWidget(strip)
            self._ch_strips.append(strip)
        leads_scroll.setWidget(leads_host)
        right_layout.addWidget(leads_scroll, 1)

        annot_box = QFrame()
        annot_box.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}")
        annot_layout = QGridLayout(annot_box)
        annot_layout.setContentsMargins(8, 6, 8, 6)
        annot_layout.setSpacing(6)
        annot_title = QLabel("Beat Annotation Editor")
        annot_title.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
        annot_layout.addWidget(annot_title, 0, 0, 1, 2)
        self._annot_event_id = QLineEdit()
        self._annot_event_id.setPlaceholderText("beat_id / event id")
        self._annot_event_id.setStyleSheet(f"QLineEdit{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_auto = QLineEdit()
        self._annot_auto.setPlaceholderText("auto label")
        self._annot_auto.setStyleSheet(self._annot_event_id.styleSheet())
        self._annot_clin = QComboBox()
        self._annot_clin.addItems(["", "N", "S", "V", "AF", "Pause", "Tachy", "Brady", "Other"])
        self._annot_clin.setStyleSheet(f"""
            QComboBox {{
                background:{COL_DARK}; color:{COL_GREEN}; border:1px solid {COL_GREEN_DRK};
                padding:5px; border-radius:4px;
            }}
            QComboBox QAbstractItemView {{
                background:{COL_DARK}; color:white; selection-background-color:{COL_GREEN_DRK};
            }}
        """)
        self._annot_conf = QDoubleSpinBox()
        self._annot_conf.setRange(0.0, 1.0)
        self._annot_conf.setSingleStep(0.05)
        self._annot_conf.setDecimals(2)
        self._annot_conf.setValue(0.0)
        self._annot_conf.setStyleSheet(f"QDoubleSpinBox{{background:{COL_DARK};color:{COL_GREEN};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_editor = QTextEdit()
        self._annot_editor.setPlaceholderText("Optional note or reviewer context")
        self._annot_editor.setFixedHeight(56)
        self._annot_editor.setStyleSheet(f"QTextEdit{{background:{COL_DARK};color:{COL_WHITE};border:1px solid {COL_GREEN_DRK};padding:5px;border-radius:4px;}}")
        self._annot_save_btn = QPushButton("Save Annotation")
        self._annot_save_btn.setStyleSheet(_style_active_btn())
        self._annot_save_btn.clicked.connect(self._save_annotation)
        annot_lbl = QLabel("Beat ID:")
        annot_lbl.setStyleSheet(f"color:{COL_WHITE};border:none;")
        annot_layout.addWidget(annot_lbl, 1, 0)
        annot_layout.addWidget(self._annot_event_id, 1, 1)
        annot_lbl = QLabel("Auto label:")
        annot_lbl.setStyleSheet(f"color:{COL_WHITE};border:none;")
        annot_layout.addWidget(annot_lbl, 2, 0)
        annot_layout.addWidget(self._annot_auto, 2, 1)
        annot_lbl = QLabel("Clinician label:")
        annot_lbl.setStyleSheet(f"color:{COL_WHITE};border:none;")
        annot_layout.addWidget(annot_lbl, 3, 0)
        annot_layout.addWidget(self._annot_clin, 3, 1)
        annot_lbl = QLabel("Confidence:")
        annot_lbl.setStyleSheet(f"color:{COL_WHITE};border:none;")
        annot_layout.addWidget(annot_lbl, 4, 0)
        annot_layout.addWidget(self._annot_conf, 4, 1)
        annot_layout.addWidget(self._annot_editor, 5, 0, 1, 2)
        annot_layout.addWidget(self._annot_save_btn, 6, 0, 1, 2)
        right_layout.addWidget(annot_box)

        nav_row = QHBoxLayout()
        for lbl in ["Prev Event", "Next Event", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        right_layout.addLayout(nav_row)
        layout.addWidget(right, 2)

    def load_events(self, events: list, summary: dict):
        # Filter out hidden event labels from display
        hidden_labels = ["Long QT Syndrome", "Wide QRS (non-specific)", "Frequent PVCs", "Multifocal PVCs"]
        filtered_events = []
        for ev in events:
            label = str(ev.get('label', '')).lower()
            if not any(hl.lower() in label for hl in hidden_labels):
                filtered_events.append(ev)
        
        self._events = filtered_events
        self._ev_table.setRowCount(len(filtered_events))
        for i, ev in enumerate(filtered_events):
            source = str(ev.get("source", "analysis"))
            conf = float(ev.get("confidence", 0.0) or 0.0)
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            for j, val in enumerate([ev['label'], t_str, "12", "7s", source, f"{conf:.2f}"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                if j == 0:
                    item.setData(Qt.UserRole, {
                        "beat_id": str(ev.get("beat_id", ev.get("template_label", ev.get("label", "")))),
                        "auto_label": str(ev.get("template_label", ev.get("label", ""))),
                        "clinician_label": str(ev.get("label", "")),
                        "confidence": conf,
                        "timestamp": float(ev.get("timestamp", 0.0) or 0.0),
                        "source": source,
                    })
                self._ev_table.setItem(i, j, item)
        s = summary
        for key, fmt in [("hr",f"{s.get('avg_hr',0):.0f} bpm"),
                          ("max_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("min_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("smax_hr",f"{s.get('max_hr',0):.0f} bpm"),
                          ("smin_hr",f"{s.get('min_hr',0):.0f} bpm"),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1"),("event","1"),
                          ("rr_int",f"{s.get('longest_rr_ms',0):.0f} ms"),
                          ("atrial_ecto","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def _on_click(self, row, col):
        if row < len(self._events):
            self._ev_table.selectRow(row)
            self.seek_requested.emit(self._events[row]['timestamp'])
            item = self._ev_table.item(row, 0)
            if item:
                payload = item.data(Qt.UserRole) or {}
                self._selected_payload = dict(payload)
                self._annot_event_id.setText(str(payload.get("beat_id", "")))
                self._annot_auto.setText(str(payload.get("auto_label", "")))
                self._annot_clin.setCurrentText(str(payload.get("clinician_label", "")))
                self._annot_conf.setValue(float(payload.get("confidence", 0.0) or 0.0))
                self._annot_editor.setPlainText(
                    f"Source: {payload.get('source', '')}\n"
                    f"Timestamp: {_sec_to_hms(float(payload.get('timestamp', 0.0) or 0.0))}"
                )

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""

    def _save_annotation(self):
        if not self._session_dir:
            QMessageBox.information(self, "Annotation", "No session directory is available for saving.")
            return
        beat_id = self._annot_event_id.text().strip()
        if not beat_id:
            QMessageBox.information(self, "Annotation", "Select an event or enter a beat ID first.")
            return
        annotation = {
            "beat_id": beat_id,
            "auto_label": self._annot_auto.text().strip(),
            "clinician_label": self._annot_clin.currentText().strip() or self._annot_auto.text().strip(),
            "confidence": float(self._annot_conf.value()),
            "edited_by": "clinician",
            "timestamp": float(self._selected_payload.get("timestamp", self._events[0].get("timestamp", 0.0) if self._events else 0.0)),
            "note": self._annot_editor.toPlainText().strip(),
        }
        try:
            append_annotation(self._session_dir, annotation)
            QMessageBox.information(self, "Annotation", "Annotation saved to session database.")
        except Exception as e:
            QMessageBox.warning(self, "Annotation", f"Could not save annotation: {e}")

    def set_replay_frame(self, data):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/500.0, N) if N > 0 else []
        for i, strip in enumerate(self._ch_strips):
            if i < data.shape[0] and N > 0:
                strip.set_data(x, data[i].copy())
            elif N > 0:
                strip.set_data([], [])


class ClickableSummaryTile(QFrame):
    clicked = pyqtSignal(str)

    def __init__(self, tile_key: str, parent=None):
        super().__init__(parent)
        self.tile_key = tile_key
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.tile_key)
        super().mousePressEvent(event)


# -----------------------------------------------------------------------------
# 15. EDIT STRIPS PANEL
# -----------------------------------------------------------------------------

class HolterEditStripsPanel(QWidget):
    seek_requested = pyqtSignal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"background:{COL_BG};")
        self._events = []
        self._summary = {}
        self._metrics_list = []
        self._focus_cards = {}
        self._selected_focus_key = "max_hr"
        self._selected_payload = {}
        self._tile_widgets = {}
        self._stat_labels = {}
        self._session_dir = ""
        self._build_ui()

    def set_session_dir(self, session_dir: str):
        self._session_dir = session_dir or ""


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

        # --- LEFT: Event List (20% width) ---
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        cols = ["Event name", "Start Time", "Chan.", "Print Len."]
        self._ev_table = QTableWidget(0, len(cols))
        self._ev_table.setHorizontalHeaderLabels(cols)
        self._ev_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self._ev_table.setStyleSheet(
            _table_style() +
            """
            QTableWidget::item:selected {
                background-color: rgba(66, 153, 225, 70);
                color: #F3F7FB;
                border: 1px solid rgba(255, 255, 255, 90);
            }
            QTableWidget::item:selected:active {
                background-color: rgba(66, 153, 225, 110);
            }
            """
        )
        self._ev_table.verticalHeader().setVisible(False)
        self._ev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self._ev_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self._ev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self._ev_table.cellClicked.connect(self._on_event_clicked)
        left_layout.addWidget(self._ev_table, 1)

        nav_row = QHBoxLayout()
        for lbl in ["Prev", "Next", "Remove All", "Remove"]:
            btn = QPushButton(lbl)
            btn.setStyleSheet(_style_btn())
            nav_row.addWidget(btn)
        left_layout.addLayout(nav_row)

        stats_frame = QFrame()
        stats_frame.setStyleSheet(
            f"QFrame{{background:{COL_DARK};border:1px solid {COL_GREEN_DRK};border-radius:6px;}}"
        )
        sf_layout = QGridLayout(stats_frame)
        sf_layout.setContentsMargins(8, 6, 8, 6)
        sf_layout.setSpacing(6)
        for i, (key, label) in enumerate([
            ("hr", "Avg HR"),
            ("max_hr", "HR Max"),
            ("min_hr", "HR Min"),
            ("smax_hr", "Sinus Max HR"),
            ("smin_hr", "Sinus Min HR"),
            ("rr_int", "Longest RR"),
            ("brady", "Brady"),
            ("user_ev", "User Event"),
        ]):
            row, col = divmod(i, 2)
            l = QLabel(f"{label}:")
            l.setStyleSheet(f"color:{UI_MUTED};font-size:10px;font-weight:bold;letter-spacing:0.5px;border:none;")
            v = QLabel("-")
            v.setStyleSheet(f"color:{COL_GREEN};font-size:12px;font-weight:bold;border:none;")
            sf_layout.addWidget(l, row * 2, col)
            sf_layout.addWidget(v, row * 2 + 1, col)
            self._stat_labels[key] = v
        left_layout.addWidget(stats_frame)
        layout.addWidget(left, 2)

        # --- CENTER: 2x2 Thumbnail Boxes (30% width) ---
        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(4)

        # Tool buttons
        tool_row = QHBoxLayout()
        for icon in ["Refresh", "Menu", "Prev", "Next", "<<", ">>", "Up", "Down", "End"]:
            btn = QPushButton(icon)
            btn.setStyleSheet(_style_btn())
            btn.setFixedSize(28, 28)
            tool_row.addWidget(btn)
        tool_row.addStretch()
        center_layout.addLayout(tool_row)

        thumb_grid = QGridLayout()
        thumb_grid.setSpacing(12)
        thumb_grid.setHorizontalSpacing(12)
        thumb_grid.setVerticalSpacing(12)
        self._thumb_frames = []
        self._tile_style_normal = f"QFrame{{background:{COL_DARK};border:1px solid #888888;border-radius:8px;}}"
        self._tile_style_active = f"QFrame{{background:{COL_BLACK};border:2px solid {COL_YELLOW};border-radius:8px;}}"
        self._tile_widgets = {}
        tile_specs = [
            (0, 0, "max_hr", "Maximum Heart Rate"),
            (0, 1, "min_hr", "Minimum Heart Rate"),
            (1, 0, "smax_hr", "Sinus Max HR"),
            (1, 1, "smin_hr", "Sinus Min HR"),
        ]
        for row, col, key, title in tile_specs:
            frame = ClickableSummaryTile(key)
            frame.setMinimumHeight(210)
            frame.setMinimumWidth(280)
            frame.setStyleSheet(self._tile_style_normal)
            fl = QVBoxLayout(frame)
            fl.setContentsMargins(8, 8, 8, 8)
            fl.setSpacing(4)

            header_w = QWidget()
            header_w.setStyleSheet("border:none;")
            hl = QHBoxLayout(header_w)
            hl.setContentsMargins(0, 0, 0, 0)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("color:#FFFFFF;font-size:12px;font-weight:bold;")
            hl.addWidget(t_lbl)
            hl.addStretch()
            hr_lbl = QLabel("HR: --")
            hr_lbl.setStyleSheet("color:#FFFF00;font-size:11px;font-weight:bold;")
            hl.addWidget(hr_lbl)
            fl.addWidget(header_w)

            time_lbl = QLabel("--:--:--")
            time_lbl.setStyleSheet("color:#AAAAAA;font-size:10px;")
            fl.addWidget(time_lbl)

            strips = []
            for _ in range(3):
                strip = ECGStripCanvas(height=40)
                strip.setStyleSheet("border:1px solid #444444;")
                fl.addWidget(strip)
                strips.append(strip)
                self._thumb_frames.append(strip)

            self._tile_widgets[key] = {
                "frame": frame,
                "title": t_lbl,
                "hr": hr_lbl,
                "time": time_lbl,
                "strips": strips,
            }
            frame.clicked.connect(self._select_focus_tile)
            thumb_grid.addWidget(frame, row, col)

        thumb_grid.setColumnStretch(0, 1)
        thumb_grid.setColumnStretch(1, 1)
        center_layout.addLayout(thumb_grid)
        center_layout.addStretch()
        layout.addWidget(center, 4)

        # --- RIGHT: Large Clinical Strip View (50% width) ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        detail_frame = QFrame()
        detail_frame.setStyleSheet(
            f"QFrame{{background:{COL_BLACK};border:1px solid {COL_GREEN_DRK};border-radius:8px;}}"
        )
        df_layout = QGridLayout(detail_frame)
        df_layout.setContentsMargins(10, 8, 10, 8)
        df_layout.setHorizontalSpacing(10)
        df_layout.setVerticalSpacing(4)
        self._detail_title = QLabel("Select a heart-rate tile")
        self._detail_title.setStyleSheet(
            f"color:{COL_YELLOW};font-size:14px;font-weight:bold;border:none;"
        )
        self._detail_value = QLabel("-- bpm")
        self._detail_value.setStyleSheet(
            f"color:{COL_GREEN};font-size:26px;font-weight:bold;border:none;"
        )
        self._detail_time = QLabel("-")
        self._detail_time.setStyleSheet(
            f"color:{COL_TIMESTAMP};font-size:11px;font-weight:bold;border:none;"
        )
        self._detail_note = QLabel("Click any tile to expand the matching strip view here.")
        self._detail_note.setStyleSheet(
            f"color:{COL_WHITE};font-size:11px;border:none;"
        )
        self._detail_note.setWordWrap(True)
        df_layout.addWidget(self._detail_title, 0, 0, 1, 2)
        df_layout.addWidget(self._detail_value, 1, 0, 1, 1)
        df_layout.addWidget(self._detail_time, 1, 1, 1, 1, Qt.AlignRight | Qt.AlignVCenter)
        df_layout.addWidget(self._detail_note, 2, 0, 1, 2)
        right_layout.addWidget(detail_frame, 0)

        main_frame = QFrame()
        main_frame.setStyleSheet(f"QFrame{{background:{COL_BLACK};border:1px solid #888888;border-radius:4px;}}")
        ml = QVBoxLayout(main_frame)
        ml.setContentsMargins(8, 8, 8, 8)

        self._detail_header_lbl = QLabel("Detailed View")
        self._detail_header_lbl.setStyleSheet("color:#FFFF00;font-size:12px;font-weight:bold;border:none;")
        ml.addWidget(self._detail_header_lbl)

        self._main_strips = []
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(6)
        
        lead_names = ["Lead I", "Lead II", "Lead III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        
        for name in lead_names:
            lbl = QLabel(name)
            lbl.setStyleSheet(f"color:{COL_GREEN};border:none;font-weight:bold;")
            strip = ECGStripCanvas(height=95)
            scroll_layout.addWidget(lbl)
            scroll_layout.addWidget(strip)
            self._main_strips.append(strip)
            
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        ml.addWidget(scroll_area)

        right_layout.addWidget(main_frame, 1)

        self._mini_lbl = QLabel("Lead II")
        self._mini_lbl.setStyleSheet(f"color:{COL_GREEN};border:none;font-weight:bold;")
        right_layout.addWidget(self._mini_lbl)

        self._mini = ECGStripCanvas(height=40, color="#00AA00")
        right_layout.addWidget(self._mini)
        layout.addWidget(right, 5)

    def _format_bpm(self, value: float) -> str:
        if value and value > 0:
            return f"{float(value):.0f} bpm"
        return "-- bpm"

    def _on_event_clicked(self, row, col):
        if row < len(self._events):
            self._ev_table.selectRow(row)
            self.seek_requested.emit(self._events[row]['timestamp'])

    def _build_focus_cards(self):
        summary = self._summary or {}
        focus = derive_hr_focus_summary(self._metrics_list or [])

        def _metric_value(*keys, default=0.0):
            for key in keys:
                for source in (summary, focus):
                    if source.get(key) is not None:
                        try:
                            value = float(source.get(key) or 0.0)
                            if value > 0:
                                return value
                        except Exception:
                            continue
            return float(default)

        def _metric_time(*keys):
            for key in keys:
                for source in (summary, focus):
                    val = source.get(f"{key}_time")
                    if val:
                        return str(val)
            return ""

        def _metric_timestamp(*keys):
            for key in keys:
                for source in (summary, focus):
                    val = source.get(f"{key}_timestamp")
                    if val is not None:
                        try:
                            ts = float(val or 0.0)
                            if ts > 0:
                                return ts
                        except Exception:
                            continue
            return 0.0

        self._focus_cards = {
            "max_hr": {
                "title": "Maximum Heart Rate",
                "value": _metric_value("max_hr"),
                "time": _metric_time("max_hr"),
                "timestamp": _metric_timestamp("max_hr"),
                "note": "Highest overall heart rate found in the recording.",
            },
            "min_hr": {
                "title": "Minimum Heart Rate",
                "value": _metric_value("min_hr"),
                "time": _metric_time("min_hr"),
                "timestamp": _metric_timestamp("min_hr"),
                "note": "Lowest overall heart rate found in the recording.",
            },
            "smax_hr": {
                "title": "Sinus Max HR",
                "value": _metric_value("sinus_max_hr", "max_hr"),
                "time": _metric_time("sinus_max_hr", "max_hr"),
                "timestamp": _metric_timestamp("sinus_max_hr", "max_hr"),
                "note": "Highest heart rate from sinus-like beats.",
            },
            "smin_hr": {
                "title": "Sinus Min HR",
                "value": _metric_value("sinus_min_hr", "min_hr"),
                "time": _metric_time("sinus_min_hr", "min_hr"),
                "timestamp": _metric_timestamp("sinus_min_hr", "min_hr"),
                "note": "Lowest heart rate from sinus-like beats.",
            },
        }

    def _refresh_focus_tiles(self):
        for key, widget in self._tile_widgets.items():
            card = self._focus_cards.get(key, {})
            widget["hr"].setText(f"HR: {self._format_bpm(card.get('value', 0.0))}")
            widget["time"].setText(card.get("time") or "-")
            widget["frame"].setStyleSheet(
                self._tile_style_active if key == self._selected_focus_key else self._tile_style_normal
            )

    def _update_detail_panel(self, key: str):
        card = self._focus_cards.get(key) or self._focus_cards.get("max_hr") or {}
        title = card.get("title", "Heart Rate Detail")
        value = card.get("value", 0.0)
        time_str = card.get("time") or "-"
        note = card.get("note", "")
        self._detail_title.setText(title)
        self._detail_value.setText(self._format_bpm(value))
        self._detail_time.setText(time_str)
        self._detail_note.setText(note or "Click a tile to expand the matching strip view here.")
        self._detail_header_lbl.setText(f"Detailed View  -  {title}  -  {time_str}")

    def _select_focus_tile(self, tile_key: str, emit_seek: bool = True):
        if tile_key not in self._focus_cards:
            return
        self._selected_focus_key = tile_key
        self._refresh_focus_tiles()
        self._update_detail_panel(tile_key)
        if emit_seek:
            timestamp = float(self._focus_cards.get(tile_key, {}).get("timestamp", 0.0) or 0.0)
            if timestamp > 0:
                self.seek_requested.emit(timestamp)

    def load_events(self, events: list, summary: dict, metrics_list: Optional[list] = None):
        # Filter out hidden event labels from display
        hidden_labels = ["Long QT Syndrome", "Wide QRS (non-specific)", "Frequent PVCs", "Multifocal PVCs"]
        filtered_events = []
        for ev in events:
            label = str(ev.get('label', '')).lower()
            if not any(hl.lower() in label for hl in hidden_labels):
                filtered_events.append(ev)
        
        self._events = filtered_events
        self._summary = dict(summary or {})
        self._metrics_list = list(metrics_list or [])
        self._build_focus_cards()
        self._ev_table.setRowCount(len(filtered_events))
        for i, ev in enumerate(filtered_events):
            t_str = _format_system_time(self._session_dir, ev['timestamp'])
            for j, val in enumerate([ev['label'], t_str, "12", "7s"]):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_WHITE))
                self._ev_table.setItem(i, j, item)
        if self._selected_focus_key not in self._focus_cards:
            self._selected_focus_key = "max_hr"
        self._refresh_focus_tiles()
        self._update_detail_panel(self._selected_focus_key)
        s = self._summary
        max_card = self._focus_cards.get("max_hr", {})
        min_card = self._focus_cards.get("min_hr", {})
        smax_card = self._focus_cards.get("smax_hr", {})
        smin_card = self._focus_cards.get("smin_hr", {})
        for key, fmt in [("hr",f"{s.get('avg_hr',0):.0f} bpm"),
                          ("max_hr",self._format_bpm(float(max_card.get('value', s.get('max_hr',0)) or 0.0))),
                          ("min_hr",self._format_bpm(float(min_card.get('value', s.get('min_hr',0)) or 0.0))),
                          ("smax_hr",self._format_bpm(float(smax_card.get('value', s.get('sinus_max_hr', s.get('max_hr',0))) or 0.0))),
                          ("smin_hr",self._format_bpm(float(smin_card.get('value', s.get('sinus_min_hr', s.get('min_hr',0))) or 0.0))),
                          ("brady",str(s.get('brady_beats',0))),
                          ("user_ev","1"),("event","1"),
                          ("rr_int",f"{s.get('longest_rr_ms',0):.0f} ms"),
                          ("atrial_ecto","1")]:
            if key in self._stat_labels:
                self._stat_labels[key].setText(fmt)

    def set_replay_frame(self, data, metrics_dict=None, current_sec=0.0):
        if data is None or data.shape[0] < 1: return
        N = data.shape[1]
        x = np.linspace(0, N/250.0, N) if N > 0 else []
        
        start_sec = max(0.0, current_sec - 5.0) # 10s window centered
        all_beats = metrics_dict.get('all_beats', []) if metrics_dict else []
        
        if N > 0:
            # Update thumbnails (cycle through CH1, CH2, CH3 if available)
            for i, strip in enumerate(self._thumb_frames):
                ch_idx = i % 3
                if ch_idx < data.shape[0]:
                    strip.set_data(x, data[ch_idx].copy(), beat_annotations=all_beats, start_sec=start_sec)
                else:
                    strip.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)
            
            # Update large main strips
            if hasattr(self, "_main_strips"):
                for idx, strip in enumerate(self._main_strips):
                    if idx < data.shape[0]:
                        strip.set_data(x, data[idx].copy(), beat_annotations=all_beats, start_sec=start_sec)
                    elif data.shape[0] > 0:
                        strip.set_data(x, np.zeros_like(data[0]), beat_annotations=all_beats, start_sec=start_sec)
                
            if data.shape[0] > 1:
                self._mini.set_data(x, data[1].copy(), beat_annotations=all_beats, start_sec=start_sec)
            else:
                self._mini.set_data(x, data[0].copy(), beat_annotations=all_beats, start_sec=start_sec)


# -----------------------------------------------------------------------------
# 16. REPORT TABLE PANEL
# -----------------------------------------------------------------------------

class HolterReportTablePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
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
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        title = QLabel("Hour-by-Hour Report Table")
        title.setStyleSheet(f"color:{COL_GREEN};font-size:14px;font-weight:bold;border:none;")
        layout.addWidget(title)

        cols = ["Hour", "Beats", "HR Min", "HR Avg", "HR Max",
                "VE Iso.", "VE Coup.", "VE Runs", "VE Total", "VE %",
                "SVE Iso.", "SVE Coup.", "SVE Total", "Pauses"]
        self._table = QTableWidget(0, len(cols))
        self._table.setHorizontalHeaderLabels(cols)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self._table.setStyleSheet(_table_style())
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self._table, 1)

    def update_from_metrics(self, metrics_list: list):
        hourly: Dict[int, list] = {}
        for m in metrics_list:
            h = int(m.get('t', 0) // 3600)
            hourly.setdefault(h, []).append(m)

        rows = []
        total_beats = total_pauses = 0
        for h in sorted(hourly.keys()):
            chunks = hourly[h]
            beats = sum(c.get('beat_count', 0) for c in chunks)
            hr_vals = [c.get('hr_mean', 0) for c in chunks if c.get('hr_mean', 0) > 0]
            hr_min_vals = [c.get('hr_min', 0) for c in chunks if c.get('hr_min', 0) > 0]
            hr_max_vals = [c.get('hr_max', 0) for c in chunks if c.get('hr_max', 0) > 0]
            pauses = sum(c.get('pauses', 0) for c in chunks)
            avg_hr = int(np.mean(hr_vals)) if hr_vals else 0
            min_hr = int(np.min(hr_min_vals)) if hr_min_vals else 0
            max_hr = int(np.max(hr_max_vals)) if hr_max_vals else 0
            total_beats += beats
            total_pauses += pauses
            rows.append([f"{h:02d}:00", str(beats), str(min_hr), str(avg_hr), str(max_hr),
                          "0","0","0","0","0%","0","0","0", str(pauses)])

        # Total row
        rows.append(["Total", str(total_beats), "-", "-", "-",
                      "0","0","0","0","0%","0","0","0", str(total_pauses)])

        self._table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            is_total = (i == len(rows) - 1)
            for j, val in enumerate(row):
                item = QTableWidgetItem(val)
                item.setForeground(QColor(COL_GREEN if j == 0 or is_total else COL_WHITE))
                if is_total:
                    item.setBackground(QColor(COL_GREEN_DRK))
                self._table.setItem(i, j, item)

# -----------------------------------------------------------------------------
# 17. HOLTER MAIN WINDOW  - Orchestrates everything
# -----------------------------------------------------------------------------

class HolterMainWindow(QDialog):
    def __init__(self, parent=None, session_dir: str = "",
                 patient_info: dict = None,
                 writer=None,
                 live_source=None,
                 duration_hours: int = 24):
        super().__init__(parent)
        self.setWindowTitle("Comprehensive ECG Analysis Monitor & Analysis")
        self.setMinimumSize(900, 620)

        screen = QApplication.primaryScreen()
        if screen:
            g = screen.availableGeometry()
            self.resize(max(1100, int(g.width() * 0.92)), max(750, int(g.height() * 0.92)))
        else:
            self.resize(1400, 900)

        self.setWindowFlags(Qt.Window | Qt.CustomizeWindowHint | Qt.WindowTitleHint | Qt.WindowCloseButtonHint)
        self.setStyleSheet(f"QDialog{{background:{UI_BG};}}")
        self.showMaximized()

        self.session_dir = session_dir
        self.patient_info = _normalize_patient_info(patient_info or (writer.patient_info if writer else {}))
        self._writer = writer
        self._live_source = live_source
        self._duration_hours = duration_hours
        self._replay_engine = None
        self._metrics_list = []
        self._summary = {}
        self._last_live_seq = -1
        self._tab_name_map = {}

        if not self.session_dir and writer:
            self.session_dir = getattr(writer, 'session_dir', '')

        self._load_session()
        self._build_ui()

        if self._writer:
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
            self._live_timer.start(1000)

    # ---- Session loading -----------------------------------------------------------------------------

    def _load_session(self):
        self._metrics_list = []
        metadata = read_session_metadata(self.session_dir) if self.session_dir else {}
        self.patient_info = _load_patient_info_from_session(self.session_dir, self.patient_info)
        layered_metrics = load_metrics(self.session_dir) if self.session_dir else []
        if layered_metrics:
            self._metrics_list = layered_metrics
        jsonl_path = os.path.join(self.session_dir, 'metrics.jsonl') if self.session_dir else ''
        if not self._metrics_list and os.path.exists(jsonl_path):
            try:
                with open(jsonl_path) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self._metrics_list.append(json.loads(line))
            except Exception as e:
                print(f"[HolterUI] Could not load metrics: {e}")

        ecgh_path = os.path.join(self.session_dir, 'recording.ecgh') if self.session_dir else ''
        if os.path.exists(ecgh_path):
            try:
                from .file_format import ECGHFileReader
                from .replay_engine import HolterReplayEngine
                self._replay_engine = HolterReplayEngine(ecgh_path)
                self._summary = self._replay_engine.get_summary()
            except Exception as e:
                print(f"[HolterUI] Replay engine error: {e}")
                self._summary = self._build_summary_from_metrics()
        else:
            self._summary = self._build_summary_from_metrics()
        if metadata and isinstance(metadata.get("summary"), dict):
            meta_sum = dict(metadata.get("summary"))
            if self._summary:
                meta_sum.update(self._summary)
            self._summary = meta_sum
        if self._summary is not None:
            self._summary["patient_info"] = dict(self.patient_info or {})

    def _build_summary_from_metrics(self) -> dict:
        if not self._metrics_list:
            return {}
        ml = self._metrics_list
        hr_vals = [m['hr_mean'] for m in ml if m.get('hr_mean', 0) > 0]
        beat_counts = [m.get('beat_count', 0) for m in ml]
        rr_stds = [m['rr_std'] for m in ml if m.get('rr_std', 0) > 0]
        rmssds = [m['rmssd'] for m in ml if m.get('rmssd', 0) > 0]
        pnn50s = [m['pnn50'] for m in ml if m.get('pnn50', 0) >= 0]
        qualities = [m['quality'] for m in ml if m.get('quality', 0) > 0]
        arrhy_counts: Dict[str, int] = {}
        beat_class_totals: Dict[str, int] = {}
        template_counts = []
        tachy_sec = 0.0
        brady_sec = 0.0
        for m in ml:
            for a in m.get('arrhythmias', []):
                arrhy_counts[a] = arrhy_counts.get(a, 0) + 1
            for cls, count in (m.get('beat_class_counts', {}) or {}).items():
                beat_class_totals[cls] = beat_class_totals.get(cls, 0) + int(count or 0)
            template_counts.append(int(m.get('template_count', 0) or 0))
            chunk_dur = float(m.get('duration', 4.0) or 4.0)
            hr_m = float(m.get('hr_mean', 0) or 0)
            if hr_m > 100:
                tachy_sec += chunk_dur
            elif 0 < hr_m < 60:
                brady_sec += chunk_dur
        all_rr = [m.get('longest_rr', 0) for m in ml]
        # Find max/min HR chunks for timestamps
        max_hr_chunk = max((m for m in ml if m.get('hr_mean', 0) > 0), key=lambda m: m.get('hr_mean', 0), default={})
        min_hr_chunk = min((m for m in ml if m.get('hr_mean', 0) > 0), key=lambda m: m.get('hr_mean', 0), default={})
        max_hr_t = float(max_hr_chunk.get('t', 0.0) or 0.0)
        min_hr_t = float(min_hr_chunk.get('t', 0.0) or 0.0)
        focus = derive_hr_focus_summary(ml)
        total_dur = _metrics_duration_sec(ml)
        tachy_pct = (tachy_sec / total_dur * 100) if total_dur > 0 else 0.0
        brady_pct = (brady_sec / total_dur * 100) if total_dur > 0 else 0.0
        return {
            'duration_sec': total_dur,
            'total_beats': sum(beat_counts),
            'avg_hr': float(np.mean(hr_vals)) if hr_vals else 0.0,
            'max_hr': float(np.max(hr_vals)) if hr_vals else 0.0,
            'min_hr': float(np.min(hr_vals)) if hr_vals else 0.0,
            'max_hr_time': _sec_to_hms(max_hr_t),
            'max_hr_timestamp': max_hr_t,
            'min_hr_time': _sec_to_hms(min_hr_t),
            'min_hr_timestamp': min_hr_t,
            'sinus_max_hr': focus.get('sinus_max_hr', float(np.max(hr_vals)) if hr_vals else 0.0),
            'sinus_min_hr': focus.get('sinus_min_hr', float(np.min(hr_vals)) if hr_vals else 0.0),
            'sinus_max_hr_time': focus.get('sinus_max_hr_time', _sec_to_hms(max_hr_t)),
            'sinus_max_hr_timestamp': focus.get('sinus_max_hr_timestamp', max_hr_t),
            'sinus_min_hr_time': focus.get('sinus_min_hr_time', _sec_to_hms(min_hr_t)),
            'sinus_min_hr_timestamp': focus.get('sinus_min_hr_timestamp', min_hr_t),
            'sdnn': float(np.mean(rr_stds)) if rr_stds else 0.0,
            'rmssd': float(np.mean(rmssds)) if rmssds else 0.0,
            'pnn50': float(np.mean(pnn50s)) if pnn50s else 0.0,
            'avg_quality': float(np.mean(qualities)) if qualities else 1.0,
            'arrhythmia_counts': arrhy_counts,
            'longest_rr_ms': max(all_rr) if all_rr else 0,
            'tachy_beats': sum(m.get('tachy_beats', 0) for m in ml),
            'brady_beats': sum(m.get('brady_beats', 0) for m in ml),
            'tachy_sec': tachy_sec,
            'brady_sec': brady_sec,
            'tachy_pct': tachy_pct,
            'brady_pct': brady_pct,
            'pauses': sum(m.get('pauses', 0) for m in ml),
            'avg_st_mv': float(np.mean([m.get('st_mv', 0) for m in ml])),
            'patient_info': self.patient_info,
            'chunks_analyzed': len(ml),
            'beat_class_totals': beat_class_totals,
            've_beats': int(beat_class_totals.get('VE', 0)),
            'sve_beats': int(beat_class_totals.get('SVE', 0)),
            'template_count': max(template_counts) if template_counts else 0,
        }

    def _tab_index_for(self, name: str) -> int:
        if not hasattr(self, '_tabs'):
            return -1
        target = (name or '').strip().lower()
        aliases = {
            'overview': 'overview',
            'view': 'preview',
            'report': 'preview',
            'preview': 'preview',
            'lorenz': 'lorenz',
            'histogram': 'histogram',
            'template': 'template',
            'af analysis': 'af analysis',
            'st tendency': 'st tendency',
            'edit event': 'edit event',
            'edit strips': 'edit strips',
            'report table': 'report table',
            'hrv': 'hrv',
            'recordings': 'recordings',
            'record settings': 'recordings',
            'replay': 'replay',
            'lorenz': 'replay',
        }
        target = aliases.get(target, target)
        for idx in range(self._tabs.count()):
            if self._tabs.tabText(idx).strip().lower() == target:
                return idx
        return -1

    def _focus_tab(self, name: str):
        idx = self._tab_index_for(name)
        if idx >= 0:
            self._tabs.setCurrentIndex(idx)

    def _recordings_panel(self):
        return getattr(self, '_record_mgmt_panel', None)

    def _open_recordings_folder(self):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')

    def _search_recordings(self):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_search'):
            panel._search.setFocus()
            panel._search.selectAll()

    def _apply_recordings_filter(self, label: str):
        if self._is_replay_active():
            return
        self._focus_tab('RECORDINGS')
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_filter'):
            idx = panel._filter.findText(label)
            if idx >= 0:
                panel._filter.setCurrentIndex(idx)

    def _import_recording(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_import_session'):
            panel._import_session()

    def _backup_recordings(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_backup_root'):
            panel._backup_root()

    def _delete_recording(self):
        if self._is_replay_active():
            return
        panel = self._recordings_panel()
        if panel and hasattr(panel, '_delete_session'):
            panel._delete_session()

    def _is_replay_active(self) -> bool:
        panel = getattr(self, "_replay_panel", None)
        engine = getattr(self, "_replay_engine", None)
        return bool(panel and engine and engine.is_playing())

    def _set_record_browser_enabled(self, enabled: bool):
        panel = self._recordings_panel()
        if panel is not None:
            for name in ("Browse", "Import", "Export", "Backup", "Delete"):
                btn = getattr(panel, "_action_buttons", {}).get(name)
                if btn is not None:
                    btn.setEnabled(bool(enabled))
            table = getattr(panel, "_table", None)
            if table is not None:
                table.setEnabled(bool(enabled))
        top_browse = getattr(self, "_action_buttons", {}).get("Browse")
        if top_browse is not None:
            top_browse.setEnabled(bool(enabled))

    def _generate_from_current(self):
        self._focus_tab('PREVIEW')
        self._generate_report()

    def _refresh_current_session(self):
        self._load_session()
        self._refresh_ui()

    def _confirm_reanalysis(self):
        msg_box = QMessageBox(self)
        msg_box.setWindowTitle("Reanalyse Data")
        msg_box.setText("Would you like to reanalyse the data?")
        msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg_box.setDefaultButton(QMessageBox.No)
        msg_box.setStyleSheet(f"""
            QMessageBox {{
                background:{COL_BG};
                border:2px solid {COL_GREEN_DRK};
            }}
            QMessageBox QLabel {{
                color:{COL_WHITE};
                font-size:13px;
                font-weight:bold;
                border:none;
            }}
            QPushButton {{
                background:{COL_DARK};
                color:{COL_GREEN};
                border:1px solid {COL_GREEN_DRK};
                padding:6px 22px;
                border-radius:5px;
                font-size:12px;
                font-weight:bold;
            }}
            QPushButton:hover {{
                background:{COL_GREEN_DRK};
                color:{COL_WHITE};
            }}
            QPushButton:pressed {{
                background:{COL_GREEN};
                color:{COL_BLACK};
            }}
        """)
        reply = msg_box.exec_()
        if reply == QMessageBox.Yes:
            self._run_reanalysis_with_progress()

    def _run_reanalysis_with_progress(self):
        # --- Progress dialog ---
        dlg = QDialog(self, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        dlg.setFixedSize(480, 160)
        dlg.setStyleSheet("""
            QDialog {
                background: #0D1117;
                border: 1px solid #00CC66;
                border-radius: 12px;
            }
        """)
        dlg_layout = QVBoxLayout(dlg)
        dlg_layout.setContentsMargins(28, 22, 28, 22)
        dlg_layout.setSpacing(10)

        # Header row
        hdr_row = QHBoxLayout()
        dot = QLabel(" ")
        dot.setStyleSheet("color:#00FF00; font-size:14px; border:none;")
        hdr_row.addWidget(dot)
        title_lbl = QLabel("ECG Reanalysis")
        title_lbl.setStyleSheet("color:#00FF00; font-size:14px; font-weight:bold; border:none; letter-spacing:1px;")
        hdr_row.addWidget(title_lbl)
        hdr_row.addStretch()
        pct_lbl = QLabel("0%")
        pct_lbl.setStyleSheet("color:#00CC66; font-size:12px; font-weight:bold; border:none;")
        hdr_row.addWidget(pct_lbl)
        dlg_layout.addLayout(hdr_row)

        # Progress bar
        progress = QProgressBar()
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setFixedHeight(10)
        progress.setStyleSheet("""
            QProgressBar {
                background: #161B22;
                border: 1px solid #444444;
                border-radius: 5px;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00CC66, stop:0.6 #00FF00, stop:1 #66FFB3);
                border-radius: 5px;
            }
        """)
        dlg_layout.addWidget(progress)

        # Status label
        status_lbl = QLabel("Initialising...")
        status_lbl.setStyleSheet("color:#888888; font-size:11px; border:none;")
        dlg_layout.addWidget(status_lbl)

        # Sub info
        sub_lbl = QLabel("This may take a few moments depending on recording length.")
        sub_lbl.setStyleSheet("color:#444444; font-size:10px; border:none;")
        dlg_layout.addWidget(sub_lbl)

        dlg.setModal(True)
        dlg.show()
        QApplication.processEvents()

        # --- Worker thread ---
        class _ReanalysisWorker(QThread):
            progress_changed = pyqtSignal(int, str)
            finished_ok = pyqtSignal()

            def __init__(self, fn):
                super().__init__()
                self._fn = fn

            def run(self):
                self.progress_changed.emit(10, "Loading session metadata...")
                try:
                    self._fn()
                    self.progress_changed.emit(90, "Finalising...")
                except Exception as e:
                    print(f"[Reanalysis] Error: {e}")
                self.progress_changed.emit(100, "Done!")
                self.finished_ok.emit()

        def _on_progress(val, txt):
            progress.setValue(val)
            pct_lbl.setText(f"{val}%")
            status_lbl.setText(txt)
            QApplication.processEvents()

        def _on_done():
            progress.setValue(100)
            pct_lbl.setText("100%")
            status_lbl.setText("Complete -- loading replay...")
            QApplication.processEvents()
            QTimer.singleShot(450, lambda: (dlg.accept(), _navigate_replay()))

        def _navigate_replay():
            try:
                self._refresh_ui()
            except Exception as e:
                print(f"[Reanalysis] _refresh_ui error: {e}")
            self._focus_tab('REPLAY')

        self._reanalysis_worker = _ReanalysisWorker(self._load_session)
        self._reanalysis_worker.progress_changed.connect(_on_progress)
        self._reanalysis_worker.finished_ok.connect(_on_done)

        # Animate progress bar smoothly with a timer while the thread runs
        _tick_val = [10]
        _anim_timer = QTimer(self)
        def _tick():
            if _tick_val[0] < 88:
                _tick_val[0] += 1
                progress.setValue(_tick_val[0])
                msg = ("Loading recorded data..." if _tick_val[0] < 30 else
                       "Running beat classification..." if _tick_val[0] < 55 else
                       "Computing HRV metrics..." if _tick_val[0] < 75 else
                       "Building summary...")
                status_lbl.setText(msg)
                pct_lbl.setText(f"{_tick_val[0]}%")
        _anim_timer.timeout.connect(_tick)
        _anim_timer.start(40)  # ~25 fps smooth fill

        self._reanalysis_worker.finished_ok.connect(_anim_timer.stop)
        self._reanalysis_worker.start()

    def _on_workspace_section_requested(self, section: str):
        key = (section or '').strip().lower()
        if key == 'quit':
            self.close()
        elif key in {'reanalysis', 'replay'}:
            self._confirm_reanalysis()
        elif key in {'overview'}:
            self._focus_tab('OVERVIEW')
        elif key in {'preview', 'view', 'report', 'edit report'}:
            self._focus_tab('PREVIEW')
        elif key == 'template':
            self._focus_tab('TEMPLATE')
        elif key == 'histogram':
            self._focus_tab('HISTOGRAM')
        elif key == 'lorenz':
            self._focus_tab('REPLAY')
        elif key == 'af analysis':
            self._focus_tab('AF ANALYSIS')
        elif key in {'tend. chart', 'st tendency'}:
            self._focus_tab('ST TENDENCY')
        elif key in {'edit event', 'pace spike', 'add event'}:
            self._focus_tab('EDIT EVENT')
        elif key == 'edit strips':
            self._focus_tab('EDIT STRIPS')
        elif key == 'report table':
            self._focus_tab('REPORT TABLE')
        elif key == 'hrv':
            self._focus_tab('HRV')
        elif key in {'record settings', 'advance tools'}:
            self._focus_tab('RECORDINGS')
        elif key == 'print':
            self._generate_report()
        else:
            self._focus_tab('REPLAY')

    # ----- Build UI -----------------------------------------------------------------------------

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
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        #  "  "  Top title bar  "  " 
        title_bar = QFrame()
        title_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL};border-bottom:1px solid {UI_BORDER};}}")
        title_bar.setFixedHeight(52)
        tb_layout = QHBoxLayout(title_bar)
        tb_layout.setContentsMargins(16, 0, 16, 0)
        tb_layout.setSpacing(14)
        self._mode_badge = QLabel("LIVE" if self._writer else "REVIEW")
        self._mode_badge.setStyleSheet(
            f"background:{'#7A2633' if self._writer else '#124936'};color:{UI_TEXT};"
            f"border:1px solid {UI_BORDER};border-radius:12px;padding:5px 12px;font-size:11px;font-weight:700;"
        )
        tb_layout.addWidget(self._mode_badge)
        app_title = QLabel("Comprehensive ECG Analysis")
        app_title.setStyleSheet(f"color:{UI_TEXT};font-size:18px;font-weight:700;border:none;")
        tb_layout.addWidget(app_title)
        dur_text = self._summary.get('duration_sec', 0)
        dur_h = int(dur_text // 3600)
        dur_m = int((dur_text % 3600) // 60)
        self._dur_label = QLabel(f"{dur_h:02d}h {dur_m:02d}m")
        self._dur_label.setStyleSheet(
            f"color:{UI_TEXT};font-size:13px;font-weight:700;background:{UI_PANEL_ALT};"
            f"padding:7px 12px;border-radius:8px;border:1px solid {UI_BORDER};"
        )
        tb_layout.addWidget(self._dur_label)
        tb_layout.addStretch()
        gen_report_btn = QPushButton("Generate Report")
        gen_report_btn.setStyleSheet(
            f"QPushButton{{background:{UI_ACCENT};color:{UI_TEXT};border:1px solid #61a8ff;border-radius:8px;padding:8px 14px;font-size:12px;font-weight:700;}}"
            f"QPushButton:hover{{background:{UI_ACCENT_HOVER};}}"
        )
        gen_report_btn.setFixedHeight(34)
        gen_report_btn.clicked.connect(self._generate_report)
        tb_layout.addWidget(gen_report_btn)
        close_btn = QPushButton("Close")
        close_btn.setStyleSheet(
            f"QPushButton{{background:{UI_PANEL_ALT};color:{UI_TEXT};border:1px solid {UI_BORDER};border-radius:8px;padding:8px 14px;font-size:12px;font-weight:600;}}"
            "QPushButton:hover{background:#2A3D61;}"
        )
        close_btn.setFixedHeight(34)
        close_btn.clicked.connect(self.close)
        tb_layout.addWidget(close_btn)
        main_layout.addWidget(title_bar)

        session_bar = QFrame()
        session_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL_ALT};border-bottom:1px solid {UI_BORDER};}}")
        sb_layout = QHBoxLayout(session_bar)
        sb_layout.setContentsMargins(14, 8, 14, 8)
        sb_layout.setSpacing(10)
        patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
        doctor_name = self.patient_info.get("doctor") or "No referring doctor"
        session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
        self._patient_chip = QLabel(f"Patient: {patient_name}")
        self._patient_chip.setStyleSheet(f"background:#123B2D;color:{UI_TEXT};border:1px solid #206B51;border-radius:14px;padding:6px 12px;font-size:11px;font-weight:700;")
        sb_layout.addWidget(self._patient_chip)
        self._doctor_chip = QLabel(f"Doctor: {doctor_name}")
        self._doctor_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._doctor_chip)
        self._session_chip = QLabel(f"Session: {session_name}")
        self._session_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._session_chip)
        self._rec_time_chip = QLabel("Recording: --")
        self._rec_time_chip.setStyleSheet(f"background:{UI_PANEL};color:{UI_MUTED};border:1px solid {UI_BORDER};border-radius:14px;padding:6px 12px;font-size:11px;font-weight:600;")
        sb_layout.addWidget(self._rec_time_chip)
        sb_layout.addStretch()
        self._analysis_state = QLabel("Clinical review mode")
        self._analysis_state.setStyleSheet(f"color:{UI_MUTED};font-size:11px;font-weight:600;border:none;")
        sb_layout.addWidget(self._analysis_state)
        main_layout.addWidget(session_bar)

        action_bar = QFrame()
        action_bar.setStyleSheet(f"QFrame{{background:{UI_PANEL};border-bottom:1px solid {UI_BORDER};}}")
        ab_layout = QHBoxLayout(action_bar)
        ab_layout.setContentsMargins(8, 6, 8, 6)
        ab_layout.setSpacing(6)
        self._action_buttons = {}
        for label in ["Browse", "Search", "Analyse", "View", "Import", "Backup", "Delete"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_style_btn())
            self._action_buttons[label] = btn
            ab_layout.addWidget(btn)
        ab_layout.addStretch()
        self._filter_buttons = {}
        for label in ["All", "Today", "Yesterday", "This Week", "This Month", "This Year"]:
            btn = QPushButton(label)
            btn.setFixedHeight(30)
            btn.setStyleSheet(_style_btn(UI_PANEL_ALT, UI_MUTED, "#1A2C49"))
            self._filter_buttons[label] = btn
            ab_layout.addWidget(btn)
        main_layout.addWidget(action_bar)

        # ----- Status bar (if recording) ------------------------------------------------------------
        if self._writer:
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            main_layout.addWidget(self._status_bar)

        # ----- Summary KPI cards ----------------------------------------------------------------------------

        # ----- Body: tabs fill full width (12-lead grid is inside HolterReplayPanel) -----------------------------------------
        right_frame = QFrame()
        right_frame.setStyleSheet(f"QFrame{{background:{UI_BG};}}")
        right_layout = QVBoxLayout(right_frame)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        self._tabs = QTabWidget()
        self._tabs.setDocumentMode(True)
        self._tabs.setUsesScrollButtons(True)

        # Make the tab bar scrollable with the mouse wheel instead of changing tabs
        class TabBarScroller(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Wheel:
                    # In QTabBar, there is a private QScrollArea, but sending the wheel event 
                    # as horizontal scroll to the tabbar doesn't always work cleanly.
                    # Alternatively, if we just want to suppress changing tabs, we can return True,
                    # but to scroll, we need to find the scroll buttons or adjust the scroll offset.
                    # A simple way to trigger the internal scroll is to post a wheel event to the 
                    # tab widget's internal scroll widget, but PyQt doesn't expose it easily.
                    # Wait, if usesScrollButtons is True, QTabBar has two QToolButtons as children.
                    # We can simulate clicks on them based on wheel direction.
                    delta = event.angleDelta().y()
                    if delta == 0:
                        delta = event.angleDelta().x()
                    if delta != 0:
                        buttons = obj.findChildren(QToolButton)
                        if len(buttons) >= 2:
                            # Usually button 0 is left, 1 is right
                            if delta > 0:
                                buttons[0].click()
                                buttons[0].click()
                            else:
                                buttons[1].click()
                                buttons[1].click()
                    return True # Consume event to prevent changing the selected tab
                return super().eventFilter(obj, event)

        self._tab_scroller = TabBarScroller(self)
        self._tabs.tabBar().installEventFilter(self._tab_scroller)

        self._tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                background:{UI_BG};
                border:1px solid {UI_BORDER};
                border-top:none;
            }}
            QTabBar::tab {{
                background:{UI_PANEL};
                color:{UI_MUTED};
                border:1px solid {UI_BORDER};
                border-radius:8px;
                padding:9px 14px;
                font-size:11px;
                font-weight:700;
                margin:6px 6px 8px 0;
                min-width:96px;
                text-align:center;
            }}
            QTabBar::tab:selected {{
                color:{UI_TEXT};
                background:{UI_ACCENT};
                border-color:#6EB4FF;
            }}
            QTabBar::tab:hover:!selected {{
                background:#1A2C49;
                color:{UI_TEXT};
            }}
            QGroupBox {{
                border: none;
                border-top: 1px solid {UI_BORDER};
                margin-top: 20px;
                font-weight: bold;
                font-size: 14px;
                color: {UI_TEXT};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 5px;
            }}
        """)

        #  "  "  OVERVIEW tab (12-lead scrollable expert view)  "  " 
        self._expert_panel = HolterExpertReviewPanel()
        self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        self._expert_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._expert_panel, "OVERVIEW")

        # Replay
        duration = self._summary.get('duration_sec', self._duration_hours * 3600)
        self._replay_panel = HolterReplayPanel(duration_sec=duration)
        if self._replay_engine:
            self._replay_panel.set_replay_engine(self._replay_engine)
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._replay_panel.section_requested.connect(self._on_workspace_section_requested)
        self._replay_panel.playback_state_changed.connect(self._set_record_browser_enabled)
        self._replay_panel.update_lorenz(self._metrics_list)
        self._replay_panel.update_summary(self._summary)
        self._tabs.addTab(self._replay_panel, "REPLAY")
        self._lorenz_panel = self._replay_panel

        # Beat Templates
        self._template_panel = HolterBeatTemplatePanel()
        if self._replay_engine:
            try:
                self._template_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
        self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        self._template_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._template_panel, "TEMPLATE")

        # Histogram
        self._hist_panel = HolterHistogramPanel()
        self._hist_panel.update_from_metrics(self._metrics_list)
        self._hist_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._hist_panel, "HISTOGRAM")

        # Lorenz
        self._lorenz_tab_panel = HolterLorenzPanel(replay_engine=self._replay_engine)
        self._lorenz_tab_panel.update_from_metrics(self._metrics_list)
        self._lorenz_tab_panel.seek_requested.connect(self._on_seek_requested)
        if self._replay_engine:
            self._lorenz_tab_panel.set_replay_engine(self._replay_engine)
        self._tabs.addTab(self._lorenz_tab_panel, "LORENZ")


        # AF Analysis
        self._af_panel = HolterAFPanel(session_dir=self.session_dir)
        self._af_panel.update_from_metrics(self._metrics_list, duration)
        self._tabs.addTab(self._af_panel, "AF ANALYSIS")

        events = self._build_linked_events()

        # Event timeline
        self._events_panel = HolterEventsPanel()
        self._events_panel.set_session_dir(self.session_dir)
        self._events_panel.load_events(events, self._summary)
        self._events_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._events_panel, "EVENTS")

        # ST Tendency
        self._st_panel = HolterSTPanel(replay_engine=self._replay_engine)
        self._st_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._st_panel, "ST TENDENCY")

        # Edit Event
        self._edit_event_panel = HolterEditEventPanel()
        self._edit_event_panel.set_session_dir(self.session_dir)
        self._edit_event_panel.load_events(events, self._summary)
        self._edit_event_panel.seek_requested.connect(self._on_seek_requested)
        self._tabs.addTab(self._edit_event_panel, "EDIT EVENT")

        # Edit Strips
        self._edit_strips_panel = HolterEditStripsPanel()
        self._edit_strips_panel.set_session_dir(self.session_dir)
        self._edit_strips_panel.seek_requested.connect(self._on_seek_requested)
        self._edit_strips_panel.load_events(events, self._summary, self._metrics_list)
        self._tabs.addTab(self._edit_strips_panel, "EDIT STRIPS")

        # Report Tendency
        self._report_tendency_panel = HolterSTPanel(replay_engine=self._replay_engine)
        self._tabs.addTab(self._report_tendency_panel, "REPORT TENDENCY")

        # Report Table
        self._report_table_panel = HolterReportTablePanel()
        self._report_table_panel.update_from_metrics(self._metrics_list)
        self._tabs.addTab(self._report_table_panel, "REPORT TABLE")

        # Expert Review (OVERVIEW tab) already added as first tab above
        # kept here as comment for clarity   " see OVERVIEW tab creation at top of tabs section

        # HRV Analysis
        self._hrv_panel = HolterHRVPanel()
        self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        self._tabs.addTab(self._hrv_panel, "HRV")

        # Record browser
        self._record_mgmt_panel = HolterRecordManagementPanel(
            output_dir=_resolve_recordings_dir(self.session_dir)
        )
        self._record_mgmt_panel.session_selected.connect(self.load_completed_session)
        if self.session_dir:
            self._record_mgmt_panel.set_active_session(self.session_dir)
        self._tabs.addTab(self._record_mgmt_panel, "RECORDINGS")

        # Report Preview
        scroll_insight = QScrollArea()
        scroll_insight.setWidgetResizable(True)
        scroll_insight.setFrameShape(QFrame.NoFrame)
        scroll_insight.setStyleSheet(f"QScrollArea{{background:{COL_BLACK};border:none;}}")
        self._insight_panel = HolterInsightPanel()
        self._insight_panel.update_text(self.patient_info, self._summary)
        scroll_insight.setWidget(self._insight_panel)
        self._tabs.addTab(scroll_insight, "PREVIEW")
        self._tabs.addTab(QWidget(), "PRINT")
        self._tabs.addTab(QWidget(), "REANALYSIS")
        self._tabs.addTab(QWidget(), "QUIT")

        # Track the last active content tab
        self._last_active_tab_name = "OVERVIEW"
        def _on_tab_changed(index):
            tab_name = self._tabs.tabText(index)
            if tab_name in {"PRINT", "REANALYSIS", "QUIT"}:
                if hasattr(self, "_last_active_tab_name"):
                    self._focus_tab(self._last_active_tab_name)
                if tab_name == "PRINT":
                    QTimer.singleShot(0, self._generate_report)
                elif tab_name == "REANALYSIS":
                    QTimer.singleShot(0, self._confirm_reanalysis)
                elif tab_name == "QUIT":
                    QTimer.singleShot(0, self.close)
            else:
                self._last_active_tab_name = tab_name
                if tab_name == "RECORDINGS":
                    self._tabs.tabBar().setVisible(False)
                else:
                    self._tabs.tabBar().setVisible(True)
        self._tabs.currentChanged.connect(_on_tab_changed)
        # Ensure initial state is correct
        _on_tab_changed(self._tabs.currentIndex())

        right_layout.addWidget(self._tabs)
        self._tabs.currentChanged.connect(
            lambda idx: hasattr(self, '_analysis_state') and self._analysis_state.setText(
                f"Focused view: {self._tabs.tabText(idx)}"
            )
        )
        self._action_buttons["Browse"].clicked.connect(self._open_recordings_folder)
        self._action_buttons["Search"].clicked.connect(self._search_recordings)
        self._action_buttons["Analyse"].clicked.connect(lambda: self._focus_tab("REPLAY"))
        self._action_buttons["View"].clicked.connect(lambda: self._focus_tab("PREVIEW"))
        self._action_buttons["Import"].clicked.connect(self._import_recording)
        self._action_buttons["Backup"].clicked.connect(self._backup_recordings)
        self._action_buttons["Delete"].clicked.connect(self._delete_recording)
        for label, btn in self._filter_buttons.items():
            btn.clicked.connect(lambda _, t=label: self._apply_recordings_filter(t))

        main_layout.addWidget(right_frame, 1)
        if hasattr(self, '_analysis_state'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")


    # ----- Callbacks ----------------------------------------------------------------------------

    def _current_replay_window_sec(self) -> float:
        panel = getattr(self, '_replay_panel', None)
        if panel is not None:
            try:
                return float(getattr(panel, '_strip_length_sec', 10.0) or 10.0)
            except Exception:
                pass
        return float(getattr(self, '_strip_length_sec', 10.0) or 10.0)

    def _sync_replay_window_length(self):
        if getattr(self, '_replay_engine', None) and hasattr(self._replay_engine, 'set_window_length'):
            try:
                self._replay_engine.set_window_length(self._current_replay_window_sec())
            except Exception:
                pass

    def _on_seek_requested(self, target_sec: float):
        if self._replay_engine:
            self._sync_replay_window_length()
            self._replay_engine.seek(target_sec)
            try:
                # Use the replay panel's current strip length (changes with paper speed)
                window_sec = self._current_replay_window_sec()
                data = self._replay_engine.get_all_leads_data(window_sec=float(window_sec))
                beat_annotations = self._replay_engine.get_beat_annotations(window_sec=float(window_sec))
                start_sec = self._replay_engine.current_position() - window_sec / 2
                if hasattr(self, '_wave_panel'):
                    self._wave_panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
                self._broadcast_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
            except Exception:
                pass


    def _broadcast_replay_frame(self, data, beat_annotations=None, start_sec=0.0):
        for panel in [getattr(self, p, None) for p in [
            '_replay_panel', '_lorenz_panel', '_hist_panel', '_af_panel',
            '_st_panel', '_edit_event_panel', '_edit_strips_panel', '_events_panel',
            '_expert_panel', '_template_panel', '_report_tendency_panel', '_hrv_panel',
            '_lorenz_tab_panel'
        ]]:
            if panel and hasattr(panel, 'set_replay_frame'):
                try:
                    panel.set_replay_frame(data, beat_annotations=beat_annotations, start_sec=start_sec)
                except TypeError:
                    # Fallback for panels that don't accept beat_annotations
                    try:
                        panel.set_replay_frame(data)
                    except Exception:
                        pass
                except Exception:
                    pass

    def _build_linked_events(self) -> list:
        events = []
        if self._replay_engine:
            try:
                events.extend(self._replay_engine.get_events_list() or [])
            except Exception:
                pass
        for metric in self._metrics_list or []:
            base_t = float(metric.get('t', 0.0) or 0.0)
            for label in metric.get('arrhythmias', []) or []:
                events.append({
                    'timestamp': base_t,
                    'label': str(label),
                    'time_str': _sec_to_hms(base_t),
                })
            for ev in metric.get('classified_events', []) or []:
                t_val = float(ev.get('timestamp', base_t) or base_t)
                events.append({
                    'timestamp': t_val,
                    'label': str(ev.get('label', ev.get('template_label', 'Beat Event'))),
                    'time_str': _sec_to_hms(t_val),
                })

        events.sort(key=lambda e: float(e.get('timestamp', 0.0) or 0.0))
        dedup = []
        seen = set()
        for ev in events:
            key = (round(float(ev.get('timestamp', 0.0) or 0.0), 3), str(ev.get('label', '')))
            if key in seen:
                continue
            seen.add(key)
            dedup.append(ev)
        return dedup

    def _update_live_ui(self):
        if not self._writer or not self._writer.is_running:
            if hasattr(self, '_live_timer'):
                self._live_timer.stop()
            self._load_session()
            self._refresh_ui()
            return
        stats = self._writer.get_live_stats()
        if hasattr(self, '_status_bar'):
            self._status_bar.update_stats(stats['bpm'], stats['arrhythmias'])
        snapshot = None
        if hasattr(self._writer, 'get_live_analysis_snapshot'):
            snapshot = self._writer.get_live_analysis_snapshot(getattr(self, '_last_live_seq', -1))
        if snapshot:
            self._last_live_seq = snapshot.get('seq', self._last_live_seq)
            self._metrics_list = snapshot.get('metrics', [])
            self._summary = snapshot.get('summary', {})
            self._refresh_ui()
        if hasattr(self, '_wave_panel'):
            self._wave_panel.refresh_waveforms()
        # Also refresh _replay_panel lead strips from live data on every tick
        if hasattr(self, '_replay_panel') and not self._replay_engine and self._live_source is not None:
            try:
                raw = getattr(self._live_source, 'data', None)
                if raw is not None and hasattr(raw, '__len__') and len(raw) > 0:
                    import numpy as _np
                    n_samp = max(len(raw[i]) for i in range(min(12, len(raw))))
                    arr = _np.full((12, n_samp), 2048.0)
                    for i in range(min(12, len(raw))):
                        ch = _np.asarray(raw[i], dtype=float)
                        arr[i, :len(ch)] = ch
                    self._replay_panel.set_replay_frame(arr)
            except Exception:
                pass
        if snapshot is None and stats['elapsed'] % 15 < 2:
            self._load_session()
            self._refresh_ui()

    def _refresh_ui(self):
        if hasattr(self, '_summary_cards'):
            self._summary_cards.update_summary(self._summary)
        if hasattr(self, '_insight_panel'):
            self._insight_panel.update_text(self.patient_info, self._summary)
        if hasattr(self, '_overview_panel'):
            self._overview_panel.update_summary(self._summary)
        if hasattr(self, '_expert_panel'):
            self._expert_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_hrv_panel'):
            self._hrv_panel.update_hrv(self._metrics_list, self._summary)
        if hasattr(self, '_replay_panel'):
            self._replay_panel.update_lorenz(self._metrics_list)
            self._replay_panel.update_summary(self._summary)
            # During live recording (no replay engine), push current live ECG frame
            if not self._replay_engine and self._live_source is not None:
                try:
                    raw = getattr(self._live_source, 'data', None)
                    if raw is not None and hasattr(raw, '__len__') and len(raw) > 0:
                        import numpy as _np
                        # Build a 12-channel array from live_source.data
                        n_samp = max(len(raw[i]) for i in range(min(12, len(raw))))
                        arr = _np.full((12, n_samp), 2048.0)
                        for i in range(min(12, len(raw))):
                            ch = _np.asarray(raw[i], dtype=float)
                            arr[i, :len(ch)] = ch
                        self._replay_panel.set_replay_frame(arr)
                except Exception:
                    pass
        if hasattr(self, '_hist_panel'):
            self._hist_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_lorenz_tab_panel'):
            self._lorenz_tab_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_af_panel'):
            self._af_panel.update_from_metrics(self._metrics_list, self._summary.get('duration_sec', 0))
        if hasattr(self, '_st_panel'):
            self._st_panel.update_from_metrics(self._metrics_list)
        if hasattr(self, '_report_table_panel'):
            self._report_table_panel.update_from_metrics(self._metrics_list)
        events = self._build_linked_events()
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
            self._events_panel.load_events(events, self._summary)
        if hasattr(self, '_template_panel'):
            self._template_panel.update_from_metrics(self._metrics_list, self._summary)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
            self._edit_event_panel.load_events(events, self._summary)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
            self._edit_strips_panel.load_events(events, self._summary, self._metrics_list)
        if hasattr(self, '_wave_panel'):
            self._wave_panel.set_live_source(self._live_source)
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(self.session_dir)
            self._record_mgmt_panel.refresh_records()
        if hasattr(self, '_wave_panel') and self._replay_engine:
            try:
                self._wave_panel.set_replay_engine(self._replay_engine)
            except Exception:
                pass
            self._wave_panel.refresh_waveforms()
        if hasattr(self, '_expert_panel') and self._replay_engine:
            try:
                self._expert_panel.set_replay_frame(self._replay_engine.get_all_leads_data(window_sec=self._current_replay_window_sec()))
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._broadcast_replay_frame(self._replay_engine.get_all_leads_data(window_sec=self._current_replay_window_sec()))
            except Exception:
                pass

        # Update duration label
        dur = self._summary.get('duration_sec', 0)
        dur_h = int(dur // 3600)
        dur_m = int((dur % 3600) // 60)
        if hasattr(self, '_dur_label'):
            self._dur_label.setText(f"{dur_h:02d}h {dur_m:02d}m")
        if hasattr(self, '_patient_chip'):
            patient_name = self.patient_info.get("patient_name") or self.patient_info.get("name") or "Unknown Patient"
            self._patient_chip.setText(f"Patient: {patient_name}")
        if hasattr(self, '_doctor_chip'):
            doctor_name = self.patient_info.get("doctor") or "No referring doctor"
            self._doctor_chip.setText(f"Doctor: {doctor_name}")
        if hasattr(self, '_session_chip'):
            session_name = os.path.basename(self.session_dir) if self.session_dir else "Active Session"
            self._session_chip.setText(f"Session: {session_name}")
        if hasattr(self, '_rec_time_chip'):
            st_str, end_str = _get_recording_start_end_times(self.session_dir, dur)
            if st_str != "Unknown":
                self._rec_time_chip.setText(f"Recording: {st_str} to {end_str}")
                self._rec_time_chip.show()
            else:
                self._rec_time_chip.hide()
        if hasattr(self, '_analysis_state') and hasattr(self, '_tabs'):
            self._analysis_state.setText(f"Focused view: {self._tabs.tabText(self._tabs.currentIndex())}")

    def _finalize_live_writer(self) -> dict:
        summary = {}
        if not self._writer:
            return summary
        try:
            stop_fn = getattr(self._writer, "stop", None)
            if callable(stop_fn):
                summary = stop_fn() or {}
            else:
                close_fn = getattr(self._writer, "close", None)
                if callable(close_fn):
                    summary = close_fn() or {}
        finally:
            self._writer = None
        return summary

    def _stop_recording(self):
        if self._writer:
            summary = self._finalize_live_writer()
            if hasattr(self, '_status_bar') and self._status_bar is not None:
                self._status_bar.setVisible(False)
                if hasattr(self._status_bar, 'cleanup'):
                    self._status_bar.cleanup()
            
            # Show dialog to collect patient info AFTER recording
            dialog = HolterStartDialog(self, patient_info=self.patient_info or {}, output_dir=self.session_dir)
            dialog.setWindowTitle("Save Comprehensive ECG Analysis Recording Details")
            if dialog.exec_() == QDialog.Accepted:
                patient_info, dur, out_dir = dialog.get_result()
                summary['patient_info'] = patient_info
                self.patient_info = patient_info
                import json
                try:
                    with open(os.path.join(summary.get('session_dir', ''), "patient.json"), 'w') as f:
                        json.dump(patient_info, f, indent=4)
                except Exception as e:
                    print(f"Failed to save patient.json: {e}")

            QMessageBox.information(self, "Recording Complete",
                                    f"Comprehensive ECG Analysis recording saved to:\n{summary.get('session_dir', '')}")
            self.load_completed_session(summary.get('session_dir', ''), summary.get('patient_info', {}))
            
            # Auto-generate report when recording is stopped
            self._generate_report()

    def _generate_report(self):
        from PyQt5.QtWidgets import QProgressDialog
        from PyQt5.QtCore import QThread, pyqtSignal
        
        progress = QProgressDialog("Generating Comprehensive ECG Analysis Report. Please wait...", None, 0, 0, self)
        progress.setWindowTitle("Please Wait")
        progress.setWindowModality(Qt.WindowModal)
        progress.setStyleSheet(f"QProgressDialog{{background:{COL_DARK};color:{COL_GREEN};}}")
        progress.setRange(0, 0)
        progress.show()
        
        class ReportWorker(QThread):
            finished = pyqtSignal(str)
            error = pyqtSignal(str)
            
            def __init__(self, session_dir, patient_info, summary):
                super().__init__()
                self.session_dir = session_dir
                self.patient_info = patient_info
                self.summary = summary
                
            def run(self):
                try:
                    from .report_generator import generate_holter_report
                    path = generate_holter_report(
                        session_dir=self.session_dir,
                        patient_info=self.patient_info,
                        summary=self.summary,
                    )
                    self.finished.emit(path)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    self.error.emit(str(e))
                    
        self._report_worker = ReportWorker(self.session_dir, self.patient_info, self._summary)
        
        def on_finished(path):
            progress.close()
            try:
                from dashboard.history_window import append_history_entry
                h_pat = self.patient_info.copy() if self.patient_info else {}
                if 'patient_name' not in h_pat and 'name' in h_pat:
                    h_pat['patient_name'] = h_pat['name']
                _p = self.parent()
                _uname = getattr(_p, "username", "") if _p is not None else ""
                _full = (getattr(_p, "user_details", {}) or {}).get("full_name") or _uname
                append_history_entry(
                    h_pat, path, report_type="Comprehensive ECG Analysis",
                    username=_uname, owner_full_name=_full
                )
            except Exception as h_err:
                print(f"Failed to append Holter history: {h_err}")
                
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Generated")
            msg.setText(f"Comprehensive ECG Analysis report saved:\n{path}")
            msg.setIcon(QMessageBox.Information)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        def on_error(err_str):
            progress.close()
            msg = QMessageBox(self)
            msg.setWindowTitle("Report Error")
            msg.setText(f"Could not generate report:\n{err_str}")
            msg.setIcon(QMessageBox.Warning)
            msg.setStyleSheet(f"QMessageBox {{ background: {COL_BLACK}; }} QLabel {{ color: {COL_WHITE}; font-size: 12px; }}")
            msg.exec_()
            
        self._report_worker.finished.connect(on_finished)
        self._report_worker.error.connect(on_error)
        self._report_worker.start()

    def attach_writer(self, writer, session_dir: str = "", patient_info: dict = None):
        self._writer = writer
        self._last_live_seq = -1
        if session_dir:
            self.session_dir = session_dir
        if patient_info is not None:
            self.patient_info = _normalize_patient_info(patient_info)
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
        if writer and not hasattr(self, '_status_bar'):
            self._status_bar = HolterStatusBar(self, target_hours=self._duration_hours)
            self._status_bar.stop_requested.connect(self._stop_recording)
            self.layout().insertWidget(1, self._status_bar)
        if writer and not hasattr(self, '_live_timer'):
            self._live_timer = QTimer(self)
            self._live_timer.timeout.connect(self._update_live_ui)
        if writer and hasattr(self, '_live_timer') and not self._live_timer.isActive():
            self._live_timer.start(1000)
        self._refresh_ui()

    def load_completed_session(self, session_dir: str, patient_info: dict = None):
        self.session_dir = session_dir
        self._last_live_seq = -1
        self.patient_info = _normalize_patient_info(patient_info or {})
        if hasattr(self, '_edit_event_panel'):
            self._edit_event_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_events_panel'):
            self._events_panel.set_session_dir(self.session_dir)
        if hasattr(self, '_edit_strips_panel'):
            self._edit_strips_panel.set_session_dir(self.session_dir)
        if getattr(self, "_replay_engine", None) and self._replay_engine.is_playing():
            self._replay_engine.pause()
        if hasattr(self, "_replay_panel"):
            try:
                self._replay_panel._slider.blockSignals(True)
                self._replay_panel._slider.setValue(self._replay_panel._slider_sec_to_value(0))
                self._replay_panel._slider.blockSignals(False)
                self._replay_panel._pos_label.setText(_sec_to_hms(0))
                self._replay_panel._play_btn.setText("Play")
            except Exception:
                pass
            self._set_record_browser_enabled(True)
        self._writer = None
        self._load_session()
        if hasattr(self, '_record_mgmt_panel'):
            self._record_mgmt_panel.output_dir = _resolve_recordings_dir(session_dir)
            self._record_mgmt_panel.set_active_session(self.session_dir)
        if hasattr(self, '_replay_panel') and getattr(self, '_replay_engine', None):
            self._replay_panel.set_replay_engine(self._replay_engine)
            try:
                self._replay_panel.seek_requested.disconnect(self._on_seek_requested)
            except Exception:
                pass
            self._replay_panel.seek_requested.connect(self._on_seek_requested)
        self._refresh_ui()
        if hasattr(self, '_tabs'):
            self._tabs.setCurrentIndex(0)

    def close_current_session_if_needed(self, session_to_delete: str) -> bool:
        """
        Closes the current session if it matches the session being deleted.
        Returns True if the session was closed.
        """
        if not hasattr(self, 'session_dir') or not self.session_dir:
            return False
        if not os.path.normpath(self.session_dir) == os.path.normpath(session_to_delete):
            return False
        if hasattr(self, "_replay_engine") and self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
            self._replay_engine = None
        self.session_dir = None
        return True

    def changeEvent(self, event):
        """Handle window state changes to preserve maximized state."""
        super().changeEvent(event)
        if event.type() == QEvent.WindowStateChange:
            # When window state changes, re-apply maximized state
            if self.isVisible():
                QTimer.singleShot(100, self.showMaximized)

    def showEvent(self, event):
        """Ensure window maintains maximized state when shown."""
        super().showEvent(event)
        # Restore maximized state when window is shown
        QTimer.singleShot(50, self.showMaximized)

    def closeEvent(self, event):
        if self._writer:
            try:
                self._finalize_live_writer()
            except Exception:
                pass
        if self._replay_engine:
            try:
                self._replay_engine.close()
            except Exception:
                pass
        super().closeEvent(event)
