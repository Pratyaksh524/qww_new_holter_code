"""
Hyperkalemia Detection Module - ECG-based hyperkalemia detection according to medical standards
This module provides a dedicated window for hyperkalemia testing with automatic ECG analysis.
"""

import sys
import os
import time
import json
import numpy as np
from collections import deque
from datetime import datetime
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox,
    QSizePolicy, QFrame, QGridLayout, QApplication
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
# import pyqtgraph as pg  # Lazy loaded in methods
from scipy.signal import find_peaks, butter, filtfilt
from scipy.ndimage import gaussian_filter1d

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - Hyperkalemia test hardware features disabled")
    SERIAL_AVAILABLE = False
    class Serial:
        def __init__(self, *args, **kwargs): pass
        def close(self): pass
        def readline(self): return b''
        def write(self, data): pass
        def reset_input_buffer(self): pass
    class SerialException(Exception): pass
    serial = type('Serial', (), {'Serial': Serial, 'SerialException': SerialException})()
    class MockComports:
        @staticmethod
        def comports(*args, **kwargs):
            return []
    serial.tools = type('Tools', (), {'list_ports': MockComports()})()

from utils.settings_manager import SettingsManager
from utils.crash_logger import get_crash_logger
from utils.patient_profile import resolve_patient_profile
from utils.app_paths import data_file
from utils.platform_compat import is_low_spec_mode
from ecg.ui import display_updates as shared_display_updates
from dashboard.history_window import append_history_entry
from ecg.lead_off_detection import detect_lead_off

# Import ECGTestPage + helpers to reuse EXACT same calculation + smoothing as 12‑lead test
try:
    from ecg.twelve_lead_test import ECGTestPage, SamplingRateCalculator, SerialStreamReader
    from PyQt5.QtWidgets import QStackedWidget
    from ecg.hyperkalemia_ecg_report_generator import generate_hyperkalemia_report
    ECG_TEST_AVAILABLE = True
except ImportError:
    ECG_TEST_AVAILABLE = False
    print(" ECGTestPage not available - using fallback calculations")


def create_pink_grid_brush():
    from PyQt5.QtGui import QBrush, QColor
    return QBrush(QColor("#ffffff"))

    from PyQt5.QtGui import QBrush, QPixmap, QPainter, QPen, QColor
    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    app = QApplication.instance()
    dpi = 96.0
    try:
        screen = app.primaryScreen() if app else None
        if screen is not None:
            dpi = float(screen.logicalDotsPerInchX() or screen.logicalDotsPerInch() or dpi)
    except Exception:
        dpi = 96.0

    px_per_mm = max(1.0, dpi / 25.4)
    minor_step = max(2, int(round(px_per_mm)))
    grid_size = minor_step * 5
    pixmap = QPixmap(grid_size, grid_size)
    pixmap.fill(QColor('#FFF5F5'))
    
    painter = QPainter(pixmap)
    try:
        painter.setRenderHint(QPainter.Antialiasing, False)
        # Minor lines every 1 mm
        minor_pen = QPen(QColor('#FFD9D9'), 0.5, Qt.SolidLine)
        painter.setPen(minor_pen)
        for i in range(1, 5):
            x = i * minor_step
            painter.drawLine(x, 0, x, grid_size)
            painter.drawLine(0, x, grid_size, x)
            
        # Major lines every 5 mm
        major_pen = QPen(QColor('#FFB3B3'), 1.0, Qt.SolidLine)
        painter.setPen(major_pen)
        painter.drawLine(0, 0, grid_size, 0)
        painter.drawLine(0, 0, 0, grid_size)
        painter.drawLine(grid_size - 1, 0, grid_size - 1, grid_size)
        painter.drawLine(0, grid_size - 1, grid_size, grid_size - 1)
    finally:
        painter.end()
        
    return QBrush(pixmap)


class HyperkalemiaTestWindow(QWidget):
    """Hyperkalemia Detection Window - ECG analysis for hyperkalemia detection"""
    
    def __init__(self, parent=None, username=None):
        super().__init__(parent)

        # Full screen on open
        self.showFullScreen()

        # Only show close button, disable minimize/maximize/restore
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint | Qt.MSWindowsFixedSizeDialogHint)
        # Prevent window from being moved/draggable
        self.setWindowFlag(Qt.WindowTitleHint, True)

        self.dashboard_instance = parent  # Store reference to dashboard
        self.username = username
        self.setWindowTitle("Hyperkalemia Detection Test")
        try:
            screen = QApplication.primaryScreen().availableGeometry()
            width = int(screen.width() * 0.90)
            height = int(screen.height() * 0.90)
            self.resize(width, height)
            
            # Center the window
            x = (screen.width() - width) // 2
            y = (screen.height() - height) // 2
            self.move(x, y)
        except Exception:
            # Fallback if screen geometry fails
            self.resize(1400, 1000)
            
        self.setMinimumSize(1024, 768)
        # Set window flags to make it a separate window
        # self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Data storage - use circular buffer like 12-lead test
        HISTORY_LENGTH = 10000
        self.data = np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32)  # Circular buffer for Lead II
        self.lead_ii_data = []  # Store all captured data with timestamps
        
        # Add data storage for V1-V6 leads (indices 6-11 in ecg_calculator.data)
        self.lead_data = {
            'II': [],  # Lead II
            'V1': [],  # V1
            'V2': [],  # V2
            'V3': [],  # V3
            'V4': [],  # V4
            'V5': [],  # V5
            'V6': []   # V6
        }

        # Plot buffers (bounded) to keep UI responsive even if capture lists grow.
        # Use ADC center line (2048) as flat line when a lead is detected OFF.
        self._adc_center = 2048.0
        self._plot_buffers = {}  # lead -> deque[float]
        self._plot_seconds = {}  # lead -> float seconds shown
        self._lead_off_windows = {}  # lead -> deque[float] (recent 1s window)
        self._lead_off_state = {}  # lead -> bool
        self._last_packet_time = 0.0
        self._plot_update_in_progress = False
        self._plot_render_stride = 4 if is_low_spec_mode() else 2

        
        # Lead mapping: name -> index in ecg_calculator.data
        self.lead_indices = {
            'I': 0, 'II': 1, 'III': 2, 'aVR': 3, 'aVL': 4, 'aVF': 5,
            'V1': 6, 'V2': 7, 'V3': 8, 'V4': 9, 'V5': 10, 'V6': 11
        }

        # Track lead connection like 12-lead (prevents one bad lead from freezing all updates)
        self._lead_last_valid_value = {name: self._adc_center for name in self.lead_indices.keys()}
        self._lead_connection_state = {name: True for name in self.lead_indices.keys()}
        
        self.start_time = None
        self.capture_duration = 30  # 30 seconds for hyperkalemia detection
        self.is_capturing = False
        self.serial_reader = None
        self.crash_logger = get_crash_logger()
        
        # For adaptive scaling per lead
        self.y_centers = {lead: 0.0 for lead in self.lead_data.keys()}
        self.y_ranges = {lead: 200.0 for lead in self.lead_data.keys()}
        
        # Backward compatibility
        self.y_center = 0.0
        self.y_range = 200.0  # Initial range
        self.sampling_rate = 500.0  # Default sampling rate, will be estimated
        self.sample_index = 0
        
        # Settings
        self.settings_manager = SettingsManager()
        
        # Track active sample count to avoid skewing stats with leading zeros
        self.active_samples = 0

        # Scrolling display window per lead (same flow as 12-lead, no raster sweep)
        self._display_window_sec = {}

        # Store last displayed metrics so analysis dialog matches dashboard values
        self.last_metrics = {}
        
        # Create a minimal ECGTestPage instance to reuse its calculation methods
        # This ensures we use the EXACT same functions as the 12-lead test
        self.ecg_calculator = None
        if ECG_TEST_AVAILABLE:
            try:
                # Create a dummy stacked widget for ECGTestPage initialization
                dummy_stack = QStackedWidget()
                self.ecg_calculator = ECGTestPage("12 Lead ECG Test", dummy_stack, settings_manager=self.settings_manager)
                
                # IMPORTANT: Sync sampling rate from parent dashboard if available
                if parent and hasattr(parent, 'ecg_test_page'):
                    p_page = parent.ecg_test_page
                    if hasattr(p_page, 'sampler') and p_page.sampler.sampling_rate > 0:
                        self.sampling_rate = p_page.sampler.sampling_rate
                        print(f" Synced sampling rate from dashboard: {self.sampling_rate} Hz")
                
                if not hasattr(self.ecg_calculator, 'data') or len(self.ecg_calculator.data) < 12:
                    self.ecg_calculator.data = [np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32) for _ in range(12)]
                
                if not hasattr(self.ecg_calculator, 'sampler'):
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                
                # CRITICAL: unique instance_id prevents sharing 12-lead smoothing buffers
                self.ecg_calculator._instance_id = 'hyperkalemia_test'

                print(" ECG calculator initialized for Hyperkalemia test")
            except Exception as e:
                print(f" Could not create ECG calculator: {e}")
                import traceback
                traceback.print_exc()
                self.ecg_calculator = None
        
        # ── HolterBPMController: stable BPM engine (background thread) ─────────
        try:
            from ecg.holter.holter_bpm_engine import HolterBPMController
            self._bpm_ctrl = HolterBPMController(
                parent_widget=self,
                fs=500,
                chunk_seconds=10,
            )
        except Exception as _e:
            print(f"[HyperkalemiaTestWindow] HolterBPMController init failed: {_e}")
            self._bpm_ctrl = None

        # Initialize UI
        self.init_ui()
        self._last_displayed_bpm = 0

        # Init bounded plotting + lead-off detection buffers
        self._init_plot_and_lead_off_buffers()
        
        # Timers
        self.capture_timer = QTimer(self)
        self.capture_timer.setTimerType(Qt.PreciseTimer)
        self.capture_timer.timeout.connect(self.update_plot)
        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self.check_duration)

    # ── Lead II history ────────────────────────────────────────────────────────
    # Written on every sample but only ever consumed as a whole array, so it is
    # held as an index-addressed ring. Writing used to be np.roll() per sample,
    # which copies the entire 10,000-element buffer 500 times a second — about
    # 5 million element moves per second for a buffer nothing reads during
    # capture. The chronological view is now rebuilt only when asked for.
    @property
    def data(self):
        ring = self._lead_ii_ring
        pos = int(self._lead_ii_pos)
        if pos == 0:
            return ring.copy()
        return np.concatenate((ring[pos:], ring[:pos]))

    @data.setter
    def data(self, value):
        arr = np.asarray(value, dtype=np.float32)
        self._lead_ii_ring = arr.copy()
        self._lead_ii_pos = 0

    def _push_lead_ii(self, value):
        """Append one Lead II sample in O(1)."""
        ring = self._lead_ii_ring
        ring[self._lead_ii_pos] = value
        self._lead_ii_pos = (self._lead_ii_pos + 1) % ring.size

    def _ring_push(self, lead_name, value):
        """Append one display sample for a lead in O(1)."""
        ring = self._ring_bufs.get(lead_name)
        if ring is None:
            return
        pos = int(self._ring_pos.get(lead_name, 0))
        ring[pos] = value
        self._ring_pos[lead_name] = (pos + 1) % ring.size
        self._ring_active[lead_name] = min(ring.size, self._ring_active.get(lead_name, 0) + 1)

    def _ring_tail(self, lead_name, count):
        """Most recent `count` samples for a lead, oldest first."""
        ring = self._ring_bufs.get(lead_name)
        if ring is None or count <= 0:
            return np.empty(0, dtype=float)
        count = int(min(count, ring.size))
        pos = int(self._ring_pos.get(lead_name, 0))
        start = pos - count
        if start >= 0:
            return np.asarray(ring[start:pos], dtype=float)
        return np.asarray(np.concatenate((ring[start:], ring[:pos])), dtype=float)

    # Display constants mirrored from the 12-lead test so both windows render a
    # trace the same way (twelve_lead_test.py: SMOOTH_SIGMA, INTERP_FACTOR,
    # _baseline_alpha_slow).
    _SMOOTH_SIGMA = 0.8
    _BASELINE_ALPHA_SLOW = 0.0005
    # Plot every 2nd sample: 250 Hz effective, still ten times the 25 Hz filter
    # cutoff, and about two points per pixel column in a 900 px lane.
    _DISPLAY_DECIMATION = 2

    @staticmethod
    def _decimate_for_display(signal, factor):
        """
        Thin the trace for plotting without losing peaks.

        The last sample is always kept so the newest data stays pinned to the
        right edge of the scrolling window.
        """
        try:
            arr = np.asarray(signal, dtype=float)
            if arr.size < 4 or factor <= 1:
                return arr
            thinned = arr[::int(factor)]
            if thinned.size and thinned[-1] != arr[-1]:
                thinned = np.append(thinned, arr[-1])
            return thinned
        except Exception:
            return np.asarray(signal, dtype=float)

    def _apply_lead_off_verdict(self, lead_name, looks_off):
        """
        Latch a lead-off verdict only after it holds for a sustained stretch.

        Blanking a lead replaces its samples with a constant in the display, the
        analysis buffer and the recording, so momentary agreement from the
        detector — a motion artifact, a noisy beat — must not be enough to do it.
        """
        engage = int(getattr(self, "_LEAD_OFF_ENGAGE_PACKETS", 250))
        release = int(getattr(self, "_LEAD_OFF_RELEASE_PACKETS", 100))
        currently_off = bool(self._lead_off_state.get(lead_name, False))

        if looks_off:
            self._lead_off_off_count[lead_name] = 0
            self._lead_off_on_count[lead_name] = self._lead_off_on_count.get(lead_name, 0) + 1
            if not currently_off and self._lead_off_on_count[lead_name] >= engage:
                self._lead_off_state[lead_name] = True
        else:
            self._lead_off_on_count[lead_name] = 0
            self._lead_off_off_count[lead_name] = self._lead_off_off_count.get(lead_name, 0) + 1
            if currently_off and self._lead_off_off_count[lead_name] >= release:
                self._lead_off_state[lead_name] = False

    def _init_plot_and_lead_off_buffers(self):
        # Keep a slightly larger buffer than the visible window for stable filtering.
        fs = float(self.sampling_rate or 500.0)
        if fs < 100.0 or fs > 1000.0:
            fs = 500.0

        self._plot_seconds = {}
        self._plot_buffers = {}
        for lead_name in self.lead_data.keys():
            seconds = 6.0
            self._plot_seconds[lead_name] = float(seconds)
            max_seconds = seconds + 2.0  # margin for filtering
            maxlen = int(max_seconds * fs)
            if maxlen < 200:
                maxlen = 200
            self._plot_buffers[lead_name] = deque(maxlen=maxlen)

        self._lead_off_windows = {}
        self._lead_off_state = {}
        # Debounce counters. A lead-off verdict blanks the lead everywhere — screen,
        # analysis buffer and the recording the report is drawn from — so a single
        # detector window must not be able to trigger it. The 12-lead test latches
        # its verdict the same way, which is why it keeps showing waves in
        # situations where this window used to go flat.
        self._lead_off_on_count = {}
        self._lead_off_off_count = {}
        window_len = int(1.0 * fs)
        if window_len < 50:
            window_len = 50
        # Engage after half a second of continuous agreement; release after a fifth
        # of a second, so a lead that comes back is shown again almost immediately.
        self._LEAD_OFF_ENGAGE_PACKETS = max(50, int(0.5 * fs))
        self._LEAD_OFF_RELEASE_PACKETS = max(20, int(0.2 * fs))
        for lead_name in self.lead_indices.keys():
            self._lead_off_windows[lead_name] = deque(maxlen=window_len)
            self._lead_off_state[lead_name] = False
            self._lead_off_on_count[lead_name] = 0
            self._lead_off_off_count[lead_name] = 0

        # Reset connection tracking
        self._lead_last_valid_value = {name: self._adc_center for name in self.lead_indices.keys()}
        self._lead_connection_state = {name: True for name in self.lead_indices.keys()}

        self._last_packet_time = 0.0

    def mousePressEvent(self, event):
        # Block all mouse press events (left/right click) for dragging
        event.ignore()

    def mouseMoveEvent(self, event):
        # Block all mouse move events for dragging
        event.ignore()
        
    def init_ui(self):
        """Initialize the user interface"""
        import pyqtgraph as pg
        # Antialiasing smooths the trace but roughly triples line-drawing cost, which
        # is the difference between a steady scroll and a stuttering one on the 8 GB /
        # i3 minimum spec. Off there, on everywhere else — on an ECG line the
        # difference is barely perceptible (see smooth_display.py, which never
        # enables it).
        pg.setConfigOptions(antialias=not is_low_spec_mode())
        
        self.setStyleSheet("""
            QWidget { background: #0D1117; color: #F9FAFB; }
            QFrame  { background: #111827; }
        """)
        
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(24, 24, 24, 24)
        
        # Header
        header = QHBoxLayout()
        title = QLabel("Hyperkalemia Detection Test")
        title.setFont(QFont("Segoe UI", 20, QFont.Bold))
        title.setStyleSheet("color: #FFFFFF; font-weight: 900; background: transparent;")
        header.addWidget(title)
        header.addStretch()
        
        # Lead connection status label (mirrors 12-lead behavior, placed left of status label)
        self.lead_status_label = QLabel("")
        self.lead_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.lead_status_label.setStyleSheet(
            "color: #c62828; font-size: 12px; font-weight: bold; padding: 5px; background: transparent;"
        )
        self.lead_status_label.hide()  # Hidden until leads go off
        header.addWidget(self.lead_status_label)

        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setFont(QFont("Segoe UI", 12))
        self.status_label.setStyleSheet("color: #6B7280; padding: 5px; background: transparent;")
        header.addWidget(self.status_label)
        
        # Timer label
        self.timer_label = QLabel("Time: 00:00")
        self.timer_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.timer_label.setStyleSheet("color: #EF4444; padding: 5px; font-weight: 900; background: transparent;")
        header.addWidget(self.timer_label)
        
        layout.addLayout(header)

        # Control buttons
        controls = QHBoxLayout()
        controls.setSpacing(16)
        
        # Start button
        self.start_btn = QPushButton("Start Capture")
        self.start_btn.setCursor(Qt.PointingHandCursor)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background: #1A2E1A; color: #4ADE80; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: 1px solid #22C55E;
            }
            QPushButton:hover { background: #22402A; }
            QPushButton:disabled { background: #151B15; color: #374151; border: 1px solid #1F2937; }
        """)
        self.start_btn.clicked.connect(self.start_capture)
        controls.addWidget(self.start_btn)
        
        # Stop button (initially disabled)
        self.stop_btn = QPushButton("Stop Capture")
        self.stop_btn.setCursor(Qt.PointingHandCursor)
        self.stop_btn.setStyleSheet("""
            QPushButton {
                background: #2E1A1A; color: #F87171; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI', Arial; border: 1px solid #EF4444;
            }
            QPushButton:hover { background: #3F2020; }
            QPushButton:disabled { background: #1A1515; color: #374151; border: 1px solid #1F2937; }
        """)
        self.stop_btn.clicked.connect(self.confirm_stop)
        self.stop_btn.setEnabled(False)
        controls.addWidget(self.stop_btn)
        
        controls.addStretch()
        
        # Analyze button (initially disabled)
        self.analyze_btn = QPushButton("Analyze for Hyperkalemia")
        self.analyze_btn.setCursor(Qt.PointingHandCursor)
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background: #1E2D4A; color: #60A5FA; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI'; border: 1px solid #3B82F6;
            }
            QPushButton:hover { background: #2A3F6B; }
            QPushButton:disabled { background: #1A1F2E; color: #374151; border: 1px solid #374151; }
        """)
        self.analyze_btn.clicked.connect(self.analyze_hyperkalemia)
        self.analyze_btn.setEnabled(False)
        controls.addWidget(self.analyze_btn)
        
        # Generate Report button (initially disabled)
        self.report_btn = QPushButton("Generate Report")
        self.report_btn.setCursor(Qt.PointingHandCursor)
        self.report_btn.setStyleSheet("""
            QPushButton {
                background: #1A1F2E; color: #9CA3AF; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI'; border: 1px solid #374151;
            }
            QPushButton:hover { background: #252B3B; }
            QPushButton:disabled { background: #151A25; color: #374151; border: 1px solid #2A3040; }
        """)
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setEnabled(False)
        controls.addWidget(self.report_btn)
        
        layout.addLayout(controls)
        
        # Metrics display section
        metrics_card = QFrame()
        metrics_card.setStyleSheet("QFrame { background: #111827; border: 1px solid #1E2A3A; border-radius: 16px; } QLabel { border: none; background: transparent; }")
        metrics_card.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        metrics_layout = QHBoxLayout(metrics_card)
        metrics_layout.setContentsMargins(24, 20, 24, 20)
        metrics_layout.setSpacing(20)
        
        try:
            from PyQt5.QtWidgets import QGraphicsDropShadowEffect
            from PyQt5.QtGui import QColor
            shadow = QGraphicsDropShadowEffect(self)
            shadow.setBlurRadius(20)
            shadow.setOffset(0, 4)
            shadow.setColor(QColor(16, 24, 40, 30))
            metrics_card.setGraphicsEffect(shadow)
        except Exception:
            pass
        
        # Store metric labels for live update
        self.metric_labels = {}
        metric_info = [
            ("HR", "00", "BPM", "heart_rate"),
            ("PR", "0", "ms", "pr_interval"),
            ("QRS", "0", "ms", "qrs_duration"),
            ("QT/QTc", "0", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
            lbl.setStyleSheet("color: #9CA3AF; font-size: 11px;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Segoe UI", 16, QFont.Bold))
            val.setStyleSheet("color: #FFFFFF; font-size: 16px; font-weight: bold;")
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box)
            self.metric_labels[key] = val
        
        layout.addWidget(metrics_card)
        
        # Plot area - Grid layout for 7 leads (Lead II + V1-V6)
        plot_frame = QFrame()
        plot_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        plot_frame.setStyleSheet("background: #000000; border-radius: 16px; border: 1px solid #333333;")
        plot_layout = QGridLayout(plot_frame)
        plot_layout.setContentsMargins(16, 16, 16, 16)
        plot_layout.setSpacing(10)
        
        # Create plot widgets and curves for each lead
        self.plot_widgets = {}
        self.plot_curves = {}
        self.data_lines = {} # For consistency with 12-lead naming
        
        lead_names = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        
        # Arrange in 2 columns (V1-V6 first, Lead II at the last bottom row full width)
        positions = {
            'V1': (0, 0),
            'V2': (0, 1),
            'V3': (1, 0),
            'V4': (1, 1),
            'V5': (2, 0),
            'V6': (2, 1),
            'II': (3, 0),
        }
        
        # Consistent colors from 12-lead dashboard (all black as requested)
        lead_colors = {
            'II': '#00ff00',
            'V1': '#00ff00',
            'V2': '#00ff00',
            'V3': '#00ff00',
            'V4': '#00ff00',
            'V5': '#00ff00',
            'V6': '#00ff00'
        }
        
        for i, lead_name in enumerate(lead_names):
            # Create rounded card container so the ECG paper edges don't clip.
            card = QFrame()
            card.setStyleSheet("""
                QFrame {
                    background: #000000;
                    border: 1px solid #333333;
                    border-radius: 12px;
                }
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(5, 5, 5, 5)
            card_layout.setSpacing(0)

            # Create plot widget
            plot_widget = pg.PlotWidget()
            plot_widget.setBackground("#000000")
            plot_widget.setMenuEnabled(False)
            plot_widget.setStyleSheet("border: none;")
            plot_widget.showGrid(x=False, y=False)
            plot_widget.setClipToView(True)
            # Same as the 12-lead test: no downsampling, plain polyline. Auto
            # 'peak' downsampling collapses each pixel column to a min/max pair
            # and draws the trace as vertical bars, so the background shows
            # through between them — the black speckle inside the green line.
            plot_widget.setDownsampling(ds=1, auto=False, mode='subsample')
            plot_widget.setMouseEnabled(x=False, y=False)
            
            # Hide Y-axis numeric labels (clean clinical look)
            plot_widget.getAxis('left').setPen(pg.mkPen(None))
            plot_widget.getAxis('left').setStyle(showValues=False)
            plot_widget.getAxis('bottom').setPen(pg.mkPen(None))
            plot_widget.getAxis('bottom').setStyle(showValues=False)
            
            lead_color = lead_colors.get(lead_name, '#00ff00')
            plot_widget.setTitle(f"Lead {lead_name}", color='#FFFFFF', size='11pt')
            
            # Fixed Y range — same as HRV / 12-lead (stable, no clipping)
            if lead_name == 'aVR':
                plot_widget.setYRange(0, -4096, padding=0)
            else:
                plot_widget.setYRange(0, 4096, padding=0)
            
            # Add center line at 2048
            center_pos = -2048 if lead_name == 'aVR' else 2048
            center_line = pg.InfiniteLine(pos=center_pos, angle=0, pen=pg.mkPen(color='#003300', width=1.0, style=Qt.DashLine))
            plot_widget.addItem(center_line)

            # Display window matches 6.0 second display duration for all leads
            self._display_window_sec[lead_name] = 6.0

            vb = plot_widget.getViewBox()
            if vb is not None:
                if lead_name == 'aVR':
                    vb.setLimits(yMin=-4096, yMax=0)
                else:
                    vb.setLimits(yMin=0, yMax=4096)
                try:
                    win_sec = self._display_window_sec[lead_name]
                    vb.setLimits(xMin=0, xMax=win_sec)
                    vb.setRange(xRange=(0, win_sec))
                except Exception:
                    pass

            plot_curve = plot_widget.plot(
                pen=pg.mkPen(color='#00FF00', width=2.0), connect='finite'
            )

            self.plot_widgets[lead_name] = plot_widget
            self.plot_curves[lead_name] = plot_curve
            self.data_lines[lead_name] = plot_curve

            card_layout.addWidget(plot_widget)
            row, col = positions[lead_name]
            if lead_name == 'II':
                plot_layout.addWidget(card, row, col, 1, 2) # Lead II spanning 2 columns at bottom
            else:
                plot_layout.addWidget(card, row, col)
        
        # Keep backward compatibility - also store Lead II as primary
        self.plot_widget = self.plot_widgets['II']
        self.plot_curve = self.plot_curves['II']
        
        layout.addWidget(plot_frame, stretch=1)
        
        # Info label
        info_label = QLabel("Capture 30 seconds of Lead II and V1-V6 data for hyperkalemia detection. The system will analyze T-waves, PR interval, QRS duration, and P-wave morphology according to ECG standards.")
        info_label.setFont(QFont("Segoe UI", 10))
        info_label.setStyleSheet("color: #4B5563; padding: 10px; background: transparent;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Store analysis results
        self.analysis_results = None

    def refresh_com_ports(self):
        """Refresh available COM ports"""
        pass
    
    def start_capture(self):
        """Start capturing Lead II data"""
        # CHECK: Ensure no other test is running
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            # Check if dashboard has the can_start_test method
            if hasattr(self.dashboard_instance, 'can_start_test'):
                if not self.dashboard_instance.can_start_test("hyperkalemia_test"):
                    return
                # Set state to running
                self.dashboard_instance.update_test_state("hyperkalemia_test", True)

        # Port detection and serial connection
        if not SERIAL_AVAILABLE or not ECG_TEST_AVAILABLE:
            QMessageBox.warning(self, "Serial Not Available", 
                              "Serial/ECG modules are not available. Please install pyserial and restart.")
            return
        
        # FLUSH stale smoothing state from any previous capture (including 12-lead).
        try:
            from ecg.ecg_calculations import cleanup_instance
            cleanup_instance('hyperkalemia_test')
        except Exception:
            pass
        if self.ecg_calculator:
            for attr in ('_pr_smooth_buffer_tl', '_qrs_smooth_buffer',
                         '_qt_smooth_buffer', '_p_smooth_buffer',
                         '_last_displayed_qrs', '_last_displayed_qt',
                         '_last_displayed_qtc', '_last_displayed_p',
                         '_pending_qrs_value', '_pending_qt_value',
                         '_pending_p_value'):
                if hasattr(self.ecg_calculator, attr):
                    v = getattr(self.ecg_calculator, attr)
                    if isinstance(v, list):
                        v.clear()
                    else:
                        setattr(self.ecg_calculator, attr, 0)
            self.ecg_calculator.pr_interval = 0
            self.ecg_calculator.last_qrs_duration = 0
            self.ecg_calculator.last_qt_interval = 0
            self.ecg_calculator.last_qtc_interval = 0
            self.ecg_calculator.last_qtcf_interval = 0
            self.ecg_calculator.last_rr_interval = 0
            self.ecg_calculator.last_heart_rate = 0

        try:
            shared_display_updates._last_valid.clear()
        except Exception:
            pass
        self._last_metric_update_ts = 0.0

        # Reset plot buffers + lead-off state
        try:
            self._init_plot_and_lead_off_buffers()
        except Exception:
            pass

        # Get port from settings or auto-detect
        port_to_use = self.settings_manager.get_serial_port()
        baudrate = int(self.settings_manager.get_setting("baud_rate", "115200"))

        # Check if Demo Mode is active on the dashboard's ECGTestPage
        is_demo = False
        try:
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                ecg_page = getattr(self.dashboard_instance, 'ecg_test_page', None)
                if ecg_page and getattr(ecg_page, 'demo_toggle', None):
                    is_demo = ecg_page.demo_toggle.isChecked()
        except Exception:
            pass

        if is_demo:
            print(" Starting Hyperkalemia Test in Demo Mode...")
            from ecg.demo_serial_reader import DemoSerialReader
            self.serial_reader = DemoSerialReader(self)
        else:
            # Check if we already have an active reader in GlobalHardwareManager
            from ecg.serial.serial_reader import GlobalHardwareManager
            existing_reader = GlobalHardwareManager().reader
            if existing_reader and existing_reader.ser and existing_reader.ser.is_open:
                if not port_to_use or port_to_use == "Select Port":
                    port_to_use = existing_reader.ser.port
                    print(f" Using existing active serial port: {port_to_use}")
            
            # Check if port needs scanning (not set or not in available ports)
            scan_needed = (not port_to_use or port_to_use == "Select Port")
            
            if not scan_needed:
                try:
                    available_ports = [p.device for p in serial.tools.list_ports.comports()]
                    if port_to_use not in available_ports:
                        print(f" Configured port {port_to_use} not found in available ports. forcing scan.")
                        scan_needed = True
                except Exception:
                    pass
            
            if scan_needed:
                print(" No COM port configured or port not found – will auto‑scan all ports.")
                try:
                    scan_result = SerialStreamReader.scan_and_detect_port(baudrate=baudrate, timeout=0.2)
                    if scan_result:
                        detected_port, detected_serial = scan_result
                        port_to_use = detected_port
                        print(f" Auto‑detected ECG device on port {detected_port}")
                        try:
                            if detected_serial and detected_serial.is_open:
                                detected_serial.close()
                        except Exception as e:
                            print(f" Warning: Failed to close detected serial port: {e}")
                        if hasattr(self, 'settings_manager'):
                            self.settings_manager.set_setting("serial_port", detected_port)
                            self.settings_manager.save_settings()
                    else:
                        QMessageBox.warning(self, "No Device Found",
                                          "Could not auto-detect ECG device. Please check connection.")
                        return
                except Exception as scan_err:
                    print(f" Port scan failed: {scan_err}")
                    QMessageBox.warning(self, "Scan Failed", f"Port scan failed: {scan_err}")
                    return
            
            try:
                from ecg.serial.serial_reader import GlobalHardwareManager
                self.serial_reader = GlobalHardwareManager().get_reader(port_to_use, baudrate)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to open serial port: {e}")
                return

        try:
            # Start/Resume acquisition. The start() method now handles
            # skipping hardware commands if already running.
            self.serial_reader.start()

            # Centralized authorization check
            from utils.license_manager import is_ecg_acquisition_allowed
            if not is_ecg_acquisition_allowed(self):
                self.stop_capture()
                self.status_label.setText("Status: Unauthorized Device")
                self.status_label.setStyleSheet("color: #EF5350; padding: 5px;")
                return
            
            # Reset data for all leads
            self.data = np.full(10000, 2048.0, dtype=np.float32)  # Reset circular buffer
            self.lead_ii_data = []
            for lead_name in self.lead_data.keys():
                self.lead_data[lead_name] = []

            # Per-lead circular display buffers (scrolling window, same as 12-lead)
            HISTORY_LENGTH = 10000
            self._ring_bufs = {
                lead: np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32)
                for lead in self.lead_data.keys()
            }
            self._ring_active = {lead: 0 for lead in self.lead_data.keys()}
            self._ring_pos = {lead: 0 for lead in self.lead_data.keys()}

            # Reset display anchors and warmup so trace starts clean (same as HRV test)
            if hasattr(self, "_display_anchors"):
                del self._display_anchors
            self._display_warmup_samples = int(max(600, self.sampling_rate * 2.5))
            self._display_settle_samples = int(max(250, self.sampling_rate * 0.5))
            self._display_ready_announced = False
            self._display_anchors_locked = set()

            # Discard stale serial packets so the trace does not start mid-stream
            if self.serial_reader:
                try:
                    if hasattr(self.serial_reader, "buf"):
                        self.serial_reader.buf.clear()
                    ser = getattr(self.serial_reader, "ser", None)
                    if ser and hasattr(ser, "reset_input_buffer"):
                        ser.reset_input_buffer()
                except Exception:
                    pass

            # Reset critical lead (RA/RL/LL/LA) guard flag for this new capture session
            self._critical_lead_popup_shown = False
            self._critical_lead_disconnected_count = 0

            # Reset adaptive scaling
            self.y_centers = {lead: 0.0 for lead in self.lead_data.keys()}
            self.y_ranges = {lead: 200.0 for lead in self.lead_data.keys()}
            self.y_center = 0.0
            self.y_range = 200.0
            self.sample_index = 0
            self.active_samples = 0
            self.analysis_results = None
            self.start_time = time.time()
            self.is_capturing = True
            self._last_packet_time = 0.0
            
            # Reset smoothing buffers
            if self.ecg_calculator:
                if hasattr(self.ecg_calculator, 'smoothing_buffers'):
                    self.ecg_calculator.smoothing_buffers = {}
                # ALWAYS reset data buffer to avoid step function from previous captures
                self.ecg_calculator.data = [np.full(10000, 2048.0, dtype=np.float32) for _ in range(12)]
                
                # Reset sampler to avoid stale timestamps causing bad rate calculations
                if hasattr(self.ecg_calculator, 'sampler') and self.ecg_calculator.sampler:
                    try:
                        from ecg.utils.helpers import SamplingRateCalculator
                        self.ecg_calculator.sampler = SamplingRateCalculator()
                        self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                    except Exception:
                        pass
            
            # Update UI
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(True)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Capturing from RhythmUltra Device...")
            self.status_label.setStyleSheet("color: #00E676; padding: 5px;")
            self._silent_data_warned = False

            if 'heart_rate' in self.metric_labels:
                if getattr(self, '_last_displayed_bpm', 0) > 0:
                    self.metric_labels['heart_rate'].setText(f"{int(self._last_displayed_bpm)} BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("0 ms")
            
            # ── Start HolterBPMController ───────────────────────────────────────
            # Bar removed as per user request
            try:
                if self._bpm_ctrl is not None:
                    if self._bpm_ctrl.is_running:
                        self._bpm_ctrl.stop()
                    self._bpm_ctrl.start(target_hours=0)
                    # main_layout = self.layout()
                    # if main_layout and self._bpm_ctrl.display_bar is not None:
                    #     existing = [main_layout.itemAt(i).widget()
                    #                 for i in range(main_layout.count())
                    #                 if main_layout.itemAt(i).widget() is not None]
                    #     if self._bpm_ctrl.display_bar not in existing:
                    #         main_layout.insertWidget(0, self._bpm_ctrl.display_bar)
                    #     self._bpm_ctrl.display_bar.show()
                        
                    # Start 3-second BPM UI refresh timer
                    if not hasattr(self, '_bpm_refresh_timer'):
                        self._bpm_refresh_timer = QTimer()
                        self._bpm_refresh_timer.timeout.connect(self._refresh_holter_bpm_label)
                    if not self._bpm_refresh_timer.isActive():
                        self._bpm_refresh_timer.start(2000)
            except Exception as _bpm_err:
                print(f"[HyperkalemiaTestWindow] BPM controller start error: {_bpm_err}")
            
            # Start timers
            self.capture_timer.start(50 if is_low_spec_mode() else 30)  # 20 FPS on low-spec, 33 FPS on normal
            self.duration_timer.start(1000)  # Check duration every second
            self.metrics_timer = QTimer(self)
            self.metrics_timer.timeout.connect(self.update_metrics)
            self.metrics_timer.start(500 if is_low_spec_mode() else 200)
            
            # No success message needed - status label already shows the state
            
        except Exception as e:
            QMessageBox.critical(self, "Error", 
                               f"Failed to start capture: {str(e)}")
            self.crash_logger.log_error(
                message=f"Hyperkalemia test capture start error: {e}",
                exception=e,
                category="HYPERKALEMIA_TEST_ERROR"
            )
    
    def _handle_critical_lead_disconnect(self):
        """
        RA/RL/LL(F)/LA(L) electrode loss makes every lead invalid (same as 12-lead test).
        Alert the user with a blocking popup, send STOP command, and close window.
        Fires once per disconnection event.
        """
        if getattr(self, "_critical_lead_popup_shown", False):
            return  # Guard: fire only once per disconnection
        
        self._critical_lead_popup_shown = True

        # ── STOP ALL TIMERS & DATA CAPTURE IMMEDIATELY ──────────────────────
        # Do this before showing the popup so nothing runs in the background
        # while the user sees the warning dialog.
        try:
            self.capture_timer.stop()
        except Exception:
            pass
        try:
            self.duration_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, 'metrics_timer'):
                self.metrics_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, '_bpm_refresh_timer') and self._bpm_refresh_timer.isActive():
                self._bpm_refresh_timer.stop()
        except Exception:
            pass
        # Stop serial reader so no new data packets arrive
        try:
            if self.serial_reader:
                self.serial_reader.running = False
        except Exception:
            pass
        self.is_capturing = False

        popup_title = "Warning"
        popup_text = "ECG Electrode disconnected, please check electrode connection"
        
        try:
            # Show blocking popup (waits for user to click OK)
            from dashboard.dashboard import StyledMessageBox
            StyledMessageBox.show_message(self, popup_title, popup_text, is_critical=True)
        except Exception as e:
            print(f" Falling back to QMessageBox for electrode disconnect alert: {e}")
            try:
                QMessageBox.critical(self, popup_title, popup_text)
            except Exception as e2:
                print(f" Error showing electrode disconnect popup: {e2}")
        
        # After user clicks OK, send STOP command and clean up
        try:
            self.stop_capture(device_disconnected=True)
        except Exception as e:
            print(f" Error stopping capture on critical lead disconnect: {e}")
        
        # Close test window and return to dashboard
        try:
            self.close()
        except Exception as e:
            print(f" Error closing Hyperkalemia window after critical lead disconnect: {e}")

    def stop_capture(self, device_disconnected=False, device_not_sending=False):
        """Stop capturing data"""
        # UPDATE STATE: Test stopped
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, 'update_test_state'):
                self.dashboard_instance.update_test_state("hyperkalemia_test", False)

        self.is_capturing = False
        
        # Serial reader cleanup
        if self.serial_reader:
            try:
                if hasattr(self.serial_reader, "command_handler") and self.serial_reader.command_handler:
                    self.serial_reader.command_handler.send_stop_command()
                # Reset running state so other tests can send START command
                self.serial_reader.running = False
            except Exception as e:
                print(f"[HyperkalemiaTest] Error sending stop command: {e}")
            self.serial_reader = None
        
        # ── Stop HolterBPMController ────────────────────────────────────────
        try:
            if hasattr(self, '_bpm_refresh_timer') and self._bpm_refresh_timer.isActive():
                self._bpm_refresh_timer.stop()
            if self._bpm_ctrl is not None and self._bpm_ctrl.is_running:
                self._bpm_ctrl.stop()
                if self._bpm_ctrl.display_bar is not None:
                    self._bpm_ctrl.display_bar.hide()
        except Exception as _bpm_err:
            print(f"[HyperkalemiaTestWindow] BPM controller stop error: {_bpm_err}")
        
        # Stop timers
        self.capture_timer.stop()
        self.duration_timer.stop()
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        
        # Update UI based on reason
        if device_disconnected:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device disconnected")
            
            # Reset dashboard metrics and interpretation
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                if hasattr(self.dashboard_instance, 'reset_metrics_and_interpretation'):
                    self.dashboard_instance.reset_metrics_and_interpretation()
        elif device_not_sending:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device connected but not sending data")
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            
            if len(self.lead_ii_data) > 0:
                self.analyze_btn.setEnabled(True)
                self.report_btn.setEnabled(True)
                self.status_label.setText(f"Status: Capture Complete")
            else:
                self.status_label.setText("Status: Capture Stopped (No data)")
        
        self.status_label.setStyleSheet("color: #6B7280; padding: 5px;")

    def _reset_after_report_open(self):
        """Clear the completed hyperkalemia session so the next test starts fresh."""
        self.is_capturing = False
        self.lead_ii_data = []
        self.analysis_results = None
        self.active_samples = 0
        self._last_metric_update_ts = 0.0
        try:
            for curve in getattr(self, "plot_curves", {}).values():
                curve.setData([], [])
        except Exception:
            pass
        try:
            if hasattr(self, "plot_widgets"):
                for lead_name, widget in self.plot_widgets.items():
                    win_sec = self._display_window_sec.get(lead_name, 6.0)
                    widget.setXRange(0, win_sec, padding=0)
                    if lead_name == 'aVR':
                        widget.setYRange(0, -4096, padding=0)
                    else:
                        widget.setYRange(0, 4096, padding=0)
        except Exception:
            pass
        try:
            # Reset ALL metrics to initial values when report is opened
            self._last_displayed_bpm = 0
            if 'heart_rate' in self.metric_labels:
                self.metric_labels['heart_rate'].setText("00 BPM")
            if 'pr_interval' in self.metric_labels:
                self.metric_labels['pr_interval'].setText("0 ms")
            if 'qrs_duration' in self.metric_labels:
                self.metric_labels['qrs_duration'].setText("0 ms")
            if 'qtc_interval' in self.metric_labels:
                self.metric_labels['qtc_interval'].setText("0 ms")
        except Exception:
            pass
        try:
            self.status_label.setText("Status: Ready")
            self.status_label.setStyleSheet("color: #6B7280; padding: 5px;")
        except Exception:
            pass
        try:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.analyze_btn.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.report_btn.setText("Generate Report")
        except Exception:
            pass
        self.timer_label.setText("Time: 00:00")

    def _refresh_holter_bpm_label(self):
        """Called every 2 s by _bpm_refresh_timer.
        HolterBPMController runs in background for arrhythmia detection.
        We no longer use its BPM value for the HR display or report — the
        calculate_ecg_metrics() pipeline (same as 12-lead) is the authoritative
        source. This method is intentionally a no-op for HR display.
        """
        # NOTE: Do NOT write HolterBPM to metric_labels['heart_rate'] or
        # self.last_heart_rate here. The update_metrics() method (called by
        # the metrics_timer every 500 ms) already writes the correct ECG-derived
        # BPM from calculate_hr_rr() to self.last_heart_rate and the HR label.
        # Writing HolterBPM here would overwrite the correct value with a
        # different (often much higher) estimate from a different algorithm.
        pass

    def confirm_stop(self):
        reply = QMessageBox.question(
            self,
            "Confirm Stop",
            "Are you sure you want to stop?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            try:
                self.stop_capture()
            finally:
                if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                    try:
                        self.dashboard_instance.raise_()
                        self.dashboard_instance.activateWindow()
                    except Exception:
                        pass
                self.close()
    
    def check_duration(self):
        """Check if 30 seconds have elapsed"""
        if not self.is_capturing:
            return
        
        elapsed = time.time() - self.start_time
        remaining = max(0, self.capture_duration - elapsed)
        
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        self.timer_label.setText(f"Time: {minutes:02d}:{seconds:02d}")
        
        if elapsed >= self.capture_duration:
            self.stop_capture()
            QMessageBox.information(self, "Capture Complete", 
                                  "30-second capture completed successfully!")
    
    def update_plot(self):
        """Update all plots with new data matching 12-lead dashboard style"""
        if getattr(self, "_plot_update_in_progress", False):
            return
        self._plot_update_in_progress = True
        if not self.is_capturing or not self.serial_reader:
            self._plot_update_in_progress = False
            return

        # Check if device got disconnected suddenly
        if not self.serial_reader.running:
            print("⚠️ Device disconnected during Hyperkalemia test!")
            self.stop_capture(device_disconnected=True)
            self._plot_update_in_progress = False
            return
        
        # ── RA/RL/LL(F)/LA(L) LEAD DISCONNECTION DETECTION ──────────────────
        # Check if RA, RL, LL, or LA electrodes are disconnected (same as 12-lead test)
        # These are critical electrodes - if they're off, ALL leads are invalid
        try:
            packets = self.serial_reader.read_packets(max_packets=100)
            
            if packets and len(packets) > 0:
                # Check the most recent packet for RA/RL/LL/LA presence flags
                last_packet = packets[-1]
                ra_present = last_packet.get("__ra_present__", True)
                rl_absent = not last_packet.get("__rl_present__", True)
                ll_present = last_packet.get("__ll_present__", True)
                la_present = last_packet.get("__la_present__", True)
                
                # Track RA/RL/LL/LA state for debouncing
                if not hasattr(self, '_critical_lead_disconnected_count'):
                    self._critical_lead_disconnected_count = 0
                
                # If RA is absent OR RL is absent OR LL is absent OR LA is absent, increment counter
                if not ra_present or rl_absent or not ll_present or not la_present:
                    self._critical_lead_disconnected_count += 1
                    
                    # Require 5 consecutive packets (~10ms) before triggering alert
                    if self._critical_lead_disconnected_count >= 5:
                        # Check guard flag (fire only once per event)
                        if not getattr(self, '_critical_lead_popup_shown', False):
                            print("⚠️ Critical electrode (RA/RL/LL/LA) disconnected during Hyperkalemia test!")
                            self._handle_critical_lead_disconnect()
                            self._plot_update_in_progress = False
                            return
                else:
                    # Reset counter if all critical electrodes are connected
                    self._critical_lead_disconnected_count = 0
        except Exception as e:
            print(f" Error checking critical lead status: {e}")
            packets = []  # Continue with empty packets
            
        
        try:
            elapsed = time.time() - self.start_time
            
            # Packets already read above during RA/RL check
            if not packets and hasattr(self.serial_reader, 'is_device_silent') and self.serial_reader.is_device_silent(3.0):
                if not getattr(self, '_silent_data_warned', False):
                    QMessageBox.warning(
                        self,
                        "Device Not Sending Data",
                        "Device is connected but not sending ECG packets.\n\n"
                        "Please check electrodes/cable and ensure device streaming is ON."
                    )
                    self._silent_data_warned = True
                self.stop_capture(device_not_sending=True)
                self._plot_update_in_progress = False
                return

            if not packets:
                self._plot_update_in_progress = False
                return
            
            n_new = len(packets)
            # Process each packet
            for packet in packets:
                # Ring size directly — len(self.data) would rebuild the whole
                # chronological array on every packet just to read its length.
                self.active_samples = min(self._lead_ii_ring.size, self.active_samples + 1)

                # ── Feed packet to HolterBPMController ──────────────────────────
                try:
                    if self._bpm_ctrl is not None:
                        self._bpm_ctrl.push(packet)
                except Exception:
                    pass

                self.sample_index += 1
                packet_time = self.sample_index / 500.0
                self._last_packet_time = packet_time
                
                # Update calculator's data buffer for all leads from serial data
                if self.ecg_calculator:
                    for lead_name, idx in self.lead_indices.items():
                        try:
                            # Mirror 12-lead behavior: packet may contain missing keys or None for disconnected leads.
                            raw = packet.get(lead_name, None)
                            was_connected = bool(self._lead_connection_state.get(lead_name, True))

                            if raw is None:
                                self._lead_connection_state[lead_name] = False
                                self._lead_off_state[lead_name] = True
                                raw_value = float(self._lead_last_valid_value.get(lead_name, self._adc_center))
                                value_for_buffers = self._adc_center  # user-visible flatline for disconnected lead
                            else:
                                raw_value = float(raw)
                                self._lead_last_valid_value[lead_name] = raw_value
                                self._lead_connection_state[lead_name] = True
                                if not was_connected:
                                    # Clear OFF state immediately on reconnection; detector will re-evaluate within 1s.
                                    self._lead_off_state[lead_name] = False
                                    self._lead_off_on_count[lead_name] = 0
                                    self._lead_off_off_count[lead_name] = 0

                                # Lead-off detection (same thresholds used by 12-lead) on connected
                                # streams, debounced so one window cannot blank the lead.
                                try:
                                    w = self._lead_off_windows.get(lead_name)
                                    if w is not None:
                                        w.append(raw_value)
                                        if len(w) >= 50:
                                            looks_off = bool(
                                                detect_lead_off(
                                                    np.asarray(w, dtype=float),
                                                    sampling_rate=float(self.sampling_rate or 500.0),
                                                    window_size=1.0
                                                )
                                            )
                                            self._apply_lead_off_verdict(lead_name, looks_off)
                                except Exception:
                                    pass

                                value_for_buffers = self._adc_center if bool(self._lead_off_state.get(lead_name, False)) else raw_value

                            # 1. ANALYSIS DATA (RAW/flat on OFF)
                            self.ecg_calculator.data[idx] = np.roll(self.ecg_calculator.data[idx], -1)
                            self.ecg_calculator.data[idx][-1] = value_for_buffers

                            # Store capture data for reports/analysis (bounded plotting is separate)
                            if lead_name in self.lead_data:
                                self.lead_data[lead_name].append({'time': packet_time, 'value': value_for_buffers})

                            # Push into bounded plot buffer (keeps UI responsive)
                            if lead_name in self._plot_buffers:
                                self._plot_buffers[lead_name].append(value_for_buffers)

                            # Circular display buffer for scrolling window rendering
                            if lead_name in self.lead_data and hasattr(self, "_ring_bufs"):
                                self._ring_push(lead_name, value_for_buffers)

                            # Primary Lead II backup
                            if lead_name == 'II':
                                self._push_lead_ii(value_for_buffers)
                                self.lead_ii_data.append({'time': packet_time, 'value': value_for_buffers})
                        except Exception as e:
                            # Per-lead failure must not block other leads (prevents "freeze all" bug)
                            print(f" Error updating lead {lead_name}: {e}")
                            continue
            
            # Update sampling rate counter
            if self.ecg_calculator and hasattr(self.ecg_calculator, "sampler"):
                sr = 0.0
                try:
                    for _ in range(len(packets)):
                        sr = self.ecg_calculator.sampler.add_sample()
                except Exception:
                    sr = self.ecg_calculator.sampler.add_sample()
                if sr > 0:
                    safe_sr = float(sr)
                    if safe_sr < 100.0 or safe_sr > 1000.0:
                        safe_sr = 500.0
                    self.sampling_rate = safe_sr

            from ecg.ecg_filters import (
                apply_ac_filter,
                apply_emg_filter,
                apply_dft_filter,
                apply_baseline_wander_median_mean,
            )
            from ecg.utils.helpers import get_display_gain

            # Fixed filter set for this test — AC 50 Hz, EMG 25 Hz, baseline 0.5 Hz —
            # applied identically on screen and in the report. Deliberately NOT read
            # from the settings screen: only the 12-lead test is user-configurable,
            # so a hyperkalemia trace always looks the same regardless of settings.
            ac_val = "50"
            emg_val = "25"
            dft_val = "0.5"
            fs = self.sampling_rate if self.sampling_rate > 0 else 500.0
            if fs < 100.0 or fs > 1000.0:
                fs = 500.0

            render_stride = max(1, int(getattr(self, "_plot_render_stride", 1)))
            if (self.sample_index % render_stride) != 0:
                self._plot_update_in_progress = False
                return

            warmup_needed = getattr(self, "_display_warmup_samples", int(2.5 * fs))
            settle_needed = warmup_needed + getattr(self, "_display_settle_samples", int(0.5 * fs))
            display_ready = self.active_samples >= settle_needed
            if not display_ready:
                self._plot_update_in_progress = False
                return

            gain_factor = get_display_gain(self.settings_manager.get_wave_gain())

            if not hasattr(self, "_display_anchors"):
                self._display_anchors = {}

            for lead_name in self.lead_data.keys():
                ring = getattr(self, "_ring_bufs", {}).get(lead_name)
                if ring is None:
                    continue

                lead_is_off = bool(self._lead_off_state.get(lead_name, False))
                valid_count = min(self._ring_active.get(lead_name, 0), len(ring))
                if valid_count <= 0:
                    continue

                window_seconds = self._display_window_sec.get(
                    lead_name, 6.0
                )
                window_samples = max(50, int(window_seconds * fs))
                window_samples = min(window_samples, valid_count)

                # Pull only the visible window out of the ring — the samples are
                # stored in write order now, so the tail helper reassembles them
                # chronologically without touching the rest of the buffer.
                buffer_data = self._ring_tail(lead_name, window_samples)
                if buffer_data.size == 0:
                    continue
                if lead_is_off:
                    buffer_data = np.full_like(buffer_data, self._adc_center)

                # Same filter + smooth pipeline as HRV test (clean display)
                if len(buffer_data) > 5 and not lead_is_off:
                    pad_len = min(int(fs * 2.0), max(0, len(buffer_data) - 1))
                    if pad_len > 0:
                        padded_data = np.pad(buffer_data, (pad_len, pad_len), mode='reflect')
                    else:
                        padded_data = buffer_data

                    if ac_val not in ("Off", "off"):
                        padded_data = apply_ac_filter(padded_data, fs, ac_val)
                    if emg_val not in ("Off", "off"):
                        padded_data = apply_emg_filter(padded_data, fs, emg_val)
                    if dft_val not in ("Off", "off", "", None):
                        dft_text = str(dft_val).strip()
                        if dft_text == "0.5":
                            padded_data = apply_baseline_wander_median_mean(padded_data, fs)
                        else:
                            padded_data = apply_dft_filter(padded_data, fs, dft_text)
                        padded_data = padded_data + float(self._adc_center)
                        if dft_text == "0.5":
                            edge_trim = int(0.75 * fs)
                            min_keep = max(50, pad_len * 2 + 20)
                            if edge_trim > 0 and len(padded_data) > (2 * edge_trim + min_keep):
                                padded_data = padded_data[edge_trim:-edge_trim]

                    if pad_len > 0:
                        buffer_data = padded_data[pad_len:-pad_len]
                    else:
                        buffer_data = padded_data

                    buffer_data = gaussian_filter1d(buffer_data, sigma=self._SMOOTH_SIGMA)

                if len(buffer_data) <= 0:
                    continue

                if not getattr(self, "_display_ready_announced", False):
                    self._display_anchors = {}
                    self._display_ready_announced = True

                # Slowly-tracking baseline anchor, as the 12-lead test does. The
                # anchor used to be measured once and locked forever, so any drift
                # afterwards pushed the whole trace off centre; an exponential
                # follower removes drift without a hard high-pass distorting the
                # ST segment.
                center_val = -2048.0 if lead_name == 'aVR' else 2048.0
                if lead_is_off:
                    display_values = np.full(len(buffer_data), center_val)
                else:
                    baseline_now = float(np.nanmedian(buffer_data))
                    if lead_name not in self._display_anchors:
                        self._display_anchors[lead_name] = baseline_now
                    else:
                        alpha = self._BASELINE_ALPHA_SLOW
                        self._display_anchors[lead_name] = (
                            (1.0 - alpha) * self._display_anchors[lead_name] + alpha * baseline_now
                        )

                    centered = (buffer_data - self._display_anchors[lead_name]) * gain_factor
                    centered = centered - float(np.nanmean(centered))
                    # Thin the trace to roughly two points per pixel column before
                    # plotting. Seven lanes share this window, so each is only about
                    # 900 px wide: at 500 Hz that is 3.3 points per column, and the
                    # polyline zig-zags inside each column into a hairy-looking line.
                    # The signal is already low-passed at 25 Hz, so halving the rate
                    # is still five times Nyquist and cannot lose a peak — it is
                    # thinner drawing, not less signal.
                    centered = self._decimate_for_display(centered, self._DISPLAY_DECIMATION)
                    if lead_name == 'aVR':
                        display_values = np.clip(-2048.0 + centered, -4096, 0)
                    else:
                        display_values = np.clip(2048.0 + centered, 0, 4096)

                # The curve is drawn with connect='finite', so a single NaN or inf
                # anywhere in the window breaks the line and prints as a black dot
                # inside the trace. The 12-lead test sanitises before plotting for
                # exactly this reason; do the same, parking any bad sample on the
                # lane's centre line instead of leaving a hole.
                display_values = np.nan_to_num(
                    np.asarray(display_values, dtype=float),
                    nan=center_val, posinf=center_val, neginf=center_val,
                )

                # Scrolling window display (12-lead flow — no raster sweep)
                x_axis = np.linspace(0.0, window_seconds, len(display_values))
                self.plot_curves[lead_name].setData(x_axis, display_values, connect='finite')
                self.plot_widgets[lead_name].setXRange(0, window_seconds, padding=0)
                if lead_name == 'aVR':
                    self.plot_widgets[lead_name].setYRange(0, -4096, padding=0)
                else:
                    self.plot_widgets[lead_name].setYRange(0, 4096, padding=0)

        except Exception as e:
            pass
        finally:
            self._plot_update_in_progress = False

    def update_metrics(self):
        """Calculate and update ECG metrics using same stable methods as 12-lead dashboard"""
        if not self.is_capturing:
            return
        # Wait for at least 3 seconds of real ECG data before computing BPM.
        # The buffer starts filled with flat 2048 ADC values; computing too early
        # causes the flat→signal edge to generate fake R-peaks at ~260 BPM.
        if self.active_samples < max(1500, int((self.sampling_rate or 500.0) * 3.0)):
            return

        # ── ALL-LEADS-DISCONNECTED: Reset intervals to 0 ──
        # Uses the existing _lead_connection_state/_lead_off_state already maintained
        # by the packet loop — visual flat-line display on the canvas is NOT touched.
        try:
            _conn = getattr(self, '_lead_connection_state', {})
            _off  = getattr(self, '_lead_off_state', {})
            # All monitored leads must be either disconnected (conn=False) or
            # detected as lead-off by the lead_off detector.
            _any_active = any(
                _conn.get(ln, True) and not _off.get(ln, False)
                for ln in self.lead_indices.keys()
            )
            _all_disconnected = not _any_active
        except Exception:
            _all_disconnected = False

        if _all_disconnected:
            # Reset internal metric state
            if self.ecg_calculator:
                self.ecg_calculator.last_heart_rate   = 0
                self.ecg_calculator.last_rr_interval  = 0
                self.ecg_calculator.pr_interval        = 0
                self.ecg_calculator.last_qrs_duration  = 0
                self.ecg_calculator.last_qt_interval   = 0
                self.ecg_calculator.last_qtc_interval  = 0
                for _buf in ('_pr_smooth_buffer_tl', '_qrs_smooth_buffer', '_qt_smooth_buffer'):
                    if hasattr(self.ecg_calculator, _buf):
                        getattr(self.ecg_calculator, _buf).clear()
            self.last_heart_rate = 0
            self._last_displayed_bpm = 0
            # Show the lead-off warning label (same as 12-lead)
            try:
                if hasattr(self, 'lead_status_label') and self.lead_status_label is not None:
                    self.lead_status_label.setText("No ECG signal detected. Check patient leads.")
                    self.lead_status_label.show()
            except Exception:
                pass
            # Push zeros to the display labels
            try:
                if 'heart_rate' in self.metric_labels:
                    self.metric_labels['heart_rate'].setText("0 BPM")
                if 'rr_interval' in self.metric_labels:
                    self.metric_labels['rr_interval'].setText("0 ms")
                if 'pr_interval' in self.metric_labels:
                    self.metric_labels['pr_interval'].setText("0 ms")
                if 'qrs_duration' in self.metric_labels:
                    self.metric_labels['qrs_duration'].setText("0 ms")
                if 'qtc_interval' in self.metric_labels:
                    self.metric_labels['qtc_interval'].setText("0/0 ms")
            except Exception:
                pass
            return  # Skip normal metric computation while all leads are off

        # Leads are active — clear the warning label if it was shown
        try:
            if hasattr(self, 'lead_status_label') and self.lead_status_label is not None:
                if self.lead_status_label.isVisible():
                    self.lead_status_label.hide()
                    self.lead_status_label.setText("")
        except Exception:
            pass

        
        try:
            # Sync sampling rate
            current_fs = self.sampling_rate if self.sampling_rate > 0 else 500.0
            if current_fs < 100.0 or current_fs > 1000.0:
                current_fs = 500.0
            if self.ecg_calculator and self.ecg_calculator.sampler:
                self.ecg_calculator.sampler.sampling_rate = current_fs
                self.ecg_calculator.sampling_rate = current_fs

                # TRIGGER STABLE MEDIAN-BEAT ANALYSIS
                # Use calculate_ecg_metrics() (same as 12-lead) as primary HR source.
                # HolterBPMController is only used as a last-resort fallback when
                # calculate_ecg_metrics returns 0 (signal too short / no R-peaks).
                _bpm_active = (self._bpm_ctrl is not None and self._bpm_ctrl.is_running)
                _holter_bpm = 0
                if _bpm_active:
                    try:
                        _holter_bpm = self._bpm_ctrl.current_bpm()
                    except Exception:
                        pass
                # Do NOT pre-seed ecg_calculator.last_heart_rate from HolterBPM here;
                # let calculate_ecg_metrics() derive HR from the actual RR intervals.

                try:
                    original_buffers = {}
                    for idx in self.lead_indices.values():
                        original_buffers[idx] = self.ecg_calculator.data[idx]
                        if self.active_samples < len(original_buffers[idx]):
                            self.ecg_calculator.data[idx] = original_buffers[idx][-self.active_samples:]
                    
                    self.ecg_calculator.calculate_ecg_metrics()
                    
                    for idx, original in original_buffers.items():
                        self.ecg_calculator.data[idx] = original
                except Exception as e:
                    print(f" calculate_ecg_metrics error in Hyperkalemia test: {e}")

                # FETCH METRICS DIRECTLY from the calculator's stored attributes.
                def _attr_to_str(attr_name, fallback='0'):
                    v = getattr(self.ecg_calculator, attr_name, 0)
                    try:
                        iv = int(round(float(v))) if v else 0
                        return str(iv) if iv > 0 else fallback
                    except:
                        return fallback

                def _attr_to_num(attr_name, fallback=0):
                    v = getattr(self.ecg_calculator, attr_name, fallback)
                    try:
                        return int(round(float(v))) if v else fallback
                    except Exception:
                        return fallback

                # Also call get_current_metrics as a secondary/fallback source
                metrics = self.ecg_calculator.get_current_metrics()
                self.last_metrics = dict(metrics) if isinstance(metrics, dict) else {}

                # Update UI labels
                hr_val  = metrics.get('heart_rate', '0')
                pr_val  = _attr_to_str('pr_interval') or metrics.get('pr_interval', '0')
                qrs_val = _attr_to_str('last_qrs_duration') or metrics.get('qrs_duration', '0')
                qt_val  = _attr_to_str('last_qt_interval') or metrics.get('qt_interval', '0')
                qtc_val = _attr_to_str('last_qtc_interval') or metrics.get('qtc_interval', '0')

                # Also persist to last_metrics so report generation can use them
                if pr_val != '0':  self.last_metrics['pr_interval'] = pr_val
                if qrs_val != '0': self.last_metrics['qrs_duration'] = qrs_val
                if qt_val != '0':  self.last_metrics['qt_interval'] = qt_val
                if qtc_val != '0': self.last_metrics['qtc_interval'] = qtc_val

                print(f"Heart Rate: {hr_val} BPM, PR Interval: {pr_val} ms, QRS Duration: {qrs_val} ms, QTC Interval: {qtc_val} ms")

                # ── PRIMARY: Use calculate_ecg_metrics() HR (same as 12-lead) ─────
                # hr_val comes from metrics.get('heart_rate') which is the result of
                # calculate_hr_rr() → Pan-Tompkins R-peaks → median(RR) → BPM.
                # This is exactly the same algorithm as the 12-lead display.
                display_hr = 0
                if hr_val not in ('0', '--', ''):
                    try:
                        display_hr = int(round(float(hr_val)))
                    except Exception:
                        display_hr = 0
                # If calculate_ecg_metrics returned 0 (no R-peaks yet), fall back to
                # last stable value, then HolterBPM as last resort.
                if display_hr <= 0:
                    display_hr = _attr_to_num('last_heart_rate', 0)
                if display_hr <= 0 and _holter_bpm > 0:
                    display_hr = int(round(_holter_bpm))
                if display_hr <= 0 and getattr(self, '_last_displayed_bpm', 0) > 0:
                    display_hr = int(self._last_displayed_bpm)

                display_pr = _attr_to_num('pr_interval', 0)
                display_qrs = _attr_to_num('last_qrs_duration', 0)
                display_qt = _attr_to_num('last_qt_interval', 0)
                display_qtc = _attr_to_num('last_qtc_interval', 0)
                display_rr = _attr_to_num('last_rr_interval', 0)
                display_qtcf = _attr_to_num('last_qtcf_interval', 0)

                if display_hr > 0:
                    display_hr = int(round(display_hr))
                    self.ecg_calculator.last_heart_rate = display_hr
                    self._last_displayed_bpm = display_hr
                    # Keep self.last_heart_rate in sync — this is what hyper_metric.json uses
                    self.last_heart_rate = display_hr

                self._last_metric_update_ts = shared_display_updates.update_ecg_metrics_display(
                    self.metric_labels,
                    display_hr,
                    display_pr,
                    display_qrs,
                    0,
                    display_qt,
                    display_qtc,
                    display_qtcf,
                    getattr(self, '_last_metric_update_ts', 0.0),
                    rr_interval=display_rr,
                    skip_heart_rate=False,  # always update HR label with ECG-derived value
                )

                if 'heart_rate' in self.metric_labels:
                    if display_hr > 0:
                        self.metric_labels['heart_rate'].setText(f"{display_hr} BPM")
                        self._last_displayed_bpm = display_hr

                if 'pr_interval' in self.metric_labels:
                    pr_text = self.metric_labels['pr_interval'].text().strip()
                    if pr_text and not pr_text.endswith("ms"):
                        self.metric_labels['pr_interval'].setText(f"{pr_text} ms")
                if 'qrs_duration' in self.metric_labels:
                    qrs_text = self.metric_labels['qrs_duration'].text().strip()
                    if qrs_text and not qrs_text.endswith("ms"):
                        self.metric_labels['qrs_duration'].setText(f"{qrs_text} ms")
                if 'qtc_interval' in self.metric_labels:
                    qtqtc_text = self.metric_labels['qtc_interval'].text().strip()
                    if qtqtc_text and not qtqtc_text.endswith("ms"):
                        self.metric_labels['qtc_interval'].setText(f"{qtqtc_text} ms")


        
        except Exception as e:
            pass

    def _compute_wave_amplitudes(self):
        p_amp = 0.0
        qrs_amp = 0.0
        t_amp = 0.0
        try:
            fs = float(self.sampling_rate) if getattr(self, "sampling_rate", 0) else 500.0
            if not self.ecg_calculator or not hasattr(self.ecg_calculator, "data") or len(self.ecg_calculator.data) <= 1:
                return p_amp, qrs_amp, t_amp
            lead_ii = self.ecg_calculator.data[1]
            if isinstance(lead_ii, (list, tuple)):
                lead_ii = np.asarray(lead_ii, dtype=np.float32)
            arr = lead_ii
            if self.active_samples > 0 and self.active_samples < len(arr):
                arr = arr[-self.active_samples:]
            max_len = int(10 * fs)
            if len(arr) > max_len:
                arr = arr[-max_len:]
            if arr is None or len(arr) < int(2 * fs) or np.std(arr) < 0.1:
                return p_amp, qrs_amp, t_amp
            nyq = fs / 2.0
            b, a = butter(2, [max(0.5 / nyq, 0.001), min(40.0 / nyq, 0.99)], btype="band")
            x = filtfilt(b, a, arr)
            squared = np.square(np.diff(x))
            win = max(1, int(0.15 * fs))
            env = np.convolve(squared, np.ones(win) / win, mode="same")
            thr = np.mean(env) + 0.5 * np.std(env)
            r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6 * fs))
            if len(r_peaks) < 3:
                return p_amp, qrs_amp, t_amp
            p_vals = []
            qrs_vals = []
            t_vals = []
            for r in r_peaks[1:-1]:
                p_start = max(0, r - int(0.20 * fs))
                p_end = max(0, r - int(0.12 * fs))
                if p_end > p_start:
                    seg = x[p_start:p_end]
                    base = np.mean(x[max(0, p_start - int(0.05 * fs)):p_start])
                    p_vals.append(np.max(seg) - base)
                qrs_start = max(0, r - int(0.08 * fs))
                qrs_end = min(len(x), r + int(0.08 * fs))
                if qrs_end > qrs_start:
                    seg = x[qrs_start:qrs_end]
                    qrs_vals.append(np.max(seg) - np.min(seg))
                t_start = min(len(x), r + int(0.10 * fs))
                t_end = min(len(x), r + int(0.30 * fs))
                if t_end > t_start:
                    seg = x[t_start:t_end]
                    base = np.mean(x[r:t_start]) if t_start > r else 0.0
                    t_vals.append(np.max(seg) - base)
            def med(v):
                return float(np.median(v)) if len(v) > 0 else 0.0
            p_amp = med(p_vals)
            qrs_amp = med(qrs_vals)
            t_amp = med(t_vals)
        except Exception:
            pass
        return p_amp, qrs_amp, t_amp

    def _calculate_estimated_k(self, p_amp, qrs_amp, t_amp, qrs_ms, pr_ms, max_t_amp_precordial):
        """
        Estimate serum potassium (K+) level in mmol/L based on morphological features.
        This is a heuristic-based estimation model.
        """
        if qrs_amp == 0 and t_amp == 0 and p_amp == 0:
            return 0.0
            
        # Baseline potassium (Normal average)
        est_k = 4.0
        
        # 1. Effect of QRS widening (Strong indicator)
        if qrs_ms > 100:
            # Every 10ms over 100ms adds roughly 0.3 mmol/L
            est_k += (qrs_ms - 100) / 10.0 * 0.3
            
        # 2. Effect of T-wave amplitude (Peaking)
        # Normal T-wave in Lead II is usually < 0.5mV (approx 500 ADC units)
        if t_amp > 500:
            est_k += (t_amp - 500) / 100.0 * 0.2
            
        # 3. Effect of Precordial T-wave peaking (V2-V4)
        if max_t_amp_precordial > 800:
            est_k += (max_t_amp_precordial - 800) / 200.0 * 0.2
            
        # 4. Effect of P-wave flattening
        if qrs_amp > 0:
            p_qrs_ratio = p_amp / qrs_amp
            if p_qrs_ratio < 0.1:
                est_k += 0.5
            elif p_qrs_ratio < 0.15:
                est_k += 0.2
                
        # 5. Effect of PR prolongation
        if pr_ms > 200:
            est_k += (pr_ms - 200) / 40.0 * 0.1
            
        # Cap the results in a physiological range (3.5 to 9.0)
        return float(np.clip(est_k, 3.5, 9.0))

    def analyze_hyperkalemia(self, enable=False):
        """Analyze captured ECG data for hyperkalemia indicators using clinical standards"""
        if self.active_samples < 500:
            QMessageBox.warning(self, "Insufficient Data", 
                              "Please capture more data before analysis (at least 10 seconds).")
            return
            
        try:
            self.status_label.setText("Status: Analyzing ECG Morphology...")
            self.status_label.setStyleSheet("color: #007bff; font-weight: bold;")
            
            # 1. TRIGGER FULL CLINICAL ANALYSIS
            # Sync buffers to active portion
            original_buffers = {}
            for idx in self.lead_indices.values():
                original_buffers[idx] = self.ecg_calculator.data[idx]
                if self.active_samples < len(original_buffers[idx]):
                    self.ecg_calculator.data[idx] = original_buffers[idx][-self.active_samples:]
            
            # Force calculation
            self.ecg_calculator.calculate_ecg_metrics()
            
            # Restore buffers
            for idx, original in original_buffers.items():
                self.ecg_calculator.data[idx] = original
                
            # Get latest clinical metrics (last metrics stored)
            metrics = self.last_metrics if getattr(self, "last_metrics", None) else self.ecg_calculator.get_current_metrics()
            
            # 2. EXTRACT MEASUREMENTS
            def safe_float(val):
                try:
                    # Strip common units before conversion
                    if isinstance(val, str):
                        val = val.replace("ms", "").replace("MS", "").strip()
                    return float(val)
                except (ValueError, TypeError):
                    return 0.0

            hr = safe_float(metrics.get('heart_rate', 0))
            pr = safe_float(metrics.get('pr_interval', 0))
            qrs = safe_float(metrics.get('qrs_duration', 0))
            qt = 0.0
            qtc = 0.0

            qtqtc_text = None
            # Prefer the Hyperkalemia dashboard label if present
            if hasattr(self, 'metric_labels') and 'qtc_interval' in self.metric_labels:
                try:
                    qtqtc_text = self.metric_labels['qtc_interval'].text().strip()
                except Exception:
                    qtqtc_text = None

            # Fallback to metrics dict if needed
            if (not qtqtc_text) and metrics.get('qtc_interval') is not None:
                qtqtc_text = str(metrics.get('qtc_interval')).strip()

            if qtqtc_text:
                clean = qtqtc_text.replace("ms", "").replace("MS", "").strip()
                if "/" in clean:
                    parts = [p.strip() for p in clean.split("/") if p.strip()]
                    if len(parts) >= 1:
                        qt = safe_float(parts[0])
                    if len(parts) >= 2:
                        qtc = safe_float(parts[1])
                else:
                    qtc = safe_float(clean)

            # Final fallback: use calculator's last clinical values if parsing failed
            if qt <= 0 and hasattr(self.ecg_calculator, 'last_qt_interval'):
                try:
                    qt = float(getattr(self.ecg_calculator, 'last_qt_interval') or 0)
                except Exception:
                    pass
            if qtc <= 0 and hasattr(self.ecg_calculator, 'last_qtc_interval'):
                try:
                    qtc = float(getattr(self.ecg_calculator, 'last_qtc_interval') or 0)
                except Exception:
                    pass

            p_amp, qrs_amp, t_amp = self._compute_wave_amplitudes()
            
            # 3. HYPERKALEMIA MORPHOLOGY LOGIC (GE/Philips standards)
            indicators = []
            risk_score = 0
            
            # Indicator 1: PR Interval Prolongation
            if pr > 200:
                indicators.append(f"Prolonged PR Interval ({pr}ms)")
                risk_score += 1
                if pr > 240:
                    risk_score += 1
                
            # Indicator 2: QRS Widening
            if qrs > 110:
                indicators.append(f"Widened QRS Complex ({qrs}ms)")
                risk_score += 1
                if qrs > 120:
                    risk_score += 2
                
            # Indicator 3: Peaked T-waves (Estimated from amplitude variation)
            # We check precordial leads V2-V4 for maximum amplitude
            max_t_amp = 0
            try:
                # Use Lead V2/V3 for peaked T-wave detection if available
                test_leads = [self.lead_indices.get('V2'), self.lead_indices.get('V3')]
                for l_idx in test_leads:
                    if l_idx is not None:
                        sig = self.ecg_calculator.data[l_idx]
                        if self.active_samples < len(sig):
                            sig = sig[-self.active_samples:]
                        amp = np.percentile(sig, 99) - np.percentile(sig, 1)
                        max_t_amp = max(max_t_amp, amp)
                
                if max_t_amp > 800:
                    indicators.append("Tall/Peaked T-waves detected (precordial leads)")
                    risk_score += 2
            except Exception:
                pass

            if qrs_amp > 0 and t_amp > 0 and (2.0 * t_amp) > qrs_amp:
                indicators.append("T-wave amplitude exceeds R-wave amplitude (Lead II)")
                risk_score += 2

            if qrs_amp > 0:
                if p_amp <= 0 or p_amp < 0.1 * qrs_amp:
                    if p_amp <= 0:
                        indicators.append("P-waves absent or extremely low amplitude (flattening)")
                    else:
                        indicators.append("P-wave flattening relative to QRS amplitude")
                    risk_score += 1

            sine_wave = False
            if qrs >= 160 and qrs_amp > 0 and t_amp > 0:
                if p_amp <= 0 or p_amp < 0.05 * qrs_amp:
                    sine_wave = True
            if sine_wave:
                indicators.append("Sine-wave morphology (very wide QRS with merged T-wave)")
                risk_score += 3

            # 3b. ESTIMATE SERUM POTASSIUM (K+)
            est_k = self._calculate_estimated_k(p_amp, qrs_amp, t_amp, qrs, pr, max_t_amp)

            # 4. DETERMINE RISK LEVEL
            risk_level = "Normal/Low"
            risk_color = "#28a745" # Green
            
            if risk_score >= 4:
                risk_level = "High"
                risk_color = "#dc3545" # Red
            elif risk_score >= 2:
                risk_level = "Moderate"
                risk_color = "#ffc107" # Yellow
            elif risk_score >= 1:
                risk_level = "Mild"
                risk_color = "#17a2b8" # Cyan
            
            # Calculate QTc Fridericia using HR and QT (like 12-lead)
            qtcf = 0.0
            if hr > 0 and qt > 0:
                rr_s = 60.0 / hr
                qtcf = (qt / np.cbrt(rr_s)) if rr_s > 0 else 0.0

            # Store results for report generator
            self.analysis_results = {
                "heart_rate": hr,
                "RR_ms": int(60000 / hr) if hr > 0 else 0,
                "pr_interval_ms": pr,
                "qrs_duration_ms": qrs,
                "qt_interval_ms": qt,
                "qtc_ms": qtc,
                "QTc_Fridericia": qtcf,
                "st_segment_ms": 0,
                "indicators": indicators,
                "risk_level": risk_level,
                "risk_score": risk_score,
                "estimated_k": est_k
            }
                
            self.status_label.setText(f"Status: Analysis Complete (Est. K+: {est_k:.1f} mmol/L)")
            self.status_label.setStyleSheet(f"color: {risk_color}; font-weight: bold;")
            self.report_btn.setEnabled(True)
            
            if not enable:
                msg = f"<b>Hyperkalemia Analysis Results</b><br><br>"
                msg += f"Risk Level: <span style='color:{risk_color}; font-weight:bold;'>{risk_level}</span><br>"
                msg += f"<b>Estimated Serum K+: <span style='color:{risk_color}; font-weight:bold;'>{est_k:.1f} mmol/L</span></b><br><br>"
                msg += f"Heart Rate: {hr} BPM<br>"
                msg += f"PR Interval: {pr} ms<br>"
                msg += f"QRS Duration: {qrs} ms<br>"
                msg += f"QT Interval: {qt} ms<br>"
                msg += f"QTc Interval: {qtc} ms<br><br>"
                if indicators:
                    msg += "<b>Indicators:</b><br>" + "<br>".join(["- " + i for i in indicators])
                
                QMessageBox.information(self, "Hyperkalemia Analysis", msg)
            
        except Exception as e:
            self.crash_logger.log_crash("analyze_hyperkalemia", e)
            QMessageBox.critical(self, "Analysis Error", f"Failed to complete morphological analysis: {e}")
            self.status_label.setText("Status: Analysis Failed")

    def generate_report(self):
        """Generate hyperkalemia detection report PDF"""
        # If analysis has not been run yet but enough data is present, perform a silent analysis
        if self.analysis_results is None and len(self.lead_ii_data) > 0 and self.active_samples >= 500:
            try:
                self.analyze_hyperkalemia(enable=True)
            except Exception:
                pass

        # If analysis is still unavailable, fall back to the original warning
        if self.analysis_results is None:
            QMessageBox.warning(self, "No Analysis", 
                              "Please analyze the ECG data first.")
            return
        
        if len(self.lead_ii_data) == 0:
            QMessageBox.warning(self, "No Data", 
                              "No data available to generate report.")
            return
        
        # Consistently save to the local project 'reports' directory
        reports_dir = str(data_file("reports"))
        os.makedirs(reports_dir, exist_ok=True)

        # Resolve machine serial — check all sources in priority order
        machine_serial = (
            getattr(self.dashboard_instance, "machine_serial_number", "") or ""
        ).strip() if self.dashboard_instance else ""
        if not machine_serial and hasattr(self, "settings_manager") and self.settings_manager:
            machine_serial = (self.settings_manager.get_setting("machine_serial_number", "") or "").strip()
        if not machine_serial and self.dashboard_instance and hasattr(self.dashboard_instance, "settings_manager"):
            machine_serial = (self.dashboard_instance.settings_manager.get_setting("machine_serial_number", "") or "").strip()
        if not machine_serial:
            try:
                from utils.crash_logger import get_crash_logger
                machine_serial = (get_crash_logger().machine_serial_id or "").strip()
            except Exception:
                pass

        serial_part = f"_{machine_serial}" if machine_serial and machine_serial not in ("Not Detected", "") else ""
        report_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = os.path.join(reports_dir, f"Hyperkalemia_Report{serial_part}_{report_stamp}.pdf")
        
        
        try:
            patient = resolve_patient_profile(
                explicit_patient=getattr(self.dashboard_instance, "patient_details", None),
                username=getattr(self.dashboard_instance, "username", "") or getattr(self, "username", "") or "",
                user_details=getattr(self.dashboard_instance, "user_details", {}) if self.dashboard_instance else {},
            )
            if not isinstance(patient, dict):
                patient = {}
            patient.setdefault("first_name", "")
            patient.setdefault("last_name", "")
            patient.setdefault("age", "")
            patient.setdefault("gender", "")
            patient.setdefault("date_time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            patient.setdefault("Org.", "")
            patient.setdefault("Org. Name", "")
            patient.setdefault("Org. Address", "")
            patient.setdefault("doctor_mobile", "")
            patient.setdefault("doctor", "")
            if "doctor_mobile" not in patient:
                patient["doctor_mobile"] = ""

            # Attach patient into analysis_results so the generator can render it
            if isinstance(self.analysis_results, dict):
                self.analysis_results["patient"] = patient

            # Collect raw leads in memory instead of saving to file
            raw_leads = {}
            try:
                # Save Lead II data
                if self.lead_ii_data and len(self.lead_ii_data) > 0:
                    lead_ii_values = [d['value'] for d in self.lead_ii_data]
                    raw_leads["II"] = lead_ii_values
                    print(f" Saving Lead II: {len(lead_ii_values)} samples")
                else:
                    print(f" Lead II data is empty!")
                
                # Save V1-V6 data from self.lead_data
                for lead_name in ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']:
                    if lead_name in self.lead_data:
                        if len(self.lead_data[lead_name]) > 0:
                            lead_values = [d['value'] for d in self.lead_data[lead_name]]
                            raw_leads[lead_name] = lead_values
                            print(f" Saving {lead_name}: {len(lead_values)} samples")
                
                # Also save from ecg_calculator.data if available (as fallback)
                if self.ecg_calculator and hasattr(self.ecg_calculator, 'data'):
                    for lead_name, idx in self.lead_indices.items():
                        if idx < len(self.ecg_calculator.data):
                            ecg_data = self.ecg_calculator.data[idx]
                            if isinstance(ecg_data, np.ndarray) and len(ecg_data) > 0:
                                # Get non-zero values
                                non_zero_data = ecg_data[ecg_data != 0]
                                if len(non_zero_data) > 0:
                                    if lead_name not in raw_leads or len(raw_leads[lead_name]) == 0:
                                        raw_leads[lead_name] = non_zero_data.tolist()
                
            except Exception as e:
                error_msg = f" Could not collect ECG data: {e}"
                print(error_msg)
                import traceback
                traceback.print_exc()

            # Pass raw_leads directly instead of ecg_data_file
            print(f"\n Starting report generation...")
            print(f"   PDF filepath: {filepath}")
            
            try:
                reports_dir = str(data_file("reports"))
                os.makedirs(reports_dir, exist_ok=True)
                hyper_metrics_path = os.path.join(reports_dir, 'hyper_metric.json')

                metrics_source = self.analysis_results if isinstance(self.analysis_results, dict) else {}

                def _safe_int(val):
                    try:
                        return int(round(float(val)))
                    except Exception:
                        return 0

                # Use the ECG-derived HR (same as 12-lead) stored in last_heart_rate.
                # self.last_heart_rate is now maintained by update_metrics() via
                # calculate_hr_rr() — NOT by HolterBPMController.
                # Fallback chain: last_heart_rate → _last_displayed_bpm → analysis_results.
                if hasattr(self, 'last_heart_rate') and self.last_heart_rate > 0:
                    hr = int(self.last_heart_rate)
                elif getattr(self, '_last_displayed_bpm', 0) > 0:
                    hr = int(self._last_displayed_bpm)
                else:
                    hr = _safe_int(metrics_source.get("heart_rate", 0))
                pr = _safe_int(metrics_source.get("pr_interval_ms", 0))
                qrs = _safe_int(metrics_source.get("qrs_duration_ms", 0))
                qt = _safe_int(metrics_source.get("qt_interval_ms", 0))
                qtc = _safe_int(metrics_source.get("qtc_ms", 0))

                hyper_entry = {
                    "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "file": os.path.abspath(filepath),
                    "HR_bpm": hr,
                    "RR_ms": int(60000 / hr) if hr > 0 else 0,
                    "PR_ms": pr,
                    "QRS_ms": qrs,
                    "QT_ms": qt,
                    "QTc_ms": qtc,
                    "Estimated_K": metrics_source.get("estimated_k", 0.0)
                }

                hyper_list = []
                if os.path.exists(hyper_metrics_path):
                    try:
                        with open(hyper_metrics_path, 'r') as f:
                            existing = json.load(f)
                            if isinstance(existing, list):
                                hyper_list = existing
                    except Exception:
                        hyper_list = []

                hyper_list.append(hyper_entry)

                with open(hyper_metrics_path, 'w') as f:
                    json.dump(hyper_list, f, indent=2)
                print(f" Saved Hyperkalemia metrics to {hyper_metrics_path}")
            except Exception as e:
                print(f" Could not save Hyperkalemia metrics: {e}")
            
            generate_hyperkalemia_report(filepath, self.analysis_results, self.lead_ii_data, sampling_rate=self.sampling_rate, raw_leads=raw_leads)
            
            print(f"\n Report generation completed!")
            
            print(f"✅ Hyperkalemia detection report saved successfully: {filepath}")

            # ── Create JSON twin for S3 sync ──────────────────────────────
            try:
                twin_path = os.path.splitext(filepath)[0] + '.json'
                from utils.ecg_payload_builder import build_hyperkalemia_payload
                twin_data = build_hyperkalemia_payload(
                    data=self.analysis_results if isinstance(self.analysis_results, dict) else {},
                    patient=patient,
                    signup_details=getattr(getattr(self, "dashboard_instance", None), "user_details", {}),
                    settings_manager=getattr(self, "settings_manager", None),
                    ecg_test_page=self,
                    raw_leads=raw_leads,
                    source_report_file=filepath,
                    hyperkalemia_findings=self.analysis_results.get('indicators', []) if isinstance(self.analysis_results, dict) else []
                )
                with open(twin_path, 'w') as jf:
                    json.dump(twin_data, jf, indent=2)
            except Exception as je:
                print(f"Error creating JSON twin: {je}")
                            
            try:
                h_pat = patient.copy() if patient else {}
                if 'patient_name' not in h_pat:
                    h_pat['patient_name'] = f"{h_pat.get('first_name','')} {h_pat.get('last_name','')}".strip()
                append_history_entry(
                    h_pat,
                    filepath,
                    report_type="Hyperkalemia",
                    username=self.username,
                    owner_full_name=(getattr(self.dashboard_instance, "user_details", {}) or {}).get("full_name") or self.username,
                )
            except Exception as hist_err:
                print(f" Failed to append Hyperkalemia history: {hist_err}")
            # ── Success popup ────────────────────────────────────────────────────
            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton
            dlg = QDialog(self)
            dlg.setWindowTitle("Report Generated")
            dlg.setMinimumWidth(480)
            dlg.setStyleSheet("""
                QDialog { background: #111827; border-radius: 12px; }
                QLabel  { color: #D1D5DB; font-size: 13px; font-family: 'Segoe UI', Arial; }
                QLabel#title { color: #00E676; font-size: 16px; font-weight: bold; }
                QPushButton { background: #1E2530; color: #D1D5DB; border: 1px solid #374151;
                              border-radius: 8px; padding: 8px 20px; font-size: 12px; font-weight: bold; font-family: 'Segoe UI', Arial; }
                QPushButton:hover { background: #252B3B; }
                QPushButton#open_btn { background: #3B82F6; color: white; border: none; }
                QPushButton#open_btn:hover { background: #2563EB; }
            """)
            vbox = QVBoxLayout(dlg)
            vbox.setSpacing(12)
            vbox.setContentsMargins(20, 20, 20, 20)
            title_lbl = QLabel("✅  Hyperkalemia Report Generated Successfully")
            title_lbl.setObjectName("title")
            vbox.addWidget(title_lbl)
            path_lbl = QLabel(f"<b>Saved at:</b><br>{filepath}")
            path_lbl.setWordWrap(True)
            vbox.addWidget(path_lbl)
            hint_lbl = QLabel("You can view this report on the <b>History</b> page.")
            vbox.addWidget(hint_lbl)
            btn_row = QHBoxLayout()
            open_btn = QPushButton("Open PDF")
            open_btn.setObjectName("open_btn")
            _fp = filepath
            def _open_pdf():
                from utils.platform_compat import open_file
                open_file(_fp)
            open_btn.clicked.connect(_open_pdf)
            ok_btn = QPushButton("OK")
            ok_btn.clicked.connect(dlg.accept)
            btn_row.addStretch()
            btn_row.addWidget(open_btn)
            btn_row.addWidget(ok_btn)
            vbox.addLayout(btn_row)
            dlg.finished.connect(lambda _result: self._reset_after_report_open())
            if hasattr(self, 'report_btn'):
                self.report_btn.setEnabled(False)
                self.report_btn.setText("Generate Report")
            dlg.exec_()
            
        except Exception as e:
            print(f"❌ Failed to generate Hyperkalemia report: {str(e)}")
            self.crash_logger.log_error(
                message=f"Hyperkalemia report generation error: {e}",
                exception=e,
                category="HYPERKALemia_REPORT_ERROR"
            )
    
    def closeEvent(self, event):
        """Handle window close event"""
        if self.is_capturing:
            reply = QMessageBox.question(
                self, "Capture in Progress",
                "Capture is still in progress. Do you want to stop and close?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                try:
                    self.stop_capture()
                except Exception as e:
                    print(f" Error stopping capture: {e}")
                
                # Explicitly stop BPM controller if it is running
                if hasattr(self, '_bpm_ctrl') and self._bpm_ctrl is not None:
                    try:
                        self._bpm_ctrl.stop()
                    except Exception as e:
                        print(f" Error stopping BPM controller: {e}")
                
                # Resume dashboard timers
                if hasattr(self, 'dashboard_instance') and self.dashboard_instance is not None:
                    try:
                        self.dashboard_instance.resume_dashboard_timers()
                    except Exception as e:
                        print(f" Error resuming dashboard timers: {e}")
                event.accept()
            else:
                event.ignore()
        else:
            # Explicitly stop BPM controller if it is running
            if hasattr(self, '_bpm_ctrl') and self._bpm_ctrl is not None:
                try:
                    self._bpm_ctrl.stop()
                except Exception as e:
                    print(f" Error stopping BPM controller: {e}")
            
            # Resume dashboard timers
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance is not None:
                try:
                    self.dashboard_instance.resume_dashboard_timers()
                except Exception as e:
                    print(f" Error resuming dashboard timers: {e}")
            event.accept()
