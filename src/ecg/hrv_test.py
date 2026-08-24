
import os
import time
import json
import numpy as np
from datetime import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QMessageBox,
    QSizePolicy, QFrame
)
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtCore import Qt, QTimer
# import pyqtgraph as pg  # Lazy loaded in methods
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

# Try to import serial
try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    print(" Serial module not available - HRV test hardware features disabled")
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
from ecg.ecg_filters import (
    apply_ac_filter,
    apply_emg_filter,
    apply_dft_filter,
    apply_baseline_wander_median_mean,
)
from ecg.utils.helpers import get_display_gain, build_raster_sweep_frame
from ecg.ui import display_updates as shared_display_updates
from dashboard.history_window import append_history_entry
from dashboard.dashboard import StyledMessageBox

# Import ECGTestPage + helpers to reuse EXACT same calculation + smoothing as 12‑lead test
try:
    from ecg.twelve_lead_test import ECGTestPage, SamplingRateCalculator, SerialStreamReader
    from PyQt5.QtWidgets import QStackedWidget
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


class HRVTestWindow(QWidget):
    """HRV Test Window - configurable-duration Lead II capture and report generation"""
    
    def __init__(self, parent=None, username=None, duration_minutes=5):
        super().__init__(parent)

        # Full screen on open
        self.showFullScreen()

        # Only show close button, disable minimize/maximize/restore
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.CustomizeWindowHint | Qt.MSWindowsFixedSizeDialogHint)
        # Prevent window from being moved/draggable
        self.setWindowFlag(Qt.WindowTitleHint, True)

        self.dashboard_instance = parent  # Store reference to dashboard
        self.username = username
        self.setWindowTitle("HRV Test - Lead II")
        self.setMinimumSize(1200, 700)
        self.setGeometry(100, 100, 1200, 700)
        # Set window flags to make it a separate window
        # self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowMaximizeButtonHint | Qt.WindowCloseButtonHint)
        self.setWindowModality(Qt.ApplicationModal)
        
        # Data storage - use circular buffer like 12-lead test
        HISTORY_LENGTH = 10000
        self.data = np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32)  # Circular buffer for selected lead
        self.captured_data = []  # Store all captured data with timestamps
        self.start_time = None
        try:
            self.duration_minutes = int(duration_minutes) if duration_minutes is not None else 5
        except Exception:
            self.duration_minutes = 5
        if self.duration_minutes <= 0:
            self.duration_minutes = 5

        self.capture_duration = self.duration_minutes * 60  # seconds
        self.is_capturing = False
        self.serial_reader = None
        self.crash_logger = get_crash_logger()
        
        # For adaptive scaling (simple, stable Y-axis for Lead II)
        self.y_center = 0.0
        self.y_range = 500.0  # Initial range
        self.sampling_rate = 500.0  # Default sampling rate, will be estimated
        self.sample_index = 0  # For synthetic time axis if needed
        self._plot_update_in_progress = False

        # Settings
        self.settings_manager = SettingsManager()

        # Selected lead for display (Lead II)
        self.selected_lead = "II"
        
        # Track active sample count to avoid skewing stats with leading zeros
        self.active_samples = 0
        
        # Create a minimal ECGTestPage instance to reuse its calculation methods
        # This ensures we use the EXACT same functions as the 12-lead test
        self.ecg_calculator = None
        if ECG_TEST_AVAILABLE:
            try:
                # Create a dummy stacked widget for ECGTestPage initialization
                dummy_stack = QStackedWidget()
                self.ecg_calculator = ECGTestPage("12 Lead ECG Test", dummy_stack, settings_manager=self.settings_manager)
                
                # IMPORTANT: Sync sampling rate from parent dashboard if available
                # This ensures both windows use identical frequency assumptions
                if parent and hasattr(parent, 'ecg_test_page'):
                    p_page = parent.ecg_test_page
                    if hasattr(p_page, 'sampler') and p_page.sampler.sampling_rate > 0:
                        self.sampling_rate = p_page.sampler.sampling_rate
                        print(f" Synced sampling rate from dashboard: {self.sampling_rate} Hz")
                
                # Set up minimal data structure (we only need Lead II, index 1)
                # ECGTestPage already initializes data, but we ensure it's set up
                if not hasattr(self.ecg_calculator, 'data') or len(self.ecg_calculator.data) < 12:
                    self.ecg_calculator.data = [np.zeros(HISTORY_LENGTH, dtype=np.float32) for _ in range(12)]
                
                # Ensure sampler exists with proper sampling rate
                if not hasattr(self.ecg_calculator, 'sampler'):
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = self.sampling_rate
                
                # CRITICAL: Assign a unique instance_id so this calculator
                # never pollutes or reads from the 12-lead test's smoothing buffers.
                # Without this, both share the 'twelve_lead' key in the module-level
                # _pr_buffers / _qrs_buffers / _qt_buffers / _qtc_buffers dicts.
                self.ecg_calculator._instance_id = 'hrv_test'

                print(" ECG calculator initialized for HRV test")
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
            print(f"[HRVTestWindow] HolterBPMController init failed: {_e}")
            self._bpm_ctrl = None

        # Initialize UI
        self.init_ui()
        self._last_displayed_bpm = 0
        
        # Timers
        self.capture_timer = QTimer(self)
        self.capture_timer.setTimerType(Qt.PreciseTimer)
        self.capture_timer.timeout.connect(self.update_plot)
        self.duration_timer = QTimer(self)
        self.duration_timer.timeout.connect(self.check_duration)

    def _minutes_word(self):
        return "minute" if int(self.duration_minutes) == 1 else "minutes"

    def mousePressEvent(self, event):
        # Block all mouse press events (left/right click) for dragging
        event.ignore()

    def mouseMoveEvent(self, event):
        # Block all mouse move events for dragging
        event.ignore()
        
    def init_ui(self):
        """Initialize the user interface"""
        import pyqtgraph as pg
        # Antialiasing roughly triples line-drawing cost; the scrolling trace moves
        # every point each frame, so on the 8 GB / i3 minimum spec that is the
        # difference between smooth and stuttering. Off there, on elsewhere.
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
        self.title_label = QLabel("HRV Test - Lead II")
        self.title_label.setFont(QFont("Segoe UI", 20, QFont.Bold))
        self.title_label.setStyleSheet("color: #FFFFFF; font-weight: 900; background: transparent;")
        header.addWidget(self.title_label)
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

        # Lead Selection
        lead_label = QLabel("Select Lead:")
        lead_label.setFont(QFont("Segoe UI", 11))
        lead_label.setStyleSheet("color: #9CA3AF; background: transparent;")
        controls.addWidget(lead_label)
        
        self.lead_combo = QComboBox()
        self.lead_combo.setMinimumWidth(140)
        self.lead_combo.setMinimumHeight(42)
        self.lead_combo.setCursor(Qt.PointingHandCursor)
        self.lead_combo.setStyleSheet("""
            QComboBox { padding: 6px 12px; border: 1px solid #374151; border-radius: 8px;
                        font: 11pt 'Segoe UI'; background: #1E2530; color: #F9FAFB; }
            QComboBox:focus { border: 1px solid #3B82F6; }
            QComboBox::drop-down { border: none; width: 30px; }
            QComboBox QAbstractItemView { background: #1E2530; color: #F9FAFB; border: 1px solid #374151; }
        """)
        self.lead_combo.addItems(["Lead I", "Lead II", "V1", "V2", "V3", "V4", "V5", "V6"])
        self.lead_combo.setCurrentText("Lead II")
        self.lead_combo.currentTextChanged.connect(self.on_lead_changed)
        controls.addWidget(self.lead_combo)
        
        controls.addSpacing(10)
        
        # Generate Report button (initially disabled)
        self.report_btn = QPushButton("Generate HRV Report")
        self.report_btn.setCursor(Qt.PointingHandCursor)
        self.report_btn.setStyleSheet("""
            QPushButton {
                background: #1E2D4A; color: #60A5FA; border-radius: 10px; padding: 10px 24px;
                font: bold 11pt 'Segoe UI'; border: 1px solid #3B82F6;
            }
            QPushButton:hover { background: #2A3F6B; }
            QPushButton:disabled { background: #151A25; color: #374151; border: 1px solid #2A3040; }
        """)
        self.report_btn.clicked.connect(self.generate_report)
        self.report_btn.setEnabled(False)
        controls.addWidget(self.report_btn)
        
        layout.addLayout(controls)
        
        # Metrics display section (below buttons, without Time)
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
            ("QRS Complex", "0", "ms", "qrs_duration"),
            ("QT/QTc", "0", "ms", "qtc_interval"),
        ]
        
        for title, value, unit, key in metric_info:
            box = QVBoxLayout()
            lbl = QLabel(title)
            lbl.setFont(QFont("Segoe UI", 11, QFont.Bold))
            lbl.setStyleSheet("color: #6B7280;")
            lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            val = QLabel(f"{value} {unit}")
            val.setFont(QFont("Segoe UI", 16, QFont.Bold))
            val.setStyleSheet("color: #FFFFFF;")
            val.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            box.addWidget(lbl)
            box.addWidget(val)
            metrics_layout.addLayout(box)
            self.metric_labels[key] = val  # Store reference for live update
        
        layout.addWidget(metrics_card)
        
        # Plot area
        plot_frame = QFrame()
        plot_frame.setStyleSheet("background: #000000; border-radius: 16px; border: 1px solid #333333;")
        plot_layout = QVBoxLayout(plot_frame)
        plot_layout.setContentsMargins(10, 10, 10, 10)
        
        # PyQtGraph plot
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setBackground("#000000")
        self.plot_widget.setMenuEnabled(False)
        self.plot_widget.setClipToView(True)
        # Matches the 12-lead test: no downsampling, plain polyline. Auto 'peak'
        # downsampling draws each pixel column as a min/max bar, which leaves the
        # background showing between the bars and speckles the trace.
        self.plot_widget.setDownsampling(ds=1, auto=False, mode='subsample')

        # Disable manual zoom/pan (amplitude lock)
        self.plot_widget.setMouseEnabled(x=False, y=False)
        self.plot_widget.hideButtons()  # Hide auto-scale button

        self.plot_widget.setStyleSheet("border: 1px solid #333333;")
        self.plot_widget.showGrid(x=False, y=False)
        self.plot_widget.hideAxis('left')
        self.plot_widget.hideAxis('bottom')
        
        # ── Scrolling monitor display ──────────────────────────────────────────
        # Layered curves for a realistic ECG glow effect:
        #   1. Outer glow  (dark green, thick)  — phosphor afterglow
        #   2. Inner trace (bright green, thin)  — actual ECG line
        #   3. Dot / 4. Gap eraser — kept for the raster sweep mode but left empty:
        #      the trace now scrolls, so there is no sweep head to mark and no
        #      band to erase ahead of it.

        self.plot_widget.setXRange(0, 2500, padding=0)
        self.plot_widget.setYRange(0, 4096, padding=0)

        # Layer 1 — phosphor glow (drawn first = behind)
        self.plot_curve_glow = self.plot_widget.plot(
            pen=pg.mkPen(color='#003300', width=5)
        )
        # Layer 2 — bright ECG trace (drawn on top of glow)
        self.plot_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#00FF00', width=2.0), connect='finite'
        )
        # Layer 3 — sweep head dot
        self.sweep_dot = self.plot_widget.plot(
            pen=None, symbol='o',
            symbolSize=7,
            symbolBrush=pg.mkBrush('#00FF41'),
            symbolPen=pg.mkPen('#00FF41', width=1)
        )
        # Layer 4 — black eraser gap ahead of sweep head
        self.sweep_gap_curve = self.plot_widget.plot(
            pen=pg.mkPen(color='#000000', width=14)
        )

        # Sweep state variables
        self._sweep_buf  = np.full(2500, 2048.0, dtype=float)  # circular display buffer
        self._sweep_pos  = 0                                    # current write index (0-2499)
        self._sweep_gap  = 80                                   # eraser width in samples
        
        plot_layout.addWidget(self.plot_widget)
        layout.addWidget(plot_frame, stretch=1)
        
        # Info label
        current_lead = self.lead_combo.currentText()
        m = int(self.duration_minutes)
        mw = self._minutes_word()
        self.info_label = QLabel(
            f"Capture {m} {mw} of {current_lead} data for HRV analysis. "
            f"The capture will stop automatically after {m} {mw}."
        )
        self.info_label.setFont(QFont("Segoe UI", 10))
        self.info_label.setStyleSheet("color: #4B5563; padding: 10px; background: transparent;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

    def on_lead_changed(self, text):
        """Handle lead selection change"""
        if text == "Lead I":
            self.selected_lead = "I"
        elif text == "Lead II":
            self.selected_lead = "II"
        else:
            self.selected_lead = text
            
        self.title_label.setText(f"HRV Test - {text}")

        m = int(self.duration_minutes)
        mw = self._minutes_word()
        self.info_label.setText(
            f"Capture {m} {mw} of {text} data for HRV analysis. "
            f"The capture will stop automatically after {m} {mw}."
        )
        
    def refresh_com_ports(self):
        """Refresh available COM ports"""
        pass
    
    def start_capture(self):
        """Start capturing selected lead data"""
        # CHECK: Ensure no other test is running
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            # Check if dashboard has the can_start_test method
            if hasattr(self.dashboard_instance, 'can_start_test'):
                if not self.dashboard_instance.can_start_test("hrv_test"):
                    return
                # Set state to running
                self.dashboard_instance.update_test_state("hrv_test", True)

        if not SERIAL_AVAILABLE or not ECG_TEST_AVAILABLE:
            QMessageBox.warning(self, "Serial Not Available", 
                              "Serial/ECG modules are not available. Please install pyserial and restart.")
            return
        
        # FLUSH stale smoothing state from any previous capture.
        # This prevents old 12-lead (or prior HRV) interval values from
        # bleeding into the new session via module-level buffer dicts.
        try:
            from ecg.ecg_calculations import cleanup_instance
            cleanup_instance('hrv_test')
        except Exception:
            pass
        # Also clear instance-level smoothing buffers on the calculator
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

        # Clear shared display cache so HRV never shows stale QT/QTc or other
        # carried-over values from a previous 12-lead / HRV session.
        try:
            shared_display_updates._last_valid.clear()
        except Exception:
            pass
        self._last_metric_update_ts = 0.0

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
            print(" Starting HRV Test in Demo Mode...")
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
                        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                            self.dashboard_instance.update_test_state("hrv_test", False)
                        return
                except Exception as scan_err:
                    print(f" Port scan failed: {scan_err}")
                    QMessageBox.warning(self, "Scan Failed", f"Port scan failed: {scan_err}")
                    return
            
            try:
                from ecg.serial.serial_reader import GlobalHardwareManager
                self.serial_reader = GlobalHardwareManager().get_reader(port_to_use, baudrate)
            except Exception as e:
                print(f" Error setting up serial reader: {e}")
                QMessageBox.critical(self, "Error", f"Failed to open serial port: {e}")
                return

        try:
            # Start/Resume acquisition.
            self.serial_reader.start()

            # Centralized authorization check
            from utils.license_manager import is_ecg_acquisition_allowed
            if not is_ecg_acquisition_allowed(self):
                self.stop_capture()
                self.status_label.setText("Status: Unauthorized Device")
                self.status_label.setStyleSheet("color: #EF5350; padding: 5px;")
                return
            
            # Reset data
            HISTORY_LENGTH = 10000
            self.data = np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32)
            self.captured_data = []
            self.sample_index = 0
            self.active_samples = 0
            self.start_time = time.time()
            self.is_capturing = True
            # Reset sweep display buffer so next capture starts from left edge
            self._sweep_buf[:] = 2048.0
            self._sweep_pos = 0

            # Reset display baseline anchor so centering adapts fresh for this session
            if hasattr(self, "_hrv_display_anchor"):
                del self._hrv_display_anchor
            # Baseline/DFT + 25 Hz EMG filters need ~2.5 s before output is stable
            self._display_warmup_samples = int(max(600, self.sampling_rate * 2.5))
            self._display_settle_samples = int(max(250, self.sampling_rate * 0.5))
            self._display_anchor_locked = False
            self._display_ready_announced = False
            self._last_sweep_y = 2048.0

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
            
            # Reset smoothing buffers
            if self.ecg_calculator:
                if hasattr(self.ecg_calculator, 'smoothing_buffers'):
                    self.ecg_calculator.smoothing_buffers = {}
                # ALWAYS reset data buffer to avoid step function from previous captures
                self.ecg_calculator.data = [np.full(HISTORY_LENGTH, 2048.0, dtype=np.float32) for _ in range(12)]
                
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
            self.report_btn.setEnabled(False)
            self.lead_combo.setEnabled(False)

            # Reset RA/RL/LL guard flag for this new capture session
            self._ra_rl_ll_popup_shown = False
            self._ra_rl_ll_disconnected_count = 0

            # Reset the visible metric labels immediately at capture start so the
            # display waits for fresh HRV calculations instead of showing old values.
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
                    if not hasattr(self, '_bpm_refresh_timer'):
                        self._bpm_refresh_timer = QTimer()
                        self._bpm_refresh_timer.timeout.connect(self._refresh_holter_bpm_label)
                    if not self._bpm_refresh_timer.isActive():
                        self._bpm_refresh_timer.start(2000)
            except Exception as _bpm_err:
                print(f"[HRVTestWindow] BPM controller start error: {_bpm_err}")

            # Lock display interaction during capture
            self.plot_widget.setMouseEnabled(x=False, y=False)

            self.status_label.setText("Status: Capturing from RhythmUltra Device...")
            self.status_label.setStyleSheet("color: #00E676; padding: 5px;")
            self._silent_data_warned = False
            
            # Start timers
            self.capture_timer.start(50 if is_low_spec_mode() else 30)  # 20 FPS on low-spec, 33 FPS on normal
            self.duration_timer.start(1000)
            self.metrics_timer = QTimer(self)
            self.metrics_timer.timeout.connect(self.update_metrics)
            self.metrics_timer.start(500 if is_low_spec_mode() else 200)
            
            StyledMessageBox.show_message(self, "Capture Started",
                                  f"{self.selected_lead} capture started. It will automatically stop after {int(self.duration_minutes)} {self._minutes_word()}.",
                                  is_critical=False, auto_close_ms=500)
            
        except Exception as e:
            StyledMessageBox.show_message(self, "Error",
                               f"Failed to start capture: {str(e)}",
                               is_critical=True)
            self.crash_logger.log_error(
                message=f"HRV test capture start error: {e}",
                exception=e,
                category="HRV_TEST_ERROR"
            )
    
    def _handle_ra_rl_ll_disconnect(self):
        """
        RA/RL/LL(F) electrode loss makes every lead invalid (same as 12-lead test).
        Alert the user with a blocking popup, send STOP command, and close window.
        Fires once per disconnection event.
        """
        if getattr(self, "_ra_rl_ll_popup_shown", False):
            return  # Guard: fire only once per disconnection
        
        self._ra_rl_ll_popup_shown = True

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
            print(f" Error stopping capture on RA/RL/LL disconnect: {e}")
        
        # Close test window and return to dashboard
        try:
            self.close()
        except Exception as e:
            print(f" Error closing HRV window after RA/RL/LL disconnect: {e}")

    def stop_capture(self, device_disconnected=False, device_not_sending=False):
        """Stop capturing data"""
        # UPDATE STATE: Test stopped
        if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
            if hasattr(self.dashboard_instance, 'update_test_state'):
                self.dashboard_instance.update_test_state("hrv_test", False)

        self.is_capturing = False
        
        if self.serial_reader:
            try:
                if hasattr(self.serial_reader, "command_handler") and self.serial_reader.command_handler:
                    self.serial_reader.command_handler.send_stop_command()
                # Reset running state so other tests can send START command
                self.serial_reader.running = False
            except Exception as e:
                print(f"[HRVTest] Error sending stop command: {e}")
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
            print(f"[HRVTestWindow] BPM controller stop error: {_bpm_err}")
        
        # Stop timers
        self.capture_timer.stop()
        self.duration_timer.stop()
        if hasattr(self, 'metrics_timer'):
            self.metrics_timer.stop()
        
        # Update UI based on reason
        if device_disconnected:
            self.start_btn.setEnabled(False)
            self.stop_btn.setEnabled(False)
            self.lead_combo.setEnabled(False)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device disconnected")
            
            # Reset dashboard metrics and interpretation
            if hasattr(self, 'dashboard_instance') and self.dashboard_instance:
                if hasattr(self.dashboard_instance, 'reset_metrics_and_interpretation'):
                    self.dashboard_instance.reset_metrics_and_interpretation()
        elif device_not_sending:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.lead_combo.setEnabled(True)
            self.report_btn.setEnabled(False)
            self.status_label.setText("Status: Device connected but not sending data")
        else:
            self.start_btn.setEnabled(True)
            self.stop_btn.setEnabled(False)
            self.lead_combo.setEnabled(True)

            if len(self.captured_data) > 0:
                self.report_btn.setEnabled(True)
                self.status_label.setText(f"Status: Capture Complete")
            else:
                self.status_label.setText("Status: Capture Stopped (No data)")
        
        # Re-enable display interaction after capture (optional, but allows inspection)
        self.plot_widget.setMouseEnabled(x=True, y=True)
        
        self.status_label.setStyleSheet("color: #6B7280; padding: 5px;")
        self.timer_label.setText("Time: 00:00")

    def _reset_after_report_open(self):
        """Clear the completed HRV session so the next test starts fresh."""
        self.is_capturing = False
        self.captured_data = []
        self._hrv_plot_buffer = np.array([], dtype=np.float32)
        try:
            self.plot_curve.setData([], [])
        except Exception:
            pass
        try:
            if hasattr(self, "plot_widget"):
                self.plot_widget.setXRange(0.0, 1.0, padding=0)
                self.plot_widget.setYRange(0, 4096, padding=0)
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
            self.lead_combo.setEnabled(True)
            self.report_btn.setEnabled(False)
            self.report_btn.setText("Generate HRV Report")
        except Exception:
            pass
        self._refresh_holter_bpm_label()

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
        """Check if configured duration has elapsed"""
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
                                  f"{int(self.duration_minutes)}-minute capture completed successfully!")
    
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

    def update_plot(self):
        """Update the plot with new data"""
        import pyqtgraph as pg
        if getattr(self, "_plot_update_in_progress", False):
            return
        self._plot_update_in_progress = True
        if not self.is_capturing or not self.serial_reader:
            self._plot_update_in_progress = False
            return

        # Check if device got disconnected suddenly
        if not self.serial_reader.running:
            print("⚠️ Device disconnected during HRV test!")
            self.stop_capture(device_disconnected=True)
            self._plot_update_in_progress = False
            return
            
        
        try:
            n_new = 0
            # Read multiple packets per GUI tick (same idea as 12‑lead test)
            # so we don't under‑sample and miss beats when HR changes.
            max_packets = 100
            
            # Use new packet-based reading from SerialStreamReader
            packets = self.serial_reader.read_packets(max_packets=max_packets)
            
            # ── RA/RL/LL(F) LEAD DISCONNECTION DETECTION ────────────────────────
            # Check if RA, RL, or LL(F) electrodes are disconnected (same as 12-lead test)
            # These are critical electrodes - if they're off, ALL leads are invalid
            try:
                if packets and len(packets) > 0:
                    # Check the most recent packet for RA/RL/LL presence flags
                    last_packet = packets[-1]
                    ra_present = last_packet.get("__ra_present__", True)
                    rl_absent = not last_packet.get("__rl_present__", True)
                    ll_present = last_packet.get("__ll_present__", True)
                    
                    # Track RA/RL/LL state for debouncing
                    if not hasattr(self, '_ra_rl_ll_disconnected_count'):
                        self._ra_rl_ll_disconnected_count = 0
                    
                    # If RA is absent OR RL is absent OR LL is absent, increment counter
                    if not ra_present or rl_absent or not ll_present:
                        self._ra_rl_ll_disconnected_count += 1
                        
                        # Require 5 consecutive packets (~10ms) before triggering alert
                        if self._ra_rl_ll_disconnected_count >= 5:
                            # Check guard flag (fire only once per event)
                            if not getattr(self, '_ra_rl_ll_popup_shown', False):
                                print("⚠️ Critical electrode (RA/RL/LL) disconnected during HRV test!")
                                self._handle_ra_rl_ll_disconnect()
                                self._plot_update_in_progress = False
                                return
                    else:
                        # Reset counter if all electrodes are connected
                        self._ra_rl_ll_disconnected_count = 0
            except Exception as e:
                print(f" Error checking RA/RL/LL status: {e}")
            
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
            
            # Leads required by HRV PDF header calculations:
            # - P/QRS/T axis: needs Lead I + Lead aVF
            # - RV5/SV1: needs V1 + V5
            _required_header_leads = {"I", "aVF", "V1", "V5"}

            for packet in packets:
                # Packet is a dictionary with lead names as keys (e.g., {"I": value, "II": value, ...})
                # Extract selected lead directly from the packet (used for HRV metrics + display).
                lead_value = packet.get(self.selected_lead, None)

                # ── Feed packet to HolterBPMController ──────────────────────────
                try:
                    if self._bpm_ctrl is not None:
                        self._bpm_ctrl.push(packet)
                except Exception:
                    pass
                
                # 1) ANALYSIS BUFFERS (Must be RAW for accurate interval calculation)
                # Always buffer the leads required for HRV PDF header computations.
                if self.ecg_calculator:
                    try:
                        # Map lead name to index
                        lead_indices = {
                            "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                            "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                        }

                        leads_to_write = set(_required_header_leads)
                        leads_to_write.add(self.selected_lead)

                        for lead_name in leads_to_write:
                            if lead_name not in packet:
                                continue
                            raw = packet.get(lead_name, None)
                            if raw is None:
                                continue
                            lead_value_f = float(raw)
                            lead_idx = lead_indices.get(lead_name, None)
                            if lead_idx is None or lead_idx >= len(self.ecg_calculator.data):
                                continue
                            self.ecg_calculator.data[lead_idx] = np.roll(self.ecg_calculator.data[lead_idx], -1)
                            self.ecg_calculator.data[lead_idx][-1] = lead_value_f

                        # FORCE UPDATE LEAD II (Index 1) if the selected lead is not II,
                        # to keep the dashboard-style lead II buffer aligned with the selected lead.
                        if self.selected_lead != "II" and lead_value is not None:
                            lead_value_f = float(lead_value)
                            self.ecg_calculator.data[1] = np.roll(self.ecg_calculator.data[1], -1)
                            self.ecg_calculator.data[1][-1] = lead_value_f

                    except Exception as e:
                        print(f" Error updating calculator buffer: {e}")

                if lead_value is None:
                    lead_value = 2048.0
                
                lead_value = float(lead_value)
                self.active_samples = min(len(self.data), self.active_samples + 1)
                # 2. DISPLAY DATA (Use RAW data for display as requested)
                # User requested raw plot centered at 2048
                # We skip smoothing for display to show raw signal
                
                # Define smoothed_value as raw value (since we are skipping smoothing)
                # This ensures subsequent code (report storage) works and matches display
                smoothed_value = lead_value
                
                # Update local circular buffer for plot
                self.data = np.roll(self.data, -1)
                self.data[-1] = lead_value
                n_new += 1
                
                if self.ecg_calculator and hasattr(self.ecg_calculator, "sampler"):
                    try:
                        sr = self.ecg_calculator.sampler.add_sample()
                    except Exception:
                        sr = 0.0
                    if sr and sr > 0:
                        safe_sr = float(sr)
                        if safe_sr < 50.0 or safe_sr > 1000.0:
                            safe_sr = self.sampling_rate
                        self.sampling_rate = safe_sr
                
                # Store data point with timestamp for final report generation
                elapsed = time.time() - self.start_time
                self.captured_data.append({
                    'time': elapsed,
                    'value': smoothed_value  # Reports use smoothed values for clean graphs
                })
                
            if not self.is_capturing:
                self._plot_update_in_progress = False
                return
            # Update plot using a stable sliding window.
            # Do not drop zero-valued samples here because real ECG data can cross zero,
            # and filtering them out makes the trace reflow and jerk.
            valid_count = min(self.active_samples, len(self.data))
            if valid_count <= 0:
                return

            fs = float(self.sampling_rate) if self.sampling_rate > 0 else 500.0
            window_seconds = 10.0
            window_samples = max(50, int(window_seconds * fs))
            window_samples = min(window_samples, valid_count)

            buffer_data = np.asarray(self.data[-window_samples:], dtype=float)

            warmup_needed = getattr(self, "_display_warmup_samples", int(2.5 * fs))
            settle_needed = warmup_needed + getattr(self, "_display_settle_samples", int(0.5 * fs))
            display_ready = valid_count >= settle_needed

            if len(buffer_data) > 5 and display_ready:
                # Default clinical filters: AC 50 Hz, EMG 25 Hz low-pass, 0.5 Hz baseline
                ac_val = "50"
                emg_val = "25"
                dft_val = "0.5"

                # Pad with enough values so startup filtering begins cleanly instead of
                # showing a raw transient while the display buffer is still warming up.
                pad_len = min(int(fs * 2.0), max(0, len(buffer_data) - 1))
                if pad_len > 0:
                    padded_data = np.pad(buffer_data, (pad_len, pad_len), mode='reflect')
                else:
                    padded_data = buffer_data
                    
                if ac_val != "Off" and ac_val != "off":
                    padded_data = apply_ac_filter(padded_data, fs, ac_val)

                if emg_val != "Off" and emg_val != "off":
                    padded_data = apply_emg_filter(padded_data, fs, emg_val)

                # Apply DFT (baseline) Filter
                if dft_val not in ("Off", "off", "", None):
                    dft_text = str(dft_val).strip()
                    if dft_text == "0.5":
                        padded_data = apply_baseline_wander_median_mean(padded_data, fs)
                    else:
                        padded_data = apply_dft_filter(padded_data, fs, dft_text)

                    padded_data = padded_data + 2048.0

                    # Remove edge artifacts introduced by the 0.5 Hz baseline filter.
                    # This mirrors the 12-lead display behavior and prevents left/right
                    # wobble during the first and last few seconds of the visible window.
                    if dft_text == "0.5":
                        edge_trim = int(0.75 * fs)
                        min_keep = max(50, pad_len * 2 + 20)
                        if edge_trim > 0 and len(padded_data) > (2 * edge_trim + min_keep):
                            padded_data = padded_data[edge_trim:-edge_trim]

                if pad_len > 0:
                    buffer_data = padded_data[pad_len:-pad_len]
                else:
                    buffer_data = padded_data

                # Gentle smoothing for HRV display so the trace looks continuous.
                buffer_data = gaussian_filter1d(buffer_data, sigma=0.8)

            if len(buffer_data) > 0 and display_ready:
                if not getattr(self, "_display_ready_announced", False):
                    # Start sweep from a clean baseline once filters have fully settled
                    self._sweep_buf[:] = 2048.0
                    self._sweep_pos = 0
                    self._last_sweep_y = 2048.0
                    self._display_anchor_locked = False
                    self._display_ready_announced = True
                # Center the waveform in the middle of the plot so it stays visually stable
                # across devices with different ADC offsets.
                if not getattr(self, "_display_anchor_locked", False):
                    self._hrv_display_anchor = float(np.nanmedian(buffer_data)) if len(buffer_data) else 2048.0
                    self._display_anchor_locked = True
                elif not hasattr(self, "_hrv_display_anchor"):
                    self._hrv_display_anchor = float(np.nanmedian(buffer_data)) if len(buffer_data) else 2048.0
                    self._display_anchor_locked = True

                gain_factor = get_display_gain(self.settings_manager.get_wave_gain())
                centered = (buffer_data - self._hrv_display_anchor) * gain_factor
                display_values = np.clip(2048.0 + centered, 0, 4096)

                # ── Scrolling render (no raster sweep) ────────────────────────
                # The trace slides right-to-left with the newest sample pinned to
                # the right edge, so there is no eraser bar travelling across the
                # waveform. Samples are still written into the circular buffer
                # below; only the presentation differs.
                SWEEP_N = 2500

                if n_new > 0 and len(display_values) > 0:
                    samples_to_push = min(n_new, len(display_values))
                    new_vals = display_values[-samples_to_push:]
                    max_step = 180.0  # reject single-sample spikes from packet glitches
                    for v in new_vals:
                        y = float(np.clip(v, 0, 4096))
                        # A non-finite sample would sit in the buffer forever and,
                        # because the curve uses connect='finite', print as a black
                        # dot in the trace. np.clip leaves NaN as NaN and the
                        # max_step guard below cannot catch it, so hold the last
                        # good value instead.
                        if not np.isfinite(y):
                            y = self._last_sweep_y
                        if abs(y - self._last_sweep_y) > max_step:
                            y = self._last_sweep_y + float(np.clip(y - self._last_sweep_y, -max_step, max_step))
                        self._sweep_buf[self._sweep_pos] = y
                        self._last_sweep_y = y
                        self._sweep_pos = (self._sweep_pos + 1) % SWEEP_N

                pos = self._sweep_pos
                buf = self._sweep_buf
                x_axis = np.arange(SWEEP_N, dtype=float)

                # Rotate the circular buffer so the oldest sample is on the left
                # and the newest lands on the right edge. Nothing is blanked, so
                # the waveform is continuous across the whole width.
                y_display = np.roll(buf, -int(pos))
                y_display = np.nan_to_num(y_display, nan=2048.0, posinf=4096.0, neginf=0.0)

                self.plot_curve_glow.setData(x_axis, y_display)
                self.plot_curve.setData(x_axis, y_display)
                # Eraser bar and sweep head belong to the raster mode only.
                self.sweep_gap_curve.setData([], [])
                self.sweep_dot.setData([], [])

                self.plot_widget.setXRange(0, SWEEP_N, padding=0)
                self.plot_widget.setYRange(0, 4096, padding=0)
        
        except Exception as e:
            # Silently handle errors during capture
            pass
        finally:
            self._plot_update_in_progress = False
    
    def _show_loader(self):
        """Show an animated loading dialog. Returns the dialog — caller must close it."""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QLabel, QApplication
        from PyQt5.QtCore import QTimer, Qt
        from PyQt5.QtGui import QFont

        dlg = QDialog(self)
        dlg.setWindowTitle("Generating Report")
        dlg.setWindowFlags(Qt.Dialog | Qt.CustomizeWindowHint | Qt.WindowTitleHint)
        dlg.setMinimumSize(380, 180)
        dlg.setStyleSheet("""
            QDialog {
                background: #111827;
                border-radius: 16px;
            }
            QLabel#spinner {
                font-size: 36px;
            }
            QLabel#msg {
                color: #F9FAFB;
                font-size: 14px;
                font-family: 'Segoe UI', Arial;
                font-weight: bold;
            }
            QLabel#sub {
                color: #6B7280;
                font-size: 11px;
                font-family: 'Segoe UI', Arial;
            }
        """)

        layout = QVBoxLayout(dlg)
        layout.setSpacing(6)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setAlignment(Qt.AlignCenter)

        spinner_lbl = QLabel("⌛")
        spinner_lbl.setObjectName("spinner")
        spinner_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(spinner_lbl)

        msg_lbl = QLabel("Generating HRV Report…")
        msg_lbl.setObjectName("msg")
        msg_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(msg_lbl)

        sub_lbl = QLabel("Please wait, this may take a few seconds.")
        sub_lbl.setObjectName("sub")
        sub_lbl.setAlignment(Qt.AlignCenter)
        layout.addWidget(sub_lbl)

        # Animate spinner frames
        _frames = ["⌛", "⏳"]
        _idx = [0]
        _anim_timer = QTimer(dlg)

        def _tick():
            _idx[0] = (_idx[0] + 1) % len(_frames)
            spinner_lbl.setText(_frames[_idx[0]])

        _anim_timer.timeout.connect(_tick)
        _anim_timer.start(900 if is_low_spec_mode() else 500)
        dlg._anim_timer = _anim_timer   # keep alive

        dlg.show()
        QApplication.processEvents()    # paint immediately
        return dlg

    def generate_report(self):
        """Generate HRV report without blocking the UI."""
        if len(self.captured_data) == 0:
            QMessageBox.warning(self, "No Data", "No data available to generate report.")
            return

        runner = getattr(self, "_pdf_runner", None)
        if runner is not None and runner.is_running():
            QMessageBox.information(self, "Please wait", "HRV report is already being generated.")
            return

        # Disable button immediately so user cannot double-click
        if hasattr(self, 'report_btn'):
            self.report_btn.setEnabled(False)
            self.report_btn.setText("Generating…")

        # Show loader instantly
        self._loader_dlg = self._show_loader()

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
        except Exception:
            patient = {}

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
        filepath = os.path.join(reports_dir, f"HRV_Report{serial_part}_{report_stamp}.pdf")

        # Snapshots — safe to pass to a worker thread
        _patient_snap = patient.copy()
        _captured_snap = list(self.captured_data)
        _lead_snap = self.selected_lead
        _sr_snap = self.sampling_rate
        _last_hr = getattr(self, 'last_heart_rate', 0)
        _ecg_calc = self.ecg_calculator
        _settings = self.settings_manager

        def _parse_first_int(text):
            import re
            if text is None:
                return 0
            match = re.search(r'(\d+)', str(text))
            return int(match.group(1)) if match else 0

        def _capture_display_metrics():
            metrics = {
                "HR": 0,
                "PR": 0,
                "QRS": 0,
                "QT": 0,
                "QTc": 0,
                "ST": 0,
            }
            try:
                if hasattr(self, 'metric_labels'):
                    hr_text = self.metric_labels.get('heart_rate').text() if 'heart_rate' in self.metric_labels else ""
                    pr_text = self.metric_labels.get('pr_interval').text() if 'pr_interval' in self.metric_labels else ""
                    qrs_text = self.metric_labels.get('qrs_duration').text() if 'qrs_duration' in self.metric_labels else ""
                    qtc_text = self.metric_labels.get('qtc_interval').text() if 'qtc_interval' in self.metric_labels else ""

                    metrics["HR"] = _parse_first_int(hr_text)
                    metrics["PR"] = _parse_first_int(pr_text)
                    metrics["QRS"] = _parse_first_int(qrs_text)

                    # QT/QTc card is shown as "QT/QTc" on the UI, so split the pair.
                    if qtc_text:
                        parts = str(qtc_text).replace("ms", "").split("/")
                        if len(parts) >= 2:
                            metrics["QT"] = _parse_first_int(parts[0])
                            metrics["QTc"] = _parse_first_int(parts[1])
                        else:
                            metrics["QT"] = _parse_first_int(qtc_text)
                            metrics["QTc"] = _parse_first_int(qtc_text)

                    # Keep the report's ST value aligned with the calculator if available.
                    metrics["ST"] = _parse_first_int(getattr(_ecg_calc, "last_st_interval", 0))
            except Exception:
                pass

            if metrics["HR"] <= 0:
                metrics["HR"] = int(_last_hr or 0)
            return metrics

        _display_metrics_snap = _capture_display_metrics()

        from PyQt5.QtCore import QThread, pyqtSignal as _Signal

        class _PrepWorker(QThread):
            done = _Signal(dict, str, dict)
            failed = _Signal(str)

            def __init__(self, captured_data, ecg_calculator, lead, sampling_rate,
                         last_hr, filepath, patient, settings_manager, display_metrics):
                super().__init__()
                self._captured = captured_data
                self._calc = ecg_calculator
                self._lead = lead
                self._sr = sampling_rate
                self._last_hr = last_hr
                self._filepath = filepath
                self._patient = patient
                self._sm = settings_manager
                self._display_metrics = display_metrics or {}

            def run(self):
                try:
                    hr_value = 0
                    pr_value = 0
                    qrs_value = 0
                    st_value = 0
                    qt_value = 0
                    qtc_value = 0
                    hr_max = 0
                    hr_min = 0
                    hr_avg = 0

                    if self._calc and len(self._captured) >= 200:
                        try:
                            lead_indices = {
                                "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                                "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                            }
                            lead_idx = lead_indices.get(self._lead, 1)
                            num_samples = min(2000, len(self._captured))
                            recent_data = [d['value'] for d in self._captured[-num_samples:]]
                            signal = np.array(recent_data, dtype=np.float32)

                            buffer_size = max(len(signal), 1000)
                            if len(self._calc.data[lead_idx]) < buffer_size:
                                self._calc.data[lead_idx] = np.zeros(buffer_size, dtype=np.float32)

                            if len(signal) <= len(self._calc.data[lead_idx]):
                                self._calc.data[lead_idx][-len(signal):] = signal
                            else:
                                self._calc.data[lead_idx] = signal[-len(self._calc.data[lead_idx]):]

                            if hasattr(self._calc, 'sampler') and self._calc.sampler:
                                self._calc.sampler.sampling_rate = self._sr

                            hr_value = self._calc.calculate_heart_rate(self._calc.data[lead_idx])
                            pr_value = self._calc.calculate_pr_interval(self._calc.data[lead_idx])
                            qrs_value = self._calc.calculate_qrs_duration(self._calc.data[lead_idx])
                            st_value = self._calc.calculate_st_interval(self._calc.data[lead_idx])
                            qt_value = self._calc.calculate_qt_interval(self._calc.data[lead_idx])

                            if qrs_value <= 0 and hasattr(self._calc, 'last_qrs_duration'):
                                qrs_value = int(getattr(self._calc, 'last_qrs_duration', 0) or 0)
                            if qt_value <= 0 and hasattr(self._calc, 'last_qt_interval'):
                                qt_value = int(getattr(self._calc, 'last_qt_interval', 0) or 0)
                            if qtc_value <= 0 and hasattr(self._calc, 'last_qtc_interval'):
                                qtc_value = int(getattr(self._calc, 'last_qtc_interval', 0) or 0)
                            if pr_value <= 0 and hasattr(self._calc, 'last_pr_interval'):
                                pr_value = int(getattr(self._calc, 'last_pr_interval', 0) or 0)

                            all_hr_values = []
                            window_size = 200
                            for i in range(0, len(self._captured) - window_size, window_size // 2):
                                window_data = [d['value'] for d in self._captured[i:i + window_size]]
                                window_signal = np.array(window_data, dtype=np.float32)
                                if len(self._calc.data[lead_idx]) >= len(window_signal):
                                    self._calc.data[lead_idx][-len(window_signal):] = window_signal
                                    hr = self._calc.calculate_heart_rate(self._calc.data[lead_idx])
                                    if hr > 0:
                                        all_hr_values.append(hr)

                            if all_hr_values:
                                hr_max = max(all_hr_values)
                                hr_min = min(all_hr_values)
                                hr_avg = int(np.mean(all_hr_values))

                        except Exception as e:
                            print(f" Error calculating metrics in prep worker: {e}")

                    if self._last_hr > 0:
                        hr_value = self._last_hr
                        hr_avg = hr_value
                        if hr_max < hr_value:
                            hr_max = hr_value
                        if hr_min > hr_value or hr_min == 0:
                            hr_min = hr_value

                    # Prefer the exact values the user saw on screen when the
                    # test ended.
                    if self._display_metrics:
                        hr_value = int(self._display_metrics.get("HR", hr_value) or hr_value)
                        pr_value = int(self._display_metrics.get("PR", pr_value) or pr_value)
                        qrs_value = int(self._display_metrics.get("QRS", qrs_value) or qrs_value)
                        qt_value = int(self._display_metrics.get("QT", qt_value) or qt_value)
                        qtc_value = int(self._display_metrics.get("QTc", qtc_value) or qtc_value)
                        st_value = int(self._display_metrics.get("ST", st_value) or st_value)

                    data = {
                        "HR": hr_value,
                        "beat": hr_value,
                        "PR": pr_value,
                        "QRS": qrs_value,
                        "QT": qt_value,
                        "QTc": qtc_value,
                        "ST": st_value,
                        "HR_max": hr_max,
                        "HR_min": hr_min,
                        "HR_avg": hr_avg,
                        "Heart_Rate": hr_value,
                    }

                    try:
                        rr_interval_s = float(data.get("RR", 0) or 0) / 1000.0
                        if rr_interval_s <= 0 and hr_value > 0:
                            rr_interval_s = 60.0 / hr_value
                        qtc_frid_ms = (qt_value / (np.cbrt(rr_interval_s))) if rr_interval_s > 0 else 0
                        data["QTc_Fridericia"] = qtc_frid_ms
                    except Exception:
                        data["QTc_Fridericia"] = 0

                    self.done.emit(data, self._filepath, self._patient)

                except Exception as exc:
                    self.failed.emit(str(exc))

        def _on_success(fname, metrics_entry=None):
            print(f"✅ HRV ECG report saved successfully: {fname}")
            
             # ── Create JSON twin for S3 sync ──────────────────────────────
            try:
                import json
                twin_path = os.path.splitext(fname)[0] + '.json'
                from utils.ecg_payload_builder import build_hrv_payload
                
                # Extract raw values from captured data for the selected lead
                raw_values = [d['value'] for d in _captured_snap]
                raw_leads_map = {getattr(self, "selected_lead", "II"): raw_values}
                
                # Merge display metrics with the detailed HRV metrics
                merged_data = {**_display_metrics_snap, **(metrics_entry or {})}
                
                twin_data = build_hrv_payload(
                    data=merged_data,
                    patient=_patient_snap,
                    signup_details=getattr(getattr(self, "dashboard_instance", None), "user_details", {}),
                    settings_manager=getattr(self, "settings_manager", None),
                    raw_leads=raw_leads_map,
                    source_report_file=fname,
                    selected_lead=getattr(self, "selected_lead", "II"),
                    hrv_metrics=metrics_entry
                )
                with open(twin_path, 'w') as jf:
                    json.dump(twin_data, jf, indent=2)
            except Exception as je:
                print(f"Error creating JSON twin: {je}")
                
            try:
                h_pat = _patient_snap.copy()
                if 'patient_name' not in h_pat:
                    h_pat['patient_name'] = f"{h_pat.get('first_name', '')} {h_pat.get('last_name', '')}".strip()
                append_history_entry(
                    h_pat,
                    fname,
                    report_type="HRV",
                    username=self.username,
                    owner_full_name=(getattr(self.dashboard_instance, "user_details", {}) or {}).get("full_name") or self.username,
                )
            except Exception as hist_err:
                print(f" Failed to append HRV history: {hist_err}")

            from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton

            dlg = QDialog(self)
            dlg.setWindowTitle("Report Generated")
            dlg.setMinimumWidth(480)
            dlg.setStyleSheet("""
                QDialog { background: #111827; }
                QLabel  { color: #D1D5DB; font-size: 13px; font-family: 'Segoe UI', Arial; }
                QLabel#title { color: #00E676; font-size: 16px; font-weight: bold; }
                QPushButton { background: #1E2530; color: #D1D5DB; border: 1px solid #374151;
                              border-radius: 8px; padding: 8px 20px; font-size: 12px;
                              font-weight: bold; font-family: 'Segoe UI', Arial; }
                QPushButton:hover { background: #252B3B; }
                QPushButton#open_btn { background: #3B82F6; color: white; border: none; }
                QPushButton#open_btn:hover { background: #2563EB; }
            """)
            vbox = QVBoxLayout(dlg)
            vbox.setSpacing(12)
            vbox.setContentsMargins(20, 20, 20, 20)
            title_lbl = QLabel("✅  HRV Report Generated Successfully")
            title_lbl.setObjectName("title")
            vbox.addWidget(title_lbl)
            path_lbl = QLabel(f"<b>Saved at:</b><br>{fname}")
            path_lbl.setWordWrap(True)
            vbox.addWidget(path_lbl)
            hint_lbl = QLabel("You can view this report on the <b>History</b> page.")
            vbox.addWidget(hint_lbl)
            btn_row = QHBoxLayout()
            open_btn = QPushButton("Open PDF")
            open_btn.setObjectName("open_btn")

            def _open_pdf():
                from utils.platform_compat import open_file
                open_file(fname)

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
                self.report_btn.setText("Generate HRV Report")
            # Close loader
            try:
                if hasattr(self, '_loader_dlg') and self._loader_dlg:
                    self._loader_dlg.close()
                    self._loader_dlg = None
            except Exception:
                pass
            dlg.exec_()

        def _on_failure(err):
            print(f"❌ Failed to generate HRV report: {err}")
            self.crash_logger.log_error(
                message=f"HRV report generation error: {err}",
                exception=None,
                category="HRV_REPORT_ERROR"
            )
            if hasattr(self, 'report_btn'):
                self.report_btn.setEnabled(True)
                self.report_btn.setText("Generate HRV Report")
            # Close loader
            try:
                if hasattr(self, '_loader_dlg') and self._loader_dlg:
                    self._loader_dlg.close()
                    self._loader_dlg = None
            except Exception:
                pass
            QMessageBox.critical(self, "Failed", str(err)[:300])

        def _on_prep_done(data, fp, pat):
            # Called on main thread via Qt queued signal — safe to touch UI
            from utils.pdf_process_runner import PDFProcessRunner
            self._pdf_runner = PDFProcessRunner(parent_widget=self)
            started = self._pdf_runner.start_hrv_report(
                filename=fp,
                captured_data=_captured_snap,
                data=data,
                patient=pat,
                settings_manager=_settings,
                selected_lead=_lead_snap,
                on_success=_on_success,
                on_failure=_on_failure,
            )
            if not started:
                if hasattr(self, 'report_btn'):
                    self.report_btn.setEnabled(True)
                    self.report_btn.setText("Generate HRV Report")

        def _on_prep_failed(err):
            print(f"❌ Prep worker failed: {err}")
            if hasattr(self, 'report_btn'):
                self.report_btn.setEnabled(True)
                self.report_btn.setText("Generate HRV Report")
            # Close loader
            try:
                if hasattr(self, '_loader_dlg') and self._loader_dlg:
                    self._loader_dlg.close()
                    self._loader_dlg = None
            except Exception:
                pass
            QMessageBox.critical(self, "Failed", f"Could not prepare report data:\n{err[:300]}")

        # Kick off background prep — main thread stays responsive
        self._prep_worker = _PrepWorker(
            captured_data=_captured_snap,
            ecg_calculator=_ecg_calc,
            lead=_lead_snap,
            sampling_rate=_sr_snap,
            last_hr=_last_hr,
            filepath=filepath,
            patient=_patient_snap,
            settings_manager=_settings,
            display_metrics=_display_metrics_snap,
        )
        self._prep_worker.done.connect(_on_prep_done)
        self._prep_worker.failed.connect(_on_prep_failed)
        self._prep_worker.start()

    def calculate_time_domain_hrv_metrics(self):
        """
        Calculate time‑domain HRV metrics from the full selected lead capture.
        
        Returns a dict with:
            mean_rr_ms, sdnn_ms, rmssd_ms, nn50, pnn50, num_intervals
        or None if insufficient data.
        """
        try:
            if len(self.captured_data) < 500:
                return None

            # Build ECG signal array from captured selected lead values
            signal = np.array([d['value'] for d in self.captured_data], dtype=float)
            if signal.size < 500:
                return None

            # Use the same sampling rate we have been tracking during capture
            fs = float(self.sampling_rate or 0)
            if not np.isfinite(fs) or fs <= 0:
                fs = 500.0  # sensible default matching live capture

            # Apply bandpass filter to enhance R-peaks (0.5-40 Hz) - same as 12-lead test
            from scipy.signal import butter, filtfilt
            try:
                nyquist = fs / 2
                low = max(0.001, 0.5 / nyquist)
                high = min(0.999, 40 / nyquist)
                if low < high:
                    b, a = butter(4, [low, high], btype='band')
                    signal = filtfilt(b, a, signal)
                    # Check for invalid values after filtering
                    if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
                        print(" Filter produced invalid values, using unfiltered signal")
                        signal = np.array([d['value'] for d in self.captured_data], dtype=float)
            except Exception as e:
                print(f" Error in signal filtering: {e}, using unfiltered signal")
                signal = np.array([d['value'] for d in self.captured_data], dtype=float)

            signal_std = np.std(signal)
            if signal_std == 0:
                return None
            peaks, _ = find_peaks(
                signal,
                distance=int(0.25 * fs),
                prominence=signal_std * 0.6
            )

            if len(peaks) < 3:
                return None

            # Proceed with detected peaks directly

            # R‑R intervals in milliseconds
            rr_intervals = np.diff(peaks) * (1000.0 / fs)

            rr = rr_intervals[(rr_intervals > 300.0) & (rr_intervals < 1500.0)]
            print(rr[:50])
            print(np.abs(np.diff(rr))[:50])
            if rr.size < 2:
                return None

            median_rr = np.median(rr)
            mask = np.abs(rr - median_rr) < 0.2 * median_rr
            rr_clean = rr[mask]
            if rr_clean.size < 2:
                return None
            rr_diff = np.abs(np.diff(rr_clean))
            rr_final = rr_clean[1:][rr_diff < 100.0]
            if rr_final.size < 2:
                return None

            diff_rr = np.diff(rr_final)
            mean_rr_ms = float(np.mean(rr_final))
            sdnn_ms = float(np.std(rr_final, ddof=1))
            rmssd_ms = float(np.sqrt(np.mean(diff_rr ** 2)))
            nn50 = int(np.sum(np.abs(diff_rr) > 50.0))
            pnn50 = float((nn50 / len(diff_rr)) * 100.0) if len(diff_rr) > 0 else 0.0

            return {
                "mean_rr_ms": mean_rr_ms,
                "sdnn_ms": sdnn_ms,
                "rmssd_ms": rmssd_ms,
                "nn50": nn50,
                "pnn50": pnn50,
                "num_intervals": int(rr_final.size),
            }
        except Exception as e:
            # Log but don't crash report generation
            try:
                self.crash_logger.log_error(
                    message=f"HRV metrics calculation error: {e}",
                    exception=e,
                    category="HRV_METRICS_ERROR"
                )
            except Exception:
                pass
            return None
    
    def update_metrics(self):
        """Calculate and update ECG metrics from selected lead data using same methods as 12-lead test"""
        if not self.is_capturing or len(self.captured_data) < max(1500, int((self.sampling_rate or 500.0) * 3.0)):
            # Wait for at least 3 seconds of real ECG data before computing BPM.
            # The buffer starts filled with flat 2048 ADC values; computing too early
            # causes the flat→signal edge to generate fake R-peaks at ~260 BPM.
            return

        # ── ALL-LEADS-DISCONNECTED: Reset intervals to 0 ──
        # Check if the selected lead signal is flat (all leads disconnected).
        # Lead-off visual display (flat line on canvas) is kept intact by the
        # packet loop — this block ONLY resets metric labels and state.
        try:
            import numpy as _np
            _recent_vals = [d['value'] for d in list(self.captured_data)[-500:]]
            _lead_std = float(_np.std(_recent_vals)) if len(_recent_vals) >= 10 else 999.0
            _all_disconnected = _lead_std < 5.0
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
                # Drop the cached global-QRS result too, otherwise a width
                # measured before the leads dropped can reappear for a few seconds
                # once signal returns.
                try:
                    from .ecg_calculations import reset_global_qrs_cache
                    reset_global_qrs_cache(getattr(self.ecg_calculator, '_instance_id', 'hrv_test'))
                except Exception:
                    pass
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
            current_fs = self.sampling_rate if self.sampling_rate > 0 else 500.0
            
            if self.ecg_calculator:
                # Ensure the calculator's sampler is in sync
                if not hasattr(self.ecg_calculator, 'sampler') or self.ecg_calculator.sampler is None:
                    from ecg.twelve_lead_test import SamplingRateCalculator
                    self.ecg_calculator.sampler = SamplingRateCalculator()
                self.ecg_calculator.sampler.sampling_rate = current_fs

                # Update main sampling rate too
                self.ecg_calculator.sampling_rate = current_fs

                # TRIGGER STABLE MEDIAN-BEAT ANALYSIS (Same as 12-lead test)
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

                # ECGTestPage.calculate_ecg_metrics() updates its internal metric attrs
                try:
                    self.ecg_calculator.calculate_ecg_metrics()
                except Exception as e:
                    print(f" calculate_ecg_metrics error in HRV test: {e}")

                # FETCH METRICS DIRECTLY from the calculator's stored attributes.
                # get_current_metrics() reads from UI label text, but ecg_calculator
                # is a headless instance with no visible labels → always returns '0'.
                # The attributes below are set by calculate_ecg_metrics() directly.
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

                # Also try get_current_metrics as secondary source
                metrics = self.ecg_calculator.get_current_metrics()

                hr_val  = metrics.get('heart_rate', '0')
                pr_val  = _attr_to_str('pr_interval') or metrics.get('pr_interval', '0')
                qrs_val = _attr_to_str('last_qrs_duration') or metrics.get('qrs_duration', '0')
                qt_val  = _attr_to_str('last_qt_interval') or metrics.get('qt_interval', '0')
                qtc_val = _attr_to_str('last_qtc_interval') or metrics.get('qtc_interval', '0')
                st_val  = _attr_to_str('last_st_interval') or metrics.get('st_interval', '0')

                # Use the exact same display pipeline as the 12-lead view so
                # QT/QTc clamping, formatting, and stabilization stay aligned.
                # Important: feed the shared display updater with the calculator's
                # own stabilized HR, because 12-lead uses that same path when
                # deciding how QT/QTc should be rendered/clamped.
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
                    # Keep self.last_heart_rate in sync — HRV report reads from UI label
                    # which now reflects the correct ECG-derived value.
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

                # HRV cards historically show explicit units in the value row.
                if 'pr_interval' in self.metric_labels:
                    pr_text = self.metric_labels['pr_interval'].text().strip()
                    if pr_text and not pr_text.endswith("ms"):
                        self.metric_labels['pr_interval'].setText(f"{pr_text} ms")
                if 'qrs_duration' in self.metric_labels:
                    qrs_text = self.metric_labels['qrs_duration'].text().strip()
                    if qrs_text and not qrs_text.endswith("ms"):
                        self.metric_labels['qrs_duration'].setText(f"{qrs_text} ms")
                    # The width comes from the shared 12-lead calculator, which
                    # measures it globally whenever two or more leads are live.
                    try:
                        from .qrs_detection import describe_qrs_method
                        self.metric_labels['qrs_duration'].setToolTip(
                            describe_qrs_method(
                                getattr(self.ecg_calculator, 'last_qrs_method', 'single-lead'),
                                getattr(self.ecg_calculator, 'last_qrs_leads_used', 1),
                            )
                        )
                    except Exception:
                        pass
                if 'qtc_interval' in self.metric_labels:
                    qtqtc_text = self.metric_labels['qtc_interval'].text().strip()
                    if qtqtc_text and not qtqtc_text.endswith("ms"):
                        self.metric_labels['qtc_interval'].setText(f"{qtqtc_text} ms")

            else:
                # Fallback if calculator not available
                for key in self.metric_labels:
                    if key == 'heart_rate':
                        if getattr(self, '_last_displayed_bpm', 0) > 0:
                            self.metric_labels[key].setText(f"{int(self._last_displayed_bpm)} BPM")
                    elif key == 'st_interval':
                        self.metric_labels[key].setText("0 ms")
                    else: self.metric_labels[key].setText("0 ms")
        
        except Exception as e:
            # Log but don't crash
            print(f" Error updating HRV metrics: {e}")
            pass
    
    def closeEvent(self, event):
        """Handle window close event"""
        runner = getattr(self, '_pdf_runner', None)
        if runner is not None and runner.is_running():
            runner.cancel()

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
