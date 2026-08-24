"""
ECG Analysis Window — Professional Clinical Edition
====================================================
Enhanced with:
  • Interactive Tool Modes:  Select | Ruler | Caliper | Magnifier | Annotate
  • Measurement ruler (drag → shows Δt ms, Δ amplitude mV)
  • Dual-caliper (PP/RR interval measurement like a real ECG machine)
  • Crosshair cursor + live readout (time, amplitude)
  • Floating magnifier lens (2–5× zoom, follows cursor)
  • Right-click context-menu annotation on any lead
  • Click any lead → dedicated expanded analysis popup
  • All original JSON-load, API-fetch, frame-nav, PDF-gen features kept intact
"""

import os
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from PyQt5.QtCore import Qt, QTimer, QPoint, QPointF, QRect, QRectF, pyqtSignal, QDate
from PyQt5.QtGui import (QFont, QPixmap, QCursor, QPainter, QPen,
                         QColor, QBrush, QRadialGradient, QFontMetrics, QImage)
from PyQt5.QtWidgets import (
     QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
     QTableWidget, QTableWidgetItem, QFrame, QMessageBox,
     QSizePolicy, QComboBox, QFileDialog, QTextEdit, QSlider,
     QLineEdit, QAction, QMenu, QApplication, QButtonGroup,
     QToolButton, QWidget, QSplitter, QHeaderView, QDateEdit, QProgressBar
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrow
import matplotlib.patches as mpatches

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
    from ecg.expanded_lead_view import PQRSTAnalyzer
except Exception as e1:
    try:
        from src.ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
        from src.ecg.expanded_lead_view import PQRSTAnalyzer
    except Exception as e2:
        try:
            from ..ecg.arrhythmia_detector import ArrhythmiaDetector, get_interpretation
            from ..ecg.expanded_lead_view import PQRSTAnalyzer
        except Exception as e3:
            import traceback
            try:
                from utils.crash_logger import get_crash_logger
                get_crash_logger().error(
                    f"Failed to import ArrhythmiaDetector/PQRSTAnalyzer.\n"
                    f"Attempt 1: {e1}\nAttempt 2: {e2}\nAttempt 3: {e3}\n"
                    f"{traceback.format_exc()}",
                    category="IMPORT_ERROR"
                )
            except Exception:
                pass
            ArrhythmiaDetector = None
            get_interpretation = None
            PQRSTAnalyzer = None


try:
    from ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
    from ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs
except ImportError:
    try:
        from src.ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
        from src.ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs
    except ImportError:
        from ..ecg.holter.theme import ADC_TO_MV, COL_CROSSHAIR, TOOL_ANNOTATE, TOOL_CALIPER, TOOL_MAGNIFY, TOOL_RULER, TOOL_SELECT
        from ..ecg.holter.tool_engine import caliper_label, cursor as tool_cursor, hint as tool_hint, ruler_label, tool_specs


# ─────────────────────────────────────────────────────────────────────────────
#  INTERACTIVE CANVAS  (one per lead)
# ─────────────────────────────────────────────────────────────────────────────
class InteractiveLeadCanvas(FigureCanvas):
    """
    A matplotlib canvas that supports professional ECG interaction:
      - Crosshair + live readout
      - Ruler measurement (drag)
      - Caliper (two vertical lines)
      - Magnifier lens (overlay widget)
      - Right-click annotation menu
      - Double-click to expand
    """

    # Signals
    ruler_measured    = pyqtSignal(str)          # human-readable measurement string
    caliper_measured  = pyqtSignal(str)
    annotation_req    = pyqtSignal(float, float, str)  # start_sec, end_sec, lead_name
    expand_requested  = pyqtSignal(str)           # lead name
    def __init__(self, fig, ax, lead_name, parent_window, parent=None):
        super().__init__(fig)
        self.ax = ax
        self.lead_name = lead_name
        self.parent_window = parent_window

        # Interaction state
        self._drag_start = None          # (x_data, y_data) in data coordinates
        self._drag_end   = None
        self._caliper_x  = [None, None]  # two vertical positions (data x)
        self._mag_pos    = None          # cursor pos for magnifier (widget coords)
        # Overlay items (matplotlib artists we redraw)
        self._ruler_patch    = None
        self._ruler_text     = None
        self._ruler_history  = []
        self._caliper_lines  = [None, None]
        self._crosshair_v    = None
        self._crosshair_h    = None
        self._readout_text   = None

        # Mouse tracking
        self.setMouseTracking(True)

        # Connect matplotlib events
        self.mpl_connect('motion_notify_event',  self._on_mouse_move)
        self.mpl_connect('button_press_event',   self._on_mouse_press)
        self.mpl_connect('button_release_event', self._on_mouse_release)

    # ── helpers ──────────────────────────────────────────────────────────────
    @property
    def tool(self):
        return self.parent_window.current_tool

    def _data_coords(self, event):
        """Return (x_sec, y_adc) from a matplotlib mouse event, or None."""
        if event.inaxes != self.ax:
            return None, None
        return event.xdata, event.ydata

    def _widget_to_data(self, qx, qy):
        """Convert Qt widget pixel pos → matplotlib data coordinates."""
        w, h = self.width(), self.height()
        if w <= 0 or h <= 0:
            return None, None
        # matplotlib uses bottom-left origin
        mpl_x = qx
        mpl_y = h - qy
        try:
            inv = self.ax.transData.inverted()
            xd, yd = inv.transform((mpl_x, mpl_y))
            return xd, yd
        except Exception:
            return None, None

    def _clear_overlay(self, *names):
        for name in names:
            obj = getattr(self, name, None)
            if obj is not None:
                if isinstance(obj, list):
                    # Handle list of artists (like _caliper_lines)
                    for item in obj:
                        if item is not None:
                            try:
                                item.remove()
                            except Exception:
                                pass
                    # Reset the list to its original state [None, None] or []
                    if name == '_caliper_lines':
                        setattr(self, name, [None, None])
                    else:
                        setattr(self, name, [])
                else:
                    # Handle single artist
                    try:
                        obj.remove()
                    except Exception:
                        pass
                    setattr(self, name, None)

    # ── matplotlib event handlers ─────────────────────────────────────────────
    def _on_mouse_move(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
            self.draw_idle()
            return

        if not self.parent_window.lead_has_visible_data(self.lead_name):
            self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
            self.draw_idle()
            return

        # Show crosshair only for measurement tools.
        if tool in (TOOL_RULER, TOOL_CALIPER):
            self._draw_crosshair(xd, yd)

        # Ruler: drag to measure
        if tool == TOOL_RULER and self._drag_start is not None:
            self._draw_ruler(self._drag_start[0], self._drag_start[1], xd, yd)

        # Caliper: drag second line
        if tool == TOOL_CALIPER and self._caliper_x[0] is not None and self._caliper_x[1] is None:
            self._draw_caliper_preview(self._caliper_x[0], xd)

        if tool == TOOL_MAGNIFY:
            # The Qt mouseMoveEvent keeps the magnifier position in widget
            # coordinates. Avoid redrawing the Matplotlib canvas here because
            # it can race with the overlay paint pass and cause flicker.
            return

        self.draw_idle()

    def mouseMoveEvent(self, event):
        if self.tool == TOOL_MAGNIFY:
            self._mag_pos = event.pos()
            self.update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event):
        self._mag_pos = None
        self.update()
        super().leaveEvent(event)

    def _on_mouse_press(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            return

        if not self.parent_window.lead_has_visible_data(self.lead_name):
            return

        if event.button == 1:  # Left click
            if tool == TOOL_SELECT:
                self.expand_requested.emit(self.lead_name)
                return

            if tool == TOOL_RULER:
                self._drag_start = (xd, yd)
                self._drag_end   = None

            elif tool == TOOL_CALIPER:
                if self._caliper_x[0] is None:
                    self._caliper_x[0] = xd
                elif self._caliper_x[1] is None:
                    self._caliper_x[1] = xd
                    self._finalise_caliper()
                else:
                    # Reset
                    self._caliper_x = [xd, None]
                    self._clear_overlay('_caliper_lines')
                self.draw_idle()

            elif tool == TOOL_ANNOTATE:
                if self._drag_start is None:
                    self._drag_start = (xd, yd)
                else:
                    end_x = xd
                    start_x = self._drag_start[0]
                    self._drag_start = None
                    self.annotation_req.emit(
                        min(start_x, end_x),
                        max(start_x, end_x),
                        self.lead_name
                    )

        elif event.button == 3:  # Right click — context menu
            self._show_context_menu(xd, yd, event)

    def _on_mouse_release(self, event):
        tool = self.tool
        xd, yd = self._data_coords(event)
        if xd is None:
            return

        if event.button == 1 and tool == TOOL_RULER and self._drag_start is not None:
            self._drag_end = (xd, yd)
            self._finalise_ruler()
            self._drag_start = None

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.expand_requested.emit(self.lead_name)
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    # ── drawing helpers ───────────────────────────────────────────────────────
    def _draw_crosshair(self, xd, yd):
        self._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()

        self._crosshair_v, = self.ax.plot(
            [xd, xd], ylim, color=COL_CROSSHAIR, linewidth=0.7,
            linestyle='--', alpha=0.85, zorder=10)
        self._crosshair_h, = self.ax.plot(
            xlim, [yd, yd], color=COL_CROSSHAIR, linewidth=0.7,
            linestyle='--', alpha=0.85, zorder=10)

        # Convert ADC to mV for display
        mv = (yd - 2048) * ADC_TO_MV
        txt = f" t={xd*1000:.1f}ms  {mv:+.2f}mV"
        self._readout_text = self.ax.text(
            xd, ylim[1] * 0.97, txt, fontsize=7, color=COL_CROSSHAIR,
            va='top', ha='left', zorder=11,
            bbox=dict(boxstyle='round,pad=0.15', fc='#000000', alpha=0.55, ec='none'))

    def _draw_ruler(self, x0, y0, x1, y1):
        self._clear_overlay('_ruler_patch', '_ruler_text')
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        dt_ms = dx * 1000.0
        dv_mv = dy * ADC_TO_MV

        # Shaded rectangle
        lx = min(x0, x1); rx = max(x0, x1)
        ly = min(y0, y1); ry = max(y0, y1)
        rect = mpatches.FancyBboxPatch(
            (lx, ly), rx - lx, ry - ly,
            boxstyle="square,pad=0", linewidth=1.2,
            edgecolor='#ffdd00', facecolor='#ffdd0020', zorder=8)
        self.ax.add_patch(rect)
        self._ruler_patch = rect

        # Text
        mx = (lx + rx) / 2
        my = ry
        ylim = self.ax.get_ylim()
        label = ruler_label(dt_ms, dv_mv)
        self._ruler_text = self.ax.text(
            mx, min(my + (ylim[1] - ylim[0]) * 0.05, ylim[1] * 0.95),
            label, fontsize=7.5, color='#ffdd00', ha='center',
            va='bottom', fontweight='bold', zorder=12,
            bbox=dict(boxstyle='round,pad=0.2', fc='#111111', alpha=0.7, ec='none')
        )
        self.ruler_measured.emit(label)

    def _finalise_ruler(self):
        if self._drag_start and self._drag_end:
            x0, y0 = self._drag_start
            x1, y1 = self._drag_end
            dx = abs(x1 - x0)
            dy = abs(y1 - y0)
            dt_ms = dx * 1000.0
            dv_mv = dy * ADC_TO_MV
            label = ruler_label(dt_ms, dv_mv)
            self.ruler_measured.emit(label)
            if self._ruler_patch is not None or self._ruler_text is not None:
                self._ruler_history.append((self._ruler_patch, self._ruler_text))
                self._ruler_patch = None
                self._ruler_text = None
                if hasattr(self.parent_window, "_register_ruler_measurement"):
                    self.parent_window._register_ruler_measurement(self)
        self.draw_idle()

    def undo_last_ruler(self):
        if not self._ruler_history:
            return False
        patch, text = self._ruler_history.pop()
        for item in (patch, text):
            if item is not None:
                try:
                    item.remove()
                except Exception:
                    pass
        self.draw_idle()
        return True

    def clear_measurements(self):
        self._clear_overlay('_ruler_patch', '_ruler_text', '_crosshair_v', '_crosshair_h',
                            '_readout_text', '_caliper_lines')
        for patch, text in self._ruler_history:
            for item in (patch, text):
                if item is not None:
                    try:
                        item.remove()
                    except Exception:
                        pass
        self._ruler_history.clear()
        self._caliper_x = [None, None]
        self.draw_idle()

    def _draw_caliper_preview(self, x0, x1):
        # Draw first line solid, second dashed
        for i, (xl, style, col) in enumerate([(x0, '-', '#ff9900'), (x1, '--', '#ff9900')]):
            if self._caliper_lines[i] is not None:
                try:
                    self._caliper_lines[i].remove()
                except Exception:
                    pass
            ylim = self.ax.get_ylim()
            line, = self.ax.plot([xl, xl], ylim, color=col,
                                 linewidth=1.3, linestyle=style,
                                 alpha=0.9, zorder=9)
            self._caliper_lines[i] = line

    def _finalise_caliper(self):
        x0, x1 = sorted(self._caliper_x)
        dt_ms = abs(x1 - x0) * 1000.0
        bpm   = 60000.0 / dt_ms if dt_ms > 0 else 0
        label = caliper_label(dt_ms)
        # draw bracket
        self._draw_caliper_preview(x0, x1)
        # mid-span label
        ylim = self.ax.get_ylim()
        mid  = (x0 + x1) / 2
        self.ax.annotate(
            '', xy=(x1, (ylim[0]+ylim[1])/2), xytext=(x0, (ylim[0]+ylim[1])/2),
            arrowprops=dict(arrowstyle='<->', color='#ff9900', lw=1.3), zorder=9)
        self.ax.text(mid, (ylim[0]+ylim[1])/2 + (ylim[1]-ylim[0])*0.06,
                     label, fontsize=7.5, color='#ff9900', ha='center',
                     fontweight='bold', zorder=12,
                     bbox=dict(boxstyle='round,pad=0.2', fc='#111111', alpha=0.7, ec='none'))
        self.caliper_measured.emit(label)
        self.draw_idle()

    def _show_context_menu(self, xd, yd, event):
        """Right-click context menu for quick annotation."""
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu { background:#1e2235; color:#fff4e8; border:1px solid #6a4d24;
                    font-size:11px; border-radius:6px; }
            QMenu::item:selected { background:#ff8a1f; color:white; }
            QMenu::separator { height:1px; background:#6a4d24; margin:2px 10px; }
        """)
        menu.addAction(f"📍  Mark point at {xd*1000:.0f} ms")
        menu.addSeparator()
        types = ["Atrial Fibrillation", "PVC", "PAC", "SVT", "VT",
                 "Bradycardia", "Tachycardia", "2nd Degree Block", "LBBB", "RBBB"]
        for t in types:
            act = menu.addAction(f"⚡  Annotate: {t}")
            act.setData((xd, t))

        menu.addSeparator()
        clear_act = menu.addAction("🗑   Clear all overlays on this lead")

        chosen = menu.exec_(QCursor.pos())
        if chosen and chosen.data():
            xpos, ann_type = chosen.data()
            # Create a 0.5s annotation around click point
            self.annotation_req.emit(max(0, xpos - 0.25), xpos + 0.25, self.lead_name)
            # Also update the combo in the parent
            pw = self.parent_window
            idx = pw.arrhythmia_type_combo.findText(ann_type)
            if idx >= 0:
                pw.arrhythmia_type_combo.setCurrentIndex(idx)
        elif chosen == clear_act:
            self.clear_measurements()

    # ── Magnifier paintEvent overlay ─────────────────────────────────────────
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.tool != TOOL_MAGNIFY or self._mag_pos is None:
            return
        if not self.parent_window.lead_has_visible_data(self.lead_name):
            return

        cx, cy = self._mag_pos.x(), self._mag_pos.y()
        panel_w = 280
        panel_h = 180
        zoom = self.parent_window.magnifier_zoom
        xd, yd = self._widget_to_data(cx, cy)
        if xd is None or yd is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        left = cx + 24
        top = cy - panel_h - 24
        if left + panel_w > self.width() - 8:
            left = cx - panel_w - 24
        if left < 8:
            left = 8
        if top < 8:
            top = cy + 24
        if top + panel_h > self.height() - 8:
            top = max(8, self.height() - panel_h - 8)

        dst_rect = QRect(int(left), int(top), panel_w, panel_h)
        inner_rect = dst_rect.adjusted(10, 10, -10, -10)

        painter.setBrush(QColor(7, 10, 18, 240))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(dst_rect, 12, 12)

        painter.setBrush(QColor(10, 14, 24, 28))
        painter.setPen(QPen(QColor(255, 176, 92), 3.2))
        painter.drawRoundedRect(dst_rect, 12, 12)

        x_min, x_max = self.ax.get_xlim()
        y_min, y_max = self.ax.get_ylim()
        x_span = max(1e-6, (x_max - x_min) / max(zoom, 1))
        y_span = max(1.0, abs(y_max - y_min) / max(zoom, 1))

        def _centered_window(center, span, lo, hi):
            """Keep the window span stable while clamping it to the bounds."""
            if hi <= lo:
                return lo, hi
            span = min(span, hi - lo)
            left = center - span / 2.0
            right = center + span / 2.0
            if left < lo:
                right += lo - left
                left = lo
            if right > hi:
                left -= right - hi
                right = hi
            left = max(lo, left)
            right = min(hi, right)
            if right <= left:
                return lo, hi
            return left, right

        view_x0, view_x1 = _centered_window(xd, x_span, x_min, x_max)

        center_y = yd
        if self.lead_name == 'aVR':
            view_y0, view_y1 = _centered_window(center_y, y_span, min(y_min, y_max), max(y_min, y_max))
        else:
            view_y0, view_y1 = _centered_window(center_y, y_span, 0.0, 4096.0)

        grid_pen = QPen(QColor(36, 68, 36, 160), 1)
        painter.setPen(grid_pen)
        for frac in (0.25, 0.5, 0.75):
            gx = int(inner_rect.left() + inner_rect.width() * frac)
            gy = int(inner_rect.top() + inner_rect.height() * frac)
            painter.drawLine(gx, inner_rect.top(), gx, inner_rect.bottom())
            painter.drawLine(inner_rect.left(), gy, inner_rect.right(), gy)

        data = self.parent_window.lead_data.get(self.lead_name, np.array([]))
        if len(data) > 1 and self.parent_window.sampling_rate > 0:
            start_idx = max(0, int(np.floor(view_x0 * self.parent_window.sampling_rate)))
            end_idx = min(len(data), int(np.ceil(view_x1 * self.parent_window.sampling_rate)) + 1)
            segment = data[start_idx:end_idx]
            if len(segment) > 1:
                xs = np.arange(start_idx, end_idx, dtype=float) / self.parent_window.sampling_rate
                if self.lead_name == 'aVR':
                    seg_y = -segment
                else:
                    seg_y = segment

                points = []
                x_range = max(1e-6, view_x1 - view_x0)
                y_range = max(1e-6, view_y1 - view_y0)
                for px_t, py_v in zip(xs, seg_y):
                    nx = (px_t - view_x0) / x_range
                    ny = (py_v - view_y0) / y_range
                    qx = inner_rect.left() + nx * inner_rect.width()
                    qy = inner_rect.bottom() - ny * inner_rect.height()
                    points.append(QPointF(qx, qy))

                if len(points) > 1:
                    path = __import__('PyQt5.QtGui', fromlist=['QPainterPath']).QPainterPath()
                    path.moveTo(points[0])
                    for pt in points[1:]:
                        path.lineTo(pt)
                    painter.setClipRect(inner_rect)
                    painter.setPen(QPen(QColor(0, 255, 32), 2.2))
                    painter.drawPath(path)
                    painter.setClipping(False)

        focus_x = inner_rect.left() + ((xd - view_x0) / max(1e-6, view_x1 - view_x0)) * inner_rect.width()
        focus_y = inner_rect.bottom() - ((yd - view_y0) / max(1e-6, view_y1 - view_y0)) * inner_rect.height()

        painter.setPen(QPen(QColor(255, 255, 255, 180), 1))
        painter.drawLine(int(focus_x), inner_rect.top(), int(focus_x), inner_rect.bottom())
        painter.drawLine(inner_rect.left(), int(focus_y), inner_rect.right(), int(focus_y))

        painter.setPen(QPen(QColor(255, 244, 232)))
        painter.setFont(QFont("Consolas", 10, QFont.Bold))
        painter.drawText(dst_rect.left() + 12, dst_rect.bottom() - 12, f"{zoom}x")
        painter.end()


class LeadExpandedPopup(QDialog):
    """Standalone expanded lead window with metrics, zoom/amplification, and rhythm interpretation."""

    def __init__(self, lead_name: str, lead_data: np.ndarray, sampling_rate: float, parent=None):
        super().__init__(parent)
        self.lead_name = str(lead_name)
        self.sampling_rate = float(sampling_rate or 500.0)
        self.raw_data = np.asarray(lead_data if lead_data is not None else [], dtype=float)
        self.time_zoom_factor = 1.0
        self.amplification = 1.0
        self.analysis = {}

        self.setWindowTitle(f"Lead {self.lead_name} Expanded Analysis")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMaximizeButtonHint)

        self._build_ui()
        self._fit_to_screen()
        self.refresh_from_data(self.raw_data, self.sampling_rate)

    def _fit_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is None:
            self.resize(1200, 760)
            return
        rect = screen.availableGeometry()
        width = int(rect.width() * 0.86)
        height = int(rect.height() * 0.82)
        x = rect.x() + (rect.width() - width) // 2
        y = rect.y() + (rect.height() - height) // 2
        self.setGeometry(x, y, width, height)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        header = QLabel(f"Lead {self.lead_name} - Expanded Waveform")
        header.setStyleSheet("color:#fff4e8;font-size:14px;font-weight:bold;")
        root.addWidget(header)

        controls = QHBoxLayout()
        controls.setSpacing(8)

        controls.addWidget(QLabel("Time Zoom:"))
        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems(["1x", "2x", "4x", "8x"])
        self.zoom_combo.setCurrentText("1x")
        self.zoom_combo.currentTextChanged.connect(self._on_zoom_changed)
        controls.addWidget(self.zoom_combo)

        controls.addWidget(QLabel("Amplification:"))
        self.amp_slider = QSlider(Qt.Horizontal)
        self.amp_slider.setRange(25, 400)
        self.amp_slider.setValue(100)
        self.amp_slider.valueChanged.connect(self._on_amp_changed)
        self.amp_label = QLabel("1.00x")
        controls.addWidget(self.amp_slider, 1)
        controls.addWidget(self.amp_label)

        controls.addWidget(QLabel("Position:"))
        self.pos_slider = QSlider(Qt.Horizontal)
        self.pos_slider.setRange(0, 1000)
        self.pos_slider.setValue(0)
        self.pos_slider.valueChanged.connect(self._render_plot)
        controls.addWidget(self.pos_slider, 2)
        root.addLayout(controls)

        fig = Figure(facecolor="#090b14")
        self.ax = fig.add_subplot(111)
        self.canvas = FigureCanvas(fig)
        self.canvas.setStyleSheet("background:#090b14;border:1px solid #5b4525;")
        root.addWidget(self.canvas, stretch=1)

        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(12)
        self.rr_lbl = QLabel("RR: -- ms")
        self.pr_lbl = QLabel("PR: -- ms")
        self.qrs_lbl = QLabel("QRS: -- ms")
        self.p_lbl = QLabel("P: --")
        for lbl in (self.rr_lbl, self.pr_lbl, self.qrs_lbl, self.p_lbl):
            lbl.setStyleSheet(
                "color:#ffedda;background:#1b1f34;border:1px solid #6a532e;"
                "border-radius:6px;padding:6px 10px;font-weight:bold;"
            )
            metrics_row.addWidget(lbl)
        metrics_row.addStretch()
        root.addLayout(metrics_row)

        self.interpret_lbl = QLabel("Arrhythmia Interpretation: analyzing...")
        self.interpret_lbl.setWordWrap(True)
        self.interpret_lbl.setStyleSheet(
            "color:#e8f1ff;background:#15192b;border:1px solid #5b4525;border-radius:6px;padding:8px;"
        )
        root.addWidget(self.interpret_lbl)

    def refresh_from_data(self, lead_data: np.ndarray, sampling_rate: float):
        self.raw_data = np.asarray(lead_data if lead_data is not None else [], dtype=float)
        self.sampling_rate = float(sampling_rate or 500.0)
        self._run_analysis()
        self._render_plot()

    def _on_zoom_changed(self, text: str):
        try:
            self.time_zoom_factor = max(1.0, float(str(text).replace("x", "")))
        except Exception:
            self.time_zoom_factor = 1.0
        self._render_plot()

    def _on_amp_changed(self, value: int):
        self.amplification = max(0.25, float(value) / 100.0)
        self.amp_label.setText(f"{self.amplification:.2f}x")
        self._render_plot()

    def _run_analysis(self):
        if self.raw_data.size < 20:
            self.analysis = {}
            self.rr_lbl.setText("RR: -- ms")
            self.pr_lbl.setText("PR: -- ms")
            self.qrs_lbl.setText("QRS: -- ms")
            self.p_lbl.setText("P: No waveform")
            self.interpret_lbl.setText("Arrhythmia Interpretation: Not enough data.")
            return

        analysis = {}
        try:
            if PQRSTAnalyzer is not None:
                analyzer = PQRSTAnalyzer(self.sampling_rate)
                analysis = analyzer.analyze_signal(self.raw_data) or {}
        except Exception:
            analysis = {}

        r_peaks = np.asarray(analysis.get("r_peaks", []), dtype=int)
        p_peaks = np.asarray(analysis.get("p_peaks", []), dtype=int)
        q_peaks = np.asarray(analysis.get("q_peaks", []), dtype=int)
        s_peaks = np.asarray(analysis.get("s_peaks", []), dtype=int)

        rr_ms = 0.0
        if r_peaks.size >= 2:
            rr_ms = float(np.median(np.diff(r_peaks)) * 1000.0 / max(self.sampling_rate, 1.0))

        pr_values = []
        if p_peaks.size and r_peaks.size:
            for r in r_peaks:
                prior_p = p_peaks[p_peaks < r]
                if prior_p.size:
                    pr_values.append((int(r) - int(prior_p[-1])) * 1000.0 / max(self.sampling_rate, 1.0))
        pr_ms = float(np.median(pr_values)) if pr_values else 0.0

        qrs_values = []
        if q_peaks.size and s_peaks.size:
            for r in r_peaks:
                q_before = q_peaks[q_peaks < r]
                s_after = s_peaks[s_peaks > r]
                if q_before.size and s_after.size:
                    qrs_values.append((int(s_after[0]) - int(q_before[-1])) * 1000.0 / max(self.sampling_rate, 1.0))
        qrs_ms = float(np.median(qrs_values)) if qrs_values else 0.0

        p_status = "Present" if p_peaks.size > 0 else "Absent"
        self.rr_lbl.setText(f"RR: {rr_ms:.0f} ms" if rr_ms > 0 else "RR: -- ms")
        self.pr_lbl.setText(f"PR: {pr_ms:.0f} ms" if pr_ms > 0 else "PR: -- ms")
        self.qrs_lbl.setText(f"QRS: {qrs_ms:.0f} ms" if qrs_ms > 0 else "QRS: -- ms")
        self.p_lbl.setText(f"P: {p_status} ({int(p_peaks.size)})")

        interpretation_text = "No arrhythmia interpretation available."
        try:
            if ArrhythmiaDetector is not None:
                det = ArrhythmiaDetector(self.sampling_rate, counts_per_mv=500.0)
                parent_win = getattr(self, "parent", lambda: None)()
                if parent_win is None:
                    parent_win = getattr(self, "parent_window", None)
                ls = {"II": self.raw_data}
                if parent_win and hasattr(parent_win, "lead_data"):
                    if "V1" in parent_win.lead_data and len(parent_win.lead_data["V1"]) >= len(self.raw_data):
                        ls["V1"] = parent_win.lead_data["V1"][:len(self.raw_data)]
                    if "V6" in parent_win.lead_data and len(parent_win.lead_data["V6"]) >= len(self.raw_data):
                        ls["V6"] = parent_win.lead_data["V6"][:len(self.raw_data)]
                arr = det.detect_arrhythmias(
                    self.raw_data,
                    analysis,
                    lead_signals=ls
                )
                summary_lines = []
                if get_interpretation is not None:
                    result_stub = {
                        "heart_rate_bpm": (60000.0 / rr_ms) if rr_ms > 0 else 0.0,
                        "rr_ms": rr_ms,
                        "pr_ms": pr_ms,
                        "qrs_ms": qrs_ms,
                        "qtc_bazett": 0.0,
                        "arrhythmias": arr or [],
                        "st_levels": {},
                        "is_nsr": any("Normal Sinus Rhythm" in str(x) for x in (arr or [])),
                        "nsr_failed_criteria": [],
                    }
                    summary_lines = [line for line in get_interpretation(result_stub) if line]
                if not summary_lines:
                    summary_lines = arr or ["Normal Sinus Rhythm"]
                interpretation_text = " | ".join(summary_lines)
        except Exception:
            pass

        self.interpret_lbl.setText(f"Arrhythmia Interpretation: {interpretation_text}")
        self.analysis = analysis

    def _render_plot(self):
        self.ax.clear()
        self.ax.set_facecolor("#080e08")
        self.ax.grid(True, color="#1a3a1a", linewidth=0.35, linestyle="-", alpha=1.0)

        if self.raw_data.size < 2:
            self.ax.text(0.5, 0.5, "No data", transform=self.ax.transAxes,
                         ha="center", va="center", color="#97a78f")
            self.canvas.draw_idle()
            return

        fs = max(self.sampling_rate, 1.0)
        total_samples = int(self.raw_data.size)
        total_sec = total_samples / fs
        window_sec = max(1.0, total_sec / max(self.time_zoom_factor, 1.0))
        window_samples = max(2, int(round(window_sec * fs)))
        max_start = max(0, total_samples - window_samples)
        start = int(round((self.pos_slider.value() / 1000.0) * max_start))
        end = min(total_samples, start + window_samples)

        seg = self.raw_data[start:end]
        time_axis = np.arange(start, end) / fs
        baseline = float(np.mean(seg)) if seg.size else 0.0
        display = (seg - baseline) * self.amplification + baseline

        self.ax.plot(time_axis, display, color="#00d000", linewidth=1.0, antialiased=True)
        self.ax.set_xlim(time_axis[0], time_axis[-1] if len(time_axis) > 1 else time_axis[0] + 1.0)

        # Keep the classic ECG ADC range while respecting amplification.
        margin = 200.0
        y_min = float(np.min(display)) - margin
        y_max = float(np.max(display)) + margin
        if y_max <= y_min:
            y_min, y_max = baseline - 500.0, baseline + 500.0
        self.ax.set_ylim(y_min, y_max)
        self.ax.set_title(
            f"Lead {self.lead_name} | Zoom {self.time_zoom_factor:.0f}x | Amp {self.amplification:.2f}x",
            color="#ffdcb5", fontsize=10
        )
        self.ax.tick_params(axis="x", colors="#b9c9b0", labelsize=8)
        self.ax.tick_params(axis="y", colors="#b9c9b0", labelsize=8)
        self.canvas.draw_idle()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN WINDOW
# ─────────────────────────────────────────────────────────────────────────────
class PublicReportsDialog(QDialog):
    """Browse public reports fetched by mobile number, with search + date filters."""

    def __init__(self, reports: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Report")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(900)
        self._all = [r for r in (reports or []) if isinstance(r, dict)]
        self._selected = None

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        filters = QHBoxLayout()
        filters.addWidget(QLabel("Search Name:"))
        self.name_search = QLineEdit()
        self.name_search.setPlaceholderText("Type patient name…")
        filters.addWidget(self.name_search, 2)

        filters.addWidget(QLabel("Search Date:"))
        self.date_search = QDateEdit()
        self.date_search.setCalendarPopup(True)
        self.date_search.setDisplayFormat("dd-MM-yyyy")
        self._date_any = QDate(1900, 1, 1)
        self.date_search.setMinimumDate(self._date_any)
        self.date_search.setSpecialValueText("Any")
        self.date_search.setDate(self._date_any)
        self.date_search.setFixedWidth(150)
        self.date_search.setStyleSheet(
            "QDateEdit{"
            " background:#0f1220;"
            " color:#fff4e8;"
            " border:1px solid #5e4827;"
            " border-radius:8px;"
            " padding:6px 34px 6px 10px;"
            " font-size:12px;"
            " font-family:'Segoe UI', Arial;"
            "}"
            "QDateEdit:hover{border-color:#ff8a1f;}"
            "QDateEdit:focus{border-color:#ff8a1f;}"
            "QDateEdit::drop-down{"
            " subcontrol-origin:padding;"
            " subcontrol-position:top right;"
            " width:28px;"
            " border-left:1px solid #5e4827;"
            " background:#15192b;"
            " border-top-right-radius:8px;"
            " border-bottom-right-radius:8px;"
            "}"
            "QDateEdit::drop-down:hover{background:#1b2036;}"
        )
        try:
            cal = self.date_search.calendarWidget()
            cal.setGridVisible(True)
            cal.setStyleSheet(
                "QCalendarWidget{"
                " background:#0f1220;"
                " color:#fff4e8;"
                " border:1px solid #5e4827;"
                " border-radius:10px;"
                "}"
                "QCalendarWidget QWidget#qt_calendar_navigationbar{"
                " background:#15192b;"
                " border-top-left-radius:10px;"
                " border-top-right-radius:10px;"
                "}"
                "QCalendarWidget QToolButton{"
                " color:#fff4e8;"
                " background:transparent;"
                " border:none;"
                " font-weight:bold;"
                " padding:6px 10px;"
                " margin:4px;"
                " border-radius:8px;"
                "}"
                "QCalendarWidget QToolButton:hover{background:#1b2036;}"
                "QCalendarWidget QToolButton:pressed{background:#241808;}"
                "QCalendarWidget QSpinBox{"
                " background:#0f1220;"
                " color:#fff4e8;"
                " border:1px solid #5e4827;"
                " border-radius:6px;"
                " padding:2px 6px;"
                " margin:4px;"
                "}"
                "QCalendarWidget QSpinBox::up-button, QCalendarWidget QSpinBox::down-button{width:18px;}"
                "QCalendarWidget QAbstractItemView{"
                " background:#0b0f1d;"
                " color:#fff4e8;"
                " selection-background-color:#ff8a1f;"
                " selection-color:#ffffff;"
                " gridline-color:#2a2230;"
                " outline:0;"
                "}"
                "QCalendarWidget QAbstractItemView:disabled{color:#6b5a4a;}"
                "QCalendarWidget QMenu{"
                " background:#15192b;"
                " color:#fff4e8;"
                " border:1px solid #5e4827;"
                "}"
                "QCalendarWidget QMenu::item:selected{background:#1b2036;}"
            )
        except Exception:
            pass
        filters.addWidget(self.date_search, 0)

        self.clear_filters_btn = QPushButton("Clear")
        self.clear_filters_btn.setFixedWidth(70)
        filters.addWidget(self.clear_filters_btn, 0)

        filters.addStretch(1)
        root.addLayout(filters)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Age/Gender"])
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Stretch)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        root.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.setObjectName("primary")
        self.load_btn.clicked.connect(self._accept_selected)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        actions.addStretch(1)
        actions.addWidget(self.cancel_btn)
        actions.addWidget(self.load_btn)
        root.addLayout(actions)

        self.name_search.textChanged.connect(self._refresh_table)
        self.date_search.dateChanged.connect(self._refresh_table)
        self.clear_filters_btn.clicked.connect(self._clear_filters)
        self.table.cellDoubleClicked.connect(lambda *_: self._accept_selected())
        self._refresh_table()

    def _norm(self, v):
        return ("" if v is None else str(v)).strip()

    def _get_first(self, d: dict, *keys):
        for k in keys:
            if k in d and d[k] not in (None, "", []):
                return d[k]
        return None

    def _clear_filters(self):
        try:
            self.name_search.setText("")
            self.date_search.setDate(self._date_any)
        except Exception:
            pass
        self._refresh_table()

    def _matches(self, report: dict) -> bool:
        name = self._norm(self._get_first(report, "name", "patient_name", "patientName"))
        query = self._norm(self.name_search.text()).lower()
        if query and query not in name.lower():
            return False

        # Date filter (match by YYYY-MM-DD)
        try:
            chosen = self.date_search.date()
            if chosen != self._date_any:
                rdate = self._norm(self._get_first(report, "report_date", "reportDate", "date", "created_at", "createdAt"))
                iso = (rdate.replace("Z", "").replace("z", "") or "").strip()
                date_part = ""
                if "T" in iso:
                    date_part = iso.split("T", 1)[0]
                elif " " in iso:
                    date_part = iso.split(" ", 1)[0]
                else:
                    date_part = iso[:10]
                # Compare to chosen date in ISO format
                if date_part and date_part != chosen.toString("yyyy-MM-dd"):
                    return False
        except Exception:
            # If date parsing fails, don't filter out by date
            pass

        return True

    def _refresh_table(self):
        self.table.setUpdatesEnabled(False)
        self.table.blockSignals(True)
        try:
            rows = [r for r in self._all if self._matches(r)]
            self.table.clearContents()
            self.table.setRowCount(len(rows))

            for row, report in enumerate(rows):
                name = self._norm(self._get_first(report, "name", "patient_name", "patientName")) or "Unknown"
                age_val = self._get_first(report, "age", "patient_age", "patientAge")
                gender_val = self._get_first(report, "gender", "patient_gender", "patientGender")

                age_str = self._norm(age_val)
                gender_str = self._norm(gender_val)

                if age_str and gender_str:
                    ag_str = f"{age_str}/{gender_str}"
                elif age_str:
                    ag_str = age_str
                elif gender_str:
                    ag_str = gender_str
                else:
                    ag_str = "—"

                vals = [name, ag_str]
                for col, val in enumerate(vals):
                    item = QTableWidgetItem(val)
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter if col == 0 else Qt.AlignCenter)
                    self.table.setItem(row, col, item)
                if self.table.item(row, 0):
                    self.table.item(row, 0).setData(Qt.UserRole, report)
                self.table.setRowHeight(row, 28)
        finally:
            self.table.blockSignals(False)
            self.table.setUpdatesEnabled(True)

    def _accept_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Select", "Select a report first.")
            return
        item = self.table.item(row, 0)
        self._selected = item.data(Qt.UserRole) if item else None
        self.accept()

    def get_selected_report(self):
        return self._selected


class ECGAnalysisWindow(QDialog):
    """Professional ECG Analysis Window with clinical-grade doctor tools."""

    LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Waveform Analysis — Clinical Edition")
        self.setWindowFlags(Qt.Window | Qt.FramelessWindowHint)

        # ── Tool state ──────────────────────────────────────────────────
        self.current_tool  = TOOL_SELECT
        self.magnifier_zoom = 4
        self._ruler_stack = []
        # ── Data state ──────────────────────────────────────────────────
        self.reports             = []
        self.current_report      = None
        self.current_report_path = ""
        self.lead_data           = {lead: np.array([]) for lead in self.LEADS}
        self.sampling_rate       = 500.0
        # Filter settings
        self.filter_ac = "50"
        self.filter_emg = "25"
        self.filter_dft = "off"
        try:
            from utils.settings_manager import SettingsManager
            sm = SettingsManager()
            self.filter_ac = str(sm.get_setting("filter_ac", self.filter_ac) or self.filter_ac).strip()
            self.filter_emg = str(sm.get_setting("filter_emg", self.filter_emg) or self.filter_emg).strip()
            self.filter_dft = str(sm.get_setting("filter_dft", self.filter_dft) or self.filter_dft).strip()
        except Exception:
            pass

        self.window_seconds      = 10.0
        self.step_seconds        = 0.5
        self.frame_start_sample  = 0
        self.play_timer          = QTimer(self)
        self.play_timer.timeout.connect(self.next_frame)

        self.pending_mark_start_sec = None
        self.manual_annotations     = []
        self._expanded_lead         = None   # Deprecated: old in-grid expand path
        self._lead_popup_windows    = {}     # lead -> LeadExpandedPopup
        self._active_lead_popup     = None

        # Paths
        project_root = Path(__file__).resolve().parents[2]
        self.analysis_pdf_logo_path = project_root / "assets" / "DeckmountLogo.png"

        # Public API perf helpers (mobile report browser)
        self._public_api_session = None
        self._public_reports_cache = {}

        self._apply_stylesheet()
        self._build_ui()
        self._init_loader_overlay()
        # self.load_reports() - Removed with report UI elements
        QTimer.singleShot(0, self._fit_window_to_screen)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            if hasattr(self, "_loader_overlay") and self._loader_overlay is not None:
                self._loader_overlay.setGeometry(self.rect())
        except Exception:
            pass

    def _init_loader_overlay(self):
        """Full-window loading overlay (used for public mobile report fetch)."""
        self._loader_overlay = QWidget(self)
        self._loader_overlay.setObjectName("loader_overlay")
        self._loader_overlay.setGeometry(self.rect())
        self._loader_overlay.setAttribute(Qt.WA_TransparentForMouseEvents, False)
        self._loader_overlay.hide()

        root = QVBoxLayout(self._loader_overlay)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        center = QWidget(self._loader_overlay)
        center_lay = QVBoxLayout(center)
        center_lay.setContentsMargins(0, 0, 0, 0)
        center_lay.setSpacing(10)
        center_lay.setAlignment(Qt.AlignCenter)

        box = QFrame(center)
        box.setObjectName("loader_box")
        box_lay = QVBoxLayout(box)
        box_lay.setContentsMargins(18, 14, 18, 14)
        box_lay.setSpacing(10)

        self._loader_label = QLabel("Loading…", box)
        self._loader_label.setObjectName("loader_label")
        self._loader_label.setAlignment(Qt.AlignCenter)
        box_lay.addWidget(self._loader_label)

        self._loader_bar = QProgressBar(box)
        self._loader_bar.setObjectName("loader_bar")
        self._loader_bar.setRange(0, 0)  # indeterminate
        self._loader_bar.setFixedWidth(280)
        box_lay.addWidget(self._loader_bar, 0, Qt.AlignCenter)

        center_lay.addWidget(box, 0, Qt.AlignCenter)
        root.addWidget(center, 1)

    def _show_loader(self, text: str = "Loading…"):
        try:
            if hasattr(self, "_loader_label") and self._loader_label is not None:
                self._loader_label.setText(str(text or "Loading…"))
            if hasattr(self, "_loader_overlay") and self._loader_overlay is not None:
                self._loader_overlay.setGeometry(self.rect())
                self._loader_overlay.raise_()
                self._loader_overlay.show()
                QApplication.processEvents()
        except Exception:
            pass

    def _hide_loader(self):
        try:
            if hasattr(self, "_loader_overlay") and self._loader_overlay is not None:
                self._loader_overlay.hide()
                QApplication.processEvents()
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    #  STYLESHEET
    # ─────────────────────────────────────────────────────────────────────────
    def _apply_stylesheet(self):
        self.setStyleSheet("""
            QDialog {
                background: #0f1220;
                color: #f3e8dc;
            }
            QFrame#topbar {
                background: #15192b;
                border: none;
                border-bottom: 2px solid #6d4a1f;
            }
            QFrame#plotpanel {
                background: #0f1220;
                border: none;
            }
            QFrame#bottompanel {
                background: #15192b;
                border: 1px solid #5f4523;
                border-radius: 8px;
            }
            QFrame#manual_mark_card {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #171b2e, stop:1 #12162a);
                border: 1px solid #5b4525;
                border-radius: 10px;
            }
            QLabel#section_title {
                color:#fff4e8;
                font-size:12px;
                font-weight:bold;
                font-family:'Courier New';
            }
            QLabel#field_lbl {
                color:#fff4e8;
                font-size:11px;
                font-weight:bold;
                font-family:Arial;
            }
            QLineEdit#manual_input, QLineEdit#mobile_input {
                background:#0f1220;
                color:#fff4e8;
                border:1px solid #5e4827;
                border-radius:7px;
                padding:6px 10px;
                font-size:12px;
                font-family:Arial;
            }
            QLineEdit#manual_input:focus, QLineEdit#mobile_input:focus { border-color:#ff8a1f; }
            QLineEdit#manual_input:hover, QLineEdit#mobile_input:hover { border-color:#ff8a1f; }
            QLineEdit#manual_input::placeholder, QLineEdit#mobile_input::placeholder { color:#bda68b; }
            QPushButton#mark_outline {
                background:#15192b;
                color:#fff4e8;
                border:2px solid #ff8a1f;
                border-radius:7px;
                padding:6px 12px;
                font-weight:bold;
            }
            QPushButton#mark_outline:hover { background:#1b2036; border-color:#ff9d3d; }
            QPushButton#mark_danger {
                background:#15192b;
                color:#ff8b8b;
                border:2px solid #a63d2d;
                border-radius:7px;
                padding:6px 12px;
                font-weight:bold;
            }
            QPushButton#mark_danger:hover { background:#1b2036; border-color:#d65d4a; }
            QTableWidget#annotation_table {
                background:#0b0f1d;
                border:1px solid #5b4525;
                gridline-color:#2f261d;
                selection-background-color:#ff8a1f;
            }
            QHeaderView::section {
                background: #1a1f34;
                color: #ffcb95;
                border: none;
                border-bottom: 1px solid #5b4525;
                padding: 6px;
                font-size: 10px;
                font-weight: bold;
            }
            QFrame#leadbox {
                background: #090b14;
                border: 1px solid #342a23;
                border-radius: 4px;
            }
            QFrame#leadbox_expanded {
                background: #090b14;
                border: 2px solid #ff8a1f;
                border-radius: 6px;
            }
            QFrame#toolbar_frame {
                background: #171b2e;
                border: 1px solid #5b4325;
                border-radius: 8px;
            }
            QLabel {
                color: #e9dccd;
                font-size: 11px;
                background: transparent;
                border: none;
            }
            QLabel#leadlabel {
                color: #ff4444;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Courier New';
                background: transparent;
                border: none;
            }
            QPushButton {
                background: #1b2036;
                color: #fff4e8;
                border: 1px solid #5e4827;
                border-radius: 5px;
                padding: 5px 12px;
                font-size: 11px;
            }
            QPushButton:hover {
                background: #2a2230;
                border-color: #ff8a1f;
                color: #ffffff;
            }
            QPushButton#primary {
                background: #ff8a1f;
                color: #ffffff;
                border: 1px solid #ff8a1f;
                font-weight: bold;
            }
            QPushButton#primary:hover { background: #ff9d3d; }
            QPushButton#apibtn {
                background: #ff8a1f;
                color: #ffffff;
                border: none;
                border-radius: 5px;
            }
            QPushButton#apibtn:hover { background: #ff9d3d; }
            /* Tool buttons */
            QToolButton {
                background: #191d31;
                color: #fff4e8;
                border: 1px solid #5b4b2a;
                border-radius: 6px;
                padding: 6px;
                font-size: 13px;
                min-width: 36px;
                min-height: 36px;
            }
            QToolButton:hover {
                background: #2a2230;
                border-color: #ff8a1f;
                color: #ffffff;
            }
            QToolButton:checked {
                background: #ff8a1f;
                border: 2px solid #ffd9b0;
                color: #ffffff;
            }
            QComboBox {
                background: #1a1f34;
                color: #ffffff;
                border: 1px solid #6a532e;
                border-radius: 5px;
                padding: 4px 22px 4px 8px;
                font-size: 11px;
                font-weight: bold;
            }
            QComboBox:hover {
                border-color: #ff8a1f;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left: none;
            }
            QComboBox::down-arrow {
                image: none;
                width: 0px;
                height: 0px;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #ffffff;
                margin-right: 4px;
            }
            QComboBox::down-arrow:hover {
                border-top-color: #ff8a1f;
            }
            QComboBox QAbstractItemView {
                background: #1a1f34;
                color: #fff4e8;
                selection-background-color: #ff8a1f;
                border: 1px solid #5b4525;
            }
            QLineEdit {
                background: #1a1f34;
                color: #fff4e8;
                border: 1px solid #5b4525;
                border-radius: 5px;
                padding: 4px 8px;
                font-size: 11px;
            }
            QTextEdit {
                background: #090b14;
                color: #f0e1d1;
                border: 1px solid #5b4525;
                border-radius: 5px;
                padding: 4px;
                font-size: 10px;
            }
            QTableWidget {
                background: #090b14;
                color: #f0e1d1;
                border: 1px solid #5b4525;
                gridline-color: #46341f;
                selection-background-color: #ff8a1f;
                selection-color: #ffffff;
                border-radius: 4px;
                font-size: 10px;
            }
            QHeaderView::section {
                background: #1a1f34;
                color: #ffcb95;
                border: none;
                border-bottom: 1px solid #5b4525;
                padding: 5px;
                font-size: 10px;
                font-weight: bold;
            }
            QSlider::groove:horizontal {
                height: 3px;
                background: #5b4525;
                border-radius: 1px;
            }
            QSlider::sub-page:horizontal {
                background: #ff8a1f;
                border-radius: 1px;
            }
            QSlider::handle:horizontal {
                width: 12px; height: 12px;
                background: #ffb15a;
                margin: -4px 0;
                border-radius: 6px;
            }
            QScrollBar:vertical {
                background: #090b14;
                width: 6px;
            }
            QScrollBar::handle:vertical {
                background: #6b4a23;
                border-radius: 3px;
            }
            QWidget#loader_overlay {
                background: rgba(0, 0, 0, 140);
            }
            QFrame#loader_box {
                background: #15192b;
                border: 1px solid #5f4523;
                border-radius: 10px;
            }
            QLabel#loader_label {
                color: #fff4e8;
                font-weight: bold;
                font-size: 12px;
                font-family: Arial;
            }
            QProgressBar#loader_bar {
                border: 1px solid #5e4827;
                border-radius: 6px;
                background: #0f1220;
                height: 10px;
                text-align: center;
            }
            QProgressBar#loader_bar::chunk {
                background: #ff8a1f;
                border-radius: 6px;
            }
        """)

    # ─────────────────────────────────────────────────────────────────────────
    #  UI CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._build_top_bar())

        content = QHBoxLayout()
        content.setContentsMargins(6, 6, 6, 6)
        content.setSpacing(6)
        self._tool_sidebar = self._build_tool_sidebar()
        self._plot_panel = self._build_plot_panel()
        content.addWidget(self._tool_sidebar)
        content.addWidget(self._plot_panel, stretch=1)

        mid = QWidget()
        mid.setLayout(content)
        root.addWidget(mid, stretch=4)
        self._bottom_panel = self._build_bottom_panel()
        root.addWidget(self._bottom_panel, stretch=0)

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    def _build_top_bar(self):
        frame = QFrame()
        frame.setObjectName("topbar")
        lay = QHBoxLayout(frame)
        lay.setContentsMargins(12, 8, 12, 8)
        lay.setSpacing(10)

        self.back_btn = QPushButton("⬅ Dashboard")
        self.back_btn.setStyleSheet(
            "background:#c0392b;color:white;font-weight:bold;"
            "font-size:12px;padding:7px 14px;border:none;border-radius:5px;")
        self.back_btn.clicked.connect(self.close)
        lay.addWidget(self.back_btn)

        pat_col = QVBoxLayout()
        pat_col.setSpacing(1)
        self.patient_lbl = QLabel("Patient: —")
        self.patient_lbl.setFont(QFont("Courier New", 11, QFont.Bold))
        self.patient_lbl.setStyleSheet("color:#fff5ea;font-weight:bold;")
        self.patient_meta_lbl = QLabel("ID: — | Age: — | Gender: —")
        self.patient_meta_lbl.setStyleSheet("color:#d8b28b;font-size:10px;")
        pat_col.addWidget(self.patient_lbl)
        pat_col.addWidget(self.patient_meta_lbl)
        self.patient_lbl.setVisible(False)
        self.patient_meta_lbl.setVisible(False)
        lay.addLayout(pat_col)
        lay.addStretch()

        # Measurement readout bar (ruler/caliper results)
        self.measure_lbl = QLabel("")
        self.measure_lbl.setStyleSheet(
            "color:#ffe5bf;font-family:'Courier New';font-size:11px;"
            "background:#241808;border:1px solid #8a5a1f;border-radius:4px;padding:3px 10px;")
        self.measure_lbl.setMinimumWidth(280)
        self.measure_lbl.setAlignment(Qt.AlignCenter)
        self.measure_lbl.setVisible(False)
        lay.addWidget(self.measure_lbl)
        lay.addStretch()

        self.report_lbl = QLabel("Report:")
        self.report_lbl.setVisible(False)
        lay.addWidget(self.report_lbl)
        self.report_combo = QComboBox()
        self.report_combo.currentIndexChanged.connect(self.load_selected_report)
        self.report_combo.setMinimumWidth(280)
        self.report_combo.setVisible(False)
        lay.addWidget(self.report_combo)

        self.refresh_btn = QPushButton("↻")
        self.refresh_btn.setFixedWidth(32)
        self.refresh_btn.clicked.connect(self.load_reports)
        self.refresh_btn.setVisible(False)
        lay.addWidget(self.refresh_btn)

        self.export_btn = QPushButton("⬇ JSON")
        self.export_btn.clicked.connect(self.export_report)
        self.export_btn.setVisible(False)
        lay.addWidget(self.export_btn)

        self.pdf_btn = QPushButton("📄 PDF Report")
        self.pdf_btn.setObjectName("primary")
        self.pdf_btn.clicked.connect(self.generate_pdf_report)
        lay.addWidget(self.pdf_btn)

        return frame

    # ── TOOL SIDEBAR ─────────────────────────────────────────────────────────
    def _build_tool_sidebar(self):
        frame = QFrame()
        frame.setObjectName("toolbar_frame")
        frame.setFixedWidth(88)
        lay = QVBoxLayout(frame)
        lay.setContentsMargins(6, 10, 6, 10)
        lay.setSpacing(6)

        tool_data = tool_specs(include_annotate=True)

        self._tool_btns = {}
        self._tool_btn_group = QButtonGroup(self)
        self._tool_btn_group.setExclusive(True)

        for tool_id, icon, tip in tool_data:
            btn = QToolButton()
            btn.setText(icon)
            btn.setToolTip(tip)
            btn.setCheckable(True)
            btn.setFont(QFont("Arial", 9, QFont.Bold))
            btn.setMinimumHeight(42)
            if tool_id == TOOL_SELECT:
                btn.setChecked(True)
            btn.clicked.connect(lambda checked, t=tool_id: self.set_tool(t))
            self._tool_btn_group.addButton(btn)
            self._tool_btns[tool_id] = btn
            lay.addWidget(btn)

        lay.addSpacing(10)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("background:#5b4525;border:none;max-height:1px;")
        lay.addWidget(sep)

        lay.addSpacing(4)

        zoom_lbl = QLabel("Zoom")
        zoom_lbl.setAlignment(Qt.AlignCenter)
        zoom_lbl.setStyleSheet("color:#ffd7b0;font-size:9px;font-weight:bold;")
        lay.addWidget(zoom_lbl)

        self.zoom_combo = QComboBox()
        self.zoom_combo.addItems([f"{level}x" for level in [2, 3, 4, 5, 6]])
        self.zoom_combo.setCurrentText("4x")
        self.zoom_combo.currentIndexChanged.connect(
            lambda i: setattr(self, 'magnifier_zoom', i + 2))
        self.zoom_combo.setFixedWidth(74)
        lay.addWidget(self.zoom_combo)

        lay.addStretch()

        # Clear overlays button
        clr_btn = QToolButton()
        clr_btn.setText("X")
        clr_btn.setToolTip("Clear all measurements & overlays")
        clr_btn.setFont(QFont("Arial", 11, QFont.Bold))
        clr_btn.clicked.connect(self._clear_all_overlays)
        undo_btn = QToolButton()
        undo_btn.setText("↶")
        undo_btn.setToolTip("Undo last ruler measurement (Ctrl+Z)")
        undo_btn.setFont(QFont("Arial", 11, QFont.Bold))
        undo_btn.clicked.connect(self._undo_last_ruler)
        lay.addWidget(undo_btn)
        lay.addWidget(clr_btn)

        return frame

    def set_tool(self, tool_id):
        self.current_tool = tool_id
        cursor = tool_cursor(tool_id)
        for canvas in self._lead_canvases.values():
            canvas.setCursor(cursor)
            if tool_id not in (TOOL_RULER, TOOL_CALIPER):
                canvas._clear_overlay('_crosshair_v', '_crosshair_h', '_readout_text')
                canvas.draw_idle()
        self.measure_lbl.setText(tool_hint(tool_id))

    def _clear_all_overlays(self):
        for canvas in self._lead_canvases.values():
            canvas._drag_start = None
            if hasattr(canvas, "clear_measurements"):
                canvas.clear_measurements()
            else:
                canvas._caliper_x = [None, None]
                canvas._clear_overlay('_ruler_patch', '_crosshair_v', '_crosshair_h',
                                      '_readout_text', '_caliper_lines')
                canvas.draw_idle()
        self._ruler_stack.clear()
        self.measure_lbl.setText("")

    def _register_ruler_measurement(self, canvas):
        self._ruler_stack.append(canvas)

    def _undo_last_ruler(self):
        while self._ruler_stack:
            canvas = self._ruler_stack.pop()
            if canvas is not None and hasattr(canvas, "undo_last_ruler") and canvas.undo_last_ruler():
                self.measure_lbl.setText("↶ Undid last ruler measurement")
                return
        self.measure_lbl.setText("No ruler measurement to undo")

    # ── PLOT PANEL ───────────────────────────────────────────────────────────
    def _build_plot_panel(self):
        frame = QFrame()
        frame.setObjectName("plotpanel")
        frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        v = QVBoxLayout(frame)
        v.setContentsMargins(4, 4, 4, 2)
        v.setSpacing(4)

        # Controls bar
        ctrl_frame = QFrame()
        ctrl_frame.setStyleSheet(
            "background:#15192b;border-radius:6px;border:1px solid #5b4525;")
        controls = QHBoxLayout(ctrl_frame)
        controls.setContentsMargins(10, 5, 10, 5)
        controls.setSpacing(8)

        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.clicked.connect(self.prev_frame)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.next_btn = QPushButton("Next ▶")
        self.next_btn.clicked.connect(self.next_frame)
        for b in (self.prev_btn, self.play_btn, self.next_btn):
            controls.addWidget(b)

        controls.addSpacing(8)
        controls.addWidget(QLabel("Window:"))
        self.window_combo = QComboBox()
        self.window_combo.addItems(["1.0 s","2.0 s","3.0 s","5.0 s","10.0 s"])
        self.window_combo.setCurrentText("10.0 s")
        self.window_combo.currentTextChanged.connect(self._on_window_changed)
        self.window_combo.setFixedWidth(88)
        controls.addWidget(self.window_combo)

        controls.addWidget(QLabel("Step:"))
        self.step_combo = QComboBox()
        self.step_combo.addItems(["0.2 s","0.5 s","1.0 s"])
        self.step_combo.setCurrentText("0.5 s")
        self.step_combo.currentTextChanged.connect(self._on_step_changed)
        self.step_combo.setFixedWidth(80)
        controls.addWidget(self.step_combo)

        controls.addSpacing(12)
        self.frame_label = QLabel("Frame: 0.00s – 10.00s")
        self.frame_label.setStyleSheet(
            "color:#fff4e8;font-weight:bold;font-family:'Courier New';font-size:11px;")
        controls.addWidget(self.frame_label)
        controls.addStretch()

        # Expand/collapse hint
        expand_hint = QLabel("click lead = expanded popup  |  right-click = annotate menu")
        expand_hint.setStyleSheet("color:#c59768;font-size:9px;")
        controls.addWidget(expand_hint)

        v.addWidget(ctrl_frame)

        # Timeline
        self.timeline = QSlider(Qt.Horizontal)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(0)
        self.timeline.valueChanged.connect(self._on_timeline_changed)
        self.timeline.setFixedHeight(14)
        v.addWidget(self.timeline)

        # Lead grid
        from PyQt5.QtWidgets import QGridLayout, QWidget as _QW
        self._grid_widget = _QW()
        self._grid_widget.setStyleSheet("background:#0f1220;")
        self._grid_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lead_grid = QGridLayout(self._grid_widget)
        self._lead_grid.setContentsMargins(0, 0, 0, 0)
        self._lead_grid.setSpacing(4)

        self._lead_canvases = {}
        self._lead_axes     = {}
        self._lead_figs     = {}
        self._lead_frames   = {}

        lead_order = [
            ['I',   'aVR', 'V1', 'V4'],
            ['II',  'aVL', 'V2', 'V5'],
            ['III', 'aVF', 'V3', 'V6'],
        ]

        for row, row_leads in enumerate(lead_order):
            self._lead_grid.setRowStretch(row, 1)
            for col, lead in enumerate(row_leads):
                self._lead_grid.setColumnStretch(col, 1)
                cell = QFrame()
                cell.setObjectName("leadbox")
                self._lead_frames[lead] = cell
                cell_lay = QVBoxLayout(cell)
                cell_lay.setContentsMargins(0, 0, 0, 0)
                cell_lay.setSpacing(0)

                lbl = QLabel(lead)
                lbl.setObjectName("leadlabel")
                lbl.setAlignment(Qt.AlignLeft)
                lbl.setContentsMargins(5, 2, 0, 0)
                lbl.setFixedHeight(16)
                cell_lay.addWidget(lbl)

                fig = Figure(facecolor='#090b14')
                fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
                ax  = fig.add_axes([0, 0, 1, 1], facecolor='#090b14')
                ax.set_axis_off()

                canvas = InteractiveLeadCanvas(fig, ax, lead, self)
                canvas.setStyleSheet("background:#090b14;border:none;")

                # Connect signals
                canvas.ruler_measured.connect(self.measure_lbl.setText)
                canvas.caliper_measured.connect(self.measure_lbl.setText)
                canvas.annotation_req.connect(self._on_canvas_annotation)
                canvas.expand_requested.connect(self._open_lead_expanded_popup)
                cell_lay.addWidget(canvas, stretch=1)
                self._lead_grid.addWidget(cell, row, col)

                self._lead_figs[lead]    = fig
                self._lead_canvases[lead] = canvas
                self._lead_axes[lead]    = ax

        # Legacy compat
        self.figure = list(self._lead_figs.values())[0]
        self.canvas = list(self._lead_canvases.values())[0]
        self.axes   = [self._lead_axes[l] for l in self.LEADS]

        v.addWidget(self._grid_widget, stretch=1)
        return frame

    def _on_canvas_annotation(self, start_sec, end_sec, lead_name):
        """Handle annotation request from canvas right-click or annotate tool."""
        arr_type = self.arrhythmia_type_combo.currentText().strip()
        if arr_type == 'Other':
            arr_type = self.manual_type_input.text().strip() or 'Other'

        ann = {
            'start_sec':  round(start_sec, 3),
            'end_sec':    round(end_sec, 3),
            'type':       arr_type,
            'lead':       lead_name,
            'notes':      self.notes_input.text().strip() or "Quick annotate",
            'created_at': datetime.now().isoformat(timespec='seconds')
        }
        self.manual_annotations.append(ann)
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()
        self.mark_status_lbl.setText(f"✅ Quick annotation: {arr_type} on {lead_name}")
        self.mark_status_lbl.setStyleSheet(
            "color:#7adf7a;border:none;background:transparent;")

    def _toggle_lead_expand(self, lead_name):
        """Deprecated old in-grid expand behavior. Kept commented-out logically and not used."""
        if self._expanded_lead == lead_name:
            # Collapse — show all
            self._expanded_lead = None
            for l, cell in self._lead_frames.items():
                cell.setVisible(True)
            # restore grid layout
            lead_order = [
                ['I',   'aVR', 'V1', 'V4'],
                ['II',  'aVL', 'V2', 'V5'],
                ['III', 'aVF', 'V3', 'V6'],
            ]
            for row, row_leads in enumerate(lead_order):
                for col, lead in enumerate(row_leads):
                    self._lead_grid.addWidget(self._lead_frames[lead], row, col)
            self._lead_frames[lead_name].setObjectName("leadbox")
        else:
            # Expand
            self._expanded_lead = lead_name
            for l, cell in self._lead_frames.items():
                cell.setVisible(l == lead_name)
            # Span the whole grid
            self._lead_grid.addWidget(self._lead_frames[lead_name], 0, 0, 3, 4)
            self._lead_frames[lead_name].setObjectName("leadbox_expanded")
            # Force style refresh
            self._lead_frames[lead_name].style().unpolish(self._lead_frames[lead_name])
            self._lead_frames[lead_name].style().polish(self._lead_frames[lead_name])

        self._render_current_frame()

    def _open_lead_expanded_popup(self, lead_name):
        """Open dedicated expanded lead popup with metrics + rhythm interpretation."""
        data = np.asarray(self.lead_data.get(lead_name, np.array([])), dtype=float)
        if data.size < 20:
            QMessageBox.information(self, "No Data", f"No usable waveform data for lead {lead_name}.")
            return

        popup = self._lead_popup_windows.get(lead_name)
        if popup is None or not popup.isVisible():
            popup = LeadExpandedPopup(lead_name, data, self.sampling_rate, parent=self)
            self._lead_popup_windows[lead_name] = popup
        else:
            popup.refresh_from_data(data, self.sampling_rate)

        self._active_lead_popup = popup
        popup.show()
        popup.raise_()
        popup.activateWindow()

    # ── BOTTOM PANEL ─────────────────────────────────────────────────────────
    def _build_bottom_panel(self):
        frame = QFrame()
        frame.setObjectName("bottompanel")
        frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        h = QHBoxLayout(frame)
        h.setContentsMargins(12, 8, 12, 8)
        h.setSpacing(14)

        # ── Manual marking ───────────────────────────────────────────────────
        mark_box = QFrame()
        mark_box.setObjectName("manual_mark_card")
        mark_box.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        av = QVBoxLayout(mark_box)
        av.setContentsMargins(12, 10, 12, 10)
        av.setSpacing(8)

        title_lbl = QLabel("▌ Manual Arrhythmia Marking")
        title_lbl.setObjectName("section_title")
        av.addWidget(title_lbl)

        row1 = QHBoxLayout()
        combo_lbl_style = "color:#fff4e8;font-size:11px;font-weight:bold;font-family:Arial;"
        type_lbl = QLabel("Type:")
        type_lbl.setObjectName("field_lbl")
        row1.addWidget(type_lbl)
        self.arrhythmia_type_combo = QComboBox()
        self.arrhythmia_type_combo.addItems([
            "Atrial Fibrillation", "PVC", "PAC", "SVT", "VT",
            "Bradycardia", "Tachycardia", "2nd Degree Block",
            "LBBB", "RBBB", "Normal Sinus Rhythm", "Other"
        ])
        self.arrhythmia_type_combo.setMinimumWidth(220)
        # Use an absolute path so the arrow icon works regardless of current working directory / packaging.
        arrow_icon_path = str((Path(__file__).resolve().parents[2] / "assets" / "dropdown_arrow.png").resolve()).replace("\\", "/")
        combo_style = (
            "QComboBox{"
            " background:#0f1220;"
            " color:#fff4e8;"
            " border:1px solid #5e4827;"
            " border-radius:6px;"
            " padding:6px 34px 6px 10px;"
            " font-size:12px;"
            " font-family:Arial;"
            "}"
            "QComboBox:hover{border-color:#ff8a1f;}"
            "QComboBox:focus{border-color:#ff8a1f;}"
            "QComboBox::drop-down{"
            " subcontrol-origin:padding;"
            " subcontrol-position:top right;"
            " width:28px;"
            " border-left:1px solid #5e4827;"
            " background:#15192b;"
            " border-top-right-radius:6px;"
            " border-bottom-right-radius:6px;"
            "}"
            "QComboBox::down-arrow{"
            f" image: url(\"{arrow_icon_path}\");"
            " width:18px;"
            " height:18px;"
            " margin-right:8px;"
            "}"
            "QComboBox QAbstractItemView{"
            " background:#0b0f1d;"
            " color:#fff4e8;"
            " selection-background-color:#ff8a1f;"
            " selection-color:#ffffff;"
            " outline:0;"
            "}"
        )
        self.arrhythmia_type_combo.setStyleSheet(combo_style)
        self.arrhythmia_type_combo.setMinimumHeight(30)
        row1.addWidget(self.arrhythmia_type_combo, 2)
        lead_lbl = QLabel("Lead:")
        lead_lbl.setObjectName("field_lbl")
        row1.addWidget(lead_lbl)
        self.mark_lead_combo = QComboBox()
        self.mark_lead_combo.addItems(["All Leads"] + self.LEADS)
        self.mark_lead_combo.setMinimumWidth(130)
        self.mark_lead_combo.setStyleSheet(combo_style)
        self.mark_lead_combo.setMinimumHeight(30)
        row1.addWidget(self.mark_lead_combo, 1)
        av.addLayout(row1)

        row1b = QHBoxLayout()
        self.manual_type_input = QLineEdit()
        self.manual_type_input.setObjectName("manual_input")
        self.manual_type_input.setPlaceholderText("Custom type (if Other selected)")
        self.manual_type_input.setMinimumWidth(220)
        self.notes_input = QLineEdit()
        self.notes_input.setObjectName("manual_input")
        self.notes_input.setPlaceholderText("Clinical notes...")
        self.notes_input.setMinimumWidth(220)
        row1b.addWidget(self.manual_type_input)
        row1b.addWidget(self.notes_input)
        av.addLayout(row1b)

        row2 = QHBoxLayout()
        self.mark_start_btn = QPushButton("① Mark Start")
        self.mark_start_btn.setObjectName("mark_outline")
        self.mark_start_btn.clicked.connect(self.mark_start)

        self.mark_end_btn = QPushButton("② Mark End + Save")
        self.mark_end_btn.setObjectName("mark_outline")
        self.mark_end_btn.clicked.connect(self.mark_end_and_save)
        self.mark_end_btn.setEnabled(False)

        self.delete_mark_btn = QPushButton("🗑 Delete")
        self.delete_mark_btn.setObjectName("mark_danger")
        self.delete_mark_btn.clicked.connect(self.delete_selected_annotation)

        for b in (self.mark_start_btn, self.mark_end_btn,
                  self.delete_mark_btn):
            row2.addWidget(b)
        av.addLayout(row2)

        self.mark_status_lbl = QLabel("No active mark")
        self.mark_status_lbl.setStyleSheet(
            "color:#d7b183;font-family:'Courier New';font-size:10px;")
        av.addWidget(self.mark_status_lbl)

        self.annotation_table = QTableWidget(0, 5)
        self.annotation_table.setObjectName("annotation_table")
        self.annotation_table.setHorizontalHeaderLabels(
            ["Start (s)", "End (s)", "Type", "Lead", "Notes"])
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.annotation_table.setMinimumHeight(110)
        av.addWidget(self.annotation_table)

        h.addWidget(mark_box, stretch=2)

        # ── Right: Metrics + Findings ────────────────────────────────────────
        right_col = QVBoxLayout()
        right_col.setSpacing(3)

        # Mobile report browser (public API) - right aligned (not inside manual marking frame)
        mobile_wrap = QWidget()
        mobile_lay = QHBoxLayout(mobile_wrap)
        mobile_lay.setContentsMargins(0, 0, 0, 0)
        mobile_lay.setSpacing(10)
        mobile_lay.addStretch(1)

        mobile_lbl = QLabel("Mobile No:")
        mobile_lbl.setStyleSheet("color:#fff4e8;font-size:12px;font-weight:bold;font-family:Arial;")
        mobile_lay.addWidget(mobile_lbl, 0, Qt.AlignRight)

        self.mobile_no_input = QLineEdit()
        self.mobile_no_input.setObjectName("mobile_input")
        self.mobile_no_input.setPlaceholderText("XXXXXXXXXX")
        # Digits only, at most ten. setMaxLength alone capped the length but
        # still accepted letters, so "12ab34cd56" could be typed and then
        # silently normalised to six digits before the length check rejected it
        # with a message that did not match what the user saw in the box.
        try:
            from utils.input_validation import apply_digit_only, PHONE_DIGITS
            apply_digit_only(self.mobile_no_input, PHONE_DIGITS)
        except Exception:
            self.mobile_no_input.setMaxLength(10)
        self.mobile_no_input.setFixedWidth(170)
        mobile_lay.addWidget(self.mobile_no_input, 0, Qt.AlignRight)

        self.mobile_load_btn = QPushButton("Load")
        self.mobile_load_btn.setStyleSheet(
            "background:#ff8a1f; color:#ffffff; border:none; border-radius:5px;"
            "font-family:Arial; font-weight:bold; font-size:12px; padding:7px 16px;"
        )
        self.mobile_load_btn.clicked.connect(self.load_mobile_reports)
        mobile_lay.addWidget(self.mobile_load_btn, 0, Qt.AlignRight)

        right_col.addWidget(mobile_wrap, 0, Qt.AlignRight)

        metrics_lbl = QLabel("▌ ECG Metrics")
        metrics_lbl.setStyleSheet(
            "color:#fff4e8;font-size:11px;font-weight:bold;font-family:'Courier New';")
        right_col.addWidget(metrics_lbl)
        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Parameter", "Value"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        self.metrics_table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_col.addWidget(self.metrics_table)

        findings_lbl = QLabel("▌ Clinical Findings")
        findings_lbl.setStyleSheet(
            "color:#fff4e8;font-size:11px;font-weight:bold;font-family:'Courier New';")
        right_col.addWidget(findings_lbl)
        self.findings_text = QTextEdit()
        self.findings_text.setReadOnly(True)
        self.findings_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        right_col.addWidget(self.findings_text)

        # Hide per original user instruction (kept for PDF gen logic)
        for w in (metrics_lbl, self.metrics_table, findings_lbl, self.findings_text):
            w.setVisible(False)

        h.addLayout(right_col, stretch=1)
        return frame

    def _normalize_mobile_no(self, mobile_no: str) -> str:
        digits = "".join(ch for ch in str(mobile_no or "") if ch.isdigit())
        if len(digits) == 12 and digits.startswith("91"):
            digits = digits[2:]
        return digits

    def load_mobile_reports(self):
        """Load list of public reports by mobile number and show selection dialog."""
        raw_mobile = self.mobile_no_input.text().strip() if hasattr(self, "mobile_no_input") else ""
        # Checked again here, not just at the widget: the validator above only
        # governs typing, and this value goes on to build a cloud API request.
        try:
            from utils.input_validation import validate_phone
            ok, mobile_no, err = validate_phone(raw_mobile, "Mobile number")
        except Exception:
            mobile_no = self._normalize_mobile_no(raw_mobile)
            ok, err = len(mobile_no) == 10, "Enter a valid 10-digit mobile number."
        if not ok:
            QMessageBox.warning(self, "Mobile", err)
            return

        from utils.offline_queue import get_offline_queue
        from utils.ui_feedback import is_network_error, offline_action_message, show_critical

        offline_queue = None
        try:
            offline_queue = get_offline_queue()
        except Exception:
            offline_queue = None

        def _get_public_session():
            import requests
            if self._public_api_session is None:
                s = requests.Session()
                s.headers.update({"Accept": "application/json"})
                self._public_api_session = s
            return self._public_api_session

        def _as_lightweight_rows(raw_reports: list):
            """Return (light_rows, full_map) to keep the selector fast even if API returns huge waveform payloads."""
            raw = [r for r in (raw_reports or []) if isinstance(r, dict)]
            full_map = list(raw)
            keep_keys = {
                "report_id", "reportId", "id", "reportID",
                "report_type", "reportType", "type",
                "report_date", "reportDate", "date", "created_at", "createdAt",
                "report_format", "reportFormat", "format",
                "name", "patient_name", "patientName",
                "age", "patient_age", "patientAge",
                "gender", "patient_gender", "patientGender",
            }
            light = []
            for idx, r in enumerate(full_map):
                d = {k: r.get(k) for k in keep_keys if k in r}
                d["__full_index"] = idx
                light.append(d)
            return light, full_map

        url = "https://pmltkfluqk.execute-api.us-east-1.amazonaws.com/dev/api/public/reports"
        cursor_overridden = False
        try:
            self.mobile_load_btn.setText("…")
            self.mobile_load_btn.setEnabled(False)
            self._show_loader("Loading reports…")

            cached = self._public_reports_cache.get(mobile_no)
            if isinstance(cached, list) and cached:
                reports = cached
            elif offline_queue is not None and not offline_queue.is_online():
                self._hide_loader()
                show_critical(
                    self,
                    "No Internet Connection",
                    offline_action_message(
                        "Loading public reports",
                        "Reconnect to fetch cloud reports, or use locally saved JSON reports instead.",
                    ),
                )
                return
            else:
                QApplication.setOverrideCursor(Qt.WaitCursor)
                cursor_overridden = True
                QApplication.processEvents()
                sess = _get_public_session()
                resp = sess.get(url, params={"mobile_no": mobile_no}, timeout=15)
                payload = resp.json()

                # Accept multiple shapes: list, {"data": [...]}, {"reports": [...]}
                if isinstance(payload, list):
                    reports = payload
                elif isinstance(payload, dict):
                    reports = payload.get("data") or payload.get("reports") or payload.get("items") or []
                else:
                    reports = []

                if isinstance(reports, list) and reports:
                    # Session cache (avoid refetch on repeated loads for same mobile)
                    self._public_reports_cache[mobile_no] = reports

            # Restore the cursor as soon as the list is ready (dialog selection should use normal cursor).
            if cursor_overridden:
                QApplication.restoreOverrideCursor()
                cursor_overridden = False
            self._hide_loader()

            if not isinstance(reports, list) or not reports:
                QMessageBox.information(self, "Reports", "No reports found for this mobile number.")
                return

            light_rows, full_map = _as_lightweight_rows(reports)
            dlg = PublicReportsDialog(light_rows, parent=self)
            if dlg.exec_() != QDialog.Accepted:
                return

            selected = dlg.get_selected_report()
            if not selected:
                return

            self._show_loader("Loading selected waveform…")

            # Map back to full record (if we used lightweight row objects)
            full_idx = selected.get("__full_index") if isinstance(selected, dict) else None
            selected_full = None
            try:
                if isinstance(full_idx, int) and 0 <= full_idx < len(full_map):
                    selected_full = full_map[full_idx]
            except Exception:
                selected_full = None

            meta_source = selected_full or selected or {}
            new_report = self._map_public_report_to_internal(selected_full or selected)
            if not new_report:
                # If listing is metadata-only, try fetching full details by report_id.
                rep_id = (
                    selected.get("report_id")
                    or selected.get("reportId")
                    or selected.get("id")
                    or selected.get("reportID")
                    or ""
                )
                rep_id = str(rep_id).strip()
                if rep_id:
                    detail = self._fetch_public_report_detail_by_report_id(rep_id, mobile_no=mobile_no)
                    if isinstance(detail, dict):
                        new_report = self._map_public_report_to_internal(detail) or new_report

            if not new_report:
                QMessageBox.warning(self, "Reports", "Selected report did not include usable waveform data.")
                return

            # Preserve patient metadata from list selection if detail API call returned empty patient details
            if isinstance(new_report.get("patient_details"), dict):
                pat = new_report["patient_details"]
                m_pd = meta_source.get("patient_details") if isinstance(meta_source.get("patient_details"), dict) else (meta_source.get("patient") if isinstance(meta_source.get("patient"), dict) else {})
                meta_name = selected.get("name") or selected.get("patient_name") or selected.get("patientName") or meta_source.get("name") or meta_source.get("patient_name") or meta_source.get("patientName") or m_pd.get("name")
                meta_age = selected.get("age") or selected.get("patient_age") or selected.get("patientAge") or meta_source.get("age") or meta_source.get("patient_age") or meta_source.get("patientAge") or m_pd.get("age")
                meta_gender = selected.get("gender") or selected.get("patient_gender") or selected.get("patientGender") or meta_source.get("gender") or meta_source.get("patient_gender") or meta_source.get("patientGender") or m_pd.get("gender")
                meta_id = selected.get("report_id") or selected.get("reportId") or selected.get("id") or selected.get("reportID") or meta_source.get("report_id") or meta_source.get("reportId") or meta_source.get("id") or m_pd.get("report_id")
                meta_date = selected.get("report_date") or selected.get("reportDate") or selected.get("date") or selected.get("created_at") or meta_source.get("report_date") or meta_source.get("reportDate") or meta_source.get("date") or m_pd.get("report_date")

                if (not pat.get("name") or pat.get("name") == "Unknown") and meta_name:
                    pat["name"] = meta_name
                if not pat.get("age") and meta_age:
                    pat["age"] = meta_age
                if not pat.get("gender") and meta_gender:
                    pat["gender"] = meta_gender
                if not pat.get("report_id") and meta_id:
                    pat["report_id"] = meta_id
                if not pat.get("report_date") and meta_date:
                    pat["report_date"] = meta_date

            self.reports.append(new_report)
            idx = len(self.reports) - 1
            # Load and render the selected report
            self._load_and_render_report(idx)
            self._hide_loader()
        except Exception as e:
            if is_network_error(e):
                show_critical(
                    self,
                    "No Internet Connection",
                    offline_action_message(
                        "Loading public reports",
                        "Reconnect to fetch cloud reports, or use locally saved JSON reports instead.",
                    ),
                    details=str(e),
                )
            else:
                show_critical(self, "API Error", "Failed to load reports.", details=str(e))
        finally:
            try:
                self._hide_loader()
                if cursor_overridden:
                    QApplication.restoreOverrideCursor()
                self.mobile_load_btn.setText("Load")
                self.mobile_load_btn.setEnabled(True)
            except Exception:
                pass

    def _fetch_public_report_detail_by_report_id(self, report_id: str, mobile_no: str = None):
        """Best-effort fetch for a single report by report_id (API contract not guaranteed)."""
        rid = str(report_id or "").strip()
        if not rid:
            return None

        base = "https://pmltkfluqk.execute-api.us-east-1.amazonaws.com/dev/api/public"
        candidates = []
        mn = self._normalize_mobile_no(mobile_no) if mobile_no else ""
        if len(mn) == 10:
            # Shared contract: mobile_no + report_id on /public/reports
            candidates.append(f"{base}/reports?mobile_no={mn}&report_id={rid}")
            candidates.append(f"{base}/reports?mobile_no={mn}&reportId={rid}")

        candidates += [
            f"{base}/report?report_id={rid}",
            f"{base}/report?reportId={rid}",
            f"{base}/reports?report_id={rid}",
            f"{base}/reports?reportId={rid}",
            f"{base}/reports?id={rid}",
        ]
        import requests
        sess = self._public_api_session if self._public_api_session is not None else requests.Session()
        for url in candidates:
            try:
                resp = sess.get(url, timeout=15)
                payload = resp.json()
                if isinstance(payload, dict):
                    if payload.get("status") is False:
                        continue
                    data = payload.get("data") or payload.get("report") or payload.get("item")
                    if isinstance(data, dict):
                        return data
                    if isinstance(data, list) and data:
                        first = data[0]
                        return first if isinstance(first, dict) else None
                    # some endpoints might return report object directly
                    if any(k in payload for k in ("ecg_data", "leads", "device_data", "lead1_reading")):
                        return payload
                if isinstance(payload, list) and payload:
                    first = payload[0]
                    return first if isinstance(first, dict) else None
            except Exception:
                continue
        return None

    def _map_public_report_to_internal(self, report_obj: dict):
        """Best-effort mapping from public API report record -> internal report structure used by Waveform Analysis."""
        if not isinstance(report_obj, dict):
            return None

        def _get_first(d: dict, *keys):
            if not isinstance(d, dict):
                return None
            for k in keys:
                if k in d and d[k] not in (None, "", []):
                    return d[k]
            return None

        pd = report_obj.get("patient_details") if isinstance(report_obj.get("patient_details"), dict) else (report_obj.get("patient") if isinstance(report_obj.get("patient"), dict) else {})
        patient_name = _get_first(report_obj, "name", "patient_name", "patientName") or _get_first(pd, "name", "patient_name", "patientName")
        patient_age = _get_first(report_obj, "age", "patient_age", "patientAge") or _get_first(pd, "age", "patient_age", "patientAge")
        patient_gender = _get_first(report_obj, "gender", "patient_gender", "patientGender") or _get_first(pd, "gender", "patient_gender", "patientGender")
        report_id = _get_first(report_obj, "report_id", "reportId", "id", "reportID") or _get_first(pd, "report_id", "reportId", "id", "reportID")
        report_date = _get_first(report_obj, "report_date", "reportDate", "date", "created_at", "createdAt") or _get_first(pd, "report_date", "reportDate", "date", "created_at", "createdAt")

        sampling_rate = (
            _get_first(report_obj, "sampling_rate", "samplingRate")
            or _get_first(report_obj.get("ecg_data", {}) if isinstance(report_obj.get("ecg_data"), dict) else {}, "sampling_rate", "samplingRate")
            or 500
        )
        try:
            sampling_rate = float(sampling_rate or 500)
        except Exception:
            sampling_rate = 500.0

        # Waveform sources (accept several shapes)
        ecg_data = {}
        if isinstance(report_obj.get("ecg_data"), dict):
            ecg_data = dict(report_obj.get("ecg_data") or {})
            ecg_data.setdefault("sampling_rate", sampling_rate)
        elif isinstance(report_obj.get("leads"), dict):
            ecg_data = {"sampling_rate": sampling_rate}
            for lead in self.LEADS:
                arr = report_obj["leads"].get(lead)
                if isinstance(arr, list):
                    ecg_data[lead] = arr
        elif isinstance(report_obj.get("leads_data"), dict):
            ecg_data = {"sampling_rate": sampling_rate, "leads_data": report_obj.get("leads_data")}
        elif isinstance(report_obj.get("device_data"), str):
            ecg_data = {"sampling_rate": sampling_rate, "device_data": report_obj.get("device_data")}
        else:
            # Try raw lead strings like "lead1_reading"
            possible_keys = [
                ["lead1_reading", "lead_1_reading", "lead1"],
                ["lead2_reading", "lead_2_reading", "lead2"],
                ["lead3_reading", "lead_3_reading", "lead3"],
                ["leadavr_reading", "lead_avr_reading", "leadavr"],
                ["leadavl_reading", "lead_avl_reading", "leadavl"],
                ["leadavf_reading", "lead_avf_reading", "leadavf"],
                ["leadv1_reading", "lead_v1_reading", "leadv1"],
                ["leadv2_reading", "lead_v2_reading", "leadv2"],
                ["leadv3_reading", "lead_v3_reading", "leadv3"],
                ["leadv4_reading", "lead_v4_reading", "leadv4"],
                ["leadv5_reading", "lead_v5_reading", "leadv5"],
                ["leadv6_reading", "lead_v6_reading", "leadv6"],
            ]
            lower_keys = {k.lower(): k for k in report_obj.keys()}
            import numpy as _np
            from scipy.ndimage import gaussian_filter1d as _gf
            ecg_data = {"sampling_rate": sampling_rate}
            for i, variants in enumerate(possible_keys):
                leadstr = self.LEADS[i]
                for variant in variants:
                    actual_key = lower_keys.get(variant.lower())
                    if actual_key and actual_key in report_obj:
                        val_str = str(report_obj[actual_key]).strip()
                        if val_str.endswith(","):
                            val_str = val_str[:-1]
                        if val_str:
                            arr_vals = _np.array([float(x.strip()) for x in val_str.split(",") if x.strip()])
                            filt = _gf(arr_vals, sigma=1.5)
                            c_mean = _np.mean(filt)
                            if not _np.isnan(c_mean):
                                filt = filt - c_mean
                            filt = filt + 2048
                            ecg_data[leadstr] = filt.tolist()
                        break

        # Ensure we actually have waveform arrays
        has_any = False
        if isinstance(ecg_data, dict):
            for lead in self.LEADS:
                if isinstance(ecg_data.get(lead), list) and len(ecg_data.get(lead)) > 10:
                    has_any = True
                    break
            if not has_any and isinstance(ecg_data.get("leads_data"), dict):
                for lead in self.LEADS:
                    arr = ecg_data["leads_data"].get(lead)
                    if isinstance(arr, list) and len(arr) > 10:
                        has_any = True
                        break
            if not has_any and isinstance(ecg_data.get("device_data"), str) and "|" in ecg_data["device_data"]:
                has_any = True

        if not has_any:
            return None

        internal = {
            "patient_details": {
                "name": patient_name or "Unknown",
                "age": patient_age or "",
                "gender": patient_gender or "",
                "report_id": report_id or "",
                "report_date": report_date or "",
            },
            "result_reading": report_obj.get("result_reading") or report_obj.get("metrics") or {},
            "clinical_findings": report_obj.get("clinical_findings") or {},
            "ecg_data": ecg_data,
            "report_type": report_obj.get("report_type") or report_obj.get("reportType") or report_obj.get("type") or "",
            "source": "public_mobile",
        }
        return internal


    def _fit_window_to_screen(self):
        screen = self.screen() or QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            self.setGeometry(available.adjusted(6, 6, -6, -6))
        self._apply_responsive_window_layout()

    def _apply_responsive_window_layout(self):
        size = self.size()
        screen_h = max(size.height(), 1)
        screen_w = max(size.width(), 1)

        if hasattr(self, "_tool_sidebar") and self._tool_sidebar is not None:
            sidebar_width = max(76, min(96, int(screen_w * 0.055)))
            self._tool_sidebar.setFixedWidth(sidebar_width)

        if hasattr(self, "_bottom_panel") and self._bottom_panel is not None:
            bottom_min = max(220, min(280, int(screen_h * 0.24)))
            bottom_max = max(bottom_min, min(360, int(screen_h * 0.34)))
            self._bottom_panel.setMinimumHeight(bottom_min)
            self._bottom_panel.setMaximumHeight(bottom_max)

        if hasattr(self, "annotation_table") and self.annotation_table is not None:
            self.annotation_table.setMaximumHeight(max(96, min(170, int(screen_h * 0.15))))

        if hasattr(self, "metrics_table") and self.metrics_table is not None:
            self.metrics_table.setMaximumHeight(max(72, min(120, int(screen_h * 0.11))))

        if hasattr(self, "findings_text") and self.findings_text is not None:
            self.findings_text.setMaximumHeight(max(56, min(84, int(screen_h * 0.07))))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_responsive_window_layout()

    # ─────────────────────────────────────────────────────────────────────────
    #  KEYBOARD SHORTCUTS
    # ─────────────────────────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        if (event.modifiers() & Qt.ControlModifier) and key == Qt.Key_Z:
            self._undo_last_ruler()
        elif key == Qt.Key_Backspace:
            self._undo_last_ruler()
        elif key == Qt.Key_S:
            self.set_tool(TOOL_SELECT)
            self._tool_btns[TOOL_SELECT].setChecked(True)
        elif key == Qt.Key_R:
            self.set_tool(TOOL_RULER)
            self._tool_btns[TOOL_RULER].setChecked(True)
        elif key == Qt.Key_C:
            self.set_tool(TOOL_CALIPER)
            self._tool_btns[TOOL_CALIPER].setChecked(True)
        elif key == Qt.Key_M:
            self.set_tool(TOOL_MAGNIFY)
            self._tool_btns[TOOL_MAGNIFY].setChecked(True)
        elif key == Qt.Key_A:
            self.set_tool(TOOL_ANNOTATE)
            self._tool_btns[TOOL_ANNOTATE].setChecked(True)
        elif key == Qt.Key_Left:
            self.prev_frame()
        elif key == Qt.Key_Right:
            self.next_frame()
        elif key == Qt.Key_Space:
            self.toggle_play()
        elif key == Qt.Key_Escape:
            if self._active_lead_popup is not None and self._active_lead_popup.isVisible():
                self._active_lead_popup.close()
        elif key == Qt.Key_Return or key == Qt.Key_Enter:
            # Ignore Enter/Return to prevent window from closing
            pass
        else:
            super().keyPressEvent(event)

    # ─────────────────────────────────────────────────────────────────────────
    def load_selected_report(self, index=0):
        """Load selected report from dropdown."""
        try:
            self._load_and_render_report(index)
        except Exception:
            pass

    def load_reports(self):
        """Load public/mobile reports."""
        try:
            self.load_mobile_reports()
        except Exception:
            pass

    def export_report(self):
        """Export report JSON."""
        pass

    def _load_and_render_report(self, index):
        """Load and render the selected report by index."""
        if index < 0 or index >= len(self.reports):
            return
        self.current_report = self.reports[index]
        self.current_report_path = ""  # No file path for API-loaded reports
        self._update_patient_info()
        self._load_lead_data()
        self._load_metrics_findings()
        self._load_manual_annotations()
        self.frame_start_sample = 0
        self.pending_mark_start_sec = None
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText("No active mark")
        self.mark_status_lbl.setStyleSheet("color:#5a8a5a;")
        self._update_timeline_limits()
        self._render_current_frame()

    def _extract_patient_name(self, report):
        return (report.get('patient_details', {}).get('name')
                or report.get('patient_name')
                or report.get('patient', {}).get('name')
                or 'Unknown')

    def _extract_report_date(self, report):
        return (report.get('patient_details', {}).get('report_date')
                or report.get('report_date')
                or report.get('date')
                or 'Unknown Date')

    def _update_patient_info(self):
        if not self.current_report:
            self.patient_lbl.setText("Patient: —")
            self.patient_meta_lbl.setText("ID: — | Age: — | Gender: —")
            return
        pd        = self.current_report.get('patient_details', {}) or {}
        p_fallback = self.current_report.get('patient', {}) or {}
        pd_name = pd.get('name') if pd.get('name') and pd.get('name') != 'Unknown' else None
        name   = pd_name or self.current_report.get('patient_name') or self.current_report.get('name') or p_fallback.get('name') or pd.get('name') or 'Unknown'
        pid    = pd.get('report_id') or pd.get('user_id') or self.current_report.get('patient_id') or self.current_report.get('report_id') or '—'
        age    = pd.get('age')    or self.current_report.get('age')    or self.current_report.get('patient_age') or p_fallback.get('age')    or '—'
        gender = pd.get('gender') or self.current_report.get('gender') or self.current_report.get('patient_gender') or p_fallback.get('gender') or '—'
        self.patient_lbl.setText(f"Patient: {name}")
        self.patient_meta_lbl.setText(f"ID: {pid} | Age: {age} | Gender: {gender}")

    def _load_lead_data(self):
        self.lead_data = {lead: np.array([]) for lead in self.LEADS}
        rpt = self.current_report or {}
        self.sampling_rate = float(
            rpt.get('data_details', {}).get('sampling_rate')
            or rpt.get('sampling_rate')
            or rpt.get('ecg_data', {}).get('sampling_rate')
            or 500)
        ecg_data  = rpt.get('ecg_data', {}) if isinstance(rpt.get('ecg_data', {}), dict) else {}
        leads_data = ecg_data.get('leads_data') if isinstance(ecg_data.get('leads_data'), dict) else None
        if leads_data:
            for lead in self.LEADS:
                arr = leads_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
        if any(lead in ecg_data for lead in self.LEADS):
            for lead in self.LEADS:
                arr = ecg_data.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
        if any(lead in rpt for lead in self.LEADS):
            for lead in self.LEADS:
                arr = rpt.get(lead, [])
                self.lead_data[lead] = np.array(arr, dtype=float) if isinstance(arr, list) else np.array([])
            return
        device_data = ecg_data.get('device_data') if isinstance(ecg_data, dict) else None
        if isinstance(device_data, str) and '|' in device_data:
            # For compact per-frame encoding (12 integers per frame), keep initial 5000 frames
            # so the first 10 seconds (500Hz default) plots immediately without huge memory usage.
            self._parse_compact_device_data(device_data, max_frames=5000)

    def _parse_compact_device_data(self, device_data, max_frames: int = None):
        per_lead = {lead: [] for lead in self.LEADS}
        frames = [x.strip() for x in device_data.split('|') if x.strip()]
        if isinstance(max_frames, int) and max_frames > 0:
            frames = frames[:max_frames]
        for fr in frames:
            try:
                vals = json.loads(fr)
                if isinstance(vals, list) and len(vals) >= 12:
                    for i, lead in enumerate(self.LEADS):
                        per_lead[lead].append(float(vals[i]))
            except Exception:
                continue
        for lead in self.LEADS:
            self.lead_data[lead] = np.array(per_lead[lead], dtype=float)

    def _load_metrics_findings(self):
        rpt     = self.current_report or {}
        metrics = rpt.get('result_reading') or rpt.get('metrics') or {}
        self.metrics_table.setRowCount(0)
        rv5_val = metrics.get('RV5_mV', metrics.get('RV5', metrics.get('rv5')))
        sv1_val = metrics.get('SV1_mV', metrics.get('SV1', metrics.get('sv1')))
        rv5_sv1 = metrics.get('RV5_SV1', metrics.get('rv5_sv1'))
        if rv5_sv1 is None and rv5_val is not None and sv1_val is not None:
            try:
                rv5_sv1 = f"{float(rv5_val):.3f}/{abs(float(sv1_val)):.3f}"
            except Exception:
                rv5_sv1 = 'N/A'
        elif rv5_sv1 is None:
            rv5_sv1 = 'N/A'

        rv5_plus_sv1 = metrics.get('RV5_plus_SV1_mV', metrics.get('RV5_plus_SV1', metrics.get('rv5_plus_sv1')))
        if (rv5_plus_sv1 is None or rv5_plus_sv1 == 'N/A') and rv5_val is not None and sv1_val is not None:
            try:
                rv5_plus_sv1 = f"{round(float(rv5_val) + abs(float(sv1_val)), 3)}"
            except Exception:
                rv5_plus_sv1 = 'N/A'
        elif rv5_plus_sv1 is None:
            rv5_plus_sv1 = 'N/A'

        if isinstance(rv5_sv1, (list, tuple)) and len(rv5_sv1) >= 2:
            try:
                rv5_sv1 = f"{float(rv5_sv1[0]):.3f}/{abs(float(rv5_sv1[1])):.3f}"
            except Exception:
                pass

        qtcf_val = metrics.get('QTcF', metrics.get('qtcf_interval', metrics.get('QTcF_ms', metrics.get('QTCF_ms', metrics.get('QTCF', 'N/A')))))

        items = [
            ("HR",     metrics.get('HR_bpm',  metrics.get('heart_rate', metrics.get('HR', 'N/A'))),    "bpm"),
            ("RR",     metrics.get('RR_ms',   metrics.get('rr_interval', metrics.get('RR', 'N/A'))),   "ms"),
            ("PR",     metrics.get('PR_ms',   metrics.get('pr_interval', metrics.get('PR', 'N/A'))),   "ms"),
            ("QRS",    metrics.get('QRS_ms',  metrics.get('qrs_duration', metrics.get('QRS', 'N/A'))), "ms"),
            ("QT",     metrics.get('QT_ms',   metrics.get('qt_interval', metrics.get('QT', 'N/A'))),   "ms"),
            ("QTc",    metrics.get('QTc_ms',  metrics.get('qtc_interval', metrics.get('QTc', 'N/A'))), "ms"),
            ("QTcF",   qtcf_val,                                                                       "ms"),
            ("RV5/SV1",  str(rv5_sv1).replace(' mV', ''),      "mV"),
            ("RV5+SV1",  str(rv5_plus_sv1).replace(' mV', ''), "mV"),
        ]
        self.metrics_table.setRowCount(len(items))
        for i, (k, v, unit) in enumerate(items):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(k))
            self.metrics_table.setItem(i, 1, QTableWidgetItem(
                f"{v} {unit}" if v not in ('', None, 'N/A') else 'N/A'))

        findings_lines = []
        clinical = rpt.get('clinical_findings', {})
        if isinstance(clinical, dict):
            for key in ('conclusion', 'arrhythmia', 'hyperkalemia'):
                vals = clinical.get(key, [])
                if isinstance(vals, list) and vals:
                    findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))
        for key in ('conclusion', 'arrhythmia', 'hyperkalemia', 'findings', 'recommendations'):
            vals = rpt.get(key)
            if isinstance(vals, list) and vals:
                findings_lines.append(f"{key.title()}: " + ', '.join(str(x) for x in vals))
        self.findings_text.setPlainText('\n'.join(findings_lines) or "No backend findings.")

    # ─────────────────────────────────────────────────────────────────────────
    #  FRAME NAVIGATION  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def _on_window_changed(self, text):
        self.window_seconds = float(text.replace('s', '').strip())
        self._update_timeline_limits()
        self._render_current_frame()

    def _on_step_changed(self, text):
        self.step_seconds = float(text.replace('s', '').strip())

    def _total_samples(self):
        for lead in self.LEADS:
            if len(self.lead_data[lead]) > 0:
                return len(self.lead_data[lead])
        return 0

    def _window_samples(self):
        return max(1, int(round(self.window_seconds * self.sampling_rate)))

    def _step_samples(self):
        return max(1, int(round(self.step_seconds * self.sampling_rate)))

    def _max_start_sample(self):
        return max(0, self._total_samples() - self._window_samples())

    def _update_timeline_limits(self):
        mx = self._max_start_sample()
        self.timeline.blockSignals(True)
        self.timeline.setMinimum(0)
        self.timeline.setMaximum(mx)
        self.timeline.setValue(min(self.frame_start_sample, mx))
        self.timeline.blockSignals(False)

    def _on_timeline_changed(self, value):
        self.frame_start_sample = int(value)
        self._render_current_frame()

    def prev_frame(self):
        self.frame_start_sample = max(0, self.frame_start_sample - self._step_samples())
        self.timeline.setValue(self.frame_start_sample)

    def next_frame(self):
        self.frame_start_sample = min(
            self._max_start_sample(), self.frame_start_sample + self._step_samples())
        self.timeline.setValue(self.frame_start_sample)

    def toggle_play(self):
        if self.play_timer.isActive():
            self.play_timer.stop()
            self.play_btn.setText("▶ Play")
        else:
            self.play_timer.start(250)
            self.play_btn.setText("⏸ Pause")

    # ─────────────────────────────────────────────────────────────────────────
    #  RENDERING
    # ─────────────────────────────────────────────────────────────────────────
    def _render_current_frame(self):
        ws = self._window_samples()
        st = self.frame_start_sample
        en = min(self._total_samples(), st + ws)

        t          = np.arange(st, en) / self.sampling_rate if en > st else np.array([])
        start_sec  = st / self.sampling_rate if self.sampling_rate > 0 else 0.0
        end_sec    = en / self.sampling_rate if self.sampling_rate > 0 else 0.0
        self.frame_label.setText(f"Frame: {start_sec:.2f}s – {end_sec:.2f}s")

        ECG_COLOR   = '#00d000'   # Classic green trace
        GRID_COLOR  = '#1a3a1a'
        ANNOT_COLOR = '#ff3333'

        for lead in self.LEADS:
            ax     = self._lead_axes.get(lead)
            fig    = self._lead_figs.get(lead)
            canvas = self._lead_canvases.get(lead)
            if ax is None:
                continue

            ax.clear()
            ax.set_axis_off()
            ax.set_facecolor('#080e08')

            if lead == 'aVR':
                ax.set_ylim(-4096, 0)
            else:
                ax.set_ylim(0, 4096)
            ax.set_xlim(start_sec, max(end_sec, start_sec + 1))

            # ECG paper grid
            ax.grid(True, color=GRID_COLOR, linewidth=0.35, linestyle='-', alpha=1.0)

            data = self.lead_data.get(lead, np.array([]))
            if len(data) > 0 and en > st:
                seg = data[st:en]
                if lead == 'aVR':
                    ax.plot(t, -seg, color=ECG_COLOR, linewidth=0.9, antialiased=True)
                    ax.set_ylim(-4096, 0)
                else:
                    ax.plot(t, seg, color=ECG_COLOR, linewidth=0.9, antialiased=True)
                    ax.set_ylim(0, 4096)
                ax.set_xlim(start_sec, end_sec)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                        transform=ax.transAxes, color='#334433', fontsize=9)

            # Annotation spans
            for ann in self.manual_annotations:
                if ann.get('lead', 'All Leads') not in ('All Leads', lead):
                    continue
                a0 = ann.get('start_sec', 0.0)
                a1 = ann.get('end_sec',   0.0)
                if a1 < start_sec or a0 > end_sec:
                    continue
                lft = max(a0, start_sec)
                rgt = min(a1, end_sec)
                if rgt > lft:
                    ax.axvspan(lft, rgt, color=ANNOT_COLOR, alpha=0.15)
                    # Label at top of span
                    ylim = ax.get_ylim()
                    ax.text((lft + rgt) / 2, ylim[1] * 0.92,
                             ann.get('type', '')[:12],
                             fontsize=5.5, color='#ff8888', ha='center',
                             va='top', zorder=10,
                             bbox=dict(boxstyle='round,pad=0.1',
                                       fc='#200000', alpha=0.6, ec='none'))

            fig.tight_layout(pad=0)
            canvas.draw_idle()

    def lead_has_visible_data(self, lead_name):
        data = self.lead_data.get(lead_name, np.array([]))
        if len(data) == 0:
            return False
        st = self.frame_start_sample
        en = min(len(data), st + self._window_samples())
        return en > st and len(data[st:en]) > 0

    # ─────────────────────────────────────────────────────────────────────────
    #  MANUAL ANNOTATIONS  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def mark_start(self):
        self.pending_mark_start_sec = self.frame_start_sample / max(self.sampling_rate, 1.0)
        self.mark_end_btn.setEnabled(True)
        self.mark_status_lbl.setText(
            f"✅ Start @ {self.pending_mark_start_sec:.2f}s  →  navigate to end → ② Mark End")
        self.mark_status_lbl.setStyleSheet("color:#f5c518;")

    def mark_end_and_save(self):
        if self.pending_mark_start_sec is None:
            QMessageBox.information(self, "Marking", "Click 'Mark Start' first.")
            return
        end_sec   = (self.frame_start_sample + self._window_samples()) / max(self.sampling_rate, 1.0)
        start_sec = min(self.pending_mark_start_sec, end_sec)
        end_sec   = max(self.pending_mark_start_sec, end_sec)
        arr_type  = self.arrhythmia_type_combo.currentText().strip()
        if arr_type == 'Other':
            arr_type = self.manual_type_input.text().strip() or 'Other'
        ann = {
            'start_sec':  round(start_sec, 3),
            'end_sec':    round(end_sec, 3),
            'type':       arr_type,
            'lead':       self.mark_lead_combo.currentText(),
            'notes':      self.notes_input.text().strip(),
            'created_at': datetime.now().isoformat(timespec='seconds')
        }
        self.manual_annotations.append(ann)
        self.pending_mark_start_sec = None
        self.mark_end_btn.setEnabled(False)
        self.mark_status_lbl.setText(f"✅ Saved: {arr_type}")
        self.mark_status_lbl.setStyleSheet("color:#7adf7a;")
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()

    def delete_selected_annotation(self):
        row = self.annotation_table.currentRow()
        if row < 0 or row >= len(self.manual_annotations):
            return
        del self.manual_annotations[row]
        self._refresh_annotation_table()
        self._persist_annotations_in_report()
        self._render_current_frame()

    def _refresh_annotation_table(self):
        self.annotation_table.setRowCount(len(self.manual_annotations))
        for i, ann in enumerate(self.manual_annotations):
            self.annotation_table.setItem(i, 0, QTableWidgetItem(f"{ann.get('start_sec',0):.3f}"))
            self.annotation_table.setItem(i, 1, QTableWidgetItem(f"{ann.get('end_sec',0):.3f}"))
            self.annotation_table.setItem(i, 2, QTableWidgetItem(ann.get('type', '')))
            self.annotation_table.setItem(i, 3, QTableWidgetItem(ann.get('lead', 'All Leads')))
            self.annotation_table.setItem(i, 4, QTableWidgetItem(ann.get('notes', '')))

    def _load_manual_annotations(self):
        self.manual_annotations = list((self.current_report or {}).get('manual_annotations', []))
        self._refresh_annotation_table()

    def _persist_annotations_in_report(self):
        if not self.current_report:
            return
        self.current_report['manual_annotations'] = self.manual_annotations

    # ─────────────────────────────────────────────────────────────────────────
    #  API FETCH  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    def fetch_api_report(self):
        id_text = self.api_id_input.text().strip()
        if not id_text:
            return
        url = f"https://deckmount.in/ankur_bhaiya.php?id={id_text}"
        import requests
        from scipy.ndimage import gaussian_filter1d
        try:
            self.api_fetch_btn.setText("…")
            QApplication.processEvents()
            resp     = requests.get(url, timeout=10)
            data     = resp.json()
            if not data.get("status"):
                QMessageBox.warning(self, "API Error", "ID not found")
                self.api_fetch_btn.setText("Fetch"); return
            api_data = data.get("data", {})

            res_reading = {}
            try: res_reading = json.loads(api_data.get("result_reading", "{}"))
            except: pass
            concl = []
            try: concl = json.loads(api_data.get("conclusion", "[]"))
            except: pass
            arr = []
            try: arr = json.loads(api_data.get("arrhythmia", "[]"))
            except: pass

            ecg_data = {}
            ecg_data["sampling_rate"] = float(api_data.get("sampling_rate", 500))
            possible_keys = [
                ["lead1_reading","lead_1_reading","lead1"],
                ["lead2_reading","lead_2_reading","lead2"],
                ["lead3_reading","lead_3_reading","lead3"],
                ["leadavr_reading","lead_avr_reading","leadavr"],
                ["leadavl_reading","lead_avl_reading","leadavl"],
                ["leadavf_reading","lead_avf_reading","leadavf"],
                ["leadv1_reading","lead_v1_reading","leadv1"],
                ["leadv2_reading","lead_v2_reading","leadv2"],
                ["leadv3_reading","lead_v3_reading","leadv3"],
                ["leadv4_reading","lead_v4_reading","leadv4"],
                ["leadv5_reading","lead_v5_reading","leadv5"],
                ["leadv6_reading","lead_v6_reading","leadv6"],
            ]
            lower_keys  = {k.lower(): k for k in api_data.keys()}
            leads_list  = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            for i, variants in enumerate(possible_keys):
                leadstr = leads_list[i]
                for variant in variants:
                    actual_key = lower_keys.get(variant.lower())
                    if actual_key and actual_key in api_data:
                        val_str = str(api_data[actual_key]).strip()
                        if val_str:
                            if val_str.endswith(','): val_str = val_str[:-1]
                            arr_vals = np.array(
                                [float(x.strip()) for x in val_str.split(',') if x.strip()])
                            filt   = gaussian_filter1d(arr_vals, sigma=1.5)
                            c_mean = np.mean(filt)
                            if not np.isnan(c_mean):
                                filt = filt - c_mean
                            filt = filt + 2048
                            ecg_data[leadstr] = filt.tolist()
                        break

            new_report = {
                "patient_details": {
                    "name":        api_data.get("name", "Unknown"),
                    "age":         api_data.get("age", ""),
                    "gender":      api_data.get("gender", ""),
                    "report_id":   api_data.get("report_id", id_text),
                    "report_date": api_data.get("report_date", ""),
                },
                "result_reading": res_reading,
                "clinical_findings": {"conclusion": concl, "arrhythmia": arr},
                "ecg_data": ecg_data,
                "api_id": id_text,
            }
            self.reports.append(new_report)
            # Report combo box removed - skip UI update
            # idx  = len(self.reports) - 1
            # name = api_data.get("name", "Unknown API")
            # self.report_combo.addItem(f"[API] {name} | ID:{id_text}", "")
            # self.report_combo.setCurrentIndex(idx)
            self.api_fetch_btn.setText("Fetch")
        except requests.exceptions.RequestException as e:
            from utils.ui_feedback import offline_action_message, show_critical
            self.api_fetch_btn.setText("Fetch")
            show_critical(
                self,
                "No Internet Connection",
                offline_action_message(
                    "Fetching an API report",
                    "This feature needs an active connection to the public API.",
                ),
                details=str(e),
            )
        except Exception as e:
            self.api_fetch_btn.setText("Fetch")
            from utils.ui_feedback import show_critical
            show_critical(self, "API Error", "Failed to load the report.", details=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    #  EXPORT / PDF  (identical to original)
    # ─────────────────────────────────────────────────────────────────────────
    # Removed: export_report() - JSON export button removed

    def generate_pdf_report(self):
        if not self.current_report:
            QMessageBox.warning(self, "Export", "No report loaded."); return

        rpt = self.current_report
        pat = rpt.get('patient_details', {}) or {}

        raw_metrics = rpt.get('result_reading') or rpt.get('metrics') or {}
        if isinstance(raw_metrics, str):
            try: raw_metrics = json.loads(raw_metrics)
            except: raw_metrics = {}
        if not isinstance(raw_metrics, dict): raw_metrics = {}

        def _get(d, *keys):
            for k in keys:
                v = d.get(k)
                if v is not None and str(v).strip() not in ('', '--', 'N/A', 'null'):
                    return str(v)
            return None

        hr   = _get(raw_metrics, 'HR',  'heart_rate',   'HR_bpm')
        pr   = _get(raw_metrics, 'PR',  'pr_interval',  'PR_ms')
        qrs  = _get(raw_metrics, 'QRS', 'qrs_duration', 'QRS_ms')
        qt   = _get(raw_metrics, 'QT',  'qt_interval',  'QT_ms')
        qtc  = _get(raw_metrics, 'QTc', 'qtc_interval', 'QTc_ms')
        qtcf = _get(raw_metrics, 'QTcF','qtcf_interval','QTcF_ms', 'QTCF_ms', 'QTCF', 'qtcf_ms')
        rr   = _get(raw_metrics, 'RR',  'rr_interval',  'RR_ms', 'rr_ms', 'rr')
        rv5  = _get(raw_metrics, 'RV5_mV', 'RV5', 'rv5', 'rv5_mv')
        sv1  = _get(raw_metrics, 'SV1_mV', 'SV1', 'sv1', 'sv1_mv')
        rv5sv1  = _get(raw_metrics, 'RV5_SV1',     'rv5_sv1')
        rv5plus = _get(raw_metrics, 'RV5_plus_SV1_mV', 'RV5_plus_SV1', 'rv5_plus_sv1')
        if isinstance(rv5sv1, (list, tuple)) and len(rv5sv1) >= 2:
            try:
                rv5sv1 = f"{float(rv5sv1[0]):.3f}/{abs(float(rv5sv1[1])):.3f}"
            except Exception:
                pass
        axes_s  = _get(raw_metrics, 'axes','P/QRS/T','p_qrs_t')

        clinical   = rpt.get('clinical_findings') or {}
        conclusions = clinical.get('conclusion', []) if isinstance(clinical, dict) else []
        if isinstance(conclusions, str): conclusions = [conclusions]
        elif not isinstance(conclusions, list): conclusions = []
        if not conclusions:
            c2 = rpt.get('conclusion', [])
            conclusions = [c2] if isinstance(c2, str) else (c2 if isinstance(c2, list) else [])

        p_fallback = rpt.get('patient', {}) or {}
        pd_name = pat.get('name') if pat.get('name') and pat.get('name') != 'Unknown' else None
        patient_name = pd_name or rpt.get('patient_name') or rpt.get('name') or p_fallback.get('name') or pat.get('name') or 'Unknown'
        patient_age = pat.get('age') or rpt.get('age') or rpt.get('patient_age') or p_fallback.get('age') or ''
        patient_gender = pat.get('gender') or rpt.get('gender') or rpt.get('patient_gender') or p_fallback.get('gender') or ''
        timestamp    = datetime.now().strftime('%Y%m%d_%H%M%S')

        # ── Embed device serial in filename so the Lambda regex can parse it ──
        # Lambda expects: ..._{deviceId}_{YYYYMMDD}_{HHMMSS}.pdf
        # deviceId = last 4 alphanumeric chars of the RhythmUltra serial (e.g. "A010")
        _device_serial_tag = "0000"
        try:
            from utils.license_manager import load_token_file, get_RhythmUltra_serial
            _tok = load_token_file() or {}
            _ser = (_tok.get("rhythmultra_serial")
                    or _tok.get("RhythmUltra_serial")
                    or _tok.get("rhythmulta_serial")
                    or "")
            if not _ser:
                _ser = get_RhythmUltra_serial() or ""
            _ser_clean = ''.join(c for c in _ser if c.isalnum())
            if len(_ser_clean) >= 2:
                _device_serial_tag = _ser_clean[-4:] if len(_ser_clean) >= 4 else _ser_clean
        except Exception:
            pass

        project_root = Path(__file__).resolve().parents[2]
        reports_dir  = project_root / "reports"
        reports_dir.mkdir(exist_ok=True)
        # Filename format: ECG_Analysis_{patient}_{deviceTag}_{YYYYMMDD}_{HHMMSS}.pdf
        _safe_pname = ''.join(c if c.isalnum() or c in ('_', '-') else '_' for c in (patient_name or 'Unknown'))
        _default_fname = f"ECG_Report_Analysis_{_safe_pname}_{_device_serial_tag}_{timestamp}.pdf"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save ECG PDF",
            str(reports_dir / _default_fname),
            "PDF Files (*.pdf)")
        if not path: return

        from PyQt5.QtWidgets import (
            QDialog as _QD, QVBoxLayout as _VL, QButtonGroup,
            QRadioButton, QDialogButtonBox, QLabel as _QL, QCheckBox
        )
        fmt_dlg = _QD(self)
        fmt_dlg.setWindowTitle("Report Format")
        fmt_dlg.setMinimumWidth(360)
        fmt_dlg.setStyleSheet(
            "QDialog{background:#12152a;color:white;} QLabel{color:#e0e0ff;font-size:12px;"
            "background:transparent;border:none;} QRadioButton{color:#e0e0ff;font-size:11px;"
            "background:transparent;padding:5px;} QPushButton{background:#0097e6;color:white;"
            "border:none;border-radius:5px;padding:7px 20px;font-weight:bold;}")
        fmt_lay = _VL(fmt_dlg)
        fmt_lay.setContentsMargins(22, 18, 22, 18)
        fmt_lay.addWidget(_QL("Choose ECG report layout:"))
        rb1 = QRadioButton("4:3  —  Standard 12-lead grid")
        rb2 = QRadioButton("12:1 —  Full rhythm strip roll")
        rb3 = QRadioButton("6:2  —  Compact comparative")
        rb1.setChecked(True)
        grp = QButtonGroup(fmt_dlg)
        for rb in (rb1, rb2, rb3):
            grp.addButton(rb); fmt_lay.addWidget(rb)

        anonymize_cb = QCheckBox("Hide patient details on PDF (send to server only)")
        anonymize_cb.setChecked(False)
        fmt_lay.addWidget(anonymize_cb)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(fmt_dlg.accept)
        bb.rejected.connect(fmt_dlg.reject)
        fmt_lay.addWidget(bb)
        if fmt_dlg.exec_() != _QD.Accepted: return
        pdf_format = "4_3" if rb1.isChecked() else ("12_1" if rb2.isChecked() else "6_2")
        anonymize_pdf = bool(anonymize_cb.isChecked())

        try:
            self.pdf_btn.setText("Generating…")
            QApplication.processEvents()

            st = self.frame_start_sample
            leads_order = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
            snap_raw = []
            for l in leads_order:
                d = self.lead_data.get(l, np.array([]))
                snap_raw.append(d[st:] if len(d) > st else np.array([]))

            lead_seq = 'Standard'
            try:
                from utils.settings_manager import SettingsManager
                lead_seq = str(SettingsManager().get_setting('lead_sequence', 'Standard') or 'Standard').strip()
            except Exception:
                pass

            # Read current filter settings
            filter_ac = getattr(self, "filter_ac", "50")
            filter_emg = getattr(self, "filter_emg", "25")
            filter_dft = getattr(self, "filter_dft", "off")
            try:
                from utils.settings_manager import SettingsManager
                sm = SettingsManager()
                filter_ac = str(sm.get_setting("filter_ac", filter_ac) or filter_ac).strip()
                filter_emg = str(sm.get_setting("filter_emg", filter_emg) or filter_emg).strip()
                filter_dft = str(sm.get_setting("filter_dft", filter_dft) or filter_dft).strip()
            except Exception:
                pass

            if filter_dft not in ("off", "") and filter_emg not in ("off", ""):
                filter_band = f"{filter_dft}-{filter_emg}Hz"
            elif filter_dft not in ("off", ""):
                filter_band = f"HP:{filter_dft}Hz"
            elif filter_emg not in ("off", ""):
                filter_band = f"LP:{filter_emg}Hz"
            else:
                filter_band = "Filter: Off"
            ac_freq = f"{filter_ac}Hz" if str(filter_ac) in ("50", "60") else "Off"

            frozen = {
                'HR': int(float(hr) if hr else 0),
                'RR': int(float(rr) if rr else 0),
                'PR': int(float(pr) if pr else 0),
                'QRS': int(float(qrs) if qrs else 0),
                'QT': int(float(qt) if qt else 0),
                'QTc': int(float(qtc) if qtc else 0),
                'QTcF': int(float(qtcf) if qtcf else 0),
                'rv5': 0.0, 'sv1': 0.0,
                'p_axis': '--', 'QRS_axis': '--', 't_axis': '--',
                'lead_seq': lead_seq,
                'logo_path': str(self.analysis_pdf_logo_path),
                'filter_band': filter_band,
                'ac_frequency': ac_freq,
            }
            try:
                if rv5 is not None:
                    frozen['rv5'] = float(rv5)
                if sv1 is not None:
                    frozen['sv1'] = float(sv1)
                if (frozen['rv5'] == 0.0 or frozen['sv1'] == 0.0) and rv5sv1:
                    parts = str(rv5sv1).split('/')
                    frozen['rv5'] = float(parts[0].strip(' mV+'))
                    if len(parts) > 1:
                        frozen['sv1'] = float(parts[1].strip(' mV+'))
            except: pass
            try:
                if axes_s and len(str(axes_s).split('/')) == 3:
                    p = str(axes_s).split('/')
                    frozen['p_axis'] = p[0].strip()
                    frozen['QRS_axis'] = p[1].strip()
                    frozen['t_axis'] = p[2].strip()
            except: pass

            pat_mapped = {
                'first_name': '' if anonymize_pdf else patient_name,
                'last_name': '',
                'age': '' if anonymize_pdf else patient_age,
                'gender': '' if anonymize_pdf else patient_gender,
                'date_time': pat.get('report_date') or datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'doctor_name': '' if anonymize_pdf else (pat.get('doctor_name', '') or pat.get('doctor', '')),
                'org': '' if anonymize_pdf else pat.get('Org.', ''),
                'phone': '' if anonymize_pdf else (pat.get('phone', '') or pat.get('doctor_mobile', '')),
                'name': '' if anonymize_pdf else patient_name,
            }
            extra_figs = []
            if self.manual_annotations:
                fig2 = self._generate_annotation_page()
                if fig2: extra_figs.append(fig2)

            from ecg.ecg_report_android import generate_report as _gen
            _gen(snap_raw=snap_raw, frozen=frozen, patient=pat_mapped,
                 filename=path, fmt=pdf_format, conc_list=conclusions,
                 fs=float(self.sampling_rate), extra_figs=extra_figs)

            # Build a companion ECG JSON (lead arrays + patient/meta) and send/upload in background.
            ecg_json_path = ""
            try:
                import os as _os
                import numpy as _np
                base_no_ext = _os.path.splitext(path)[0]
                # Use exact same name as PDF for the JSON companion
                ecg_json_path = base_no_ext + ".json"

                target_samples = int(round(float(self.sampling_rate) * 10.0)) if self.sampling_rate else 5000
                if target_samples <= 0:
                    target_samples = 5000

                leads_out = {}
                for lead_name in leads_order:
                    arr = self.lead_data.get(lead_name, _np.array([]))
                    if hasattr(arr, "__len__") and len(arr) > st:
                        seg = arr[st:st + target_samples]
                    else:
                        seg = _np.array([])
                    try:
                        seg_i = _np.rint(_np.asarray(seg, dtype=float)).astype(int).tolist()
                    except Exception:
                        seg_i = []
                    leads_out[lead_name] = seg_i

                ecg_json_obj = {
                    "patient_details": pat,
                    "report_format": pdf_format,
                    "sampling_rate": float(self.sampling_rate),
                    "frame_start_sample": int(st),
                    "generated_at": datetime.now().isoformat(timespec="seconds"),
                    "metrics": raw_metrics,
                    "clinical_findings": clinical,
                    "leads": leads_out,
                }
                with open(ecg_json_path, "w", encoding="utf-8") as f:
                    json.dump(ecg_json_obj, f, indent=2, ensure_ascii=False)
            except Exception as json_err:
                print(f"ECG JSON save failed: {json_err}")
                ecg_json_path = ""

            upload_metadata = {
                "patient_name": pat.get("name", "Unknown"),
                "patient_age": str(pat.get("age", "")),
                "patient_gender": str(pat.get("gender", "")),
                "report_date": str(pat.get("report_date", "")),
                "report_type": "analysis",
                "report_format": str(pdf_format),
            }

            def _send_to_server():
                try:
                    from utils.cloud_uploader import get_cloud_uploader
                    cloud_uploader = get_cloud_uploader()
                    if cloud_uploader.is_configured():
                        cloud_uploader.upload_report(path, metadata=upload_metadata)
                        if ecg_json_path:
                            cloud_uploader.upload_report(ecg_json_path, metadata=upload_metadata)
                except Exception as up_err:
                    print(f"Cloud upload failed: {up_err}")

                try:
                    from utils.ecg_payload_builder import dispatch_12lead_report
                    data_payload = {
                        "HR": hr or 0,
                        "RR": rr or 0,
                        "PR": pr or 0,
                        "QRS": qrs or 0,
                        "QT": qt or 0,
                        "QTc": qtc or 0,
                        "QTcF": qtcf or 0,
                        "report_date": pat.get("report_date", ""),
                        "patient": pat,
                    }
                    dispatch_12lead_report(
                        data=data_payload,
                        patient=pat,
                        pdf_path=path,
                        settings_manager=None,
                        signup_details=None,
                        ecg_test_page=None,
                        ecg_data_file=ecg_json_path or None,
                        report_format=str(pdf_format),
                        conclusions=conclusions,
                        arrhythmia=None,
                        reports_dir=str((Path(__file__).resolve().parents[2] / "reports")),
                        save_local=True,
                        send=True,
                    )
                except Exception as payload_err:
                    print(f"Unified payload send failed: {payload_err}")

            try:
                import threading as _threading
                _threading.Thread(target=_send_to_server, daemon=True).start()
            except Exception:
                _send_to_server()

            try:
                from dashboard.history_window import append_history_entry
                _p = self.parent()
                _uname = getattr(_p, "username", "") if _p is not None else ""
                _full = (getattr(_p, "user_details", {}) or {}).get("full_name") or _uname
                append_history_entry({
                    "patient_name": pat.get('name','Unknown'),
                    "age": str(pat.get('age','')),
                    "gender": pat.get('gender',''),
                    "doctor": pat.get('doctor',''),
                    "Org.": pat.get('Org.',''),
                }, path, report_type="Analysis", username=_uname, owner_full_name=_full)
            except Exception as h_err:
                print(f"History append failed: {h_err}")

            self.pdf_btn.setText("📄 PDF Report")
            QMessageBox.information(self, "PDF Saved", f"Report saved:\n{path}")
        except Exception as e:
            self.pdf_btn.setText("📄 PDF Report")
            QMessageBox.critical(self, "PDF Error", f"Failed:\n{e}")

    def _generate_annotation_page(self):
        """Annotation summary + wave strips (identical to original)."""
        if not self.manual_annotations:
            return None
        import matplotlib.patches as mpa
        PAGE_W = 210.0; PAGE_H = 297.0
        ML = 15.0; MR = 15.0; MT = 15.0; MB = 15.0
        fig = Figure(figsize=(PAGE_W/25.4, PAGE_H/25.4), dpi=150, facecolor='white')
        ax  = fig.add_axes([0,0,1,1], facecolor='white')
        ax.set_xlim(0, PAGE_W); ax.set_ylim(PAGE_H, 0); ax.set_aspect('equal')
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.set_xticklabels([]); ax.set_yticklabels([])
        ax.tick_params(left=False, bottom=False, which='both')
        ax.text(PAGE_W/2, MT+5, "ECG ARRHYTHMIA & FINDINGS REPORT",
                fontsize=12, fontweight='bold', ha='center', va='top', color='#0000cc')
        y_cursor = MT+20
        ax.text(ML, y_cursor, "Summary of Manual Annotations:",
                fontsize=9, fontweight='bold', va='top')
        y_cursor += 8
        cols = [("Start (s)",20),("End (s)",20),("Type",50),("Lead",20),("Notes",60)]
        xc = ML
        for lbl, w in cols:
            ax.text(xc, y_cursor, lbl, fontsize=8, fontweight='bold', va='top'); xc += w
        y_cursor += 5
        ax.plot([ML, PAGE_W-MR], [y_cursor]*2, color='black', linewidth=0.5)
        y_cursor += 2
        for ann in self.manual_annotations[:12]:
            xc = ML
            ax.text(xc, y_cursor, f"{ann.get('start_sec',0):.2f}", fontsize=7, va='top'); xc += 20
            ax.text(xc, y_cursor, f"{ann.get('end_sec',0):.2f}", fontsize=7, va='top');   xc += 20
            ax.text(xc, y_cursor, ann.get('type',''), fontsize=7, fontweight='bold', va='top'); xc += 50
            ax.text(xc, y_cursor, ann.get('lead',''), fontsize=7, va='top');                    xc += 20
            ax.text(xc, y_cursor, ann.get('notes',''), fontsize=7, va='top')
            y_cursor += 5
            if y_cursor > PAGE_H/2-10: break
        y_cursor = max(y_cursor+10, PAGE_H/2-20)
        ax.text(ML, y_cursor, "Waveform Strips for Marked Events:",
                fontsize=9, fontweight='bold', va='top')
        y_cursor += 10
        important = [a for a in self.manual_annotations if "Rhythm" not in str(a.get('type',''))]
        if not important: important = self.manual_annotations
        ADC_PER_MM = 128.0
        MM_PER_SAMPLE = 25.0 / float(self.sampling_rate)
        for i, ann in enumerate(important[:3]):
            if y_cursor > PAGE_H-50: break
            lead_name = ann.get('lead','II')
            if lead_name == "All Leads": lead_name = 'II'
            data = self.lead_data.get(lead_name, np.array([]))
            if len(data) == 0: continue
            start_s = ann.get('start_sec', 0); end_s = ann.get('end_sec', 0)
            duration_s = end_s - start_s
            strip_w_mm = PAGE_W - ML - MR
            time_shown_s = min(10.0, max(3.0, duration_s*1.5))
            center_s  = (start_s+end_s)/2
            strip_start_s = max(0, center_s - time_shown_s/2)
            strip_end_s   = strip_start_s + time_shown_s
            st_idx = int(strip_start_s*self.sampling_rate)
            en_idx = int(strip_end_s*self.sampling_rate)
            segment = data[st_idx:en_idx]
            if len(segment) < 10: continue
            strip_h = 30.0
            rect = mpa.FancyBboxPatch((ML, y_cursor), strip_w_mm, strip_h,
                                       boxstyle="square,pad=0", linewidth=0.5,
                                       edgecolor='#e09696', facecolor='#fff5f5')
            ax.add_patch(rect)
            for gy in np.arange(y_cursor, y_cursor+strip_h, 5):
                ax.plot([ML, ML+strip_w_mm], [gy,gy], color='#f5d8d8', linewidth=0.2, zorder=1)
            for gx in np.arange(ML, ML+strip_w_mm, 5):
                ax.plot([gx,gx], [y_cursor, y_cursor+strip_h], color='#f5d8d8', linewidth=0.2, zorder=1)
            baseline = np.median(segment)
            seg_mm   = (segment - baseline) / ADC_PER_MM
            wx_mm    = ML + np.arange(len(segment))*MM_PER_SAMPLE
            mask     = wx_mm <= (ML+strip_w_mm)
            wx_mm    = wx_mm[mask]
            wy_mm    = y_cursor + strip_h/2 - seg_mm[:len(wx_mm)]
            ax.plot(wx_mm, wy_mm, color='black', linewidth=0.5, zorder=2)
            hl_start_mm = ML + (start_s-strip_start_s)*25.0
            hl_end_mm   = ML + (end_s  -strip_start_s)*25.0
            if hl_start_mm < ML+strip_w_mm and hl_end_mm > ML:
                hl_s = max(ML, hl_start_mm); hl_e = min(ML+strip_w_mm, hl_end_mm)
                ax.axvspan(hl_s, hl_e, color='#ff0000', alpha=0.1,
                           ymin=1-(y_cursor+strip_h)/PAGE_H, ymax=1-y_cursor/PAGE_H)
            ax.text(ML+2, y_cursor+3, f"Event {i+1}: {ann.get('type','')} (Lead {lead_name})",
                    fontsize=8, fontweight='bold', color='black', va='top', zorder=3)
            ax.text(ML+2, y_cursor+strip_h-2, f"Time: {start_s:.2f}s – {end_s:.2f}s",
                    fontsize=6, color='#555', va='bottom', zorder=3)
            y_cursor += strip_h + 10
        ax.text(PAGE_W/2, PAGE_H-MB+5,
                "Deckmount Electronics Pvt Ltd | RhythmPro ECG | Made in India",
                fontsize=6, ha='center', va='top', color='#333', zorder=9)
        return fig
