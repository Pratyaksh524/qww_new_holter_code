"""
utils/license_dialog.py
=======================
CardioX — Registration & Startup Guard Dialog (Three-Pillar Architecture v2.0).

Registration form (§3.1 of SDD):
  User fills only:  Full Name, Doctor Name, Organisation Name,
                    Organisation Address, Phone Number, Password + Confirm.
  Software collects automatically (not shown to user):
                    Hardware fingerprint, BIOS/machine serial, PC name,
                    Windows version, RhythmUltra USB serial.

RhythmUltra guard (§3.2):
  Sign Up button is DISABLED by default.
  A background QTimer scans USB every 2 seconds.
  If RhythmUltra detected -> button enables + green status shown.
  If not detected -> button stays disabled + orange warning shown.
  Server is never contacted if device is missing.

Startup block dialogs:
  BlockedDialog — shown when run_startup_checks() fails.
  Displays the failed check number and reason clearly.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QDialog, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QMessageBox, QPushButton, QScrollArea,
    QSizePolicy, QVBoxLayout, QWidget,
)

try:
    from utils.license_manager import (
        StartupCheckResult,
        format_key,
        get_hardware_fingerprint,
        get_machine_context,
        get_RhythmUltra_serial,
        is_RhythmUltra_connected,
        load_stored_key,
        register_device,
        remember_valid_license,
        save_stored_key,
        save_token_file,
        tier_name,
        SOFTWARE_VERSION,
        RhythmUltra_VID,
        RhythmUltra_PID,
    )
except ImportError:
    from license_manager import (  # type: ignore
        StartupCheckResult,
        format_key,
        get_hardware_fingerprint,
        get_machine_context,
        get_RhythmUltra_serial,
        is_RhythmUltra_connected,
        load_stored_key,
        register_device,
        remember_valid_license,
        save_stored_key,
        save_token_file,
        tier_name,
        SOFTWARE_VERSION,
        RhythmUltra_VID,
        RhythmUltra_PID,
    )

# ── Shared styles ─────────────────────────────────────────────────────────────

_HEADER_BG = "background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #1a1a2e,stop:1 #16213e);"
_ORANGE_BTN = """
    QPushButton {
        background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #ff7a12,stop:1 #ff950f);
        color: white; border: none; border-radius: 8px;
        font-size: 14px; font-weight: bold; padding: 0 28px;
    }
    QPushButton:hover { background: #e86f00; }
    QPushButton:disabled { background: #ffb87a; color: rgba(255,255,255,0.6); }
"""
_GHOST_BTN = """
    QPushButton {
        background: #eee; color: #555; border: 1px solid #ccc;
        border-radius: 8px; font-size: 14px; padding: 0 20px;
    }
    QPushButton:hover { background: #e0e0e0; }
"""
_INPUT_STYLE = """
    QLineEdit {
        border: 2px solid #ccd; border-radius: 8px;
        padding: 9px 14px; background: white; color: #1a1a2e;
        font-size: 14px;
    }
    QLineEdit:focus { border: 2px solid #ff8c00; background: #fffbf5; }
"""


def _make_header(title: str, version: str = SOFTWARE_VERSION) -> QWidget:
    header = QWidget()
    header.setFixedHeight(64)
    header.setStyleSheet(_HEADER_BG)
    lay = QHBoxLayout(header)
    lay.setContentsMargins(28, 0, 28, 0)
    lbl = QLabel(title)
    lbl.setFont(QFont("Arial", 18, QFont.Bold))
    lbl.setStyleSheet("color: #ff8c00; background: transparent;")
    ver = QLabel(f"v{version}")
    ver.setStyleSheet("color: rgba(255,255,255,0.45); font-size:11px; background:transparent;")
    ver.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
    lay.addWidget(lbl)
    lay.addStretch()
    lay.addWidget(ver)
    return header


# ══════════════════════════════════════════════════════════════════════════════
# Background Device Scanning Worker
# ══════════════════════════════════════════════════════════════════════════════

class DeviceScanWorker(QThread):
    """Worker thread for non-blocking serial port scanning in license dialogs"""
    scan_finished = pyqtSignal(bool, str, str, str) # success, port, version, serial

    def __init__(self):
        super().__init__()

    def run(self):
        try:
            import serial
            import serial.tools.list_ports
            from ecg.serial.hardware_commands import HardwareCommandHandler

            ports = list(serial.tools.list_ports.comports())
            if sys.platform == "darwin":
                ports = [p for p in ports if ("usbserial" in p.device) or ("usbmodem" in p.device)]
            else:
                # Avoid probing non-USB / legacy ports that frequently hang or always fail.
                filtered = []
                for p in ports:
                    desc = str(getattr(p, "description", "") or "")
                    dev = str(getattr(p, "device", "") or "")
                    if dev.upper() == "COM1" and "Communications Port" in desc:
                        continue
                    if "Bluetooth" in desc:
                        continue
                    filtered.append(p)
                ports = filtered

            if not ports:
                self.scan_finished.emit(False, "", "", "")
                return

            # Sort ports to prioritize last saved port if possible
            try:
                from utils.settings_manager import SettingsManager
                sm = SettingsManager()
                saved_port = sm.get_setting("serial_port")
                if saved_port:
                    ports.sort(key=lambda p: 0 if p.device == saved_port else 1)
            except Exception:
                pass

            for port in ports:
                try:
                    ser = serial.Serial(
                        port.device,
                        115200,
                        timeout=0.2,
                        write_timeout=0.2,
                    )
                    try:
                        handler = HardwareCommandHandler(ser)

                        # Query MACHINE SERIAL directly first
                        success_s, serial_num, _ = handler.send_machine_serial_command(
                            timeout=0.4, quiet=True
                        )

                        if success_s and serial_num:
                            # Optionally fetch version
                            success_v, version, _ = handler.send_version_command(
                                timeout=0.4, quiet=True
                            )
                            version_str = version if (success_v and version) else ""

                            self.scan_finished.emit(True, port.device, version_str, serial_num)
                            return
                    finally:
                        try:
                            ser.close()
                        except Exception:
                            pass
                except Exception:
                    continue

            self.scan_finished.emit(False, "", "", "")
        except Exception as e:
            print(f"[DeviceScanWorker] Error: {e}")
            self.scan_finished.emit(False, "", "", "")


# ══════════════════════════════════════════════════════════════════════════════
# Background Registration Worker
# ══════════════════════════════════════════════════════════════════════════════

class _RegisterWorker(QThread):
    """Runs register_device() off the UI thread."""
    result = pyqtSignal(dict)

    def __init__(
        self,
        license_key: str,
        full_name: str,
        doctor_name: str,
        org_name: str,
        org_address: str,
        phone: str,
        password: str,
    ):
        super().__init__()
        self._license_key = license_key
        self._full_name = full_name
        self._doctor_name = doctor_name
        self._org_name = org_name
        self._org_address = org_address
        self._phone = phone
        self._password = password

    def run(self):
        try:
            import hashlib
            pw_hash = hashlib.sha256(self._password.encode("utf-8")).hexdigest()
            res = register_device(
                license_key=self._license_key,
                full_name=self._full_name,
                doctor_name=self._doctor_name,
                org_name=self._org_name,
                org_address=self._org_address,
                phone=self._phone,
                password_hash=pw_hash,
            )
        except Exception as e:
            res = {"valid": False, "message": str(e)}
        self.result.emit(res)


# ── Legacy validation worker (kept for old key-based dialog) ──────────────────

class _ValidateWorker(QThread):
    result = pyqtSignal(dict)

    def __init__(self, license_key: str):
        super().__init__()
        self._key = license_key

    def run(self):
        try:
            from utils.license_manager import activate_with_server, check_license, get_hardware_fingerprint
            fingerprint = get_hardware_fingerprint()
            res = activate_with_server(self._key, fingerprint)
            if res.get("valid"):
                res.setdefault("source", "server")
            elif res.get("offline"):
                res = check_license(self._key)
        except Exception as e:
            res = {"valid": False, "message": str(e)}
        self.result.emit(res)


# ══════════════════════════════════════════════════════════════════════════════
# Registration Dialog (New Architecture)
# ══════════════════════════════════════════════════════════════════════════════

class RegistrationDialog(QDialog):
    """
    Full registration form per SDD §3.1.
    Only user-visible fields are shown; all hardware data is collected silently.
    Sign Up button is disabled until RhythmUltra is detected.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CardioX — Device Registration")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(600, 560)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._worker: Optional[_RegisterWorker] = None
        self._assigned_key: str = ""
        self._registration_result: dict = {}

        self._build_ui()
        self._fit_to_screen()

        # Populate machine serial ID immediately (WMI, non-blocking enough at startup)
        self._populate_machine_serial()

        self._device_scan_in_progress = False
        self._had_device_connected = False
        self._last_connected_serial = ""

        # USB scan timer — every 2 seconds (SDD §3.2)
        self._usb_timer = QTimer(self)
        self._usb_timer.timeout.connect(self._scan_RhythmUltra)
        self._usb_timer.start(2000)
        # Initial scan immediately
        self._scan_RhythmUltra()

    def _fit_to_screen(self):
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.resize(660, 600)
                return
            geom = screen.availableGeometry()
            w = min(max(int(geom.width() * 0.62), self.minimumWidth()), 820)
            h = min(max(int(geom.height() * 0.82), self.minimumHeight()), 760)
            self.resize(w, h)
            self.move(
                geom.left() + (geom.width() - w) // 2,
                geom.top() + (geom.height() - h) // 2,
            )
        except Exception:
            self.resize(660, 600)

    # ── UI Construction ───────────────────────────────────────────────────────

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(_make_header("CardioX  ·  Device Registration"))

        # Scrollable card
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        card = QWidget()
        card.setStyleSheet("background: #f5f6fa;")
        c = QVBoxLayout(card)
        c.setContentsMargins(44, 28, 44, 24)
        c.setSpacing(14)

        # ── Title ─────────────────────────────────────────────────────────────
        title = QLabel("Register This Device")
        title.setFont(QFont("Arial", 17, QFont.Bold))
        title.setStyleSheet("color: #1a1a2e;")
        c.addWidget(title)

        sub = QLabel(
            "Enter your details below.  Hardware information is collected automatically — "
            "no need to type serial numbers."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #666; font-size: 13px;")
        c.addWidget(sub)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("background:#dde; max-height:1px; border:none;")
        c.addWidget(div)

        # ── License Key ───────────────────────────────────────────────────────
        c.addWidget(self._field_label("License Key"))
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("CRDX-XXXX-XXXX")
        self._key_input.setMinimumHeight(46)
        self._key_input.setFont(QFont("Courier New", 13, QFont.Bold))
        self._key_input.setMaxLength(14)
        self._key_input.setAlignment(Qt.AlignCenter)
        self._key_input.setStyleSheet(_INPUT_STYLE + "letter-spacing: 2px;")
        self._key_input.textChanged.connect(self._on_key_typed)
        stored = load_stored_key()
        if stored:
            self._key_input.setText(format_key(stored))
        c.addWidget(self._key_input)

        # ── User fields ───────────────────────────────────────────────────────
        c.addWidget(self._field_label("Full Name"))
        self._full_name = self._make_input("Your full name")
        c.addWidget(self._full_name)

        c.addWidget(self._field_label("Doctor Name"))
        self._doctor = self._make_input("Treating / responsible doctor")
        c.addWidget(self._doctor)

        c.addWidget(self._field_label("Organisation Name"))
        self._org_name = self._make_input("Hospital or clinic name")
        c.addWidget(self._org_name)

        c.addWidget(self._field_label("Organisation Address"))
        self._org_address = self._make_input("Full address")
        c.addWidget(self._org_address)

        c.addWidget(self._field_label("Phone Number"))
        self._phone = self._make_input("10-digit mobile number")
        c.addWidget(self._phone)

        c.addWidget(self._field_label("Password"))
        self._password = self._make_input("Create a secure password", password=True)
        c.addWidget(self._password)

        c.addWidget(self._field_label("Confirm Password"))
        self._confirm = self._make_input("Re-enter your password", password=True)
        c.addWidget(self._confirm)

        # ── Hardware Info Section ─────────────────────────────────────────────
        hw_div = QFrame()
        hw_div.setFrameShape(QFrame.HLine)
        hw_div.setStyleSheet("background:#dde; max-height:1px; border:none; margin-top:4px;")
        c.addWidget(hw_div)

        hw_title = QLabel("Hardware Info  (auto-detected)")
        hw_title.setStyleSheet("color: #888; font-size: 11px; font-weight: bold; letter-spacing: 0.5px;")
        c.addWidget(hw_title)

        # Machine Serial ID (WMI BIOS serial)
        machine_row = QHBoxLayout()
        machine_row.setSpacing(8)
        machine_lbl = QLabel("Machine Serial ID:")
        machine_lbl.setStyleSheet("color: #555; font-size: 12px; font-weight: bold; min-width: 130px;")
        self._machine_serial_field = QLineEdit()
        self._machine_serial_field.setReadOnly(True)
        self._machine_serial_field.setPlaceholderText("Detecting...")
        self._machine_serial_field.setMinimumHeight(34)
        self._machine_serial_field.setStyleSheet(
            "QLineEdit { border: 1px solid #c8e6c9; border-radius: 6px; padding: 5px 10px; "
            "background: #f1f8f1; color: #2e7d32; font-size: 12px; font-weight: bold; }"
        )
        machine_row.addWidget(machine_lbl)
        machine_row.addWidget(self._machine_serial_field, 1)
        c.addLayout(machine_row)

        # ── RhythmUltra Status ─────────────────────────────────────────────────
        self._device_status = QLabel()
        self._device_status.setAlignment(Qt.AlignCenter)
        self._device_status.setMinimumHeight(36)
        self._device_status.setWordWrap(True)
        self._device_status.setStyleSheet("font-size: 12px; border-radius: 6px; padding: 6px 12px;")
        self._set_device_status_disconnected()
        c.addWidget(self._device_status)

        # ── Status / feedback ─────────────────────────────────────────────────
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(28)
        self._status.setStyleSheet("font-size: 12px; color: #888;")
        c.addWidget(self._status)

        c.addStretch()

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setFixedHeight(44)
        self._cancel_btn.setStyleSheet(_GHOST_BTN)
        self._cancel_btn.clicked.connect(self.reject)

        self._register_btn = QPushButton("Register Device")
        self._register_btn.setFixedHeight(44)
        self._register_btn.setDefault(True)
        self._register_btn.setEnabled(False)  # Disabled until RhythmUltra detected
        self._register_btn.setStyleSheet(_ORANGE_BTN)
        self._register_btn.clicked.connect(self._on_register)

        btn_row.addWidget(self._cancel_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._register_btn)
        c.addLayout(btn_row)

        # ── Help ──────────────────────────────────────────────────────────────
        help_lbl = QLabel(
            "Need a license key?  Contact "
            "<a href='mailto:cardiocare@deckmount.in' style='color:#ff8c00;'>"
            "cardiocare@deckmount.in</a>"
        )
        help_lbl.setOpenExternalLinks(True)
        help_lbl.setAlignment(Qt.AlignCenter)
        help_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        c.addWidget(help_lbl)

        scroll.setWidget(card)
        root.addWidget(scroll, 1)

    @staticmethod
    def _field_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #333; font-weight: bold; font-size: 13px;")
        return lbl

    @staticmethod
    def _make_input(placeholder: str, *, password: bool = False) -> QLineEdit:
        inp = QLineEdit()
        inp.setPlaceholderText(placeholder)
        inp.setMinimumHeight(42)
        inp.setStyleSheet(_INPUT_STYLE)
        if password:
            inp.setEchoMode(QLineEdit.Password)
        return inp

    # ── Machine Serial ID (WMI) ───────────────────────────────────────────────

    def _populate_machine_serial(self):
        """Read BIOS machine serial via WMI and show it in the field."""
        try:
            ctx = get_machine_context()
            serial = ctx.get("machine_serial_id", "").strip()
            if serial:
                self._machine_serial_field.setText(serial)
                self._machine_serial_field.setStyleSheet(
                    "QLineEdit { border: 1px solid #a5d6a7; border-radius: 6px; padding: 5px 10px; "
                    "background: #e8f5e9; color: #2e7d32; font-size: 12px; font-weight: bold; }"
                )
            else:
                self._machine_serial_field.setText("Not available")
                self._machine_serial_field.setStyleSheet(
                    "QLineEdit { border: 1px solid #ccc; border-radius: 6px; padding: 5px 10px; "
                    "background: #f5f5f5; color: #999; font-size: 12px; }"
                )
        except Exception:
            self._machine_serial_field.setText("Detection failed")

    # ── RhythmUltra USB polling ────────────────────────────────────────────────

    def _set_device_status_connected(self, serial: str):
        self._device_status.setText(
            f"[OK]  RhythmUltra connected  |  Serial: {serial}"
        )
        self._device_status.setStyleSheet(
            "font-size: 12px; border-radius: 6px; padding: 6px 12px; "
            "background: #e8f5e9; color: #2e7d32; border: 1px solid #a5d6a7;"
        )

    def _set_device_status_disconnected(self):
        vid_str = f"0x{RhythmUltra_VID:04X}" if RhythmUltra_VID else "?"
        pid_str = f"0x{RhythmUltra_PID:04X}" if RhythmUltra_PID else "?"
        if RhythmUltra_VID == 0 and RhythmUltra_PID == 0:
            msg = "[WARN]  RhythmUltra VID/PID not configured.  Set RhythmUltra_VID and RhythmUltra_PID in .env"
        else:
            if getattr(self, "_had_device_connected", False):
                msg = (
                    "RhythmUltra disconnected"
                )
            else:
                msg = (
                    f"🟠  RhythmUltra not detected  (VID={vid_str} PID={pid_str})  —  "
                    "please connect the RhythmUltra device to enable Sign Up"
                )
        self._device_status.setText(msg)
        if getattr(self, "_had_device_connected", False):
            self._device_status.setStyleSheet(
                "font-size: 12px; border-radius: 6px; padding: 6px 12px; "
                "background: #ffebee; color: #c62828; border: 1px solid #ffcdd2;"
            )
        else:
            self._device_status.setStyleSheet(
                "font-size: 12px; border-radius: 6px; padding: 6px 12px; "
                "background: #fff3e0; color: #e65100; border: 1px solid #ffcc80;"
            )

    def _check_device_connection(self):
        """Monitor USB connection to auto-populate machine serial ID non-blockingly"""
        if self._device_scan_in_progress:
            return

        try:
            self._device_scan_in_progress = True
            self._scan_worker = DeviceScanWorker()
            self._scan_worker.scan_finished.connect(self._on_scan_finished)
            self._scan_worker.start()
        except Exception as e:
            self._device_scan_in_progress = False
            print(f"[RegistrationDialog] Error checking connection: {e}")

    def _on_scan_finished(self, success, port, version, serial_num):
        """Update RegistrationDialog machine serial field when device is detected"""
        self._device_scan_in_progress = False
        
        if success and serial_num:
            # Cache the serial in the license manager so it passes checks
            from utils.license_manager import set_detected_device_serial
            set_detected_device_serial(serial_num)

            self._had_device_connected = True
            self._last_connected_serial = serial_num

            self._machine_serial_field.setText(serial_num)
            self._machine_serial_field.setStyleSheet(
                "QLineEdit { border: 1px solid #a5d6a7; border-radius: 6px; padding: 5px 10px; "
                "background: #e8f5e9; color: #2e7d32; font-size: 12px; font-weight: bold; }"
            )
            self._set_device_status_connected(serial_num)
            
            # Enable Register button
            if not self._register_btn.isEnabled() and not self._is_submitting():
                self._register_btn.setEnabled(True)
        else:
            # Not found - try to fall back to WMI serial
            from utils.license_manager import set_detected_device_serial, get_machine_context
            set_detected_device_serial(None)
            
            if getattr(self, "_had_device_connected", False):
                self._machine_serial_field.setText(f"{self._last_connected_serial} (Disconnected)")
                self._machine_serial_field.setStyleSheet(
                    "QLineEdit { border: 1px solid #ffcdd2; border-radius: 6px; padding: 5px 10px; "
                    "background: #ffebee; color: #c62828; font-size: 12px; font-weight: bold; }"
                )
            else:
                ctx = get_machine_context()
                wmi_serial = ctx.get("machine_serial_id", "").strip()
                if wmi_serial:
                    self._machine_serial_field.setText(wmi_serial)
                    self._machine_serial_field.setStyleSheet(
                        "QLineEdit { border: 1px solid #a5d6a7; border-radius: 6px; padding: 5px 10px; "
                        "background: #e8f5e9; color: #2e7d32; font-size: 12px; font-weight: bold; }"
                    )
                else:
                    self._machine_serial_field.setText("Not available")
                    self._machine_serial_field.setStyleSheet(
                        "QLineEdit { border: 1px solid #ccc; border-radius: 6px; padding: 5px 10px; "
                        "background: #f5f5f5; color: #999; font-size: 12px; }"
                    )
                
            self._set_device_status_disconnected()
            if not self._is_submitting():
                self._register_btn.setEnabled(False)

    def _scan_RhythmUltra(self):
        """Called every 2 seconds by QTimer. Updates button state without page reload."""
        try:
            # 1. Fast path: if VID/PID is configured and device is connected
            serial = get_RhythmUltra_serial()
            if serial:
                from utils.license_manager import set_detected_device_serial
                set_detected_device_serial(serial)
                
                self._had_device_connected = True
                self._last_connected_serial = serial
                
                self._machine_serial_field.setText(serial)
                self._machine_serial_field.setStyleSheet(
                    "QLineEdit { border: 1px solid #a5d6a7; border-radius: 6px; padding: 5px 10px; "
                    "background: #e8f5e9; color: #2e7d32; font-size: 12px; font-weight: bold; }"
                )
                self._set_device_status_connected(serial)
                if not self._register_btn.isEnabled() and not self._is_submitting():
                    self._register_btn.setEnabled(True)
            else:
                # 2. Slow path: run non-blocking active scan of COM ports (same logic as signup page)
                self._check_device_connection()
        except Exception:
            pass

    def _is_submitting(self) -> bool:
        return self._worker is not None and self._worker.isRunning()

    # ── Key formatting ────────────────────────────────────────────────────────

    def _on_key_typed(self, text: str):
        clean = "".join(c for c in text.upper() if c.isalnum())
        if len(clean) > 0 and not clean.startswith("CRDX"):
            clean = "CRDX" + clean
        clean = clean[:12]
        parts = []
        if len(clean) > 0:
            parts.append(clean[0:4])
        if len(clean) > 4:
            parts.append(clean[4:8])
        if len(clean) > 8:
            parts.append(clean[8:12])
        formatted = "-".join(parts)
        self._key_input.blockSignals(True)
        self._key_input.setText(formatted)
        self._key_input.setCursorPosition(len(formatted))
        self._key_input.blockSignals(False)

    # ── Registration submission ───────────────────────────────────────────────

    def _set_status(self, msg: str, color: str = "#888"):
        self._status.setText(msg)
        self._status.setStyleSheet(f"font-size: 12px; color: {color};")

    def _on_register(self):
        # Client-side guard: RhythmUltra must be connected (SDD §3.2)
        if not is_RhythmUltra_connected():
            self._set_status(
                "RhythmUltra device not detected.  Please connect it before registering.",
                "#e74c3c",
            )
            return

        key_text = self._key_input.text().strip().upper()
        clean_key = key_text.replace("-", "")
        if len(clean_key) != 12 or not clean_key.startswith("CRDX"):
            self._set_status("Please enter a complete 12-character license key (CRDX-XXXX-XXXX).", "#e67e22")
            return

        full_name = self._full_name.text().strip()
        doctor = self._doctor.text().strip()
        org_name = self._org_name.text().strip()
        org_address = self._org_address.text().strip()
        phone = self._phone.text().strip()
        password = self._password.text()
        confirm = self._confirm.text()

        if not all([full_name, doctor, org_name, org_address, phone, password, confirm]):
            self._set_status("All fields are required.", "#e74c3c")
            return
        if not phone.isdigit() or len(phone) != 10:
            self._set_status("Phone number must be exactly 10 digits.", "#e74c3c")
            return
        if password != confirm:
            self._set_status("Passwords do not match.", "#e74c3c")
            return
        if len(password) < 6:
            self._set_status("Password must be at least 6 characters.", "#e74c3c")
            return

        self._set_status("Contacting license server…", "#2980b9")
        self._register_btn.setEnabled(False)
        self._cancel_btn.setEnabled(False)
        self._usb_timer.stop()

        self._worker = _RegisterWorker(
            license_key=key_text,
            full_name=full_name,
            doctor_name=doctor,
            org_name=org_name,
            org_address=org_address,
            phone=phone,
            password=password,
        )
        self._worker.result.connect(self._on_registration_result)
        self._worker.start()

    def _on_registration_result(self, result: dict):
        self._cancel_btn.setEnabled(True)
        self._usb_timer.start(2000)

        success = result.get("valid") or result.get("success") or result.get("authorized")
        if success:
            self._registration_result = result
            key = result.get("license_key", self._key_input.text().strip())
            seat = result.get("seat_number", 1)
            tier = result.get("tier", 0)

            # Save token to cardiox.lic if server returned one
            token_str = result.get("token")
            if token_str:
                try:
                    import json as _json
                    if isinstance(token_str, dict):
                        save_token_file(token_str.get("payload", token_str))
                    elif token_str.count(".") == 2:
                        save_token_file(token_str)
                    else:
                        token_envelope = _json.loads(token_str)
                        save_token_file(token_envelope.get("payload", token_envelope))
                except Exception as te:
                    print(f"[License] Could not save token: {te}")
                    remember_valid_license(key, get_hardware_fingerprint(), result)
            else:
                remember_valid_license(key, get_hardware_fingerprint(), result)

            save_stored_key(key)
            self._assigned_key = key

            # Show key to user — they must note it down (SDD §3.4)
            self._show_key_dialog(key, seat, tier_name(tier))
        else:
            msg = result.get("message") or result.get("error") or "Registration failed."
            err = str(result.get("error", "")).strip().upper()
            # Match on the machine-readable code AND on the server's prose. The
            # deployed Lambda reports an exhausted licence as the sentence
            # "No available seats for this license." rather than a code, so keying
            # only off DEVICE_ALREADY_REGISTERED left the useful dialog below
            # unreachable and the user got a one-line red label instead.
            haystack = f"{err} {str(msg).strip().upper()}"

            def _fail(title: str, body: str) -> None:
                self._set_status(body, "#c0392b")
                QMessageBox.critical(self, title, body)
                self._register_btn.setEnabled(True)

            if err in ("LICENSE_REVOKED", "ACCOUNT_REVOKED") or "REVOKED" in haystack:
                # Revocation blocks re-registration; only Deckmount can free the seat.
                from utils.license_manager import REVOKED_MESSAGE
                _fail("License Revoked", REVOKED_MESSAGE)
                return

            if (
                err in ("DEVICE_ALREADY_REGISTERED", "NO_SEATS", "DEVICE_LIMIT_REACHED")
                or "NO AVAILABLE SEATS" in haystack
                or "SEAT(S) FOR THIS LICENSE ARE IN USE" in haystack
                or "ALL SEATS" in haystack
            ):
                _fail(
                    "Device Limit Reached",
                    "Device limit reached — all machines for this licence are in use.\n\n"
                    "Please contact Deckmount support to free up a slot.",
                )
                return

            if result.get("offline") or "INTERNAL SERVER ERROR" in haystack:
                _fail(
                    "Registration Unavailable",
                    "Could not reach the Deckmount licence server.\n\n"
                    "Check this computer's internet connection and try again. "
                    "If the problem continues, contact Deckmount support.",
                )
                return

            self._set_status(f"Error: {msg}", "#e74c3c")
            self._register_btn.setEnabled(True)

    def _show_key_dialog(self, key: str, seat: int, tier: str):
        """Show the assigned license key once — user must note it down."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Registration Successful")
        dlg.setMinimumWidth(500)
        dlg.setStyleSheet("background: #f5f6fa;")
        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(32, 28, 32, 24)
        lay.setSpacing(16)

        icon = QLabel("[OK]")
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 40px;")
        lay.addWidget(icon)

        title = QLabel("Device Registered Successfully!")
        title.setFont(QFont("Arial", 15, QFont.Bold))
        title.setStyleSheet("color: #2e7d32;")
        title.setAlignment(Qt.AlignCenter)
        lay.addWidget(title)

        info = QLabel(
            f"Plan: <b>{tier}</b>  ·  Seat: <b>#{seat}</b><br>"
            "Your license key is shown below.  <b>Please note it down</b> — "
            "it will not be shown again."
        )
        info.setWordWrap(True)
        info.setAlignment(Qt.AlignCenter)
        info.setStyleSheet("color: #555; font-size: 13px;")
        lay.addWidget(info)

        key_box = QLineEdit(format_key(key))
        key_box.setReadOnly(True)
        key_box.setAlignment(Qt.AlignCenter)
        key_box.setFont(QFont("Courier New", 14, QFont.Bold))
        key_box.setStyleSheet(
            "border: 2px solid #ff8c00; border-radius: 8px; padding: 10px; "
            "background: white; color: #1a1a2e; letter-spacing: 2px;"
        )
        lay.addWidget(key_box)

        copy_btn = QPushButton("[PARSED]  Copy Key")
        copy_btn.setStyleSheet(_ORANGE_BTN)
        copy_btn.setFixedHeight(40)
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(format_key(key)))
        lay.addWidget(copy_btn)

        ok_btn = QPushButton("Continue to Sign In")
        ok_btn.setFixedHeight(44)
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(_ORANGE_BTN)
        ok_btn.clicked.connect(dlg.accept)
        lay.addWidget(ok_btn)

        dlg.exec_()
        self.accept()

    def get_registration_result(self) -> dict:
        return self._registration_result

    def get_assigned_key(self) -> str:
        return self._assigned_key

    def get_registration_details(self) -> dict:
        return {
            "full_name": self._full_name.text().strip(),
            "doctor": self._doctor.text().strip(),
            "org_name": self._org_name.text().strip(),
            "org_address": self._org_address.text().strip(),
            "phone": self._phone.text().strip(),
            "password": self._password.text(),
            "serial_id": self._machine_serial_field.text().strip(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# Startup Block Dialog (shown when run_startup_checks() fails)
# ══════════════════════════════════════════════════════════════════════════════

class StartupBlockDialog(QDialog):
    """
    Shown when a startup check fails.  Non-dismissible for hard blocks.
    """

    _STEP_LABELS = {
        1: "Check 1 — License File",
        2: "Check 2 — License Integrity",
        3: "Check 3 — Machine Identity",
        4: "Check 4 — RhythmUltra Device",
        5: "Check 5 — Server Verification",
    }

    _STEP_ICONS = {
        1: "[PARSED]",
        2: "[CLOSE]",
        3: "💻",
        4: "🔌",
        5: "🌐",
    }

    def __init__(self, check_result: "StartupCheckResult", parent=None):
        super().__init__(parent)
        self.setWindowTitle("CardioX — Access Blocked")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint)
        self.setMinimumSize(500, 320)
        self._result = check_result
        self._build_ui()
        self._fit_to_screen()

        # For Check 4 (RhythmUltra), poll USB every 2s and auto-close when detected
        if check_result.step_failed == 4:
            self._device_scan_in_progress = False
            self._poll_timer = QTimer(self)
            self._poll_timer.timeout.connect(self._check_device)
            self._poll_timer.start(2000)

    def _fit_to_screen(self):
        try:
            screen = QApplication.primaryScreen()
            if screen:
                geom = screen.availableGeometry()
                self.move(
                    geom.left() + (geom.width() - self.width()) // 2,
                    geom.top() + (geom.height() - self.height()) // 2,
                )
        except Exception:
            pass

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(_make_header("CardioX"))

        card = QWidget()
        card.setStyleSheet("background: #f5f6fa;")
        c = QVBoxLayout(card)
        c.setContentsMargins(40, 28, 40, 24)
        c.setSpacing(14)

        step = self._result.step_failed
        icon_text = self._STEP_ICONS.get(step, "[WARN]")
        step_label = self._STEP_LABELS.get(step, f"Check {step}")

        # Icon
        icon = QLabel(icon_text)
        icon.setAlignment(Qt.AlignCenter)
        icon.setStyleSheet("font-size: 48px;")
        c.addWidget(icon)

        # Failed check label
        check_lbl = QLabel(f"BLOCKED AT  ·  {step_label}")
        check_lbl.setAlignment(Qt.AlignCenter)
        check_lbl.setStyleSheet(
            "color: #c0392b; font-size: 11px; font-weight: bold; letter-spacing: 1px;"
        )
        c.addWidget(check_lbl)

        # Reason message
        self._reason_lbl = QLabel(self._result.reason)
        self._reason_lbl.setWordWrap(True)
        self._reason_lbl.setAlignment(Qt.AlignCenter)
        self._reason_lbl.setStyleSheet(
            "color: #333; font-size: 14px; background: white; border-radius: 8px; "
            "padding: 14px 18px; border: 1px solid #e0e0e0;"
        )
        c.addWidget(self._reason_lbl)

        c.addStretch()

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)

        quit_btn = QPushButton("Quit")
        quit_btn.setFixedHeight(40)
        quit_btn.setStyleSheet(_GHOST_BTN)
        quit_btn.clicked.connect(self.reject)
        btn_row.addWidget(quit_btn)

        # Check 1 & 2 -> show Register button
        if step in (1, 2):
            reg_btn = QPushButton("Register This Device")
            reg_btn.setFixedHeight(40)
            reg_btn.setDefault(True)
            reg_btn.setStyleSheet(_ORANGE_BTN)
            reg_btn.clicked.connect(self._open_registration)
            btn_row.addStretch()
            btn_row.addWidget(reg_btn)
        elif step == 4:
            # RhythmUltra block — show retry button
            self._retry_btn = QPushButton("⟳  Retry Device Scan")
            self._retry_btn.setFixedHeight(40)
            self._retry_btn.setStyleSheet(_ORANGE_BTN)
            self._retry_btn.clicked.connect(self._check_device)
            btn_row.addStretch()
            btn_row.addWidget(self._retry_btn)
        else:
            # Checks 3 & 5 — hard block; user must contact support
            support_lbl = QLabel(
                "Contact <a href='mailto:cardiocare@deckmount.in' style='color:#ff8c00;'>"
                "cardiocare@deckmount.in</a> for assistance."
            )
            support_lbl.setOpenExternalLinks(True)
            support_lbl.setAlignment(Qt.AlignCenter)
            support_lbl.setStyleSheet("color: #888; font-size: 12px;")
            c.addWidget(support_lbl)

        c.addLayout(btn_row)
        root.addWidget(card, 1)

    def _open_registration(self):
        dlg = RegistrationDialog(parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self.accept()

    def _check_device(self):
        """Check if RhythmUltra has been plugged in — auto-close if detected."""
        try:
            # 1. Fast path: if VID/PID is configured and device is connected
            serial = get_RhythmUltra_serial()
            if serial:
                from utils.license_manager import set_detected_device_serial
                set_detected_device_serial(serial)
                if hasattr(self, "_poll_timer"):
                    self._poll_timer.stop()
                self.accept()
                return

            # 2. Slow path: run non-blocking active scan of COM ports if not already scanning
            if not getattr(self, "_device_scan_in_progress", False):
                import serial.tools.list_ports
                current_ports = [p.device for p in serial.tools.list_ports.comports()]
                
                if not hasattr(self, '_last_ports') or current_ports != self._last_ports:
                    self._last_ports = current_ports
                    self._device_scan_in_progress = True
                    self._scan_worker = DeviceScanWorker()
                    self._scan_worker.scan_finished.connect(self._on_scan_finished)
                    self._scan_worker.start()
        except Exception:
            pass

    def _on_scan_finished(self, success, port, version, serial_num):
        """Called when background device scan finishes in the block dialog"""
        self._device_scan_in_progress = False
        if success and serial_num:
            from utils.license_manager import set_detected_device_serial
            set_detected_device_serial(serial_num)
            if hasattr(self, "_poll_timer"):
                self._poll_timer.stop()
            self.accept()


# ══════════════════════════════════════════════════════════════════════════════
# Legacy LicenseDialog (kept for backward compat with old activation flow)
# ══════════════════════════════════════════════════════════════════════════════

class LicenseDialog(QDialog):
    """
    Legacy activation dialog — shown when a stored license key exists but
    needs re-validation.  Used by the old /activate code path.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("CardioX — License Activation")
        self.setWindowFlags(Qt.Window | Qt.WindowMinimizeButtonHint | Qt.WindowCloseButtonHint)
        self.setMinimumSize(520, 380)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._worker: Optional[_ValidateWorker] = None
        self._license_result: dict = {}
        self._build_ui()
        self._fit_to_screen()
        stored = load_stored_key()
        if stored:
            self._key_input.setText(format_key(stored))

    def _fit_to_screen(self):
        try:
            screen = QApplication.primaryScreen()
            if screen is None:
                self.resize(640, 480)
                return
            geom = screen.availableGeometry()
            w = min(max(int(geom.width() * 0.65), self.minimumWidth()), 860)
            h = min(max(int(geom.height() * 0.68), self.minimumHeight()), 680)
            self.resize(w, h)
            self.move(geom.left() + (geom.width() - w) // 2, geom.top() + (geom.height() - h) // 2)
        except Exception:
            self.resize(640, 480)

    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(_make_header("CardioX"))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        card = QWidget()
        card.setStyleSheet("background: #f5f6fa;")
        c = QVBoxLayout(card)
        c.setContentsMargins(40, 28, 40, 24)
        c.setSpacing(16)

        title = QLabel("Software License Activation")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #1a1a2e;")
        c.addWidget(title)

        sub = QLabel(
            "Enter the license key provided by Deckmount to activate this software.\n"
            "An internet connection is required for initial activation."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #666; font-size: 13px;")
        c.addWidget(sub)

        div = QFrame()
        div.setFrameShape(QFrame.HLine)
        div.setStyleSheet("background:#dde; max-height:1px; border:none;")
        c.addWidget(div)

        c.addWidget(QLabel("License Key").setStyleSheet("color:#333; font-weight:bold; font-size:13px;") or QLabel("License Key"))
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("CRDX-XXXX-XXXX")
        self._key_input.setMinimumHeight(48)
        self._key_input.setFont(QFont("Courier New", 14, QFont.Bold))
        self._key_input.setMaxLength(14)
        self._key_input.setAlignment(Qt.AlignCenter)
        self._key_input.setStyleSheet(_INPUT_STYLE + "letter-spacing: 2px;")
        self._key_input.textChanged.connect(self._on_key_typed)
        self._key_input.returnPressed.connect(self._on_activate)
        c.addWidget(self._key_input)

        import platform
        fp = get_hardware_fingerprint()
        sys_info = QLabel(
            f"Machine ID:\n{fp}\nOS: {platform.system()} {platform.release()}  |  Host: {platform.node()}"
        )
        sys_info.setWordWrap(True)
        sys_info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        sys_info.setStyleSheet("color: #999; font-size: 10px; line-height: 1.2;")
        sys_info.setAlignment(Qt.AlignCenter)
        c.addWidget(sys_info)

        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setWordWrap(True)
        self._status.setMinimumHeight(32)
        self._status.setStyleSheet("font-size: 12px; color: #888;")
        c.addWidget(self._status)
        c.addStretch()

        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self._quit_btn = QPushButton("Quit")
        self._quit_btn.setFixedHeight(44)
        self._quit_btn.setStyleSheet(_GHOST_BTN)
        self._quit_btn.clicked.connect(self.reject)
        self._activate_btn = QPushButton("Activate")
        self._activate_btn.setFixedHeight(44)
        self._activate_btn.setDefault(True)
        self._activate_btn.setStyleSheet(_ORANGE_BTN)
        self._activate_btn.clicked.connect(self._on_activate)
        btn_row.addWidget(self._quit_btn)
        btn_row.addStretch()
        btn_row.addWidget(self._activate_btn)
        c.addLayout(btn_row)

        help_lbl = QLabel(
            "Need a license key? Contact "
            "<a href='mailto:cardiocare@deckmount.in' style='color:#ff8c00;'>cardiocare@deckmount.in</a>"
        )
        help_lbl.setOpenExternalLinks(True)
        help_lbl.setAlignment(Qt.AlignCenter)
        help_lbl.setStyleSheet("color: #aaa; font-size: 11px; margin-top: 4px;")
        c.addWidget(help_lbl)

        scroll.setWidget(card)
        root.addWidget(scroll, 1)

    def _on_key_typed(self, text: str):
        clean = "".join(c for c in text.upper() if c.isalnum())
        if len(clean) > 0 and not clean.startswith("CRDX"):
            clean = "CRDX" + clean
        clean = clean[:12]
        parts = []
        if len(clean) > 0:
            parts.append(clean[0:4])
        if len(clean) > 4:
            parts.append(clean[4:8])
        if len(clean) > 8:
            parts.append(clean[8:12])
        formatted = "-".join(parts)
        self._key_input.blockSignals(True)
        self._key_input.setText(formatted)
        self._key_input.setCursorPosition(len(formatted))
        self._key_input.blockSignals(False)
        self._activate_btn.setEnabled(len(clean) == 12)

    def _set_status(self, msg: str, color: str = "#888"):
        self._status.setText(msg)
        self._status.setStyleSheet(f"font-size: 12px; color: {color};")

    def _on_activate(self):
        key_text = self._key_input.text().strip().upper()
        clean = key_text.replace("-", "")
        if len(clean) != 12 or not clean.startswith("CRDX"):
            self._set_status("Please enter a complete 12-character license key (CRDX-XXXX-XXXX).", "#e67e22")
            return
        self._set_status("Contacting license server…", "#2980b9")
        self._activate_btn.setEnabled(False)
        self._quit_btn.setEnabled(False)
        self._worker = _ValidateWorker(key_text)
        self._worker.result.connect(self._on_validation_result)
        self._worker.start()

    def _on_validation_result(self, result: dict):
        self._activate_btn.setEnabled(True)
        self._quit_btn.setEnabled(True)
        if result.get("valid"):
            self._license_result = result
            tier = result.get("tier", 0)
            exp = result.get("expires", 0)
            exp_str = "Perpetual" if exp == 0 else __import__("datetime").datetime.fromtimestamp(exp).strftime("%Y-%m-%d")
            self._set_status(f"Activated — {tier_name(tier)} | Expires: {exp_str}", "#27ae60")
            key_text = self._key_input.text().strip()
            save_stored_key(key_text)
            remember_valid_license(key_text, get_hardware_fingerprint(), result)
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(1200, self.accept)
        else:
            msg = result.get("message", "Validation failed.")
            err = str(result.get("error", "")).strip().upper()
            if err == "DEVICE_ALREADY_REGISTERED":
                QMessageBox.critical(
                    self,
                    "Device Limit Reached",
                    "Device limit reached (5/5 machines active). Please contact Deckmount support to free up a slot.",
                )
                self.reject()
                return
            if result.get("revoked") or str(result.get("error_code", "")).strip().upper() == "LICENSE_REVOKED":
                QMessageBox.critical(self, "License Revoked", f"{msg}\n\nContact support to restore access.")
                self.reject()
                return
            self._set_status(f"Error: {msg}", "#e74c3c")

    def get_license_result(self) -> dict:
        return self._license_result

    def get_license_key(self) -> str:
        return self._key_input.text().strip()


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = RegistrationDialog()
    if dlg.exec_() == QDialog.Accepted:
        print("Registered:", dlg.get_registration_result())
    else:
        print("Cancelled.")
    sys.exit(0)
