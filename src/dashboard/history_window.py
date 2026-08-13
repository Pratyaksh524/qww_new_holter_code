"""
ECG Report History Window — redesigned with:
  • Split-pane layout (table left, PDF preview right)
  • In-app PDF preview via pymupdf (page-by-page, scroll)
  • Email-send dialog (smtplib, attachment)
  • Cloud review workflow preserved
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter,
    QTableWidget, QTableWidgetItem, QPushButton,
    QMessageBox, QSizePolicy, QApplication, QFileDialog,
    QLineEdit, QComboBox, QLabel, QDateEdit, QFrame,
    QScrollArea, QWidget, QProgressDialog, QFormLayout,
    QTextEdit, QCheckBox, QGridLayout, QListWidget,
    QListWidgetItem, QAbstractItemView, QGroupBox, QStackedWidget, QHeaderView,
)
import sys, os, json, datetime, shutil, smtplib, traceback, tempfile, re
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import requests
import webbrowser

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

try:
    from utils.cloud_uploader import get_cloud_uploader
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from utils.cloud_uploader import get_cloud_uploader

try:
    from utils.app_paths import data_file
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from utils.app_paths import data_file

from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal, QSize, QEvent
from PyQt5.QtGui import QFont, QPixmap, QColor, QImage
from PyQt5.QtWidgets import QShortcut
from PyQt5.QtGui import QKeySequence
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if load_dotenv is not None:
    for candidate in (
        os.path.join(BASE_DIR, ".env"),
        os.path.join(os.path.dirname(BASE_DIR), ".env"),
    ):
        if os.path.exists(candidate):
            load_dotenv(candidate, override=False)
            break
HISTORY_FILE = str(data_file("ecg_history.json"))
ECG_DATA_FILE = str(data_file("ecg_data.txt"))
REPORTS_DIR = str(data_file("reports"))
REPORTS_INDEX_FILE = os.path.join(REPORTS_DIR, "index.json")
BACKEND_API_URL = "https://your-backend-api.com/api/reports"
API_TIMEOUT = 30
PUBLIC_REVIEWED_REPORTS_URL = os.getenv(
    "REVIEWED_REPORTS_API_URL",
    "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/public/reviewed-reports",
).strip()


def _get_RhythmUltra_serial() -> str:
    try:
        from utils.license_manager import load_token_file, get_RhythmUltra_serial
        token = load_token_file()
        serial = (token or {}).get("rhythmultra_serial", (token or {}).get("RhythmUltra_serial", (token or {}).get("rhythmulta_serial", "")))
        if not serial:
            serial = get_RhythmUltra_serial() or ""
    except Exception:
        serial = ""
    return serial or "DM ECG V1.0 A010"

HL7_VERSION = "2.5.1"
HL7_APP_NAME = "CardioX"
HL7_REPORT_DIR = os.path.join(REPORTS_DIR, "hl7")


def _hl7_escape(value) -> str:
    """Escape values for HL7 v2 pipe-delimited fields."""
    text = "" if value is None else str(value)
    return (
        text.replace("\\", "\\E\\")
        .replace("|", "\\F\\")
        .replace("^", "\\S\\")
        .replace("&", "\\T\\")
        .replace("~", "\\R\\")
    )


def _hl7_clean_key(key: str) -> str:
    key = str(key or "").strip().upper()
    key = re.sub(r"[^A-Z0-9_]+", "_", key)
    return key[:30] or "FIELD"


def _hl7_datetime(value=None) -> str:
    if isinstance(value, datetime.datetime):
        return value.strftime("%Y%m%d%H%M%S")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            return datetime.datetime.fromisoformat(text).strftime("%Y%m%d%H%M%S")
        except Exception:
            pass
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


def _history_entry_to_hl7(entry: dict) -> str:
    """Build a medically standard HL7 v2.5.1 ORU^R01 message with LOINC codes & Base64 PDF attachment."""
    import base64
    entry = entry or {}
    now = datetime.datetime.now()
    ts = _hl7_datetime(now)
    report_dt = f"{entry.get('date', '')} {entry.get('time', '')}".strip()
    report_ts = ""
    if entry.get("date") and entry.get("time"):
        try:
            report_ts = datetime.datetime.strptime(report_dt, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d%H%M%S")
        except Exception:
            report_ts = _hl7_datetime(report_dt) or ts
    else:
        report_ts = ts

    patient_name = str(entry.get("patient_name", "") or "").strip()
    first_name = str(entry.get("first_name", "") or "").strip()
    last_name = str(entry.get("last_name", "") or "").strip()
    if not first_name and not last_name and patient_name:
        parts = patient_name.split(" ", 1)
        first_name = parts[0]
        last_name = parts[1] if len(parts) > 1 else ""

    patient_id = str(
        entry.get("patient_id")
        or entry.get("id")
        or entry.get("mrn")
        or entry.get("patient_number")
        or "PAT0001"
    ).strip()

    # Format DOB to YYYYMMDD
    dob_raw = str(entry.get("dob") or entry.get("date_of_birth") or "").strip()
    dob_hl7 = ""
    if dob_raw:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d", "%Y/%m/%d"):
            try:
                dob_hl7 = datetime.datetime.strptime(dob_raw, fmt).strftime("%Y%m%d")
                break
            except ValueError:
                pass
        if not dob_hl7:
            dob_hl7 = re.sub(r"\D", "", dob_raw)[:8]

    # Format Gender (HL7 Table 0001: M, F, O, U)
    raw_gender = str(entry.get("gender", "")).strip().upper()
    gender_hl7 = "M" if raw_gender in ("M", "MALE") else "F" if raw_gender in ("F", "FEMALE") else "O" if raw_gender in ("O", "OTHER") else "U"

    report_type = str(entry.get("report_type", "ECG") or "ECG").strip()
    doctor = str(entry.get("doctor", "") or "").strip()
    doctor_xcn = f"101^{_hl7_escape(doctor)}" if doctor else ""
    org_name = str(entry.get("org_name", "") or entry.get("Org.", "") or "CardioX Medical").strip()
    org_address = str(entry.get("org_address", "") or "").strip()
    report_file = str(entry.get("report_file", "") or "").strip()
    review_status = str(entry.get("review_status", "Pending") or "Pending").strip()
    owner_full_name = str(entry.get("owner_full_name", "") or "").strip()
    username = str(entry.get("username", "") or "").strip()

    message_id = re.sub(r"[^A-Za-z0-9]", "", f"{ts}{patient_id}")[:20] or ts
    msh_sender = _hl7_escape(org_name or HL7_APP_NAME)
    msh_receiver = _hl7_escape(owner_full_name or username or "EMR_PACS")

    lines = [
        f"MSH|^~\\&|{HL7_APP_NAME}|{msh_sender}|HISTORY|{msh_receiver}|{ts}||ORU^R01^ORU_R01|{message_id}|P|{HL7_VERSION}",
        f"PID|1||{_hl7_escape(patient_id)}||{_hl7_escape(last_name)}^{_hl7_escape(first_name)}||{dob_hl7}|{gender_hl7}|||{_hl7_escape(entry.get('address', '') or '')}||{_hl7_escape(entry.get('phone', '') or entry.get('mobile', '') or '')}",
        f"PV1|1|O|{_hl7_escape(org_name)}^^^{_hl7_escape(org_address)}|||||{doctor_xcn}",
        f"OBR|1|{_hl7_escape(patient_id)}|{_hl7_escape(message_id)}|8884-9^ECG Report^LN|||{report_ts}||||||||{doctor_xcn}",
    ]

    obx_index = 1

    # ── Standard LOINC ECG Observation Results ──
    hr_val = entry.get("heart_rate") or entry.get("hr")
    if hr_val:
        lines.append(f"OBX|{obx_index}|NM|8867-4^Heart Rate^LN|1|{_hl7_escape(str(hr_val))}|bpm|60-100|N|||F")
        obx_index += 1

    pr_val = entry.get("pr_interval") or entry.get("pr")
    if pr_val:
        lines.append(f"OBX|{obx_index}|NM|8834-4^PR Interval^LN|1|{_hl7_escape(str(pr_val))}|ms|120-200|N|||F")
        obx_index += 1

    qrs_val = entry.get("qrs_duration") or entry.get("qrs")
    if qrs_val:
        lines.append(f"OBX|{obx_index}|NM|8838-5^QRS Duration^LN|1|{_hl7_escape(str(qrs_val))}|ms|60-120|N|||F")
        obx_index += 1

    qt_val = entry.get("qt_interval") or entry.get("qt")
    if qt_val:
        lines.append(f"OBX|{obx_index}|NM|8889-8^QT Interval^LN|1|{_hl7_escape(str(qt_val))}|ms||N|||F")
        obx_index += 1

    qtc_val = entry.get("qtc_interval") or entry.get("qtc")
    if qtc_val:
        lines.append(f"OBX|{obx_index}|NM|8890-6^QTc Interval^LN|1|{_hl7_escape(str(qtc_val))}|ms|<440|N|||F")
        obx_index += 1

    findings = entry.get("findings") or entry.get("interpretation") or entry.get("conclusion") or ""
    if isinstance(findings, list):
        findings = "; ".join(str(f) for f in findings if f)
    if findings:
        lines.append(f"OBX|{obx_index}|TX|8884-9^ECG Impression^LN|1|{_hl7_escape(str(findings))}||||||F")
        obx_index += 1

    # ── Additional Metadata OBX items ──
    meta_items = [
        ("REPORT_TYPE", report_type),
        ("DOCTOR", doctor),
        ("ORG_NAME", org_name),
        ("REVIEW_STATUS", review_status),
    ]
    for code, val in meta_items:
        if val:
            lines.append(f"OBX|{obx_index}|TX|{code}^{code.replace('_', ' ').title()}||{_hl7_escape(str(val))}||||||F")
            obx_index += 1

    # ── Encapsulate PDF Report inside HL7 OBX ED segment if PDF exists ──
    if report_file and os.path.isfile(report_file) and report_file.lower().endswith(".pdf"):
        try:
            with open(report_file, "rb") as pdf_f:
                pdf_b64 = base64.b64encode(pdf_f.read()).decode("ascii")
            lines.append(f"OBX|{obx_index}|ED|11502-2^Laboratory Report PDF^LN|1|PDF^Application^pdf^Base64^{pdf_b64}||||||F")
            obx_index += 1
        except Exception as e:
            print(f"[HL7] Could not encapsulate PDF: {e}")

    return "\n".join(lines) + "\n"


def _write_hl7_history_file(entry: dict) -> str:
    """Persist an HL7 copy of the history entry next to the report archive."""
    try:
        os.makedirs(HL7_REPORT_DIR, exist_ok=True)
        source_name = os.path.splitext(os.path.basename(str(entry.get("report_file", "") or "")))[0]
        if not source_name:
            source_name = f"history_{entry.get('date', '')}_{entry.get('time', '')}".strip("_")
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", source_name).strip("._-") or "history_entry"
        hl7_path = os.path.join(HL7_REPORT_DIR, f"{safe_name}.hl7")
        with open(hl7_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(entry.get("hl7_message") or _history_entry_to_hl7(entry))
        return os.path.abspath(hl7_path)
    except Exception:
        return ""


def _normalize_owner(value: str) -> str:
    try:
        return str(value or "").strip().casefold()
    except Exception:
        return ""


def _reviewed_reports_headers() -> dict:
    headers = {}
    try:
        uploader = get_cloud_uploader()
        api_key = (
            getattr(uploader, "reviewed_reports_api_key", "")
            or os.getenv("PUBLIC_API_KEY", "")
            or os.getenv("REVIEWED_REPORTS_API_KEY", "")
        ).strip()
        if api_key:
            headers["x-api-key"] = api_key
    except Exception:
        api_key = (
            os.getenv("PUBLIC_API_KEY", "")
            or os.getenv("REVIEWED_REPORTS_API_KEY", "")
        ).strip()
        if api_key:
            headers["x-api-key"] = api_key
    return headers


def _history_email_defaults() -> dict:
    """Load locked email-sender settings from environment/config."""
    return {
        "smtp_host": os.getenv("EMAIL_SMTP_HOST", "smtp.gmail.com").strip(),
        "smtp_port": os.getenv("EMAIL_SMTP_PORT", "587").strip(),
        "sender_email": os.getenv("EMAIL_SENDER", "").strip(),
        "sender_password": os.getenv("EMAIL_PASSWORD", "").strip(),
    }

# ── pymupdf (fitz) optional import ─────────────────────────────────────────
try:
    import fitz as _fitz
    _HAS_FITZ = True
except ImportError:
    _HAS_FITZ = False


# ── helper: render one PDF page → QPixmap ──────────────────────────────────
def _pdf_page_to_pixmap(pdf_path: str, page_index: int = 0, zoom: float = 1.5) -> QPixmap:
    if not _HAS_FITZ:
        return QPixmap()
    try:
        doc = _fitz.open(pdf_path)
        if page_index >= len(doc):
            return QPixmap()
        page = doc[page_index]
        mat = _fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
        return QPixmap.fromImage(img)
    except Exception:
        return QPixmap()


# ══════════════════════════════════════════════════════════════════════════════
#  PDF Preview Panel
# ══════════════════════════════════════════════════════════════════════════════
class PdfPreviewPanel(QWidget):
    """Renders a local PDF file page-by-page inside a scroll area."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pdf_path = None
        self._page_index = 0
        self._total_pages = 0
        self._zoom = 1.4
        self._page_pixmap = QPixmap()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # ── toolbar ────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        self.page_label = QLabel("No report")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-weight:700;color:#111;")

        self.zoom_in = QPushButton("＋")
        self.zoom_in.setFixedWidth(32)
        self.zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom + 0.2))

        self.zoom_out = QPushButton("－")
        self.zoom_out.setFixedWidth(32)
        self.zoom_out.clicked.connect(lambda: self._set_zoom(max(0.4, self._zoom - 0.2)))

        bar.addStretch(1)
        for w in (self.page_label, self.zoom_out, self.zoom_in):
            bar.addWidget(w)
        bar.addStretch(1)
        root.addLayout(bar)

        # ── scroll area with page image ────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.viewport().installEventFilter(self)
        self.scroll.setStyleSheet("background:#fff7ed;border:1px solid #fed7aa;border-radius:12px;")

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.img_label.setMinimumSize(240, 240)
        self.img_label.setWordWrap(True)
        self.scroll.setWidget(self.img_label)
        root.addWidget(self.scroll, 1)

        self._update_nav()

    # ── public ─────────────────────────────────────────────────────────────
    def load_pdf(self, path: str):
        self._pdf_path = path
        self._page_index = 0
        if not path or not os.path.exists(path):
            self._total_pages = 0
            self._page_pixmap = QPixmap()
            self.img_label.setPixmap(QPixmap())
            self.img_label.setText("Report file not found.")
            self.img_label.setStyleSheet("color:#7c2d12;font-size:14px;font-weight:600;")
            self._update_nav()
            return

        if not _HAS_FITZ:
            self._total_pages = 0
            self._page_pixmap = QPixmap()
            self.img_label.setPixmap(QPixmap())
            self.img_label.setText(
                "Preview renderer is not installed on this system.\n"
                "Install `pymupdf` to enable in-app preview, or use 'Open Selected Report'."
            )
            self.img_label.setStyleSheet("color:#7c2d12;font-size:13px;font-weight:600;")
            self._update_nav()
            return

        if path and os.path.exists(path):
            try:
                doc = _fitz.open(path)
                self._total_pages = len(doc)
                doc.close()
            except Exception:
                self._total_pages = 0
        else:
            self._total_pages = 0
        self._render()

    def clear(self):
        self._pdf_path = None
        self._page_index = 0
        self._total_pages = 0
        self._page_pixmap = QPixmap()
        self.img_label.setPixmap(QPixmap())
        self.img_label.setText("Select a report from history to preview")
        self.img_label.setStyleSheet("color:#7c2d12;font-size:15px;font-weight:600;")
        self._update_nav()

    # ── private ────────────────────────────────────────────────────────────
    def eventFilter(self, obj, event):
        if obj is self.scroll.viewport() and event.type() == QEvent.Resize:
            self._apply_scaled_pixmap()
        return super().eventFilter(obj, event)

    def _render(self):
        if not self._pdf_path or self._total_pages == 0:
            self.clear()
            return
        px = _pdf_page_to_pixmap(self._pdf_path, self._page_index, self._zoom)
        if px.isNull():
            self._page_pixmap = QPixmap()
            if not _HAS_FITZ:
                self.img_label.setText("In-app PDF preview is unavailable on this system.\nUse 'Open Selected Report' or double-click a row.")
            else:
                self.img_label.setText("Could not render page.")
            self.img_label.setStyleSheet("color:#7c2d12;font-size:13px;font-weight:600;")
        else:
            self._page_pixmap = px
            self.img_label.setText("")
            self.img_label.setStyleSheet("")
            self._apply_scaled_pixmap()
        self._update_nav()

    def _apply_scaled_pixmap(self):
        if self._page_pixmap.isNull():
            self.img_label.setPixmap(QPixmap())
            return
        viewport_size = self.scroll.viewport().size()
        if viewport_size.width() <= 0 or viewport_size.height() <= 0:
            self.img_label.setPixmap(self._page_pixmap)
            return
        scaled = self._page_pixmap.scaled(
            max(100, viewport_size.width() - 24),
            max(100, viewport_size.height() - 24),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.img_label.setPixmap(scaled)
        self.img_label.resize(scaled.size())

    def _set_zoom(self, z):
        self._zoom = z
        self._render()

    def _update_nav(self):
        self.zoom_in.setVisible(False)
        self.zoom_out.setVisible(False)
        if self._total_pages > 1:
            self.page_label.setText(f"Page {self._page_index+1} / {self._total_pages}")
            self.page_label.setVisible(True)
        else:
            self.page_label.setText("")
            self.page_label.setVisible(False)


# ══════════════════════════════════════════════════════════════════════════════
#  Email-Send Dialog
# ══════════════════════════════════════════════════════════════════════════════
class SendEmailDialog(QDialog):
    """Send ECG report PDF via email using SMTP."""

    def __init__(self, report_path: str, patient_name: str = "", parent=None):
        super().__init__(parent)
        self.report_path = report_path
        self._email_cfg = _history_email_defaults()
        self._smtp_host = self._email_cfg["smtp_host"] or "smtp.gmail.com"
        self._smtp_port = self._email_cfg["smtp_port"] or "587"
        self._sender_email = self._email_cfg["sender_email"]
        self._sender_password = self._email_cfg["sender_password"]
        self.setWindowTitle("Send Report by Email")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumSize(520, 420)
        self.setStyleSheet("""
            QDialog{background:#fffaf4;color:#1f2937;}
            QLabel{color:#1f2937;}
            QLineEdit,QTextEdit{
                border:1px solid #f7c58b;
                border-radius:10px;
                padding:10px 12px;
                background:#ffffff;
                color:#1f2937;
                font-size:13px;
            }
            QTextEdit{selection-background-color:#fed7aa;}
            QPushButton{
                border-radius:10px;
                padding:10px 18px;
                font-weight:700;
                color:#ffffff;
                background:#ff7a00;
                border:none;
                min-width:92px;
            }
            QPushButton:hover{background:#ff8a1f;}
            QPushButton#cancel{background:#4b5563;}
            QPushButton#cancel:hover{background:#374151;}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(12)

        title = QLabel("Send ECG Report by Email")
        title.setStyleSheet("font-size:18px;font-weight:800;color:#111827;")
        layout.addWidget(title)

        subtitle = QLabel("Compose the message and choose who should receive the report.")
        subtitle.setStyleSheet("color:#7c2d12;font-size:11px;font-weight:600;")
        layout.addWidget(subtitle)

        form = QFormLayout()
        form.setSpacing(12)
        form.setLabelAlignment(Qt.AlignRight)

        self.to_field = QLineEdit()
        self.to_field.setPlaceholderText("recipient@example.com")
        self.cc_field = QLineEdit()
        self.cc_field.setPlaceholderText("Optional CC addresses, comma-separated")
        self.subject = QLineEdit(f"ECG Report — {patient_name}".strip(" —"))
        self.body = QTextEdit()
        self.body.setPlainText(
            f"Dear Doctor,\n\nPlease find attached the ECG report for patient: {patient_name}.\n\n"
            f"Regards,\nCardioX"
        )
        self.body.setFixedHeight(120)

        form.addRow("To:", self.to_field)
        form.addRow("CC:", self.cc_field)
        form.addRow("Subject:", self.subject)
        form.addRow("Body:", self.body)
        layout.addLayout(form)

        info_row = QFrame()
        info_row.setStyleSheet("QFrame{background:#fff3e6;border:1px solid #ffd199;border-radius:10px;}")
        info_layout = QHBoxLayout(info_row)
        info_layout.setContentsMargins(12, 8, 12, 8)
        info_layout.setSpacing(8)
        if self._sender_email:
            sender_text = f"Sending from {self._sender_email}"
        else:
            sender_text = "Sender account is not configured"
        info_lbl = QLabel(sender_text)
        info_lbl.setStyleSheet("color:#9a3412;font-size:11px;font-weight:600;")
        info_layout.addWidget(info_lbl)
        info_layout.addStretch()
        layout.addWidget(info_row)

        # Attachment label
        fname = os.path.basename(report_path) if report_path else "—"
        att_lbl = QLabel(f"📎 Attachment: {fname}")
        att_lbl.setStyleSheet("color:#6b7280;font-weight:600;margin-top:2px;")
        layout.addWidget(att_lbl)

        # Buttons
        btns = QHBoxLayout()
        send_btn = QPushButton("Send")
        send_btn.clicked.connect(self._do_send)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setObjectName("cancel")
        cancel_btn.clicked.connect(self.reject)
        btns.addStretch()
        btns.addWidget(send_btn)
        btns.addWidget(cancel_btn)
        layout.addLayout(btns)

    def _do_send(self):
        to_addr = self.to_field.text().strip()
        if not to_addr:
            QMessageBox.warning(self, "Missing", "Please enter a recipient email address.")
            return
        sender_email = self._sender_email
        sender_password = self._sender_password
        if not sender_email or not sender_password:
            QMessageBox.warning(
                self,
                "Missing Sender Config",
                "Sender email credentials are not configured. Set EMAIL_SENDER and EMAIL_PASSWORD in .env.",
            )
            return
        if not os.path.exists(self.report_path):
            QMessageBox.warning(self, "File Missing", "Report PDF not found.")
            return

        # Build message
        msg = MIMEMultipart()
        msg["From"] = sender_email
        msg["To"] = to_addr
        cc_list = [x.strip() for x in self.cc_field.text().split(",") if x.strip()]
        if cc_list:
            msg["Cc"] = ", ".join(cc_list)
        msg["Subject"] = self.subject.text()
        msg.attach(MIMEText(self.body.toPlainText(), "plain"))

        # Attach PDF
        with open(self.report_path, "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f'attachment; filename="{os.path.basename(self.report_path)}"')
        msg.attach(part)

        try:
            port = int(self._smtp_port or "587")
            server = smtplib.SMTP(self._smtp_host, port, timeout=15)
            server.starttls()
            server.login(sender_email, sender_password)
            all_recipients = [to_addr] + cc_list
            server.sendmail(msg["From"], all_recipients, msg.as_string())
            server.quit()
            QMessageBox.information(self, "Sent", "Report emailed successfully!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Send Failed", f"Email could not be sent:\n{e}")


# ══════════════════════════════════════════════════════════════════════════════
#  Upload worker
# ══════════════════════════════════════════════════════════════════════════════
class UploadWorker(QThread):
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, uploader, file_path, doctor_name, metadata=None):
        super().__init__()
        self.uploader = uploader
        self.file_path = file_path
        self.doctor_name = doctor_name
        self.metadata = metadata

    def run(self):
        try:
            result = self.uploader.send_for_doctor_review(
                self.file_path, self.doctor_name, metadata=self.metadata
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


# ══════════════════════════════════════════════════════════════════════════════
#  Main History Window
# ══════════════════════════════════════════════════════════════════════════════
class WaveformPlotPanel(QWidget):
    """Embedded ECG waveform viewer for JSON reports."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lead_map = {}
        self._sampling_rate = 500.0

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Lead:"))
        self.lead_combo = QComboBox()
        self.lead_combo.setMinimumWidth(120)
        self.lead_combo.currentTextChanged.connect(self._plot_selected_lead)
        top.addWidget(self.lead_combo)
        self.meta_label = QLabel("Select a JSON report to plot waves")
        self.meta_label.setStyleSheet("color:#7c2d12;font-size:11px;font-weight:600;")
        self.meta_label.setWordWrap(True)
        top.addWidget(self.meta_label, 1)
        root.addLayout(top)

        fig = Figure(figsize=(6, 3), dpi=100)
        self.canvas = FigureCanvas(fig)
        self.axes = fig.add_subplot(111)
        fig.patch.set_facecolor("#ffffff")
        root.addWidget(self.canvas, 1)
        self.clear_plot()

    def clear_plot(self, message: str = "Select a JSON report to plot waves"):
        self._lead_map = {}
        self.lead_combo.blockSignals(True)
        self.lead_combo.clear()
        self.lead_combo.blockSignals(False)
        self.meta_label.setText(message)
        self.axes.clear()
        self.axes.set_title("ECG Waveform", fontsize=11, fontweight="bold")
        self.axes.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            transform=self.axes.transAxes,
            color="#9a3412",
            fontsize=11,
        )
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.canvas.draw_idle()

    def load_waveforms(self, json_path: str, payload: dict):
        lead_map = {}
        sampling_rate = 500.0
        if isinstance(payload, dict):
            sampling_rate = float(payload.get("sampling_rate") or payload.get("fs") or 500.0)
            leads = payload.get("leads")
            if isinstance(leads, dict):
                for lead_name, values in leads.items():
                    arr = self._to_numeric_array(values)
                    if arr is not None and arr.size:
                        lead_map[str(lead_name)] = arr
            else:
                for key, values in payload.items():
                    arr = self._to_numeric_array(values)
                    if arr is not None and arr.size:
                        lead_map[str(key)] = arr

        if not lead_map:
            self.clear_plot("This JSON file does not contain plottable ECG wave data.")
            return

        self._lead_map = dict(sorted(lead_map.items()))
        self._sampling_rate = sampling_rate if sampling_rate > 0 else 500.0

        self.lead_combo.blockSignals(True)
        self.lead_combo.clear()
        self.lead_combo.addItems(list(self._lead_map.keys()))
        default_lead = "II" if "II" in self._lead_map else next(iter(self._lead_map))
        self.lead_combo.setCurrentText(default_lead)
        self.lead_combo.blockSignals(False)

        self.meta_label.setText(
            f"{os.path.basename(json_path)}  |  {len(self._lead_map)} lead(s)  |  {self._sampling_rate:.1f} Hz"
        )
        self._plot_selected_lead(default_lead)

    def _to_numeric_array(self, values):
        if not isinstance(values, list) or not values:
            return None
        try:
            arr = np.asarray(values, dtype=float).reshape(-1)
        except Exception:
            return None
        return arr if arr.size else None

    def _plot_selected_lead(self, lead_name: str):
        signal = self._lead_map.get(lead_name)
        if signal is None or signal.size == 0:
            return

        plot_signal = signal
        if plot_signal.size > 5000:
            step = max(1, plot_signal.size // 5000)
            plot_signal = plot_signal[::step]
        time_axis = np.arange(plot_signal.size, dtype=float) / max(self._sampling_rate, 1.0)

        self.axes.clear()
        self.axes.set_facecolor("#fffaf5")
        self.axes.plot(time_axis, plot_signal, color="#f97316", linewidth=1.0)
        self.axes.set_title(f"Lead {lead_name}", fontsize=11, fontweight="bold")
        self.axes.set_xlabel("Time (s)")
        self.axes.set_ylabel("Amplitude")
        self.axes.grid(True, color="#fed7aa", linewidth=0.6, alpha=0.9)
        for spine in self.axes.spines.values():
            spine.set_color("#fdba74")
        self.canvas.draw_idle()


class HistoryWindow(QDialog):
    """ECG Report History — split pane: report list (left) + PDF preview (right)."""

    # ── black/white clean theme ─────────────────────────────────────────────
    STYLE = """
        QDialog{
            background:#fffaf5;
            color:#1f2937;
            font-family:'Segoe UI',Helvetica,Arial,sans-serif;
        }
        QTableWidget{
            border:1px solid #fed7aa;
            background:#ffffff;
            gridline-color:#ffedd5;
            selection-background-color:#ffedd5;
            selection-color:#9a3412;
            alternate-background-color:#fff7ed;
        }
        QTableWidget::item{
            padding:6px 8px;
            border-bottom:1px solid #ffedd5;
            color:#1f2937;
        }
        QHeaderView::section{
            background:#111111;
            color:#ffffff;
            font-weight:700;
            font-size:12px;
            padding:10px 6px;
            border:none;
            border-right:1px solid #2a2a2a;
        }
        QPushButton{
            background:#f97316;
            color:#ffffff;
            border:1px solid #f97316;
            border-radius:10px;
            padding:8px 16px;
            font-weight:600;
            font-size:12px;
        }
        QPushButton:hover{background:#ea580c;}
        QPushButton:pressed{background:#c2410c;}
        QPushButton#btn_secondary,
        QPushButton#btn_close{
            background:#ffffff;
            color:#9a3412;
            border:1px solid #fdba74;
        }
        QPushButton#btn_secondary:hover,
        QPushButton#btn_close:hover{background:#fff7ed;}
        QLineEdit,QComboBox,QDateEdit{
            border:1px solid #fdba74;
            border-radius:10px;
            padding:6px 10px;
            background:#ffffff;
            color:#1f2937;
            font-size:13px;
        }
        QLineEdit:focus,QComboBox:focus,QDateEdit:focus{border-color:#f97316;}
        QComboBox::drop-down{border:none;width:22px;}
        QLabel{color:#1f2937;font-weight:600;}
        QGroupBox{
            border:1px solid #fed7aa;
            border-radius:14px;
            background:#ffffff;
            margin-top:12px;
            padding:12px 10px 10px 10px;
            font-weight:bold;
        }
        QGroupBox::title{
            color:#9a3412;
            font-weight:700;
            subcontrol-origin:margin;
            left:12px;
            padding:0 4px;
        }
        QSplitter::handle{background:#fed7aa;width:3px;}
        QScrollBar:vertical{background:#fff7ed;width:10px;border-radius:5px;}
        QScrollBar::handle:vertical{background:#fdba74;border-radius:5px;min-height:20px;}
        QScrollBar::handle:vertical:hover{background:#fb923c;}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
        QTabBar::tab{
            background:#fff7ed;
            color:#9a3412;
            border:1px solid #fed7aa;
            border-bottom:none;
            border-radius:10px 10px 0 0;
            padding:8px 18px;
            font-weight:600;
            font-size:12px;
            margin-right:2px;
            min-width:120px;
        }
        QTabBar::tab:selected{background:#111111;color:#ffffff;}
        QTabBar::tab:hover:!selected{background:#ffedd5;}
        QTabWidget::pane{border:1px solid #fed7aa;border-radius:0 12px 12px 12px;background:#ffffff;}
        QListWidget{border:1px solid #fed7aa;border-radius:10px;background:#ffffff;}
        QListWidget::item{padding:6px 10px;color:#1f2937;}
        QListWidget::item:selected{background:#111111;color:#ffffff;}
        QListWidget::item:hover:!selected{background:#fff7ed;}
    """

    def __init__(self, parent=None, username=None, owner_full_name=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Report History")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.username = username
        self.owner_full_name = owner_full_name or username
        self.all_history_entries = []
        self.all_json_reports = []
        self._cloud_preview_map = {}
        self._preview_temp_pdf = ""
        self.setStyleSheet(self.STYLE)

        # Load doctor profile defaults from users.json for table fallback
        try:
            from dashboard.doctor_profile_dialog import _load_users_db, _find_user_key_and_record
            users_db = _load_users_db()
            _, user_rec = _find_user_key_and_record(users_db, self.username or "")
            self._user_profile = user_rec or {}
        except Exception:
            self._user_profile = {}

        # Responsive: use 90% of screen but never less than 800x500
        screen = QApplication.desktop().availableGeometry()
        self.resize(max(800, int(screen.width() * 0.90)),
                    max(500, int(screen.height() * 0.82)))
        self.setMinimumSize(800, 480)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self._build_ui()

        try:
            import threading
            threading.Thread(target=self._prefetch_doctors, daemon=True).start()
        except Exception:
            pass

        self.load_history()
        self._setup_shortcuts()   # keyboard shortcuts — added last so all widgets exist

    def _get_profile_fallback(self, entry, field_name):
        val = str(entry.get(field_name, "") or "").strip()
        if val:
            return val

        prof = getattr(self, "_user_profile", {}) or {}

        if field_name == "doctor":
            return str(entry.get("doctor", "") or prof.get("doctor", "") or prof.get("doctor_name", "") or prof.get("full_name", "") or self.owner_full_name or self.username or "").strip()
        elif field_name == "org_name":
            return str(entry.get("org_name", "") or entry.get("Org.", "") or prof.get("org_name", "") or prof.get("Org. Name", "") or prof.get("Org.", "") or "").strip()
        elif field_name == "org_address":
            return str(entry.get("org_address", "") or prof.get("org_address", "") or prof.get("Org. Address", "") or "").strip()
        return val

    # ── Keyboard Shortcuts ──────────────────────────────────────────────────
    def _setup_shortcuts(self):
        """Register keyboard shortcuts for common history-window actions."""
        def _sc(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.activated.connect(slot)
            return sc

        _sc("Ctrl+P", self._preview_selected)                      # Preview
        # _sc("Ctrl+E", self._send_email)                          # Email (hidden)
        _sc("Ctrl+O", self._open_in_system)                        # Open in viewer
        _sc("Ctrl+F", lambda: self.search_input.setFocus())        # Focus search
        _sc("F5",     self.load_history)                           # Refresh table
        _sc("Ctrl+D", self._export_dicom)                          # Export DICOM
        _sc("F1",     self._show_shortcuts_dialog)                 # Show shortcuts popup

    # ── Shortcuts Help Dialog ────────────────────────────────────────────────
    def _show_shortcuts_dialog(self):
        """Display a popup dialog listing all system keyboard shortcuts."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Keyboard Shortcuts Reference")
        dialog.setWindowFlags(dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dialog.setMinimumSize(640, 560)
        dialog.setStyleSheet("""
            QDialog { background: #ffffff; border-radius: 12px; }
            QLabel { color: #0f172a; font-family: 'Segoe UI', Arial; }
            QTableWidget {
                background: #ffffff; color: #0f172a; gridline-color: #e2e8f0;
                border: 1px solid #cbd5e1; border-radius: 8px; font-size: 13px;
                alternate-background-color: #f8fafc;
            }
            QHeaderView::section {
                background: #f1f5f9; color: #0f172a; font-weight: bold; font-size: 13px;
                padding: 10px; border: none; border-bottom: 2px solid #cbd5e1;
            }
            QPushButton {
                background: #ff6600; color: #ffffff; border-radius: 8px; padding: 10px 28px;
                font-size: 14px; font-weight: bold; border: none;
            }
            QPushButton:hover { background: #e65c00; }
        """)
        layout = QVBoxLayout(dialog)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(16)

        header_box = QHBoxLayout()
        icon_lbl = QLabel("⌨️")
        icon_lbl.setStyleSheet("font-size: 24px;")
        title = QLabel("Keyboard Shortcuts Cheat Sheet")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #ff6600;")
        header_box.addWidget(icon_lbl)
        header_box.addWidget(title)
        header_box.addStretch()
        layout.addLayout(header_box)

        shortcuts = [
            ("History Window", "Ctrl + P", "Preview selected report in side panel"),
            ("History Window", "Ctrl + O", "Open selected report in system PDF viewer"),
            ("History Window", "Ctrl + F", "Jump cursor focus to Search bar"),
            ("History Window", "Ctrl + D", "Export selected report as DICOM file"),
            ("History Window", "F5", "Refresh report history table"),
            ("Dashboard", "Ctrl + N", "Open New Patient Registration"),
            ("Dashboard", "Ctrl + H", "Open ECG Report History Window"),
            ("General", "F1", "Open this Keyboard Shortcuts cheat sheet"),
            ("General", "Esc", "Close active popup / window"),
        ]

        table = QTableWidget()
        table.setColumnCount(3)
        table.setHorizontalHeaderLabels(["Scope", "Shortcut Key", "Action Description"])
        table.setRowCount(len(shortcuts))
        table.verticalHeader().setVisible(False)
        table.setAlternatingRowColors(True)

        for row, (scope, key, desc) in enumerate(shortcuts):
            item_scope = QTableWidgetItem(scope)
            item_scope.setForeground(QColor("#475569"))
            item_scope.setFont(QFont("Segoe UI", 9, QFont.Bold))
            item_scope.setTextAlignment(Qt.AlignCenter)

            item_key = QTableWidgetItem(key)
            item_key.setTextAlignment(Qt.AlignCenter)
            item_key.setFont(QFont("Consolas", 10, QFont.Bold))
            item_key.setForeground(QColor("#0284c7"))
            item_key.setBackground(QColor("#e0f2fe"))

            item_desc = QTableWidgetItem(desc)
            item_desc.setForeground(QColor("#0f172a"))
            item_desc.setFont(QFont("Segoe UI", 10, QFont.Bold if "F1" in key or "Ctrl" in key else QFont.Normal))

            table.setItem(row, 0, item_scope)
            table.setItem(row, 1, item_key)
            table.setItem(row, 2, item_desc)
            table.setRowHeight(row, 32)

        table.setColumnWidth(0, 140)
        table.setColumnWidth(1, 150)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        layout.addWidget(table)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setCursor(Qt.PointingHandCursor)
        close_btn.clicked.connect(dialog.accept)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)

        dialog.exec_()



    def _get_selected_entry(self):
        """Retrieve the exact history entry dictionary for the currently selected table row."""
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if not item:
            return None
        stored = item.data(Qt.UserRole)
        if isinstance(stored, dict):
            return stored

        # Fallback string/fuzzy search across all_history_entries
        date_item = self.table.item(row, 0)
        name_item = self.table.item(row, 4)
        if date_item and name_item:
            d = date_item.text().strip()
            n = name_item.text().strip()
            for e in self.all_history_entries:
                if e.get("date", "") == d and (
                    e.get("patient_name", "") == n or
                    (e.get("first_name", "") + " " + e.get("last_name", "")).strip() == n
                ):
                    return e
        return None

    # ── DICOM Export ─────────────────────────────────────────────────────────
    def _export_dicom(self):
        """Export the selected history row as a DICOM file."""
        try:
            from utils.dicom_exporter import entry_to_dicom, is_available
        except ImportError:
            QMessageBox.warning(
                self, "DICOM Export",
                "DICOM exporter module not found.\n"
                "Ensure dicom_exporter.py is in src/utils/."
            )
            return

        if not is_available():
            QMessageBox.warning(
                self, "DICOM Export — Not Available",
                "pydicom is not installed.\n"
                "Run the following command and restart:\n\n"
                "    pip install pydicom"
            )
            return

        entry = self._get_selected_entry()
        if entry is None:
            QMessageBox.warning(self, "DICOM Export", "Please select a report row first.")
            return

        # Ask user where to save
        patient_safe = re.sub(r"\W+", "_", str(entry.get("patient_name") or "patient")).strip("_")
        default_name = f"ECG_{patient_safe}_{entry.get('date','')}.dcm"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export as DICOM",
            default_name,
            "DICOM Files (*.dcm);;All Files (*.*)"
        )
        if not save_path:
            return

        try:
            out = entry_to_dicom(entry, save_path)
            reply = QMessageBox.information(
                self,
                "DICOM Export — Saved",
                f"DICOM file saved:\n{out}\n\nOpen containing folder?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply == QMessageBox.Yes:
                import subprocess
                import sys
                if sys.platform == "win32":
                    subprocess.Popen(f'explorer /select,"{out}"')
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", "-R", out])
                else:
                    subprocess.Popen(["xdg-open", os.path.dirname(out)])
        except Exception as exc:
            QMessageBox.critical(
                self, "DICOM Export Failed",
                f"Could not create DICOM file:\n{exc}"
            )

    # ── HL7 Export ───────────────────────────────────────────────────────────
    def _export_hl7(self):
        """Export or copy the HL7 v2.5.1 ORU^R01 message for the selected report."""
        entry = self._get_selected_entry()
        if entry is None:
            QMessageBox.warning(self, "Export HL7", "Please select a report row first.")
            return


        hl7_text = entry.get("hl7_message") or _history_entry_to_hl7(entry)

        # Save dialog
        patient_safe = re.sub(r"\W+", "_", str(entry.get("patient_name") or "patient")).strip("_")
        default_name = f"HL7_{patient_safe}_{entry.get('date','')}.hl7"
        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export HL7 v2.5.1 Message",
            default_name,
            "HL7 Files (*.hl7);;Text Files (*.txt);;All Files (*.*)"
        )
        if not save_path:
            return

        try:
            with open(save_path, "w", encoding="utf-8", newline="\n") as f:
                f.write(hl7_text)

            QApplication.clipboard().setText(hl7_text)
            reply = QMessageBox.information(
                self,
                "HL7 Message Exported",
                f"HL7 v2.5.1 message saved to:\n{save_path}\n\n"
                f"✅ HL7 message has also been copied to your clipboard!\n"
                f"You can paste it directly into your EMR/HIS integration terminal.",
                QMessageBox.Ok,
            )
        except Exception as exc:
            QMessageBox.critical(self, "HL7 Export Failed", f"Could not write HL7 file:\n{exc}")

    # ── UI construction ──────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt5.QtWidgets import QTabWidget
        root = QVBoxLayout(self)
        root.setContentsMargins(16, 14, 16, 14)
        root.setSpacing(0)

        # ── Header bar ─────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(72)
        header.setStyleSheet(
            "QFrame{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #111111,stop:0.55 #1f1f1f,stop:1 #f97316);border-radius:16px 16px 0 0;}"
        )
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 0, 18, 0)
        logo = QLabel("⚡")
        logo.setStyleSheet("font-size:20px;color:#fff;")
        title_lbl = QLabel("ECG Report History")
        title_lbl.setStyleSheet(
            "font-size:20px;font-weight:700;color:#fff;letter-spacing:0.5px;"
        )
        sub_lbl = QLabel("Search, preview, open and manage patient ECG reports")
        sub_lbl.setStyleSheet("font-size:11px;color:#ffedd5;font-weight:500;")
        txt_col = QVBoxLayout()
        txt_col.setSpacing(1)
        txt_col.addWidget(title_lbl)
        txt_col.addWidget(sub_lbl)
        hh.addWidget(logo)
        hh.addSpacing(10)
        hh.addLayout(txt_col)
        hh.addStretch()
        shortcuts_btn = QPushButton("⌨️  Shortcuts (F1)")
        shortcuts_btn.setFixedSize(155, 36)
        shortcuts_btn.setCursor(Qt.PointingHandCursor)
        shortcuts_btn.setToolTip("View all keyboard shortcuts cheat sheet (F1)")
        shortcuts_btn.setStyleSheet(
            "QPushButton { background: #ff6600; color: #ffffff; "
            "border: 1px solid #ff8533; border-radius: 10px; font-weight: bold; font-size: 12px; padding: 2px 10px; }"
            "QPushButton:hover { background: #e65c00; border-color: #ffffff; }"
        )
        shortcuts_btn.clicked.connect(self._show_shortcuts_dialog)
        hh.addWidget(shortcuts_btn)
        hh.addSpacing(8)

        close_btn = QPushButton("✕  Close")
        close_btn.setObjectName("btn_close")
        close_btn.setFixedSize(98, 36)
        close_btn.clicked.connect(self.close)
        hh.addWidget(close_btn)
        root.addWidget(header)

        # ── Search bar ─────────────────────────────────────────────────────
        search_frame = self._build_search_bar()
        search_frame.setStyleSheet(
            "QFrame{background:#ffffff;border:1px solid #fed7aa;"
            "border-top:none;border-bottom:none;padding:10px 12px;}"
        )
        root.addWidget(search_frame)

        # ── Main splitter ──────────────────────────────────────────────────
        self.splitter = QSplitter(Qt.Horizontal)
        self.splitter.setHandleWidth(3)

        # Left: tab widget — Reports | Reviewed from API
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)

        # Tab 1 — Local reports
        reports_tab = QWidget()
        rt_layout = QVBoxLayout(reports_tab)
        rt_layout.setContentsMargins(0, 6, 0, 0)
        rt_layout.setSpacing(6)
        self._build_table(rt_layout)
        self._build_action_buttons(rt_layout)
        self.tabs.addTab(reports_tab, "Reports")

        # Tab 2 — Reviewed reports from API
        reviewed_tab = self._build_reviewed_tab()
        self.tabs.addTab(reviewed_tab, "Reviewed")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Right: PDF preview
        preview_group = QGroupBox("  Report Preview")
        pg_layout = QVBoxLayout(preview_group)
        pg_layout.setContentsMargins(10, 10, 10, 10)
        pg_layout.setSpacing(8)
        preview_hint = QLabel("Single-click a row to preview. Double-click a row to open the report.")
        preview_hint.setStyleSheet("color:#7c2d12;font-size:11px;font-weight:600;")
        preview_hint.setWordWrap(True)
        pg_layout.addWidget(preview_hint)
        preview_open_btn = QPushButton("Open Selected Report")
        preview_open_btn.setObjectName("btn_secondary")
        preview_open_btn.setFixedHeight(34)
        preview_open_btn.clicked.connect(self._open_in_system)
        pg_layout.addWidget(preview_open_btn)
        self.preview_panel = PdfPreviewPanel()
        pg_layout.addWidget(self.preview_panel)

        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(preview_group)
        self.splitter.setChildrenCollapsible(False)
        self.tabs.setMinimumWidth(520)
        preview_group.setMinimumWidth(360)
        self.splitter.setStretchFactor(0, 56)
        self.splitter.setStretchFactor(1, 44)
        self.splitter.setSizes([max(520, int(self.width() * 0.64)), max(360, int(self.width() * 0.36))])
        root.addWidget(self.splitter, 1)

    def _on_tab_changed(self, idx):
        """When user switches to the Reviewed tab, load doctors then auto-fetch."""
        if idx == 1:
            # Load doctors into combo if not already done
            if self.rev_doctor_combo.count() <= 1:   # only -- Select -- present
                self._populate_doctor_combo()

    def _populate_doctor_combo(self):
        """Fill the doctor combo from the cloud uploader in a background thread."""
        import threading
        self.rev_status_lbl.setText("Loading doctor list from cloud...")
        QApplication.processEvents()

        def _do():
            try:
                uploader = get_cloud_uploader()
                docs = uploader.get_available_doctors() or []
            except Exception as e:
                docs = []
                # Update on main thread via a single-shot timer trick
                import functools
                from PyQt5.QtCore import QTimer
                QTimer.singleShot(0, functools.partial(
                    self.rev_status_lbl.setText,
                    "Offline mode: doctor list could not be refreshed from the cloud."
                ))
                QTimer.singleShot(0, functools.partial(
                    self.rev_status_lbl.setStyleSheet,
                    "color:#b91c1c;font-size:11px;font-weight:600;"
                ))
                return

            from PyQt5.QtCore import QTimer
            import functools
            QTimer.singleShot(0, functools.partial(self._fill_doctor_combo, docs))

        threading.Thread(target=_do, daemon=True).start()

    def _fill_doctor_combo(self, doctors):
        """Populate combo on the main thread and auto-select first doctor."""
        self.rev_doctor_combo.blockSignals(True)
        current = self.rev_doctor_combo.currentText()
        self.rev_doctor_combo.clear()
        self.rev_doctor_combo.addItem("-- Select Doctor --")
        for d in doctors:
            if d:
                self.rev_doctor_combo.addItem(str(d))
        idx = self.rev_doctor_combo.findText(current)
        self.rev_doctor_combo.setCurrentIndex(max(0, idx))
        self.rev_doctor_combo.blockSignals(False)
        if doctors:
            self.rev_status_lbl.setStyleSheet("color:#166534;font-size:11px;font-weight:500;")
            self.rev_status_lbl.setText(
                f"{len(doctors)} doctor(s) loaded. Select one to fetch reports."
            )
        else:
            self.rev_status_lbl.setStyleSheet("color:#b91c1c;font-size:11px;font-weight:600;")
            self.rev_status_lbl.setText("Offline mode or empty cloud response: no doctors available right now.")

    def _build_reviewed_tab(self) -> QWidget:
        """Reviewed-by-doctor reports fetched from the public API."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        # ── info banner ────────────────────────────────────────────────────
        info = QLabel(
            "Browse reviewed reports by doctor and open the shared report link directly from the list."
        )
        info.setStyleSheet(
            "background:#fff7ed;border:1px solid #fed7aa;border-radius:10px;"
            "color:#7c2d12;font-weight:500;font-size:12px;padding:10px 12px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── toolbar ────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        dl = QLabel("Doctor:")
        dl.setStyleSheet("color:#9a3412;font-size:12px;font-weight:700;")

        # Combo populated from get_available_doctors()
        self.rev_doctor_combo = QComboBox()
        self.rev_doctor_combo.setMinimumWidth(200)
        self.rev_doctor_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.rev_doctor_combo.setMaxVisibleItems(20)
        self.rev_doctor_combo.addItem("-- Select Doctor --")
        # Auto-fetch when doctor changes
        self.rev_doctor_combo.currentIndexChanged.connect(
            lambda idx: self._load_reviewed_reports() if idx > 0 else None
        )

        refresh_docs_btn = QPushButton("Load Doctors")
        refresh_docs_btn.setMinimumHeight(32)
        refresh_docs_btn.setObjectName("btn_secondary")
        refresh_docs_btn.clicked.connect(self._populate_doctor_combo)

        open_rev_btn = QPushButton("Open URL")
        open_rev_btn.setObjectName("btn_secondary")
        open_rev_btn.setMinimumHeight(32)
        open_rev_btn.clicked.connect(self._open_reviewed_selected)

        copy_rev_btn = QPushButton("Copy Link")
        copy_rev_btn.setObjectName("btn_secondary")
        copy_rev_btn.setMinimumHeight(32)
        copy_rev_btn.clicked.connect(self._copy_reviewed_link)

        top.addWidget(dl)
        top.addWidget(self.rev_doctor_combo, 1)
        top.addWidget(refresh_docs_btn)
        top.addStretch()
        top.addWidget(open_rev_btn)
        top.addWidget(copy_rev_btn)
        layout.addLayout(top)

        # ── reviewed table ─────────────────────────────────────────────────
        # Columns: Date | Time | Patient | Type | Doctor | Presigned URL
        self.rev_table = QTableWidget(0, 6)
        self.rev_table.setHorizontalHeaderLabels(
            ["Date", "Time", "Report Name", "Type", "Doctor", "Presigned / Preview URL"]
        )
        self.rev_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.rev_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.rev_table.setAlternatingRowColors(True)
        hdr = self.rev_table.horizontalHeader()
        hdr.setStretchLastSection(True)          # URL column fills remaining space
        hdr.setSectionResizeMode(0, hdr.ResizeToContents)
        hdr.setSectionResizeMode(1, hdr.ResizeToContents)
        hdr.setSectionResizeMode(2, hdr.Interactive)
        hdr.setSectionResizeMode(3, hdr.ResizeToContents)
        hdr.setSectionResizeMode(4, hdr.ResizeToContents)
        self.rev_table.setSortingEnabled(True)
        self.rev_table.setWordWrap(False)
        self.rev_table.cellDoubleClicked.connect(
            lambda r, _c: self._open_reviewed_selected()
        )
        layout.addWidget(self.rev_table, 1)

        # ── status bar ─────────────────────────────────────────────────────
        hint_row = QHBoxLayout()
        self.rev_status_lbl = QLabel(
            "Load a doctor to view reviewed reports."
        )
        self.rev_status_lbl.setStyleSheet(
            "color:#7c2d12;font-size:11px;font-weight:500;"
        )
        hint_row.addWidget(self.rev_status_lbl, 1)
        hint_lbl = QLabel("Double-click a row to open URL")
        hint_lbl.setStyleSheet("color:#9a3412;font-size:10px;font-weight:600;")
        hint_row.addWidget(hint_lbl)
        layout.addLayout(hint_row)
        return w

    def _build_json_tab(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(8)

        top = QHBoxLayout()
        top.addWidget(QLabel("Search JSON:"))
        self.json_search_input = QLineEdit()
        self.json_search_input.setPlaceholderText("Patient / file / date...")
        self.json_search_input.textChanged.connect(self._filter_json_table)
        top.addWidget(self.json_search_input, 1)

        refresh_btn = QPushButton("Refresh JSON")
        refresh_btn.setObjectName("btn_secondary")
        refresh_btn.clicked.connect(self._refresh_json_reports)
        top.addWidget(refresh_btn)
        layout.addLayout(top)

        self.json_table = QTableWidget(0, 4)
        self.json_table.setHorizontalHeaderLabels(["Date", "Patient", "Type", "JSON File"])
        self.json_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.json_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.json_table.setAlternatingRowColors(True)
        self.json_table.verticalHeader().setVisible(False)
        self.json_table.setWordWrap(False)
        jhdr = self.json_table.horizontalHeader()
        jhdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        jhdr.setSectionResizeMode(1, QHeaderView.Stretch)
        jhdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        jhdr.setSectionResizeMode(3, QHeaderView.Stretch)
        self.json_table.cellClicked.connect(self._on_json_clicked)
        self.json_table.cellDoubleClicked.connect(lambda _r, _c: self._open_json_in_waveform_analysis())
        layout.addWidget(self.json_table, 1)

        action_row = QHBoxLayout()
        plot_btn = QPushButton("Plot Selected JSON")
        plot_btn.clicked.connect(self._plot_selected_json)
        action_row.addWidget(plot_btn)

        open_btn = QPushButton("Open in Waveform Analysis")
        open_btn.setObjectName("btn_secondary")
        open_btn.clicked.connect(self._open_json_in_waveform_analysis)
        action_row.addWidget(open_btn)
        action_row.addStretch()
        layout.addLayout(action_row)

        wave_group = QGroupBox("  JSON Waveform Preview")
        wave_layout = QVBoxLayout(wave_group)
        wave_layout.setContentsMargins(10, 10, 10, 10)
        self.waveform_panel = WaveformPlotPanel()
        wave_layout.addWidget(self.waveform_panel)
        layout.addWidget(wave_group, 1)
        return w

    def _collect_json_reports(self):
        reports = []
        seen = set()
        for root_dir in (REPORTS_DIR, str(data_file("recordings"))):
            if not os.path.exists(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                for fn in files:
                    if not fn.lower().endswith(".json"):
                        continue
                    if fn.lower() in {"index.json", "upload_log.json"}:
                        continue
                    full_path = os.path.abspath(os.path.join(root, fn))
                    if full_path.lower() in seen:
                        continue
                    seen.add(full_path.lower())
                    try:
                        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
                    except Exception:
                        mtime = datetime.datetime.now()
                    report_type = self._infer_report_type(full_path, "")
                    patient_name = os.path.splitext(fn)[0].replace("_", " ")
                    reports.append({
                        "date": mtime.strftime("%Y-%m-%d %H:%M:%S"),
                        "patient_name": patient_name,
                        "report_type": report_type,
                        "json_file": full_path,
                    })
        reports.sort(key=lambda e: e.get("date", ""), reverse=True)
        return reports

    def _refresh_json_reports(self):
        self.all_json_reports = self._collect_json_reports()
        self._filter_json_table(self.json_search_input.text() if hasattr(self, "json_search_input") else "")

    def _filter_json_table(self, text):
        if not hasattr(self, "json_table"):
            return
        query = (text or "").strip().lower()
        self.json_table.setRowCount(0)
        for entry in self.all_json_reports:
            file_path = entry.get("json_file", "")
            row_text = " ".join([
                str(entry.get("date", "")),
                str(entry.get("patient_name", "")),
                str(entry.get("report_type", "")),
                os.path.basename(file_path),
            ]).lower()
            if query and query not in row_text:
                continue
            row = self.json_table.rowCount()
            self.json_table.insertRow(row)
            vals = [
                entry.get("date", ""),
                entry.get("patient_name", ""),
                entry.get("report_type", ""),
                os.path.basename(file_path),
            ]
            for col, val in enumerate(vals):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter if col != 3 else Qt.AlignLeft | Qt.AlignVCenter)
                self.json_table.setItem(row, col, item)
            if self.json_table.item(row, 3):
                self.json_table.item(row, 3).setData(Qt.UserRole, file_path)
            self.json_table.setRowHeight(row, 26)

    def _selected_json_path(self):
        if not hasattr(self, "json_table"):
            return ""
        row = self.json_table.currentRow()
        if row < 0:
            return ""
        item = self.json_table.item(row, 3)
        if not item:
            return ""
        return str(item.data(Qt.UserRole) or "")

    def _on_json_clicked(self, _row, _col):
        self._plot_selected_json()

    def _plot_selected_json(self):
        json_path = self._selected_json_path()
        if not json_path:
            if hasattr(self, "waveform_panel"):
                self.waveform_panel.clear_plot("Select a JSON report to plot waves")
            return
        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            self.waveform_panel.load_waveforms(json_path, payload)
        except Exception as exc:
            self.waveform_panel.clear_plot(f"Failed to read JSON: {exc}")

    def _open_json_in_waveform_analysis(self):
        json_path = self._selected_json_path()
        if not json_path:
            QMessageBox.information(self, "Waveform Analysis", "Select a JSON row first.")
            return
        if not os.path.exists(json_path):
            QMessageBox.warning(self, "Waveform Analysis", "Selected JSON file was not found.")
            return

        try:
            from dashboard.analysis_window import ECGAnalysisWindow
        except Exception as exc:
            QMessageBox.critical(self, "Waveform Analysis", f"Failed to open analysis window:\n{exc}")
            return

        if not hasattr(self, "_analysis_window") or self._analysis_window is None:
            self._analysis_window = ECGAnalysisWindow(self)
        self._analysis_window.show()
        self._analysis_window.raise_()
        self._analysis_window.activateWindow()
        ok = self._analysis_window.load_external_report(json_path)
        if not ok:
            QMessageBox.warning(self, "Waveform Analysis", "JSON loaded, but waveform data could not be plotted.")

    def _load_reviewed_reports(self):
        """Fetch reviewed reports for the selected doctor from the cloud API."""
        doc_name = self.rev_doctor_combo.currentText().strip()
        if not doc_name or doc_name == "-- Select Doctor --":
            self.rev_status_lbl.setText(
                "Select a doctor from the list to fetch their reviewed reports."
            )
            return

        self.rev_status_lbl.setText(f"Fetching reports for '{doc_name}' from cloud...")
        QApplication.processEvents()
        try:
            # The backend Lambda expects `doctor`, not `doctorName`.
            # Keep both for compatibility with any older deployments.
            params = {
                "doctor": doc_name,
                "doctorName": doc_name,
                "RhythmUltra_serial": _get_RhythmUltra_serial(),
            }
            resp = requests.get(
                PUBLIC_REVIEWED_REPORTS_URL,
                params=params,
                headers=_reviewed_reports_headers(),
                timeout=12,
            )
            if resp.status_code != 200:
                self.rev_status_lbl.setStyleSheet("color:#b91c1c;font-size:11px;font-weight:600;")
                self.rev_status_lbl.setText(
                    f"Cloud unavailable ({resp.status_code}). Reviewed reports could not be loaded."
                )
                from utils.ui_feedback import show_critical
                show_critical(
                    self,
                    "Cloud Unavailable",
                    f"Reviewed reports could not be loaded for {doc_name}.",
                    details=f"HTTP {resp.status_code}: {resp.reason}",
                )
                return
            
            ct = resp.headers.get("Content-Type", "")
            try:
                data = resp.json() if "application/json" in ct or resp.text.strip().startswith("[") else []
            except Exception as json_err:
                self.rev_status_lbl.setStyleSheet("color:#b91c1c;font-size:11px;font-weight:600;")
                self.rev_status_lbl.setText("Cloud data could not be read right now.")
                return
            if not isinstance(data, list):
                data = []

            # Fill table — Date | Time | Patient | Type | Doctor | Presigned URL
            self.rev_table.setRowCount(0)
            self.rev_table.setSortingEnabled(False)
            for entry in data:
                # Try all known URL field names from the API
                purl = (
                    entry.get("preview_url")
                    or entry.get("presigned_url")
                    or entry.get("file_url")
                    or entry.get("fileUrl")
                    or entry.get("url")
                    or ""
                )
                vals = [
                    str(entry.get("date", "")),
                    str(entry.get("time", "")),
                    str(entry.get("patient", entry.get("name", ""))),
                    str(entry.get("report_type", entry.get("type", ""))),
                    str(entry.get("doctor", entry.get("doctorName", doc_name))),
                    purl,
                ]
                r = self.rev_table.rowCount()
                self.rev_table.insertRow(r)
                self.rev_table.setRowHeight(r, 26)
                for c, v in enumerate(vals):
                    item = QTableWidgetItem(v)
                    item.setTextAlignment(
                        Qt.AlignLeft | Qt.AlignVCenter if c == 5 else Qt.AlignCenter
                    )
                    if c == 5:   # URL column
                        item.setData(Qt.UserRole, v)
                        item.setForeground(QColor("#111111"))
                        item.setToolTip(v)   # full presigned URL on hover
                    self.rev_table.setItem(r, c, item)
                    
            self.rev_table.setSortingEnabled(True) 

            if data:
                self.rev_status_lbl.setText(
                    f"{len(data)} reviewed report(s) for {doc_name}. "
                    "Double-click or press 'Open URL' to view."
                )
            else:
                self.rev_status_lbl.setText(
                    f"No reviewed reports found for '{doc_name}'. "
                    "Check the doctor name spelling and try again."
                )
        except requests.exceptions.RequestException as e:
            self.rev_status_lbl.setStyleSheet("color:#b91c1c;font-size:11px;font-weight:600;")
            self.rev_status_lbl.setText(
                "Offline mode: reviewed reports cannot be loaded until the internet is restored."
            )
            from utils.ui_feedback import show_critical, is_network_error, offline_action_message
            show_critical(
                self,
                "No Internet Connection",
                offline_action_message(
                    "Loading reviewed reports",
                    "You can still browse local reports and queued items while offline.",
                ),
            )
        except Exception as e:
            self.rev_status_lbl.setStyleSheet("color:#b91c1c;font-size:11px;font-weight:600;")
            self.rev_status_lbl.setText("Could not load reviewed reports.")
            from utils.ui_feedback import show_critical
            show_critical(
                self,
                "Reviewed Reports Error",
                "Could not load reviewed reports from the cloud.",
                details=str(e),
            )

    def _open_reviewed_selected(self):
        row = self.rev_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Open", "Select a reviewed report row first.")
            return
        item = self.rev_table.item(row, 5)   # URL is now column 5
        url = item.data(Qt.UserRole) if item else ""
        if url:
            webbrowser.open(url)
        else:
            QMessageBox.information(self, "Open",
                                    "No presigned/preview URL found for this entry.")

    def _copy_reviewed_link(self):
        row = self.rev_table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Copy", "Select a row first.")
            return
        item = self.rev_table.item(row, 5)   # URL is column 5
        url = item.data(Qt.UserRole) if item else ""
        if url:
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "Copied",
                                    "Presigned URL copied to clipboard.")
        else:
            QMessageBox.information(self, "Copy", "No URL in selected row.")

    def _build_search_bar(self) -> QFrame:
        frame = QFrame()
        h = QHBoxLayout(frame)
        h.setContentsMargins(8, 4, 8, 4)
        h.setSpacing(10)

        srch_lbl = QLabel("Search by:")
        srch_lbl.setStyleSheet("color:#111111;font-size:12px;")
        self.search_type_combo = QComboBox()
        # Phase-2B: 'All Fields' is now the default (index 0)
        self.search_type_combo.addItems(["All Fields", "Patient Name", "Date Range", "Single Date"])
        self.search_type_combo.currentTextChanged.connect(self._on_search_type_changed)
        self.search_type_combo.setFixedWidth(130)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search name, doctor, diagnosis, date…")
        self.search_input.setMaxLength(30)
        self.search_input.textChanged.connect(self.filter_table)

        self.start_date_edit = QDateEdit()
        self.start_date_edit.setCalendarPopup(True)
        self.start_date_edit.setDate(QDate.currentDate().addDays(-30))
        self.start_date_edit.dateChanged.connect(self.filter_table)
        self.to_label = QLabel("→")
        self.to_label.setStyleSheet("color:#111111;")
        self.end_date_edit = QDateEdit()
        self.end_date_edit.setCalendarPopup(True)
        self.end_date_edit.setDate(QDate.currentDate())
        self.end_date_edit.dateChanged.connect(self.filter_table)

        self.single_date_edit = QDateEdit()
        self.single_date_edit.setCalendarPopup(True)
        self.single_date_edit.setDate(QDate.currentDate())
        self.single_date_edit.dateChanged.connect(self.filter_table)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setFixedHeight(32)
        refresh_btn.clicked.connect(self.load_history)

        for w in (self.start_date_edit, self.to_label, self.end_date_edit):
            w.hide()
        self.single_date_edit.hide()

        h.addWidget(srch_lbl)
        h.addWidget(self.search_type_combo)
        h.addWidget(self.search_input, 1)
        h.addWidget(self.start_date_edit)
        h.addWidget(self.to_label)
        h.addWidget(self.end_date_edit)
        h.addWidget(self.single_date_edit)
        h.addWidget(refresh_btn)
        return frame

    # ── Phase-1: risk severity helpers ──────────────────────────────────────
    _CRITICAL_KEYWORDS = (
        "ventricular tachycardia", "vt", "ventricular fibrillation", "vf",
        "atrial fibrillation", "afib", "af ", "stemi", "complete heart block",
        "third degree", "asystole", "bigeminy", "trigeminy",
        "wide complex", "flutter", "torsade",
    )
    _WARNING_KEYWORDS = (
        "tachycardia", "bradycardia", "pac", "pvc", "lbbb", "rbbb",
        "first degree block", "second degree", "mobitz", "wenckebach",
        "prolonged qt", "qtc", "st elevation", "st depression",
        "ischemia", "infarction", "hypertrophy",
    )
    _NORMAL_KEYWORDS = (
        "normal sinus", "sinus rhythm", "within normal", "no significant",
        "regular rhythm", "normal ecg", "normal study",
    )

    @staticmethod
    def _get_row_risk(entry: dict):
        """Return ('critical'|'warning'|'normal'|'') based on entry findings."""
        # Collect all text fields that may contain findings
        text_parts = []
        for key in ("findings", "interpretation", "conclusion", "rhythm",
                     "arrhythmia", "diagnosis", "report_type"):
            val = entry.get(key)
            if isinstance(val, list):
                text_parts.extend(str(v) for v in val)
            elif isinstance(val, str) and val:
                text_parts.append(val)
        combined = " ".join(text_parts).lower()
        if not combined:
            return ""
        for kw in HistoryWindow._CRITICAL_KEYWORDS:
            if kw in combined:
                return "critical"
        for kw in HistoryWindow._WARNING_KEYWORDS:
            if kw in combined:
                return "warning"
        for kw in HistoryWindow._NORMAL_KEYWORDS:
            if kw in combined:
                return "normal"
        return ""

    @staticmethod
    def _get_findings_summary(entry: dict) -> str:
        """Return a short one-line findings string for the Findings column."""
        findings = entry.get("findings") or entry.get("interpretation") or entry.get("conclusion") or ""
        if isinstance(findings, list):
            # Join first 2 items and cap at 60 chars
            text = "; ".join(str(f) for f in findings[:2] if f)
        else:
            text = str(findings)
        return text[:60] + ("…" if len(text) > 60 else "")

    def _build_table(self, layout):
        self.table = QTableWidget()
        self.table.setColumnCount(9)
        self.table.setHorizontalHeaderLabels([
            "Date", "Time", "Org.", "Doctor", "Patient Name",
            "Org. Name", "Org. Address", "Type", "Status",
        ])
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.table.setWordWrap(False)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(26)
        self.table.verticalHeader().setMinimumSectionSize(26)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        self._configure_table_columns()
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_selection_changed_preview)
        layout.addWidget(self.table, 1)

    def _configure_table_columns(self):
        hh = self.table.horizontalHeader()
        hh.setMinimumSectionSize(52)
        fixed_columns = {
            0: 92,   # Date
            1: 84,   # Time
            2: 120,  # Org.
            3: 110,  # Doctor
            5: 130,  # Org. Name
            6: 150,  # Org. Address
            7: 90,   # Type
            8: 86,   # Status
        }
        for col in range(self.table.columnCount()):
            if col in fixed_columns:
                hh.setSectionResizeMode(col, QHeaderView.Interactive)
                self.table.setColumnWidth(col, fixed_columns[col])
            elif col == 4:
                hh.setSectionResizeMode(col, QHeaderView.Stretch)
            else:
                hh.setSectionResizeMode(col, QHeaderView.Interactive)
        self.table.setColumnWidth(2, 120)
        self.table.setColumnWidth(3, 110)
        self.table.setColumnWidth(4, max(140, self.table.viewport().width() // 5))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        try:
            total_w = max(1, self.width())
            left_w = max(520, int(total_w * 0.62))
            right_w = max(360, total_w - left_w - 40)
            self.splitter.setSizes([left_w, right_w])
            if hasattr(self, "table"):
                self._configure_table_columns()
        except Exception:
            pass

    def _build_action_buttons(self, layout):
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame{background:#fffaf5;border-top:1px solid #fed7aa;"
            "border-radius:0 0 14px 14px;padding:6px 4px;}"
        )
        row = QHBoxLayout(bar)
        row.setContentsMargins(8, 4, 8, 4)
        row.setSpacing(6)

        def btn(label, slot, secondary=False):
            b = QPushButton(label)
            b.setFixedHeight(34)
            if secondary:
                b.setObjectName("btn_secondary")
            b.clicked.connect(slot)
            row.addWidget(b)
            return b

        btn("Preview",        self._preview_selected)
        # btn("Email",           self._send_email)
        btn("Open Report",     self._open_in_system,      secondary=True)
        btn("Send for Review", self.send_report_for_review)
        btn("Export All",      self.export_all_reports,   secondary=True)

        # DICOM export button (disabled gracefully if pydicom not installed)
        try:
            from utils.dicom_exporter import is_available as _dicom_ok
            _dicom_enabled = _dicom_ok()
        except Exception:
            _dicom_enabled = False
        dicom_btn = QPushButton("Export DICOM")
        dicom_btn.setFixedHeight(34)
        dicom_btn.setObjectName("btn_secondary")
        dicom_btn.setToolTip(
            "Export selected report as DICOM ECG file (Ctrl+D)"
            if _dicom_enabled else
            "pydicom not installed — run: pip install pydicom"
        )
        dicom_btn.setEnabled(_dicom_enabled)
        dicom_btn.clicked.connect(self._export_dicom)
        row.addWidget(dicom_btn)

        btn("Export HL7", self._export_hl7, secondary=True)

        row.addStretch()

        layout.addWidget(bar)

    # ── search ──────────────────────────────────────────────────────────────
    def _on_search_type_changed(self, search_type):
        # Phase-2B: 'All Fields' and 'Patient Name' both use the text input
        self.search_input.setVisible(search_type in ("All Fields", "Patient Name"))
        for w in (self.start_date_edit, self.to_label, self.end_date_edit):
            w.setVisible(search_type == "Date Range")
        self.single_date_edit.setVisible(search_type == "Single Date")
        # Update placeholder to help the doctor understand what to type
        if search_type == "All Fields":
            self.search_input.setPlaceholderText("Search name, doctor, diagnosis, date…")
        else:
            self.search_input.setPlaceholderText("Patient name…")
        self.filter_table()

    def filter_table(self):
        search_type = self.search_type_combo.currentText()
        self.table.setRowCount(0)
        for entry in self.all_history_entries:
            show = False
            if search_type == "All Fields":
                # Phase-2B: search across every relevant field
                txt = self.search_input.text().strip().lower()
                if not txt:
                    show = True
                else:
                    # Build a combined string from all searchable fields
                    findings = entry.get("findings", [])
                    findings_str = (" ".join(str(f) for f in findings)
                                    if isinstance(findings, list) else str(findings or ""))
                    haystack = " ".join([
                        str(entry.get("patient_name", "")),
                        str(entry.get("first_name", "")),
                        str(entry.get("last_name", "")),
                        str(entry.get("doctor", "")),
                        str(entry.get("report_type", "")),
                        str(entry.get("date", "")),
                        str(entry.get("org_name", "")),
                        str(entry.get("interpretation", "")),
                        str(entry.get("conclusion", "")),
                        str(entry.get("review_status", "")),
                        findings_str,
                    ]).lower()
                    show = txt in haystack
            elif search_type == "Patient Name":
                txt = self.search_input.text().strip().lower()
                show = txt == "" or txt in entry.get("patient_name", "").lower()
            elif search_type == "Date Range":
                d0 = self.start_date_edit.date().toPyDate()
                d1 = self.end_date_edit.date().toPyDate()
                try:
                    ed = datetime.datetime.strptime(entry.get("date", ""), "%Y-%m-%d").date()
                    show = d0 <= ed <= d1
                except ValueError:
                    show = False
            elif search_type == "Single Date":
                sd = self.single_date_edit.date().toPyDate()
                try:
                    ed = datetime.datetime.strptime(entry.get("date", ""), "%Y-%m-%d").date()
                    show = ed == sd
                except ValueError:
                    show = False
            if show:
                self._add_row(entry)

    def _add_stray_pdf(self, full_path, history_entries):
        """Helper to add a stray PDF to the history list if not already present."""
        if any(e.get("report_file") == full_path for e in history_entries):
            return

        fn = os.path.basename(full_path).lower()
        # Only add files that look like ECG reports
        if not any(x in fn for x in ["ecg", "hrv", "hyper", "holter", "analysis"]):
            return

        rt = "ECG"
        if "holter" in fn: rt = "Comprehensive ECG Analysis"
        elif "hyper" in fn: rt = "Hyperkalemia"
        elif "hrv" in fn: rt = "HRV"
        elif "analysis" in fn: rt = "Analysis"

        try:
            mtime = os.path.getmtime(full_path)
            dt = datetime.datetime.fromtimestamp(mtime)
            
            history_entries.append({
                "date": dt.strftime("%Y-%m-%d"),
                "time": dt.strftime("%H:%M:%S"),
                "report_type": rt,
                "patient_name": os.path.basename(full_path).replace(".pdf", "")
                                 .replace("ECG_Report_", "")
                                 .replace("HRV_Report_", "")
                                 .replace("ECG_Analysis_", "")
                                 .replace("holter_report", "Comprehensive_ECG_Analysis_")
                                 .replace("_", " "),
                "report_file": full_path
            })
        except Exception:
            pass

    # ── data loading ─────────────────────────────────────────────────────────
    def _infer_report_type(self, report_file: str = "", report_type: str = "") -> str:
        rt = (report_type or "").strip()
        if rt:
            return rt
        fl = (report_file or "").lower()
        if "hyper" in fl:
            return "Hyperkalemia"
        if "hrv" in fl:
            return "HRV"
        if "holter" in fl:
            return "Comprehensive ECG Analysis"
        if "analysis" in fl:
            return "Analysis"
        return "ECG"

    def _entry_matches_current_user(self, entry: dict, active_owner_norm: str, active_username_norm: str) -> bool:
        """Match entries by owner name first, then by the signed-in username."""
        if not active_owner_norm and not active_username_norm:
            return True

        entry_owner = _normalize_owner(
            entry.get("owner_full_name")
            or entry.get("full_name")
            or ""
        )
        entry_username = _normalize_owner(entry.get("username") or "")

        if active_owner_norm and entry_owner and entry_owner == active_owner_norm:
            return True
        if active_username_norm and entry_username and entry_username == active_username_norm:
            return True
        return False

    def _add_report_file_entries(self, history_entries):
        """Ensure all PDF reports under reports/ appear in history table."""
        try:
            if not os.path.exists(REPORTS_DIR):
                return
            known = {
                os.path.abspath((e.get("report_file") or "")).lower()
                for e in history_entries if e.get("report_file")
            }
            for root, _dirs, files in os.walk(REPORTS_DIR):
                for fn in files:
                    if not fn.lower().endswith('.pdf'):
                        continue
                    full_path = os.path.abspath(os.path.join(root, fn))
                    if full_path.lower() in known:
                        continue

                    date_str, time_str = "", ""
                    m = re.search(r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})", fn)
                    if m:
                        date_str = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                        time_str = f"{m.group(4)}:{m.group(5)}:{m.group(6)}"
                    else:
                        ts = datetime.datetime.fromtimestamp(os.path.getmtime(full_path))
                        date_str = ts.strftime("%Y-%m-%d")
                        time_str = ts.strftime("%H:%M:%S")

                    history_entries.append({
                        "date": date_str,
                        "time": time_str,
                        "report_type": self._infer_report_type(full_path, ""),
                        "Org.": "",
                        "doctor": "",
                        "patient_name": "",
                        "org_name": "",
                        "org_address": "",
                        "height": "",
                        "weight": "",
                        "report_file": full_path,
                        "username": "",
                    })
                    known.add(full_path.lower())
        except Exception:
            pass

    def load_history(self):
        self.table.setRowCount(0)
        history_entries = []

        if os.path.exists(HISTORY_FILE):
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    history_entries = json.load(f)
                if not isinstance(history_entries, list):
                    history_entries = []
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load history: {e}")

        if os.path.exists(REPORTS_INDEX_FILE):
            try:
                with open(REPORTS_INDEX_FILE, "r", encoding="utf-8") as f:
                    idx = json.load(f)
                if isinstance(idx, list):
                    for entry in idx:
                        fn = entry.get("filename", "").lower()
                        # Better detection logic for report types
                        if "holter" in fn:
                            rt = "Comprehensive ECG Analysis"
                        elif "hyper" in fn:
                            rt = "Hyperkalemia"
                        elif "hrv" in fn:
                            rt = "HRV"
                        elif "analysis" in fn:
                            rt = "Analysis"
                        else:
                            rt = "ECG"
                        
                        history_entries.append({
                            "date": entry.get("date", ""),
                            "time": entry.get("time", ""),
                            "report_type": rt,
                            "Org.": entry.get("org", entry.get("Org.", "")),
                            "doctor": entry.get("doctor", ""),
                            "patient_name": entry.get("patient", entry.get("patient_name", "")),
                            "org_name": entry.get("org_name", entry.get("organisation_name", entry.get("org", ""))),
                            "org_address": entry.get("org_address", entry.get("organisation_address", "")),
                            "height": entry.get("height", ""),
                            "weight": entry.get("weight", ""),
                            "report_file": os.path.join(REPORTS_DIR, entry.get("filename", "")),
                            "username": entry.get("username", ""),
                            "owner_full_name": entry.get("owner_full_name", entry.get("full_name", entry.get("username", ""))),
                        })
            except Exception:
                pass

        # Also scan reports directory for any stray PDFs that might not be in index
        RECORDINGS_DIR = str(data_file("recordings"))
        scan_dirs = [REPORTS_DIR, RECORDINGS_DIR]
        try:
            from PyQt5.QtCore import QStandardPaths
            downloads = QStandardPaths.writableLocation(QStandardPaths.DownloadLocation)
            if downloads and os.path.exists(downloads):
                scan_dirs.append(downloads)
        except Exception:
            pass

        for sdir in scan_dirs:
            if not os.path.exists(sdir): continue
            
            # Recursive scan for Holter reports which are often in subdirectories
            if sdir == RECORDINGS_DIR:
                for root_dir, dirs, files in os.walk(sdir):
                    for f in files:
                        if f.lower().endswith(".pdf") and "holter" in f.lower():
                            self._add_stray_pdf(os.path.join(root_dir, f), history_entries)
            else:
                for f in os.listdir(sdir):
                    if f.lower().endswith(".pdf"):
                        self._add_stray_pdf(os.path.join(sdir, f), history_entries)

        # Fallback
        if not history_entries:
            pf = str(data_file("all_patients.json"))
            if os.path.exists(pf):
                try:
                    with open(pf, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    for p in data.get("patients", []):
                        dt = p.get("date_time", "")
                        ds, ts = ("", "")
                        if dt and " " in dt:
                            ds, ts = dt.split(" ", 1)
                        elif dt:
                            ds = dt
                        pname = p.get("patient_name") or (
                            (p.get("first_name", "") + " " + p.get("last_name", "")).strip()
                        )
                        history_entries.append({
                            "date": ds, "time": ts, "report_type": "ECG",
                            "Org.": p.get("Org.", ""), "doctor": p.get("doctor", ""),
                            "patient_name": pname,
                            "org_name": p.get("org_name", p.get("organisation_name", p.get("org", ""))),
                            "org_address": p.get("org_address", p.get("organisation_address", "")),
                            "height": str(p.get("height", "")),
                            "weight": str(p.get("weight", "")), "report_file": "",
                        })
                except Exception:
                    pass

        # Ensure reports folder PDFs are also visible in history list
        self._add_report_file_entries(history_entries)

        # Filter by owner and normalize report type
        active_owner_norm = _normalize_owner(self.owner_full_name)
        active_username_norm = _normalize_owner(self.username)
        self.all_history_entries = []
        for entry in history_entries:
            if not self._entry_matches_current_user(entry, active_owner_norm, active_username_norm):
                continue
            rf = entry.get("report_file", "") or ""
            rt = entry.get("report_type", "")
            if not rt:
                fl = rf.lower()
                if "holter" in fl: rt = "Comprehensive ECG Analysis"
                elif "hyper" in fl: rt = "Hyperkalemia"
                elif "hrv" in fl: rt = "HRV"
                elif "analysis" in fl: rt = "Analysis"
                else: rt = "ECG"
            
            entry["report_type"] = rt
            self.all_history_entries.append(entry)

        # Sort newest first
        def _key(e):
            try:
                d = tuple(map(int, e.get("date", "0-0-0").split("-")))
                t = tuple(map(int, (e.get("time", "0:0:0") + ":0").split(":")[:3]))
                return (d, t)
            except Exception:
                return ((0, 0, 0), (0, 0, 0))

        self.all_history_entries.sort(key=_key, reverse=True)
        migrated = False
        for entry in self.all_history_entries:
            if not entry.get("hl7_message"):
                entry["hl7_message"] = _history_entry_to_hl7(entry)
                migrated = True
            if not entry.get("hl7_file"):
                entry["hl7_file"] = _write_hl7_history_file(entry)
                migrated = True
        if migrated:
            self._save_history_to_file()
        for entry in self.all_history_entries:
            self._add_row(entry)
        self._refresh_json_reports()

        self.preview_panel.clear()

    def _add_row(self, entry):
        row = self.table.rowCount()
        self.table.insertRow(row)
        status = entry.get("review_status", "Pending")

        # ── Findings summary for inline column ───────────────────────────────
        findings_text = self._get_findings_summary(entry)

        # Profile fallbacks for doctor and clinic details
        doc = self._get_profile_fallback(entry, "doctor")
        org_name = self._get_profile_fallback(entry, "org_name")
        org_addr = self._get_profile_fallback(entry, "org_address")

        values = [
            entry.get("date", ""),
            entry.get("time", ""),
            org_name,
            doc,
            entry.get("patient_name", "") or (
                (str(entry.get("first_name") or "") + " " + str(entry.get("last_name") or "")).strip()),
            org_name,
            org_addr,
            entry.get("report_type", ""),
            status,
        ]

        # ── Determine row risk level for color coding ─────────────────────────
        risk = self._get_row_risk(entry)
        row_bg_colors = {
            "critical": "#fde8e8",   # soft red
            "warning":  "#fff8e1",   # soft amber
            "normal":   "#e8f5e9",   # soft green
        }
        row_bg = row_bg_colors.get(risk, "")

        status_colors = {
            "Pending":     ("#f2f2f2", "#333333"),
            "Under Review":("#e8e8e8", "#111111"),
            "Reviewed":    ("#d9d9d9", "#111111"),
            "Queued":      ("#efefef", "#333333"),
        }

        # ── risk badge prefix for Status cell (col 8) ─────────────────────────
        risk_badge = {"critical": "🔴 ", "warning": "🟡 ", "normal": "🟢 "}.get(risk, "")

        for col, val in enumerate(values):
            # Prepend badge only on Status column (col 8)
            display_val = (risk_badge + val) if (col == 8 and risk_badge) else val
            item = QTableWidgetItem(display_val)
            item.setTextAlignment(
                Qt.AlignLeft | Qt.AlignVCenter if col == 9 else Qt.AlignCenter
            )

            # Apply row background tint
            if row_bg and col != 8:
                item.setBackground(QColor(row_bg))

            # Status column colors
            if col == 8:
                bg, fg = status_colors.get(status, ("#e9ecef", "#6c757d"))
                item.setBackground(QColor(bg))
                item.setForeground(QColor(fg))
                item.setFont(QFont("Segoe UI", 9, QFont.Bold))

            # Findings column tooltip shows full text
            if col == 9 and findings_text:
                full_findings = entry.get("findings") or entry.get("interpretation") or entry.get("conclusion") or ""
                if isinstance(full_findings, list):
                    full_findings = "; ".join(str(f) for f in full_findings if f)
                item.setToolTip(str(full_findings))
                item.setForeground(QColor(
                    "#b91c1c" if risk == "critical" else
                    "#92400e" if risk == "warning" else
                    "#166534" if risk == "normal" else "#374151"
                ))

            self.table.setItem(row, col, item)

        rf = entry.get("report_file", "")
        if self.table.item(row, 0):
            self.table.item(row, 0).setData(Qt.UserRole, entry)
            self.table.item(row, 0).setData(Qt.UserRole + 1, rf)
        self.table.setRowHeight(row, 28)  # slightly taller for readability
        self._configure_table_columns()

    # ── interactions ─────────────────────────────────────────────────────────
    def _on_selection_changed_preview(self):
        """Auto-preview currently selected report row in side panel."""
        row = self.table.currentRow()
        if row >= 0:
            self._load_preview_for_row(row, silent=True)

    def _get_cloud_preview_url(self, row: int) -> str:
        pi = self.table.item(row, 4)
        di = self.table.item(row, 0)
        if not pi or not di:
            return ""
        return self._cloud_preview_map.get((pi.text().strip(), di.text().strip()), "")

    def _download_preview_pdf(self, url: str) -> str:
        if not url:
            return ""
        try:
            r = requests.get(url, timeout=12)
            if r.status_code != 200:
                return ""
            fd, tmp_path = tempfile.mkstemp(prefix="ecg_preview_", suffix=".pdf")
            os.close(fd)
            with open(tmp_path, "wb") as f:
                f.write(r.content)
            self._preview_temp_pdf = tmp_path
            return tmp_path
        except Exception:
            return ""

    def _load_preview_for_row(self, row: int, silent: bool = False) -> bool:
        rf = self._get_report_file(row)
        if rf and os.path.exists(rf):
            self.preview_panel.load_pdf(rf)
            return True

        cloud_url = self._get_cloud_preview_url(row)
        cloud_pdf = self._download_preview_pdf(cloud_url) if cloud_url else ""
        if cloud_pdf and os.path.exists(cloud_pdf):
            self.preview_panel.load_pdf(cloud_pdf)
            return True

        self.preview_panel.clear()
        if not silent:
            QMessageBox.information(self, "Preview", "No preview available for this report yet.")
        return False

    def _on_cell_clicked(self, row, col):
        """Single-click: load selected report in side preview panel."""
        if col == 10:
            self._show_status_menu(row)
        self._load_preview_for_row(row, silent=True)

    def _on_row_double_clicked(self, row, col):
        self._open_report_for_row(row)

    def _get_report_file(self, row) -> str:
        item = self.table.item(row, 0)
        if item:
            data = item.data(Qt.UserRole)
            if isinstance(data, dict):
                path_val = data.get("report_file", "") or ""
            else:
                path_val = data or ""
            rf = self._resolve_report_path(path_val)
            if rf:
                return rf


        pi = self.table.item(row, 4)
        di = self.table.item(row, 0)
        ti = self.table.item(row, 1)
        pname = pi.text().strip() if pi else ""
        date_str = di.text().strip() if di else ""
        time_str = ti.text().strip() if ti else ""
        return self._find_report_file(pname, date_str, time_str) or ""

    def _preview_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Preview", "Select a report row first.")
            return
        self._load_preview_for_row(row, silent=False)

    def _open_report_for_row(self, row: int):
        rf = self._get_report_file(row)
        if rf and os.path.exists(rf):
            self._open_pdf_file(rf)
            return
        cloud_url = self._get_cloud_preview_url(row)
        if cloud_url:
            webbrowser.open(cloud_url)
            return
        QMessageBox.information(self, "Open Report", "No report file or review link is available for this row.")

    def _open_in_system(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Open", "Select a report row first.")
            return
        rf = self._get_report_file(row)
        if rf and os.path.exists(rf):
            self._open_pdf_file(rf)
        else:
            QMessageBox.information(self, "Not Found", "No local PDF found for this entry.")

    # def _send_email(self):
    #     row = self.table.currentRow()
    #     if row < 0:
    #         QMessageBox.information(self, "Send Email", "Select a report row first.")
    #         return
    #     rf = self._get_report_file(row)
    #     if not rf or not os.path.exists(rf):
    #         QMessageBox.warning(self, "Send Email", "No local PDF found for this entry.")
    #         return
    #     pi = self.table.item(row, 4)
    #     pname = pi.text() if pi else ""
    #     dlg = SendEmailDialog(rf, patient_name=pname, parent=self)
    #     dlg.exec_()

    # ── status context menu ───────────────────────────────────────────────
    def _show_status_menu(self, row):
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QCursor
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#fff;border:1px solid #cfcfcf;border-radius:5px;}"
                           "QMenu::item{padding:7px 18px;color:#111;}"
                           "QMenu::item:selected{background:#111;color:#fff;}")
        si = self.table.item(row, 8)
        current = si.text() if si else "Pending"
        acts = {
            menu.addAction("⚪ Pending"): "Pending",
            menu.addAction("🟡 Under Review"): "Under Review",
            menu.addAction("🟢 Reviewed"): "Reviewed",
        }
        for a, s in acts.items():
            if s == current:
                a.setEnabled(False)
        chosen = menu.exec_(QCursor.pos())
        if chosen in acts:
            self._update_review_status(row, acts[chosen])

    def _update_review_status(self, row, new_status):
        from PyQt5.QtGui import QColor
        si = self.table.item(row, 8)
        if not si:
            return
        si.setText(new_status)
        colors = {"Pending": ("#f2f2f2", "#333333"),
                  "Under Review": ("#e8e8e8", "#111111"),
                  "Reviewed": ("#d9d9d9", "#111111")}
        bg, fg = colors.get(new_status, ("#f2f2f2", "#333333"))
        si.setBackground(QColor(bg))
        si.setForeground(QColor(fg))
        pi = self.table.item(row, 4)
        di = self.table.item(row, 0)
        ti = self.table.item(row, 1)
        rfp = self._get_report_file(row) or ""
        pname = pi.text() if pi else ""
        dstr = di.text() if di else ""
        tstr = ti.text() if ti else ""
        owner_norm = _normalize_owner(self.owner_full_name)
        active_username_norm = _normalize_owner(self.username)
        for entry in self.all_history_entries:
            same_owner = self._entry_matches_current_user(entry, owner_norm, active_username_norm)
            same_report = False
            try:
                if rfp and entry.get("report_file"):
                    same_report = os.path.abspath(entry.get("report_file")).lower() == os.path.abspath(rfp).lower()
            except Exception:
                same_report = False

            if same_owner and (same_report or (
                entry.get("patient_name") == pname and entry.get("date") == dstr and entry.get("time") == tstr
            )):
                entry["review_status"] = new_status
                entry["review_updated_at"] = datetime.datetime.now().isoformat()
                entry["review_updated_by"] = self.username or "unknown"
                break
        self._save_history_to_file()

    # ── file helpers ─────────────────────────────────────────────────────────
    def _resolve_report_path(self, report_path) -> str:
        if isinstance(report_path, dict):
            report_path = report_path.get("report_file", "") or ""
        candidate = str(report_path or "").strip()
        if not candidate:
            return ""


        normalized = os.path.abspath(candidate)
        if os.path.exists(normalized):
            return normalized

        base_name = os.path.basename(candidate)
        for root_dir in (REPORTS_DIR, str(data_file("recordings"))):
            if not os.path.exists(root_dir):
                continue
            for root, _dirs, files in os.walk(root_dir):
                for fn in files:
                    if fn.lower() == base_name.lower():
                        found = os.path.join(root, fn)
                        if os.path.exists(found):
                            return found
        return ""

    def _find_report_file(self, patient_name, date_str="", time_str="") -> str:
        if not os.path.exists(REPORTS_DIR):
            return ""

        pdf_paths = []
        for root, _dirs, files in os.walk(REPORTS_DIR):
            for pdf in files:
                if pdf.lower().endswith(".pdf"):
                    pdf_paths.append(os.path.join(root, pdf))

        pclean = patient_name.replace(" ", "_").replace(",", "").upper()
        for pdf_path in pdf_paths:
            pdf_name = os.path.basename(pdf_path)
            if pclean and pclean in pdf_name.upper():
                return pdf_path

        if date_str and time_str:
            try:
                row_dt = datetime.datetime.strptime(
                    f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S"
                )
                best_match = None
                best_delta = None
                for pdf_path in pdf_paths:
                    match = re.search(
                        r"(20\d{2})(\d{2})(\d{2})[_-]?(\d{2})(\d{2})(\d{2})",
                        os.path.basename(pdf_path),
                    )
                    if not match:
                        continue
                    pdf_dt = datetime.datetime(
                        int(match.group(1)),
                        int(match.group(2)),
                        int(match.group(3)),
                        int(match.group(4)),
                        int(match.group(5)),
                        int(match.group(6)),
                    )
                    delta = abs((pdf_dt - row_dt).total_seconds())
                    if best_delta is None or delta < best_delta:
                        best_delta = delta
                        best_match = pdf_path
                if best_match and best_delta is not None and best_delta <= 5:
                    return best_match
            except Exception:
                pass

        if date_str:
            try:
                dp = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
                for pdf_path in pdf_paths:
                    if dp in os.path.basename(pdf_path):
                        return pdf_path
            except Exception:
                pass

        ecg = [f for f in pdf_paths if os.path.basename(f).startswith("ECG_Report_")]
        if ecg:
            ecg.sort(key=lambda f: os.path.getmtime(f), reverse=True)
            return ecg[0]
        return ""

    def _open_pdf_file(self, path):
        try:
            from utils.platform_compat import open_file
            open_file(path)
        except Exception as e:
            QMessageBox.critical(self, "Open Report", f"Failed: {e}")

    # ── cloud review ─────────────────────────────────────────────────────────
    def send_report_for_review(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Send for Review", "Select a report row first.")
            return
        rf = self._get_report_file(row)
        if not rf or not os.path.exists(rf):
            QMessageBox.warning(self, "Send for Review", "Report file not found.")
            return
        try:
            uploader = get_cloud_uploader()
            doctors = uploader.get_available_doctors()
            if not doctors:
                QMessageBox.warning(self, "Error", "No Internet Connection is availabe. Could not fetch available doctors list.")
                return
            pi = self.table.item(row, 3)
            current_doc = pi.text() if pi else ""
            doctor_name = self._select_doctor_from_list(doctors, current_doc)
            if not doctor_name:
                return
            reply = QMessageBox.question(self, "Send for Review",
                                         f"Send report to {doctor_name}?",
                                         QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            self.progress_dialog = QProgressDialog(f"Uploading to {doctor_name}…", "Cancel", 0, 0, self)
            self.progress_dialog.setWindowModality(Qt.WindowModal)
            self.progress_dialog.setMinimumDuration(0)
            self.progress_dialog.show()
            rd = self._get_report_data_from_row(row)
            self.upload_worker = UploadWorker(uploader, rf, doctor_name, metadata=rd)
            self.upload_worker.finished.connect(lambda res: self._on_upload_finished(res, row, doctor_name))
            self.upload_worker.error.connect(self._on_upload_error)
            self.upload_worker.start()
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def _on_upload_finished(self, result, row, doctor_name):
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()
        if result.get("status") == "success":
            QMessageBox.information(self, "Sent", f"Report sent to {doctor_name}!\n{result.get('message')}")
            self._update_review_status(row, "Under Review")
        elif result.get("status") == "queued":
            QMessageBox.information(self, "Queued", f"Offline: {result.get('message')}")
            self._update_review_status(row, "Queued")
        else:
            QMessageBox.warning(self, "Failed", f"Failed.\n{result.get('message')}")

    def _on_upload_error(self, err):
        if hasattr(self, "progress_dialog"):
            self.progress_dialog.close()
        QMessageBox.critical(self, "Error", f"Upload error: {err}")

    def refresh_reviewed_reports(self):
        try:
            params = {"RhythmUltra_serial": _get_RhythmUltra_serial()}
            resp = requests.get(
                PUBLIC_REVIEWED_REPORTS_URL,
                params=params,
                headers=_reviewed_reports_headers(),
                timeout=10,
            )
            if resp.status_code != 200:
                QMessageBox.information(self, "Cloud", "Could not fetch reviewed reports.")
                return
            data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else []
            if not isinstance(data, list):
                data = []
            def norm(s): return str(s or "").strip().lower()
            lookup = {}
            for e in data:
                pn = norm(e.get("patient", e.get("name", "")))
                dt = str(e.get("date", "")).strip()
                url = e.get("preview_url") or e.get("file_url") or e.get("url")
                if pn or dt:
                    lookup[(pn, dt)] = url
            updated = 0
            for row in range(self.table.rowCount()):
                pi = self.table.item(row, 4)
                di = self.table.item(row, 0)
                if not pi or not di:
                    continue
                key = (norm(pi.text()), di.text().strip())
                url = lookup.get(key)
                if url:
                    self._update_review_status(row, "Reviewed")
                    self._cloud_preview_map[(pi.text().strip(), di.text().strip())] = url
                    updated += 1
            QMessageBox.information(self, "Cloud Status", f"Updated {updated} row(s) to Reviewed.")
        except Exception as e:
            QMessageBox.warning(self, "Cloud Status", f"Error: {e}")

    def export_all_reports(self):
        export_dir = QFileDialog.getExistingDirectory(self, "Export All Reports",
                                                      os.path.expanduser("~/Desktop"))
        if not export_dir:
            return
        try:
            pdf_paths = []
            for entry in self.all_history_entries:
                rp = self._resolve_report_path(entry.get("report_file", ""))
                if rp and os.path.exists(rp) and rp.lower().endswith(".pdf"):
                    pdf_paths.append(rp)

            # De-duplicate
            pdf_paths = list(dict.fromkeys([os.path.abspath(p) for p in pdf_paths]))

            if not pdf_paths:
                QMessageBox.information(self, "Export", "No PDF reports found.")
                return
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(export_dir, f"ECG_Reports_Export_{ts}")
            os.makedirs(dest, exist_ok=True)
            ok, fail = 0, 0
            for pdf_path in pdf_paths:
                try:
                    shutil.copy2(pdf_path, os.path.join(dest, os.path.basename(pdf_path)))
                    ok += 1
                except Exception:
                    fail += 1
            QMessageBox.information(self, "Export Done",
                                    f"Exported {ok} PDF(s) to:\n{dest}" +
                                    (f"\n{fail} failed." if fail else ""))
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", str(e))

    # ── helpers ───────────────────────────────────────────────────────────────
    def _get_report_data_from_row(self, row) -> dict:
        cols = ["date", "time", "organization", "doctor", "patient_name",
                "org_name", "org_address", "report_type", "review_status"]
        data = {}
        for i, key in enumerate(cols):
            item = self.table.item(row, i)
            data[key] = item.text() if item else ""
        data["report_file_path"] = self._get_report_file(row)
        data["username"] = self.username
        return data

    def _select_doctor_from_list(self, doctors, current="") -> str:
        dlg = QDialog(self)
        dlg.setWindowTitle("Select Doctor")
        dlg.setMinimumSize(320, 420)
        dlg.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        dlg.setStyleSheet(
            "QDialog{background:#ffffff;font-family:'Segoe UI',Arial,sans-serif;}"
            "QLabel{color:#111111;font-weight:600;}"
            "QLineEdit{border:1px solid #cfcfcf;border-radius:5px;"
            "  padding:7px 10px;background:#fff;color:#111111;}"
            "QLineEdit:focus{border-color:#111111;}"
            "QPushButton{background:#111111;color:#fff;border:none;"
            "  border-radius:5px;padding:8px 20px;font-weight:600;}"
            "QPushButton:hover{background:#000000;}"
            "QPushButton#cancel{background:#fff;color:#111111;"
            "  border:1px solid #111111;}"
            "QPushButton#cancel:hover{background:#f2f2f2;}"
            "QListWidget{border:1px solid #d5d5d5;border-radius:5px;background:#fff;}"
            "QListWidget::item{padding:8px 12px;color:#111111;}"
            "QListWidget::item:selected{background:#111111;color:#fff;}"
            "QListWidget::item:hover:!selected{background:#f2f2f2;}"
            "QScrollBar:vertical{background:#f5f5f5;width:10px;border-radius:5px;}"
            "QScrollBar::handle:vertical{background:#9b9b9b;border-radius:5px;min-height:20px;}"
            "QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}"
        )
        v = QVBoxLayout(dlg)
        v.setContentsMargins(16, 16, 16, 16)
        v.setSpacing(10)

        # Header
        hdr = QLabel("Select Reviewing Doctor")
        hdr.setStyleSheet("font-size:15px;font-weight:700;color:#111111;")
        v.addWidget(hdr)

        sub = QLabel("Choose the doctor to receive this ECG report:")
        sub.setStyleSheet("color:#555555;font-weight:400;font-size:12px;")
        v.addWidget(sub)

        # Filter box  (no emoji — avoids blank rendering on some platforms)
        box = QLineEdit()
        box.setPlaceholderText("Filter doctors...")
        box.setFixedHeight(36)
        v.addWidget(box)

        # Scrollable doctor list — plain text, no emoji prefix
        lw = QListWidget()
        lw.setSelectionMode(QAbstractItemView.SingleSelection)
        lw.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        lw.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        for doc in doctors:
            item = QListWidgetItem(doc)      # plain name, no emoji
            item.setData(Qt.UserRole, doc)
            lw.addItem(item)
            if doc == current:
                item.setSelected(True)
                lw.setCurrentItem(item)
        v.addWidget(lw, 1)

        # Live filter
        box.textChanged.connect(lambda t: [
            lw.item(i).setHidden(t.lower() not in lw.item(i).text().lower())
            for i in range(lw.count())
        ])

        # Buttons
        btns = QHBoxLayout()
        ok_b = QPushButton("Select")
        ca_b = QPushButton("Cancel")
        ca_b.setObjectName("cancel")
        ok_b.setMinimumHeight(36)
        ca_b.setMinimumHeight(36)
        btns.addStretch()
        btns.addWidget(ok_b)
        btns.addWidget(ca_b)
        v.addLayout(btns)

        result = [None]

        def accept():
            cur = lw.currentItem()
            if cur:
                result[0] = cur.data(Qt.UserRole)
                dlg.accept()
            else:
                QMessageBox.warning(dlg, "Select", "Please select a doctor.")

        ok_b.clicked.connect(accept)
        ca_b.clicked.connect(dlg.reject)
        lw.itemDoubleClicked.connect(accept)
        if dlg.exec_() == QDialog.Accepted:
            return result[0]
        return None

    def _save_history_to_file(self):
        try:
            all_e = []
            if os.path.exists(HISTORY_FILE):
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    all_e = json.load(f)
            if not isinstance(all_e, list):
                all_e = []

            owner_norm = _normalize_owner(self.owner_full_name)
            for entry in self.all_history_entries:
                if not entry.get("hl7_message"):
                    entry["hl7_message"] = _history_entry_to_hl7(entry)
                if not entry.get("hl7_file"):
                    entry["hl7_file"] = _write_hl7_history_file(entry)
                pn = entry.get("patient_name", "")
                ds = entry.get("date", "")
                ts = entry.get("time", "")
                rf = entry.get("report_file", "") or ""
                found = False
                for se in all_e:
                    se_owner_norm = _normalize_owner(se.get("owner_full_name") or se.get("full_name") or "")
                    se_username_norm = _normalize_owner(se.get("username") or "")
                    if owner_norm and se_owner_norm and se_owner_norm == owner_norm:
                        pass
                    elif _normalize_owner(self.username) and se_username_norm and se_username_norm == _normalize_owner(self.username):
                        pass
                    elif owner_norm:
                        continue

                    same_report = False
                    try:
                        if rf and se.get("report_file"):
                            same_report = os.path.abspath(se.get("report_file")).lower() == os.path.abspath(rf).lower()
                    except Exception:
                        same_report = False

                    if same_report or (se.get("patient_name") == pn and se.get("date") == ds and se.get("time") == ts):
                        se["review_status"] = entry.get("review_status", "Pending")
                        se["review_updated_at"] = entry.get("review_updated_at", "")
                        se["review_updated_by"] = entry.get("review_updated_by", "")
                        found = True
                        break
                if not found:
                    all_e.append(entry)
            with open(HISTORY_FILE, "w", encoding="utf-8") as f:
                json.dump(all_e, f, indent=2)
        except Exception as e:
            print(f"Error saving history: {e}")

    def _prefetch_doctors(self):
        """Pre-warm the doctor list so it's ready when the Reviewed tab opens."""
        try:
            uploader = get_cloud_uploader()
            docs = uploader.get_available_doctors() or []
            if docs:
                from PyQt5.QtCore import QTimer
                import functools
                QTimer.singleShot(0, functools.partial(self._fill_doctor_combo, docs))
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
#  Reviewed Reports Dialog (unchanged from original)
# ══════════════════════════════════════════════════════════════════════════════
class ReviewedReportsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reviewed Reports")
        self.setMinimumSize(800, 500)
        v = QVBoxLayout(self)
        h = QHBoxLayout()
        self.doctor_combo = QComboBox()
        self.refresh_btn = QPushButton("Refresh")
        self.open_btn = QPushButton("Open Selected")
        self.copy_btn = QPushButton("Copy Link")
        for w in (QLabel("Doctor"), self.doctor_combo,
                  self.refresh_btn, self.open_btn, self.copy_btn):
            h.addWidget(w)
        v.addLayout(h)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Date", "Time", "Patient", "Type", "Filename", "URL"])
        self.table.horizontalHeader().setStretchLastSection(True)
        v.addWidget(self.table)
        try:
            doctors = get_cloud_uploader().get_available_doctors()
        except Exception:
            doctors = []
        self.doctor_combo.addItems(doctors or [])
        self.refresh_btn.clicked.connect(self.refresh_list)
        self.open_btn.clicked.connect(self.open_selected)
        self.copy_btn.clicked.connect(self.copy_selected)
        if self.doctor_combo.count() > 0:
            self.refresh_list()

    def refresh_list(self):
        dname = self.doctor_combo.currentText().strip()
        rows = []
        try:
            params = {
                "doctor": dname,
                "doctorName": dname,
                "RhythmUltra_serial": _get_RhythmUltra_serial(),
            } if dname else {"RhythmUltra_serial": _get_RhythmUltra_serial()}
            resp = requests.get(
                PUBLIC_REVIEWED_REPORTS_URL,
                params=params,
                headers=_reviewed_reports_headers(),
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json() if "application/json" in resp.headers.get("Content-Type", "") else []
                if isinstance(data, list):
                    for e in data:
                        rows.append((str(e.get("date", "")), str(e.get("time", "")),
                                     str(e.get("patient", e.get("name", ""))),
                                     str(e.get("report_type", "")), str(e.get("filename", "")),
                                     e.get("preview_url") or e.get("file_url") or ""))
        except Exception:
            pass
        self.table.setRowCount(0)
        for r in rows:
            i = self.table.rowCount()
            self.table.insertRow(i)
            for c, val in enumerate(r):
                item = QTableWidgetItem(val)
                if c == 5:
                    item.setData(Qt.UserRole, val)
                self.table.setItem(i, c, item)

    def open_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 5)
        url = item.data(Qt.UserRole) if item else ""
        if url:
            webbrowser.open(url)

    def copy_selected(self):
        row = self.table.currentRow()
        if row < 0:
            return
        item = self.table.item(row, 5)
        url = item.data(Qt.UserRole) if item else ""
        if url:
            QApplication.clipboard().setText(url)
            QMessageBox.information(self, "Copy", "Link copied.")


# ══════════════════════════════════════════════════════════════════════════════
#  Module-level helper (called by report generators)
# ══════════════════════════════════════════════════════════════════════════════
def append_history_entry(patient_details, report_file_path, report_type="ECG", username=None, owner_full_name=None):
    """Append a new history entry when a report is generated."""
    try:
        entries = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                entries = json.load(f)
        if not isinstance(entries, list):
            entries = []
    except Exception:
        entries = []

    now = datetime.datetime.now()
    base = {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "report_type": report_type,
        "username": username,
        "owner_full_name": owner_full_name or username or "",
        "report_file": os.path.abspath(report_file_path) if report_file_path else "",
        "review_status": "Pending",
        "review_updated_at": "",
        "review_updated_by": "",
    }
    if isinstance(patient_details, dict):
        base.update(patient_details)
        if not base.get("org_name"):
            base["org_name"] = (
                base.get("Org. Name")
                or base.get("organisation_name")
                or base.get("Org.")
                or ""
            )
        if not base.get("org_address"):
            base["org_address"] = (
                base.get("Org. Address")
                or base.get("organisation_address")
                or ""
            )

    # Auto-fill doctor and organization profile from users.json if blank
    try:
        from dashboard.doctor_profile_dialog import _load_users_db, _find_user_key_and_record
        users_db = _load_users_db()
        _, user_rec = _find_user_key_and_record(users_db, username or "")
        if user_rec:
            if not base.get("doctor"):
                base["doctor"] = user_rec.get("doctor") or user_rec.get("doctor_name") or user_rec.get("full_name") or owner_full_name or username or ""
            if not base.get("doctor_name"):
                base["doctor_name"] = base.get("doctor")
            if not base.get("org_name"):
                base["org_name"] = user_rec.get("org_name") or user_rec.get("Org. Name") or user_rec.get("Org.") or ""
            if not base.get("Org."):
                base["Org."] = base.get("org_name")
            if not base.get("org_address"):
                base["org_address"] = user_rec.get("org_address") or user_rec.get("Org. Address") or ""
    except Exception:
        pass

    # Preserve the full structured record and emit an HL7 sidecar so the history
    # archive can be exchanged with other systems later without losing context.
    base["hl7_format"] = f"ORU^R01|{HL7_VERSION}"
    base["hl7_generated_at"] = now.isoformat()
    base["hl7_message"] = _history_entry_to_hl7(base)
    base["hl7_file"] = _write_hl7_history_file(base)

    entries.append(base)

    try:
        hdir = os.path.dirname(HISTORY_FILE)
        if hdir and not os.path.exists(hdir):
            os.makedirs(hdir, exist_ok=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2)
        print(f"💾 History entry saved: {base.get('patient_name','')} — {report_type}")
    except Exception as e:
        print(f"⚠️ Failed to save history entry: {e}")
