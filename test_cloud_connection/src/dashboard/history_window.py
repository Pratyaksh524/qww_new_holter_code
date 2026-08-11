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
    QListWidgetItem, QAbstractItemView, QGroupBox, QStackedWidget,
)
import sys, os, json, datetime, shutil, smtplib, traceback, tempfile
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email.mime.text import MIMEText
from email import encoders
import requests
import webbrowser

try:
    from utils.cloud_uploader import get_cloud_uploader
except ImportError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from utils.cloud_uploader import get_cloud_uploader

from PyQt5.QtCore import Qt, QDate, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QFont, QPixmap, QColor, QImage

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HISTORY_FILE = os.path.join(BASE_DIR, "ecg_history.json")
ECG_DATA_FILE = os.path.join(BASE_DIR, "ecg_data.txt")
REPORTS_INDEX_FILE = os.path.join(BASE_DIR, "reports", "index.json")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
BACKEND_API_URL = "https://your-backend-api.com/api/reports"
API_TIMEOUT = 30
PUBLIC_REVIEWED_REPORTS_URL = "https://6jhix49qt6.execute-api.us-east-1.amazonaws.com/api/public/reviewed-reports"

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

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        # ── toolbar ────────────────────────────────────────────────────────
        bar = QHBoxLayout()
        self.prev_btn = QPushButton("◀ Prev")
        self.prev_btn.setFixedWidth(70)
        self.prev_btn.clicked.connect(self._prev_page)

        self.page_label = QLabel("No report")
        self.page_label.setAlignment(Qt.AlignCenter)
        self.page_label.setStyleSheet("font-weight:700;color:#111;")

        self.next_btn = QPushButton("Next ▶")
        self.next_btn.setFixedWidth(70)
        self.next_btn.clicked.connect(self._next_page)

        self.zoom_in = QPushButton("＋")
        self.zoom_in.setFixedWidth(32)
        self.zoom_in.clicked.connect(lambda: self._set_zoom(self._zoom + 0.2))

        self.zoom_out = QPushButton("－")
        self.zoom_out.setFixedWidth(32)
        self.zoom_out.clicked.connect(lambda: self._set_zoom(max(0.4, self._zoom - 0.2)))

        for w in (self.prev_btn, self.page_label, self.next_btn, self.zoom_out, self.zoom_in):
            bar.addWidget(w)
        root.addLayout(bar)

        # ── scroll area with page image ────────────────────────────────────
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setAlignment(Qt.AlignCenter)
        self.scroll.setStyleSheet("background:#f4f4f4;border:1px solid #d9d9d9;")

        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scroll.setWidget(self.img_label)
        root.addWidget(self.scroll, 1)

        self._update_nav()

    # ── public ─────────────────────────────────────────────────────────────
    def load_pdf(self, path: str):
        self._pdf_path = path
        self._page_index = 0
        if _HAS_FITZ and path and os.path.exists(path):
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
        self.img_label.setPixmap(QPixmap())
        self.img_label.setText("Select a report to preview")
        self.img_label.setStyleSheet("color:#666;font-size:15px;")
        self._update_nav()

    # ── private ────────────────────────────────────────────────────────────
    def _render(self):
        if not self._pdf_path or self._total_pages == 0:
            self.clear()
            return
        px = _pdf_page_to_pixmap(self._pdf_path, self._page_index, self._zoom)
        if px.isNull():
            if not _HAS_FITZ:
                self.img_label.setText("Install pymupdf for in-app preview\npip install pymupdf")
            else:
                self.img_label.setText("Could not render page.")
            self.img_label.setStyleSheet("color:#555;font-size:13px;")
        else:
            self.img_label.setPixmap(px)
            self.img_label.setStyleSheet("")
        self._update_nav()

    def _prev_page(self):
        if self._page_index > 0:
            self._page_index -= 1
            self._render()

    def _next_page(self):
        if self._page_index < self._total_pages - 1:
            self._page_index += 1
            self._render()

    def _set_zoom(self, z):
        self._zoom = z
        self._render()

    def _update_nav(self):
        has = self._total_pages > 0
        self.prev_btn.setEnabled(has and self._page_index > 0)
        self.next_btn.setEnabled(has and self._page_index < self._total_pages - 1)
        if has:
            self.page_label.setText(f"Page {self._page_index+1} / {self._total_pages}")
        else:
            self.page_label.setText("No report loaded")


# ══════════════════════════════════════════════════════════════════════════════
#  Email-Send Dialog
# ══════════════════════════════════════════════════════════════════════════════
class SendEmailDialog(QDialog):
    """Send ECG report PDF via email using SMTP."""

    def __init__(self, report_path: str, patient_name: str = "", parent=None):
        super().__init__(parent)
        self.report_path = report_path
        self.setWindowTitle("Send Report by Email")
        self.setMinimumSize(500, 480)
        self.setStyleSheet("""
            QDialog{background:#ffffff;}
            QLabel{font-weight:bold;color:#111111;}
            QLineEdit,QTextEdit{border:1px solid #ced4da;border-radius:5px;padding:6px;background:#fff;}
            QPushButton{border-radius:5px;padding:8px 18px;font-weight:bold;color:#fff;background:#111111;border:none;}
            QPushButton:hover{background:#000000;}
            QPushButton#cancel{background:#444444;}
            QPushButton#cancel:hover{background:#222222;}
        """)

        layout = QVBoxLayout(self)

        title = QLabel("📧  Send ECG Report by Email")
        title.setStyleSheet("font-size:15px;font-weight:bold;color:#111111;margin-bottom:8px;")
        layout.addWidget(title)

        form = QFormLayout()
        form.setSpacing(10)

        self.smtp_host = QLineEdit("smtp.gmail.com")
        self.smtp_port = QLineEdit("587")
        self.smtp_user = QLineEdit()
        self.smtp_user.setPlaceholderText("sender@example.com")
        self.smtp_pass = QLineEdit()
        self.smtp_pass.setEchoMode(QLineEdit.Password)
        self.smtp_pass.setPlaceholderText("App password / SMTP password")
        self.to_field = QLineEdit()
        self.to_field.setPlaceholderText("recipient@example.com")
        self.cc_field = QLineEdit()
        self.cc_field.setPlaceholderText("Optional CC addresses, comma-separated")
        self.subject = QLineEdit(f"ECG Report — {patient_name}")
        self.body = QTextEdit()
        self.body.setPlainText(
            f"Dear Doctor,\n\nPlease find attached the ECG report for patient: {patient_name}.\n\n"
            f"Regards,\nCardioX"
        )
        self.body.setFixedHeight(100)

        form.addRow("SMTP Host:", self.smtp_host)
        form.addRow("Port:", self.smtp_port)
        form.addRow("Your Email:", self.smtp_user)
        form.addRow("Password:", self.smtp_pass)
        form.addRow("To:", self.to_field)
        form.addRow("CC:", self.cc_field)
        form.addRow("Subject:", self.subject)
        form.addRow("Body:", self.body)
        layout.addLayout(form)

        # Attachment label
        fname = os.path.basename(report_path) if report_path else "—"
        att_lbl = QLabel(f"📎 Attachment: {fname}")
        att_lbl.setStyleSheet("color:#333333;font-weight:normal;margin-top:4px;")
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
        if not self.smtp_user.text().strip():
            QMessageBox.warning(self, "Missing", "Please enter your email address.")
            return
        if not os.path.exists(self.report_path):
            QMessageBox.warning(self, "File Missing", "Report PDF not found.")
            return

        # Build message
        msg = MIMEMultipart()
        msg["From"] = self.smtp_user.text().strip()
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
            port = int(self.smtp_port.text().strip() or "587")
            server = smtplib.SMTP(self.smtp_host.text().strip(), port, timeout=15)
            server.starttls()
            server.login(self.smtp_user.text().strip(), self.smtp_pass.text())
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
class HistoryWindow(QDialog):
    """ECG Report History — split pane: report list (left) + PDF preview (right)."""

    # ── black/white clean theme ─────────────────────────────────────────────
    STYLE = """
        QDialog{
            background:#ffffff;
            color:#111111;
            font-family:'Segoe UI',Helvetica,Arial,sans-serif;
        }
        QTableWidget{
            border:1px solid #d9d9d9;
            background:#ffffff;
            gridline-color:#ececec;
            selection-background-color:#111111;
            selection-color:#ffffff;
            alternate-background-color:#fafafa;
        }
        QTableWidget::item{
            padding:6px 8px;
            border-bottom:1px solid #efefef;
            color:#111111;
        }
        QHeaderView::section{
            background:#111111;
            color:#ffffff;
            font-weight:700;
            font-size:12px;
            padding:8px 6px;
            border:none;
            border-right:1px solid #2a2a2a;
        }
        QPushButton{
            background:#111111;
            color:#ffffff;
            border:1px solid #111111;
            border-radius:6px;
            padding:7px 14px;
            font-weight:600;
            font-size:12px;
        }
        QPushButton:hover{background:#000000;}
        QPushButton:pressed{background:#2a2a2a;}
        QPushButton#btn_secondary,
        QPushButton#btn_close{
            background:#ffffff;
            color:#111111;
            border:1px solid #111111;
        }
        QPushButton#btn_secondary:hover,
        QPushButton#btn_close:hover{background:#f3f3f3;}
        QLineEdit,QComboBox,QDateEdit{
            border:1px solid #cfcfcf;
            border-radius:6px;
            padding:6px 10px;
            background:#ffffff;
            color:#111111;
            font-size:13px;
        }
        QLineEdit:focus,QComboBox:focus,QDateEdit:focus{border-color:#111111;}
        QComboBox::drop-down{border:none;width:22px;}
        QLabel{color:#111111;font-weight:600;}
        QGroupBox{
            border:1px solid #d5d5d5;
            border-radius:8px;
            background:#ffffff;
            margin-top:12px;
            padding:10px 8px 8px 8px;
            font-weight:bold;
        }
        QGroupBox::title{
            color:#111111;
            font-weight:700;
            subcontrol-origin:margin;
            left:12px;
            padding:0 4px;
        }
        QSplitter::handle{background:#d0d0d0;width:2px;}
        QScrollBar:vertical{background:#f5f5f5;width:10px;border-radius:5px;}
        QScrollBar::handle:vertical{background:#9b9b9b;border-radius:5px;min-height:20px;}
        QScrollBar::handle:vertical:hover{background:#7d7d7d;}
        QScrollBar::add-line:vertical,QScrollBar::sub-line:vertical{height:0;}
        QTabBar::tab{
            background:#f3f3f3;
            color:#111111;
            border:1px solid #d5d5d5;
            border-bottom:none;
            border-radius:5px 5px 0 0;
            padding:7px 18px;
            font-weight:600;
            font-size:12px;
            margin-right:2px;
            min-width:100px;
        }
        QTabBar::tab:selected{background:#111111;color:#ffffff;}
        QTabBar::tab:hover:!selected{background:#e8e8e8;}
        QTabWidget::pane{border:1px solid #d5d5d5;border-radius:0 6px 6px 6px;background:#ffffff;}
        QListWidget{border:1px solid #d5d5d5;border-radius:5px;background:#ffffff;}
        QListWidget::item{padding:6px 10px;color:#111111;}
        QListWidget::item:selected{background:#111111;color:#ffffff;}
        QListWidget::item:hover:!selected{background:#f2f2f2;}
    """

    def __init__(self, parent=None, username=None):
        super().__init__(parent)
        self.setWindowTitle("ECG Report History")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.username = username
        self.all_history_entries = []
        self._cloud_preview_map = {}
        self._preview_temp_pdf = ""
        self.setStyleSheet(self.STYLE)

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

    # ── UI construction ─────────────────────────────────────────────────────
    def _build_ui(self):
        from PyQt5.QtWidgets import QTabWidget
        root = QVBoxLayout(self)
        root.setContentsMargins(12, 10, 12, 10)
        root.setSpacing(0)

        # ── Header bar ─────────────────────────────────────────────────────
        header = QFrame()
        header.setFixedHeight(54)
        header.setStyleSheet(
            "QFrame{background:#111111;border-radius:8px 8px 0 0;}"
        )
        hh = QHBoxLayout(header)
        hh.setContentsMargins(18, 0, 18, 0)
        logo = QLabel("⚡")
        logo.setStyleSheet("font-size:20px;color:#fff;")
        title_lbl = QLabel("ECG Report History")
        title_lbl.setStyleSheet(
            "font-size:18px;font-weight:700;color:#fff;letter-spacing:0.5px;"
        )
        sub_lbl = QLabel("View, preview and send ECG reports")
        sub_lbl.setStyleSheet("font-size:11px;color:#d9d9d9;font-weight:400;")
        txt_col = QVBoxLayout()
        txt_col.setSpacing(1)
        txt_col.addWidget(title_lbl)
        txt_col.addWidget(sub_lbl)
        hh.addWidget(logo)
        hh.addSpacing(10)
        hh.addLayout(txt_col)
        hh.addStretch()
        close_btn = QPushButton("✕  Close")
        close_btn.setObjectName("btn_close")
        close_btn.setFixedSize(90, 32)
        close_btn.clicked.connect(self.close)
        hh.addWidget(close_btn)
        root.addWidget(header)

        # ── Search bar ─────────────────────────────────────────────────────
        search_frame = self._build_search_bar()
        search_frame.setStyleSheet(
            "QFrame{background:#ffffff;border:none;"
            "border-bottom:1px solid #e3e3e3;padding:6px 12px;}"
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
        self.tabs.addTab(reviewed_tab, "Reviewed (Cloud)")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Right: PDF preview
        preview_group = QGroupBox("  Report Preview")
        pg_layout = QVBoxLayout(preview_group)
        pg_layout.setContentsMargins(4, 4, 4, 4)
        self.preview_panel = PdfPreviewPanel()
        pg_layout.addWidget(self.preview_panel)

        self.splitter.addWidget(self.tabs)
        self.splitter.addWidget(preview_group)
        self.splitter.setStretchFactor(0, 56)
        self.splitter.setStretchFactor(1, 44)
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
                    f"Could not load doctors: {e}"
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
            self.rev_status_lbl.setText(
                f"{len(doctors)} doctor(s) loaded. Select one to fetch reports."
            )
        else:
            self.rev_status_lbl.setText("No doctors found in cloud.")

    def _build_reviewed_tab(self) -> QWidget:
        """Reviewed-by-doctor reports fetched from the public API."""
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(6, 8, 6, 6)
        layout.setSpacing(8)

        # ── info banner ────────────────────────────────────────────────────
        info = QLabel(
            "Enter doctor name exactly as registered (e.g. Dr_Neha) and click "
            "'Fetch Reports' to load reviewed ECG reports from the cloud."
        )
        info.setStyleSheet(
            "background:#e8f0fe;border:1px solid #c5d4f0;border-radius:5px;"
            "color:#1a2340;font-weight:400;font-size:12px;padding:8px 10px;"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # ── toolbar ────────────────────────────────────────────────────────
        top = QHBoxLayout()
        top.setSpacing(8)

        dl = QLabel("Doctor:")
        dl.setStyleSheet("color:#1a73e8;font-size:12px;font-weight:600;")

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
            ["Date", "Time", "Patient", "Type", "Doctor", "Presigned / Preview URL"]
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
            "Type a doctor name and press 'Fetch Reports' to load from cloud."
        )
        self.rev_status_lbl.setStyleSheet(
            "color:#444;font-size:11px;font-weight:400;"
        )
        hint_row.addWidget(self.rev_status_lbl, 1)
        hint_lbl = QLabel("Double-click a row to open URL")
        hint_lbl.setStyleSheet("color:#666;font-size:10px;font-weight:400;")
        hint_row.addWidget(hint_lbl)
        layout.addLayout(hint_row)
        return w

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
            url = f"{PUBLIC_REVIEWED_REPORTS_URL}?doctorName={doc_name}"
            resp = requests.get(url, timeout=12)
            if resp.status_code != 200:
                self.rev_status_lbl.setText(
                    f"Cloud API Error: {resp.status_code} - {resp.reason}. Doctor: {doc_name}"
                )
                return
            
            ct = resp.headers.get("Content-Type", "")
            try:
                data = resp.json() if "application/json" in ct or resp.text.strip().startswith("[") else []
            except Exception as json_err:
                self.rev_status_lbl.setText(f"Cloud Data Error: Failed to parse JSON response. {json_err}")
                return
            if not isinstance(data, list):
                data = []

            # Fill table — Date | Time | Patient | Type | Doctor | Presigned URL
            self.rev_table.setRowCount(0)
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
                    str(entry.get("doctorName", entry.get("doctor", doc_name))),
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
        except Exception as e:
            self.rev_status_lbl.setText(f"Error fetching cloud data: {e}")

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
        self.search_type_combo.addItems(["Patient Name", "Date Range", "Single Date"])
        self.search_type_combo.currentTextChanged.connect(self._on_search_type_changed)
        self.search_type_combo.setFixedWidth(130)

        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Patient name...")
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

    def _build_table(self, layout):
        self.table = QTableWidget()
        self.table.setColumnCount(11)
        self.table.setHorizontalHeaderLabels([
            "Date", "Time", "Org.", "Doctor", "Patient Name",
            "Age", "Gender", "Height", "Weight", "Type", "Status"
        ])
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        hh = self.table.horizontalHeader()
        hh.setStretchLastSection(False)
        hh.setSectionResizeMode(hh.Stretch)
        self.table.cellClicked.connect(self._on_cell_clicked)
        self.table.cellDoubleClicked.connect(self._on_row_double_clicked)
        self.table.itemSelectionChanged.connect(self._on_selection_changed_preview)
        layout.addWidget(self.table, 1)

    def _build_action_buttons(self, layout):
        bar = QFrame()
        bar.setStyleSheet(
            "QFrame{background:#ffffff;border-top:1px solid #e3e3e3;"
            "border-radius:0 0 8px 8px;padding:4px 2px;}"
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

        btn("Preview", self._preview_selected)
        btn("Email", self._send_email)
        btn("System Viewer", self._open_in_system, secondary=True)
        btn("Send for Review", self.send_report_for_review)
        btn("Export All", self.export_all_reports, secondary=True)
        btn("Cloud Status", self.refresh_reviewed_reports, secondary=True)
        row.addStretch()

        layout.addWidget(bar)

    # ── search ──────────────────────────────────────────────────────────────
    def _on_search_type_changed(self, search_type):
        self.search_input.setVisible(search_type == "Patient Name")
        for w in (self.start_date_edit, self.to_label, self.end_date_edit):
            w.setVisible(search_type == "Date Range")
        self.single_date_edit.setVisible(search_type == "Single Date")
        self.filter_table()

    def filter_table(self):
        search_type = self.search_type_combo.currentText()
        self.table.setRowCount(0)
        for entry in self.all_history_entries:
            show = False
            if search_type == "Patient Name":
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
        if "holter" in fn: rt = "Holter"
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
                                 .replace("holter_report", "Holter_")
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
            return "Holter"
        if "analysis" in fl:
            return "Analysis"
        return "ECG"

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
                        "age": "",
                        "gender": "",
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
                            rt = "Holter"
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
                            "age": entry.get("age", ""),
                            "gender": entry.get("gender", ""),
                            "height": entry.get("height", ""),
                            "weight": entry.get("weight", ""),
                            "report_file": os.path.join(REPORTS_DIR, entry.get("filename", "")),
                            "username": entry.get("username", ""),
                        })
            except Exception:
                pass

        # Also scan reports directory for any stray PDFs that might not be in index
        RECORDINGS_DIR = os.path.join(BASE_DIR, "recordings")
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
            pf = os.path.join(BASE_DIR, "all_patients.json")
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
                            "patient_name": pname, "age": str(p.get("age", "")),
                            "gender": p.get("gender", ""), "height": str(p.get("height", "")),
                            "weight": str(p.get("weight", "")), "report_file": "",
                        })
                except Exception:
                    pass

        # Ensure reports folder PDFs are also visible in history list
        self._add_report_file_entries(history_entries)

        # Filter by username and normalize report type
        self.all_history_entries = []
        for entry in history_entries:
            if self.username and entry.get("username") and entry.get("username") != self.username:
                continue
            rf = entry.get("report_file", "") or ""
            rt = entry.get("report_type", "")
            if not rt:
                fl = rf.lower()
                if "holter" in fl: rt = "Holter"
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
        for entry in self.all_history_entries:
            self._add_row(entry)

        self.preview_panel.clear()

    def _add_row(self, entry):
        row = self.table.rowCount()
        self.table.insertRow(row)
        status = entry.get("review_status", "Pending")
        values = [
            entry.get("date", ""), entry.get("time", ""), entry.get("Org.", ""),
            entry.get("doctor", ""),
            entry.get("patient_name", "") or (
                (entry.get("first_name", "") + " " + entry.get("last_name", "")).strip()),
            str(entry.get("age", "")), entry.get("gender", ""),
            str(entry.get("height", "")), str(entry.get("weight", "")),
            entry.get("report_type", ""), status,
        ]
        status_colors = {
            "Pending":     ("#f2f2f2", "#333333"),
            "Under Review":("#e8e8e8", "#111111"),
            "Reviewed":    ("#d9d9d9", "#111111"),
            "Queued":      ("#efefef", "#333333"),
        }
        for col, val in enumerate(values):
            item = QTableWidgetItem(val)
            item.setTextAlignment(Qt.AlignCenter)
            if col == 10:
                bg, fg = status_colors.get(val, ("#e9ecef", "#6c757d"))
                item.setBackground(QColor(bg))
                item.setForeground(QColor(fg))
                item.setFont(QFont("Segoe UI", 9, QFont.Bold))
            self.table.setItem(row, col, item)

        rf = entry.get("report_file", "")
        if self.table.item(row, 0):
            self.table.item(row, 0).setData(Qt.UserRole, rf)
        self.table.setRowHeight(row, 26)

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
        self._preview_selected()

    def _get_report_file(self, row) -> str:
        item = self.table.item(row, 0)
        if item:
            rf = item.data(Qt.UserRole) or ""
            if rf and os.path.exists(rf):
                return rf
        # Try to find by patient name
        pi = self.table.item(row, 4)
        di = self.table.item(row, 0)
        pname = pi.text().strip() if pi else ""
        date_str = di.text().strip() if di else ""
        return self._find_report_file(pname, date_str) or ""

    def _preview_selected(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Preview", "Select a report row first.")
            return
        self._load_preview_for_row(row, silent=False)

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

    def _send_email(self):
        row = self.table.currentRow()
        if row < 0:
            QMessageBox.information(self, "Send Email", "Select a report row first.")
            return
        rf = self._get_report_file(row)
        if not rf or not os.path.exists(rf):
            QMessageBox.warning(self, "Send Email", "No local PDF found for this entry.")
            return
        pi = self.table.item(row, 4)
        pname = pi.text() if pi else ""
        dlg = SendEmailDialog(rf, patient_name=pname, parent=self)
        dlg.exec_()

    # ── status context menu ───────────────────────────────────────────────
    def _show_status_menu(self, row):
        from PyQt5.QtWidgets import QMenu
        from PyQt5.QtGui import QCursor
        menu = QMenu(self)
        menu.setStyleSheet("QMenu{background:#fff;border:1px solid #cfcfcf;border-radius:5px;}"
                           "QMenu::item{padding:7px 18px;color:#111;}"
                           "QMenu::item:selected{background:#111;color:#fff;}")
        si = self.table.item(row, 10)
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
        si = self.table.item(row, 10)
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
        pname = pi.text() if pi else ""
        dstr = di.text() if di else ""
        for entry in self.all_history_entries:
            if entry.get("patient_name") == pname and entry.get("date") == dstr:
                entry["review_status"] = new_status
                entry["review_updated_at"] = datetime.datetime.now().isoformat()
                entry["review_updated_by"] = self.username or "unknown"
                break
        self._save_history_to_file()

    # ── file helpers ─────────────────────────────────────────────────────────
    def _find_report_file(self, patient_name, date_str="") -> str:
        if not os.path.exists(REPORTS_DIR):
            return ""
        pdfs = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")]
        pclean = patient_name.replace(" ", "_").replace(",", "").upper()
        for pdf in pdfs:
            if pclean and pclean in pdf.upper():
                return os.path.join(REPORTS_DIR, pdf)
        if date_str:
            try:
                dp = datetime.datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y%m%d")
                for pdf in pdfs:
                    if dp in pdf:
                        return os.path.join(REPORTS_DIR, pdf)
            except Exception:
                pass
        ecg = [f for f in pdfs if f.startswith("ECG_Report_")]
        if ecg:
            ecg.sort(key=lambda f: os.path.getmtime(os.path.join(REPORTS_DIR, f)), reverse=True)
            return os.path.join(REPORTS_DIR, ecg[0])
        return ""

    def _open_pdf_file(self, path):
        try:
            if os.name == "nt":
                os.startfile(path)
            elif sys.platform == "darwin":
                os.system(f'open "{path}"')
            else:
                os.system(f'xdg-open "{path}"')
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
                QMessageBox.warning(self, "Error", "No Internet Connection is available. Could not fetch available doctors list.")
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
            resp = requests.get(PUBLIC_REVIEWED_REPORTS_URL, timeout=10)
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
            if not os.path.exists(REPORTS_DIR):
                QMessageBox.warning(self, "Export", "Reports directory not found.")
                return
            pdfs = [f for f in os.listdir(REPORTS_DIR) if f.lower().endswith(".pdf")]
            if not pdfs:
                QMessageBox.information(self, "Export", "No PDF reports found.")
                return
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = os.path.join(export_dir, f"ECG_Reports_Export_{ts}")
            os.makedirs(dest, exist_ok=True)
            ok, fail = 0, 0
            for pdf in pdfs:
                try:
                    shutil.copy2(os.path.join(REPORTS_DIR, pdf), os.path.join(dest, pdf))
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
                "age", "gender", "height", "weight", "report_type", "review_status"]
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
            for entry in self.all_history_entries:
                pn = entry.get("patient_name", "")
                ds = entry.get("date", "")
                found = False
                for se in all_e:
                    if se.get("patient_name") == pn and se.get("date") == ds:
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
            doctors = ["Dr_Rohit", "Dr_Neha", "Dr_Arjun"]
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
            resp = requests.get(PUBLIC_REVIEWED_REPORTS_URL,
                                params={"doctorName": dname} if dname else {}, timeout=10)
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
def append_history_entry(patient_details, report_file_path, report_type="ECG", username=None):
    """Append a new history entry when a report is generated."""
    try:
        entries = []
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, "r") as f:
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
        "report_file": os.path.abspath(report_file_path) if report_file_path else "",
        "review_status": "Pending",
        "review_updated_at": "",
        "review_updated_by": "",
    }
    if isinstance(patient_details, dict):
        base.update(patient_details)

    entries.append(base)

    try:
        hdir = os.path.dirname(HISTORY_FILE)
        if hdir and not os.path.exists(hdir):
            os.makedirs(hdir, exist_ok=True)
        with open(HISTORY_FILE, "w") as f:
            json.dump(entries, f, indent=2)
        print(f"💾 History entry saved: {base.get('patient_name','')} — {report_type}")
    except Exception as e:
        print(f"⚠️ Failed to save history entry: {e}")
