from .metrics.reference_intervals import lookup_reference_intervals
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Path, String
from reportlab.graphics import renderPDF

class TransparentDrawing(Drawing):
    """Drawing subclass that suppresses ReportLab's default white background rect."""
    def drawOn(self, canvas, x, y, _sW=0):
        canvas.saveState()
        canvas.translate(x, y)
        renderPDF.draw(self, canvas, 0, 0, showBoundary=0)
        canvas.restoreState()
import os
from utils.app_paths import data_file
import sys
import json
import matplotlib.pyplot as plt  
import matplotlib
import numpy as np

# matplotlib.use('Agg') # Removed - isolated FigureCanvasAgg used instead

# ------------------------ ECG grid scale constants ------------------------
# 40 large boxes across A4 width (210mm) => 1 large box = 5.25mm
ECG_BASE_BOX_MM = 5.0
ECG_LARGE_BOX_MM = 5.0  # Standard ECG: 1 large box = 5mm (A4 standard)
ECG_SMALL_BOX_MM = ECG_LARGE_BOX_MM / 5.0
# Scale wave speed so 1 second equals 5 large boxes at 25 mm/s on 40-box grid
ECG_SPEED_SCALE = ECG_LARGE_BOX_MM / ECG_BASE_BOX_MM
STANDARD_REPORT_WINDOW_SECONDS = 10.0
REPORT_STRIP_WIDTH_POINTS = 460
# Rendering-only trim so a waveform currently spanning ~13 large boxes
# prints closer to the requested 12 large boxes without changing metric math.
REPORT_WAVEFORM_ADC_DIVISOR = 1083.3333333333333


def _samples_for_standard_report_window(sampling_rate):
    """Return sample count for the standard last-10-second ECG strip."""
    fs = _safe_float(sampling_rate, 500.0)
    if not fs or fs <= 0:
        fs = 500.0
    return max(1, int(round(STANDARD_REPORT_WINDOW_SECONDS * fs)))

LEAD_SEQUENCES = {
    "Standard": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
    "Cabrera": ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
}
Y_POSITIONS_MM = [247.0625, 227.0625, 207.0625, 187.0625, 167.0625, 147.0625, 127.0625, 107.0625, 87.0625, 67.0625, 47.0625, 27.0625]
def _add_patient_header(master_drawing, full_name, age, gender, patient, date_time_str, data, settings_manager):
    from reportlab.graphics.shapes import String
    from reportlab.lib.units import mm
    from reportlab.lib import colors
    
    # 1. Prepare Date and Time
    if date_time_str:
        parts = date_time_str.split()
        date_part = parts[0] if parts else ""
        time_part = parts[1] if len(parts) > 1 else ""
    else:
        from datetime import datetime
        now = datetime.now()
        date_part = now.strftime("%d/%m/%Y")
        time_part = now.strftime("%H:%M:%S")
    
    # 2. Prepare Metrics Column 2
    HR = data.get('HR_avg') or data.get('HR') or 0
    PR = data.get('PR') or 0
    QRS = data.get('QRS') or 0
    QT = _safe_float(data.get('QT'), 0.0)
    QTc = _safe_float(data.get('QTc'), 0.0)
    RR = int(60000 / float(HR)) if HR and float(HR) > 0 else 0
    
    # 3. Prepare Metrics Column 3
    _qtcf_val = data.get('QTc_Fridericia') or data.get('QTcF_ms') or data.get('QTcF') or data.get('QTcF_interval')
    qtcf_display = f"{int(round(float(_qtcf_val)))} ms" if _qtcf_val and float(_qtcf_val) > 0 else "-- ms"
    
    # RV5/SV1
    raw_hr_check = data.get('HR_avg') or data.get('HR') or data.get('HR_bpm') or 0
    try:
        hr_val_check = float(str(raw_hr_check).replace("bpm","").strip())
    except Exception:
        hr_val_check = 0.0

    if hr_val_check <= 0:
        rv5_mv = 0.0
        sv1_mv = 0.0
    else:
        rv5_mv = data.get('rv5_mv') if data.get('rv5_mv') is not None else data.get('rv5')
        sv1_mv = data.get('sv1_mv') if data.get('sv1_mv') is not None else data.get('sv1')

    # If the source lead is missing, zero only the matching metric.
    if data.get('_rv5_source_present') is False:
        rv5_mv = 0.0
    if data.get('_sv1_source_present') is False:
        sv1_mv = 0.0

    if rv5_mv is None:
        rv5_mv = 0.0
    if sv1_mv is None:
        sv1_mv = 0.0

    rv5_text = f"{rv5_mv:.3f} mV"
    sv1_text = f"{sv1_mv:.3f} mV"    # RV5+SV1
    rv5_sv1_sum = (rv5_mv - abs(sv1_mv)) if (rv5_mv is not None and sv1_mv is not None and hr_val_check > 0) else 0.0
    rv5_sv1_sum_text = f"{rv5_sv1_sum:.3f} mV"
    
    # P/QRS/T Axis
    p_axis = data.get('p_axis', '--')
    qrs_axis = data.get('QRS_axis', '--')
    t_axis = data.get('t_axis', '--')
    def _fmt_axis(v):
        if v is None or v == '--': return '--'
        try:
            val = str(v).replace('°','')
            return f"{int(round(float(val)))}°"
        except: return '--'
    p_qrs_t_text = f"{_fmt_axis(p_axis)}/{_fmt_axis(qrs_axis)}/{_fmt_axis(t_axis)}"

    # 4. Prepare Filter Info
    emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
    ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
    ac_frequency = f"{ac_setting}Hz" if ac_setting in ("50", "60") else "Off"
    if dft_setting not in ("off", "") and emg_setting not in ("off", ""):
        filter_band = f"{dft_setting}-{emg_setting}Hz"
    elif dft_setting not in ("off", ""):
        filter_band = f"HP: {dft_setting}Hz"
    elif emg_setting not in ("off", ""):
        filter_band = f"LP: {emg_setting}Hz"
    else:
        filter_band = "Filter: Off"
    filter_info = f"25.0 mm/s   {filter_band}   AC : {ac_frequency}   10.0 mm/mV"

def _fit_pdf_text(text, max_width_pt, font_name="Helvetica", font_size=9):
    """Truncate string with '...' if width in points exceeds max_width_pt so text never overflows PDF box or margins."""
    text = str(text or "").strip()
    if not text:
        return ""
    try:
        from reportlab.pdfbase.pdfmetrics import stringWidth
        if stringWidth(text, font_name, font_size) <= max_width_pt:
            return text
        ellipsis = "..."
        while len(text) > 0 and stringWidth(text + ellipsis, font_name, font_size) > max_width_pt:
            text = text[:-1]
        return (text + ellipsis) if text else ""
    except Exception:
        return text[:18] + "..." if len(text) > 18 else text

    # 5. Prepare Org Info
    org_name = patient.get('Org.', '') if patient else ''
    org_address = patient.get('org_address', '') if patient else ''
    phone_no = patient.get('doctor_mobile', '') if patient else ''

    # 6. Define Layout Coordinates
    y_start = 282.0 * mm
    y_step = 4.2 * mm
    
    col1_x = 0.0 * mm
    col2_x = 62.0 * mm
    col3_x = 105.0 * mm
    col4_x = 152.0 * mm

    # --- COLUMN 1 ---
    master_drawing.add(String(col1_x, y_start, f"Age: {age}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col1_x, y_start - y_step, f"Gender: {gender}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col1_x, y_start - 2*y_step, f"ECG Type: Standard", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    # Use YYYY-MM-DD format for Date & Time to match Image 2
    try:
        from datetime import datetime
        if date_part and "/" in date_part:
            d_obj = datetime.strptime(date_part, "%d/%m/%Y")
            formatted_date = d_obj.strftime("%Y-%m-%d")
        else:
            formatted_date = date_part
    except:
        formatted_date = date_part
    
    master_drawing.add(String(col1_x, y_start - 3*y_step, f"Date & Time: {formatted_date} {time_part}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    
    # Reformat filter_info to match Image 2 (25.0 mm/s  0.5-150Hz  AC:50Hz  10.0 mm/mV)
    clean_filter_band = filter_band.replace("Filter: ", "").replace("HP: ", "").replace("LP: ", "")
    clean_ac = ac_frequency.replace("Off", "Off").replace("Hz", "Hz")
    display_filter_info = f"25.0 mm/s  {clean_filter_band}  AC:{clean_ac}  10.0 mm/mV"
    master_drawing.add(String(col1_x, y_start - 4*y_step, display_filter_info, fontSize=9, fontName="Helvetica", fillColor=colors.black))

    # --- COLUMN 2 ---
    fitted_name = _fit_pdf_text(f"Name: {full_name}", 115, "Helvetica", 9)
    master_drawing.add(String(col2_x, y_start, fitted_name, fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col2_x, y_start - y_step, f"HR: {int(round(float(HR)))} bpm", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col2_x, y_start - 2*y_step, f"RR: {RR} ms", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col2_x, y_start - 3*y_step, f"PR: {PR} ms", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col2_x, y_start - 4*y_step, f"QRS: {QRS} ms", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col2_x, y_start - 5*y_step, f"QT: {int(round(QT))} ms", fontSize=9, fontName="Helvetica", fillColor=colors.black))

    # --- COLUMN 3 ---
    master_drawing.add(String(col3_x, y_start, f"QTc: {int(round(QTc))} ms", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col3_x, y_start - y_step, f"QTcF: {qtcf_display}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col3_x, y_start - 2*y_step, f"RV5/SV1: {rv5_text}/{sv1_text}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    master_drawing.add(String(col3_x, y_start - 3*y_step, f"RV5+SV1: {rv5_sv1_sum_text}", fontSize=9, fontName="Helvetica", fillColor=colors.black))
    # master_drawing.add(String(col3_x, y_start - 4*y_step, f"P/QRS/T: {p_qrs_t_text}", fontSize=9, fontName="Helvetica", fillColor=colors.black))

    # --- COLUMN 4 ---
    fitted_org = _fit_pdf_text(org_name, 130, "Helvetica-Bold", 9)
    fitted_address = _fit_pdf_text(org_address, 130, "Helvetica-Bold", 9)
    fitted_phone = _fit_pdf_text(phone_no, 130, "Helvetica-Bold", 9)

    master_drawing.add(String(col4_x, y_start, fitted_org, fontSize=9, fontName="Helvetica-Bold", fillColor=colors.black))
    master_drawing.add(String(col4_x, y_start - y_step, fitted_address, fontSize=9, fontName="Helvetica-Bold", fillColor=colors.black))
    master_drawing.add(String(col4_x, y_start - 2*y_step, fitted_phone, fontSize=9, fontName="Helvetica-Bold", fillColor=colors.black))



def _build_vital_table(data):
    """Compact report summary: show only HR/PR/QRS/QT/QTc."""
    HR = _safe_float(data.get('HR_avg') or data.get('HR') or data.get('HR_bpm'), 0)
    PR = _safe_float(data.get('PR'), 0)
    QRS = _safe_float(data.get('QRS'), 0)
    QT = _safe_float(data.get('QT'), 0)
    QTc = _safe_float(data.get('QTc'), 0)

    vital_table_data = [
        [f"HR : {int(round(HR))} bpm", f"QT : {int(round(QT))} ms"],
        [f"PR : {int(round(PR))} ms", f"QTc: {int(round(QTc))} ms"],
        [f"QRS: {int(round(QRS))} ms", ""]
    ]
    vital_params_table = Table(vital_table_data, colWidths=[100, 100])
    vital_params_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0, 0, 0, alpha=0)),
        ("GRID", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("BOX", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return vital_params_table

# ------------------------ Resource path helper for PyInstaller compatibility ------------------------

def _get_resource_path(relative_path):
    """
    Get resource path that works both in development and when packaged as exe.
    For PyInstaller: resources are in sys._MEIPASS
    For development: resources are relative to project root
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        if hasattr(sys, '_MEIPASS'):
            base_path = sys._MEIPASS
        else:
            # Development mode - get path relative to this file
            base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.join(base_path, relative_path)
    except Exception:
        # Fallback to relative path
        return os.path.join(os.path.abspath(os.path.dirname(__file__)), "..", "..", relative_path)

# ------------------------ Conservative interpretation helpers ------------------------

def _safe_float(value, default=None):
    try:
        return float(value)
    except Exception:
        return default


def _align_report_intervals_to_reference(data: dict) -> None:
    """Fill missing report interval fields from HR reference table.

    IMPORTANT:
    We must not overwrite measured values (QT/QTc/PR/QRS/RR) with reference
    table values. Overwriting caused clinically incorrect report numbers,
    especially for QTc at low/high heart rates.
    """
    try:
        hr_val = _safe_float(data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR"), 0)
        if not hr_val or hr_val <= 0:
            return

        ref = lookup_reference_intervals(float(hr_val))
        if not ref:
            return

        # Only backfill missing/invalid values.
        # Measured values from analysis pipeline always take priority.
        mapping = {
            "RR_ms": "RR",
            "PR": "PR",
            "QRS": "QRS",
            "QT": "QT",
            "QTc": "QTc",
        }
        for data_key, ref_key in mapping.items():
            current = _safe_float(data.get(data_key), 0)
            if not current or current <= 0:
                data[data_key] = int(round(ref[ref_key]))
    except Exception:
        return


def _build_conservative_conclusions(metrics, settings_manager=None, sampling_rate=None, recording_duration=None):
    """
    Build a conservative, hospital-style conclusions list.
    Uses only provided metrics (assumed pre-display / raw-based).
    """
    conclusions = []

    # Extract measurements
    hr = _safe_float(metrics.get("HR_bpm") or metrics.get("Heart_Rate") or metrics.get("HR"), None)
    pr = _safe_float(metrics.get("PR"), None)
    qrs = _safe_float(metrics.get("QRS"), None)
    qt = _safe_float(metrics.get("QT"), None)
    qtc_bazett = _safe_float(metrics.get("QTc_Bazett") or metrics.get("QTc"), None)
    qtc_frid = _safe_float(metrics.get("QTc_Fridericia"), None)

    rv5 = _safe_float(metrics.get("RV5"), None)
    sv1 = _safe_float(metrics.get("SV1"), None)
    rv5_sv1 = _safe_float(metrics.get("RV5_SV1"), None)
    st_dev = _safe_float(metrics.get("ST_deviation") or metrics.get("ST"), None)

    # Acquisition params
    wave_gain = None
    wave_speed = None
    if settings_manager:
        try:
            wave_gain = _safe_float(settings_manager.get_wave_gain(), None)
        except Exception:
            pass
        try:
            wave_speed = _safe_float(settings_manager.get_wave_speed(), None)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ArrhythmiaEngine – use pre-computed diagnoses if available
    # ------------------------------------------------------------------
    engine_diagnoses = metrics.get("arrhythmias") or []

    # Rhythm classification
    if engine_diagnoses:
        diagnosis_str = ", ".join(engine_diagnoses)
        if hr is not None:
            rhythm = f"Rhythm: {diagnosis_str} (HR ≈ {hr:.0f} bpm)"
        else:
            rhythm = f"Rhythm: {diagnosis_str}"
    elif hr is not None:
        if hr > 100:
            rhythm = f"Sinus tachycardia (HR ≈ {hr:.0f} bpm)"
        elif hr < 60:
            rhythm = f"Sinus bradycardia (HR ≈ {hr:.0f} bpm)"
        else:
            rhythm = f"Normal sinus rhythm (HR ≈ {hr:.0f} bpm)"
    else:
        rhythm = "Rhythm: not enough data"

    # QTc assessment (suppress Bazett when HR > 100)
    qtc_line = "QTc: not available"
    if hr is not None and hr > 100:
        if qtc_frid is not None:
            qtc_line = f"QTcF (Fridericia): {qtc_frid:.0f} ms"
    else:
        if qtc_bazett is not None:
            if qtc_bazett < 440:
                band = "Normal"
            elif qtc_bazett <= 470:
                band = "Borderline"
            else:
                band = "Prolonged"
            qtc_line = f"QTcB (Bazett): {qtc_bazett:.0f} ms ({band})"
    if qtc_frid is not None:
        qtc_sec = qtc_frid / 1000.0
        if qtc_line == "QTc: not available":
            qtc_line = f"QTcF (Fridericia): {qtc_frid:.0f} ms ({qtc_sec:.3f} s)"  # GE/Philips/BPL: ms and seconds
        else:
            qtc_line += f"; QTcF (Fridericia): {qtc_frid:.0f} ms ({qtc_sec:.3f} s)"  # GE/Philips/BPL: ms and seconds



    # ST deviation (conservative)
    st_line = "ST deviation: not assessed"
    if st_dev is not None:
        st_line = f"ST deviation: {st_dev:.2f} mV (J+60ms); report only as deviation"

    # Measured values summary
    measured = "Measured Values: "
    parts = []
    if hr is not None:
        parts.append(f"HR {hr:.0f} bpm")
    if pr is not None:
        parts.append(f"PR {pr:.0f} ms")
    if qrs is not None:
        parts.append(f"QRS {qrs:.0f} ms")
    if qt is not None:
        parts.append(f"QT {qt:.0f} ms")
    if qtc_bazett is not None:
        parts.append(f"QTc {qtc_bazett:.0f} ms")

    # Acquisition info
    acq_parts = []
    if sampling_rate:
        acq_parts.append(f"Sampling rate {sampling_rate} Hz")
    # ── Always add standard speed + gain ──
    acq_parts.append("25.0 mm/s")
    acq_parts.append("10.0 mm/mV")
    # ── FIX: Always show standard 10mm/mV on report ─────────────────────
    acq_parts.append("10.0 mm/mV")  # Standard — always 10mm/mV for reports
    # ── FIX: Always show standard 25mm/s on report ───────────────────────
    acq_parts.append("25.0 mm/s")  # Standard — always 25mm/s for reports
    if recording_duration:
        acq_parts.append(f"Duration {recording_duration}")
    acq_info = "Acquisition: " + "; ".join(acq_parts) if acq_parts else "Acquisition: not available"

    # Build conservative list (max 12 entries downstream)
    conclusions.append(measured)
    conclusions.append(rhythm)
    # If engine produced extra diagnoses beyond the first, add them individually
    if len(engine_diagnoses) > 1:
        for dx in engine_diagnoses[1:]:
            conclusions.append(f"  • {dx}")
    conclusions.append(qtc_line)

    conclusions.append(st_line)
    conclusions.append("Automated interpretation (conservative): Normal unless measurements suggest otherwise")
    conclusions.append(acq_info)
    conclusions.append("This is an automated ECG analysis and must be reviewed by a qualified physician.")

    return conclusions

# ==================== ECG DATA SAVE/LOAD FUNCTIONS ====================

def save_ecg_data_to_file(ecg_test_page, output_file=None):
    """
    Save ECG data from ecg_test_page.data to a JSON file
    Returns: path to saved file or None if failed
    
    Example:
        saved_file = save_ecg_data_to_file(ecg_test_page)
        # Saved to: reports/ecg_data/ecg_data_20241119_143022.json
    """
    from datetime import datetime
    
    if not ecg_test_page or not hasattr(ecg_test_page, 'data'):
        print(" No ECG test page data available to save")
        return None
    
    # Create output directory
    ecg_data_dir = os.path.join(str(data_file("reports")), 'ecg_data')
    os.makedirs(ecg_data_dir, exist_ok=True)
    
    # Generate filename with timestamp
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(ecg_data_dir, f'ecg_data_{timestamp}.json')
    
    # Prepare data for saving
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    
    saved_data = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "sampling_rate": 500.0,
        "leads": {}
    }
    
    # Get sampling rate if available
    if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            sampled_rate = float(ecg_test_page.sampler.sampling_rate)
            # Guard against invalid/low sampling rates (report expects 500 Hz)
            if sampled_rate < 50.0 or sampled_rate > 1000.0:
                sampled_rate = 500.0
            saved_data["sampling_rate"] = sampled_rate
    
    # Save each lead's data - use FULL buffer (ecg_buffers if available, otherwise data)
    # Priority: Use ecg_buffers (5000 samples) if available, otherwise use data (1000 samples)
    
    # Debug: Check what attributes ecg_test_page has
    print(f" DEBUG: ecg_test_page attributes check:")
    print(f"  has ecg_buffers: {hasattr(ecg_test_page, 'ecg_buffers')}")
    print(f"   has data: {hasattr(ecg_test_page, 'data')}")
    print(f"   has ptrs: {hasattr(ecg_test_page, 'ptrs')}")
    if hasattr(ecg_test_page, 'ecg_buffers'):
        print(f"   ecg_buffers length: {len(ecg_test_page.ecg_buffers) if ecg_test_page.ecg_buffers else 0}")
    if hasattr(ecg_test_page, 'data'):
        print(f"   data length: {len(ecg_test_page.data) if ecg_test_page.data else 0}")
        if ecg_test_page.data and len(ecg_test_page.data) > 0:
            print(f"   data[0] length: {len(ecg_test_page.data[0]) if isinstance(ecg_test_page.data[0], (list, np.ndarray)) else 'N/A'}")
    
    for i, lead_name in enumerate(lead_names):
        data_to_save = []
        
        # Priority 1: Try to use ecg_buffers (larger buffer, 5000 samples)
        if hasattr(ecg_test_page, 'ecg_buffers') and i < len(ecg_test_page.ecg_buffers):
            buffer = ecg_test_page.ecg_buffers[i]
            if isinstance(buffer, np.ndarray) and len(buffer) > 0:
                # Check if this is a rolling buffer with ptrs
                if hasattr(ecg_test_page, 'ptrs') and i < len(ecg_test_page.ptrs):
                    ptr = ecg_test_page.ptrs[i]
                    window_size = getattr(ecg_test_page, 'window_size', 1000)
                    
                    # For report generation: use FULL buffer (5000 samples), not just window_size (1000)
                    # Get all available data from buffer, starting from ptr
                    if ptr + len(buffer) <= len(buffer):
                        # No wrap needed: get from ptr to end, then from start to ptr
                        part1 = buffer[ptr:].tolist()
                        part2 = buffer[:ptr].tolist()
                        data_to_save = part1 + part2  # Full circular buffer
                    else:
                        # Simple case: use all buffer data
                        data_to_save = buffer.tolist()
                else:
                    # No ptrs: use ALL available data (full buffer)
                    data_to_save = buffer.tolist()
        
        # Priority 2: Fallback to ecg_test_page.data (smaller buffer, 1000 samples)
        if not data_to_save and i < len(ecg_test_page.data):
            lead_data = ecg_test_page.data[i]
            if isinstance(lead_data, np.ndarray):
                # Use ALL available data (not just window_size)
                data_to_save = lead_data.tolist()
            elif isinstance(lead_data, (list, tuple)):
                data_to_save = list(lead_data)
        
        saved_data["leads"][lead_name] = data_to_save if data_to_save else []
    
    # Check if we have sufficient data for report generation
    sample_counts = [len(saved_data["leads"][lead]) for lead in saved_data["leads"] if saved_data["leads"][lead]]
    if sample_counts:
        max_samples = max(sample_counts)
        min_samples = min(sample_counts)
        print(f" Buffer analysis: Max samples={max_samples}, Min samples={min_samples}")
        
        # Calculate expected samples for 13.2s window at current sampling rate
        sampling_rate = saved_data.get("sampling_rate", 500.0)
        expected_samples_for_13_2s = int(13.2 * sampling_rate)
        
        if max_samples < expected_samples_for_13_2s:
            print(f" WARNING: Buffer has only {max_samples} samples, need {expected_samples_for_13_2s} for 13.2s window")
            print(f"   Current time window: {max_samples/sampling_rate:.2f}s")
            print(f"   Expected time window: 13.2s")
            print(f"    TIP: Run ECG for at least 15-20 seconds to accumulate sufficient data")
    
    # Save to file
    try:
        with open(output_file, 'w') as f:
            json.dump(saved_data, f, indent=2)
        print(f"Saved ECG data to: {output_file}")
        print(f"   Leads saved: {list(saved_data['leads'].keys())}")
        print(f"   Sampling rate: {saved_data['sampling_rate']} Hz")
        print(f"   Total data points per lead: {[len(saved_data['leads'][lead]) for lead in saved_data['leads']]}")
        return output_file
    except Exception as e:
        print(f" Error saving ECG data: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_ecg_data_from_file(file_path):
    """
    Load ECG data from JSON file
    Returns: dict with 'leads', 'sampling_rate', 'timestamp' or None if failed
    
    Example:
        data = load_ecg_data_from_file('reports/ecg_data/ecg_data_20241119_143022.json')
        # Returns: {'leads': {'I': [...], 'II': [...]}, 'sampling_rate': 500.0, ...}
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Convert lists back to numpy arrays
        if 'leads' in data:
            for lead_name in data['leads']:
                if isinstance(data['leads'][lead_name], list):
                    data['leads'][lead_name] = np.array(data['leads'][lead_name])
        
        print(f" Loaded ECG data from: {file_path}")
        print(f"   Leads loaded: {list(data.get('leads', {}).keys())}")
        print(f"   Sampling rate: {data.get('sampling_rate', 500.0)} Hz")
        return data
    except Exception as e:
        print(f" Error loading ECG data: {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_time_window_from_bpm_and_wave_speed(hr_bpm, wave_speed_mm_s, desired_beats=6):
    """
    Calculate optimal time window based on BPM and wave_speed
    
    Important: Report ECG graph width = 37 boxes × ECG_LARGE_BOX_MM
     wave_speed time calculate factor use :
        Time from wave_speed = (graph_width_mm / effective_wave_speed_mm_s) seconds
    
    Formula:
        - Time window = (graph_width_mm / effective_wave_speed_mm_s) seconds ONLY
          (37 boxes × ECG_LARGE_BOX_MM)
        - BPM window is NOT used - only wave speed window
        - Beats = (BPM / 60) × time_window
        - Final window clamped maximum 20 seconds (NO minimum clamp)
    
    
    Returns: (time_window_seconds, num_samples)
    """
    # Calculate time window from wave_speed ONLY (BPM window NOT used)
    # Report ECG graph width = 37 boxes × ECG_LARGE_BOX_MM
    # Time = Distance / Speed (scaled for 40-box grid)
    graph_boxes = 18.5 if abs(float(wave_speed_mm_s) - 12.5) < 0.01 else 37.0
    ecg_graph_width_mm = graph_boxes * ECG_LARGE_BOX_MM
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    calculated_time_window = ecg_graph_width_mm / max(1e-6, effective_wave_speed_mm_s)
    
    # Only clamp maximum to 20 seconds (NO minimum clamp)
    calculated_time_window = min(calculated_time_window, 20.0)
    
    # Calculate number of samples (assuming 500 Hz default)
    num_samples = int(calculated_time_window * 500.0)
    
    # Calculate expected beats: beats = (BPM / 60) × time_window
    # Formula: beats per second = BPM / 60, then multiply by time window
    beats_per_second = hr_bpm / 60.0 if hr_bpm > 0 else 0
    expected_beats = beats_per_second * calculated_time_window
    
    print(f" Time Window Calculation (Wave Speed ONLY):")
    print(f"   Graph Width: {ecg_graph_width_mm:.2f}mm ({graph_boxes} boxes × {ECG_LARGE_BOX_MM:.2f}mm)")
    print(f"   Wave Speed: {wave_speed_mm_s}mm/s (effective {effective_wave_speed_mm_s:.2f}mm/s)")
    print(f"   Time Window: {ecg_graph_width_mm:.2f} / {effective_wave_speed_mm_s:.2f} = {calculated_time_window:.2f}s")
    print(f"   BPM: {hr_bpm} → Beats per second: {hr_bpm}/60 = {beats_per_second:.2f} beats/sec")
    print(f"   Expected Beats: {beats_per_second:.2f} × {calculated_time_window:.2f} = {expected_beats:.1f} beats")
    print(f"   Estimated Samples: {num_samples} (at 500Hz)")
    
    return calculated_time_window, num_samples

def _prepare_report_strip_signal(signal, sampling_rate, settings_manager, target_samples=None, emg_override=None, dft_override=None):
    """Prepare the visible report strip using a stable segment-local pipeline."""
    arr = np.asarray(signal, dtype=float)
    if arr.size < 2:
        return arr

    fs = _safe_float(sampling_rate, 500.0)
    if fs is None or fs <= 0:
        fs = 500.0

    if target_samples is not None:
        try:
            core_n = max(1, min(arr.size, int(target_samples)))
        except Exception:
            core_n = arr.size
        start_idx = max(0, arr.size - core_n)
    else:
        try:
            win_sec = float(str(settings_manager.get_setting("report_window_seconds", "10")).strip() or "10")
        except Exception:
            win_sec = 10.0
        core_n = min(arr.size, int(max(1.0, win_sec) * fs))
        start_idx = max(0, arr.size - core_n)

    pad_n = min(max(8, int(0.5 * fs)), start_idx)
    work_start = max(0, start_idx - pad_n)
    work = np.asarray(arr[work_start:], dtype=float)
    if work.size < 2:
        return work

    try:
        from ecg.ecg_filters import apply_ecg_filters

        if settings_manager is not None:
            ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip()
            emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
            dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
        else:
            ac_setting, emg_setting, dft_setting = "50", "25", "0.5"

        if emg_override is not None:
            emg_setting = emg_override
        if dft_override is not None:
            dft_setting = dft_override

        if ac_setting in ("50", "60"):
            required_fs = float(ac_setting) * 2.0 + 1.0
            if float(fs) <= required_fs:
                ac_setting = "off"

        # Reflect-pad both sides before filtering to prevent right-edge artifacts
        # (terminal jump/curve) in rendered report strips.
        pad_filt_n = min(max(12, int(0.35 * fs)), max(0, work.size // 3))
        if pad_filt_n > 0:
            work = np.pad(work, (pad_filt_n, pad_filt_n), mode="reflect")

        work = apply_ecg_filters(
            signal=work,
            sampling_rate=float(fs),
            ac_filter=ac_setting if ac_setting not in ("off", "") else None,
            emg_filter=emg_setting if emg_setting not in ("off", "") else None,
            dft_filter=dft_setting if dft_setting not in ("off", "") else None,
        )
        if pad_filt_n > 0 and work.size > (2 * pad_filt_n):
            work = work[pad_filt_n:-pad_filt_n]
    except Exception:
        pass

    if work.size > core_n:
        work = work[-core_n:]

    try:
        from ecg.signal.signal_processing import extract_low_frequency_baseline

        baseline_est = extract_low_frequency_baseline(work, float(fs))
        if np.isfinite(baseline_est):
            work = work - float(baseline_est)
    except Exception:
        pass

    try:
        dc = float(np.nanmean(work)) if work.size > 0 else 0.0
        if np.isfinite(dc):
            work = work - dc
    except Exception:
        pass

    try:
        from scipy.ndimage import gaussian_filter1d

        if work.size > 5:
            work = gaussian_filter1d(work, sigma=0.8)
    except Exception:
        pass

    return np.asarray(work, dtype=float)


def create_report_strip_paths(
    values,
    strip_x,
    strip_y,
    strip_width,
    strip_height,
    sampling_rate,
    settings_manager,
    wave_speed_mm_s=25.0,
    wave_gain_mm_mv=10.0,
    lead_name="II",
    adc_per_box_multiplier=6400.0,
    fill_strip=False,
):
    """
    Draw one ECG strip aligned box-by-box to standard landscape grid paper.
    Uses time-based horizontal scaling (25 mm/s × speed scale) and ADC-per-box
    vertical scaling — same algorithm as the HRV Lead II report strips.
    """
    from reportlab.graphics.shapes import Path
    from reportlab.lib import colors
    from reportlab.lib.units import mm as mm_unit

    center_y = strip_y + (strip_height / 2.0)
    notch_path = None
    notch_x = strip_x + (3.0 * mm_unit)

    try:
        notch_boxes = settings_manager.get_calibration_notch_boxes()
    except Exception:
        notch_boxes = 2.0

    notch_width = 5.0 * mm_unit
    notch_tail = 2.0 * mm_unit
    notch_height = (notch_boxes * 5.0) * mm_unit

    notch_path = Path(
        fillColor=None,
        strokeColor=colors.HexColor("#000000"),
        strokeWidth=0.8,
        strokeLineCap=1,
        strokeLineJoin=0,
    )
    notch_path.moveTo(strip_x, center_y)
    notch_path.lineTo(notch_x, center_y)
    notch_path.lineTo(notch_x, center_y + notch_height)
    notch_path.lineTo(notch_x + notch_width, center_y + notch_height)
    notch_path.lineTo(notch_x + notch_width, center_y)
    notch_path.lineTo(notch_x + notch_width + notch_tail + (1.0 * mm_unit), center_y)

    if values is None or len(values) < 2:
        baseline_path = Path(
            fillColor=None,
            strokeColor=colors.HexColor("#000000"),
            strokeWidth=0.4,
            strokeLineCap=1,
            strokeLineJoin=1,
        )
        baseline_path.moveTo(strip_x, center_y)
        baseline_path.lineTo(strip_x + strip_width, center_y)
        return baseline_path, notch_path, None

    adc_data = np.asarray(values, dtype=float)
    fs = _safe_float(sampling_rate, 500.0) or 500.0

    try:
        emg_val = str(settings_manager.get_setting("filter_emg", "25")).strip() if settings_manager else "25"
        dft_val = str(settings_manager.get_setting("filter_dft", "0.5")).strip() if settings_manager else "0.5"
        if dft_val in ("off", ""):
            dft_val = "0.5"
        centered_adc = _prepare_report_strip_signal(
            adc_data,
            float(fs),
            settings_manager,
            target_samples=adc_data.size,
            emg_override=emg_val,
            dft_override=dft_val,
        )
    except Exception:
        data_mean = float(np.mean(adc_data))
        if abs(data_mean - 2000.0) < 500:
            centered_adc = adc_data - 2000.0
        else:
            centered_adc = adc_data.copy()
        centered_adc = centered_adc - float(np.mean(centered_adc))

    # Skip linear detrending if the robust median-mean baseline filter (0.5 Hz) was applied,
    # because linear fitting on short asymmetric ECG segments can introduce artificial slants/drift.
    dft_val_clean = str(settings_manager.get_setting("filter_dft", "0.5")).strip() if settings_manager else "0.5"
    if dft_val_clean in ("off", ""):
        dft_val_clean = "0.5"
    if dft_val_clean != "0.5" and centered_adc.size > 20:
        x_idx = np.arange(centered_adc.size, dtype=float)
        trend = np.polyval(np.polyfit(x_idx, centered_adc, 1), x_idx)
        centered_adc = centered_adc - trend

    # NOTE: no edge taper here. stabilize_report_edges() cross-fades the first and
    # last samples into a flat baseline, which flattens any QRS that happens to fall
    # in that window — a beat 60 ms from the strip end printed at ~15% of its real
    # height. Filter transients are already handled without touching real samples:
    # _prepare_report_strip_signal() prepends 0.5 s of genuine pre-roll and
    # reflect-pads around the filters before trimming back to the visible window.
    # The strip must print the recorded waveform to its last sample.

    grid_box_mm = 5.0
    box_height_points = grid_box_mm * mm_unit
    mm_per_sample = float(wave_speed_mm_s) / max(1e-6, float(fs))
    strip_end_x = strip_x + strip_width
    wave_start_x = notch_x + notch_width + notch_tail + (1.0 * mm_unit)

    if not fill_strip:
        strip_width_mm = float(strip_width) / mm_unit
        visible_n = max(2, int(strip_width_mm / mm_per_sample) + 1)
        if centered_adc.size > visible_n:
            centered_adc = centered_adc[-visible_n:]

    adc_per_box = float(adc_per_box_multiplier) / max(1e-6, float(wave_gain_mm_mv))
    boxes_offset = centered_adc / adc_per_box
    ecg_normalized = center_y + (boxes_offset * box_height_points)

    if fill_strip:
        # Legacy 4:3 / hyperkalemia: map every sample across full strip width
        t = np.linspace(wave_start_x, strip_end_x, len(centered_adc))
        visible_mask = np.ones(len(centered_adc), dtype=bool)
    else:
        t = wave_start_x + (np.arange(centered_adc.size, dtype=float) * mm_per_sample * mm_unit)
        visible_mask = t <= strip_end_x

    if not np.any(visible_mask):
        baseline_path = Path(
            fillColor=None,
            strokeColor=colors.HexColor("#000000"),
            strokeWidth=0.4,
            strokeLineCap=1,
            strokeLineJoin=1,
        )
        baseline_path.moveTo(strip_x, center_y)
        baseline_path.lineTo(strip_x + strip_width, center_y)
        return baseline_path, notch_path, None

    t = t[visible_mask]
    ecg_normalized = ecg_normalized[visible_mask]

    trace_path = Path(
        fillColor=None,
        strokeColor=colors.HexColor("#000000"),
        strokeWidth=0.4,
        strokeLineCap=1,
        strokeLineJoin=1,
    )
    trace_path.moveTo(t[0], ecg_normalized[0])
    for i in range(1, len(t)):
        trace_path.lineTo(t[i], ecg_normalized[i])

    dotted_path = None
    gap_mm = (strip_end_x - float(t[-1])) / mm_unit
    if not fill_strip and gap_mm > 3.0:
        dotted_path = Path(
            fillColor=None,
            strokeColor=colors.HexColor("#000000"),
            strokeWidth=0.4,
            strokeLineCap=1,
            strokeLineJoin=1,
            strokeDashArray=[2, 3],
        )
        dotted_path.moveTo(t[-1], center_y)
        dotted_path.lineTo(strip_end_x, center_y)

    return trace_path, notch_path, dotted_path


def apply_report_ecg_filters(signal, sampling_rate, settings_manager):
    # Prepare strip (padding, settings-aware filters, baseline removal, smoothing)
    work = _prepare_report_strip_signal(signal, sampling_rate, settings_manager)

    try:
        # Post-process: ensure a stable edge and robust notch removal for reports
        from ecg.ecg_filters import notch_filter_butterworth

        # Always apply a gentle Butterworth notch at configured AC frequency (default 50Hz)
        try:
            ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip() if settings_manager else "50"
            notch_freq = float(ac_setting) if ac_setting and ac_setting not in ("off", "") else 50.0
        except Exception:
            notch_freq = 50.0

        # Apply AC (notch) filter using the dedicated implementation (IIR notch -> sosfiltfilt)
        try:
            from ecg.ecg_filters import apply_ac_filter
            work = apply_ac_filter(work, float(sampling_rate), str(int(notch_freq)))
        except Exception:
            try:
                work = notch_filter_butterworth(work, float(sampling_rate), freq=notch_freq, q=25.0)
            except Exception:
                pass

        # No edge taper — the strip keeps its recorded amplitude right to the last
        # sample. Transients are prevented upstream by real pre-roll plus reflect
        # padding in _prepare_report_strip_signal(), so fading the ends only
        # destroyed genuine beats near the strip boundary.
    except Exception:
        pass

    return work


def _estimate_rr_from_report_strip(ecg_test_page, ecg_data_file, sampling_rate, settings_manager, width_points=REPORT_STRIP_WIDTH_POINTS):
    """Estimate RR from the same Lead II strip window used in the 12:1 report."""
    try:
        leads_payload, fs = _collect_12_lead_payload(
            ecg_test_page,
            sampling_rate,
            ecg_data_file=ecg_data_file,
            window_seconds=STANDARD_REPORT_WINDOW_SECONDS,
        )
        lead_ii = leads_payload.get("II") if isinstance(leads_payload, dict) else None
        if lead_ii is None:
            return None

        signal = apply_report_ecg_filters(lead_ii, fs, settings_manager)
        if signal is None or len(signal) < max(3, int(fs * 2.0)):
            return None

        wave_speed_setting = settings_manager.get_setting("wave_speed")
        speed_mm_s = float(wave_speed_setting) if wave_speed_setting else None
        if not speed_mm_s or speed_mm_s <= 0:
            return None

        width_mm = float(width_points) / mm
        effective_speed_mm_s = speed_mm_s * ECG_SPEED_SCALE
        max_display_seconds = width_mm / max(effective_speed_mm_s, 1e-6)
        display_samples = max(1, min(len(signal), int(round(max_display_seconds * fs))))
        strip_signal = np.asarray(signal[-display_samples:], dtype=float)
        if len(strip_signal) < max(3, int(fs * 2.0)):
            return None

        from scipy.signal import find_peaks

        centered = strip_signal - np.nanmedian(strip_signal)
        diff_signal = np.diff(centered, prepend=centered[0])
        energy = diff_signal * diff_signal
        window = max(1, int(0.12 * fs))
        envelope = np.convolve(energy, np.ones(window) / window, mode="same")
        threshold = np.mean(envelope) + 0.8 * np.std(envelope)
        min_distance = max(1, int(0.25 * fs))
        peaks, _ = find_peaks(envelope, height=threshold, distance=min_distance)
        if len(peaks) < 2:
            return None

        rr_ms = np.diff(peaks) * (1000.0 / fs)
        rr_ms = rr_ms[(rr_ms >= 250.0) & (rr_ms <= 2000.0)]
        if len(rr_ms) == 0:
            return None

        return int(round(float(np.median(rr_ms))))
    except Exception as exc:
        print(f" RR strip estimation failed: {exc}")
        return None

def create_ecg_grid_with_waveform(ecg_data, lead_name, width=6, height=2):
    """
    Create ECG graph with pink grid background and dark ECG waveform
    Returns: matplotlib figure with pink ECG grid background
    """
    # Create figure with pink background
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    fig = _Figure(figsize=(width, height), facecolor='#ffe6e6')
    _FCA(fig)
    ax = fig.add_subplot(111)
    
    # STEP 1: Create pink ECG grid background
    # ECG grid colors (even lighter pink/red like medical ECG paper)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    bg_color = '#ffe6e6'  # Very light pink background
    
    # Set both figure and axes background to pink
    fig.patch.set_facecolor(bg_color)  # Figure background pink
    ax.set_facecolor(bg_color)         # Axes background pink
    
    # STEP 2: Draw pink ECG grid lines
    # Minor grid lines (1mm equivalent spacing) - LIGHT PINK
    minor_spacing_x = width / 60  # 60 minor divisions across width
    minor_spacing_y = height / 20  # 20 minor divisions across height
    
    # Draw vertical minor pink grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (5mm equivalent spacing) - DARKER PINK
    major_spacing_x = width / 12  # 12 major divisions across width
    major_spacing_y = height / 4   # 4 major divisions across height
    
    # Draw vertical major pink grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        ax.axhline(y=y_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # STEP 3: Plot DARK ECG waveform on top of pink grid
    if ecg_data is not None and len(ecg_data) > 0:
        # Scale ECG data to fit in the grid
        t = np.linspace(0, width, len(ecg_data))
        # Normalize ECG data to fit in height with some margin
        if np.max(ecg_data) != np.min(ecg_data):
            ecg_normalized = ((ecg_data - np.min(ecg_data)) / (np.max(ecg_data) - np.min(ecg_data))) * (height * 0.8) + (height * 0.1)
        else:
            ecg_normalized = np.full_like(ecg_data, height / 2)
        
        # DARK ECG LINE - clearly visible on pink grid
        ax.plot(t, ecg_normalized, color='#000000', linewidth=2.8, solid_capstyle='round', alpha=0.9)
    # REMOVE ENTIRE else BLOCK - just comment it out or delete lines 78-96
    
    # STEP 4: Set axis limits to match grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # STEP 5: Remove axis elements but keep the pink grid background
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    
    return fig

from reportlab.graphics.shapes import Drawing, Group, Line, Rect
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.lib.units import mm

def create_reportlab_ecg_drawing(lead_name, width=REPORT_STRIP_WIDTH_POINTS, height=45):
    """
    Create ECG drawing using ReportLab (NO matplotlib - NO white background issues)
    Returns: ReportLab Drawing with guaranteed pink background
    """
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines using the same 40-box report paper scale
    # used by waveform timing (1 large box = 5.25mm, 1 small box = 1.05mm).
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid

    minor_spacing_x = ECG_SMALL_BOX_MM * mm
    minor_spacing_y = ECG_SMALL_BOX_MM * mm
    
    # Vertical minor grid lines
    for i in range(int(width / minor_spacing_x) + 2):
        x_pos = i * minor_spacing_x
        if x_pos > width:
            break
        line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Horizontal minor grid lines
    for i in range(int(height / minor_spacing_y) + 2):
        y_pos = i * minor_spacing_y
        if y_pos > height:
            break
        line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)

    # Major grid lines: 1 large box horizontally, 2 large boxes vertically.
    major_spacing_x = ECG_LARGE_BOX_MM * mm
    major_spacing_y = (2.0 * ECG_LARGE_BOX_MM) * mm
    
    # Vertical major grid lines
    for i in range(int(width / major_spacing_x) + 2):
        x_pos = i * major_spacing_x
        if x_pos > width:
            break
        line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # Horizontal major grid lines
    for i in range(int(height / major_spacing_y) + 2):
        y_pos = i * major_spacing_y
        if y_pos > height:
            break
        line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # REMOVE ENTIRE "STEP 3: Draw ECG waveform as series of lines" section (lines ~166-214)
    
    return drawing

def capture_real_ecg_graphs_from_dashboard(dashboard_instance=None, ecg_test_page=None, samples_per_second=150, settings_manager=None):
    """
    Capture REAL ECG data from the live test page and create drawings
    Returns: dict with ReportLab Drawing objects containing REAL ECG data
    """
    lead_drawings = {}
    
    print(" Capturing REAL ECG data from live test page...")
    
    if settings_manager is None:
        from utils.settings_manager import SettingsManager
        settings_manager = SettingsManager()

    lead_sequence = settings_manager.get_setting("lead_sequence", "Standard")
    
    # Use the appropriate sequence for REPORT ONLY
    ordered_leads = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    # Map lead names to indices
    lead_to_index = {
        "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
    }
    
# Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = samples_per_second or 150
    
    # FORCE REAL DATA ONLY - DISABLE DEMO MODE FOR REPORTS
    is_demo_mode = False  # Always use real data for reports
    print(" REPORT GENERATION: Using REAL data only (Demo mode disabled for consistency)")
    
    # Try to get ACTUAL sampling rate from the test page
    if ecg_test_page and hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            samples_per_second = float(ecg_test_page.sampler.sampling_rate)
            print(f" Found sampler sampling rate: {samples_per_second}Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'sampling_rate'):
        if ecg_test_page.sampling_rate:
            samples_per_second = float(ecg_test_page.sampling_rate)
            print(f" Found page sampling rate: {samples_per_second}Hz")
    real_ecg_data = {}
    if ecg_test_page and hasattr(ecg_test_page, "data"):
        # ALWAYS USE REAL DATA - No demo mode windowing
        num_samples_to_capture = 10000
        print(f" REAL DATA MODE: Capturing up to {num_samples_to_capture} samples from live buffer")
        
        for lead in ordered_leads:
            if lead == "-aVR":
                if hasattr(ecg_test_page, "data") and len(ecg_test_page.data) > 3:
                    avr_data = np.array(ecg_test_page.data[3])
                    real_ecg_data[lead] = -avr_data[-num_samples_to_capture:]
                    print(f" Captured REAL -aVR data: {len(real_ecg_data[lead])} points")
            else:
                lead_index = lead_to_index.get(lead)
                if lead_index is not None and len(ecg_test_page.data) > lead_index:
                    lead_data = np.array(ecg_test_page.data[lead_index])
                    if len(lead_data) > 0:
                        real_ecg_data[lead] = lead_data[-num_samples_to_capture:]
                        print(f" Captured REAL {lead} data: {len(real_ecg_data[lead])} points")
                    else:
                        print(f" No data found for {lead}")
                else:
                    print(f" Lead {lead} index not found")
    else:
        print(" No live ECG test page found - using grid only")
    
    # Get wave_gain from settings_manager for amplitude scaling
    wave_gain_mm_mv = 10.0  # Default
    if settings_manager:
        try:
            wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
            wave_gain_mm_mv = float(wave_gain_setting) if wave_gain_setting else 10.0
            print(f" Using wave_gain from ecg_settings.json: {wave_gain_mm_mv} mm/mV (for amplitude scaling)")
        except Exception:
            wave_gain_mm_mv = 10.0
            print(f" Could not get wave_gain from settings, using default: {wave_gain_mm_mv} mm/mV")
    
    # Create ReportLab drawings with REAL ECG data
    for lead in ordered_leads:
        try:
            # Create ReportLab drawing with REAL ECG data (with wave_gain applied)
            drawing = create_reportlab_ecg_drawing_with_real_data(
                lead, 
                real_ecg_data.get(lead), 
                width=REPORT_STRIP_WIDTH_POINTS, 
                height=45,
                wave_gain_mm_mv=wave_gain_mm_mv,
                sampling_rate=samples_per_second,
                settings_manager=settings_manager
            )
            lead_drawings[lead] = drawing
            
            if lead in real_ecg_data:
                print(f" Created drawing with MAXIMUM data for Lead {lead} - showing 7+ heartbeats")
            else:
                print(f"Created grid-only drawing for Lead {lead}")
            
        except Exception as e:
            print(f" Error creating drawing for Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    if is_demo_mode and time_window_seconds is not None:
        print(f" Successfully created {len(lead_drawings)}/12 ECG drawings with DEMO window filtering ({time_window_seconds}s window - visible peaks only)!")
    else:
        print(f" Successfully created {len(lead_drawings)}/12 ECG drawings with MAXIMUM heartbeats!")
    return lead_drawings

def create_reportlab_ecg_drawing_with_real_data(lead_name, ecg_data, width=REPORT_STRIP_WIDTH_POINTS, height=45, wave_gain_mm_mv=None, sampling_rate=500.0, settings_manager=None):
    """
    Create ECG drawing using ReportLab with REAL ECG data using proper time-based scaling
    Returns: ReportLab Drawing with guaranteed pink background and REAL ECG waveform
    
    Parameters:
        wave_gain_mm_mv: Wave gain in mm/mV (dynamic from settings)
        sampling_rate: Sampling rate in Hz (default: 500.0 Hz)
        settings_manager: Settings manager for dynamic wave_speed
    """
    # VALIDATION: Check for required parameters
    if wave_gain_mm_mv is None:
        print(f" ERROR: wave_gain_mm_mv is None for {lead_name} - cannot generate drawing")
        return drawing
        
    if settings_manager is None:
        print(f" ERROR: settings_manager is None for {lead_name} - cannot generate drawing")
        return drawing
    
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines using the report's actual paper scale.
    # The report layout uses 40 large boxes across A4 width, so:
    #   large box = ECG_LARGE_BOX_MM (5.25mm)
    #   small box = ECG_SMALL_BOX_MM (1.05mm)
    # The waveform speed scaling already uses ECG_SPEED_SCALE, so the paper grid
    # must use the same box dimensions or the waves-per-box timing drifts.
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    from reportlab.lib.units import mm
    
    # Minor grid: 1 small box spacing.
    minor_spacing_mm = ECG_SMALL_BOX_MM * mm
    minor_spacing_x_points = minor_spacing_mm
    minor_spacing_y_points = minor_spacing_mm
    
    # Vertical minor lines
    num_minor_x = int(width / minor_spacing_x_points) + 1
    for i in range(num_minor_x):
        x_pos = i * minor_spacing_x_points
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Horizontal minor lines
    num_minor_y = int(height / minor_spacing_y_points) + 1
    for i in range(num_minor_y):
        y_pos = i * minor_spacing_y_points
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
            drawing.add(line)
    
    # Major grid: 1 large box horizontally, 2 large boxes vertically.
    major_spacing_x_mm = ECG_LARGE_BOX_MM * mm
    major_spacing_y_mm = (2.0 * ECG_LARGE_BOX_MM) * mm
    
    # Vertical major lines (every 0.20s = 5mm)
    num_major_x = int(width / major_spacing_x_mm) + 1
    for i in range(num_major_x):
        x_pos = i * major_spacing_x_mm
        if x_pos <= width:
            line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
    # Horizontal major lines (every 1.0mV = 10mm)
    num_major_y = int(height / major_spacing_y_mm) + 1
    for i in range(num_major_y):
        y_pos = i * major_spacing_y_mm
        if y_pos <= height:
            line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
            drawing.add(line)
    
    # STEP 3: Plot ECG in fixed diagnostic scale (25 mm/s, 10 mm/mV) with no autoscale
    if ecg_data is None or len(ecg_data) == 0:
        print(f" No real data available for {lead_name} - showing grid only")
        return drawing

    # ── FIX: Report ALWAYS at 25mm/s regardless of display setting ─────────
    speed_mm_s = 25.0  # Standard clinical ECG paper speed
    # (Display setting may be 12.5 or 50 mm/s but report is always standard)
    # ─────────────────────────────────────────────────────────────────────────
        
    width_mm = width / mm  # Convert width points to mm
    total_seconds = width_mm / (speed_mm_s * 1.05)  # ECG_SPEED_SCALE = 1.05
    height_mm_physical = height / mm  # Convert height points to mm

    ecg_array = np.asarray(ecg_data, dtype=float)
    med_abs = np.nanmedian(np.abs(ecg_array)) if len(ecg_array) else 0.0
    ecg_mv = ecg_array / REPORT_WAVEFORM_ADC_DIVISOR if med_abs > 20.0 else ecg_array

    fs = float(sampling_rate)

    # PROPER ECG SCALING: Use actual ECG paper scaling (25mm/s with proper grid alignment)
    # This ensures medically accurate time representation
    effective_speed_mm_s = speed_mm_s * 1.05  # ECG_SPEED_SCALE = 1.05
    
    # Calculate how many seconds of data we can show in the available width
    max_display_seconds = width_mm / effective_speed_mm_s
    
    # Use the smaller of: available data time or max display time
    actual_data_seconds = len(ecg_mv) / fs
    display_seconds = min(actual_data_seconds, max_display_seconds)
    display_samples = max(1, min(len(ecg_mv), int(round(display_seconds * fs))))

    print(f" Available data: {len(ecg_mv)} points, Time window: {total_seconds:.2f}s")

    if len(ecg_mv) == 0:
        print(f" ECG data empty for {lead_name}")
        return drawing

    try:
        ecg_mv = _prepare_report_strip_signal(
            ecg_mv,
            fs,
            settings_manager,
            target_samples=display_samples,
        )
    except Exception as _fe:
        print(f"[Report] Stable strip prep failed for {lead_name}: {_fe}")
        ecg_mv = np.asarray(ecg_mv[-display_samples:], dtype=float)

    if len(ecg_mv) == 0:
        return drawing

    display_seconds = len(ecg_mv) / fs
    print(f" {lead_name}: Using latest {len(ecg_mv)} samples for report strip")

    t_sec = np.arange(len(ecg_mv)) / fs

    # Gain once: mm per mV (AFTER all processing and final window selection)
    y_mm = ecg_mv * wave_gain_mm_mv
    baseline_mm = height_mm_physical / 2.0
    y_mm = baseline_mm + y_mm

    # Convert to physical mm positions (proper ECG scaling)
    x_mm = t_sec * effective_speed_mm_s
    
    print(f" Display: {len(ecg_mv)} points, {display_seconds:.2f}s of {actual_data_seconds:.2f}s available")

    # Clip to panel
    y_mm = np.clip(y_mm, 0.0, height_mm_physical)
    x_mm = np.clip(x_mm, 0.0, width_mm)

    # Convert to points for plotting
    points = list(zip(x_mm, y_mm))

    # Draw as line segments
    ecg_color = colors.HexColor("#000000")
    for i in range(len(points) - 1):
        x1 = points[i][0] * mm
        y1 = points[i][1] * mm
        x2 = points[i+1][0] * mm
        y2 = points[i+1][1] * mm
        drawing.add(Line(x1, y1, x2, y2, strokeColor=ecg_color, strokeWidth=0.6))
    
    return drawing

def create_clean_ecg_image(lead_name, width=6, height=2):
    """
    Create COMPLETELY CLEAN ECG image with GUARANTEED pink background
    NO labels, NO time markers, NO axes, NO white background
    """
    # FORCE matplotlib to use proper backend
    import matplotlib
    # matplotlib.use('Agg') # Safe
    import matplotlib.pyplot as plt
    
    # STEP 1: Create figure with FORCED pink background
    from matplotlib.figure import Figure as _Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg as _FCA
    fig = _Figure(figsize=(width, height), facecolor='#ffe6e6')
    _FCA(fig)
    
    # FORCE figure background to pink
    fig.patch.set_facecolor('#ffe6e6')
    fig.patch.set_alpha(1.0)  # Full opacity
    
    # Create axes with FORCED pink background
    ax = fig.add_subplot(111)
    ax.set_facecolor('#ffe6e6')  # FORCE axes background pink
    ax.patch.set_facecolor('#ffe6e6')  # FORCE axes patch pink
    ax.patch.set_alpha(1.0)  # Full opacity
    
    # STEP 2: Draw pink ECG grid lines OVER pink background (darker for clarity)
    light_grid_color = '#ffd1d1'  # Darker minor grid
    major_grid_color = '#ffb3b3'  # Darker major grid
    
    # Minor grid lines (1mm equivalent spacing)
    minor_spacing_x = width / 60  # 60 minor divisions
    minor_spacing_y = height / 20  # 20 minor divisions
    
    # Draw vertical minor pink grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        ax.axvline(x=x_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Draw horizontal minor pink grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        ax.axhline(y=y_pos, color=light_grid_color, linewidth=0.6, alpha=0.8)
    
    # Major grid lines (5mm equivalent spacing)
    major_spacing_x = width / 12  # 12 major divisions
    major_spacing_y = height / 4   # 4 major divisions
    
    # Draw vertical major pink grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        ax.axvline(x=x_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # Draw horizontal major pink grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        ax.axhline(y=y_pos, color=major_grid_color, linewidth=1.0, alpha=0.9)
    
    # REMOVE ENTIRE "STEP 3: Create realistic ECG waveform" section (lines ~315-356)
    # REMOVE ENTIRE "STEP 4: Plot DARK ECG line" section
    
    # STEP 5: Set limits and remove ALL visual elements except grid
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    
    # COMPLETELY remove ALL spines, ticks, labels
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('')
    ax.axis('off')  # FORCE turn off all axis elements
    
    # Remove any text objects
    for text in ax.texts:
        text.set_visible(False)
    
    # FORCE tight layout with pink background
    fig.tight_layout(pad=0)
    
    return fig


def get_dashboard_conclusions_from_image(dashboard_instance):
    """
    Load dynamic conclusions from JSON file (saved by dashboard)
    Returns: List of clean conclusion headings (up to 12 conclusions)
    """
    conclusions = []
    
    # **NEW: Try to load from JSON file first (DYNAMIC)**
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        conclusions_file = str(data_file("last_conclusions.json"))
        
        print(f" Looking for conclusions at: {conclusions_file}")
        
        if os.path.exists(conclusions_file):
            with open(conclusions_file, 'r') as f:
                conclusions_data = json.load(f)
            
            print(f" Loaded JSON data: {conclusions_data}")
            
            # Extract findings from JSON
            findings = _normalize_report_conclusions(conclusions_data.get('findings', []))
            
            if findings:
                conclusions = findings[:12]  # Take up to 12 conclusions
                print(f" Loaded {len(conclusions)} DYNAMIC conclusions from JSON file")
                for i, conclusion in enumerate(conclusions, 1):
                    print(f"   {i}. {conclusion}")
            else:
                print(" No findings in JSON file")
        else:
            print(f" Conclusions JSON file not found: {conclusions_file}")
    
    except Exception as json_err:
        print(f" Error loading conclusions from JSON: {json_err}")
        import traceback
        traceback.print_exc()
    
    # **REMOVED: Old code that extracted from dashboard_instance.conclusion_box**
    # **REMOVED: Fallback default conclusions**
    
    # If still no conclusions found, use minimal fallback
    if not conclusions:
        conclusions = [
            "No ECG data available",
            "Please connect device",
           
            
        ]
        print(" Using zero-value fallback (no ECG data available)")
    
    # Ensure we have exactly 12 conclusions (pad with empty strings if needed)
    MAX_CONCLUSIONS = 12
    while len(conclusions) < MAX_CONCLUSIONS:
        conclusions.append("---")  # Use "---" for empty slots
    
    # Limit to maximum 12 conclusions
    conclusions = conclusions[:MAX_CONCLUSIONS]
    
    print(f" Final conclusions list (12 total): {len([c for c in conclusions if c and c != '---'])} filled, {len([c for c in conclusions if not c or c == '---'])} blank")
    
    return conclusions


def get_conclusions_from_ecg_test_page(ecg_test_page, data=None):
    """
    Prefer conclusions generated by the expanded lead view / ECG test page.
    Returns a list of short conclusion strings (may be empty).
    """
    conclusions = []
    try:
        if not ecg_test_page:
            return conclusions

        # 1) Latest arrhythmia interpretation (expanded view keeps this in sync)
        arr = None
        try:
            if hasattr(ecg_test_page, 'get_latest_rhythm_interpretation'):
                arr = ecg_test_page.get_latest_rhythm_interpretation()
        except Exception:
            arr = getattr(ecg_test_page, '_latest_rhythm_interpretation', None)

        if arr and arr not in ("", "Analyzing Rhythm...", "Detecting..."):
            conclusions.extend(_split_conclusion_text(arr))

        # 2) Use _last_analysis metrics if available to add interval conclusions
        la = getattr(ecg_test_page, '_last_analysis', None)
        if isinstance(la, dict):
            for label in la.get('arrhythmias', []) or []:
                if label:
                    conclusions.extend(_split_conclusion_text(str(label)))

        # If _last_analysis exists, also include numeric metric summaries (HR/PR/QRS/QTc)
        if la is not None:
            try:
                hr = int(float(data.get('HR_bpm') or data.get('Heart_Rate') or data.get('HR') or 0)) if data else None
            except Exception:
                hr = None
            try:
                pr = int(float(data.get('PR') or 0)) if data else None
            except Exception:
                pr = None
            try:
                qrs = int(float(data.get('QRS') or 0)) if data else None
            except Exception:
                qrs = None
            try:
                qtc = int(float(data.get('QTc') or 0)) if data else None
            except Exception:
                qtc = None

            if hr and hr > 0:
                conclusions.append(f"Heart Rate: {hr} BPM")
            if pr and pr > 0:
                conclusions.append(f"PR Interval: {pr} ms")
            if qrs and qrs > 0:
                conclusions.append(f"QRS Duration: {qrs} ms")
            if qtc and qtc > 0:
                conclusions.append(f"QTc: {qtc} ms")

        has_abnormal = _has_abnormal_conclusion(conclusions)
        if not has_abnormal:
            for label in _build_metric_conclusions(data or {}):
                conclusions.append(label)

        # Deduplicate while preserving order
        return _normalize_report_conclusions(conclusions)
    except Exception:
        return []


def _split_conclusion_text(text):
    """Split combined rhythm text into report-safe individual findings."""
    if not text:
        return []
    raw = str(text).replace("•", ",").replace("\n", ",").strip()
    if raw.lower().startswith("rhythm:"):
        raw = raw.split(":", 1)[1].strip()
    if raw.lower().startswith("arrhythmia interpretation:"):
        raw = raw.split(":", 1)[1].strip()

    parts = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        # Strip metric suffixes from dashboard lines like "Wide QRS - 116 ms".
        if " - " in item and any(key in item for key in ("Wide QRS", "Borderline Wide QRS", "Prolonged PR", "Short PR")):
            item = item.split(" - ", 1)[0].strip()
        parts.append(item)
    return parts


def _has_abnormal_conclusion(conclusions):
    abnormal_keywords = (
        "Block", "Fibrillation", "Flutter", "Tachycardia", "Bradycardia",
        "PVC", "PAC", "Wide QRS", "Prolonged", "Short", "Asystole",
        "Ventricular"
    )
    return any(
        any(keyword.lower() in str(item).lower() for keyword in abnormal_keywords)
        for item in conclusions
    )


# ══════════════════════════════════════════════════════════════════════════════
# REPORT CONCLUSION ALLOW-LIST
# ══════════════════════════════════════════════════════════════════════════════
# The printed CONCLUSION box carries these five findings and nothing else.
#
# They share one property: each is derived directly from a measured value that
# is also printed in the report header (HR, QRS, QTc), so a reader can check the
# conclusion against the numbers on the same page. The morphology- and
# rhythm-classifier labels are deliberately excluded — those detectors proved
# unreliable in the field (a normal 65 bpm sinus ECG was reported as
# "Ventricular Fibrillation"), and a wrong lethal label on a signed report is
# worse than no label.
#
# CONSEQUENCE, STATED PLAINLY: the report will not name Asystole, Ventricular
# Fibrillation, Ventricular Tachycardia, Atrial Fibrillation/Flutter, AV block
# or bundle branch block even when the analyser detects them. The waveform is
# still printed in full and the intervals are still measured and shown; the
# interpretation of anything beyond rate, QRS width and QTc is left to the
# reading clinician.
REPORT_ALLOWED_CONCLUSIONS = (
    "Normal Sinus Rhythm",
    "Sinus Bradycardia",
    "Sinus Tachycardia",
    "Wide QRS",
    "Prolonged QTc",
)

# Different spellings of the SAME permitted finding are folded in rather than
# dropped, so a real finding is never lost to wording alone.
_CONCLUSION_CANONICAL = {
    "normal sinus rhythm": "Normal Sinus Rhythm",
    "sinus rhythm": "Normal Sinus Rhythm",          # emitted when P waves undetected
    "nsr": "Normal Sinus Rhythm",
    "sinus bradycardia": "Sinus Bradycardia",
    "bradycardia": "Sinus Bradycardia",
    "athlete bradycardia": "Sinus Bradycardia",
    "sinus tachycardia": "Sinus Tachycardia",
    "tachycardia": "Sinus Tachycardia",
    "wide qrs": "Wide QRS",
    "wide qrs complex": "Wide QRS",
    "prolonged qtc": "Prolonged QTc",
    "prolonged qtc interval": "Prolonged QTc",
    "long qt syndrome": "Prolonged QTc",            # QTc > 500 is still prolonged
    "prolonged qtcf interval (fridericia)": "Prolonged QTc",
}


def restrict_to_allowed_conclusions(conclusions):
    """Fold spelling variants, then keep only the permitted findings.

    Order is preserved: rhythm first (it is always produced first), then Wide
    QRS, then Prolonged QTc — matching REPORT_ALLOWED_CONCLUSIONS.
    """
    kept = []
    for item in conclusions or []:
        text = str(item).strip()
        if not text or text == "---":
            continue
        canonical = _CONCLUSION_CANONICAL.get(text.lower(), text)
        if canonical in REPORT_ALLOWED_CONCLUSIONS and canonical not in kept:
            kept.append(canonical)
    order = {label: i for i, label in enumerate(REPORT_ALLOWED_CONCLUSIONS)}
    kept.sort(key=lambda c: order[c])
    return kept


def ensure_rate_conclusion(conclusions, data):
    """Guarantee the box always states the rate finding when HR is measurable.

    The rhythm rules are an if/elif chain whose first matches are labels the
    allow-list removes — e.g. HR < 40 with no measurable PR yields
    "Third-degree AV Block". Filtering that left a profoundly bradycardic
    patient with a completely EMPTY conclusion box, which reads as "nothing
    found" rather than "we are not reporting that class of finding".

    The rate itself is always reportable, so it is restated here. Note this
    uses the same wording the chain already uses at HR 40-59 with no PR
    ("Sinus Bradycardia"), so it introduces no claim the rules were not
    already making.
    """
    rhythm_labels = ("Normal Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia")
    if any(c in rhythm_labels for c in (conclusions or [])):
        return list(conclusions or [])

    hr = 0.0
    for key in ("HR_bpm", "Heart_Rate", "HR", "beat", "HR_avg"):
        try:
            raw = (data or {}).get(key)
            if raw is not None and str(raw).strip() != "":
                hr = float(str(raw).lower().replace("bpm", "").strip())
                break
        except Exception:
            continue

    if hr <= 0:
        return list(conclusions or [])

    if hr < 60:
        label = "Sinus Bradycardia"
    elif hr > 100:
        label = "Sinus Tachycardia"
    else:
        label = "Normal Sinus Rhythm"
    return [label] + list(conclusions or [])


def _normalize_report_conclusions(conclusions):
    """Normalize report conclusions so real arrhythmias beat metric fallbacks."""
    expanded = []
    for item in conclusions or []:
        expanded.extend(_split_conclusion_text(item))

    # Only a finding that will ACTUALLY BE PRINTED may suppress the rhythm line.
    #
    # Without this, a QRS of 116 ms produced an empty conclusion box: it raised
    # "Borderline Wide QRS", which set abnormal=True and dropped "Normal Sinus
    # Rhythm", and the allow-list then removed Borderline Wide QRS as well —
    # leaving nothing at all on the report.
    #
    # Wide QRS and Prolonged QTc are also excluded from the decision. They are
    # conduction/repolarisation findings, not competing rhythm diagnoses, so
    # "Normal Sinus Rhythm" plus "Wide QRS" is a coherent pair and the rhythm
    # should still be stated. Within the permitted set the rhythm labels are
    # mutually exclusive anyway, so nothing else can contradict it.
    _printable_now = restrict_to_allowed_conclusions(expanded)
    abnormal = _has_abnormal_conclusion(
        [c for c in _printable_now if c not in ("Wide QRS", "Prolonged QTc")]
    )
    drop_when_abnormal = {
        "Normal Sinus Rhythm",
        "Normal sinus rhythm",
        "Normal heart rate",
    }
    metric_prefixes = (
        "Heart Rate:",
        "PR Interval:",
        "QRS Duration:",
        "QTc:",
    )

    priority = [
        "Asystole",
        "Ventricular Fibrillation",
        "Ventricular Tachycardia",
        "Atrial Fibrillation",
        "Atrial Flutter",
        "Third-degree AV Block",
        "Second-degree AV Block (Mobitz II)",
        "Second-degree AV Block (Mobitz I)",
        "Sinus Bradycardia",
        "Bradycardia (non-sinus)",
        "Bradycardia",
        "Sinus Tachycardia",
        "Tachycardia (non-sinus)",
        "Tachycardia",
        "Complete Left Bundle Branch Block",
        "Complete Right Bundle Branch Block",
        "Left Bundle Branch Block",
        "Right Bundle Branch Block",
        "First-degree AV Block (Prolonged PR)",
        "Wide QRS",
        "Borderline Wide QRS",
        "Prolonged QTc",
        "Long QT Syndrome",
        "Normal Sinus Rhythm",
    ]

    seen = set()
    cleaned = []
    canonical_expanded = []
    for item in expanded:
        s = str(item).strip()
        if " (HR" in s:
            s = s.split(" (HR", 1)[0].strip()
        canonical_expanded.append(s)
    has_atrial_rhythm = any(
        str(item).lower() in {
            "atrial fibrillation",
            "atrial fibrillation detected",
            "atrial flutter",
            "possible atrial flutter",
        }
        for item in canonical_expanded
    )
    has_bundle_branch_block = any(
        "bundle branch block" in str(item).lower()
        for item in canonical_expanded
    )
    ignore_values = {
        "No specific arrhythmia detected.",
        "No specific arrhythmia detected",
        "Analyzing Rhythm...",
        "Detecting...",
    }
    for item in expanded:
        s = str(item).strip()
        if not s or s == "---" or s in ignore_values:
            continue
        if " (HR" in s:
            s = s.split(" (HR", 1)[0].strip()
        if s.lower().startswith(("bradycardia - hr:", "tachycardia - hr:")):
            s = s.split(" - HR", 1)[0].strip()
        aliases = {
            "possible atrial flutter": "Atrial Flutter",
            "atrial flutter": "Atrial Flutter",
            "atrial fibrillation": "Atrial Fibrillation",
            "atrial fibrillation detected": "Atrial Fibrillation",
            "1st-degree av block": "First-degree AV Block (Prolonged PR)",
            "first-degree av block": "First-degree AV Block (Prolonged PR)",
            "first-degree av block (prolonged pr)": "First-degree AV Block (Prolonged PR)",
        }
        s = aliases.get(s.lower(), s)
        if has_atrial_rhythm and "av block" in s.lower():
            continue
        if abnormal and (s in drop_when_abnormal or s.startswith(metric_prefixes)):
            continue
        if has_bundle_branch_block and s in ("Wide QRS", "Borderline Wide QRS"):
            continue
        key = s.lower()
        if key not in seen:
            cleaned.append(s)
            seen.add(key)

    def _rank(label):
        return priority.index(label) if label in priority else len(priority)

    cleaned.sort(key=_rank)

    # Final gate: the printed conclusion carries only the permitted findings.
    # Applied here because every conclusion source in this module funnels
    # through this function, so no path can bypass it.
    return restrict_to_allowed_conclusions(cleaned)


def _build_metric_conclusions(data):
    """Create value-based report findings from measured HR/PR/QRS/QTc."""
    if not isinstance(data, dict):
        return []

    def _num(*keys):
        for key in keys:
            try:
                value = data.get(key)
                if value is not None and value != "":
                    s = str(value).strip().lower()
                    clean = s.replace("bpm", "").replace("ms", "").replace("mv", "").replace("deg", "").replace("°", "").strip()
                    return float(clean)
            except Exception:
                continue
        return 0.0

    hr = _num("HR_bpm", "Heart_Rate", "HR", "beat", "HR_avg")
    pr = _num("PR", "PR_ms", "pr_ms")
    qrs = _num("QRS", "QRS_ms", "qrs_ms")
    qtc = _num("QTc", "QTc_ms", "qtc_bazett")

    if all(v <= 0 for v in (hr, pr, qrs, qtc)):
        return []

    findings = []
    if hr < 5:
        findings.append("Asystole")
    elif hr > 150 and qrs == 0:
        findings.append("Ventricular Fibrillation")
    elif 0 < hr < 40 and pr == 0:
        findings.append("Third-degree AV Block")
    elif 0 < hr < 60:
        findings.append("Sinus Bradycardia")
    elif hr > 100:
        findings.append("Sinus Tachycardia")
    elif hr >= 60:
        findings.append("Normal Sinus Rhythm")

    if pr > 200:
        findings.append("First-degree AV Block (Prolonged PR)")
    elif 0 < pr < 120:
        findings.append("Short PR Interval")

    if qrs >= 120:
        findings.append("Wide QRS")
    elif qrs >= 110:
        findings.append("Borderline Wide QRS")

    if qtc > 500:
        findings.append("Long QT Syndrome")
    elif qtc > 460:
        findings.append("Prolonged QTc")

    return findings


def load_latest_metrics_entry(reports_dir):
    """
    Return the most recent metrics entry from reports/metrics.json, if available.
    """
    metrics_path = os.path.join(reports_dir, 'metrics.json')
    if not os.path.exists(metrics_path):
        return None
    try:
        with open(metrics_path, 'r') as f:
            data = json.load(f)

        if isinstance(data, list) and data:
            return data[-1]

        if isinstance(data, dict):
            # support older shape where 'entries' may list the items
            entries = data.get('entries')
            if isinstance(entries, list) and entries:
                return entries[-1]

            # if dict already looks like one entry, return it
            if data.get('timestamp'):
                return data
    except Exception as e:
        print(f" Could not read metrics file for HR: {e}")

    return None



def _load_signup_details_for_username(username):
    """Load signup/profile details for a username from users.json if available."""
    if not username:
        return {}
    try:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
        users_path = str(data_file("users.json"))
        if not os.path.exists(users_path):
            return {}
        with open(users_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            key = str(username)
            val = raw.get(key, {})
            if isinstance(val, dict):
                return val
            key_lower = key.strip().lower()
            for uname, record in raw.items():
                if not isinstance(record, dict):
                    continue
                if str(uname).strip() == key:
                    return record
                phone = str(record.get("phone", "") or record.get("contact", "")).strip()
                full_name = str(record.get("full_name", "") or "").strip()
                login_key = str(
                    record.get("username")
                    or record.get("login_username")
                    or record.get("login_identifier")
                    or record.get("canonical_username")
                    or ""
                ).strip()
                if key and (phone == key or login_key == key):
                    return record
                if key_lower and (full_name.lower() == key_lower or phone.lower() == key_lower or login_key.lower() == key_lower):
                    return record
            return {}
        return {}
    except Exception:
        return {}


def _collect_12_lead_payload(ecg_test_page, sampling_rate, ecg_data_file=None, window_seconds=10.0):
    """Collect latest 12-lead ECG payload for backend sync."""
    try:
        fs = float(sampling_rate) if sampling_rate else 500.0
    except Exception:
        fs = 500.0
    n = max(1, int(round(window_seconds * fs)))

    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    # Priority: saved ecg_data_file when provided (stable/report-consistent)
    if ecg_data_file and os.path.exists(ecg_data_file):
        try:
            with open(ecg_data_file, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            leads = saved.get('leads') if isinstance(saved, dict) else None
            if isinstance(leads, dict) and leads:
                out = {}
                for ln in lead_names:
                    arr = leads.get(ln) or leads.get(f'Lead_{ln}') or []
                    if isinstance(arr, list):
                        out[ln] = arr[-n:] if len(arr) > n else arr
                if out:
                    return out, fs
        except Exception:
            pass

    # Fallback: live ecg_test_page data buffers
    out = {}
    try:
        if ecg_test_page and hasattr(ecg_test_page, 'data') and isinstance(ecg_test_page.data, (list, tuple)):
            for idx, ln in enumerate(lead_names):
                if idx < len(ecg_test_page.data):
                    arr = ecg_test_page.data[idx]
                    if isinstance(arr, (list, tuple, np.ndarray)) and len(arr) > 0:
                        arr_list = list(arr)
                        out[ln] = arr_list[-n:] if len(arr_list) > n else arr_list
        return out, fs
    except Exception:
        return {}, fs




def _lead_has_signal(lead_samples) -> bool:
    """Return True when a lead has any non-zero samples worth treating as present."""
    try:
        arr = np.asarray(lead_samples, dtype=float)
        if arr.size == 0:
            return False
        return not np.allclose(arr, 0.0)
    except Exception:
        return False
def _sync_report_package_to_backend(
    filename,
    patient,
    data,
    metrics_payload,
    username,
    ecg_test_page,
    sampling_rate,
    ecg_data_file=None,
    report_type="12_lead_ecg",
):
    """Sync generated report package (metrics + waveform + signup + ECG details) to backend."""
    try:
        from utils.backend_api import get_backend_api

        backend = get_backend_api()
        if not backend.is_enabled():
            print('  Backend sync disabled')
            return

        signup_details = _load_signup_details_for_username(username)
        leads_payload, fs = _collect_12_lead_payload(ecg_test_page, sampling_rate, ecg_data_file=ecg_data_file)

        device_serial = str((patient or {}).get('serial_number') or data.get('machine_serial') or signup_details.get('serial_id') or 'UNKNOWN')
        device_info = {
            'machine_serial': data.get('machine_serial', ''),
            'app': 'cardiox',
            'report_type': report_type,
            'report_file': os.path.abspath(filename),
        }

        sid = backend.start_session(device_serial=device_serial, device_info=device_info)
        print(f'  Backend session started: {sid}')

        metric_result = backend.upload_metrics({
            'report_generated_at': datetime.now().isoformat(),
            'username': username or '',
            'master_phone': (signup_details or {}).get('phone') or (signup_details or {}).get('master_phone') or username or '',
            'patient': patient or {},
            'signup': signup_details,
            'metrics': metrics_payload or {},
        })
        print(f"  Backend metrics sync: {metric_result.get('status')}")

        if leads_payload:
            wave_result = backend.upload_waveform(leads_payload, int(round(fs)))
            print(f"  Backend waveform sync: {wave_result.get('status')} ({len(leads_payload)} leads)")
        else:
            print('  Backend waveform sync skipped: no lead data available')

        report_meta = {
            'username': username or '',
            'master_phone': (signup_details or {}).get('phone') or (signup_details or {}).get('master_phone') or username or '',
            'patient': patient or {},
            'signup': signup_details,
            'metrics': metrics_payload or {},
            'ecg_details': {
                'sampling_rate': fs,
                'lead_count': len(leads_payload),
                'ecg_data_file': os.path.abspath(ecg_data_file) if ecg_data_file else '',
            }
        }
        report_result = backend.upload_report(filename, metadata=report_meta)
        print(f"  Backend report sync: {report_result.get('status')}")

        end_result = backend.end_session({
            'status': 'report_generated',
            'report_file': os.path.abspath(filename),
            'lead_count': len(leads_payload),
            'username': username or '',
        })
        print(f"  Backend session end: {end_result.get('status')}")

    except Exception as be:
        print(f"  Backend sync error: {be}")

def generate_ecg_report(
    filename="ecg_report.pdf",
    data=None,
    lead_images=None,
    dashboard_instance=None,
    ecg_test_page=None,
    patient=None,
    ecg_data_file=None,
    conclusions=None,
    log_history=False,
    username=None,
):
    """
    Generate ECG report PDF
    
    Parameters:
        ecg_data_file: Optional path to saved ECG data file. 
                       If provided, will load from file instead of live ecg_test_page.
                       If None and ecg_test_page provided, will save data first.
    
    Example:
        # Option 1: Save data first, then generate report
        saved_file = save_ecg_data_to_file(ecg_test_page)
        generate_ecg_report("report.pdf", data=metrics, ecg_test_page=ecg_test_page, ecg_data_file=saved_file)
        
        # Option 2: Generate report and auto-save data
        generate_ecg_report("report.pdf", data=metrics, ecg_test_page=ecg_test_page)
        # Data will be automatically saved before report generation
    """
   
    # Ensure we use a single canonical PDF path everywhere (save + upload).
    filename = os.path.abspath(filename)
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    except Exception:
        pass

    # Ensure mm is available in local scope
    from reportlab.lib.units import mm
    
    # Main function body starts here
    if data is None:
        # When no device connected or demo off - show ZERO values (not dummy values)
        data = {
            "HR": 0,
            "beat": 0,
            "PR": 0,
            "QRS": 0,
            "QT": 0,
            "QTc": 0,
            "ST": 0,
            "HR_max": 0,
            "HR_min": 0,
            "HR_avg": 0,
            "Heart_Rate": 0,  # Add for compatibility with dashboard
        }

    # Define base_dir and reports_dir for file operations
    reports_dir = str(data_file("reports"))
    os.makedirs(reports_dir, exist_ok=True)

    from utils.settings_manager import SettingsManager
    settings_manager = SettingsManager()

    def _safe_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    # ==================== STEP 1: Resolve HR_bpm with click-snapshot priority ====================
    # IMPORTANT:
    # Report generation is triggered from a live dashboard snapshot. Those values
    # must win over historical metrics.json values; otherwise, first-click report
    # can use stale HR/RR and produce wrong QTc/QTcF. metrics.json is fallback only.
    latest_metrics = load_latest_metrics_entry(reports_dir)
    hr_bpm_value = 0
    
    # Priority 1: current report payload (captured at click time)
    if hr_bpm_value == 0:
        hr_candidate = data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR")
        hr_bpm_value = _safe_int(hr_candidate)
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from report snapshot: {hr_bpm_value} bpm")

    # Priority 2: metrics.json latest (fallback only)
    if hr_bpm_value == 0 and latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from metrics.json fallback: {hr_bpm_value} bpm")
    
    # Priority 3: Fallback to HR_avg
    if hr_bpm_value == 0 and data.get("HR_avg"):
        hr_bpm_value = _safe_int(data.get("HR_avg"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from HR_avg: {hr_bpm_value} bpm")

    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value

    report_sampling_rate = (
        getattr(getattr(ecg_test_page, "sampler", None), "sampling_rate", None)
        or getattr(ecg_test_page, "sampling_rate", None)
        or 500.0
    )
    rr_from_strip_ms = _estimate_rr_from_report_strip(
        ecg_test_page=ecg_test_page,
        ecg_data_file=ecg_data_file,
        sampling_rate=report_sampling_rate,
        settings_manager=settings_manager,
    )
    if rr_from_strip_ms and rr_from_strip_ms > 0:
        data["RR_ms"] = rr_from_strip_ms
        if not hr_bpm_value:
            data["HR_bpm"] = int(round(60000.0 / rr_from_strip_ms))
            data["Heart_Rate"] = data["HR_bpm"]
            data["HR"] = data["HR_bpm"]
        print(f" Using RR from plotted Lead II strip: {data['RR_ms']} ms")
    elif hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = data.get("RR_ms", 0)

    _align_report_intervals_to_reference(data)

    # Compute QTc / QTcF from QT and RR only when missing:
    #   QTc  (Bazett)     = QT / sqrt(RR)
    #   QTcF (Fridericia) = QT / cbrt(RR)
    # where QT and RR are in seconds.
    try:
        qt_ms = _safe_float(data.get("QT"))
        rr_ms = _safe_float(data.get("RR_ms"))
        existing_qtc = _safe_float(data.get("QTc"), 0)
        existing_qtcf = _safe_float(
            data.get("QTc_Fridericia") or data.get("QTcF") or data.get("QTcF_ms"),
            0,
        )
        qtc_bazett_ms = None
        qtc_frid_ms = None
        if qt_ms and qt_ms > 0 and rr_ms and rr_ms > 0:
            qt_sec = qt_ms / 1000.0
            rr_sec = rr_ms / 1000.0
            qtc_bazett_ms = qt_sec / (rr_sec ** 0.5) * 1000.0
            qtc_frid_ms = qt_sec / (rr_sec ** (1.0 / 3.0)) * 1000.0
        if (not existing_qtc or existing_qtc <= 0) and qtc_bazett_ms and qtc_bazett_ms > 0:
            data["QTc"] = qtc_bazett_ms
        if (not existing_qtcf or existing_qtcf <= 0) and qtc_frid_ms and qtc_frid_ms > 0:
            data["QTc_Fridericia"] = qtc_frid_ms
    except Exception:
        pass

    # ==================== STEP 2: Get wave_speed from ecg_settings.json (PRIORITY) ====================
    # Priority: ecg_settings.json wave_speed (FULLY DYNAMIC)
    # ── FIX: Reports ALWAYS at 25mm/s + 10mm/mV (standard clinical) ────────
    # User display setting (12.5/50mm/s) NEVER affects the report
    wave_speed_mm_s = 25.0   # Standard paper speed — hardcoded for reports
    wave_gain_mm_mv = 10.0   # Standard gain — hardcoded for reports
    # ─────────────────────────────────────────────────────────────────────────
    print(f" Report fixed at standard: {wave_speed_mm_s} mm/s, {wave_gain_mm_mv} mm/mV")
    computed_sampling_rate = 500

    data["wave_speed_mm_s"] = wave_speed_mm_s
    data["wave_gain_mm_mv"] = wave_gain_mm_mv

    print(f" Pre-plot checks: HR_bpm={hr_bpm_value}, RR_ms={data['RR_ms']}, wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV, sampling_rate={computed_sampling_rate}Hz")
    graph_boxes = 18.5 if abs(float(wave_speed_mm_s) - 12.5) < 0.01 else 37.0
    ecg_graph_width_mm = graph_boxes * ECG_LARGE_BOX_MM
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    print(f" Calculation-based beats formula:")
    print(f"   Graph width: {graph_boxes} boxes × {ECG_LARGE_BOX_MM:.2f}mm = {ecg_graph_width_mm:.2f}mm")
    print(f"   BPM window: (desired_beats × 60) / {hr_bpm_value} = {(6 * 60.0 / hr_bpm_value) if hr_bpm_value > 0 else 0:.2f}s")
    print(f"   Wave speed window: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {ecg_graph_width_mm / max(1e-6, effective_wave_speed_mm_s):.2f}s")
    
    # ==================== STEP 3: SAVE ECG DATA TO FILE (ALWAYS) ====================
    # IMPORTANT:  data file  save ,    load  (calculation-based beats  )
    saved_ecg_data = None
    saved_data_file_path = None
    
    if ecg_data_file and os.path.exists(ecg_data_file):
        # Use provided file
        print(f" Using provided ECG data file: {ecg_data_file}")

        
        saved_data_file_path = ecg_data_file
        saved_ecg_data = load_ecg_data_from_file(ecg_data_file)
        if saved_ecg_data:
            # Override sampling rate from saved data
            saved_sampling_rate = saved_ecg_data.get('sampling_rate', computed_sampling_rate)
            computed_sampling_rate = int(saved_sampling_rate)
            print(f" Using sampling rate from provided file: {computed_sampling_rate} Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'data'):
        # ALWAYS save current data to file before generating report (REQUIRED for calculation-based beats)
        print(" Saving ECG data to file (required for calculation-based beats)...")
        # Use companion JSON name derived from PDF filename
        json_filename = os.path.splitext(filename)[0] + ".json"
        saved_data_file_path = save_ecg_data_to_file(ecg_test_page, output_file=json_filename)
        if saved_data_file_path:
            saved_ecg_data = load_ecg_data_from_file(saved_data_file_path)
            if saved_ecg_data:
                saved_sampling_rate = saved_ecg_data.get('sampling_rate', computed_sampling_rate)
                computed_sampling_rate = int(saved_sampling_rate)
                print(f" Using sampling rate from saved file: {computed_sampling_rate} Hz")
            else:
                print(" Warning: Could not load saved ECG data file")
        else:
            print(" Warning: Could not save ECG data to file")
    
    if not saved_ecg_data:
        print(" Warning: No saved ECG data available - beats will not be calculation-based")

    # Track whether the source leads are actually present so missing leads can
    # be rendered as 0.000 instead of inheriting the other lead's value.
    rv5_source_present = None
    sv1_source_present = None
    try:
        if saved_ecg_data and isinstance(saved_ecg_data, dict):
            saved_leads = saved_ecg_data.get("leads") or {}
            if isinstance(saved_leads, dict):
                if rv5_source_present is None:
                    v5_samples = saved_leads.get("V5") or saved_leads.get("Lead_V5")
                    if v5_samples is not None:
                        rv5_source_present = _lead_has_signal(v5_samples)
                if sv1_source_present is None:
                    v1_samples = saved_leads.get("V1") or saved_leads.get("Lead_V1")
                    if v1_samples is not None:
                        sv1_source_present = _lead_has_signal(v1_samples)

        if ecg_test_page and hasattr(ecg_test_page, 'data') and isinstance(ecg_test_page.data, (list, tuple)):
            if rv5_source_present is None and len(ecg_test_page.data) > 10:
                rv5_source_present = _lead_has_signal(ecg_test_page.data[10])
            if sv1_source_present is None and len(ecg_test_page.data) > 6:
                sv1_source_present = _lead_has_signal(ecg_test_page.data[6])
    except Exception as _lead_state_err:
        print(f" Lead presence detection skipped: {_lead_state_err}")

    if isinstance(data, dict):
        data["_rv5_source_present"] = rv5_source_present
        data["_sv1_source_present"] = sv1_source_present


    # Get conclusions from frozen report request, dashboard/JSON, or expanded view.
    # Background PDF generation runs in a child process, so live widgets are not
    # available there; explicit conclusions keep the report aligned with the UI.
    # Prefer expanded-view / ECG test page conclusions when available (more detailed, updated form)
    dashboard_conclusions = _normalize_report_conclusions(conclusions or [])
    try:
        dashboard_conclusions = _normalize_report_conclusions(dashboard_conclusions)
        if not dashboard_conclusions and ecg_test_page:
            dashboard_conclusions = get_conclusions_from_ecg_test_page(ecg_test_page, data)
            if dashboard_conclusions:
                print(f" Using conclusions from expanded ECG view: {len(dashboard_conclusions)} items")
    except Exception:
        dashboard_conclusions = []

    if not dashboard_conclusions:
        dashboard_conclusions = get_dashboard_conclusions_from_image(dashboard_instance)

    # The permitted findings are all derived from measured values, so derive them
    # here unconditionally and merge. Previously the value rules ran only as a
    # last-resort fallback: a report measuring QRS 116 ms printed no QRS finding
    # because an earlier source had already supplied a rhythm label and won.
    # restrict_to_allowed_conclusions() de-duplicates, so merging is safe.
    try:
        dashboard_conclusions = _normalize_report_conclusions(
            list(dashboard_conclusions or []) + _build_metric_conclusions(data or {})
        )
        # The rate finding is always reportable; never leave the box blank
        # just because the rhythm label the rules picked is not printable.
        dashboard_conclusions = ensure_rate_conclusion(dashboard_conclusions, data or {})
    except Exception as _mc_err:
        print(f" Could not derive value-based conclusions: {_mc_err}")

    # SAFEGUARD: If there is no real data (HR <= 0 or all core metrics are zero), ignore any
    # persisted conclusions from last_conclusions.json and show explicit "No ECG data available" instead.
    try:
        def _parse_val(val):
            if val is None:
                return 0.0
            s = str(val).strip().lower()
            if s in ("", "--", "none", "null", "0", "0.0"):
                return 0.0
            clean = s.replace("bpm", "").replace("ms", "").replace("mv", "").replace("deg", "").replace("°", "").strip()
            try:
                return float(clean)
            except Exception:
                return 0.0

        hr_val = _parse_val(data.get("HR") or data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR_avg") or 0)
        pr_val = _parse_val(data.get("PR") or data.get("PR_ms") or 0)
        qrs_val = _parse_val(data.get("QRS") or data.get("QRS_ms") or 0)
        qt_val = _parse_val(data.get("QT") or data.get("QT_ms") or 0)
        qtc_val = _parse_val(data.get("QTc") or data.get("QTc_ms") or 0)

        is_no_data = (hr_val <= 0) or (hr_val == 0 and pr_val == 0 and qrs_val == 0 and qt_val == 0 and qtc_val == 0)

        if is_no_data:
            # Every value reads zero. There are two very different reasons for
            # that and telling a clinician the wrong one is dangerous:
            #
            #   electrodes off   -> a device problem, "please connect device"
            #   electrodes on,   -> the patient has no cardiac output; telling
            #   flat trace          them to check the cable is actively harmful
            #
            # The 12-lead page distinguishes them and sets _asystole_active.
            # Neither line is a rhythm diagnosis, so neither is subject to the
            # findings allow-list — they are status text, like the existing
            # "No ECG data available".
            _asystole = False
            _low_rate = False
            try:
                _asystole = bool(getattr(ecg_test_page, "_asystole_active", False))
                _low_rate = bool(getattr(ecg_test_page, "_rate_below_measurable", False))
            except Exception:
                _asystole = _low_rate = False

            if _low_rate and not _asystole:
                # QRS complexes are present but too far apart for the analysis
                # window to measure — a profoundly slow rhythm, not a fault and
                # not a flat line. Saying "please connect device" here would send
                # the operator after equipment that is working correctly.
                dashboard_conclusions = [
                    "Rate below measurable range",
                    "Review trace - measurements unavailable",
                ]
                print(" Rate below measurable range: R-peaks present but too few to measure")
            elif _asystole:
                dashboard_conclusions = [
                    "No cardiac activity detected",
                    "All measurements zero - review trace",
                ]
                print(" Asystole: all metrics zero with electrodes attached")
            else:
                dashboard_conclusions = [
                    "No ECG data available",
                    "Please connect device",
                ]
                print(" Overriding conclusions because HR is 0 or all core metrics are zero (no data)")
        else:
            dashboard_conclusions = [c for c in dashboard_conclusions if "No ECG data available" not in c and "Please connect device" not in c]
            if hr_val <= 0:
                dashboard_conclusions = [c for c in dashboard_conclusions if "Normal Sinus Rhythm" not in c and "Normal sinus rhythm" not in c]

            # Merge value-based metric conclusions derived directly from data.
            #
            # These MUST go through _normalize_report_conclusions(): appending
            # them raw bypassed the findings allow-list entirely, so labels that
            # are meant to be filtered — Third-degree AV Block, Borderline Wide
            # QRS, First-degree AV Block, Long QT Syndrome — reached the printed
            # report through this branch.
            merged = list(dashboard_conclusions) + _build_metric_conclusions(data)
            dashboard_conclusions = _normalize_report_conclusions(merged)
            dashboard_conclusions = ensure_rate_conclusion(dashboard_conclusions, data)
    except Exception as e:
        print(f" Safeguard check error: {e}")

    # Use ONLY conclusions from last_conclusions.json (loaded above)
    # Strip placeholders so only real conclusions appear in report
    # Per user request: exclude "Rhythm Analysis" and HRV-related lines
    # from 12-lead ECG reports.
    hrv_blocklist = [
        "HRV",
        "Heart Rate Variability",
        "variability",
        "HRV Test",
        "Stress Level",
        "Average Variability",
        "HRV:",
    ]

    def _is_hrv_line(s):
        if not s:
            return False
        low = s.lower()
        for b in hrv_blocklist:
            if b.lower() in low:
                return True
        return False

    filtered_conclusions = [
        c
        for c in _normalize_report_conclusions(dashboard_conclusions)
        if c and c != "---" and "Rhythm Analysis" not in c and not _is_hrv_line(c)
    ]
    # Ensure max 12
    filtered_conclusions = filtered_conclusions[:12]

    #  FORCE DELETE ALL OLD WHITE BACKGROUND IMAGES
    if lead_images is None:
        print("  DELETING ALL OLD WHITE BACKGROUND IMAGES...") 
        
        # Get both possible locations
        current_dir = os.path.dirname(os.path.abspath(__file__)) 
        project_root = os.path.join(current_dir, '..', '..')
        project_root = os.path.abspath(project_root)
        src_dir = os.path.join(current_dir, '..')
        
        leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"] 
        
        # DELETE from both locations
        for lead in leads:
            # Location 1: project root
            img_path1 = os.path.join(project_root, f"lead_{lead}.png")
            if os.path.exists(img_path1):
                os.remove(img_path1)
                print(f"  Deleted OLD image: {img_path1}")
            
            # Location 2: src directory  
            img_path2 = os.path.join(src_dir, f"lead_{lead}.png")
            if os.path.exists(img_path2):
                os.remove(img_path2)
                print(f"  : {img_path2}")
        
        print(" CREATING NEW PINK GRID IMAGES...")
        
        # Create NEW pink grid images
        lead_images = {}
        for lead in leads:
            try:
                # Create pink grid ECG
                fig = create_ecg_grid_with_waveform(None, lead, width=6, height=2)
                
                # Save to project root with pink background
                img_path = os.path.join(project_root, f"lead_{lead}.png")
                fig.savefig(img_path, 
                           dpi=200, 
                           bbox_inches='tight', 
                           pad_inches=0.05,
                           facecolor='#ffe6e6',  # PINK background
                           edgecolor='none',
                           format='png')
                plt.close(fig)
                
                lead_images[lead] = img_path
                print(f" Created NEW PINK GRID image: {img_path}")
                
            except Exception as e:
                print(f" Error creating {lead}: {e}")
        
        if not lead_images:
            return "Error: Could not create PINK GRID ECG images"
    
    # Get REAL ECG drawings from live test page
    print(" Capturing REAL ECG data from live test page...")
    
    # Check if demo mode is active and data is available
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo = ecg_test_page.demo_toggle.isChecked()
        if is_demo:
            print(" DEMO MODE DETECTED - Checking data availability...")
            if hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 0:
                # Check if data has actual variation (not just zeros)
                sample_data = ecg_test_page.data[0] if len(ecg_test_page.data) > 0 else []
                if len(sample_data) > 0:
                    std_val = np.std(sample_data)
                    print(f"    Data buffer size: {len(sample_data)}, Std deviation: {std_val:.4f}")
                    if std_val < 0.01:
                        print("    WARNING: Demo data appears to be flat/empty!")
                        print("    TIP: Make sure demo has been running for at least 5 seconds before generating report")
                    else:
                        print(f"    Demo data looks good (variation detected)")
                else:
                    print("    WARNING: Data buffer is empty!")
            else:
                print("    ERROR: No data structure found!")
    
    lead_drawings = capture_real_ecg_graphs_from_dashboard(
        dashboard_instance,
        ecg_test_page,
        samples_per_second=computed_sampling_rate,
        settings_manager=settings_manager
    )
    
    # Get lead sequence from settings (already initialized above)
    lead_sequence = settings_manager.get_setting("lead_sequence", "Standard")
    
    # Use the appropriate sequence for REPORT ONLY
    lead_order = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    print(f" Using lead sequence for REPORT: {lead_sequence}")
    print(f" Lead order for REPORT: {lead_order}")

    doc = SimpleDocTemplate(filename, pagesize=A4,
                            rightMargin=5 * mm, leftMargin=5 * mm,  # 5mm margins for 40 boxes
                            topMargin=6 * mm, bottomMargin=6 * mm)  # 6mm margins for 57 boxes

    story = []
    styles = getSampleStyleSheet()
    
    # Skip all Page 1 content - go directly to Page 2 content
    # Page 2 will now become Page 1 (portrait)
    
    # Patient details for what was Page 2 (now Page 1)
    if patient is None:
        patient = {}
    
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip()
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    date_time_str = patient.get("date_time", "")

    # Get real ECG data from dashboard
    HR = _safe_int(data.get('HR') or data.get('HR_bpm') or data.get('HR_avg'), 0)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = _safe_float(data.get('QT',), 0.0)
    QTc = _safe_float(data.get('QTc',), 0.0)
    QTcF = data.get('QTc_Fridericia') or data.get('QTcF') or 0
    ST = data.get('ST',)
    # Prefer RR calculated from the same plotted strip; only fall back to HR-derived RR.
    RR = _safe_int(data.get('RR_ms'), 0)
    if RR <= 0 and HR > 0:
        RR = int(60000 / HR)

    # Formatting functions
    def _fmt_bpm(value):
        return f"{value:.0f} bpm" if value and value > 0 else "--"

    def _fmt_ms(value):
        return f"{value:.0f} ms" if value and _safe_float(value) and _safe_float(value) > 0 else "--"

    def _fmt_mv(value):
        try:
            vf = _safe_float(value)
            if vf is not None:
                return f"{int(round(vf))}"
        except Exception:
            pass
        return "--"

    vital_params_table = _build_vital_table(data)

   

    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - A4 portrait standard (210x297mm), 5mm ECG boxes
    # 38 boxes × 5mm = 190mm width, 57 boxes × 5mm = 285mm height (within A4 297mm)
    total_width = 190 * mm
    total_height = 285 * mm
    # DEBUG: Print actual dimensions being used
    print(f" DEBUG: Drawing dimensions - Width: {total_width/mm:.1f}mm ({total_width/mm/5:.1f} boxes), Height: {total_height/mm:.1f}mm ({total_height/mm/5:.1f} boxes)")
    
    # Create ONE master drawing
    master_drawing = TransparentDrawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (SHIFTED UP to match Image 2 gap)
    y_positions = [239.6 * mm, 222.0 * mm, 204.3 * mm, 186.7 * mm, 169.1 * mm, 151.4 * mm, 133.8 * mm, 116.1 * mm, 98.5 * mm, 80.9 * mm, 63.2 * mm, 45.6 * mm]  
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        lead_positions.append({
            "lead": lead, 
            "x": 60 - (3.0 * ECG_LARGE_BOX_MM * mm), 
            "y": y_positions[i]
        })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # STEP 3: Draw ALL ECG content directly in master drawing
    successful_graphs = 0
    
# Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = computed_sampling_rate
    
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo_mode = ecg_test_page.demo_toggle.isChecked()
        if is_demo_mode:
            # Get time window from demo manager
            if hasattr(ecg_test_page, 'demo_manager') and ecg_test_page.demo_manager:
                time_window_seconds = getattr(ecg_test_page.demo_manager, 'time_window', None)
                samples_per_second = getattr(ecg_test_page.demo_manager, 'samples_per_second', samples_per_second)
                print(f" Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    graph_boxes = 18.5 if abs(float(wave_speed) - 12.5) < 0.01 else 37.0
                    ecg_graph_width_mm = graph_boxes * ECG_LARGE_BOX_MM
                    effective_wave_speed_mm_s = wave_speed * ECG_SPEED_SCALE
                    time_window_seconds = ecg_graph_width_mm / effective_wave_speed_mm_s
                    print(f" Report Generator: Demo mode ON - Calculated window: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {time_window_seconds:.2f}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f" Report Generator: Demo mode is OFF")
    
    # Always use the latest standard 10-second strip (hospital-style ECG printout)
    calculated_time_window = STANDARD_REPORT_WINDOW_SECONDS
    if is_demo_mode:
        if not computed_sampling_rate or computed_sampling_rate <= 0:
            computed_sampling_rate = 500.0
        num_samples_to_capture = _samples_for_standard_report_window(computed_sampling_rate)
        print(
            f" DEMO MODE: Using latest {STANDARD_REPORT_WINDOW_SECONDS:.1f}s "
            f"({num_samples_to_capture} samples at {computed_sampling_rate}Hz)"
        )
    else:
        num_samples_to_capture = _samples_for_standard_report_window(computed_sampling_rate)
        print(
            f" NORMAL MODE: Using latest {STANDARD_REPORT_WINDOW_SECONDS:.1f}s "
            f"({num_samples_to_capture} samples at {computed_sampling_rate}Hz)"
        )
    
    for pos_info in lead_positions:
        lead = pos_info['lead']
        x_pos = pos_info['x']
        y_pos = pos_info['y']
        try:
            from reportlab.graphics.shapes import String, Group
            lead_label = String(3.5 * mm, y_pos + 7.1 * mm + (1.5 * ECG_LARGE_BOX_MM * mm), f"{lead}", fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            if lead in lead_drawings:
                sub = lead_drawings[lead]
                grp = Group(*sub.contents)
                grp.translate(x_pos, y_pos)
                master_drawing.add(grp)
                successful_graphs += 1
            else:
                print(f"No ECG drawing available for lead {lead}")
        except Exception as e:
            print(f"Error drawing lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # ── CALL RESTRUCTURED HEADER ──────────────────────────────────────────
    _add_patient_header(master_drawing, full_name, age, gender, patient, date_time_str, data, settings_manager)

    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT', )
    QTc = data.get('QTc',)
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   

    # Create table data: 2 rows × 2 columns (as per your changes)
    vital_params_table = _build_vital_table(data)

   

    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - A4 portrait standard (210x297mm), 5mm ECG boxes
    total_width = 190 * mm
    total_height = 265 * mm  
    
    # Create ONE master drawing
    master_drawing = TransparentDrawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (SHIFTED UP by 80 points total: 40+25+15)
    y_positions = [v * mm for v in Y_POSITIONS_MM]
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        lead_positions.append({
            "lead": lead, 
            "x": 60 - (3.0 * ECG_LARGE_BOX_MM * mm), 
            "y": y_positions[i]
        })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # STEP 3: Draw ALL ECG content directly in master drawing
    successful_graphs = 0
    
    # Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = computed_sampling_rate
    
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo_mode = ecg_test_page.demo_toggle.isChecked()
        if is_demo_mode:
            # Get time window from demo manager
            if hasattr(ecg_test_page, 'demo_manager') and ecg_test_page.demo_manager:
                time_window_seconds = getattr(ecg_test_page.demo_manager, 'time_window', None)
                samples_per_second = getattr(ecg_test_page.demo_manager, 'samples_per_second', samples_per_second)
                print(f" Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    graph_boxes = 18.5 if abs(float(wave_speed) - 12.5) < 0.01 else 37.0
                    ecg_graph_width_mm = graph_boxes * ECG_LARGE_BOX_MM
                    effective_wave_speed_mm_s = wave_speed * ECG_SPEED_SCALE
                    time_window_seconds = ecg_graph_width_mm / effective_wave_speed_mm_s
                    print(f" Report Generator: Demo mode ON - Calculated window: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {time_window_seconds:.2f}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f" Report Generator: Demo mode is OFF")
    
    # Always use the latest standard 10-second strip (hospital-style ECG printout)
    calculated_time_window = STANDARD_REPORT_WINDOW_SECONDS
    if is_demo_mode:
        if not computed_sampling_rate or computed_sampling_rate <= 0:
            computed_sampling_rate = 500.0
        num_samples_to_capture = _samples_for_standard_report_window(computed_sampling_rate)
        print(
            f" DEMO MODE: Using latest {STANDARD_REPORT_WINDOW_SECONDS:.1f}s "
            f"({num_samples_to_capture} samples at {computed_sampling_rate}Hz)"
        )
    else:
        num_samples_to_capture = _samples_for_standard_report_window(computed_sampling_rate)
        print(
            f" NORMAL MODE: Using latest {STANDARD_REPORT_WINDOW_SECONDS:.1f}s "
            f"({num_samples_to_capture} samples at {computed_sampling_rate}Hz)"
        )
    
    for pos_info in lead_positions:
        lead = pos_info["lead"]
        x_pos = pos_info["x"]
        y_pos = pos_info["y"]
        
        try:
            # STEP 3A: Add lead label directly
            from reportlab.graphics.shapes import String
            lead_label = String(3.5 * mm, y_pos + 7.1 * mm + (1.5 * ECG_LARGE_BOX_MM * mm), f"{lead}", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # STEP 3B: Get REAL ECG data for this lead (ONLY from saved file - calculation-based)
            # IMPORTANT:  saved file  data use , live dashboard   (calculation-based beats  )
            real_data_available = False
            real_ecg_data = None
            
            # Helper function to calculate derived leads from I and II
            def calculate_derived_lead(lead_name, lead_i_data, lead_ii_data):
                """Calculate derived leads: III, aVR, aVL, aVF from I and II"""
                lead_i = np.array(lead_i_data, dtype=float)
                lead_ii = np.array(lead_ii_data, dtype=float)
                
                if lead_name == "III":
                    return lead_ii - lead_i  # III = II - I
                elif lead_name == "aVR":
                    return -(lead_i + lead_ii) / 2.0  # aVR = -(I + II) / 2
                elif lead_name == "aVL":
                    # aVL = (Lead I - Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_i - lead_iii) / 2.0  # aVL = (I - III) / 2
                elif lead_name == "aVF":
                    # aVF = (Lead II + Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_ii + lead_iii) / 2.0  # aVF = (II + III) / 2
                elif lead_name == "-aVR":
                    return -(-(lead_i + lead_ii) / 2.0)  # -aVR = -aVR = (I + II) / 2
                else:
                    return None
            
            # Priority 1: Use ONLY live dashboard data (ignore saved data completely)
            real_data_available = False
            real_ecg_data = None
            
            # Use live dashboard data only
            # Check if live data has MORE samples than saved data
            if ecg_test_page and hasattr(ecg_test_page, 'data'):
                lead_to_index = {
                    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                }
                
                live_data_available = False
                live_data_samples = 0
                
                # For calculated leads, calculate from live I and II
                if lead in ["III", "aVR", "aVL", "aVF", "-aVR"]:
                    if len(ecg_test_page.data) > 1:  # Need at least I and II
                        lead_i_data = ecg_test_page.data[0]  # I
                        lead_ii_data = ecg_test_page.data[1]  # II
                        
                        if len(lead_i_data) > 0 and len(lead_ii_data) > 0:
                            # Ensure same length
                            min_len = min(len(lead_i_data), len(lead_ii_data))
                            lead_i_slice = lead_i_data[-min_len:] if len(lead_i_data) >= min_len else lead_i_data
                            lead_ii_slice = lead_ii_data[-min_len:] if len(lead_ii_data) >= min_len else lead_ii_data
                            
                            # IMPORTANT: Subtract baseline from Lead I and Lead II BEFORE calculating derived leads
                            # This ensures calculated leads are centered around 0, not around baseline
                            baseline_adc = 2000.0
                            lead_i_centered = np.array(lead_i_slice, dtype=float) - baseline_adc
                            lead_ii_centered = np.array(lead_ii_slice, dtype=float) - baseline_adc
                            
                            # Calculate derived lead from centered values
                            calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                            if calculated_data is not None:
                                live_data_samples = len(calculated_data)
                                use_live_data = False
                                if not real_data_available:
                                    use_live_data = True
                                elif live_data_samples > saved_data_samples:
                                    use_live_data = True
                                
                                if use_live_data:
                                    raw_data = calculated_data
                                    if len(raw_data) >= num_samples_to_capture:
                                        raw_data = raw_data[-num_samples_to_capture:]
                                    if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                        real_ecg_data = np.array(raw_data)
                                        real_data_available = True
                                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                
                # For non-calculated leads, use existing logic
                if not real_data_available:
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        live_data_samples = len(ecg_test_page.data[3])
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        live_data_samples = len(ecg_test_page.data[lead_to_index[lead]])
                    
                    # Always use live dashboard data (ignore any saved data)
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        # For -aVR, use filtered inverted aVR data
                        raw_data = ecg_test_page.data[3]
                        # Check if we have enough samples, otherwise use all available
                        if len(raw_data) >= num_samples_to_capture:
                            raw_data = raw_data[-num_samples_to_capture:]
                        # Check if data is not all zeros or flat
                        if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                            # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                            real_ecg_data = np.array(raw_data)
                            real_data_available = True
                            actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                            if is_demo_mode and time_window_seconds is not None:
                                pass
                            else:
                                time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        # Get filtered real data for this lead
                        lead_index = lead_to_index[lead]
                        if len(ecg_test_page.data[lead_index]) > 0:
                            raw_data = ecg_test_page.data[lead_index]
                            # Check if we have enough samples, otherwise use all available
                            if len(raw_data) >= num_samples_to_capture:
                                raw_data = raw_data[-num_samples_to_capture:]
                            # Check if data has variation (not all zeros or flat line)
                            if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                real_ecg_data = np.array(raw_data)
                                real_data_available = True
                                actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                if is_demo_mode and time_window_seconds is not None:
                                    pass
                                else:
                                    time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                            else:
                                pass
            
            # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
            # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
            ADC_PER_BOX_CONFIG = {
                'I': 6400.0,
                'II': 6400.0,
                'III': 6400.0,
                'aVR': 6400.0,
                'aVL': 6400.0,
                'aVF': 6400.0,
                'V1': 6400.0,
                'V2': 6400.0,
                'V3': 6400.0,
                'V4': 6400.0,
                'V5': 6400.0,
                'V6': 6400.0,
                '-aVR': 6400.0,  # For Cabrera sequence
            }

            if real_data_available and len(real_ecg_data) > 0:
                # Draw ALL REAL ECG data - NO LIMITS
                ecg_width = graph_boxes * ECG_LARGE_BOX_MM * mm
                ecg_height = 45
                
                adc_data = _prepare_report_strip_signal(
                    real_ecg_data,
                    computed_sampling_rate,
                    settings_manager,
                    target_samples=len(real_ecg_data),
                )
                if len(adc_data) < 2:
                    continue

                # Calculate physical dimensions and samples that fit
                ecg_width_mm = graph_boxes * ECG_LARGE_BOX_MM  # 37.0 * 5.0 = 185.0 mm
                visible_seconds = ecg_width_mm / 25.0  # 185.0 / 25.0 = 7.4 seconds
                visible_samples = int(round(visible_seconds * computed_sampling_rate))
                
                # Slices to exactly fit the visible width, cropping extra samples instead of stretching
                if len(adc_data) > visible_samples:
                    adc_data = adc_data[-visible_samples:]
                
                t_sec = np.arange(len(adc_data)) / computed_sampling_rate
                t = x_pos + t_sec * 25.0 * mm

                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                from reportlab.lib.units import mm
                box_height_points = ECG_LARGE_BOX_MM * mm
                boxes_offset = adc_data / adc_per_box
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                # DEBUG: Verify Y position calculation
                
                # Draw ALL REAL ECG data points
                from reportlab.graphics.shapes import Path
                ecg_path = Path(fillColor=None, 
                               strokeColor=colors.HexColor("#000000"), 
                               strokeWidth=0.4,
                               strokeLineCap=1,
                               strokeLineJoin=1)
                
                # DEBUG: Verify actual plotted values
                actual_min_y = np.min(ecg_normalized)
                actual_max_y = np.max(ecg_normalized)
                actual_span_points = actual_max_y - actual_min_y
                actual_span_boxes = actual_span_points / box_height_points
                
                # Trim a small render window at both ends to suppress any residual edge artifacts
                # Start path
                ecg_path.moveTo(t[0], ecg_normalized[0])
                # Add ALL points
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                # Add calibration notch 15 points after ECG strip starts for all 12 leads
                print(f" DEBUG: Adding calibration notch for Lead {lead}")
                from reportlab.graphics.shapes import Path
                
                # Calibration notch dimensions (1 box wide, 2 boxes tall)
                notch_width_mm = ECG_LARGE_BOX_MM
                notch_height_mm = 2.0 * ECG_LARGE_BOX_MM
                notch_width = notch_width_mm * mm
                notch_height = notch_height_mm * mm
                
                # Position notch 15 points after where ECG strip starts, then shift 40 points left
                notch_x = x_pos + 15.0 - 40.0
                notch_y_base = center_y  # Use same center_y as ECG data
                
                print(f" DEBUG: Notch position for Lead {lead} - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                
                # Create calibration notch path
                notch_path = Path(
                    fillColor=None,
                    strokeColor=colors.HexColor("#000000"),
                    strokeWidth=0.8,
                    strokeLineCap=1,
                    strokeLineJoin=0
                )
                notch_path.moveTo(notch_x, notch_y_base)
                notch_path.lineTo(notch_x, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base)
                # Small forward tick to the right (extra 2mm) for clearer notch end
                notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                
                # Add notch to master drawing
                master_drawing.add(notch_path)
                print(f" DEBUG: Calibration notch added for Lead {lead}")
                
                print(f" Drew {len(adc_data)} ECG data points for Lead {lead}")
            else:
                print(f" No real data for Lead {lead} - showing flat line")
                # Draw flat line when no real data available (like dashboard)
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Line, Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of graph in points
                
                # Draw flat line at center (baseline)
                flat_line_start_x = x_pos + 15.0  # Same start as real data
                flat_line_end_x = x_pos + graph_boxes * ECG_LARGE_BOX_MM * mm
                flat_line_y = center_y  # Center/baseline position
                
                flat_line = Line(flat_line_start_x, flat_line_y, flat_line_end_x, flat_line_y,
                              strokeColor=colors.HexColor("#000000"), strokeWidth=1.2)
                master_drawing.add(flat_line)
                
                # Add calibration notch 15 points after ECG strip starts even when no data is available
                print(f" DEBUG: Adding calibration notch for Lead {lead} (no data case)")
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                
                # Calibration notch dimensions (1 box wide, 2 boxes tall)
                notch_width_mm = ECG_LARGE_BOX_MM
                notch_height_mm = 2.0 * ECG_LARGE_BOX_MM
                notch_width = notch_width_mm * mm
                notch_height = notch_height_mm * mm
                
                # Position notch 15 points after where ECG strip starts, then shift 40 points left
                notch_x = x_pos + 15.0 - 40.0
                notch_y_base = center_y  # Use same center_y calculation as real data section
                
                print(f" DEBUG: Notch position for Lead {lead} (no data) - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                
                # Create calibration notch path
                notch_path = Path(
                    fillColor=None,
                    strokeColor=colors.HexColor("#000000"),
                    strokeWidth=0.8,
                    strokeLineCap=1,
                    strokeLineJoin=0
                )
                notch_path.moveTo(notch_x, notch_y_base)
                notch_path.lineTo(notch_x, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base)
                # Small forward tick to the right (extra 2mm) for clearer notch end
                notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                
                # Add notch to master drawing
                master_drawing.add(notch_path)
                print(f" DEBUG: Calibration notch added for Lead {lead} (no data case)")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 4: Add Patient Info, Date/Time and Vital Parameters to master drawing
    # POSITIONED ABOVE ECG GRAPH (not mixed inside graph)
    # ── CALL RESTRUCTURED HEADER ──────────────────────────────────────────
    _add_patient_header(master_drawing, full_name, age, gender, patient, date_time_str, data, settings_manager)

    # RIGHT SIDE: Vital Parameters calculation (Labels now handled by _add_patient_header)
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = _safe_float(data.get('QT',), 0.0)
    QTc = _safe_float(data.get('QTc',), 0.0)
    ST = data.get('ST',)
    RR = int(60000 / HR) if HR and HR > 0 else 0 
   

    vital_params_table = _build_vital_table(data)

   

    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - A4 portrait standard (210x297mm), 5mm ECG boxes
    total_width = 190 * mm
    total_height = 265 * mm
    
    # Create ONE master drawing
    master_drawing = TransparentDrawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads based on selected sequence (SHIFTED UP by 80 points total: 40+25+15)
    y_positions = [v * mm for v in Y_POSITIONS_MM]
    lead_positions = []
    
    for i, lead in enumerate(lead_order):
        lead_positions.append({
            "lead": lead, 
            "x": 60 - (3.0 * ECG_LARGE_BOX_MM * mm), 
            "y": y_positions[i]
        })
    
    print(f" Using lead positions in {lead_sequence} sequence: {[pos['lead'] for pos in lead_positions]}")
    
    # STEP 3: Draw ALL ECG content directly in master drawing
    successful_graphs = 0
    
    # Check if demo mode is active and get time window for filtering
    is_demo_mode = False
    time_window_seconds = None
    samples_per_second = computed_sampling_rate
    
    if ecg_test_page and hasattr(ecg_test_page, 'demo_toggle'):
        is_demo_mode = ecg_test_page.demo_toggle.isChecked()
        if is_demo_mode:
            # Get time window from demo manager
            if hasattr(ecg_test_page, 'demo_manager') and ecg_test_page.demo_manager:
                time_window_seconds = getattr(ecg_test_page.demo_manager, 'time_window', None)
                samples_per_second = getattr(ecg_test_page.demo_manager, 'samples_per_second', samples_per_second)
                print(f" Report Generator: Demo mode ON - Wave speed window: {time_window_seconds}s, Sampling rate: {samples_per_second}Hz")
            else:
                # Fallback: calculate from wave speed setting
                try:
                    from utils.settings_manager import SettingsManager
                    sm = SettingsManager()
                    wave_speed = float(sm.get_wave_speed())
                    graph_boxes = 18.5 if abs(float(wave_speed) - 12.5) < 0.01 else 37.0
                    ecg_graph_width_mm = graph_boxes * ECG_LARGE_BOX_MM
                    effective_wave_speed_mm_s = wave_speed * ECG_SPEED_SCALE
                    time_window_seconds = ecg_graph_width_mm / effective_wave_speed_mm_s
                    print(f" Report Generator: Demo mode ON - Calculated window: {ecg_graph_width_mm:.2f}mm / {effective_wave_speed_mm_s:.2f}mm/s = {time_window_seconds:.2f}s")
                except Exception as e:
                    print(f" Could not get demo time window: {e}")
                    time_window_seconds = None
        else:
            print(f" Report Generator: Demo mode is OFF")
    
    # Calculate number of samples to capture based on demo mode OR BPM + wave_speed
    calculated_time_window = None  # Initialize for use in data loading section
    if is_demo_mode and time_window_seconds is not None:
        # In demo mode: only capture data visible in one window frame
        calculated_time_window = time_window_seconds
        num_samples_to_capture = int(time_window_seconds * samples_per_second)
        print(f" DEMO MODE: Master drawing will capture only {num_samples_to_capture} samples ({time_window_seconds}s window)")
    else:
        # Normal mode: Calculate time window based on wave_speed ONLY (NEW LOGIC)
        # This ensures proper number of beats are displayed based on graph width
        # Formula: 
        #   - Time window = 165mm / wave_speed ONLY (33 boxes × 5mm = 165mm)
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,  # From metrics.json (priority) - for calculation-based beats
            wave_speed_mm_s,  # From ecg_settings.json - for calculation-based beats
            desired_beats=6  # Default: 6 beats desired
        )
        
        # Recalculate with actual sampling rate
        num_samples_to_capture = int(calculated_time_window * computed_sampling_rate)
        print(f" NORMAL MODE: Calculated time window: {calculated_time_window:.2f}s")
        print(f"   Based on BPM={hr_bpm_value} and wave_speed={wave_speed_mm_s}mm/s")
        print(f"   Will capture {num_samples_to_capture} samples (at {computed_sampling_rate}Hz)")
        if hr_bpm_value > 0:
            expected_beats = int((calculated_time_window * hr_bpm_value) / 60)
            print(f"   Expected beats shown: ~{expected_beats} beats")
    
    for pos_info in lead_positions:
        lead = pos_info["lead"]
        x_pos = pos_info["x"]
        y_pos = pos_info["y"]
        
        try:
            # STEP 3A: Add lead label directly
            from reportlab.graphics.shapes import String
            lead_label = String(3.5 * mm, y_pos + 7.1 * mm + (1.5 * ECG_LARGE_BOX_MM * mm), f"{lead}", 
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # STEP 3B: Get REAL ECG data for this lead (ONLY from saved file - calculation-based)
            # IMPORTANT:  saved file  data use , live dashboard   (calculation-based beats  )
            real_data_available = False
            real_ecg_data = None
            
            # Helper function to calculate derived leads from I and II
            def calculate_derived_lead(lead_name, lead_i_data, lead_ii_data):
                """Calculate derived leads: III, aVR, aVL, aVF from I and II"""
                lead_i = np.array(lead_i_data, dtype=float)
                lead_ii = np.array(lead_ii_data, dtype=float)
                
                if lead_name == "III":
                    return lead_ii - lead_i  # III = II - I
                elif lead_name == "aVR":
                    return -(lead_i + lead_ii) / 2.0  # aVR = -(I + II) / 2
                elif lead_name == "aVL":
                    # aVL = (Lead I - Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_i - lead_iii) / 2.0  # aVL = (I - III) / 2
                elif lead_name == "aVF":
                    # aVF = (Lead II + Lead III) / 2
                    lead_iii = lead_ii - lead_i  # Calculate Lead III first
                    return (lead_ii + lead_iii) / 2.0  # aVF = (II + III) / 2
                elif lead_name == "-aVR":
                    return -(-(lead_i + lead_ii) / 2.0)  # -aVR = -aVR = (I + II) / 2
                else:
                    return None
            
            # Priority 1: Use ONLY live dashboard data (ignore saved data completely)
            real_data_available = False
            real_ecg_data = None
            
            # Use live dashboard data only
            # Check if live data has MORE samples than saved data
            if ecg_test_page and hasattr(ecg_test_page, 'data'):
                lead_to_index = {
                    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
                    "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11
                }
                
                live_data_available = False
                live_data_samples = 0
                
                # For calculated leads, calculate from live I and II
                if lead in ["III", "aVR", "aVL", "aVF", "-aVR"]:
                    if len(ecg_test_page.data) > 1:  # Need at least I and II
                        lead_i_data = ecg_test_page.data[0]  # I
                        lead_ii_data = ecg_test_page.data[1]  # II
                        
                        if len(lead_i_data) > 0 and len(lead_ii_data) > 0:
                            # Ensure same length
                            min_len = min(len(lead_i_data), len(lead_ii_data))
                            lead_i_slice = lead_i_data[-min_len:] if len(lead_i_data) >= min_len else lead_i_data
                            lead_ii_slice = lead_ii_data[-min_len:] if len(lead_ii_data) >= min_len else lead_ii_data
                            
                            # IMPORTANT: Subtract baseline from Lead I and Lead II BEFORE calculating derived leads
                            # This ensures calculated leads are centered around 0, not around baseline
                            baseline_adc = 2000.0
                            lead_i_centered = np.array(lead_i_slice, dtype=float) - baseline_adc
                            lead_ii_centered = np.array(lead_ii_slice, dtype=float) - baseline_adc
                            
                            # Calculate derived lead from centered values
                            calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                            if calculated_data is not None:
                                live_data_samples = len(calculated_data)
                                use_live_data = False
                                if not real_data_available:
                                    use_live_data = True
                                elif live_data_samples > saved_data_samples:
                                    use_live_data = True
                                
                                if use_live_data:
                                    raw_data = calculated_data
                                    if len(raw_data) >= num_samples_to_capture:
                                        raw_data = raw_data[-num_samples_to_capture:]
                                    if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                        real_ecg_data = np.array(raw_data)
                                        real_data_available = True
                                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                
                # For non-calculated leads, use existing logic
                if not real_data_available:
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        live_data_samples = len(ecg_test_page.data[3])
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        live_data_samples = len(ecg_test_page.data[lead_to_index[lead]])
                    
                    # Always use live dashboard data (ignore any saved data)
                    if lead == "-aVR" and len(ecg_test_page.data) > 3:
                        # For -aVR, use filtered inverted aVR data
                        raw_data = ecg_test_page.data[3]
                        # Check if we have enough samples, otherwise use all available
                        if len(raw_data) >= num_samples_to_capture:
                            raw_data = raw_data[-num_samples_to_capture:]
                        # Check if data is not all zeros or flat
                        if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                            # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                            real_ecg_data = np.array(raw_data)
                            real_data_available = True
                            actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                            if is_demo_mode and time_window_seconds is not None:
                                pass
                            else:
                                time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                        # Get filtered real data for this lead
                        lead_index = lead_to_index[lead]
                        if len(ecg_test_page.data[lead_index]) > 0:
                            raw_data = ecg_test_page.data[lead_index]
                            # Check if we have enough samples, otherwise use all available
                            if len(raw_data) >= num_samples_to_capture:
                                raw_data = raw_data[-num_samples_to_capture:]
                            # Check if data has variation (not all zeros or flat line)
                            if len(raw_data) > 0 and np.std(raw_data) > 0.01:
                                # STEP 1: Capture ORIGINAL dashboard data (NO gain applied)
                                real_ecg_data = np.array(raw_data)
                                real_data_available = True
                                actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                                if is_demo_mode and time_window_seconds is not None:
                                    pass
                                else:
                                    time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                            else:
                                pass
            
            if real_data_available and len(real_ecg_data) > 0:
                # Draw ALL REAL ECG data - NO LIMITS
                ecg_width = graph_boxes * ECG_LARGE_BOX_MM * mm
                ecg_height = 45
                
                adc_data = _prepare_report_strip_signal(
                    real_ecg_data,
                    computed_sampling_rate,
                    settings_manager,
                    target_samples=len(real_ecg_data),
                )
                if len(adc_data) < 2:
                    continue

                # Calculate physical dimensions and samples that fit
                ecg_width_mm = graph_boxes * ECG_LARGE_BOX_MM  # 37.0 * 5.0 = 185.0 mm
                visible_seconds = ecg_width_mm / 25.0  # 185.0 / 25.0 = 7.4 seconds
                visible_samples = int(round(visible_seconds * computed_sampling_rate))
                
                # Slices to exactly fit the visible width, cropping extra samples instead of stretching
                if len(adc_data) > visible_samples:
                    adc_data = adc_data[-visible_samples:]
                
                t_sec = np.arange(len(adc_data)) / computed_sampling_rate
                t = x_pos + t_sec * 25.0 * mm

                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                box_height_points = ECG_LARGE_BOX_MM * mm
                boxes_offset = adc_data / adc_per_box
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                
                # Draw ALL REAL ECG data points
                from reportlab.graphics.shapes import Path
                ecg_path = Path(fillColor=None, 
                               strokeColor=colors.HexColor("#000000"), 
                               strokeWidth=0.4,
                               strokeLineCap=1,
                               strokeLineJoin=1)
                
                # DEBUG: Verify actual plotted values
                actual_min_y = np.min(ecg_normalized)
                actual_max_y = np.max(ecg_normalized)
                actual_span_points = actual_max_y - actual_min_y
                actual_span_boxes = actual_span_points / box_height_points
                
                ecg_path.moveTo(t[0], ecg_normalized[0])
                for i in range(1, len(t)):
                    ecg_path.lineTo(t[i], ecg_normalized[i])
                
                # Add path to master drawing
                master_drawing.add(ecg_path)
                
                # Add calibration notch 15 points after ECG strip starts for all 12 leads
                print(f" DEBUG: Adding calibration notch for Lead {lead}")
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Path
                
                # Dynamic calibration notch based on wave gain
                try:
                    from utils.settings_manager import SettingsManager
                    settings_mgr = SettingsManager()
                    notch_boxes = settings_mgr.get_calibration_notch_boxes()
                    print(f" Dynamic notch: {notch_boxes} boxes for gain {settings_mgr.get_wave_gain()}mm/mV")
                except Exception as e:
                    print(f" Could not get dynamic notch, using default: {e}")
                    notch_boxes = 2.0  # Default fallback
                
                # Calibration notch dimensions (1 box wide, dynamic boxes tall)
                notch_width_mm = ECG_LARGE_BOX_MM
                notch_height_mm = notch_boxes * ECG_LARGE_BOX_MM
                notch_width = notch_width_mm * mm
                notch_height = notch_height_mm * mm
                
                # Position notch 15 points after where ECG strip starts, then shift 40 points left
                notch_x = x_pos + 15.0 - 40.0
                notch_y_base = center_y  # Use same center_y as ECG data
                
                print(f" DEBUG: Notch position for Lead {lead} - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                
                # Create calibration notch path
                notch_path = Path(
                    fillColor=None,
                    strokeColor=colors.HexColor("#000000"),
                    strokeWidth=0.8,
                    strokeLineCap=1,
                    strokeLineJoin=0
                )
                notch_path.moveTo(notch_x, notch_y_base)
                notch_path.lineTo(notch_x, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base)
                # Small forward tick to the right (extra 2mm) for clearer notch end
                notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                
                # Add notch to master drawing
                master_drawing.add(notch_path)
                print(f" DEBUG: Calibration notch added for Lead {lead}")
                
                print(f" Drew {len(adc_data)} ECG data points for Lead {lead}")
            else:
                print(f" No real data for Lead {lead} - showing flat line")
                
                # Draw flat line when no real data available (like dashboard)
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Line, Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of graph in points
                
                # Draw flat line at center (baseline)
                flat_line_start_x = x_pos + 15.0  # Same start as real data
                flat_line_end_x = x_pos + graph_boxes * ECG_LARGE_BOX_MM * mm
                flat_line_y = center_y  # Center/baseline position
                
                flat_line = Line(flat_line_start_x, flat_line_y, flat_line_end_x, flat_line_y,
                              strokeColor=colors.HexColor("#000000"), strokeWidth=1.2)
                master_drawing.add(flat_line)
                # Add calibration notch 15 points after ECG strip starts even when no data is available
                print(f" DEBUG: Adding calibration notch for Lead {lead} (no data case)")
                from reportlab.lib.units import mm
                from reportlab.graphics.shapes import Path
                
                # Calculate center_y same as real data section
                ecg_height = 45  # Same as real data section
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                
                # Calibration notch dimensions (1 box wide, 2 boxes tall)
                notch_width_mm = ECG_LARGE_BOX_MM
                notch_height_mm = 2.0 * ECG_LARGE_BOX_MM
                notch_width = notch_width_mm * mm
                notch_height = notch_height_mm * mm
                
                # Position notch 15 points after where ECG strip starts, then shift 40 points left
                notch_x = x_pos + 15.0 - 40.0
                notch_y_base = center_y  # Use same center_y calculation as real data section
                
                print(f" DEBUG: Notch position for Lead {lead} (no data) - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                
                # Create calibration notch path
                notch_path = Path(
                    fillColor=None,
                    strokeColor=colors.HexColor("#000000"),
                    strokeWidth=0.8,
                    strokeLineCap=1,
                    strokeLineJoin=0
                )
                notch_path.moveTo(notch_x, notch_y_base)
                notch_path.lineTo(notch_x, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
                notch_path.lineTo(notch_x + notch_width, notch_y_base)
                # Small forward tick to the right (extra 2mm) for clearer notch end
                notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                
                # Add notch to master drawing
                master_drawing.add(notch_path)
                print(f" DEBUG: Calibration notch added for Lead {lead} (no data case)")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # RIGHT SIDE: Vital Parameters calculation
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = _safe_float(data.get('QT',), 0.0)
    QTc = _safe_float(data.get('QTc',), 0.0)
    ST = data.get('ST',)
    RR = int(60000 / HR) if HR and HR > 0 else 0 


    # CALCULATED wave amplitudes and lead-specific measurements
    # Prefer values passed in data; if missing/zero, compute from live ecg_test_page data (last 10s)
    p_amp_mv = data.get('p_amp', 0.0)
    qrs_amp_mv = data.get('qrs_amp', 0.0)
    t_amp_mv = data.get('t_amp', 0.0)
    
    print(f" Report Generator - Received wave amplitudes from data:")
    print(f"   p_amp: {p_amp_mv}, qrs_amp: {qrs_amp_mv}, t_amp: {t_amp_mv}")
    print(f"   Available keys in data: {list(data.keys())}")
    
    # If not provided or zero, compute quickly from Lead II in ecg_test_page (robust fallback)
    def _compute_from_data_array(arr, fs):
        from scipy.signal import butter, filtfilt, find_peaks
        if arr is None or len(arr) < int(2*fs) or np.std(arr) < 0.1:
            return 0.0, 0.0, 0.0
        nyq = fs/2.0
        b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
        x = filtfilt(b,a,arr)
        # Simple R detection via Pan-Tompkins style envelope
        squared = np.square(np.diff(x))
        win = max(1, int(0.15*fs))
        env = np.convolve(squared, np.ones(win)/win, mode='same')
        thr = np.mean(env) + 0.5*np.std(env)
        r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6*fs))
        if len(r_peaks) < 3:
            return 0.0, 0.0, 0.0
        p_vals, qrs_vals, t_vals = [], [], []
        for r in r_peaks[1:-1]:
            # P: 120-200ms before R
            p_start = max(0, r-int(0.20*fs)); p_end = max(0, r-int(0.12*fs))
            if p_end>p_start:
                seg = x[p_start:p_end]
                base = np.mean(x[max(0,p_start-int(0.05*fs)):p_start])
                p_vals.append(max(seg)-base)
            # QRS: +-80ms around R
            qrs_start = max(0, r-int(0.08*fs)); qrs_end = min(len(x), r+int(0.08*fs))
            if qrs_end>qrs_start:
                seg = x[qrs_start:qrs_end]
                qrs_vals.append(max(seg)-min(seg))
            # T: 100-300ms after R
            t_start = min(len(x), r+int(0.10*fs)); t_end = min(len(x), r+int(0.30*fs))
            if t_end>t_start:
                seg = x[t_start:t_end]
                base = np.mean(x[r:t_start]) if t_start>r else 0.0
                t_vals.append(max(seg)-base)
        def med(v):
            return float(np.median(v)) if len(v)>0 else 0.0
        return med(p_vals), med(qrs_vals), med(t_vals)

    if (p_amp_mv<=0 or qrs_amp_mv<=0 or t_amp_mv<=0) and ecg_test_page is not None and hasattr(ecg_test_page,'data'):
        try:
            fs = 500.0
            if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler,'sampling_rate') and ecg_test_page.sampler.sampling_rate:
                fs = float(ecg_test_page.sampler.sampling_rate)
            arr = None
            if len(ecg_test_page.data)>1:
                lead_ii = ecg_test_page.data[1]
                if isinstance(lead_ii, (list, tuple)):
                    lead_ii = np.asarray(lead_ii)
                arr = lead_ii[-int(10*fs):] if lead_ii is not None and len(lead_ii)>int(10*fs) else lead_ii
            cp, cqrs, ct = _compute_from_data_array(arr, fs)
            if p_amp_mv<=0: p_amp_mv = cp
            if qrs_amp_mv<=0: qrs_amp_mv = cqrs
            if t_amp_mv<=0: t_amp_mv = ct
            print(f" Fallback computed amplitudes from Lead II: P={p_amp_mv:.4f}, QRS={qrs_amp_mv:.4f}, T={t_amp_mv:.4f}")
        except Exception as e:
            print(f" Fallback amplitude computation failed: {e}")

    # Calculate P/QRS/T Axis in degrees (using Lead I and Lead aVF)
    def _to_axis_degree(value):
        """Normalize axis input to integer degrees in [-180, 180], or None."""
        if value is None:
            return None
        try:
            if isinstance(value, str):
                value = value.replace("°", "").strip()
                if not value:
                    return None
            axis = float(value)
            if not np.isfinite(axis):
                return None
            while axis > 180:
                axis -= 360
            while axis < -180:
                axis += 360
            return int(round(axis))
        except Exception:
            return None

    def _axis_delta_deg(a, b):
        if a is None or b is None:
            return None
        return abs(((a - b + 180) % 360) - 180)

    def _sanitize_axis_for_report(axis_name, axis_value, qrs_value=None, hr_value=None):
        axis_num = _to_axis_degree(axis_value)
        qrs_num = _to_axis_degree(qrs_value)
        hr_num = _safe_float(hr_value, None)
        if axis_num is None:
            return "--"

        if axis_name == "P":
            if abs(axis_num) > 120:
                return "--"
            delta = _axis_delta_deg(axis_num, qrs_num)
            if delta is not None and delta > 120:
                return "--"

        if axis_name == "T":
            delta = _axis_delta_deg(axis_num, qrs_num)
            if delta is not None and delta > 150:
                return "--"
            if hr_num is not None and hr_num >= 100 and abs(axis_num) > 150:
                return "--"

        return f"{axis_num}°"

    # PRIORITY 1: Use standardized values from data dictionary (passed from dashboard)
    p_axis_deg = "--"
    qrs_axis_deg = "--"
    t_axis_deg = "--"
    
    if data is not None:
        p_axis_data = _to_axis_degree(data.get('p_axis'))
        qrs_axis_data = _to_axis_degree(data.get('QRS_axis'))
        t_axis_data = _to_axis_degree(data.get('t_axis'))

        if p_axis_data is not None:
            p_axis_deg = f"{p_axis_data}°"
            print(f" Using P axis from data: {p_axis_deg}")
        if qrs_axis_data is not None:
            qrs_axis_deg = f"{qrs_axis_data}°"
            print(f" Using QRS axis from data: {qrs_axis_deg}")
        if t_axis_data is not None:
            t_axis_deg = f"{t_axis_data}°"
            print(f" Using T axis from data: {t_axis_deg}")
    
    # PRIORITY 2: Try to get axis values from ECG test page (standardized median beat method)
    if (p_axis_deg == "--" or qrs_axis_deg == "--" or t_axis_deg == "--") and ecg_test_page is not None:
        try:
            # Get P axis from standardized calculation
            if p_axis_deg == "--" and hasattr(ecg_test_page, 'calculate_p_axis_from_median'):
                p_axis_calc = ecg_test_page.calculate_p_axis_from_median()
                p_axis_calc = _to_axis_degree(p_axis_calc)
                if p_axis_calc is not None:
                    p_axis_deg = f"{p_axis_calc}°"
                    print(f" Using standardized P axis from ECG test page: {p_axis_deg}")
            
            # Get QRS axis from standardized calculation
            if qrs_axis_deg == "--" and hasattr(ecg_test_page, 'calculate_qrs_axis_from_median'):
                qrs_axis_calc = ecg_test_page.calculate_qrs_axis_from_median()
                qrs_axis_calc = _to_axis_degree(qrs_axis_calc)
                if qrs_axis_calc is not None:
                    qrs_axis_deg = f"{qrs_axis_calc}°"
                    print(f" Using standardized QRS axis from ECG test page: {qrs_axis_deg}")
            
            # Get T axis from standardized calculation
            if t_axis_deg == "--" and hasattr(ecg_test_page, 'calculate_t_axis_from_median'):
                t_axis_calc = ecg_test_page.calculate_t_axis_from_median()
                t_axis_calc = _to_axis_degree(t_axis_calc)
                if t_axis_calc is not None:
                    t_axis_deg = f"{t_axis_calc}°"
                    print(f"🔬 Using standardized T axis from ECG test page: {t_axis_deg}")
        except Exception as e:
            print(f" Error getting axis values from ECG test page: {e}")
            import traceback
            traceback.print_exc()
    
    # Fallback: Recalculate if not available from ECG test page
    # Disable fallback - it uses a different wrong algorithm
    # T axis should come from the cached value only
    if (p_axis_deg == "--" or qrs_axis_deg == "--" or t_axis_deg == "--"):
        pass  # Don't recalculate - use cached value or "--"
        
    if False: # Fallback disabled
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            
            # Get Lead I (index 0) and Lead aVF (index 5)
            lead_I = ecg_test_page.data[0] if len(ecg_test_page.data) > 0 else None
            lead_aVF = ecg_test_page.data[5] if len(ecg_test_page.data) > 5 else None
            
            # Get sampling rate
            fs = 500.0
            if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate') and ecg_test_page.sampler.sampling_rate:
                fs = float(ecg_test_page.sampler.sampling_rate)
            
            if lead_I is not None and lead_aVF is not None:
                # Convert to numpy arrays
                if isinstance(lead_I, (list, tuple)):
                    lead_I = np.asarray(lead_I)
                if isinstance(lead_aVF, (list, tuple)):
                    lead_aVF = np.asarray(lead_aVF)
                
                # Get last 10 seconds of data
                def _get_last(arr):
                    return arr[-int(10*fs):] if arr is not None and len(arr) > int(10*fs) else arr
                
                lead_I_data = _get_last(lead_I)
                lead_aVF_data = _get_last(lead_aVF)
                
                if len(lead_I_data) > int(2*fs) and len(lead_aVF_data) > int(2*fs):
                    # Filter signals
                    nyq = fs/2.0
                    b, a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq, 0.99)], btype='band')
                    lead_I_filt = filtfilt(b, a, lead_I_data)
                    lead_aVF_filt = filtfilt(b, a, lead_aVF_data)
                    
                    # Detect R peaks using Pan-Tompkins style
                    squared = np.square(np.diff(lead_aVF_filt))
                    win = max(1, int(0.15*fs))
                    env = np.convolve(squared, np.ones(win)/win, mode='same')
                    thr = np.mean(env) + 0.5*np.std(env)
                    r_peaks, _ = find_peaks(env, height=thr, distance=int(0.6*fs))
                    
                    if len(r_peaks) >= 3:
                        # Calculate QRS Axis
                        from .twelve_lead_test import calculate_qrs_axis
                        qrs_axis_result = calculate_qrs_axis(lead_I_filt, lead_aVF_filt, r_peaks, fs=fs, window_ms=100)
                        if qrs_axis_result != "--":
                            qrs_axis_deg = qrs_axis_result
                        
                        # Helper function to calculate axis for any wave
                        def calculate_wave_axis(lead_I_sig, lead_aVF_sig, wave_peaks, fs, window_before_ms, window_after_ms):
                            """Calculate axis for P or T wave"""
                            if len(lead_I_sig) < 100 or len(lead_aVF_sig) < 100 or len(wave_peaks) == 0:
                                return "--"
                            window_before = int(window_before_ms * fs / 1000)
                            window_after = int(window_after_ms * fs / 1000)
                            net_I = []
                            net_aVF = []
                            for peak in wave_peaks:
                                start = max(0, peak - window_before)
                                end = min(len(lead_I_sig), peak + window_after)
                                if end > start:
                                    net_I.append(np.sum(lead_I_sig[start:end]))
                                    net_aVF.append(np.sum(lead_aVF_sig[start:end]))
                            if len(net_I) == 0:
                                return "--"
                            mean_I = np.mean(net_I)
                            mean_aVF = np.mean(net_aVF)
                            if abs(mean_I) < 1e-6 and abs(mean_aVF) < 1e-6:
                                return "--"
                            axis_rad = np.arctan2(mean_aVF, mean_I)
                            axis_deg = np.degrees(axis_rad)
                            
                            # Normalize to -180 to +180 (clinical standard, matches standardized function)
                            # This ensures consistency with calculate_axis_from_median_beat()
                            if axis_deg > 180:
                                axis_deg -= 360
                            if axis_deg < -180:
                                axis_deg += 360
                            
                            return f"{int(round(axis_deg))}°"
                        
                        # Detect P peaks (adaptive window based on HR)
                        # Calculate HR from R-peaks for adaptive detection
                        if len(r_peaks) >= 2:
                            rr_intervals = np.diff(r_peaks) / fs  # in seconds
                            mean_rr = np.mean(rr_intervals)
                            estimated_hr = 60.0 / mean_rr if mean_rr > 0 else 100
                        else:
                            estimated_hr = 100
                        
                        # Adaptive P wave detection window based on HR
                        # At very high HR (>140), P waves are hard to detect due to T-P overlap
                        # At high HR (>100), use narrower window to avoid T wave overlap
                        if estimated_hr > 140:
                            # Very high HR: use very narrow window or skip P detection
                            p_window_before_ms = 0.12  # 120ms - very narrow
                            p_window_after_ms = 0.08   # 80ms - very narrow
                            use_lead_I_for_p = True  # Prefer Lead I at very high HR
                        elif estimated_hr > 100:
                            p_window_before_ms = 0.15  # 150ms instead of 200ms
                            p_window_after_ms = 0.10   # 100ms instead of 120ms
                            use_lead_I_for_p = False
                        else:
                            p_window_before_ms = 0.20  # Standard 200ms
                            p_window_after_ms = 0.12   # Standard 120ms
                            use_lead_I_for_p = False
                        
                        # For very high HR, try Lead I first (usually clearer P waves)
                        if use_lead_I_for_p:
                            p_peaks = []
                            for r in r_peaks[1:-1]:  # Skip first and last
                                p_start = max(0, r - int(p_window_before_ms*fs))
                                p_end = max(0, r - int(p_window_after_ms*fs))
                                if p_end > p_start:
                                    # Try Lead I first at very high HR
                                    segment = lead_I_filt[p_start:p_end]
                                    if len(segment) > 0:
                                        # Look for positive deflection (P wave is usually positive)
                                        # Use argmax but validate it's actually a peak
                                        p_idx = p_start + np.argmax(segment)
                                        # Validate: peak should be above baseline
                                        if segment[np.argmax(segment)] > np.mean(segment) + 0.1 * np.std(segment):
                                            p_peaks.append(p_idx)
                        else:
                            # Standard detection using Lead aVF
                            p_peaks = []
                            for r in r_peaks[1:-1]:  # Skip first and last
                                p_start = max(0, r - int(p_window_before_ms*fs))
                                p_end = max(0, r - int(p_window_after_ms*fs))
                                if p_end > p_start:
                                    segment = lead_aVF_filt[p_start:p_end]
                                    if len(segment) > 0:
                                        p_idx = p_start + np.argmax(segment)
                                        p_peaks.append(p_idx)
                        
                        # Try to calculate P axis even with fewer peaks if possible
                        if len(p_peaks) >= 2:
                            p_axis_result = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks, fs, 20, 60)
                            if p_axis_result != "--":
                                # Validate P axis is in normal range (0-75°)
                                p_axis_num = int(str(p_axis_result).replace("°", ""))
                                # Normalize to -180 to +180 range for comparison
                                if p_axis_num > 180:
                                    p_axis_num_normalized = p_axis_num - 360
                                else:
                                    p_axis_num_normalized = p_axis_num
                                
                                # Debug: Print HR and P axis for troubleshooting
                                print(f" P axis validation: HR={estimated_hr:.1f} BPM, P_axis={p_axis_num}°, normalized={p_axis_num_normalized}°")
                                
                                # Check if P axis is in normal range (0 to 75°)
                                # P axis normal range: 0° to +75°
                                # For values > 180°, normalize to negative (e.g., 174° stays 174°, but 200° becomes -160°)
                                # But 174° is still abnormal (> 75°)
                                is_normal = False
                                if p_axis_num_normalized >= 0 and p_axis_num_normalized <= 75:
                                    is_normal = True
                                elif p_axis_num >= 0 and p_axis_num <= 75:
                                    is_normal = True
                                
                                if is_normal:
                                    p_axis_deg = p_axis_result
                                else:
                                    # P axis abnormal - try multiple fallback methods to get best possible value
                                    # Always try to return a value instead of "--"
                                    hr_from_data = data.get('HR', 0) if data else 0
                                    hr_from_data = hr_from_data if isinstance(hr_from_data, (int, float)) else 0
                                    
                                    # Try multiple fallback methods
                                    p_axis_candidates = []
                                    
                                    # Method 1: Try Lead I detection (if not already used)
                                    if not use_lead_I_for_p:
                                        p_peaks_alt1 = []
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(p_window_before_ms*fs))
                                            p_end = max(0, r - int(p_window_after_ms*fs))
                                            if p_end > p_start:
                                                segment = lead_I_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt1.append(p_idx)
                                        
                                        if len(p_peaks_alt1) >= 2:
                                            p_axis_result_alt1 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt1, fs, 20, 60)
                                            if p_axis_result_alt1 != "--":
                                                p_axis_candidates.append(p_axis_result_alt1)
                                    
                                    # Method 2: Try Lead aVF detection (if not already used)
                                    if use_lead_I_for_p:
                                        p_peaks_alt2 = []
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(p_window_before_ms*fs))
                                            p_end = max(0, r - int(p_window_after_ms*fs))
                                            if p_end > p_start:
                                                segment = lead_aVF_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt2.append(p_idx)
                                        
                                        if len(p_peaks_alt2) >= 2:
                                            p_axis_result_alt2 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt2, fs, 20, 60)
                                            if p_axis_result_alt2 != "--":
                                                p_axis_candidates.append(p_axis_result_alt2)
                                    
                                    # Method 3: Try wider window for high HR
                                    if estimated_hr > 100:
                                        p_peaks_alt3 = []
                                        wider_window_before = 0.18 if estimated_hr > 140 else 0.16
                                        wider_window_after = 0.11 if estimated_hr > 140 else 0.10
                                        for r in r_peaks[1:-1]:
                                            p_start = max(0, r - int(wider_window_before*fs))
                                            p_end = max(0, r - int(wider_window_after*fs))
                                            if p_end > p_start:
                                                segment = lead_I_filt[p_start:p_end]
                                                if len(segment) > 0:
                                                    p_idx = p_start + np.argmax(segment)
                                                    p_peaks_alt3.append(p_idx)
                                        
                                        if len(p_peaks_alt3) >= 2:
                                            p_axis_result_alt3 = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks_alt3, fs, 15, 50)
                                            if p_axis_result_alt3 != "--":
                                                p_axis_candidates.append(p_axis_result_alt3)
                                    
                                    # Add original result as candidate
                                    p_axis_candidates.append(p_axis_result)
                                    
                                    # Select best candidate: prefer values in normal range, otherwise use closest to normal
                                    best_p_axis = None
                                    best_score = -1
                                    
                                    for candidate in p_axis_candidates:
                                        if candidate == "--":
                                            continue
                                        cand_num = int(str(candidate).replace("°", ""))
                                        if cand_num > 180:
                                            cand_normalized = cand_num - 360
                                        else:
                                            cand_normalized = cand_num
                                        
                                        # Score: prefer values in normal range (0-75°)
                                        if 0 <= cand_normalized <= 75:
                                            score = 100 - abs(cand_normalized - 37.5)  # Closer to middle (37.5°) is better
                                        else:
                                            # For abnormal values, prefer closer to normal range
                                            if cand_normalized > 75:
                                                score = max(0, 50 - (cand_normalized - 75))
                                            else:
                                                score = max(0, 50 - abs(cand_normalized))
                                        
                                        if score > best_score:
                                            best_score = score
                                            best_p_axis = candidate
                                    
                                    # Use best candidate or original if no better option
                                    if best_p_axis:
                                        p_axis_deg = best_p_axis
                                        if best_p_axis != p_axis_result:
                                            print(f" P axis adjusted using fallback method: {p_axis_deg} (original: {p_axis_result}, HR: {estimated_hr:.0f} BPM)")
                                        else:
                                            print(f" P axis value: {p_axis_deg} (may be less accurate at HR {estimated_hr:.0f} BPM)")
                                    else:
                                        # Last resort: use original value even if abnormal
                                        p_axis_deg = p_axis_result
                                        print(f" P axis value: {p_axis_deg} (calculated at HR {estimated_hr:.0f} BPM, may be less accurate)")
                        else:
                            # If less than 2 P peaks detected, try to calculate with available peaks
                            if len(p_peaks) >= 1:
                                # Try with single peak (less accurate but better than "--")
                                p_axis_result_single = calculate_wave_axis(lead_I_filt, lead_aVF_filt, p_peaks, fs, 20, 60)
                                if p_axis_result_single != "--":
                                    p_axis_deg = p_axis_result_single
                                    print(f" P axis calculated with limited peaks: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, may be less accurate)")
                            else:
                                # Last resort: try to estimate from R-peaks timing
                                # Use average PR interval assumption (150ms) to estimate P wave position
                                if len(r_peaks) >= 3:
                                    estimated_p_peaks = []
                                    for r in r_peaks[1:-1]:
                                        estimated_p_idx = max(0, r - int(0.15*fs))  # Assume 150ms PR interval
                                        if estimated_p_idx < len(lead_I_filt):
                                            estimated_p_peaks.append(estimated_p_idx)
                                    
                                    if len(estimated_p_peaks) >= 2:
                                        p_axis_result_est = calculate_wave_axis(lead_I_filt, lead_aVF_filt, estimated_p_peaks, fs, 20, 60)
                                        if p_axis_result_est != "--":
                                            p_axis_deg = p_axis_result_est
                                            print(f" P axis estimated from R-peaks timing: {p_axis_deg} (HR: {estimated_hr:.0f} BPM, estimated)")
                        
                        # Detect T peaks (100-300ms after R peaks)
                        t_peaks = []
                        for r in r_peaks[1:-1]:  # Skip first and last
                            t_start = min(len(lead_aVF_filt), r + int(0.10*fs))
                            t_end = min(len(lead_aVF_filt), r + int(0.30*fs))
                            if t_end > t_start:
                                segment = lead_aVF_filt[t_start:t_end]
                                if len(segment) > 0:
                                    t_idx = t_start + np.argmax(segment)
                                    t_peaks.append(t_idx)
                        
                        if len(t_peaks) >= 2:
                            t_axis_result = calculate_wave_axis(lead_I_filt, lead_aVF_filt, t_peaks, fs, 40, 80)
                            if t_axis_result != "--":
                                t_axis_deg = t_axis_result
                        
                        print(f" Calculated P/QRS/T Axis: P={p_axis_deg}, QRS={qrs_axis_deg}, T={t_axis_deg}")
        except Exception as e:
            print(f" Axis calculation failed: {e}")
            import traceback
            traceback.print_exc()
        
    sanitized_p_axis = _sanitize_axis_for_report("P", p_axis_deg, qrs_axis_deg, data.get("HR_bpm") or data.get("HR"))
    sanitized_qrs_axis = _sanitize_axis_for_report("QRS", qrs_axis_deg, qrs_axis_deg, data.get("HR_bpm") or data.get("HR"))
    sanitized_t_axis = _sanitize_axis_for_report("T", t_axis_deg, qrs_axis_deg, data.get("HR_bpm") or data.get("HR"))

    # Format axis values for display (remove ° symbol for compact display)
    p_axis_display = str(sanitized_p_axis).replace("°", "") if sanitized_p_axis != "--" else "--"
    qrs_axis_display = str(sanitized_qrs_axis).replace("°", "") if sanitized_qrs_axis != "--" else "--"
    t_axis_display = str(sanitized_t_axis).replace("°", "") if sanitized_t_axis != "--" else "--"
    
    # Extract numeric values for JSON storage (convert from string format like "45°" to int)
    def extract_axis_value(axis_str):
        """Extract numeric value from axis string like '45°' or '--'"""
        if axis_str == "--":
            return 0  # Default value if not calculated
        try:
            # Remove ° symbol and convert to int
            return int(str(axis_str).replace("°", "").strip())
        except (ValueError, AttributeError):
            return 0
    
    p_mm = extract_axis_value(sanitized_p_axis)
    qrs_mm = extract_axis_value(sanitized_qrs_axis)
    t_mm = extract_axis_value(sanitized_t_axis)
    
    # Calculate axis values for data dictionary
    rv5_mv = data.get('rv5_mv') if data.get('rv5_mv') is not None else (data.get('rv5') or 0.0)
    sv1_mv = data.get('sv1_mv') if data.get('sv1_mv') is not None else (data.get('sv1') or 0.0)

    # If we know a source lead is absent, force that metric to zero instead of
    # leaking the other lead's value into the missing slot.
    if data.get('_rv5_source_present') is False:
        rv5_mv = 0.0
    if data.get('_sv1_source_present') is False:
        sv1_mv = 0.0

    if rv5_mv is None:
        rv5_mv = 0.0
    if sv1_mv is None:
        sv1_mv = 0.0

    data['rv5_mv'] = rv5_mv
    data['sv1_mv'] = sv1_mv
    data['p_axis'] = sanitized_p_axis
    data['QRS_axis'] = sanitized_qrs_axis
    data['t_axis'] = sanitized_t_axis

    # ── CALL RESTRUCTURED HEADER (Handles all 4 columns) ─────────────────
    _add_patient_header(master_drawing, full_name, age, gender, patient, date_time_str, data, settings_manager)

    


    
    from reportlab.pdfbase.pdfmetrics import stringWidth
    label_text = "Doctor Name: "
    
    # Value from Save ECG -> passed in 'patient'
    # Accept both keys: `doctor` (legacy) and `doctor_name` (signup/profile mapping)
    doctor = ""
    try:
        if patient:
            doctor = str(patient.get("doctor", "") or patient.get("doctor_name", "") or "").strip()
    except Exception:
        doctor = ""
  
    # Reference Report Confirmed by (above Doctor Name)
    confirmed_label = String(3.6 * mm, 24.3 * mm, "Reference Report Confirmed by ", 
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(confirmed_label)

    # Doctor Name (below V6 lead)
    doctor_name_label = String(3.6 * mm, 19.0 * mm, "Doctor Name: ", 
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(doctor_name_label)
    
    if doctor:
        value_x = 3.6 * mm + stringWidth("Doctor Name: ", "Helvetica", 8) + 5 * mm
        doctor_name_value = String(value_x, 19.0 * mm, doctor,
                                fontSize=8, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(doctor_name_value)

    # Doctor Signature (below Doctor Name)
    doctor_sign_label = String(3.6 * mm, 13.7 * mm, "Doctor Sign: ", 
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(doctor_sign_label)

    # Add RIGHT-SIDE Conclusion Box (moved to the right) - NOW DYNAMIC FROM DASHBOARD (12 conclusions max) - MADE SMALLER
    # SHIFTED DOWN further (additional 5 points)
    conclusion_y_start = 26.8 * mm  # Shifted up by 20mm  # Shifted down from 0 to -5 (5 more points down to shift container lower)
    
    # Create a rectangular box for conclusions (shifted right) - INCREASED HEIGHT (same position)
    # Height increased: bottom extended down (top position same). Length increased by 20 (x position fixed)
    from reportlab.graphics.shapes import Rect
    conclusion_box = Rect(70.6 * mm, conclusion_y_start - 26.5 * mm, 125.2 * mm, 26.5 * mm,  # Width 325→345 (+20); height 65→75 (+10)
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # CENTERED and STYLISH "Conclusion" header - DYNAMIC - SMALLER (AT TOP OF CONTAINER - CLOSE TO TOP LINE)
    # Box center: 200 + (325/2) = 362.5, so text should be centered around 362.5
    # Box top is at conclusion_y_start - 55, so header should be very close to top line
    conclusion_header = String(127.9 * mm, conclusion_y_start - 3.0 * mm, "CONCLUSION",  # Moved very close to top line: y=0→-53 (just below top edge at -55)
                              fontSize=7, fontName="Helvetica-Bold",  # Reduced from 9 to 7
                              fillColor=colors.HexColor("#2c3e50"),
                              textAnchor="middle")  # This centers the text
    master_drawing.add(conclusion_header)
    
    # DYNAMIC conclusions from dashboard in the box - SINGLE COLUMN to avoid overlapping
    print(f" Drawing conclusions in graph from filtered list: {filtered_conclusions}")
    
    # Draw conclusions vertically in a single column
    row_spacing = 4.0 * mm  # Increased vertical spacing
    start_y = conclusion_y_start - 8.0 * mm  # Starting Y position (further down from top)
    box_bottom = conclusion_y_start - 26.5 * mm  # Bottom edge of the box
    
    # FIRST LINE: Patient Name (Bold)
    name_text = f"Name: {full_name}"
    x_pos = 74.1 * mm  # Align with the box's left-ish side
    master_drawing.add(String(x_pos, start_y, name_text, 
                             fontSize=9, fontName="Helvetica-Bold", fillColor=colors.black))
    
    # THEN: Findings "uske baad"
    for idx, conclusion in enumerate(filtered_conclusions):
        row_y = start_y - ((idx + 1) * row_spacing)
        
        # User request: If getting cropped (exceeds box height), don't put in this.
        if row_y < box_bottom + 2.0 * mm:  # 2mm padding from bottom
            print(f" Skipping conclusion {idx+1} as it would be cropped")
            continue
            
        conc_text = f"{idx + 1}. {conclusion}"
        
        conc = String(x_pos, row_y, conc_text, 
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(conc)
        
    print(f" Added {len(filtered_conclusions)} REAL Conclusions in single column (no cropping)")

    print(f" Added Patient Info, Vital Parameters, {len(filtered_conclusions)} REAL Conclusions (no empty/---), and Doctor Name/Signature to ECG grid")
    
    # STEP 5: Add SINGLE master drawing to story (NO containers)
    story.append(master_drawing)
    
    print(f" Added SINGLE master drawing with {successful_graphs}/12 ECG leads (ZERO containers)!")
    
    # Final summary
    if is_demo_mode:
        print(f"\n{'='*60}")
        print(f" DEMO MODE REPORT SUMMARY:")
        print(f"   • Total leads processed: {successful_graphs}/12")
        print(f"   • Demo mode: {'ON' if is_demo_mode else 'OFF'}")
        if successful_graphs == 0:
            print(f"    WARNING: No ECG graphs were added to the report!")
            print(f"    SOLUTION: Ensure demo is running for 5-10 seconds before generating report")
        elif successful_graphs < 12:
            print(f"    WARNING: Only {successful_graphs} graphs added (expected 12)")
        else:
            print(f"    SUCCESS: All 12 ECG graphs added successfully!")
        print(f"{'='*60}\n")

    # REFERENCE METRICS TABLE — Removed

    # Measurement info (NO background)
    measurement_style = ParagraphStyle(
        'MeasurementStyle',
        fontSize=8,
        textColor=colors.HexColor("#000000"),
        alignment=1  # center
        # backColor removed
    )


    # Summary (NO background)
    summary_style = ParagraphStyle( 
        'SummaryStyle',
        fontSize=10,
        textColor=colors.HexColor("#000000"),
        alignment=1  # center
        # backColor removed
    )
    # summary_para = Paragraph(f"ECG Report: {successful_graphs}/12 leads displayed", summary_style)
    # story.append(summary_para)

    # Extract patient data for use in canvas drawing
    patient_org = patient.get("Org.", "") if patient else ""
    patient_doctor_mobile = patient.get("doctor_mobile", "") if patient else ""
    
    # Helper: draw logo on every page AND ALIGNED pink grid background on Page 2
    def _draw_logo_and_footer(canvas, doc):
        import os
        from reportlab.lib.units import mm
        
        # STEP 1: Draw ECG grid across full A4 with standard 5mm boxes
        if canvas.getPageNumber() == 1:  # Changed from 3 to 2
            a4_width, a4_height = canvas._pagesize
            grid_x = 5.0 * mm
            grid_y = 6.0 * mm
            grid_width = 190.0 * mm
            grid_height = 265.0 * mm
            
            # Fill exact grid area with pink background
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(grid_x, grid_y, grid_width, grid_height, fill=1, stroke=0)
            
            # ECG grid colors - darker for better visibility
            light_grid_color = colors.HexColor("#ffd1d1")  
            major_grid_color = colors.HexColor("#ffb3b3")   
            
            # Exact physical spacings
            minor_spacing = 1.0 * mm
            major_spacing = 5.0 * mm
            
            # Draw minor grid lines FIRST - bottom layer
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.4)
            
            # Vertical minor lines
            x = grid_x
            while x <= grid_x + grid_width + 0.01:
                canvas.line(x, grid_y, x, grid_y + grid_height)
                x += minor_spacing
            
            # Horizontal minor lines
            y = grid_y
            while y <= grid_y + grid_height + 0.01:
                canvas.line(grid_x, y, grid_x + grid_width, y)
                y += minor_spacing
            
            # Draw major grid lines ON TOP - standard 5mm spacing
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(0.8)
            
            # Vertical major lines
            x = grid_x
            while x <= grid_x + grid_width + 0.01:
                canvas.line(x, grid_y, x, grid_y + grid_height)
                x += major_spacing
            
            # Horizontal major lines
            y = grid_y
            while y <= grid_y + grid_height + 0.01:
                canvas.line(grid_x, y, grid_x + grid_width, y)
                y += major_spacing
            

        
        # STEP 1.5: Draw Org. and Phone No. labels on Page 1 (TOP LEFT) - REMOVED
        if canvas.getPageNumber() == 1:
            canvas.saveState()
            
            # Position in top-left corner (below margin) - REMOVED
            # x_pos = doc.leftMargin  # 30 points from left
            # y_pos = doc.height + doc.bottomMargin - 5  # 20 points from top
            
            # Always draw "Org." label with value - REMOVED
            # canvas.setFont("Helvetica-Bold", 10)
            # canvas.setFillColor(colors.black)
            # org_label = "Org:"
            # canvas.drawString(x_pos, y_pos, org_label)
            
            # Calculate width of label and add small gap - REMOVED
            # org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
            # canvas.setFont("Helvetica", 10)
            # canvas.drawString(x_pos + org_label_width + 5, y_pos, patient_org if patient_org else "")
            
            # y_pos -= 15  # Move down for next line
            
            # Always draw "Phone No." label with value - REMOVED
            # canvas.setFont("Helvetica-Bold", 10)
            # canvas.setFillColor(colors.black)
            # phone_label = "Phone No:"
            # canvas.drawString(x_pos, y_pos, phone_label)
            
            # Calculate width of label and add small gap - REMOVED
            # phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
            # canvas.setFont("Helvetica", 10)
            # canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient_doctor_mobile if patient_doctor_mobile else "")
            
            canvas.restoreState()
        
        # STEP 2: Draw logo on all pages (existing code)
        # Prefer PNG (ReportLab-friendly); fallback to WebP if PNG missing
        # Use resource_path helper for PyInstaller compatibility
        logo_filename = "DeckmountLogo.png"
        logo_path = _get_resource_path(f"assets/{logo_filename}")
        
        # Fallback to old names if the new one is missing
        if not os.path.exists(logo_path):
            png_path = _get_resource_path("assets/Deckmountimg.png")
            webp_path = _get_resource_path("assets/Deckmount.webp")
            logo_path = png_path if os.path.exists(png_path) else webp_path

        if os.path.exists(logo_path):
            canvas.saveState()
            # Different positioning for different pages
            if canvas.getPageNumber() == 1:
                logo_w, logo_h = 120, 40  # bigger size for ECG page
                # SHIFTED LEFT FROM RIGHT TOP CORNER
                page_width, page_height = canvas._pagesize
                x = page_width - logo_w - 35  # Shifted 50 pixels left from right edge
                y = page_height - logo_h  # Top edge touch
            else:
                logo_w, logo_h = 120, 40  # normal size for other pages
                x = doc.width + doc.leftMargin - logo_w
                y = doc.height + doc.bottomMargin - logo_h  # top positioning
            try:
                canvas.drawImage(logo_path, x, y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask='auto')
            except Exception:
                # If WebP unsupported, silently skip
                pass
            canvas.restoreState()
        
        # STEP 3: Add footer with company address on all pages
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.black)  # Ensure text is black on pink background
        
        serial_num = ""
        if data:
            serial_num = data.get("machine_serial", "") or data.get("machine_serial_number", "")
        if not serial_num:
            try:
                from utils.settings_manager import SettingsManager
                sm = SettingsManager()
                serial_num = sm.get_setting("machine_serial_number", "")
            except Exception:
                pass
        serial_suffix = serial_num[-4:] if len(serial_num) >= 4 else serial_num
        
        if serial_suffix:
            footer_text = f"Deckmount Electronics Pvt Ltd | Rhythm Ultra Max | IEC 60601 | {serial_suffix} | Made in India"
        else:
            footer_text = "Deckmount Electronics Pvt Ltd | Rhythm Ultra Max | IEC 60601 | Made in India"
            
        # Center the footer text at bottom of page
        text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
        x = (doc.width + doc.leftMargin + doc.rightMargin - text_width) / 2
        y = 10  # 20 points from bottom
        canvas.drawString(x, y, footer_text)
        canvas.restoreState()

    # Save parameters to a JSON index for later reuse
    try:
        from datetime import datetime
        reports_dir = str(data_file("reports"))
        os.makedirs(reports_dir, exist_ok=True)
        index_path = os.path.join(reports_dir, 'index.json')
        metrics_path = os.path.join(reports_dir, 'metrics.json')

        # Get username from dashboard_instance or ecg_test_page
        username = ""
        try:
            if dashboard_instance and hasattr(dashboard_instance, 'username'):
                username = dashboard_instance.username or ""
            elif ecg_test_page:
                # Try to get username from ecg_test_page's dashboard reference
                if hasattr(ecg_test_page, 'dashboard_instance') and ecg_test_page.dashboard_instance:
                    username = getattr(ecg_test_page.dashboard_instance, 'username', '') or ""
                # Try to traverse parent widgets to find dashboard
                widget = ecg_test_page
                for _ in range(10):  # prevent infinite loops
                    if widget is None:
                        break
                    if hasattr(widget, 'username'):
                        username = widget.username or ""
                        break
                    widget = widget.parent()
        except Exception:
            username = ""

        params_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "file": os.path.abspath(filename),
            "patient": {
                "name": full_name,
                "age": str(age),
                "gender": gender,
                "date_time": date_time_str,
            },
            "metrics": {
                "HR_bpm": HR,
                "PR_ms": PR,
                "QRS_ms": QRS,
                "QT_ms": QT,
                "QTc_ms": QTc,
                "ST_ms": ST,
                "RR_ms": RR,
                "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3) if rv5_sv1_sum is not None else None,
                "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
                "RV5_SV1_mV": [
                    round(rv5_mv, 3) if rv5_mv is not None else None,
                    round(sv1_mv, 3) if sv1_mv is not None else None,
                ],
                "QTCF_ms": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val else None,
            },
            "username": username  # Add username to track report ownership
        }

        existing_list = []
        if os.path.exists(index_path):
            try:
                with open(index_path, 'r') as f:
                    existing_json = json.load(f)
                    if isinstance(existing_json, list):
                        existing_list = existing_json
                    elif isinstance(existing_json, dict) and isinstance(existing_json.get('entries'), list):
                        existing_list = existing_json['entries']
            except Exception:
                existing_list = []

        existing_list.append(params_entry)

        # Persist as a flat list for simplicity
        with open(index_path, 'w') as f:
            json.dump(existing_list, f, indent=2)
        print(f" Saved parameters to {index_path}")

        # Save ONLY the 11 metrics in a lightweight separate JSON file (append to list)
        metrics_entry = {
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "file": os.path.abspath(filename),
            "HR_bpm": HR,
            "PR_ms": PR,
            "QRS_ms": QRS,
            "QT_ms": QT,
            "QTc_ms": QTc,
            "ST_ms": ST,
            "RR_ms": RR,
            "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3) if rv5_sv1_sum is not None else None,
            "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
            "QTCF": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val and qtcf_val > 0 else None,
            "RV5_SV1_mV": [
                round(rv5_mv, 3) if rv5_mv is not None else None,
                round(sv1_mv, 3) if sv1_mv is not None else None,
            ]
        }

        metrics_list = []
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    mj = json.load(f)
                    if isinstance(mj, list):
                        metrics_list = mj
            except Exception:
                metrics_list = []

        metrics_list.append(metrics_entry)

        with open(metrics_path, 'w') as f:
            json.dump(metrics_list, f, indent=2)
        print(f" Saved 11 metrics to {metrics_path}")
    except Exception as e:
        print(f" Could not save parameters JSON: {e}")

    # Build PDF - single page only
    doc.build(story, onFirstPage=_draw_logo_and_footer)
    print(f"✓ ECG Report generated: {filename}")

    # Optionally log history entry for ECG reports
    if log_history:
        try:
            from dashboard.history_window import append_history_entry
            entry_patient = patient if isinstance(patient, dict) else {}
            # Get username from dashboard_instance if not provided
            if not username and dashboard_instance:
                username = getattr(dashboard_instance, 'username', None)
            owner_full_name = ""
            try:
                if dashboard_instance:
                    owner_full_name = (getattr(dashboard_instance, "user_details", {}) or {}).get("full_name") or ""
            except Exception:
                owner_full_name = ""
            append_history_entry(
                entry_patient,
                os.path.abspath(filename),
                report_type="ECG",
                username=username,
                owner_full_name=owner_full_name or username,
            )
        except Exception as hist_err:
            print(f" Failed to append ECG history entry: {hist_err}")
    
    # Sync full 12-lead report package to backend (metrics + signup + ECG details)
    try:
        backend_metrics_payload = {
            "HR_bpm": HR, "PR_ms": PR, "QRS_ms": QRS, "QT_ms": QT, "QTc_ms": QTc,
            "ST_ms": ST, "RR_ms": RR,
            "RV5_plus_SV1_mV": round(rv5_sv1_sum, 3) if rv5_sv1_sum is not None else None,
            "P_QRS_T_mm": [p_mm, qrs_mm, t_mm],
            "QTCF_ms": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val else None,
            "RV5_SV1_mV": [
                round(rv5_mv, 3) if rv5_mv is not None else None,
                round(sv1_mv, 3) if sv1_mv is not None else None,
            ],
        }
        _sync_report_package_to_backend(
            filename=filename,
            patient=patient if isinstance(patient, dict) else {},
            data=data if isinstance(data, dict) else {},
            metrics_payload=backend_metrics_payload,
            username=username,
            ecg_test_page=ecg_test_page,
            sampling_rate=computed_sampling_rate,
            ecg_data_file=saved_data_file_path if 'saved_data_file_path' in locals() else ecg_data_file,
        )
    except Exception as _be:
        print(f"  Backend package sync failed: {_be}")

    # Upload to cloud if configured
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        # --- NEW UNIFIED PAYLOAD DISPATCH ---
        from utils.ecg_payload_builder import dispatch_12lead_report
        
        _uname = data.get('username') or (username if 'username' in locals() else None)
        _signup = _load_signup_details_for_username(_uname) if '_load_signup_details_for_username' in globals() and _uname else {}
        master_phone = str(
            (patient or {}).get('phone')
            or (patient or {}).get('mobile_no')
            or _signup.get('phone')
            or _signup.get('contact')
            or username
            or ""
        ).strip()
        
        dispatch_12lead_report(
            data=data,
            patient=patient or {},
            pdf_path=filename,
            settings_manager=settings_manager if 'settings_manager' in locals() else None,
            signup_details=_signup,
            ecg_test_page=ecg_test_page if 'ecg_test_page' in locals() else None,
            ecg_data_file=saved_data_file_path if 'saved_data_file_path' in locals() else (ecg_data_file if 'ecg_data_file' in locals() else None),
            report_format="12_1",
            conclusions=filtered_conclusions if 'filtered_conclusions' in locals() else None,
            arrhythmia=None,
        )
        print("  Dispatched 12-lead unified payload")
        # -------------------------------------
        
        from utils.cloud_uploader import get_cloud_uploader
        
        cloud_uploader = get_cloud_uploader()
        if cloud_uploader.is_configured():
            print(f"  Uploading report package to cloud ({cloud_uploader.cloud_service})...")
            
            # Prepare clinical measurements
            clinical_measurements = {
                "heart_rate": HR,
                "pr_interval": PR,
                "qrs_duration": QRS,
                "qt_interval": QT,
                "qtc": QTc,
                "p_axis": p_mm,
                "qrs_axis": qrs_mm,
                "t_axis": t_mm
            }
            
            # Prepare metadata
            report_metadata = {
                "patient_name": data.get('patient', {}).get('name', 'Unknown'),
                "patient_age": str(data.get('patient', {}).get('age', '')),
                "report_date": data.get('date', ''),
                "machine_serial": data.get('machine_serial', ''),
                "master_phone": master_phone,
                "report_type": "12_lead_ecg"
            }
            
            # Upload the complete report package
            result = cloud_uploader.upload_complete_report_package(
                pdf_path=filename,
                patient_data=patient if isinstance(patient, dict) else {},
                ecg_data_file=saved_data_file_path if 'saved_data_file_path' in locals() else None,
                report_metadata=report_metadata,
                report_type="12_LEAD_ECG",
                clinical_measurements=clinical_measurements
            )
            
            if result.get('status') == 'success':
                print(f"✓ Report package uploaded successfully to {cloud_uploader.cloud_service}")
            else:
                print(f"  Cloud uploader fallback: attempting direct report file uploads...")
                # If package upload failed or was skipped, try uploading files directly
                cloud_uploader.upload_report(filename, metadata=report_metadata)
                if saved_data_file_path and os.path.exists(saved_data_file_path):
                    cloud_uploader.upload_report(saved_data_file_path, metadata=report_metadata)
            
            # ALWAYS ensure the companion JSON is uploaded with the same name as the PDF
            # for easy identification in the S3 File Browser.
            if saved_data_file_path and os.path.exists(saved_data_file_path):
                # Only upload if not already handled by the package uploader (to avoid double work, 
                # although upload_report handles duplicates anyway)
                cloud_uploader.upload_report(saved_data_file_path, metadata=report_metadata)
        else:
            print("  Cloud upload not configured (see cloud_config_template.txt)")
            
    except ImportError:
        print("  Cloud uploader not available")
    except Exception as e:
        print(f"  Cloud upload error: {e}")
    
    return filename, metrics_entry


# REMOVE ENTIRE create_sample_ecg_images function (lines ~1222-1257)

# REMOVE ENTIRE main execution block (lines ~1260-1265)
# if __name__ == "__main__":
#     # Create sample images with transparency (force recreation)
#     create_sample_ecg_images(force_recreate=True)
#     
#     # Generate report
#     generate_ecg_report("test_ecg_report.pdf")





