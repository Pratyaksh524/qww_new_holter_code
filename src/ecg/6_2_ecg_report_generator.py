from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer, Image, PageBreak,
    PageTemplate, Frame, NextPageTemplate, BaseDocTemplate
)
from reportlab.graphics.shapes import Drawing, Line, Rect, Path, String
from reportlab.graphics import renderPDF
from reportlab.lib.units import mm

class TransparentDrawing(Drawing):
    """Drawing subclass that suppresses ReportLab's default white background rect."""
    def drawOn(self, canvas, x, y, _sW=0):
        canvas.saveState()
        canvas.translate(x, y)
        renderPDF.draw(self, canvas, 0, 0, showBoundary=0)
        canvas.restoreState()
from reportlab.pdfbase.pdfmetrics import stringWidth
import os
from utils.app_paths import data_file
import sys
import json
import matplotlib.pyplot as plt  
import matplotlib
import os
import logging
from datetime import datetime

# Setup logging - use a path relative to this script to work cross-platform
_log_base = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'logs'))
os.makedirs(_log_base, exist_ok=True)
log_file = os.path.join(_log_base, f"6_2_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(filename=log_file, level=logging.DEBUG,
                   format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logging.getLogger().addHandler(console_handler)

import numpy as np

# matplotlib.use('Agg') # Removed to prevent main thread Qt canvas corruption


# ------------------------ ECG grid scale constants ------------------------
# Clinical ECG standard: 1 large box = exactly 5.0 mm, 1 small box = exactly 1.0 mm
# At 25 mm/s: 5 large boxes per second (100 samples/box at 500 Hz)
# At 50 mm/s: 10 large boxes per second
ECG_BASE_BOX_MM = 5.0
ECG_LARGE_BOX_MM = 5.0   # Clinical standard: exactly 5.0 mm per large box
ECG_SMALL_BOX_MM = 1.0   # Clinical standard: exactly 1.0 mm per small box
ECG_SPEED_SCALE = 1.0    # No artificial scaling – waveform speed matches clinical paper exactly
SIX_TWO_SAMPLES_COLUMN = 2500
SIX_TWO_SAMPLES_EXTRA_II = 5000  # 5 seconds at 500Hz = 5000 samples for Lead II rhythm strip

# UNIFIED BOX CONFIGURATION
COLUMN1_BOXES = 26.3
COLUMN2_BOXES = 26.3  
EXTRA_LEAD2_BOXES = 52.0  


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
            saved_data["sampling_rate"] = float(ecg_test_page.sampler.sampling_rate)
    
    # Save each lead's data - use FULL buffer (ecg_buffers if available, otherwise data)
    # Priority: Use ecg_buffers (5000 samples) if available, otherwise use data (1000 samples)
    
    # Debug: Check what attributes ecg_test_page has
    print(f" DEBUG: ecg_test_page attributes check:")
    print(f"   has ecg_buffers: {hasattr(ecg_test_page, 'ecg_buffers')}")
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

def calculate_time_window_from_bpm_and_wave_speed(hr_bpm, wave_speed_mm_s, desired_beats=None, num_boxes=None):
    """
    Calculate optimal time window based on BPM and wave_speed
    
    Important: Report ECG strip width = (boxes × ECG_LARGE_BOX_MM)
     wave_speed  time calculate    factor use :
        Time from wave_speed = (graph_width_mm / effective_wave_speed_mm_s) seconds
    
    Formula:
        - Time window = (graph_width_mm / effective_wave_speed_mm_s) seconds ONLY
        - BPM window is NOT used - only wave speed window
        - Beats = (BPM / 60) × time_window
        - Final window clamped maximum 20 seconds (NO minimum clamp)
    

    Returns: (time_window_seconds, num_samples)
    """
    # Calculate time window from wave_speed ONLY (BPM window NOT used)
    # Time = Distance / Speed (scaled for 40-box grid)
    ecg_graph_width_mm = (num_boxes if num_boxes is not None else COLUMN1_BOXES) * ECG_LARGE_BOX_MM
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
    print(f"   Graph Width: {ecg_graph_width_mm:.2f}mm ({(ecg_graph_width_mm/ECG_LARGE_BOX_MM):.2f} boxes × {ECG_LARGE_BOX_MM:.2f}mm)")
    print(f"   Wave Speed: {wave_speed_mm_s}mm/s (effective {effective_wave_speed_mm_s:.2f}mm/s)")
    print(f"   Time Window: {ecg_graph_width_mm:.2f} / {effective_wave_speed_mm_s:.2f} = {calculated_time_window:.2f}s")
    print(f"   BPM: {hr_bpm} → Beats per second: {hr_bpm}/60 = {beats_per_second:.2f} beats/sec")
    print(f"   Expected Beats: {beats_per_second:.2f} × {calculated_time_window:.2f} = {expected_beats:.1f} beats")
    print(f"   Estimated Samples: {num_samples} (at 500Hz)")
    
    return calculated_time_window, num_samples

def calculate_time_window_from_width_points(wave_speed_mm_s, width_points):
    width_mm = width_points / mm
    effective_wave_speed_mm_s = wave_speed_mm_s * ECG_SPEED_SCALE
    return width_mm / max(1e-6, effective_wave_speed_mm_s)

def apply_report_ecg_filters(signal, sampling_rate, settings_manager):
    from ecg.ecg_filters import apply_ecg_filters, apply_baseline_wander_median_mean
    arr = np.asarray(signal, dtype=float)
    if arr.size < 10:
        return arr
    try:
        fs = float(sampling_rate)
    except Exception:
        fs = 500.0
    try:
        win_sec = float(str(settings_manager.get_setting("report_window_seconds", "10")).strip() or "10")
    except Exception:
        win_sec = 10.0
    window_samples = int(max(1.0, win_sec) * fs)
    if arr.size > window_samples:
        arr = arr[-window_samples:]
    pad_seconds = 4.0
    pad_samples = int(pad_seconds * fs)
    pad = min(pad_samples, max(0, arr.size - 1))
    if pad > 0:
        work = np.pad(arr, pad_width=pad, mode="reflect")
    else:
        work = arr
    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
    emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
    ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip()
    dft_param = dft_setting if dft_setting not in ("off", "") else None
    emg_param = emg_setting if emg_setting not in ("off", "") else None
    ac_param = ac_setting if ac_setting not in ("off", "") else None
    filtered = apply_ecg_filters(
        work,
        sampling_rate=fs,
        ac_filter=ac_param,
        emg_filter=emg_param,
        dft_filter=dft_param,
    )
    filtered = apply_baseline_wander_median_mean(filtered, fs)
    try:
        if filtered.size > 5:
            from scipy.ndimage import gaussian_filter1d
            filtered = gaussian_filter1d(filtered, sigma=0.8)  # Match display SMOOTH_SIGMA
    except Exception:
        pass
    # No noise-based edge trimming: a burst of activity at the edge is often a real
    # QRS, and cutting 5% of the strip on that suspicion shortened the printed
    # waveform. The recorded signal is shown end to end.
    try:
        if filtered.size > 5:
            dif = np.diff(filtered, prepend=filtered[0])
            th = 5.0 * (np.std(dif) + 1e-6)
            bad = np.where(np.abs(dif) > th)[0]
            for i in bad:
                if 1 <= i < filtered.size - 1:
                    filtered[i] = 0.5 * (filtered[i - 1] + filtered[i + 1])
    except Exception:
        pass
    if pad > 0 and filtered.size > 2 * pad:
        filtered = filtered[pad:-pad]
    # The second noise-based trim and the unconditional 3%-per-side "hard trim" are
    # gone for the same reason — together they discarded up to 16% of the recorded
    # strip, and the 3% cut applied even to perfectly clean traces.
    #
    # stabilize_report_edges() is not applied either: it cross-fades the ends into a
    # flat baseline, so a beat near the strip end printed at a fraction of its real
    # amplitude. Filter transients are already prevented by the reflect padding
    # applied around the filter chain above.
    try:
        from ecg.ecg_filters import notch_filter_butterworth
        try:
            ac_setting = str(settings_manager.get_setting("filter_ac", "50")).strip() if settings_manager else "50"
            notch_freq = float(ac_setting) if ac_setting and ac_setting not in ("off", "") else 50.0
        except Exception:
            notch_freq = 50.0
        try:
            from ecg.ecg_filters import apply_ac_filter
            filtered = apply_ac_filter(filtered, fs, str(int(notch_freq)))
        except Exception:
            try:
                filtered = notch_filter_butterworth(filtered, fs, freq=notch_freq, q=25.0)
            except Exception:
                pass
    except Exception:
        pass
    return filtered

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

def create_reportlab_ecg_drawing(lead_name, width=460, height=45):
    """
    Create ECG drawing using ReportLab (NO matplotlib - NO white background issues)
    Returns: ReportLab Drawing with guaranteed pink background
    """
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines (even lighter colors)
    light_grid_color = colors.HexColor("#ffd1d1")  # Darker minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Darker major grid
    
    # Minor grid lines (1mm spacing equivalent)
    minor_spacing_x = width / 60  # 60 divisions across width
    minor_spacing_y = height / 20  # 20 divisions across height
    
    # Vertical minor grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Horizontal minor grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Major grid lines (5mm spacing equivalent)
    major_spacing_x = width / 12  # 12 divisions across width
    major_spacing_y = height / 4   # 4 divisions across height
    
    # Vertical major grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # Horizontal major grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
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
    
    # Define lead orders based on sequence 
    LEAD_SEQUENCES = {
        "Standard": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Cabrera": ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    # Use the appropriate sequence for REPORT ONLY
    ordered_leads = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    # Map lead names to indices
    lead_to_index = {
        "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11,
        "Extra Lead II": 12
    }
    
    # FORCE REAL DATA ONLY - DISABLE DEMO MODE FOR REPORTS
    is_demo_mode = False  # Always use real data for reports
    time_window_seconds = None
    samples_per_second = samples_per_second or 150
    print(" 6_2 REPORT GENERATION: Using REAL data only (Demo mode disabled for consistency)")
    
    # Try to get ACTUAL sampling rate from the test page
    if ecg_test_page and hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler, 'sampling_rate'):
        if ecg_test_page.sampler.sampling_rate:
            samples_per_second = float(ecg_test_page.sampler.sampling_rate)
            print(f" Found sampler sampling rate: {samples_per_second}Hz")
    elif ecg_test_page and hasattr(ecg_test_page, 'sampling_rate'):
        if ecg_test_page.sampling_rate:
            samples_per_second = float(ecg_test_page.sampling_rate)
            print(f" Found page sampling rate: {samples_per_second}Hz")
    # Try to get REAL ECG data from test page
    real_ecg_data = {}
    if ecg_test_page and hasattr(ecg_test_page, 'data'):
        
        # ALWAYS USE REAL DATA - No demo mode windowing
        num_samples_to_capture = 10000
        print(f" REAL DATA MODE: Capturing up to {num_samples_to_capture} samples from live buffer")
        
        for lead in ordered_leads:
            if lead == "-aVR":
                # For -aVR, we need to invert aVR data
                if hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 3:
                    avr_data = np.array(ecg_test_page.data[3])  # aVR is at index 3
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
    
    # Apply report filters per lead using wrapper (parity with 4:3/main)
    filtered_ecg_data = {}
    for lead, signal in real_ecg_data.items():
        try:
            filtered_ecg_data[lead] = apply_report_ecg_filters(signal, samples_per_second, settings_manager)
        except Exception:
            filtered_ecg_data[lead] = signal

    # Create ReportLab drawings with REAL (filtered) data
    for lead in ordered_leads:
        try:
            # Create ReportLab drawing with REAL ECG data (with wave_gain applied)
            drawing = create_reportlab_ecg_drawing_with_real_data(
                lead, 
                filtered_ecg_data.get(lead), 
                width=460, 
                height=45,
                wave_gain_mm_mv=wave_gain_mm_mv,
                sampling_rate=samples_per_second,
                settings_manager=settings_manager
            )
            lead_drawings[lead] = drawing
            
            if lead in filtered_ecg_data:
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

# ADC-per-box configuration (shared across report sections; kept near baseline/plot logic).
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
REPORT_WAVEFORM_ADC_DIVISOR = 1083.3333333333333

def create_reportlab_ecg_drawing_with_real_data(lead_name, ecg_data, width=460, height=45, wave_gain_mm_mv=10.0, sampling_rate=500.0, settings_manager=None):
    """
    Create ECG drawing using ReportLab with REAL ECG data.
    Uses proper sample-rate-based time axis so beats are displayed at correct speed.
    """
    drawing = Drawing(width, height)
    
    # STEP 1: Create solid pink background rectangle
    bg_color = colors.HexColor("#ffe6e6")  # Light pink background
    bg_rect = Rect(0, 0, width, height, fillColor=bg_color, strokeColor=None)
    drawing.add(bg_rect)
    
    # STEP 2: Draw pink ECG grid lines
    light_grid_color = colors.HexColor("#ffd1d1")  # Light minor grid
    major_grid_color = colors.HexColor("#ffb3b3")   # Major grid
    
    # Minor grid lines (1mm spacing equivalent)
    minor_spacing_x = width / 60  # 60 divisions across width
    minor_spacing_y = height / 20  # 20 divisions across height
    
    # Vertical minor grid lines
    for i in range(61):
        x_pos = i * minor_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Horizontal minor grid lines
    for i in range(21):
        y_pos = i * minor_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=light_grid_color, strokeWidth=0.4)
        drawing.add(line)
    
    # Major grid lines (5mm spacing equivalent)
    major_spacing_x = width / 12  # 12 divisions across width
    major_spacing_y = height / 4   # 4 divisions across height
    
    # Vertical major grid lines
    for i in range(13):
        x_pos = i * major_spacing_x
        line = Line(x_pos, 0, x_pos, height, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # Horizontal major grid lines
    for i in range(5):
        y_pos = i * major_spacing_y
        line = Line(0, y_pos, width, y_pos, strokeColor=major_grid_color, strokeWidth=0.8)
        drawing.add(line)
    
    # STEP 3: Draw ECG waveform on proper time axis
    if ecg_data is None or len(ecg_data) == 0:
        print(f" No real data available for {lead_name} - showing flat line")
        return drawing
    
    print(f" Drawing ECG data for {lead_name}: {len(ecg_data)} points")
    
    fs = float(sampling_rate) if sampling_rate and float(sampling_rate) > 0 else 500.0
    
    # Get wave_speed for proper beat timing
    speed_mm_s = 25.0  # default
    if settings_manager:
        try:
            ws = settings_manager.get_setting("wave_speed")
            if ws:
                speed_mm_s = float(ws)
        except Exception:
            pass
    
    width_mm = width / mm  # Convert width points to mm
    effective_speed_mm_s = speed_mm_s * ECG_SPEED_SCALE
    max_display_seconds = width_mm / max(effective_speed_mm_s, 1e-6)
    max_samples = int(max_display_seconds * fs)
    
    ecg_array = np.asarray(ecg_data, dtype=float)
    
    # Amplitude conversion: if raw ADC values, center and scale
    med_abs = np.nanmedian(np.abs(ecg_array)) if len(ecg_array) else 0.0
    if med_abs > 20.0:  # Raw ADC
        ecg_mv = ecg_array / REPORT_WAVEFORM_ADC_DIVISOR
    else:
        ecg_mv = ecg_array  # Already in mV
    
    # Remove DC offset and baseline slope
    if len(ecg_mv) > 0:
        dc_offset = np.nanmean(ecg_mv)
        ecg_mv = ecg_mv - dc_offset
        # Skip linear detrending if the robust median-mean baseline filter (0.5 Hz) was applied,
        # because linear fitting on short asymmetric ECG segments can introduce artificial slants/drift.
        dft_val_clean = str(settings_manager.get_setting("filter_dft", "0.5")).strip() if settings_manager else "0.5"
        if dft_val_clean in ("off", ""):
            dft_val_clean = "0.5"
        if dft_val_clean != "0.5" and len(ecg_mv) > 20:
            x_idx = np.arange(len(ecg_mv))
            coeffs = np.polyfit(x_idx, ecg_mv, 1)
            ecg_mv = ecg_mv - np.polyval(coeffs, x_idx)
        
        # No head or tail ramp. Despite the old "no forced zero tail" comment, the
        # tail ramp did fade the last samples into a flat median, shrinking a QRS
        # that landed there; the head ramp scaled the opening samples toward zero.
        # Both are cosmetic and the printed strip must show the recorded amplitude.

    # Time window clip
    if len(ecg_mv) > max_samples:
        ecg_mv = ecg_mv[-max_samples:]
    
    # Build time axis
    t_sec = np.arange(len(ecg_mv)) / fs
    x_pos_mm = t_sec * effective_speed_mm_s
    
    # mV -> Y position in points
    height_mm_physical = height / mm
    y_mm = (height_mm_physical / 2.0) + ecg_mv * wave_gain_mm_mv
    y_mm = np.clip(y_mm, 0.0, height_mm_physical)
    x_pos_mm = np.clip(x_pos_mm, 0.0, width_mm)
    
    # Draw ECG line
    ecg_color = colors.HexColor("#000000")
    for i in range(len(x_pos_mm) - 1):
        x1 = x_pos_mm[i] * mm
        y1 = y_mm[i] * mm
        x2 = x_pos_mm[i+1] * mm
        y2 = y_mm[i+1] * mm
        drawing.add(Line(x1, y1, x2, y2, strokeColor=ecg_color, strokeWidth=0.5))
    
    print(f" Drew {len(ecg_mv)} ECG points for {lead_name} at fs={fs:.0f}Hz, speed={speed_mm_s}mm/s")
    return drawing

def create_clean_ecg_image(lead_name, width=6, height=2):
    """
    Create COMPLETELY CLEAN ECG image with GUARANTEED pink background
    NO labels, NO time markers, NO axes, NO white background
    """
    # FORCE matplotlib to use proper backend
    import matplotlib
    # matplotlib.use('Agg')
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
            findings = conclusions_data.get('findings', [])
            
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
        
def _draw_logo_and_footer_callback(canvas, doc_obj, patient=None, data=None, settings_manager=None):
    from reportlab.lib.units import mm
    
    # STEP 1: Draw pink ECG grid background on Page 1 (now the only landscape page)
    if canvas.getPageNumber() == 1:
        page_width, page_height = canvas._pagesize
        
        # ========== 57 BOXES IN FULL 297MM PAGE WIDTH ==========
        # Page width: 297mm (full A4 landscape)
        # Number of boxes: 57
        # Box size: 297mm / 57 = 5.2105mm per box
        num_boxes_width = 57
        page_width_mm = 297.0
        box_width_mm = page_width_mm / num_boxes_width  # 297/57 = 5.2105mm per box
        box_width_pts = box_width_mm * mm
        
        # Pink background - FULL PAGE (297mm width, no white space)
        canvas.setFillColor(colors.HexColor("#ffe6e6"))
        canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
        
        # Grid colors
        light_grid_color = colors.HexColor("#ffd1d1")
        major_grid_color = colors.HexColor("#ffb3b3")
        
        # Minor grid lines - 5 minor boxes per major box (scaled proportionally)
        # Width: 57 boxes across 297mm → 5.2105mm per box → minor = 1.042mm
        # Height: 40 boxes across 210mm → 5.25mm per box → minor = 1.05mm
        minor_spacing_mm = box_width_mm / 5.0  # 1.042mm per minor division
        minor_spacing_pts = minor_spacing_mm * mm
        
        canvas.setStrokeColor(light_grid_color)
        canvas.setLineWidth(0.6)  # Minor grid lines (1mm spacing) - keep original thickness
        # Vertical minor lines - full page width (297mm)
        x = 0
        while x <= page_width:
            canvas.line(x, 0, x, page_height)
            x += minor_spacing_pts
            if x > page_width:
                break
        
        # Horizontal minor lines - 5 minor boxes per major box
        # Use proportional spacing to match 40 major boxes across 210mm height.
        num_boxes_height = 40
        page_height_mm = 210.0
        box_height_mm = page_height_mm / num_boxes_height  # 210/40 = 5.25mm per box
        minor_spacing_y = (box_height_mm / 5.0) * mm
        y = 0
        while y <= page_height:
            canvas.line(0, y, page_width, y)
            y += minor_spacing_y
        
        # Major grid lines - exactly 57 boxes across full 297mm width
        canvas.setStrokeColor(major_grid_color)
        canvas.setLineWidth(0.6)  # Thinner major grid lines (5mm spacing) - was 1.2
        # Vertical major lines - 57 boxes (297mm width, 5.2105mm per box)
        x = 0
        for i in range(num_boxes_width + 1):  # 58 lines for 57 boxes
            canvas.line(x, 0, x, page_height)
            x += box_width_pts
        
        # Horizontal major lines - 40 boxes (210mm height, 5.25mm per box)
        box_height_pts = box_height_mm * mm
        y = 0
        for i in range(num_boxes_height + 1):  # 41 lines for 40 boxes
            canvas.line(0, y, page_width, y)
            y += box_height_pts
    
    # STEP 1.5: Draw Org. and Phone No. on Page 1 (REPOSITIONED - slightly higher, more left)
    if canvas.getPageNumber() == 1:
        canvas.saveState()
        # Portrait A4 height = 842 points, position very close to top
        page_height = 842  # A4 portrait height
        x_pos = 15  # More to the left (was 30, now 15)
        y_pos = page_height - 30
        
        canvas.setFont("Helvetica-Bold", 10)
        canvas.setFillColor(colors.black)
        org_label = "Org:"
        canvas.drawString(x_pos, y_pos, org_label)
        
        org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 10)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x_pos + org_label_width + 5, y_pos, patient.get("Org.", "") if patient else "")
        
        y_pos -= 15

        canvas.setFont("Helvetica-Bold", 10)
        address_label = "Address:"
        canvas.drawString(x_pos, y_pos, address_label)

        address_label_width = canvas.stringWidth(address_label, "Helvetica-Bold", 10)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(
            x_pos + address_label_width + 5,
            y_pos,
            (patient.get("org_address", "") or patient.get("Org. Address", "")) if patient else "",
        )

        y_pos -= 15
        
        canvas.setFont("Helvetica-Bold", 10)
        phone_label = "Phone No:"
        canvas.drawString(x_pos, y_pos, phone_label)
        
        phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 10)
        canvas.setFont("Helvetica", 10)
        canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient.get("doctor_mobile", "") if patient else "")
        canvas.restoreState()
    
    # STEP 2: Draw logo (REPOSITIONED - lower from top)
    # Use resource_path helper for PyInstaller compatibility
    png_path = _get_resource_path("assets/Deckmountimg.png")
    webp_path = _get_resource_path("assets/Deckmount.webp")
    logo_path = png_path if os.path.exists(png_path) else webp_path
    
    if os.path.exists(logo_path):
        canvas.saveState()
        if canvas.getPageNumber() == 1:
            # Page 1 is now LANDSCAPE - logo at top right
            logo_w, logo_h = 120, 40
            page_width, page_height = canvas._pagesize
            x = page_width - logo_w - 35
            y = page_height - logo_h
        else:
            # Portrait page (not used anymore)
            logo_w, logo_h = 120, 40
            page_height = 842  # A4 portrait height
            x = 595 - logo_w - 30  # 595 = A4 width, 30 = right margin
            y = page_height - 35  # 35 points from top
        try:
            canvas.drawImage(logo_path, x, y, width=logo_w, height=logo_h, preserveAspectRatio=True, mask="auto")
        except Exception:
            pass
        canvas.restoreState()
    
    # STEP 2.5: Add Date and Time labels below logo on Page 1
    if canvas.getPageNumber() == 1:
        from datetime import datetime
        canvas.saveState()
        canvas.setFont("Helvetica", 9)
        canvas.setFillColor(colors.black)
        
        # Get current date and time
        now = datetime.now()
        date_str = now.strftime("%d/%m/%Y")
        time_str = now.strftime("%H:%M:%S")
        
        # Position below logo (logo ends at y, so date at y-15, time at y-30)
        # Page 1 is now LANDSCAPE - use same x position as logo (right aligned)
        page_width, page_height = canvas._pagesize
        logo_w = 120
        date_x = page_width - logo_w - 35  # Same x as logo
        time_x = page_width - logo_w - 35  # Same x as logo
        date_y = page_height - 40  # Right below logo (shifted up 5 more points from -45)
        time_y = page_height - 55  # 15 points below logo (shifted up 5 more points from -60)
        
        # Draw date label
        canvas.drawString(date_x, date_y, f"Date: {date_str}")
        # Draw time label  
        canvas.drawString(time_x, time_y, f"Time: {time_str}")
        
        # STEP 2.6: Add Org and Phone No. labels below date/time on Page 2
        canvas.setFont("Helvetica-Bold", 9)
        canvas.setFillColor(colors.black)
        
        # Org label (15 points below time)
        org_y = page_height - 70  # 15 points below time (time_y = -55, so -55 - 15 = -70)
        org_label = "Org:"
        canvas.drawString(date_x, org_y, org_label)
        
        org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 9)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(date_x + org_label_width + 5, org_y, patient.get("Org.", "") if patient else "")
        
        # Phone No. label (15 points below org)
        phone_y = page_height - 85  # 15 points below org (org_y = -70, so -70 - 15 = -85)
        canvas.setFont("Helvetica-Bold", 9)
        phone_label = "Phone No:"
        canvas.drawString(date_x, phone_y, phone_label)
        
        phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 9)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(date_x + phone_label_width + 5, phone_y, patient.get("doctor_mobile", "") if patient else "")
        
        canvas.restoreState()
    
    # STEP 3: Footer
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.black)
    
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
        
    text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
    
    if canvas.getPageNumber() == 1:
        # Page 2 is LANDSCAPE
        page_width, page_height = canvas._pagesize
        x = (page_width - text_width) / 2
    else:
        # Page 1 is PORTRAIT
        x = (595 - text_width) / 2
    
    y = 5  # Shifted down 5 points from 10 to 5
    canvas.drawString(x, y, footer_text)
    canvas.restoreState()

    return None



def _load_signup_details_for_username(username):
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
            val = raw.get(str(username), {})
            return val if isinstance(val, dict) else {}
        return {}
    except Exception:
        return {}


def _collect_12_lead_payload(ecg_test_page, sampling_rate, ecg_data_file=None, window_seconds=10.0):
    try:
        fs = float(sampling_rate) if sampling_rate else 500.0
    except Exception:
        fs = 500.0
    n = max(1, int(round(window_seconds * fs)))
    lead_names = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

    if ecg_data_file and os.path.exists(ecg_data_file):
        try:
            with open(ecg_data_file, 'r', encoding='utf-8') as f:
                saved = json.load(f)
            leads = saved.get('leads') if isinstance(saved, dict) else None
            if isinstance(leads, dict):
                out = {}
                for ln in lead_names:
                    arr = leads.get(ln) or leads.get(f'Lead_{ln}') or []
                    if isinstance(arr, list):
                        out[ln] = arr[-n:] if len(arr) > n else arr
                if out:
                    return out, fs
        except Exception:
            pass

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


def _sync_report_package_to_backend(filename, patient, data, metrics_payload, username, ecg_test_page, sampling_rate, ecg_data_file=None):
    try:
        from utils.backend_api import get_backend_api
        backend = get_backend_api()
        if not backend.is_enabled():
            print('  Backend sync disabled')
            return

        signup_details = _load_signup_details_for_username(username)
        leads_payload, fs = _collect_12_lead_payload(ecg_test_page, sampling_rate, ecg_data_file=ecg_data_file)

        device_serial = str((patient or {}).get('serial_number') or data.get('machine_serial') or signup_details.get('serial_id') or 'UNKNOWN')
        sid = backend.start_session(device_serial=device_serial, device_info={
            'machine_serial': data.get('machine_serial', ''),
            'app': 'cardiox',
            'report_type': '12_lead_ecg_6_2',
            'report_file': os.path.abspath(filename),
        })
        print(f'  Backend session started: {sid}')

        mres = backend.upload_metrics({
            'report_generated_at': datetime.now().isoformat(),
            'username': username or '',
            'patient': patient or {},
            'signup': signup_details,
            'metrics': metrics_payload or {},
        })
        print(f"  Backend metrics sync: {mres.get('status')}")

        if leads_payload:
            wres = backend.upload_waveform(leads_payload, int(round(fs)))
            print(f"  Backend waveform sync: {wres.get('status')} ({len(leads_payload)} leads)")
        else:
            print('  Backend waveform sync skipped: no lead data available')

        rres = backend.upload_report(filename, metadata={
            'username': username or '',
            'patient': patient or {},
            'signup': signup_details,
            'metrics': metrics_payload or {},
            'ecg_details': {
                'sampling_rate': fs,
                'lead_count': len(leads_payload),
                'ecg_data_file': os.path.abspath(ecg_data_file) if ecg_data_file else '',
            }
        })
        print(f"  Backend report sync: {rres.get('status')}")

        eres = backend.end_session({
            'status': 'report_generated',
            'report_file': os.path.abspath(filename),
            'lead_count': len(leads_payload),
            'username': username or '',
        })
        print(f"  Backend session end: {eres.get('status')}")

    except Exception as be:
        print(f"  Backend sync error: {be}")

def generate_6_2_ecg_report(filename="ecg_report.pdf", data=None, lead_images=None, dashboard_instance=None, ecg_test_page=None, patient=None, ecg_data_file=None):
    """
    Generate ECG report PDF
    """
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
            "QRS_axis": "--",
        }

    # Define base_dir and reports_dir for file operations
    reports_dir = str(data_file("reports"))
    os.makedirs(reports_dir, exist_ok=True)

    from utils.settings_manager import SettingsManager
    settings_manager = SettingsManager()

    def _safe_float(value, default):
        try:
            return float(value)
        except Exception:
            return default

    def _safe_int(value, default=0):
        try:
            return int(float(value))
        except Exception:
            return default

    # ==================== STEP 1: Get HR_bpm from metrics.json (PRIORITY) ====================
    # Priority: metrics.json  latest HR_bpm  (calculation-based beats  )
    latest_metrics = load_latest_metrics_entry(reports_dir)
    hr_bpm_value = 0
    
    # Priority 1: metrics.json  latest HR_bpm (CALCULATION-BASED BEATS   REQUIRED)
    if latest_metrics:
        hr_bpm_value = _safe_int(latest_metrics.get("HR_bpm"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from metrics.json: {hr_bpm_value} bpm (for calculation-based beats)")
    
    # Priority 2: Fallback to data parameter
    if hr_bpm_value == 0:
        hr_candidate = data.get("HR_bpm") or data.get("Heart_Rate") or data.get("HR")
        hr_bpm_value = _safe_int(hr_candidate)
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from data parameter: {hr_bpm_value} bpm")
    
    # Priority 3: Fallback to HR_avg
    if hr_bpm_value == 0 and data.get("HR_avg"):
        hr_bpm_value = _safe_int(data.get("HR_avg"))
        if hr_bpm_value > 0:
            print(f" Using HR_bpm from HR_avg: {hr_bpm_value} bpm")

    data["HR_bpm"] = hr_bpm_value
    data["Heart_Rate"] = hr_bpm_value
    data["HR"] = hr_bpm_value
    if hr_bpm_value > 0:
        data["RR_ms"] = int(60000 / hr_bpm_value)
    else:
        data["RR_ms"] = data.get("RR_ms", 0)

    # ==================== STEP 2: Get wave_speed from ecg_settings.json (PRIORITY) ====================
    # Priority: ecg_settings.json  wave_speed  (calculation-based beats  )
    wave_speed_setting = settings_manager.get_setting("wave_speed", "25")
    wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
    wave_speed_mm_s = _safe_float(wave_speed_setting, 25.0)  # Default: 25.0 mm/s
    wave_gain_mm_mv = _safe_float(wave_gain_setting, 10.0)   # Default: 10.0 mm/mV
    print(f" Using wave_speed from ecg_settings.json: {wave_speed_mm_s} mm/s (for calculation-based beats)")
    computed_sampling_rate = 500

    data["wave_speed_mm_s"] = wave_speed_mm_s
    data["wave_gain_mm_mv"] = wave_gain_mm_mv

    print(f" Pre-plot checks: HR_bpm={hr_bpm_value}, RR_ms={data['RR_ms']}, wave_speed={wave_speed_mm_s}mm/s, wave_gain={wave_gain_mm_mv}mm/mV, sampling_rate={computed_sampling_rate}Hz")
    print(f" Calculation-based beats formula:")
    width_mm = COLUMN1_BOXES * ECG_LARGE_BOX_MM
    print(f"   Graph width: {COLUMN1_BOXES} boxes × {ECG_LARGE_BOX_MM}mm = {width_mm:.2f}mm")
    print(f"   BPM window: (desired_beats × 60) / {hr_bpm_value} = {(6 * 60.0 / hr_bpm_value) if hr_bpm_value > 0 else 0:.2f}s")
    print(f"   Wave speed window: {width_mm:.2f}mm / {wave_speed_mm_s}mm/s = {width_mm / wave_speed_mm_s:.2f}s")
    
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
            computed_sampling_rate = 500
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
                computed_sampling_rate = 500
                print(f" Using sampling rate from saved file: {computed_sampling_rate} Hz")
            else:
                print(" Warning: Could not load saved ECG data file")
        else:
            print(" Warning: Could not save ECG data to file")
    
    if not saved_ecg_data:
        print(" Warning: No saved ECG data available - beats will not be calculation-based")

    # Get conclusions from dashboard/JSON
    dashboard_conclusions = get_dashboard_conclusions_from_image(dashboard_instance)

    # SAFEGUARD: If there is no real data (all core metrics are zero), ignore any
    # persisted conclusions and use the explicit "no data" conclusions instead.
    try:
        core_keys = ["HR", "PR", "QRS", "QT", "QTc", "ST"]
        all_zero = True
        for k in core_keys:
            v = data.get(k, 0)
            try:
                all_zero = all_zero and (float(v) == 0.0)
            except Exception:
                all_zero = all_zero and (str(v).strip() in ["0", "--", "", "None"])
        if all_zero:
            dashboard_conclusions = [
                " No ECG data available",
                "Please connect device",
           
                
                
                
                

                

                


               

                
            ]
            print(" Overriding conclusions because all core metrics are zero (no data)")
    except Exception:
        pass

    # FILTER: Remove empty conclusions and "---" placeholders - ONLY SHOW REAL CONCLUSIONS
    # MAXIMUM 12 CONCLUSIONS (because only 12 boxes available)
    filtered_conclusions = []
    for conclusion in dashboard_conclusions:
        # Keep only non-empty conclusions that are not "---"
        if conclusion and conclusion.strip() and conclusion.strip() != "---":
            filtered_conclusions.append(conclusion.strip())
            # LIMIT: Maximum 12 conclusions (only 12 boxes available)
            if len(filtered_conclusions) >= 12:
                break
    
    print(f"\n Original conclusions: {len(dashboard_conclusions)}")
    print(f" Filtered conclusions (removed empty/---): {len(filtered_conclusions)}")
    print(f" Final conclusions to show (MAX 12): {filtered_conclusions}\n")

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
                del fig
                
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
    
    # Define lead orders based on sequence
    LEAD_SEQUENCES = {
        "Standard": ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"],
        "Cabrera": ["aVL", "I", "-aVR", "II", "aVF", "III", "V1", "V2", "V3", "V4", "V5", "V6"]
    }
    
    # Use the appropriate sequence for REPORT ONLY
    lead_order = LEAD_SEQUENCES.get(lead_sequence, LEAD_SEQUENCES["Standard"])
    
    print(f" Using lead sequence for REPORT: {lead_sequence}")
    print(f" Lead order for REPORT: {lead_order}")

    # Create BaseDocTemplate for mixed page orientations (from hyperkalemia)
    doc = BaseDocTemplate(filename, pagesize=A4,
                         rightMargin=30, leftMargin=30,
                         topMargin=30, bottomMargin=30)
    # Create wrapper function for callback with patient parameter
    def callback_wrapper(canvas, doc):
        return _draw_logo_and_footer_callback(canvas, doc, patient, data, settings_manager)


    # Define Landscape template as DEFAULT (only template) with onPage callback
    landscape_width, landscape_height = landscape(A4)
    # Reduce margins to increase frame size (from 30 to 20 on each side)
    landscape_frame = Frame(20, 20,  # reduced margins for landscape to fit taller drawing
                           landscape_width - 40, landscape_height - 40,
                           id="landscape_frame")
    landscape_template = PageTemplate(id="landscape", frames=[landscape_frame],
                                     pagesize=landscape(A4), onPage=callback_wrapper)

    # Add ONLY landscape template as default
    doc.addPageTemplates([landscape_template])

    story = []
    styles = getSampleStyleSheet()
    
    # Skip title and other Page 1 content - go directly to landscape ECG content

    # Skip all Page 1 content - go directly to landscape ECG graphs
    # Patient details will be shown in the landscape page master drawing

    # Patient details for landscape page (will be used in master drawing)
    if patient is None:
        patient = {}
    first_name = patient.get("first_name", "")
    last_name = patient.get("last_name", "")
    full_name = f"{first_name} {last_name}".strip()
    age = patient.get("age", "")
    gender = patient.get("gender", "")
    date_time_str = patient.get("date_time", "")

    # REMOVED: Date/Time table from story - will be added in master drawing instead
    # Patient info and vital parameters are now in master drawing above ECG graph
    # No extra spacing needed as they're positioned in drawing coordinates

    
    # Vital Parameters Header (completely transparent)
    vital_style = ParagraphStyle(
        'VitalStyle',
        fontSize=12,  # Increased from 11
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=15,
        alignment=1,  # center
        # Add white background 
        # for better visibility on pink grid
        backColor=colors.white,
    )

    # Vital Parameters Header (on top of background)
    vital_style = ParagraphStyle(


            
        'VitalStyle',
        fontSize=11,
        fontName='Helvetica-Bold',
        textColor=colors.black,
        spaceAfter=15,
        alignment=1,  # center
        
    )

    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT',)
    QTc = data.get('QTc',)
    QTcF = data.get('QTc_Fridericia') or data.get('QTcF') or 0
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   

    # Create table data: 4 rows × 2 columns
    vital_table_data = [
        [f"HR : {int(round(HR))} bpm", f"QT: {int(round(QT))} ms"],
        [f"PR : {int(round(PR))} ms", f"QTc: {int(round(QTc))} ms"],
        [f"QRS: {int(round(QRS))} ms", f"ST: {int(round(ST))} ms"],
        [f"RR : {int(round(RR))} ms", f"QTcF: {QTcF/1000.0:.3f} s"]  
    ]

    # Create vital parameters table with MORE LEFT and TOP positioning
    vital_params_table = Table(vital_table_data, colWidths=[100, 100])  # Even smaller widths for more left

    vital_params_table.setStyle(TableStyle([
        # Transparent background to show pink grid
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0, 0, 0, alpha=0)),
        ("GRID", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("BOX", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # Left align
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),  # Normal font
        ("FONTSIZE", (0, 0), (-1, -1), 10),  # Same size as Name, Age, Gender
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),  # Zero left padding for extreme left
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),  # Zero right padding too
        ("TOPPADDING", (0, 0), (-1, -1), 0),   # Zero top padding for top shift
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

   

    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - ADJUSTED HEIGHT to fit within page frame (max ~770)
    total_width = landscape(A4)[0] - 40  # Exact landscape frame width (A4 - margins) to prevent page overflow
    total_height = 540  # Reduced to 720 to fit within page frame (max ~770) with margin
    
    # Create ONE master drawing
    master_drawing = TransparentDrawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads in VERTICAL COLUMNS
    # Column 1: vertical stack from top to bottom, Column 2: vertical stack from top to bottom
    lead_positions = []
    
    # Define which leads go in which column
    column1_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]  # First 7 leads including Extra Lead II
    column2_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]     # Last 6 leads - vertical column
    
    # QTc label is at Y=455, so start leads 60 points below that
    qtc_y_position = 455
    start_y = qtc_y_position - 60 + 17  # 412 (60 points below QTc + 17 points up shift)
    
    for i, lead in enumerate(lead_order):
        # Determine X position based on which column this lead belongs to
        if lead in column1_leads:
            # Column 1: Vertical stack - all at same X, different Y
            lead_index = column1_leads.index(lead)
            # Shift ALL column1 leads 45 points to the left
            x_pos = 20 - 45  # -25 (shifted left 45 points)
            y_pos = start_y - (lead_index * 60)  # GO DOWNWARD: 395, 335, 275, 215, 155, 95
        else:
            # Column 2: Vertical stack - all at same X, different Y
            lead_index = column2_leads.index(lead)
            x_pos = 400  # ALL column2 leads at same X position (center/right side)
            y_pos = start_y - (lead_index * 60)  # GO DOWNWARD: 395, 335, 275, 215, 155, 95
        
        lead_positions.append({
            "lead": lead, 
            "x": x_pos, 
            "y": y_pos
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
                    # NEW LOGIC: Time window = graph_width_mm / effective_wave_speed (column boxes × ECG_LARGE_BOX_MM)
                    ecg_graph_width_mm = COLUMN1_BOXES * ECG_LARGE_BOX_MM
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
        #   - Time window = (strip width mm) / wave_speed ONLY
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,      # 90 bpm
            wave_speed_mm_s,   # 25 mm/s (current)
            desired_beats=15,
            num_boxes=COLUMN1_BOXES
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
        
        # Determine per-lead sample window based on wave speed (NEW LOGIC)
        # 12.5 mm/s -> Full samples (2500/5200) to plot half width (~13/26 boxes)
        # 25 mm/s -> Full samples (2500/5200) to plot full width (~26/52 boxes)
        # 50 mm/s -> Half samples (1250/2600) to fit in full width (~26/52 boxes)
        if abs(wave_speed_mm_s - 50.0) < 0.1:
            # 50 mm/s -> Use half samples to stay within full 26/52 boxes space
            current_num_samples = SIX_TWO_SAMPLES_COLUMN // 2  # 1250 samples -> 25 boxes at 50mm/s
            extra_ii_samples = SIX_TWO_SAMPLES_EXTRA_II // 2   # 2600 samples -> 52 boxes at 50mm/s
            print(f" Wave speed 50.0 detected: Using half-samples ({current_num_samples}/{extra_ii_samples})")
        else:
            # 25 or 12.5 mm/s -> Standard full samples
            # At 12.5: 2500 samples = 5s = 12.5 boxes (half-width)
            # At 25: 2500 samples = 5s = 25 boxes (full-width)
            current_num_samples = SIX_TWO_SAMPLES_COLUMN       # 2500 samples
            extra_ii_samples = SIX_TWO_SAMPLES_EXTRA_II        # 5200 samples
            print(f" Wave speed {wave_speed_mm_s} detected: Using full-samples ({current_num_samples}/{extra_ii_samples})")

        print(f" DEBUG: Processing Lead {lead} at position ({x_pos}, {y_pos})")
        print(f" DEBUG: Drawing dimensions: width={total_width}, height={total_height}")
        
        try:
            # STEP 3A: Add lead labels 15 points above ECG graph strips
            from reportlab.graphics.shapes import String
            
            # Calculate label position - shift 45 points right for column 1 leads, then 10 points left
            if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                label_x_pos = x_pos + 45.0 - 10.0  # Shift 45 points right, then 10 points left
            else:
                label_x_pos = x_pos  # Keep original position for other leads
                
            lead_label = String(label_x_pos, y_pos + 52, f"{lead}",  # Y+52 (35+17) - labels 17 points higher
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # Add vertical dotted line to separate columns - positioned to prevent column 1 data overlap
            if lead in column2_leads:  # Draw for ALL V leads to debug
                print(f" DEBUG: Drawing dotted line for {lead} at X=397.5, Y=430 to 105")
                # Draw vertical dotted line from V1 to V6, positioned between columns
                from reportlab.graphics.shapes import Line
                from reportlab.lib.units import mm
                
                # Position line 14.5 boxes (217.5 points) to the right
                # Original X=180, now X=180 + 217.5 = 397.5
                line_x = 397.5  # Shifted right 14.5 boxes from 180
                # Reduce line length by 2 boxes (30 points) from bottom
                line_y_start = 447  # 430 + 17 = 447 (shifted 17 points up)
                line_y_end = 122     # 105 + 17 = 122 (shifted 17 points up)
                
                # Create dotted line effect with multiple short lines
                dotted_line_spacing = 3  # points between dots
                dot_length = 2  # length of each dot
                
                current_y = line_y_start
                dots_count = 0
                while current_y > line_y_end:
                    # Draw small line segments to create dotted effect
                    segment_end = max(current_y - dot_length, line_y_end)
                    dotted_line = Line(line_x, current_y, line_x, segment_end,
                                    strokeColor=colors.black, strokeWidth=0.5)
                    master_drawing.add(dotted_line)
                    dots_count += 1
                    current_y = segment_end - dotted_line_spacing
                
                print(f" DEBUG: Added {dots_count} dotted segments to vertical line (reduced by 2 boxes from bottom)")
            
            real_data_available = False
            real_ecg_data = None
            
            if lead == "II":
                target_samples = SIX_TWO_SAMPLES_EXTRA_II
            else:
                target_samples = SIX_TWO_SAMPLES_COLUMN
            lead_samples_to_capture = target_samples
            lead_time_window_seconds = lead_samples_to_capture / max(1e-6, computed_sampling_rate)
            
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
            # Use live dashboard data only
            # Check if live data has MORE samples than saved data
            if ecg_test_page and hasattr(ecg_test_page, 'data'):
                lead_to_index = {
                    "I": 0, "II": 1, "III": 2, "aVR": 3, "aVL": 4, "aVF": 5,
        "V6": 11, "Extra Lead II": 12
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
                                # Always use live dashboard data (ignore any saved data)
                                raw_data = calculated_data
                                if len(raw_data) >= lead_samples_to_capture:
                                    raw_data = raw_data[-lead_samples_to_capture:]
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
                        if len(raw_data) >= lead_samples_to_capture:
                            raw_data = raw_data[-lead_samples_to_capture:]
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
                        else:
                            pass
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                            # Get filtered real data for this lead
                            lead_index = lead_to_index[lead]
                            if len(ecg_test_page.data[lead_index]) > 0:
                                raw_data = ecg_test_page.data[lead_index]
                                # Check if we have enough samples, otherwise use all available
                                if len(raw_data) >= lead_samples_to_capture:
                                    raw_data = raw_data[-lead_samples_to_capture:]
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
                            else:
                                pass
            
            if real_data_available and len(real_ecg_data) > 0:
                # Draw ALL REAL ECG data - NO LIMITS
                # Adjust ECG width based on column to ensure column 1 stops before dotted line
                if lead in column1_leads:
                    # Column 1: Graph should stop exactly at dotted line at X=397.5
                    # Lead starts at adjusted_x_pos (x_pos + 30), so calculate exact width
                    if lead == "II":
                        ecg_width = COLUMN1_BOXES * ECG_LARGE_BOX_MM * mm  # 52.6 boxes × 5.25mm = 276.15mm
                    elif lead in column1_leads:
                        ecg_width = COLUMN1_BOXES * ECG_LARGE_BOX_MM * mm  # 26.3 boxes × 5.25mm = 138.075mm
                    else:
                        ecg_width = COLUMN2_BOXES * ECG_LARGE_BOX_MM * mm  # 26.3 boxes × 5.25mm = 138.075mm
                ecg_height = 45
                
                # Create time array for ALL data with adjusted width and position
                # For column 1 leads, shift start position 2 boxes (30 points) to the right
                if lead in column1_leads:
                    # Shift column 1 leads 2 boxes (30 points) to the right
                    adjusted_x_pos = x_pos + 30  # Move 30 points right
                else:
                    # Column 2 leads keep original position
                    adjusted_x_pos = x_pos
                
                # NEW LOGIC: Use time * speed * scale to align with 5.25mm boxes
                # 25mm/s speed, scaled by ECG_SPEED_SCALE (1.05) to match 5.25mm boxes
                # result: 1 second = 5 boxes = 26.25mm (exactly aligned with grid)
                # Create time array for ALL data (same approach as ecg_report_generator.py)
                # Slices to exactly fit the visible width, cropping extra samples instead of stretching
                visible_seconds = ecg_width / (25.0 * mm)
                visible_samples = int(round(visible_seconds * computed_sampling_rate))
                if len(adc_data) > visible_samples:
                    adc_data = adc_data[-visible_samples:]
                
                t_sec = np.arange(len(adc_data)) / computed_sampling_rate
                t = adjusted_x_pos + t_sec * 25.0 * mm
                
                adc_data = np.array(real_ecg_data, dtype=float)

                # Apply SAME filters as dashboard (AC notch → EMG → DFT → Gaussian)
                try:
                    from ecg.ecg_filters import apply_ecg_filters
                    from scipy.ndimage import gaussian_filter1d as _gf1d
                    
                    ac_setting  = str(settings_manager.get_setting("filter_ac",  "50")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
                    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
                    
                    # Nyquist guard: AC notch at F Hz requires sampling rate > 2*F Hz
                    if ac_setting in ("50", "60"):
                        required_fs = float(ac_setting) * 2.0 + 1.0
                        if float(computed_sampling_rate) <= required_fs:
                            print(f" AC filter disabled (rate {computed_sampling_rate} Hz too low for {ac_setting} Hz notch)")
                            ac_setting = "off"
                    
                    adc_data = apply_ecg_filters(
                        signal=adc_data,
                        sampling_rate=float(computed_sampling_rate),
                        ac_filter=ac_setting  if ac_setting  not in ("off", "") else None,
                        emg_filter=emg_setting if emg_setting not in ("off", "") else None,
                        dft_filter=dft_setting if dft_setting not in ("off", "") else None,
                    )
                    
                    # Light Gaussian smoothing (sigma=0.8 — same as dashboard)
                    if len(adc_data) > 5:
                        adc_data = _gf1d(adc_data, sigma=0.8)
                    
                    print(f" Applied dashboard filters: AC={ac_setting}, EMG={emg_setting}, DFT={dft_setting} for {lead}")
                except Exception as filter_err:
                    print(f" Dashboard filter apply failed for {lead}: {filter_err}")

                # Remove filter transients at edges (same as dashboard approach)
                # Trim 50 samples from start and end to remove filter artifacts
                trim_samples = 50
                logging.info(f"=== LEAD {lead} TRIMMING ANALYSIS ===")
                logging.info(f"Before trimming - adc_data length: {len(adc_data)}, t length: {len(t)}")
                logging.info(f"adc_data first 5 values: {adc_data[:5]}")
                logging.info(f"adc_data last 5 values: {adc_data[-5:]}")
                logging.info(f"t first 5 values: {t[:5]}")
                logging.info(f"t last 5 values: {t[-5:]}")
                
                if len(adc_data) > (trim_samples * 2):
                    logging.info(f"Proceeding with trimming {trim_samples} samples")
                    adc_data = adc_data[trim_samples:-trim_samples]
                    # Also trim time array to match data length (exact same approach as ecg_report_generator.py)
                    t = t[trim_samples:-trim_samples]
                    
                    logging.info(f"After trimming - adc_data length: {len(adc_data)}, t length: {len(t)}")
                    logging.info(f"adc_data first 5 values: {adc_data[:5]}")
                    logging.info(f"adc_data last 5 values: {adc_data[-5:]}")
                    logging.info(f"t first 5 values: {t[:5]}")
                    logging.info(f"t last 5 values: {t[-5:]}")
                    logging.info(f"Trimmed {trim_samples} samples from start and end for {lead}")
                else:
                    logging.warning(f"Not enough data to trim for {lead}, using full data")
                    logging.warning(f"Data length: {len(adc_data)}, Required: {trim_samples * 2}")
                
                logging.info(f"=== END LEAD {lead} TRIMMING ANALYSIS ===")
                
                # DEBUG: Check if data is already processed (baseline-subtracted)
                # If data range is far from 2000 baseline, it might already be processed
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                
                # Step 2: Apply baseline 2000 (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads (III, aVR, aVL, aVF), data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = 2000.0
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
                    centered_adc = adc_data - baseline_adc
                elif is_calculated_lead:
                    # For calculated leads, data is already centered from calculation (II - I, etc.)
                    # The calculated value is already the difference, so it's centered around 0
                    # We use it directly without baseline subtraction
                    centered_adc = adc_data  # Use data as-is (already centered from calculation)
                else:  # Data is already processed (baseline-subtracted or filtered)
                    centered_adc = adc_data  # Use data as-is (already centered)
                
                # Step 3: Calculate ADC per box based on wave_gain and lead-specific multiplier
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
                # Get lead-specific ADC per box multiplier (default: 6400)
                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
                # Formula: ADC_per_box = adc_per_box_multiplier / wave_gain_mm_mv
                # IMPORTANT: Each lead can have different ADC per box multiplier
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                # This means: 640 ADC offset = 1 box (5mm) vertical movement
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # DEBUG: Log actual ADC values for troubleshooting
                max_centered_adc = np.max(np.abs(centered_adc))
                min_centered_adc = np.min(centered_adc)
                max_centered_adc_abs = np.max(np.abs(centered_adc))
                expected_boxes = max_centered_adc_abs / adc_per_box
                
                # Step 4: Convert ADC offset to boxes (vertical units)
                # Direct calculation: boxes_offset = centered_adc / adc_per_box
                # Example: 2000 ADC offset / 750 ADC per box = 2.6666 boxes
                # BUT: If actual ADC values are smaller (e.g., 375 ADC), then:
                # 375 ADC / 750 ADC per box = 0.5 boxes (which matches what user sees!)
                boxes_offset = centered_adc / adc_per_box
                
                # Log boxes offset for verification
                
                # Step 5: Convert boxes to Y position
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                # IMPORTANT: Standard ECG paper uses 5mm per box
                # 5mm = 5 * 2.834645669 points = 14.17 points per box
                from reportlab.lib.units import mm
                box_height_points = 5.0 * mm  # Standard ECG: 5mm = 14.17 points per box
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 4: Add Patient Info, Date/Time and Vital Parameters to master drawing
    # POSITIONED ABOVE ECG GRAPH (not mixed inside graph)
    from reportlab.graphics.shapes import String

    # LEFT SIDE: Patient Info (TOP LEFT CORNER - moved up by 1 box and left by 1 box) - SHIFTED LEFT 10 points
    patient_name_label = String(-45, 555, f"Name: {full_name}",  # Moved left 10 points from -25 to -35
                           fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_name_label)

    patient_age_label = String(-45, 535, f"Age: {age}",  # Moved left 10 points from -25 to -35
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_age_label)

    patient_gender_label = String(-45, 515, f"Gender: {gender}",  # Moved left 10 points from -25 to -35
                             fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_gender_label)

    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT',)
    QTc = data.get('QTc',)
    QTcF = data.get('QTc_Fridericia') or data.get('QTcF') or 0
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   

    # Create table data: 4 rows × 2 columns
    vital_table_data = [
        [f"HR : {int(round(HR))} bpm", f"QT: {int(round(QT))} ms"],
        [f"PR : {int(round(PR))} ms", f"QTc: {int(round(QTc))} ms"],
        [f"QRS: {int(round(QRS))} ms", f"ST: {int(round(ST))} ms"],
        [f"RR : {int(round(RR))} ms", f"QTcF: {QTcF/1000.0:.3f} s"]  
    ]

    # Create vital parameters table with MORE LEFT and TOP positioning
    vital_params_table = Table(vital_table_data, colWidths=[100, 100])  # Even smaller widths for more left

    vital_params_table.setStyle(TableStyle([
        # Transparent background to show pink grid
        ("BACKGROUND", (0, 0), (-1, -1), colors.Color(0, 0, 0, alpha=0)),
        ("GRID", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("BOX", (0, 0), (-1, -1), 0, colors.Color(0, 0, 0, alpha=0)),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),  # Left align
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),  # Normal font
        ("FONTSIZE", (0, 0), (-1, -1), 10),  # Same size as Name, Age, Gender
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),  # Zero left padding for extreme left
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),  # Zero right padding too
        ("TOPPADDING", (0, 0), (-1, -1), 0),   # Zero top padding for top shift
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))

   

    #  CREATE SINGLE MASSIVE DRAWING with ALL ECG content (NO individual drawings)
    print("Creating SINGLE drawing with all ECG content...")
    
    # Single drawing dimensions - ADJUSTED HEIGHT to fit within page frame (max ~770)
    total_width = landscape(A4)[0] - 40  # Exact landscape frame width (A4 - margins) to prevent page overflow
    total_height = 540  # Reduced to 720 to fit within page frame (max ~770) with margin
    
    # Create ONE master drawing
    master_drawing = TransparentDrawing(total_width, total_height)
    
    # STEP 1: NO background rectangle - let page pink grid show through
    
    # STEP 2: Define positions for all 12 leads in VERTICAL COLUMNS
    # Column 1: vertical stack from top to bottom, Column 2: vertical stack from top to bottom
    lead_positions = []
    
    # Define which leads go in which column
    column1_leads = ["I", "II", "III", "aVR", "aVL", "aVF"]  # First 7 leads including Extra Lead II
    column2_leads = ["V1", "V2", "V3", "V4", "V5", "V6"]     # Last 6 leads - vertical column
    
    # QTc label is at Y=455, so start leads 60 points below that
    qtc_y_position = 455
    start_y = qtc_y_position - 60 + 17  # 412 (60 points below QTc + 17 points up shift)
    
    for i, lead in enumerate(lead_order):
        # Determine X position based on which column this lead belongs to
        if lead in column1_leads:
            # Column 1: Vertical stack - all at same X, different Y
            lead_index = column1_leads.index(lead)
            # Shift ALL column1 leads 45 points to the left
            x_pos = 20 - 45  # -25 (shifted left 45 points)
            y_pos = start_y - (lead_index * 60)  # GO DOWNWARD: 395, 335, 275, 215, 155, 95
        else:
            # Column 2: Vertical stack - all at same X, different Y
            lead_index = column2_leads.index(lead)
            x_pos = 400  # ALL column2 leads at same X position (center/right side)
            y_pos = start_y - (lead_index * 60)  # GO DOWNWARD: 395, 335, 275, 215, 155, 95
        
        lead_positions.append({
            "lead": lead, 
            "x": x_pos, 
            "y": y_pos
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
                    ecg_graph_width_mm = COLUMN1_BOXES * ECG_LARGE_BOX_MM
                    time_window_seconds = ecg_graph_width_mm / wave_speed
                    print(f" Report Generator: Demo mode ON - Calculated window using NEW LOGIC: {ecg_graph_width_mm:.2f}mm / {wave_speed}mm/s = {time_window_seconds}s")
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
        #   - Time window = (strip width mm) / wave_speed ONLY
        #   - BPM window is NOT used - only wave speed determines time window
        #   - Beats = (BPM / 60) × time_window
        #   - Maximum clamp: 20 seconds (NO minimum clamp)
        calculated_time_window, _ = calculate_time_window_from_bpm_and_wave_speed(
            hr_bpm_value,  # From metrics.json (priority) - for calculation-based beats
            wave_speed_mm_s,  # From ecg_settings.json - for calculation-based beats
            desired_beats=6,  # Default: 6 beats desired
            num_boxes=COLUMN1_BOXES
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
        
        print(f" DEBUG: Processing Lead {lead} at position ({x_pos}, {y_pos})")
        print(f" DEBUG: Drawing dimensions: width={total_width}, height={total_height}")
        
        try:
            # STEP 3A: Add lead labels 15 points above ECG graph strips
            from reportlab.graphics.shapes import String
            
            # Calculate label position - shift 45 points right for column 1 leads, then 10 points left
            if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                label_x_pos = x_pos + 45.0 - 10.0  # Shift 45 points right, then 10 points left
            else:
                label_x_pos = x_pos  # Keep original position for other leads
                
            lead_label = String(label_x_pos, y_pos + 52, f"{lead}",  # Y+52 (35+17) - labels 17 points higher
                              fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
            master_drawing.add(lead_label)
            
            # Add vertical dotted line to separate columns - positioned to prevent column 1 data overlap
            if lead in column2_leads:  # Draw for ALL V leads to debug
                print(f" DEBUG: Drawing dotted line for {lead} at X=397.5, Y=430 to 105")
                # Draw vertical dotted line from V1 to V6, positioned between columns
                from reportlab.graphics.shapes import Line
                from reportlab.lib.units import mm
                
                # Position line 14.5 boxes (217.5 points) to the right
                # Original X=180, now X=180 + 217.5 = 397.5
                line_x = 397.5  # Shifted right 14.5 boxes from 180
                # Reduce line length by 2 boxes (30 points) from bottom
                line_y_start = 447  # 430 + 17 = 447 (shifted 17 points up)
                line_y_end = 122     # 105 + 17 = 122 (shifted 17 points up)
                
                # Create dotted line effect with multiple short lines
                dotted_line_spacing = 3  # points between dots
                dot_length = 2  # length of each dot
                
                current_y = line_y_start
                dots_count = 0
                while current_y > line_y_end:
                    # Draw small line segments to create dotted effect
                    segment_end = max(current_y - dot_length, line_y_end)
                    dotted_line = Line(line_x, current_y, line_x, segment_end,
                                    strokeColor=colors.black, strokeWidth=0.5)
                    master_drawing.add(dotted_line)
                    dots_count += 1
                    current_y = segment_end - dotted_line_spacing
                
                print(f" DEBUG: Added {dots_count} dotted segments to vertical line (reduced by 2 boxes from bottom)")
            
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
            
            # Priority 1: Use saved_ecg_data (REQUIRED for calculation-based beats)
            saved_data_samples = 0  # Initialize for comparison with live data
            if saved_ecg_data and 'leads' in saved_ecg_data:
                # For calculated leads, calculate from I and II
                if lead in ["III", "aVR", "aVL", "aVF", "-aVR"]:
                    if "I" in saved_ecg_data['leads'] and "II" in saved_ecg_data['leads']:
                        lead_i_data = saved_ecg_data['leads']["I"]
                        lead_ii_data = saved_ecg_data['leads']["II"]
                        
                        # Ensure same length
                        min_len = min(len(lead_i_data), len(lead_ii_data))
                        lead_i_data = lead_i_data[:min_len]
                        lead_ii_data = lead_ii_data[:min_len]
                        
                        # IMPORTANT: Subtract baseline from Lead I and Lead II BEFORE calculating derived leads
                        # This ensures calculated leads are centered around 0, not around baseline
                        baseline_adc = 2000.0
                        lead_i_centered = np.array(lead_i_data, dtype=float) - baseline_adc
                        lead_ii_centered = np.array(lead_ii_data, dtype=float) - baseline_adc
                        
                        # Calculate derived lead from centered values
                        calculated_data = calculate_derived_lead(lead, lead_i_centered, lead_ii_centered)
                        if calculated_data is not None:
                            raw_data = calculated_data.tolist() if isinstance(calculated_data, np.ndarray) else calculated_data
                            print(f" Calculated {lead} from saved I and II data (baseline-subtracted): {len(raw_data)} points")
                        else:
                            # Fallback to saved data if calculation fails
                            lead_name_for_saved = lead.replace("-aVR", "aVR")
                            if lead_name_for_saved in saved_ecg_data['leads']:
                                raw_data = saved_ecg_data['leads'][lead_name_for_saved]
                                if lead == "-aVR":
                                    raw_data = [-x for x in raw_data]  # Invert for -aVR
                            else:
                                raw_data = []
                    else:
                        print(f" Cannot calculate {lead}: I or II data missing in saved file")
                        raw_data = []
                else:
                    # For non-calculated leads, use saved data directly
                    lead_name_for_saved = lead.replace("-aVR", "aVR")  # Handle -aVR case
                    if lead_name_for_saved in saved_ecg_data['leads']:
                        raw_data = saved_ecg_data['leads'][lead_name_for_saved]
                        if lead == "-aVR":
                            raw_data = [-x for x in raw_data]  # Invert for -aVR
                    else:
                        raw_data = []
                
                # Check if saved data has enough samples for calculated time window
                    saved_data_samples = len(raw_data)
                    # Use dynamically calculated current_num_samples/extra_ii_samples instead of fixed num_samples_to_capture
                    target_samples = extra_ii_samples if (lead == "II" and y_pos == 90) else current_num_samples
                    
                    if saved_data_samples < target_samples:
                        print(f" SAVED FILE {lead} has only {saved_data_samples} samples, need {target_samples} for calculated window")
                        print(f"   Will use ALL saved data ({saved_data_samples} samples)")
                        # Use all available saved data (don't filter)
                        raw_data_to_use = raw_data
                    else:
                        # Apply time window filtering based on calculated window
                        raw_data_to_use = raw_data[-target_samples:]
                    
                    if len(raw_data_to_use) > 0 and np.std(raw_data_to_use) > 0.01:
                        real_ecg_data = np.array(raw_data_to_use)
                        real_data_available = True
                        time_window_str = f"{calculated_time_window:.2f}s" if calculated_time_window else "auto"
                        actual_time_window = len(real_ecg_data) / computed_sampling_rate if computed_sampling_rate > 0 else 0
                        print(f" Using SAVED FILE {lead} data: {len(real_ecg_data)} points (requested: {time_window_str}, actual: {actual_time_window:.2f}s, std: {np.std(real_ecg_data):.2f})")
            
            # Priority 2: Fallback to live dashboard data (if saved data not available OR has insufficient samples)
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
                                # Always use live dashboard data (ignore any saved data)
                                raw_data = calculated_data
                                target_samples = extra_ii_samples if (lead == "II" and y_pos == 90) else current_num_samples
                                if len(raw_data) >= target_samples:
                                    raw_data = raw_data[-target_samples:]
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
                        # Use dynamically calculated target_samples
                        target_samples = extra_ii_samples if (lead == "II" and y_pos == 90) else current_num_samples
                        # Check if we have enough samples, otherwise use all available
                        if len(raw_data) >= target_samples:
                            raw_data = raw_data[-target_samples:]
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
                        else:
                            pass
                    elif lead in lead_to_index and len(ecg_test_page.data) > lead_to_index[lead]:
                            # Get filtered real data for this lead
                            lead_index = lead_to_index[lead]
                            if len(ecg_test_page.data[lead_index]) > 0:
                                raw_data = ecg_test_page.data[lead_index]
                                # Use dynamically calculated target_samples
                                target_samples = extra_ii_samples if (lead == "II" and y_pos == 90) else current_num_samples
                                # Check if we have enough samples, otherwise use all available
                                if len(raw_data) >= target_samples:
                                    raw_data = raw_data[-target_samples:]
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
                            else:
                                pass
            
            if real_data_available and len(real_ecg_data) > 0:
                # Draw ALL REAL ECG data - NO LIMITS
                # Adjust ECG width based on column to ensure column 1 stops before dotted line
                if lead in column1_leads:
                    # Column 1: Graph should stop exactly at dotted line at X=397.5
                    # Lead starts at adjusted_x_pos (x_pos + 30), so calculate exact width
                    if lead == "II":
                        ecg_width = COLUMN1_BOXES * ECG_LARGE_BOX_MM * mm  # 52.6 boxes × 5.25mm = 276.15mm
                    elif lead in column1_leads:
                        ecg_width = COLUMN1_BOXES * ECG_LARGE_BOX_MM * mm  # 26.3 boxes × 5.25mm = 138.075mm
                    else:
                        ecg_width = COLUMN2_BOXES * ECG_LARGE_BOX_MM * mm  # 26.3 boxes × 5.25mm = 138.075mm
                else:
                    # Column 2: Full width available (from X=400 onwards)
                        ecg_width = COLUMN2_BOXES * ECG_LARGE_BOX_MM * mm  # 26.3 boxes × 5.25mm = 138.075mm
                
                ecg_height = 45
                
                # Create time array for ALL data with adjusted width and position
                # For column 1 leads, shift start position 2 boxes (30 points) to the right
                if lead in column1_leads:
                    # Shift column 1 leads 2 boxes (30 points) to the right
                    adjusted_x_pos = x_pos + 30  # Move 30 points right
                else:
                    # Column 2 leads keep original position
                    adjusted_x_pos = x_pos
                
                # Clinical calibration: 1 second = 5 large boxes at 25 mm/s (exactly 25 mm)
                fs = float(computed_sampling_rate)
                
                
                # Step 1: Convert ADC data to numpy array
                adc_data = np.array(real_ecg_data, dtype=float)

                # Step 1.1: Apply report filters (DFT -> EMG -> AC) on raw ADC data
                try:
                    from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
                    dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
                    emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
                    ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
                    if dft_setting not in ("off", ""):
                        adc_data = apply_dft_filter(adc_data, float(computed_sampling_rate), dft_setting)
                    if emg_setting not in ("off", ""):
                        adc_data = apply_emg_filter(adc_data, float(computed_sampling_rate), emg_setting)
                    if ac_setting in ("50", "60"):
                        adc_data = apply_ac_filter(adc_data, float(computed_sampling_rate), ac_setting)
                except Exception as filter_err:
                    print(f" Report filter apply failed for {lead}: {filter_err}")
                
                # DEBUG: Check if data is already processed (baseline-subtracted)
                data_mean = np.mean(adc_data)
                data_std = np.std(adc_data)
                is_calculated_lead = lead in ["III", "aVR", "aVL", "aVF", "-aVR"]
                
                # Step 2: Apply baseline 2000 (subtract baseline from ADC values)
                # IMPORTANT: For calculated leads, data is already calculated from processed I and II
                # So it's already centered (mean ~0), but we still need to scale it properly
                baseline_adc = 2000.0
                
                if abs(data_mean - 2000.0) < 500:  # Data is close to baseline 2000 (raw ADC)
                    centered_adc = adc_data - baseline_adc
                elif is_calculated_lead:
                    # For calculated leads, data is already centered from calculation (II - I, etc.)
                    # The calculated value is already the difference, so it's centered around 0
                    # We use it directly without baseline subtraction
                    centered_adc = adc_data  # Use data as-is (already centered from calculation)
                else:  # Data is already processed (baseline-subtracted or filtered)
                    centered_adc = adc_data  # Use data as-is (already centered)
                
                # Step 3: Calculate ADC per box based on wave_gain and lead-specific multiplier
                # LEAD-SPECIFIC ADC PER BOX CONFIGURATION
                # Each lead can have different ADC per box multiplier (will be divided by wave_gain)
                # Get lead-specific ADC per box multiplier (default: 6400)
                adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get(lead, 6400.0)
                # Formula: ADC_per_box = adc_per_box_multiplier / wave_gain_mm_mv
                # IMPORTANT: Each lead can have different ADC per box multiplier
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                # For 10mm/mV with multiplier 6400: 6400 / 10 = 640 ADC per box
                adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)  # Avoid division by zero
                
                # DEBUG: Log actual ADC values for troubleshooting
                max_centered_adc_abs = np.max(np.abs(centered_adc))
                expected_boxes = max_centered_adc_abs / adc_per_box
                
                
                
                boxes_offset = centered_adc / adc_per_box
                
                # Step 5: Convert boxes to Y position (in mm, then to points)
                # Center of graph is at y_pos + (ecg_height / 2.0)
                # IMPORTANT: User changed to height/3 = 45/3 = 15.0 points per box
                # This matches the actual grid spacing the user wants
                center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                major_spacing_y = ecg_height / 3.0  # height/3 = 15.0 points per box (user's choice)
                box_height_points = major_spacing_y  # Use actual grid spacing (height/3)
                
                # Slices to exactly fit the visible width, cropping extra samples instead of stretching
                visible_seconds = ecg_width / (25.0 * mm)
                visible_samples = int(round(visible_seconds * fs))
                if len(boxes_offset) > visible_samples:
                    boxes_offset = boxes_offset[-visible_samples:]
                
                ecg_normalized = center_y + (boxes_offset * box_height_points)
                
                gap_points = mm
                t_sec = np.arange(len(boxes_offset)) / fs
                t = adjusted_x_pos + gap_points + (t_sec * 25.0 * mm)
                
                
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
                
                # Calibration notch (only for I, II, III, aVR, aVL, aVF)
                from reportlab.lib.units import mm
                notch_path = None
                print(f" DEBUG: Checking notch for Lead {lead} - is in column1: {lead in ['I', 'II', 'III', 'aVR', 'aVL', 'aVF']}")
                if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                    print(f" DEBUG: Creating calibration notch for Lead {lead}")
                    
                    # Dynamic calibration notch based on wave gain
                    try:
                        from utils.settings_manager import SettingsManager
                        settings_mgr = SettingsManager()
                        notch_boxes = settings_mgr.get_calibration_notch_boxes()
                        print(f" Dynamic notch: {notch_boxes} boxes for gain {settings_mgr.get_wave_gain()}mm/mV")
                    except Exception as e:
                        print(f" Could not get dynamic notch, using default: {e}")
                        notch_boxes = 2.0  # Default fallback
                    
                    notch_width_mm = 5.0   # width 5mm (fixed)
                    notch_height_mm = notch_boxes * 5.0  # height = boxes × 5mm per box
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    # For 6:2 format, ECG data starts at adjusted_x_pos, so notch should be relative to that
                    notch_x = adjusted_x_pos + 10.0 - 30.0  # Same relative position as 4:3 format
                    notch_y_base = center_y
                    print(f" DEBUG: Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
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
                    print(f" DEBUG: Notch added to master drawing for Lead {lead}")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in column1 leads)")
                
                print(f" Drew {len(real_ecg_data)} ECG data points for Lead {lead}")
                if lead in ["I", "II", "III", "aVR", "aVL", "aVF"] and notch_path:
                    print(f" Added calibration notch for Lead {lead}")
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
                flat_line_end_x = x_pos + 100  # Default end position
                flat_line_y = center_y  # Center/baseline position
                
                flat_line = Line(flat_line_start_x, flat_line_y, flat_line_end_x, flat_line_y,
                              strokeColor=colors.HexColor("#000000"), strokeWidth=1.2)
                master_drawing.add(flat_line)
                if saved_ecg_data and 'leads' in saved_ecg_data:
                    print(f" DEBUG: Available leads in saved data: {list(saved_ecg_data['leads'].keys())}")
                
                # ALWAYS create calibration notch for first 6 leads regardless of data availability
                # This ensures notch appears in both conditions: with real data AND without real data
                if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                    print(f" DEBUG: Creating GUARANTEED calibration notch for Lead {lead} (no data case)")
                    from reportlab.lib.units import mm
                    from reportlab.graphics.shapes import Path
                    
                    # Dynamic calibration notch based on wave gain
                    try:
                        from utils.settings_manager import SettingsManager
                        settings_mgr = SettingsManager()
                        notch_boxes = settings_mgr.get_calibration_notch_boxes()
                        print(f" Dynamic notch (no data): {notch_boxes} boxes for gain {settings_mgr.get_wave_gain()}mm/mV")
                    except Exception as e:
                        print(f" Could not get dynamic notch (no data), using default: {e}")
                        notch_boxes = 2.0  # Default fallback
                    
                    notch_width_mm = 5.0   # width 5mm (fixed)
                    notch_height_mm = notch_boxes * 5.0  # height = boxes × 5mm per box
                    notch_width = notch_width_mm * mm
                    notch_height = notch_height_mm * mm
                    
                    # Calculate center_y same as real data section
                    ecg_height = 45  # Same as real data section
                    center_y = y_pos + (ecg_height / 2.0)  # Center of the graph in points
                    
                    # Place notch 10 points after where ECG strip starts, then shift 30 points left
                    # For 6:2 format, ECG data would start at adjusted_x_pos, so notch should be relative to that
                    if lead in ["I", "II", "III", "aVR", "aVL", "aVF"]:
                        adjusted_x_pos = x_pos + 30  # Column 1 leads are shifted 30 points right
                    else:
                        adjusted_x_pos = x_pos  # Column 2 leads keep original position
                    notch_x = adjusted_x_pos + 10.0 - 30.0  # Same relative position as real data
                    notch_y_base = center_y  # Use same center_y calculation as real data section
                    print(f" DEBUG: GUARANTEED Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
                    
                    guaranteed_notch_path = Path(
                        fillColor=None,
                        strokeColor=colors.HexColor("#000000"),
                        strokeWidth=0.8,
                        strokeLineCap=1,
                        strokeLineJoin=0
                    )
                    guaranteed_notch_path.moveTo(notch_x, notch_y_base)
                    guaranteed_notch_path.lineTo(notch_x, notch_y_base + notch_height)
                    guaranteed_notch_path.lineTo(notch_x + notch_width, notch_y_base + notch_height)
                    guaranteed_notch_path.lineTo(notch_x + notch_width, notch_y_base)
                    # Small forward tick to the right (extra 2mm) for clearer notch end
                    guaranteed_notch_path.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
                    
                    # Add notch to master drawing
                    master_drawing.add(guaranteed_notch_path)
                    print(f" DEBUG: GUARANTEED calibration notch added for Lead {lead} (no data case)")
                    print(f" Added GUARANTEED calibration notch for Lead {lead} (no data case)")
                else:
                    print(f" DEBUG: Skipping notch for Lead {lead} (not in first 6 leads, no data case)")
            
            successful_graphs += 1
            
        except Exception as e:
            print(f" Error adding Lead {lead}: {e}")
            import traceback
            traceback.print_exc()
    
    # STEP 3.5: Add extra Lead II label below aVF (45 points below aVF position)
    # Extra Lead II is at index 6 in column1_leads, position = start_y - (6 * 60) = 412 - 360 = 52
    # Extra Lead II label position = 112 - 45 = 67
    from reportlab.graphics.shapes import String
    
    # Calculate aVF position to place Lead II label 45 points below it
    avf_lead_index = 5  # aVF is 6th lead (index 5) in column1_leads
    avf_y_pos = start_y - (avf_lead_index * 60)  # 412 - (5 * 60) = 112
    extra_lead_ii_y_pos = avf_y_pos - 45 - 15  # Original vertical position for label/notch
    
    # Use same X position as other column1 leads (shifted 45 points right, then 10 points left)
    extra_lead_ii_x_pos = (20 - 45) + 45.0 - 10.0  # Same as other column1 leads
    
    # Add extra Lead II label (shifted 50 points up from strip: 60 - 10 = 50)
    extra_lead_ii_label_y = extra_lead_ii_y_pos + 50  # Label 50 points above ECG strip (60 - 10 = 50)
    extra_lead_ii_label = String(extra_lead_ii_x_pos, extra_lead_ii_label_y, "II",  # 50 points above ECG strip
                                fontSize=10, fontName="Helvetica-Bold", fillColor=colors.black)
    master_drawing.add(extra_lead_ii_label)
    
    print(f" Added extra Lead II label at position ({extra_lead_ii_x_pos}, {extra_lead_ii_label_y}) - 50 points above ECG strip")
    
    # STEP 3.6: Add extra Lead II ECG strip and calibration notch (45 points below aVF)
    # Get Lead II data for the extra strip - EXACTLY SAME AS ORIGINAL LEAD II PROCESSING
    extra_lead_ii_raw_data = None
    extra_lead_ii_data = None
    
    # Try to get Lead II data from saved_ecg_data - EXACTLY LIKE ORIGINAL LEAD II
    if saved_ecg_data and 'leads' in saved_ecg_data and "II" in saved_ecg_data['leads']:
        extra_lead_ii_raw_data = saved_ecg_data['leads']["II"]
        print(f" Found Lead II raw data from saved_ecg_data: {len(extra_lead_ii_raw_data)} points")
        
        if len(extra_lead_ii_raw_data) > 0:
            saved_data_samples = len(extra_lead_ii_raw_data)
            from reportlab.lib.units import mm as _mm_units
            right_margin = 10
            extra_lead_ii_x_pos_calc = 20 - 45
            extra_lead_ii_adjusted_x_pos_calc = extra_lead_ii_x_pos_calc + 30
            width_points_calc = max(0, (total_width - extra_lead_ii_adjusted_x_pos_calc) - right_margin)
            effective_wave_speed = wave_speed_mm_s * ECG_SPEED_SCALE
            time_window_seconds = (width_points_calc / _mm_units) / max(1e-6, effective_wave_speed)
            num_samples_to_capture = int(time_window_seconds * computed_sampling_rate)
            if num_samples_to_capture <= 0:
                num_samples_to_capture = SIX_TWO_SAMPLES_EXTRA_II
            if saved_data_samples < num_samples_to_capture:
                print(f" Extra Lead II has only {saved_data_samples} samples, using all available data")
                extra_lead_ii_data_to_use = extra_lead_ii_raw_data
            else:
                extra_lead_ii_data_to_use = extra_lead_ii_raw_data[-num_samples_to_capture:]
            
            extra_lead_ii_data = extra_lead_ii_data_to_use
    
    # Try to get Lead II data from live dashboard - EXACTLY LIKE ORIGINAL LEAD II
    elif ecg_test_page and hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 1:
        extra_lead_ii_raw_data = ecg_test_page.data[1]
        print(f" Found Lead II raw data from live dashboard: {len(extra_lead_ii_raw_data)} points")
        from reportlab.lib.units import mm as _mm_units
        right_margin = 10
        extra_lead_ii_x_pos_calc = 20 - 45
        extra_lead_ii_adjusted_x_pos_calc = extra_lead_ii_x_pos_calc + 30
        width_points_calc = max(0, (total_width - extra_lead_ii_adjusted_x_pos_calc) - right_margin)
        effective_wave_speed = wave_speed_mm_s * ECG_SPEED_SCALE
        time_window_seconds = (width_points_calc / _mm_units) / max(1e-6, effective_wave_speed)
        num_samples_to_capture = int(time_window_seconds * computed_sampling_rate)
        if num_samples_to_capture <= 0:
            num_samples_to_capture = SIX_TWO_SAMPLES_EXTRA_II
        if len(extra_lead_ii_raw_data) > num_samples_to_capture:
            extra_lead_ii_data = extra_lead_ii_raw_data[-num_samples_to_capture:]
        else:
            extra_lead_ii_data = extra_lead_ii_raw_data
    
    # Draw extra Lead II ECG strip if data is available
    if extra_lead_ii_data is not None and len(extra_lead_ii_data) > 0:
        print(f" Drawing extra Lead II ECG strip with {len(extra_lead_ii_data)} points")
        
        extra_lead_ii_x_pos = 20 - 45
        extra_lead_ii_adjusted_x_pos = extra_lead_ii_x_pos + 30
        extra_lead_ii_y = extra_lead_ii_y_pos
        from reportlab.lib.units import mm as _mm_units
        right_margin = 10
        extra_lead_ii_width_points = max(0, (total_width - extra_lead_ii_adjusted_x_pos) - right_margin)
        extra_lead_ii_height = 45
        
        # Get wave_gain for amplitude scaling
        wave_gain_mm_mv = 10.0  # Default
        if settings_manager:
            try:
                wave_gain_setting = settings_manager.get_setting("wave_gain", "10")
                wave_gain_mm_mv = float(wave_gain_setting) if wave_gain_setting else 10.0
            except Exception:
                wave_gain_mm_mv = 10.0
        
        # Process Lead II data for drawing
        adc_per_box_multiplier = ADC_PER_BOX_CONFIG.get("II", 6400.0)
        
        # Convert to numpy array and process
        adc_data = np.array(extra_lead_ii_data, dtype=float)

        # Apply report filters (DFT -> EMG -> AC) on extra Lead II data
        try:
            from ecg.ecg_filters import apply_dft_filter, apply_emg_filter, apply_ac_filter
            dft_setting = str(settings_manager.get_setting("filter_dft", "off")).strip()
            emg_setting = str(settings_manager.get_setting("filter_emg", "25")).strip()
            ac_setting = str(settings_manager.get_setting("filter_ac", "off")).strip()
            if dft_setting not in ("off", ""):
                adc_data = apply_dft_filter(adc_data, float(computed_sampling_rate), dft_setting)
            if emg_setting not in ("off", ""):
                adc_data = apply_emg_filter(adc_data, float(computed_sampling_rate), emg_setting)
            if ac_setting in ("50", "60"):
                adc_data = apply_ac_filter(adc_data, float(computed_sampling_rate), ac_setting)
        except Exception as filter_err:
            print(f" Report filter apply failed for extra Lead II: {filter_err}")
        baseline_adc = 2000.0
        data_mean = float(np.mean(adc_data)) if adc_data.size > 0 else 0.0
        is_calculated_lead = False
        if abs(data_mean - baseline_adc) < 500.0:
            baseline_corrected = adc_data - baseline_adc
        elif is_calculated_lead:
            baseline_corrected = adc_data
        else:
            baseline_corrected = adc_data
        centered_adc = baseline_corrected - (float(np.mean(baseline_corrected)) if baseline_corrected.size > 0 else 0.0)
        adc_per_box = adc_per_box_multiplier / max(1e-6, wave_gain_mm_mv)
        boxes_offset = centered_adc / adc_per_box
        
        # Slices to exactly fit the visible width, cropping extra samples instead of stretching
        visible_seconds = extra_lead_ii_width_points / (25.0 * mm)
        visible_samples = int(round(visible_seconds * computed_sampling_rate))
        if len(boxes_offset) > visible_samples:
            boxes_offset = boxes_offset[-visible_samples:]
            
        ecg_normalized = center_y_wave + (boxes_offset * box_height_points)
        
        # Clinical calibration: exact time axis – 25 mm/s means 25 mm per second on paper
        fs = float(computed_sampling_rate)
        t_sec = np.arange(len(boxes_offset)) / fs
        
        # Calculate X positions with 1mm gap
        gap_points = mm # 1mm gap
        t = extra_lead_ii_adjusted_x_pos + gap_points + (t_sec * 25.0 * mm)
        
        # Draw ECG strip
        from reportlab.graphics.shapes import Path
        extra_lead_ii_path = Path(fillColor=None, 
                                 strokeColor=colors.HexColor("#000000"), 
                                 strokeWidth=0.4,
                                 strokeLineCap=1,
                                 strokeLineJoin=1)
        
        # Render trim for extra Lead II to suppress residual edge artifacts
        extra_lead_ii_path.moveTo(t[0], ecg_normalized[0])
        for i in range(1, len(t)):
            extra_lead_ii_path.lineTo(t[i], ecg_normalized[i])
        
        # Add ECG strip to master drawing
        master_drawing.add(extra_lead_ii_path)
        
        print(f" Drew extra Lead II ECG strip with {len(extra_lead_ii_data)} points")
        
        # Add calibration notch for extra Lead II
        from reportlab.lib.units import mm
        
        # Dynamic calibration notch based on wave gain
        try:
            from utils.settings_manager import SettingsManager
            settings_mgr = SettingsManager()
            notch_boxes = settings_mgr.get_calibration_notch_boxes()
            print(f" Dynamic notch (extra Lead II): {notch_boxes} boxes for gain {settings_mgr.get_wave_gain()}mm/mV")
        except Exception as e:
            print(f" Could not get dynamic notch (extra Lead II), using default: {e}")
            notch_boxes = 2.0  # Default fallback
        
        notch_width_mm = 5.0   # width 5mm (fixed)
        notch_height_mm = notch_boxes * 5.0  # height = boxes × 5mm per box
        notch_width = notch_width_mm * mm
        notch_height = notch_height_mm * mm
        
        # Calculate notch position (same as other column1 leads - at ECG strip start)
        # Place notch 10 points after where ECG strip starts, then shift 30 points left
        # For 6:2 format, ECG data starts at adjusted_x_pos, so notch should be relative to that
        notch_x = extra_lead_ii_adjusted_x_pos + 10.0 - 30.0  # Same relative position as other leads
        notch_y_base = center_y_notch  # Keep notch at original center position
        print(f" DEBUG: Extra Lead II Notch position - X: {notch_x}, Y: {notch_y_base}, Width: {notch_width}, Height: {notch_height}")
        
        # Create calibration notch (same styling as other leads)
        extra_lead_ii_notch = Path(
            fillColor=None,
            strokeColor=colors.HexColor("#000000"),
            strokeWidth=0.8,
            strokeLineCap=1,
            strokeLineJoin=1
        )
        extra_lead_ii_notch.moveTo(notch_x, notch_y_base)
        extra_lead_ii_notch.lineTo(notch_x, notch_y_base + notch_height)
        extra_lead_ii_notch.lineTo(notch_x + notch_width, notch_y_base + notch_height)
        extra_lead_ii_notch.lineTo(notch_x + notch_width, notch_y_base)
        # Small forward tick to the right (extra 2mm) for clearer notch end
        extra_lead_ii_notch.lineTo(notch_x + notch_width + (2.0 * mm), notch_y_base)
        
        # Add notch to master drawing
        master_drawing.add(extra_lead_ii_notch)
        
        print(f" Added calibration notch for extra Lead II")
        
    else:
        print(f" No Lead II data available for extra strip")
    
    # STEP 4: Add Patient Info, Date/Time and Vital Parameters to master drawing
    # POSITIONED ABOVE ECG GRAPH (not mixed inside graph)
    from reportlab.graphics.shapes import String

    # LEFT SIDE: Patient Info (TOP LEFT CORNER - moved up by 10 points)
    patient_name_label = String(-5, 550, f"Name: {full_name}",  # Moved up 10 points (550)
                           fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_name_label)

    patient_age_label = String(-5, 530, f"Age: {age}",  # Moved up 10 points (530)
                          fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_age_label)

    patient_gender_label = String(-5, 510, f"Gender: {gender}",  # Moved up 10 points (510)
                             fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(patient_gender_label)

    # RIGHT SIDE: Vital Parameters at SAME LEVEL as patient info (ABOVE ECG GRAPH)
    # Get real ECG data from dashboard
    HR = data.get('HR_avg',)
    PR = data.get('PR',) 
    QRS = data.get('QRS',)
    QT = data.get('QT',)
    QTc = data.get('QTc',)
    ST = data.get('ST',)
    # DYNAMIC RR interval calculation from heart rate (instead of hard-coded 857)
    RR = int(60000 / HR) if HR and HR > 0 else 0  # RR interval in ms from heart rate
   
    # Add vital parameters in TWO COLUMNS (ALIGNED with patient info - moved up by 10 points) - SHIFTED UP 5 more points
    # FIRST COLUMN (Left side - x=130) - ALIGNED and moved up
    hr_label = String(130, 555, f"HR    : {HR} bpm",  # Moved up 5 points from 550 to 555
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(hr_label)

    pr_label = String(130, 541, f"PR    : {PR} ms",  # Moved up 5 points from 530 to 535
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(pr_label)

    qrs_label = String(130, 527, f"QRS : {QRS} ms",  # Moved up 5 points from 510 to 515
                      fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qrs_label)
    
    rr_label = String(130, 510, f"RR    : {RR} ms",  # Moved up 5 points from 490 to 495
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rr_label)

    qt_label = String(130, 496, f"QT    : {int(round(QT))} ms",  # Moved up 5 points from 470 to 475
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qt_label)

    qtc_label = String(130, 482, f"QTc  : {int(round(QTc))} ms",  # Moved up 5 points from 450 to 455
                      fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtc_label)

    # SECOND COLUMN (Right side) - QTcF replaces ST
    _qtcf_val = data.get('QTc_Fridericia') or data.get('QTcF_ms') or data.get('QTcF') or data.get('QTcF_interval')
    _qtcf_display = f"{int(round(float(_qtcf_val)))} ms" if _qtcf_val and float(_qtcf_val) > 0 else "-- ms"
    qtcf_header_label = String(240, 496, f"QTcF : {_qtcf_display}",
                               fontSize=9, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_header_label)

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
            fs = 250.0
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
    p_axis_deg = "--"
    qrs_axis_deg = "--"
    t_axis_deg = "--"
    
    if ecg_test_page is not None and hasattr(ecg_test_page, 'data') and len(ecg_test_page.data) > 5:
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            
            # Get Lead I (index 0) and Lead aVF (index 5)
            lead_I = ecg_test_page.data[0] if len(ecg_test_page.data) > 0 else None
            lead_aVF = ecg_test_page.data[5] if len(ecg_test_page.data) > 5 else None
            
            # Get sampling rate
            fs = 250.0
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
                                            score = 100 - abs(cand_normalized - 37.5)  
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
        
    # Format axis values for display (remove ° symbol for compact display)
    # Convert to string first in case they're integers
    p_axis_display = str(p_axis_deg).replace("°", "") if p_axis_deg != "--" else "--"
    qrs_axis_display = str(qrs_axis_deg).replace("°", "") if qrs_axis_deg != "--" else "--"
    t_axis_display = str(t_axis_deg).replace("°", "") if t_axis_deg != "--" else "--"
    
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
    
    p_mm = extract_axis_value(p_axis_deg)
    qrs_mm = extract_axis_value(qrs_axis_deg)
    t_mm = extract_axis_value(t_axis_deg)
    
    # SECOND COLUMN - P/QRS/T Axis (ABOVE ECG GRAPH - same position)
    # p_qrs_label = String(240, 555, f"P/QRS/T  : {p_axis_display}/{qrs_axis_display}/{t_axis_display}°",  # Changed to axis values
    #                      fontSize=10, fontName="Helvetica", fillColor=colors.black)
    # master_drawing.add(p_qrs_label)

    # Get RV5 and SV1 amplitudes from the actual raw ECG graph whenever available.
    rv5_amp = None
    sv1_amp = None
    used_live_measurement = False

    if ecg_test_page is not None:
        try:
            from .metrics.intervals import calculate_rv5_sv1_from_test_page
            rv5_calc, sv1_calc = calculate_rv5_sv1_from_test_page(ecg_test_page)
            if rv5_calc is not None and rv5_calc > 0:
                rv5_amp = float(rv5_calc)
                used_live_measurement = True
            if sv1_calc is not None and sv1_calc != 0.0:
                sv1_amp = float(sv1_calc)
                used_live_measurement = True
        except Exception as e:
            print(f" Error getting live RV5/SV1 measurement: {e}")

    print(" Report Generator - Final RV5/SV1 source values:")
    print(f"   rv5: {rv5_amp}, sv1: {sv1_amp}, source={'live_raw_ecg' if used_live_measurement else 'lead_data_unavailable'}")
    
    # If missing/zero, compute from V5 and V1 of last 10 seconds (GE/Hospital Standard)
    # CRITICAL: Use RAW ECG data, not display-filtered signals
    # Measurements must be from median beat, relative to TP baseline (isoelectric segment before P-wave)
    # NOTE: sv1_amp can be negative (SV1 is negative by definition), so check for == 0.0, not <= 0
    if False:
        try:
            from scipy.signal import butter, filtfilt, find_peaks
            fs = 250.0
            if hasattr(ecg_test_page, 'sampler') and hasattr(ecg_test_page.sampler,'sampling_rate') and ecg_test_page.sampler.sampling_rate:
                fs = float(ecg_test_page.sampler.sampling_rate)
            def _get_last(arr):
                return arr[-int(10*fs):] if arr is not None and len(arr)>int(10*fs) else arr
            # V5 index 10, V1 index 6
            v5 = _get_last(ecg_test_page.data[10]) if len(ecg_test_page.data)>10 else None
            v1 = _get_last(ecg_test_page.data[6]) if len(ecg_test_page.data)>6 else None
            # V5 index 10, V1 index 6 - Get RAW data
            v5_raw = _get_last(ecg_test_page.data[10]) if len(ecg_test_page.data)>10 else None
            v1_raw = _get_last(ecg_test_page.data[6]) if len(ecg_test_page.data)>6 else None
            
            if v5_raw is not None and len(v5_raw)>int(2*fs):
                # Apply filter ONLY for R-peak detection (0.5-40 Hz)
                # Use RAW data for amplitude measurements
                nyq = fs/2.0
                b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
                v5f = filtfilt(b,a, np.asarray(v5_raw))
                env = np.convolve(np.square(np.diff(v5f)), np.ones(int(0.15*fs))/(0.15*fs), mode='same')
                r,_ = find_peaks(env, height=np.mean(env)+0.5*np.std(env), distance=int(0.6*fs))
                vals=[]
                for rr in r[1:-1]:
                    # QRS window: ±80ms around R-peak
                    qs = max(0, rr-int(0.08*fs))
                    qe = min(len(v5_raw), rr+int(0.08*fs))
                    if qe > qs:
                        # Use RAW data for amplitude measurement
                        qrs_segment = np.asarray(v5_raw[qs:qe])
                        
                        # TP baseline: isoelectric segment before P-wave onset (150-350ms before R)
                        tp_start = max(0, rr-int(0.35*fs))
                        tp_end = max(0, rr-int(0.15*fs))
                        if tp_end > tp_start:
                            tp_segment = np.asarray(v5_raw[tp_start:tp_end])
                            tp_baseline = np.median(tp_segment)  # Median for robustness
                        else:
                            # Fallback: short segment before QRS
                            tp_baseline = np.median(np.asarray(v5_raw[max(0,qs-int(0.05*fs)):qs]))
                        
                        # RV5 = max(QRS) - TP_baseline (positive, in mV)
                        # Convert from ADC counts to mV: V5 uses 6400 multiplier, 10mm/mV → 1 mV = 1280 ADC
                        r_amp_adc = np.max(qrs_segment) - tp_baseline
                        if r_amp_adc > 0:
                            r_amp_mv = r_amp_adc / 1280.0  # Convert ADC to mV
                            vals.append(r_amp_mv)
                if len(vals)>0 and rv5_amp<=0: 
                    rv5_amp = float(np.median(vals))  # Median beat approach
                    
            if v1_raw is not None and len(v1_raw)>int(2*fs):
                # Apply filter ONLY for R-peak detection (0.5-40 Hz)
                # Use RAW data for amplitude measurements
                nyq = fs/2.0
                b,a = butter(2, [max(0.5/nyq, 0.001), min(40.0/nyq,0.99)], btype='band')
                v1f = filtfilt(b,a, np.asarray(v1_raw))
                env = np.convolve(np.square(np.diff(v1f)), np.ones(int(0.15*fs))/(0.15*fs), mode='same')
                r,_ = find_peaks(env, height=np.mean(env)+0.5*np.std(env), distance=int(0.6*fs))
                vals=[]
                for rr in r[1:-1]:
                    # QRS window: ±80ms around R-peak
                    ss = rr
                    se = min(len(v1_raw), rr+int(0.08*fs))
                    if se > ss:
                        # Use RAW data for amplitude measurement
                        qrs_segment = np.asarray(v1_raw[ss:se])
                        
                        # TP baseline: isoelectric segment before P-wave onset (150-350ms before R)
                        tp_start = max(0, rr-int(0.35*fs))
                        tp_end = max(0, rr-int(0.15*fs))
                        if tp_end > tp_start:
                            tp_segment = np.asarray(v1_raw[tp_start:tp_end])
                            tp_baseline = np.median(tp_segment)  # Median for robustness
                        else:
                            # Fallback: short segment before QRS
                            tp_baseline = np.median(np.asarray(v1_raw[max(0, ss-int(0.05*fs)):ss]))
                        
                        # SV1 = min(QRS) - TP_baseline (negative, preserve sign, in mV)
                        # Convert from ADC counts to mV: V1 uses 6400 multiplier, 10mm/mV → 1 mV = 1280 ADC
                        s_amp_adc = np.min(qrs_segment) - tp_baseline
                        if s_amp_adc < 0:  # SV1 must be negative
                            s_amp_mv = s_amp_adc / 1280.0  # Convert ADC to mV (preserve sign)
                            vals.append(s_amp_mv)
                if len(vals)>0 and sv1_amp==0.0:
                    sv1_amp = float(np.median(vals))  # Median beat approach, negative value
            print(f" Fallback computed RV5/SV1: RV5={rv5_amp:.4f}, SV1={sv1_amp:.4f}")
        except Exception as e:
            print(f" Fallback RV5/SV1 computation failed: {e}")

    # Unit conversion: GE/Hospital Standard - Values must be in mV
    # CRITICAL: calculate_wave_amplitudes() now returns values in mV (converted from ADC counts)
    # No additional conversion needed - use values directly
    # GE Standard ranges: RV5 typically 0.5-3.0 mV, SV1 typically -0.5 to -2.0 mV
    rv5_mv = float(rv5_amp) if rv5_amp is not None and rv5_amp > 0 else None
    sv1_mv = float(sv1_amp) if sv1_amp is not None and sv1_amp != 0.0 else None
    
    print(f"   Converted to mV: RV5={rv5_mv if rv5_mv is not None else '--'}, SV1={sv1_mv if sv1_mv is not None else '--'}")
    
    # SECOND COLUMN - RV5/SV1 (ALIGNED and moved up 5 points)
    # Display SV1 as negative mV (GE/Hospital standard)
    # Use 3 decimal places for precision (not rounded to integers)
    rv5_text = f"{rv5_mv:.3f} mV" if rv5_mv is not None else "--"
    sv1_text = f"{sv1_mv:.3f} mV" if sv1_mv is not None else "--"
    rv5_sv_label = String(240, 541, f"RV5/SV1  : {rv5_text}/{sv1_text}",
                          fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv_label)

    # Calculate RV5+SV1 from displayed magnitudes: RV5 - SV1_magnitude.
    rv5_sv1_sum = (rv5_mv - abs(sv1_mv)) if rv5_mv is not None and sv1_mv is not None else None
    
    # SECOND COLUMN - RV5+SV1 (ALIGNED and moved up 5 points)
    # Use 3 decimal places for precision
    rv5_sv1_sum_text = f"{rv5_sv1_sum:.3f} mV" if rv5_sv1_sum is not None else "--"
    rv5_sv1_sum_label = String(240, 527, f"RV5+SV1 : {rv5_sv1_sum_text}",  # Moved up 5 points from 510 to 515
                               fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(rv5_sv1_sum_label)

    # SECOND COLUMN - QTCF (ABOVE ECG GRAPH - shifted further up)
    # Calculate QTcF using Fridericia formula: QTcF = QT / RR^(1/3)
    qtcf_val = _safe_float(data.get("QTc_Fridericia") or data.get("QTcF_ms") or data.get("QTcF"), None)
    if qtcf_val and qtcf_val > 0:
        qtcf_sec = qtcf_val / 1000.0
        qtcf_text = f"QTCF       : {qtcf_val:.0f} ms ({qtcf_sec:.3f} s)"
    else:
        qtcf_text = "QTCF       : --"
    qtcf_label = String(240, 510, qtcf_text,  # Moved up from 642 to 652
                        fontSize=10, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(qtcf_label)

    # SECOND COLUMN - Speed/Gain (merged in one line) (ABOVE ECG GRAPH - shifted further up)
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
    master_drawing.add(String(
        240,
        482,  # Moved up from 606 to 616
        f"{wave_speed_mm_s} mm/s   {filter_band}   AC : {ac_frequency}   {wave_gain_mm_mv} mm/mV",
        fontSize=10,
        fontName="Helvetica",
        fillColor=colors.black,
    ))

    


    
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
    confirmed_label = String(-15, 35, "Reference Report Confirmed by ", 
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(confirmed_label)

    # Doctor Name (below V6 lead) - SHIFTED 15 points right and 30 points up
    doctor_name_label = String(-15, 20, "Doctor Name: ",  # X: -30→-15 (right 15), Y: -10→20 (up 30)
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(doctor_name_label)
    
    if doctor:
        value_x = -15 + stringWidth("Doctor Name: ", "Helvetica", 8) + 5  # Updated X position
        doctor_name_value = String(value_x, 20, doctor,  # Updated Y position
                                fontSize=8, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(doctor_name_value)

    # Doctor Signature (below Doctor Name) - SHIFTED 15 points right and 30 points up
    doctor_sign_label = String(-15, 5, "Doctor Sign: ",  # X: -30→-15 (right 15), Y: -25→5 (up 30)
                              fontSize=8, fontName="Helvetica", fillColor=colors.black)
    master_drawing.add(doctor_sign_label)

    # Add RIGHT-SIDE Conclusion Box (moved to the right) - NOW DYNAMIC FROM DASHBOARD (12 conclusions max) - MADE SMALLER
    # SHIFTED UP by 50 points (45 + 5) and RIGHT by 75 points
    conclusion_y_start = 41.  # Shifted up 5 more points from 36 to 41
    
    # Create a rectangular box for conclusions (shifted right 75 points) - INCREASED HEIGHT (same position)
    # Height increased: bottom extended down (top position same). Length increased by 20 (x position fixed)
    # Rect already imported at top
    conclusion_box = Rect(450, conclusion_y_start - 55, 355, 75,  # Shifted right 75 points from 375 to 450
                         fillColor=None, strokeColor=colors.black, strokeWidth=1.5)
    master_drawing.add(conclusion_box)
    
    # CENTERED and STYLISH "Conclusion" header - DYNAMIC - SMALLER (AT TOP OF CONTAINER - CLOSE TO TOP LINE)
    # Box center: 450 + (355/2) = 627.5, so text should be centered around 627.5
    # Box top is at conclusion_y_start - 55, so header should be very close to top line
    conclusion_header = String(627.5, conclusion_y_start + 8, "CONCLUSION",  # Centered in shifted box
                              fontSize=9, fontName="Helvetica-Bold",  # Reduced from 11 to 9
                              fillColor=colors.HexColor("#2c3e50"),
                              textAnchor="middle")  # This centers the text
    master_drawing.add(conclusion_header)
    
    # DYNAMIC conclusions from dashboard in the box - SINGLE COLUMN to avoid overlapping
    print(f" Drawing conclusions in graph from filtered list: {filtered_conclusions}")
    
    # Draw conclusions vertically in a single column
    row_spacing = 10  # Increased vertical spacing
    start_y = conclusion_y_start - 12  # Starting Y position (further down from top)
    box_bottom = conclusion_y_start - 55  # Bottom edge of the box
    
    for idx, conclusion in enumerate(filtered_conclusions):
        row_y = start_y - (idx * row_spacing)
        
        # User request: If getting cropped (exceeds box height), don't put in this.
        if row_y < box_bottom + 5:  # 5 points padding from bottom
            print(f" Skipping conclusion {idx+1} as it would be cropped")
            continue
            
        conc_text = f"{idx + 1}. {conclusion}"
        
        # Position horizontally in a single column - shifted right
        x_pos = 460  # Align with the box's left side
        
        conc = String(x_pos, row_y, conc_text, 
                     fontSize=9, fontName="Helvetica", fillColor=colors.black)
        master_drawing.add(conc)
            
    print(f" Added {len(filtered_conclusions)} REAL Conclusions in single column (no cropping)")
    
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

    # Helper: draw logo on every page AND ALIGNED pink grid background on Page 2
    def _draw_logo_and_footer(canvas, doc):
        import os
        from reportlab.lib.units import mm
        
        # STEP 1: Draw FULL PAGE pink ECG grid background on Page 2 (ECG graphs page)
        if canvas.getPageNumber() == 1:  # Changed from 3 to 2
            page_width, page_height = canvas._pagesize
            
            # Fill entire page with pink background
            canvas.setFillColor(colors.HexColor("#ffe6e6"))
            canvas.rect(0, 0, page_width, page_height, fill=1, stroke=0)
            
            # ECG grid colors - darker for better visibility
            light_grid_color = colors.HexColor("#ffd1d1")  
            
            major_grid_color = colors.HexColor("#ffb3b3")   
            
            # Draw minor grid lines (1mm spacing) - 59 boxes complete (0 to 295mm)
            canvas.setStrokeColor(light_grid_color)
            canvas.setLineWidth(0.6)
            
            minor_spacing = 1 * mm
            
            # Vertical minor lines - Draw up to 295mm (includes 295mm line)
            max_x_limit = 59 * 5 * mm  # 295mm = right edge of 59th box
            x = 0
            while x <= max_x_limit:  # Draw lines 0 to 295mm (complete 59 boxes)
                canvas.line(x, 0, x, page_height)
                x += minor_spacing
                if x > max_x_limit:  # Stop immediately after 295mm
                    break
            
            # Horizontal minor lines - full page
            y = 0
            while y <= page_height:
                canvas.line(0, y, page_width, y)
                y += minor_spacing
                
            
            # Draw major grid lines - FULL PAGE
            # IMPORTANT: Match waveform calculation: height/3 = 15.0 points per box
            # For individual lead graphs: ecg_height = 45 points, so 15 points = 1 box
            canvas.setStrokeColor(major_grid_color)
            canvas.setLineWidth(1.2)
            
            # Use standard ECG paper spacing: 5mm per box
            # 5mm = 5 * 2.834645669 points = 14.17 points per box
            from reportlab.lib.units import mm
            major_spacing = 5 * mm  # Standard ECG: 5mm = 14.17 points per box
            
            # Vertical major lines - Draw 60 lines (0, 5, 10...295mm) for 59 complete boxes
            max_x_limit = 59 * 5 * mm  # 295mm = right edge of 59th box
            x = 0
            while x <= max_x_limit:  # Include 295mm line (completes 59 boxes)
                canvas.line(x, 0, x, page_height)
                x += major_spacing
                if x > max_x_limit:  # Stop after 295mm
                    break
            
            # Horizontal major lines - STRICT: Only up to 295mm width (not full page_width)
            y = 0
            while y <= page_height:
                canvas.line(0, y, max_x_limit, y)  # End at 295mm (not page_width) ✅
                y += major_spacing
            

        
        # STEP 1.5: Draw Org. and Phone No. labels on Page 1 (TOP LEFT)
        if canvas.getPageNumber() == 1:
            canvas.saveState()
            
            # Position in top-left corner (below margin)
            x_pos = doc.leftMargin  # 30 points from left
            y_pos = doc.height + doc.bottomMargin - 5  # 20 points from top
            
            # Always draw "Org." label with value
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.black)
            org_label = "Org:"
            canvas.drawString(x_pos, y_pos, org_label)
            
            # Calculate width of label and add small gap
            org_label_width = canvas.stringWidth(org_label, "Helvetica-Bold", 9)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(x_pos + org_label_width + 5, y_pos, patient.get("Org.", "") if patient else "")
            
            y_pos -= 15  # Move down for next line
            
            # Always draw "Phone No." label with value
            canvas.setFont("Helvetica-Bold", 9)
            canvas.setFillColor(colors.black)
            phone_label = "Phone No:"
            canvas.drawString(x_pos, y_pos, phone_label)
            
            # Calculate width of label and add small gap
            phone_label_width = canvas.stringWidth(phone_label, "Helvetica-Bold", 9)
            canvas.setFont("Helvetica", 9)
            canvas.drawString(x_pos + phone_label_width + 5, y_pos, patient.get("doctor_mobile", "") if patient else "")
            
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
        footer_text = "Deckmount Electronics, Plot No. 683, Phase V, Udyog Vihar, Sector 19, Gurugram, Haryana 122016"
        # Center the footer text at bottom of page
        text_width = canvas.stringWidth(footer_text, "Helvetica", 8)
        x = (doc.width + doc.leftMargin + doc.rightMargin - text_width) / 2
        y = 5  # Shifted down 5 points from 10 to 5  # 20 points from bottom
        canvas.drawString(x, y, footer_text)
        canvas.restoreState()

    # Save parameters to a JSON index for later reuse
    try:
        from datetime import datetime
        reports_dir = str(data_file("reports"))
        os.makedirs(reports_dir, exist_ok=True)
        index_path = os.path.join(reports_dir, 'index.json')
        metrics_path = os.path.join(reports_dir, 'metrics.json')

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
                "QTCF": round(qtcf_val, 1) if 'qtcf_val' in locals() and qtcf_val and qtcf_val > 0 else None,
            }
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

    # Build PDF
    doc.build(story)
    print(f" ECG Report generated: {filename}")
    
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
        
        dispatch_12lead_report(
            data=data,
            patient=patient or {},
            pdf_path=filename,
            settings_manager=settings_manager if 'settings_manager' in locals() else None,
            signup_details={},
            ecg_test_page=ecg_test_page if 'ecg_test_page' in locals() else None,
            ecg_data_file=saved_data_file_path if 'saved_data_file_path' in locals() else (ecg_data_file if 'ecg_data_file' in locals() else None),
            report_format="6_2",
            conclusions=filtered_conclusions if 'filtered_conclusions' in locals() else None,
            arrhythmia=None,
        )
        print("  Dispatched 6-2 unified payload")
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
            # Upload the complete package
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
                print(f"  Cloud upload failed: {result.get('message', 'Unknown error')}")
        else:
            print("  Cloud upload not configured (see cloud_config_template.txt)")
            
    except ImportError:
        print("  Cloud uploader not available")
    except Exception as e:
        print(f"  Cloud upload error: {e}")
    
    return filename, metrics_entry


# ====================  REPORT WRAPPER 6_2 ====================
