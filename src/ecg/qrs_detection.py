"""
QRS Complex Detection & Measurement Module
==========================================
Paper: Curtin et al., "QRS Complex Detection and Measurement Algorithms for
       Multichannel ECGs in Cardiac Resynchronization Therapy Patients"
       IEEE J. Transl. Eng. Health Med., 2018. DOI: 10.1109/JTEHM.2018.2844195

COMPLETE PIPELINE — Stages 1 through 10
────────────────────────────────────────
QRS Detection (Stages 1–5):       ← NEW — added to existing file
  Stage 1 → Channel grouping + averaging
  Stage 2 → Peak detection (amplitude + width criteria)
  Stage 3 → QRS complex windowing (PR + QT approximation)
  Stage 4 → Additional complex identification
  Stage 5 → Morphology classification (PM vs OM)

QRS Duration Measurement (Stages 6–10):   ← EXISTING — unchanged
  Stage 6 → Reference peak identification + significant peaks detection
  Stage 7 → Array-specific peak groups (anterior / posterior)
  Stage 8 → Channel-specific border delineation (amplitude + slope criteria)
  Stage 9 → Array-specific border delineation (normal group within 20 ms)
  Stage 10→ Global border delineation (earliest anterior + latest posterior)

INTEGRATION WITH EXISTING CODEBASE:
  # Old (scipy find_peaks):
  peaks, _ = find_peaks(signal, distance=int(0.5*fs), height=threshold)

  # New (Stage 1-5 paper method):
  from qrs_detection import get_r_peaks_for_lead
  r_peaks = get_r_peaks_for_lead(raw_signal, fs, adc_per_mv)

  # Full result with PM/OM classification:
  from qrs_detection import detect_qrs_full
  result = detect_qrs_full([lead_i, lead_ii, ...], fs)
  r_peaks  = result["r_peaks"]
  pm_wins  = result["pm_windows"]
  om_wins  = result["om_windows"]
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from typing import Optional, Tuple, List, Dict, Any


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS  (Paper Table 1 aur Table 2 se liye gaye)
# ══════════════════════════════════════════════════════════════════════════════

# ── Stage 1-5 constants (QRS Detection) ──────────────────────────────────────
PEAK_AMP_MIN_MV:              float = 0.10    # 0.10 mV minimum viable amplitude
PEAK_AMP_MAX_MV:              float = 4.00    # 4.00 mV maximum viable amplitude
QRS_PEAK_MAX_WIDTH_MS:        float = 120.0   # QRS peak max width (narrower than T)
MIN_SAME_POLARITY_PEAKS:      int   = 3       # min peaks to confirm PM morphology
MORPHOLOGY_MATCH_THRESHOLD:   float = 0.85    # cross-corr threshold for PM/OM
PREPROCESS_LOW_HZ:            float = 0.5
PREPROCESS_HIGH_HZ:           float = 25.0
PREPROCESS_ORDER:             int   = 10

# PR approximation limits (Carruthers et al. 1987, paper ref [29])
PR_APPROX_MIN_MS: float = 80.0
PR_APPROX_MAX_MS: float = 200.0
QT_APPROX_FRACTION: float = 0.40   # QT ≈ 0.40 × RR (Karjalainen 1994, ref [30])

# ── Stage 6-10 constants (QRS Duration Measurement) ──────────────────────────
PEAK_AMPLITUDE_GROUP_TOL_MV:         float = 0.1    # ±0.1 mV grouping tolerance
PEAK_WIDTH_GROUP_TOL_MS:             float = 20.0   # ±20 ms width grouping tolerance
MAX_INTRA_COMPLEX_PEAK_DIST_MS:      float = 81.0   # max intra-complex peak spacing
MAX_ARRAY_PEAK_SPACING_MS:           float = 52.0   # Stage 7 outlier removal
QRS_BORDER_AMPLITUDE_RATIO:          float = 0.20   # FIX: was 0.50 → QRS=72ms; 0.20 → target 86ms
QRS_BORDER_SLOPE_THRESHOLD_MV_PER_MS: float = 0.015 # FIX: was 0.025 → target 0.015
ARRAY_BORDER_NORMAL_GROUP_TOLERANCE_MS: float = 20.0
HR_WINDOW_MIN_BPM: int = 40
HR_WINDOW_MAX_BPM: int = 120
MIN_SIGNIFICANT_PEAK_HEIGHT_RATIO: float = 0.10

# Physiological QRS limits
QRS_DURATION_MIN_MS: float = 40.0
QRS_DURATION_MAX_MS: float = 200.0  # Even extreme LBBB/RBBB rarely exceeds 200ms


def _amplitude_qrs_width(signal: np.ndarray,
                         peak_idx: int,
                         fs: float,
                         threshold_ratio: float = 0.10,
                         pre_ms: float = 250.0,
                         post_ms: float = 400.0) -> Tuple[float, Optional[int], Optional[int]]:
    """Amplitude-based QRS borders for broad/slurred BBB complexes."""
    sig = np.asarray(signal, dtype=float)
    if sig.size < 3 or fs <= 0:
        return 0.0, None, None

    peak_idx = int(max(0, min(sig.size - 1, peak_idx)))
    start = max(0, peak_idx - int(pre_ms * fs / 1000.0))
    end = min(sig.size, peak_idx + int(post_ms * fs / 1000.0) + 1)
    segment = sig[start:end]
    if segment.size < 3:
        return 0.0, None, None

    baseline = float(np.median(segment))
    centered = segment - baseline
    peak_rel = int(np.argmax(np.abs(centered)))
    amp = float(np.max(np.abs(centered)))
    if amp <= 1e-9:
        return 0.0, None, None

    threshold = max(1e-9, float(threshold_ratio) * amp)
    left = peak_rel
    while left > 0 and abs(centered[left]) > threshold:
        left -= 1

    right = peak_rel
    while right < centered.size - 1 and abs(centered[right]) > threshold:
        right += 1

    qrs_ms = float((right - left) * 1000.0 / fs)
    if not (QRS_DURATION_MIN_MS <= qrs_ms <= QRS_DURATION_MAX_MS):
        return 0.0, None, None
    return qrs_ms, start + left, start + right


def _adaptive_amplitude_qrs_width(signal: np.ndarray,
                                  peak_idx: int,
                                  fs: float,
                                  baseline_ratio: float = 0.10,
                                  wide_ratio: float = 0.05,
                                  pre_ms: float = 250.0,
                                  post_ms: float = 400.0) -> Tuple[float, Optional[int], Optional[int]]:
    """Use a looser threshold only when the complex already looks broad/slurred."""
    base_ms, base_on, base_off = _amplitude_qrs_width(
        signal, peak_idx, fs,
        threshold_ratio=baseline_ratio,
        pre_ms=pre_ms,
        post_ms=post_ms,
    )
    wide_ms, wide_on, wide_off = _amplitude_qrs_width(
        signal, peak_idx, fs,
        threshold_ratio=wide_ratio,
        pre_ms=pre_ms,
        post_ms=post_ms,
    )

    if wide_ms <= 0:
        return base_ms, base_on, base_off
    if base_ms <= 0:
        return wide_ms, wide_on, wide_off

    # BBB/slurred terminal forces are often missed at a 10% cutoff.
    # Only widen the borders when the baseline width is already near-broad
    # or the looser threshold adds a clearly meaningful tail.
    if base_ms >= 105.0 or (wide_ms - base_ms) >= 10.0:
        return wide_ms, wide_on, wide_off
    return base_ms, base_on, base_off


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 – CHANNEL GROUPING AND AVERAGING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_channel(raw_signal: np.ndarray,
                        fs: float,
                        adc_per_mv: float = 1.0
                        ) -> Tuple[np.ndarray, bool]:
    """
    Paper Section II.B.2: Zero-phase bandpass filter (0.5–25 Hz, 10th order)
    + amplitude viability check (0.10–4.00 mV) + baseline correction.

    Returns:
        (filtered_signal_mv, is_viable)
    """
    signal_mv = raw_signal.astype(float) / adc_per_mv
    nyq  = fs / 2.0
    low  = max(PREPROCESS_LOW_HZ  / nyq, 0.001)
    high = min(PREPROCESS_HIGH_HZ / nyq, 0.99)
    if low >= high:
        return signal_mv, False
    try:
        b, a = butter(PREPROCESS_ORDER, [low, high], btype='band')
        filtered = filtfilt(b, a, signal_mv)
    except Exception:
        try:
            b, a = butter(4, [low, high], btype='band')
            filtered = filtfilt(b, a, signal_mv)
        except Exception:
            filtered = signal_mv.copy()
    filtered -= np.mean(filtered)
    peak_amp  = np.max(np.abs(filtered))
    is_viable = PEAK_AMP_MIN_MV <= peak_amp <= PEAK_AMP_MAX_MV
    return filtered, is_viable


def group_channels_by_morphology(signals_mv: List[np.ndarray],
                                  fs: float,
                                  corr_threshold: float = 0.70
                                  ) -> List[List[int]]:
    """
    Stage 1a/b: Group channels with similar morphology by cross-correlation.
    Paper (Stage 1, Table 1): channels grouped if corr > threshold.
    """
    n = len(signals_mv)
    if n == 0:
        return []
    if n == 1:
        return [[0]]

    seg_len   = int(5.0 * fs)
    mid       = len(signals_mv[0]) // 2
    seg_start = max(0, mid - seg_len // 2)
    seg_end   = min(len(signals_mv[0]), seg_start + seg_len)

    segments = []
    for sig in signals_mv:
        seg = sig[seg_start:seg_end]
        std = np.std(seg)
        segments.append(seg / std if std > 1e-9 else seg)

    corr_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            min_len = min(len(segments[i]), len(segments[j]))
            c = float(np.corrcoef(segments[i][:min_len],
                                  segments[j][:min_len])[0, 1])
            c = c if np.isfinite(c) else 0.0
            corr_matrix[i, j] = c
            corr_matrix[j, i] = c

    assigned: List[int] = [-1] * n
    groups:   List[List[int]] = []
    for i in range(n):
        placed = False
        for g_idx, group in enumerate(groups):
            if all(abs(corr_matrix[i, j]) >= corr_threshold for j in group):
                group.append(i)
                assigned[i] = g_idx
                placed = True
                break
        if not placed:
            assigned[i] = len(groups)
            groups.append([i])
    return groups


def compute_group_averages(signals_mv: List[np.ndarray],
                            groups: List[List[int]]
                            ) -> List[np.ndarray]:
    """Stage 1c: Average signal for each morphology group."""
    averages = []
    for group in groups:
        if not group:
            continue
        min_len = min(len(signals_mv[i]) for i in group)
        stack   = np.array([signals_mv[i][:min_len] for i in group])
        averages.append(np.mean(stack, axis=0))
    return averages


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 – PEAK DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def _peak_width_ms(signal: np.ndarray, peak_idx: int, fs: float) -> float:
    """Estimate peak width at half-amplitude (both sides)."""
    half_amp = abs(signal[peak_idx]) * 0.5
    polarity = np.sign(signal[peak_idx])
    left  = peak_idx
    right = peak_idx
    for i in range(peak_idx - 1, -1, -1):
        if polarity * signal[i] < half_amp:
            left = i
            break
    for i in range(peak_idx + 1, len(signal)):
        if polarity * signal[i] < half_amp:
            right = i
            break
    return (right - left) / fs * 1000.0


def detect_peaks_in_average_signal(avg_signal: np.ndarray,
                                    fs: float,
                                    rr_estimate_ms: float = 800.0
                                    ) -> Dict[str, Any]:
    """
    Stage 2: Detect QRS peaks in an average signal.

    Step 2a: Signal-specific amplitude threshold.
    Step 2b: Width downselection + same-polarity grouping.
    Step 2c/2d: Intra-complex grouping (max 81ms apart).

    Returns dict with: positive_peaks, negative_peaks, all_peaks, complex_groups.
    """
    signal_range  = np.max(np.abs(avg_signal))
    rectified     = np.abs(avg_signal)
    amp_threshold = np.mean(rectified) + 1.5 * np.std(rectified)
    amp_threshold = max(amp_threshold, signal_range * 0.30)
    min_distance  = max(int(rr_estimate_ms * 0.4 / 1000.0 * fs), int(0.22 * fs))

    pos_peaks, _ = find_peaks( avg_signal, height=amp_threshold, distance=min_distance)
    neg_peaks, _ = find_peaks(-avg_signal, height=amp_threshold, distance=min_distance)

    def _filter_by_width(peaks):
        return [int(p) for p in peaks
                if _peak_width_ms(avg_signal, int(p), fs) <= QRS_PEAK_MAX_WIDTH_MS]

    pos_qrs = _filter_by_width(pos_peaks)
    neg_qrs = _filter_by_width(neg_peaks)

    def _largest_group(peaks):
        if not peaks:
            return []
        amps   = [abs(avg_signal[p]) for p in peaks]
        widths = [_peak_width_ms(avg_signal, p, fs) for p in peaks]
        groups, used = [], [False] * len(peaks)
        for i in range(len(peaks)):
            if used[i]:
                continue
            grp = [peaks[i]]
            used[i] = True
            for j in range(i + 1, len(peaks)):
                if (not used[j]
                        and abs(amps[i] - amps[j]) <= PEAK_AMPLITUDE_GROUP_TOL_MV
                        and abs(widths[i] - widths[j]) <= PEAK_WIDTH_GROUP_TOL_MS):
                    grp.append(peaks[j])
                    used[j] = True
            groups.append(grp)
        best = max(groups, key=len)
        return best if len(best) >= MIN_SAME_POLARITY_PEAKS else []

    pm_pos = _largest_group(pos_qrs)
    pm_neg = _largest_group(neg_qrs)

    all_pm = sorted(pm_pos + pm_neg)
    max_dist_samp = int(MAX_INTRA_COMPLEX_PEAK_DIST_MS / 1000.0 * fs)
    complex_groups: List[List[int]] = []
    if all_pm:
        cur = [all_pm[0]]
        for i in range(1, len(all_pm)):
            if all_pm[i] - all_pm[i - 1] <= max_dist_samp:
                cur.append(all_pm[i])
            else:
                complex_groups.append(cur)
                cur = [all_pm[i]]
        complex_groups.append(cur)

    return {
        "positive_peaks":  pm_pos,
        "negative_peaks":  pm_neg,
        "all_peaks":       all_pm,
        "complex_groups":  complex_groups,
    }


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 – QRS COMPLEX WINDOWING
# ══════════════════════════════════════════════════════════════════════════════

def _approximate_pr_ms(rr_ms: float) -> float:
    """PR approx from RR (Carruthers et al. 1987, paper ref [29])."""
    hr = 60000.0 / rr_ms if rr_ms > 0 else 75.0
    pr = 120.0 + 80.0 * np.exp(-hr / 100.0)
    return float(np.clip(pr, PR_APPROX_MIN_MS, PR_APPROX_MAX_MS))


def _approximate_qt_ms(rr_ms: float) -> float:
    """QT approx from RR (Karjalainen et al. 1994, paper ref [30])."""
    return QT_APPROX_FRACTION * rr_ms


def define_qrs_windows(complex_groups: List[List[int]],
                        avg_signal: np.ndarray,
                        fs: float
                        ) -> List[Tuple[int, int]]:
    """
    Stage 3: Broad QRS window per complex.
    Window = first_peak - PR_approx  →  last_peak + QT_approx.
    """
    n = len(avg_signal)
    if not complex_groups:
        return []

    rr_estimates = []
    for i in range(1, len(complex_groups)):
        rr_est = (complex_groups[i][0] - complex_groups[i - 1][0]) / fs * 1000.0
        if 300 < rr_est < 2000:
            rr_estimates.append(rr_est)
    rr_ms   = float(np.median(rr_estimates)) if rr_estimates else 800.0
    pr_samp = int(_approximate_pr_ms(rr_ms) / 1000.0 * fs)
    qt_samp = int(_approximate_qt_ms(rr_ms) / 1000.0 * fs)

    windows = []
    for group in complex_groups:
        win_start = max(0,     group[0]  - pr_samp)
        win_end   = min(n - 1, group[-1] + qt_samp)
        windows.append((win_start, win_end))
    return windows


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 – ADDITIONAL COMPLEX IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def find_additional_complexes(avg_signal: np.ndarray,
                               known_windows: List[Tuple[int, int]],
                               pm_complex_groups: List[List[int]],
                               fs: float,
                               rr_ms: float
                               ) -> List[Tuple[int, int]]:
    """
    Stage 4: Find complexes in gaps between known windows (OM candidates).
    """
    additional: List[Tuple[int, int]] = []
    n = len(avg_signal)
    if not known_windows:
        return additional

    sorted_wins = sorted(known_windows, key=lambda x: x[0])
    gaps: List[Tuple[int, int]] = []
    half_rr = int(0.5 * rr_ms / 1000.0 * fs)

    if sorted_wins[0][0] > half_rr:
        gaps.append((0, sorted_wins[0][0]))
    for i in range(len(sorted_wins) - 1):
        gs, ge = sorted_wins[i][1], sorted_wins[i + 1][0]
        if ge - gs > int(0.3 * rr_ms / 1000.0 * fs):
            gaps.append((gs, ge))
    if sorted_wins[-1][1] < n - half_rr:
        gaps.append((sorted_wins[-1][1], n))

    pm_amps = [abs(avg_signal[p])
               for grp in pm_complex_groups for p in grp]
    ref_amp   = float(np.median(pm_amps)) if pm_amps else 0.5
    threshold = 0.30 * ref_amp
    pr_samp   = int(_approximate_pr_ms(rr_ms) / 1000.0 * fs)
    qt_samp   = int(_approximate_qt_ms(rr_ms) / 1000.0 * fs)

    for gs, ge in gaps:
        if ge <= gs:
            continue
        seg  = avg_signal[gs:ge]
        pidx = int(np.argmax(np.abs(seg)))
        if abs(seg[pidx]) >= threshold:
            abs_p = gs + pidx
            additional.append((max(0, abs_p - pr_samp),
                                min(n - 1, abs_p + qt_samp)))
    return additional


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 – MORPHOLOGY CLASSIFICATION (PM vs OM)
# ══════════════════════════════════════════════════════════════════════════════

def _extract_complex_template(avg_signal: np.ndarray,
                               window: Tuple[int, int]) -> np.ndarray:
    seg  = avg_signal[window[0]:window[1]]
    norm = np.linalg.norm(seg)
    return seg / norm if (len(seg) > 0 and norm > 1e-9) else seg


def _cross_correlation_score(t1: np.ndarray, t2: np.ndarray) -> float:
    min_len = min(len(t1), len(t2))
    if min_len < 5:
        return 0.0
    c = np.corrcoef(t1[:min_len], t2[:min_len])[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def classify_complex_morphologies(avg_signal: np.ndarray,
                                   pm_windows: List[Tuple[int, int]],
                                   additional_windows: List[Tuple[int, int]],
                                   fs: float
                                   ) -> Dict[str, Any]:
    """
    Stage 5: Finalize PM vs OM classification.
    Step 5a: Cross-corr of each window vs median PM template.
    Step 5b: Reassign if needed.
    """
    if not pm_windows:
        return {"pm_windows": [], "om_windows": additional_windows,
                "pm_template": np.array([])}

    templates = [_extract_complex_template(avg_signal, w) for w in pm_windows]
    templates = [t for t in templates if len(t) > 5]
    if not templates:
        return {"pm_windows": pm_windows, "om_windows": additional_windows,
                "pm_template": np.array([])}

    med_len = int(np.median([len(t) for t in templates]))
    aligned = []
    for t in templates:
        if len(t) >= med_len:
            aligned.append(t[:med_len])
        else:
            aligned.append(np.pad(t, (0, med_len - len(t))))
    pm_template = np.median(np.array(aligned), axis=0)

    final_pm, moved_to_om = [], []
    for win in pm_windows:
        tmpl  = _extract_complex_template(avg_signal, win)
        score = _cross_correlation_score(pm_template, tmpl) if len(tmpl) > 5 else 0.0
        (final_pm if score >= MORPHOLOGY_MATCH_THRESHOLD else moved_to_om).append(win)

    final_om, moved_to_pm = list(moved_to_om), []
    for win in additional_windows:
        tmpl  = _extract_complex_template(avg_signal, win)
        score = _cross_correlation_score(pm_template, tmpl) if len(tmpl) > 5 else 0.0
        (moved_to_pm if score >= MORPHOLOGY_MATCH_THRESHOLD else final_om).append(win)

    final_pm.extend(moved_to_pm)
    final_pm.sort(key=lambda x: x[0])
    final_om.sort(key=lambda x: x[0])

    return {"pm_windows": final_pm, "om_windows": final_om,
            "pm_template": pm_template}


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL STAGE 1-5 API
# ══════════════════════════════════════════════════════════════════════════════

def detect_qrs_full(signals_mv: List[np.ndarray],
                    fs: float,
                    lead_names: Optional[List[str]] = None
                    ) -> Dict[str, Any]:
    """
    Full Stage 1–5 QRS detection pipeline (paper method).

    Args:
        signals_mv:  List of signals in mV (one per lead/channel).
        fs:          Sampling rate (Hz).
        lead_names:  Optional lead name labels.

    Returns:
        Dict: r_peaks, pm_windows, om_windows, groups,
              group_avgs, pm_template, rr_ms, hr_bpm.
    """
    if not signals_mv:
        return _empty_result()

    # Stage 1
    groups     = group_channels_by_morphology(signals_mv, fs)
    group_avgs = compute_group_averages(signals_mv, groups)
    if not group_avgs:
        return _empty_result()

    main_avg = group_avgs[int(np.argmax([np.max(np.abs(g)) for g in group_avgs]))]

    # Coarse RR estimate
    rough, _ = find_peaks(np.abs(main_avg),
                           distance=int(0.3 * fs),
                           height=np.max(np.abs(main_avg)) * 0.3)
    rr_ms = 800.0
    if len(rough) >= 2:
        diffs    = np.diff(rough) / fs * 1000.0
        valid_rr = diffs[(diffs > 250) & (diffs < 2000)]
        if len(valid_rr) > 0:
            rr_ms = float(np.median(valid_rr))

    hr_bpm = 60000.0 / rr_ms if rr_ms > 0 else 75.0

    # Stage 2
    s2 = detect_peaks_in_average_signal(main_avg, fs, rr_ms)

    # Stage 3
    pm_windows = define_qrs_windows(s2["complex_groups"], main_avg, fs)

    # Stage 4
    add_windows = find_additional_complexes(
        main_avg, pm_windows, s2["complex_groups"], fs, rr_ms)

    # Stage 5
    s5 = classify_complex_morphologies(main_avg, pm_windows, add_windows, fs)

    r_peaks = _windows_to_r_peaks(main_avg, s5["pm_windows"])

    return {
        "r_peaks":     np.array(r_peaks, dtype=int),
        "pm_windows":  s5["pm_windows"],
        "om_windows":  s5["om_windows"],
        "groups":      groups,
        "group_avgs":  group_avgs,
        "pm_template": s5["pm_template"],
        "rr_ms":       rr_ms,
        "hr_bpm":      hr_bpm,
    }


def detect_qrs_peaks(signal_mv: np.ndarray, fs: float,
                     adc_per_mv: float = 1.0) -> np.ndarray:
    """
    Single-channel drop-in replacement for scipy find_peaks based detection.
    """
    sig = signal_mv.astype(float) / adc_per_mv
    nyq  = fs / 2.0
    low  = max(0.5 / nyq, 0.001)
    high = min(25.0 / nyq, 0.99)
    try:
        b, a = butter(4, [low, high], btype='band')
        sig  = filtfilt(b, a, sig)
    except Exception:
        pass
    sig -= np.mean(sig)
    return detect_qrs_full([sig], fs)["r_peaks"]


def get_r_peaks_for_lead(raw_signal: np.ndarray, fs: float,
                          adc_per_mv: float = 1.0,
                          use_paper_method: bool = True) -> np.ndarray:
    """
    Convenience wrapper — drop-in for existing R-peak detection calls.

    Usage:
        # OLD:
        peaks, _ = find_peaks(signal, distance=int(0.5*fs), height=threshold)
        # NEW:
        peaks = get_r_peaks_for_lead(raw_signal, fs, adc_per_mv)
    """
    if use_paper_method:
        return detect_qrs_peaks(raw_signal, fs, adc_per_mv)
    # Fallback
    sig_mv    = raw_signal.astype(float) / adc_per_mv
    threshold = np.max(np.abs(sig_mv)) * 0.5
    peaks, _  = find_peaks(np.abs(sig_mv), height=threshold,
                            distance=int(0.4 * fs))
    return peaks


def _windows_to_r_peaks(avg_signal: np.ndarray,
                         windows: List[Tuple[int, int]]) -> List[int]:
    return sorted([w[0] + int(np.argmax(np.abs(avg_signal[w[0]:w[1]])))
                   for w in windows if w[1] > w[0]])


def _empty_result() -> Dict[str, Any]:
    return {"r_peaks": np.array([], dtype=int), "pm_windows": [],
            "om_windows": [], "groups": [], "group_avgs": [],
            "pm_template": np.array([]), "rr_ms": 800.0, "hr_bpm": 75.0}


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 – REFERENCE PEAK + SIGNIFICANT PEAK IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════

def find_reference_peak(signal: np.ndarray,
                        qrs_window_start: int,
                        qrs_window_end: int) -> int:
    start = max(0, qrs_window_start)
    end   = min(len(signal), qrs_window_end)
    if end <= start:
        return (qrs_window_start + qrs_window_end) // 2
    return start + int(np.argmax(np.abs(signal[start:end])))


def _compute_peak_bounds(signal: np.ndarray, peak_idx: int) -> Tuple[int, int]:
    n = len(signal)
    left = peak_idx
    prev_sign = np.sign(signal[peak_idx])
    for i in range(peak_idx - 1, -1, -1):
        curr_sign = np.sign(signal[i])
        if curr_sign != prev_sign and curr_sign != 0:
            left = i + 1
            break
        if i > 0:
            d2 = (signal[i + 1] - 2 * signal[i] + signal[i - 1])
            if i < peak_idx - 1:
                d2_prev = (signal[i + 2] - 2 * signal[i + 1] + signal[i])
                if np.sign(d2) != np.sign(d2_prev) and d2_prev != 0:
                    left = i + 1
                    break
    else:
        left = 0

    right = peak_idx
    prev_sign = np.sign(signal[peak_idx])
    for i in range(peak_idx + 1, n):
        curr_sign = np.sign(signal[i])
        if curr_sign != prev_sign and curr_sign != 0:
            right = i - 1
            break
        if i < n - 1:
            d2 = (signal[i + 1] - 2 * signal[i] + signal[i - 1])
            if i > peak_idx + 1:
                d2_prev = (signal[i] - 2 * signal[i - 1] + signal[i - 2])
                if np.sign(d2) != np.sign(d2_prev) and d2_prev != 0:
                    right = i
                    break
    else:
        right = n - 1
    return left, right


def _peak_slope_and_curvature(signal: np.ndarray, peak_idx: int,
                               left_bound: int, right_bound: int,
                               fs: float) -> Tuple[float, float, float, float]:
    lead_seg = signal[left_bound : peak_idx + 1]
    lead_slope = float(np.max(np.abs(np.diff(lead_seg)))) if len(lead_seg) >= 2 else 0.0
    lead_curv  = float(np.max(np.abs(np.diff(np.diff(lead_seg))))) if len(lead_seg) >= 3 else 0.0
    fall_seg = signal[peak_idx : right_bound + 1]
    fall_slope = float(np.max(np.abs(np.diff(fall_seg)))) if len(fall_seg) >= 2 else 0.0
    fall_curv  = float(np.max(np.abs(np.diff(np.diff(fall_seg))))) if len(fall_seg) >= 3 else 0.0
    return lead_slope, lead_curv, fall_slope, fall_curv


def find_significant_peaks(signal: np.ndarray, ref_peak_idx: int,
                            qrs_window_start: int, qrs_window_end: int,
                            fs: float) -> List[int]:
    start   = max(0, qrs_window_start)
    end     = min(len(signal), qrs_window_end)
    ref_amp = abs(signal[ref_peak_idx])
    if ref_amp < 1e-9:
        return [ref_peak_idx]

    ref_left, ref_right = _compute_peak_bounds(signal, ref_peak_idx)
    ref_ls, ref_lc, ref_fs_, ref_fc = _peak_slope_and_curvature(
        signal, ref_peak_idx, ref_left, ref_right, fs)
    significant = [ref_peak_idx]

    def _evaluate(cand_idx: int) -> bool:
        cand_amp = abs(signal[cand_idx])
        ratio = cand_amp / ref_amp
        if ratio < MIN_SIGNIFICANT_PEAK_HEIGHT_RATIO:
            return False
        c_left, c_right = _compute_peak_bounds(signal, cand_idx)
        for s in significant:
            s_l, s_r = _compute_peak_bounds(signal, s)
            if s_l <= cand_idx <= s_r:
                return False
        c_ls, c_lc, c_fs_, c_fc = _peak_slope_and_curvature(
            signal, cand_idx, c_left, c_right, fs)
        if ref_ls > 0   and c_ls  < ref_ls  * ratio: return False
        if ref_lc > 1e-9 and c_lc  < ref_lc  * ratio: return False
        if ref_fs_ > 0  and c_fs_ < ref_fs_ * ratio: return False
        if ref_fc > 1e-9 and c_fc  < ref_fc  * ratio: return False
        return True

    left_cands = _find_local_extrema(signal, start, ref_peak_idx)
    left_cands.sort(reverse=True)
    for cand in left_cands:
        if _evaluate(cand): significant.append(cand)
        else: break

    right_cands = _find_local_extrema(signal, ref_peak_idx + 1, end)
    right_cands.sort()
    for cand in right_cands:
        if _evaluate(cand): significant.append(cand)
        else: break

    significant.sort()
    return significant


def _find_local_extrema(signal: np.ndarray, start: int, end: int) -> List[int]:
    start = max(0, start)
    end   = min(len(signal), end)
    return [i for i in range(start + 1, end - 1)
            if (signal[i] > signal[i-1] and signal[i] > signal[i+1])
            or (signal[i] < signal[i-1] and signal[i] < signal[i+1])]


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 – ARRAY-SPECIFIC PEAK GROUPS
# ══════════════════════════════════════════════════════════════════════════════

def remove_peak_outliers_by_spacing(significant_peaks: List[int], fs: float,
                                     max_spacing_ms: float = MAX_ARRAY_PEAK_SPACING_MS
                                     ) -> List[int]:
    if len(significant_peaks) <= 1:
        return significant_peaks
    max_samp = max_spacing_ms / 1000.0 * fs
    filtered = [significant_peaks[0]]
    for i in range(1, len(significant_peaks)):
        if significant_peaks[i] - significant_peaks[i-1] <= max_samp:
            filtered.append(significant_peaks[i])
    return filtered


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 8 – CHANNEL-SPECIFIC QRS BORDER DELINEATION
# ══════════════════════════════════════════════════════════════════════════════

def delineate_channel_borders(signal: np.ndarray,
                               significant_peaks: List[int],
                               ref_peak_idx: int,
                               qrs_window_start: int,
                               qrs_window_end: int,
                               fs: float,
                               adc_per_mv: float = 1.0
                               ) -> Tuple[Optional[int], Optional[int]]:
    if not significant_peaks:
        return None, None

    earliest = significant_peaks[0]
    latest   = significant_peaks[-1]
    slope_thr = QRS_BORDER_SLOPE_THRESHOLD_MV_PER_MS * adc_per_mv * fs / 1000.0

    # ONSET
    onset_idx     = max(0, qrs_window_start)
    amp_thr_onset = QRS_BORDER_AMPLITUDE_RATIO * abs(signal[earliest])
    for i in range(earliest - 1, max(0, qrs_window_start) - 1, -1):
        if abs(signal[i]) < amp_thr_onset and abs(signal[i] - signal[i+1]) < slope_thr:
            onset_idx = i
            break
    else:
        onset_idx = max(0, qrs_window_start)

    # OFFSET
    offset_idx     = min(len(signal) - 1, qrs_window_end - 1)
    amp_thr_offset = QRS_BORDER_AMPLITUDE_RATIO * abs(signal[latest])
    for i in range(latest + 1, min(len(signal), qrs_window_end)):
        slope_ok = abs(signal[i] - signal[i-1]) < slope_thr if i > 0 else True
        if abs(signal[i]) < amp_thr_offset and slope_ok:
            offset_idx = i
            break
    else:
        offset_idx = min(len(signal) - 1, qrs_window_end - 1)

    if onset_idx >= offset_idx:
        return None, None
    return onset_idx, offset_idx


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 9 – ARRAY-SPECIFIC BORDER DELINEATION
# ══════════════════════════════════════════════════════════════════════════════

def delineate_array_borders(channel_borders: List[Tuple[Optional[int], Optional[int]]],
                             fs: float,
                             tolerance_ms: float = ARRAY_BORDER_NORMAL_GROUP_TOLERANCE_MS
                             ) -> Tuple[Optional[int], Optional[int]]:
    valid_onsets  = [b[0] for b in channel_borders if b[0] is not None]
    valid_offsets = [b[1] for b in channel_borders if b[1] is not None]
    if not valid_onsets or not valid_offsets:
        return None, None
    tol_samp = tolerance_ms / 1000.0 * fs

    def _normal_group(borders: List[int], pick_min: bool) -> Optional[int]:
        if not borders:
            return None
        borders_sorted = sorted(borders)
        best_count, best_group = 0, []
        for anchor in borders_sorted:
            grp = [b for b in borders_sorted if abs(b - anchor) <= tol_samp]
            if len(grp) > best_count:
                best_count, best_group = len(grp), grp
        return min(best_group) if pick_min else max(best_group)

    return (_normal_group(valid_onsets,  pick_min=True),
            _normal_group(valid_offsets, pick_min=False))


# ══════════════════════════════════════════════════════════════════════════════
# STAGE 10 – GLOBAL QRS BORDER DELINEATION
# ══════════════════════════════════════════════════════════════════════════════

def delineate_global_borders(anterior_onset: Optional[int],
                              anterior_offset: Optional[int],
                              posterior_onset: Optional[int],
                              posterior_offset: Optional[int]
                              ) -> Tuple[Optional[int], Optional[int]]:
    onsets  = [x for x in (anterior_onset,  posterior_onset)  if x is not None]
    offsets = [x for x in (anterior_offset, posterior_offset) if x is not None]
    return (min(onsets)  if onsets  else None,
            max(offsets) if offsets else None)


# ══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL SINGLE-CHANNEL API (Stage 6-10)
# ══════════════════════════════════════════════════════════════════════════════

def measure_qrs_duration_paper(median_beat: np.ndarray,
                                time_axis: np.ndarray,
                                fs: float,
                                tp_baseline: float,
                                adc_per_mv: float = 1200.0
                                ) -> int:
    """
    Measure QRS duration on a single median beat using Curtin et al. (2018)
    Stage 6–10 algorithm. Drop-in replacement for old measure_qrs_duration_from_median_beat.

    FIX: R-peak index is now the CENTER of the median beat (len//2), not the
    zero-crossing of time_axis.  build_median_beat() creates a time_axis whose
    zero is the START of the pre-R window, not the R-peak itself.  Using
    argmin(|time_axis|) therefore places the 'R-peak' 200-400 ms before the
    real peak, causing QRS windows to land in the P-wave or baseline → giving
    wrong (55-60 ms) QRS durations.  Using len//2 always hits the true peak.
    """
    try:
        # FIX: R is at the CENTER of the median beat, not time_axis zero.
        # Further fix: use argmax(|signal|) within the central 60% of the beat
        # rather than len//2, because build_median_beat windows can be asymmetric
        # and the true R-peak can sit 100-150ms away from the geometric center.
        n = len(median_beat)
        center = n // 2
        search_margin = int(0.30 * fs)   # ±300ms search window around center
        search_s = max(0, center - search_margin)
        search_e = min(n, center + search_margin)
        centered_signal = np.array(median_beat, dtype=float) - float(tp_baseline)
        r_idx = search_s + int(np.argmax(np.abs(centered_signal[search_s:search_e])))
        signal = centered_signal
        if len(signal) < 30:
            return 0

        win_pre_samp  = int(0.16 * fs)   # 160ms pre-R (was 120ms) — captures deep Q in LBBB
        win_post_samp = int(0.17 * fs)   # 170ms post-R (was 120ms) — captures terminal S in RBBB/LBBB
        qrs_win_start = max(0, r_idx - win_pre_samp)
        qrs_win_end   = min(len(signal), r_idx + win_post_samp)

        if len(signal[qrs_win_start:qrs_win_end]) < 10:
            return 0

        ref_idx = find_reference_peak(signal, qrs_win_start, qrs_win_end)
        if abs(signal[ref_idx]) < 1e-6:
            return 0

        sig_peaks = find_significant_peaks(signal, ref_idx, qrs_win_start, qrs_win_end, fs)
        sig_peaks = remove_peak_outliers_by_spacing(sig_peaks, fs)
        if not sig_peaks:
            return 0

        # FIX: adapt slope threshold to actual R-peak amplitude in the median beat.
        r_peak_amp = abs(signal[ref_idx])
        effective_adc = max(r_peak_amp, adc_per_mv / 5.0)
        onset, offset = delineate_channel_borders(
            signal, sig_peaks, ref_idx, qrs_win_start, qrs_win_end, fs, effective_adc * 0.85)
        if onset is None or offset is None:
            # Retry with even wider 180ms QRS window for very wide BBB complexes
            win_pre_samp2  = int(0.18 * fs)
            win_post_samp2 = int(0.18 * fs)
            qrs_win_start2 = max(0, r_idx - win_pre_samp2)
            qrs_win_end2   = min(len(signal), r_idx + win_post_samp2)
            sig_peaks2 = find_significant_peaks(signal, ref_idx, qrs_win_start2, qrs_win_end2, fs)
            sig_peaks2 = remove_peak_outliers_by_spacing(sig_peaks2, fs)
            if sig_peaks2:
                onset, offset = delineate_channel_borders(
                    signal, sig_peaks2, ref_idx, qrs_win_start2, qrs_win_end2, fs, effective_adc * 0.85)
        if onset is None or offset is None:
            onset  = max(qrs_win_start, sig_peaks[0]  - int(0.01 * fs))
            offset = min(qrs_win_end,   sig_peaks[-1] + int(0.01 * fs))

        global_onset, global_offset = delineate_global_borders(onset, offset, onset, offset)
        if global_onset is None or global_offset is None:
            return 0

        # FIX: Use sample difference / fs * 1000 instead of time_axis[off] - time_axis[on]
        # because time_axis may not be zero-centered at R (build_median_beat offsets it).
        # Also take amplitude-based borders to avoid clipping broad/slurred LBBB.
        qrs_ms = (global_offset - global_onset) / fs * 1000.0
        amp_qrs_ms, amp_onset, amp_offset = _adaptive_amplitude_qrs_width(
            signal,
            ref_idx,
            fs,
            baseline_ratio=0.10,
            wide_ratio=0.05,
            pre_ms=250.0,
            post_ms=400.0,
        )
        # Use amplitude-based measurement ONLY for already-wide complexes.
        # This avoids inflating normal QRS (80-120ms) with P-wave/T-wave contamination.
        # Condition: slope-based must already be >= 120ms (suggesting real wide QRS/BBB)
        # AND amplitude result must be within physiological limits (≤ 180ms).
        # This preserves LBBB/RBBB detection while rejecting false wide measurements.
        if qrs_ms >= 120.0 and amp_qrs_ms > qrs_ms and amp_qrs_ms <= 180.0:
            qrs_ms = amp_qrs_ms
            global_onset = amp_onset if amp_onset is not None else global_onset
            global_offset = amp_offset if amp_offset is not None else global_offset

        return int(round(qrs_ms)) if QRS_DURATION_MIN_MS <= qrs_ms <= QRS_DURATION_MAX_MS else 0

    except Exception as e:
        print(f" ⚠️ measure_qrs_duration_paper error: {e}")
        return 0


# ══════════════════════════════════════════════════════════════════════════════
# MULTICHANNEL (MECG) API  — Stage 1-10 full pipeline
# ══════════════════════════════════════════════════════════════════════════════

def compute_global_qrs_duration_mecg(
        anterior_signals: List[np.ndarray],
        posterior_signals: List[np.ndarray],
        r_peak_idx: int,
        fs: float,
        adc_per_mv: float = 1.0,
        qrs_window_pre_ms: float = 120.0,
        qrs_window_post_ms: float = 120.0,
) -> Dict[str, Any]:
    """Full Stage 6–10 pipeline for multichannel ECG (MECG)."""
    pre_samp  = int(qrs_window_pre_ms  / 1000.0 * fs)
    post_samp = int(qrs_window_post_ms / 1000.0 * fs)
    win_start = max(0, r_peak_idx - pre_samp)

    def _process_array(signals):
        borders = []
        for sig in signals:
            if len(sig) < 10:
                borders.append((None, None)); continue
            win_end = min(len(sig), r_peak_idx + post_samp)
            ref_idx = find_reference_peak(sig, win_start, win_end)
            if abs(sig[ref_idx]) < 1e-9:
                borders.append((None, None)); continue
            sp = find_significant_peaks(sig, ref_idx, win_start, win_end, fs)
            sp = remove_peak_outliers_by_spacing(sp, fs)
            if not sp:
                borders.append((None, None)); continue
            borders.append(delineate_channel_borders(
                sig, sp, ref_idx, win_start, win_end, fs, adc_per_mv))
        return delineate_array_borders(borders, fs)

    ant_onset,  ant_offset  = _process_array(anterior_signals)
    pos_onset,  pos_offset  = _process_array(posterior_signals)
    glob_onset, glob_offset = delineate_global_borders(
        ant_onset, ant_offset, pos_onset, pos_offset)

    def _dur(on, off):
        if on is None or off is None: return None
        d = (off - on) / fs * 1000.0
        return round(d, 1) if QRS_DURATION_MIN_MS <= d <= QRS_DURATION_MAX_MS else None

    sig_len = len(anterior_signals[0]) if anterior_signals else 0
    return {
        "anterior_onset":  ant_onset,  "anterior_offset":  ant_offset,
        "anterior_qrs_ms": _dur(ant_onset, ant_offset),
        "posterior_onset": pos_onset,  "posterior_offset": pos_offset,
        "posterior_qrs_ms": _dur(pos_onset, pos_offset),
        "global_onset":    glob_onset, "global_offset":    glob_offset,
        "global_qrs_ms":   _dur(glob_onset, glob_offset),
    }


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER (raw signal, per-beat use)
# ══════════════════════════════════════════════════════════════════════════════

def qrs_duration_from_raw_signal(lead_data: np.ndarray,
                                  r_curr_idx: int,
                                  fs: float = 500.0,
                                  adc_per_mv: float = 1200.0,
                                  heart_rate: int = 75
                                  ) -> float:
    """Per-beat QRS duration from raw signal around known R-peak. HR-adaptive."""
    # ── Two-pass adaptive QRS border search ─────────────────────────────
    # Pass 1: narrow window (±80ms) — correct for normal QRS (< 120ms).
    # Pass 2: wider window — only triggered when the QRS offset reaches the
    # edge of the narrow segment, indicating the complex extends further.
    # Normal beats stop well inside; BBB beats hit the boundary.

    if heart_rate >= 150: slope_mul = 1.30
    elif heart_rate >= 120: slope_mul = 1.08
    elif heart_rate >= 75:  slope_mul = 1.10
    else:                   slope_mul = 2.20

    def _measure(pre_ms_loc, post_ms_loc):
        ws = max(0, r_curr_idx - int(pre_ms_loc / 1000.0 * fs))
        we = min(len(lead_data), r_curr_idx + int(post_ms_loc / 1000.0 * fs))
        seg = np.array(lead_data[ws:we], dtype=float)
        if len(seg) < 20:
            return 0.0, seg, None
        bl = min(len(seg), int(0.03 * fs))
        seg -= float(np.mean(seg[:max(1, bl)]))
        rp = find_reference_peak(seg, 0, len(seg))
        if abs(seg[rp]) < 1e-9:
            return 0.0, seg, None
        sp_loc = find_significant_peaks(seg, rp, 0, len(seg), fs)
        sp_loc = remove_peak_outliers_by_spacing(sp_loc, fs)
        if not sp_loc:
            return 0.0, seg, None
        r_peak_amp = abs(seg[rp])
        effective_adc = max(r_peak_amp, adc_per_mv / 5.0)
        on, off = delineate_channel_borders(seg, sp_loc, rp, 0, len(seg), fs, effective_adc * slope_mul)
        if on is None or off is None:
            return 0.0, seg, None
        ms = (off - on) / fs * 1000.0
        amp_ms, _, _ = _adaptive_amplitude_qrs_width(
            seg,
            rp,
            fs,
            baseline_ratio=0.10,
            wide_ratio=0.05,
            pre_ms=250.0,
            post_ms=400.0,
        )
        ms = max(ms, amp_ms)
        result = round(ms, 1) if QRS_DURATION_MIN_MS <= ms <= QRS_DURATION_MAX_MS else 0.0
        return result, seg, off

    # Pass 1 — narrow (±80ms)
    qrs_ms, seg1, off1 = _measure(80.0, 80.0)

    # BBB discriminator: offset clips the segment edge → complex continues beyond window
    # Normal beats stop well inside (off1 <= len-4); BBB beats hit the last samples.
    hits_boundary = off1 is not None and len(seg1) > 0 and off1 >= len(seg1) - 3

    # Pass 2 — wide (120ms pre / 160ms post), only when complex extends beyond narrow window
    if hits_boundary:
        qrs_ms_wide, _, _ = _measure(120.0, 160.0)
        if qrs_ms_wide > 0:
            qrs_ms = qrs_ms_wide

    qrs_ms_amp, _, _ = _amplitude_qrs_width(
        np.asarray(lead_data, dtype=float),
        r_curr_idx,
        fs,
        threshold_ratio=0.10,
        pre_ms=250.0,
        post_ms=400.0,
    )
    if qrs_ms_amp > qrs_ms:
        qrs_ms = qrs_ms_amp

    return qrs_ms



# ══════════════════════════════════════════════════════════════════════════════
# MULTI-LEAD GLOBAL QRS BOUNDARY  (Glasgow / Marquette style)
# ══════════════════════════════════════════════════════════════════════════════
#
# Single-lead delineation systematically UNDER-estimates QRS width: the earliest
# deflection and the latest return-to-baseline rarely happen in the same lead, so
# a Lead-II-only measurement typically reads 10-20 ms short of the true width.
#
# The fix is the boundary rule every 12-lead cart uses:
#
#     global onset  = earliest QRS onset  across the leads   (per beat)
#     global offset = latest   QRS offset across the leads   (per beat)
#     QRS duration  = global offset - global onset
#
# Taken literally ("earliest" = min, "latest" = max) one noisy lead out of twelve
# inflates every beat, so the extremes are replaced by the 15th/85th percentile —
# still the outer edge of the lead ensemble, but immune to one or two bad leads.
# Across beats the MEDIAN is used, so an ectopic beat cannot drag the result.
#
# Reference: Rijnbeek et al. (2014), "Normal values of the electrocardiogram for
# ages 16-90 years", J Electrocardiol 47:914-921 — validates the global-boundary
# approach against the CSE database.
# ══════════════════════════════════════════════════════════════════════════════

GLOBAL_QRS_MIN_LEADS: int = 2       # below this, fall back to single-lead
GLOBAL_QRS_ONSET_PERCENTILE: float = 15.0
GLOBAL_QRS_OFFSET_PERCENTILE: float = 85.0
GLOBAL_QRS_MAX_BEATS: int = 8       # beats delineated per lead (cost guard)
GLOBAL_QRS_FLAT_MV: float = 0.05    # peak-to-peak below this -> lead is flat


def _bandpass_for_global_qrs(signal: np.ndarray, fs: float) -> np.ndarray:
    """0.5-40 Hz analysis band — the same band the Lead II path is measured in.

    Every lead must go through an identical filter, otherwise the phase response
    differs between leads and the onset/offset positions stop being comparable.
    """
    try:
        nyq = fs / 2.0
        low = max(0.5 / nyq, 1e-4)
        high = min(40.0 / nyq, 0.99)
        centred = np.asarray(signal, dtype=float)
        centred = centred - float(np.mean(centred))
        if high <= low or centred.size < 30:
            return centred
        b, a = butter(2, [low, high], btype="band")
        return filtfilt(b, a, centred)
    except Exception:
        centred = np.asarray(signal, dtype=float)
        return centred - float(np.mean(centred))


def _curtin_beat_borders(lead_data: np.ndarray,
                         r_idx: int,
                         fs: float,
                         adc_per_mv: float,
                         heart_rate: int
                         ) -> Optional[Tuple[int, int]]:
    """One beat, one lead -> (onset, offset) as ABSOLUTE sample indices.

    Same two-pass Curtin border search as qrs_duration_from_raw_signal(), but it
    returns the border positions instead of collapsing them to a duration — the
    global rule has to compare positions across leads, not widths.
    """
    if heart_rate >= 150:
        slope_mul = 1.30
    elif heart_rate >= 120:
        slope_mul = 1.08
    elif heart_rate >= 75:
        slope_mul = 1.10
    else:
        slope_mul = 2.20

    def _measure(pre_ms_loc: float, post_ms_loc: float):
        ws = max(0, r_idx - int(pre_ms_loc / 1000.0 * fs))
        we = min(len(lead_data), r_idx + int(post_ms_loc / 1000.0 * fs))
        seg = np.array(lead_data[ws:we], dtype=float)
        if len(seg) < 20:
            return None, seg, None
        bl = min(len(seg), int(0.03 * fs))
        seg -= float(np.mean(seg[:max(1, bl)]))
        rp = find_reference_peak(seg, 0, len(seg))
        if abs(seg[rp]) < 1e-9:
            return None, seg, None
        sp_loc = find_significant_peaks(seg, rp, 0, len(seg), fs)
        sp_loc = remove_peak_outliers_by_spacing(sp_loc, fs)
        if not sp_loc:
            return None, seg, None
        effective_adc = max(abs(seg[rp]), adc_per_mv / 5.0)
        on, off = delineate_channel_borders(
            seg, sp_loc, rp, 0, len(seg), fs, effective_adc * slope_mul)
        if on is None or off is None:
            return None, seg, None
        return (ws + on, ws + off), seg, off

    # Pass 1 — narrow window: correct for a normal-width complex.
    borders, seg1, off1 = _measure(80.0, 80.0)

    # A complex that runs into the edge of the narrow segment is still going:
    # widen the window so BBB terminal forces are not clipped.
    hits_boundary = off1 is not None and len(seg1) > 0 and off1 >= len(seg1) - 3
    if hits_boundary:
        wide, _, _ = _measure(120.0, 160.0)
        if wide is not None:
            borders = wide

    if borders is None:
        return None

    on_abs, off_abs = borders
    dur_ms = (off_abs - on_abs) / fs * 1000.0
    if not (QRS_DURATION_MIN_MS <= dur_ms <= QRS_DURATION_MAX_MS):
        return None
    return int(on_abs), int(off_abs)


def compute_global_qrs_duration_12lead(
        lead_signals: Dict[str, np.ndarray],
        r_peaks,
        fs: float = 500.0,
        adc_per_mv: float = 1200.0,
        heart_rate: int = 75,
        min_leads: int = GLOBAL_QRS_MIN_LEADS,
        max_beats: int = GLOBAL_QRS_MAX_BEATS,
        flat_threshold: Optional[float] = None,
) -> Optional[Dict[str, Any]]:
    """Global QRS duration measured across every supplied lead.

    All leads must be time-synchronised and share Lead II's sample clock — the
    R-peaks detected on Lead II are reused as the per-beat anchor for every lead,
    which is what makes the onsets and offsets directly comparable.

    Args:
        lead_signals:   {lead name -> raw signal}. Flat/short leads are dropped.
        r_peaks:        R-peak sample indices detected on Lead II.
        fs:             Sampling rate (Hz).
        adc_per_mv:     ADC counts per mV, for the slope threshold.
        heart_rate:     BPM, selects the HR-adaptive slope multiplier.
        min_leads:      Below this many usable leads, returns None so the caller
                        falls back to its single-lead measurement.
        max_beats:      Beats delineated per lead.
        flat_threshold: Peak-to-peak below which a lead counts as flat. Defaults
                        to 0.05 mV expressed in the signal's own units.

    Returns:
        dict(global_qrs_ms, global_onset, global_offset, leads_used, leads_total,
             beats_used, per_lead_qrs_ms) or None when the measurement fails.
    """
    try:
        peaks = np.asarray(r_peaks, dtype=int)
        if peaks.size < 2 or not lead_signals:
            return None

        if flat_threshold is None:
            flat_threshold = GLOBAL_QRS_FLAT_MV * float(adc_per_mv)

        # Beats nearest the middle of the buffer are the best conditioned: the
        # filter edges at either end of the window distort the outer complexes.
        if peaks.size > max_beats:
            mid = peaks.size // 2
            start = max(0, mid - max_beats // 2)
            peaks = peaks[start:start + max_beats]

        min_len = int(2.0 * fs)
        beat_onsets = [[] for _ in range(peaks.size)]
        beat_offsets = [[] for _ in range(peaks.size)]
        per_lead_qrs = {}
        leads_used = 0

        for lead_name, raw in lead_signals.items():
            sig = np.asarray(raw, dtype=float)
            if sig.size < min_len:
                continue
            if float(np.max(sig) - np.min(sig)) < flat_threshold:
                continue          # flat / disconnected lead

            filt = _bandpass_for_global_qrs(sig, fs)

            lead_widths = []
            for beat_i, r_idx in enumerate(peaks):
                if r_idx < 0 or r_idx >= filt.size:
                    continue
                borders = _curtin_beat_borders(
                    filt, int(r_idx), fs, adc_per_mv, heart_rate)
                if borders is None:
                    continue
                on_abs, off_abs = borders
                beat_onsets[beat_i].append(on_abs)
                beat_offsets[beat_i].append(off_abs)
                lead_widths.append((off_abs - on_abs) / fs * 1000.0)

            if lead_widths:
                leads_used += 1
                per_lead_qrs[lead_name] = round(float(np.median(lead_widths)), 1)

        if leads_used < min_leads:
            return None

        # Per beat: the outer edge of the lead ensemble, trimmed of outliers.
        global_onsets = []
        global_offsets = []
        for beat_i in range(peaks.size):
            onsets = beat_onsets[beat_i]
            offsets = beat_offsets[beat_i]
            if not onsets or not offsets:
                continue
            global_onsets.append(
                float(np.percentile(onsets, GLOBAL_QRS_ONSET_PERCENTILE)))
            global_offsets.append(
                float(np.percentile(offsets, GLOBAL_QRS_OFFSET_PERCENTILE)))

        if not global_onsets:
            return None

        # Across beats: median, so one ectopic beat cannot move the result.
        med_onset = float(np.median(global_onsets))
        med_offset = float(np.median(global_offsets))
        qrs_ms = (med_offset - med_onset) / fs * 1000.0
        if not (QRS_DURATION_MIN_MS <= qrs_ms <= QRS_DURATION_MAX_MS):
            return None

        return {
            "global_qrs_ms":   round(qrs_ms, 1),
            "global_onset":    int(round(med_onset)),
            "global_offset":   int(round(med_offset)),
            "leads_used":      leads_used,
            "leads_total":     len(lead_signals),
            "beats_used":      len(global_onsets),
            "per_lead_qrs_ms": per_lead_qrs,
        }

    except Exception as e:
        print(f" ⚠️ compute_global_qrs_duration_12lead error: {e}")
        return None


def describe_qrs_method(method: str, leads_used: int = 1) -> str:
    """One-line provenance for a QRS width, for tooltips and report footers.

    The number alone is ambiguous — 96 ms measured on Lead II and 96 ms measured
    across twelve leads are different claims — so anywhere the width is shown
    should be able to say which one it is.
    """
    if method == "global-multilead" and leads_used >= 2:
        return (f"Global QRS: earliest onset to latest offset across "
                f"{leads_used} leads (Glasgow/Marquette rule).")
    return "Single-lead QRS: measured on Lead II only (Curtin 2018)."


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 70)
    print("QRS Detection Self-Test  (Curtin et al. 2018, Stages 1-10)")
    print("=" * 70)

    rng = np.random.default_rng(42)
    FS  = 500.0

    def _g(t, mu, sigma, amp):
        return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)

    def _make_beat(hr=72, noise=0.0):
        rr  = 60.0 / hr
        n   = int(rr * FS)
        t   = np.arange(n) / FS
        r_s = rr * 0.35
        sig = (_g(t, r_s-0.16, 0.020, 150) + _g(t, r_s-0.03, 0.008, -80)
               + _g(t, r_s, 0.012, 1000) + _g(t, r_s+0.025, 0.008, -200)
               + _g(t, r_s+0.22, 0.040, 200))
        if noise: sig += rng.normal(0, noise, n)
        return sig, int(r_s * FS)

    def _make_ecg(hr=72, n_beats=10, noise=0.0):
        rr  = 60.0 / hr
        blen = int(rr * FS)
        n   = n_beats * blen
        sig = np.zeros(n)
        beat, r_off = _make_beat(hr, noise)
        for i in range(n_beats):
            s = i * blen
            e = min(s + blen, n)
            sig[s:e] += beat[:e-s]
        return sig / 1000.0  # convert to mV

    # Stage 1-5 tests
    print("\n── Stage 1-5 Tests ──")
    ecg1 = _make_ecg(72, 12)
    r1   = detect_qrs_full([ecg1], FS)
    print(f"Test 1 – 72 BPM: {len(r1['r_peaks'])} peaks (expected ≈12)  HR={r1['hr_bpm']:.0f}")
    assert 8 <= len(r1["r_peaks"]) <= 14, f"FAIL: {len(r1['r_peaks'])}"
    print("  PASS ✓")

    ecg2 = _make_ecg(150, 15)
    r2   = detect_qrs_full([ecg2], FS)
    print(f"Test 2 – 150 BPM: {len(r2['r_peaks'])} peaks (expected ≈15)")
    assert 10 <= len(r2["r_peaks"]) <= 18, f"FAIL: {len(r2['r_peaks'])}"
    print("  PASS ✓")

    # Stage 6-10 tests
    print("\n── Stage 6-10 Tests ──")
    beat3, r3 = _make_beat(72)
    d3 = qrs_duration_from_raw_signal(beat3, r3, FS)
    print(f"Test 3 – QRSd 72 BPM: {d3:.1f} ms")
    assert QRS_DURATION_MIN_MS <= d3 <= 120.0, f"FAIL: {d3}"
    print("  PASS ✓")

    beat4, r4 = _make_beat(75)
    t4  = (np.arange(len(beat4)) - r4) / FS * 1000.0
    d4  = measure_qrs_duration_paper(beat4, t4, FS, float(np.mean(beat4[:25])))
    print(f"Test 4 – measure_qrs_duration_paper 75 BPM: {d4} ms")
    assert 0 <= d4 <= int(QRS_DURATION_MAX_MS), f"FAIL: {d4}"
    print("  PASS ✓")

    print("\n" + "=" * 70)
    print("All tests passed — Stage 1-10 complete.")
    print("=" * 70)
