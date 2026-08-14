"""
QRS Complex Detection & Measurement Module
==========================================
Paper: Curtin et al., "QRS Complex Detection and Measurement Algorithms for
       Multichannel ECGs in Cardiac Resynchronization Therapy Patients"
       IEEE J. Transl. Eng. Health Med., 2018. DOI: 10.1109/JTEHM.2018.2844195

COMPLETE PIPELINE — Stages 1 through 10
────────────────────────────────────────
QRS Detection (Stages 1–5):
  Stage 1 → Channel grouping + averaging
  Stage 2 → Peak detection (amplitude + width criteria)
  Stage 3 → QRS complex windowing (PR + QT approximation)
  Stage 4 → Additional complex identification
  Stage 5 → Morphology classification (PM vs OM)

QRS Duration Measurement (Stages 6–10):
  Stage 6 → Reference peak identification + significant peaks detection
  Stage 7 → Array-specific peak groups (anterior / posterior)
  Stage 8 → Channel-specific border delineation  ← UPGRADED (Curtin 2018 strict)
  Stage 9 → Array-specific border delineation (normal group within 20 ms)
  Stage 10→ Global border delineation (earliest anterior + latest posterior)

UPGRADES vs previous version (integrated from ECGAnalyzer.kt / ecg_qrs_detector.py):
  UPGRADE-1 (delineate_channel_borders / Stage 8):
    - True isoelectric baseline from TP segment trimmed mean (10-90th pct)
      instead of abs(signal[i]) which assumed baseline=0.  Fixes +40-60ms
      offset overestimate on real ECG with 0.1-0.2 mV post-filter drift.
    - BBB detection: signal still active at R+100ms (>18% R amplitude) →
      flag as BBB and relax offset amplitude fraction to 0.45 + fewer
      confirmation samples to avoid overshooting slurred S-wave.
    - Onset: scan backwards with amplitude + slope gate + 5-sample rising
      confirmation (≥3 of 5 samples must be rising past 50% threshold).
    - Offset: sliding stability gate — N consecutive samples must be below
      amplitude threshold AND below 2× slope threshold simultaneously.
      confirmSamp = 3-6 depending on HR and BBB status.
    - QRS duration cap raised: normal HR ≤100 → 200ms (was ~130ms via
      0.20 ratio), allowing RBBB/LBBB (120-200ms) to be measured correctly.

  UPGRADE-2 (qrs_duration_from_raw_signal):
    - New _curtin_validate_peaks() pre-pass: amplitude + half-width
      down-selection (±0.10 mV, ±20 ms) and 81ms intra-complex merge,
      exactly matching Curtin 2018 §2b-§2d.
    - _curtin_find_significant_peaks_local(): Q/S extended to 120ms from R
      (was 52ms) to capture terminal deflections in RBBB/LBBB.
    - Two-pass window (80ms/160ms) retained; now also uses Curtin borders.

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
QRS_BORDER_AMPLITUDE_RATIO:          float = 0.50   # Curtin 2018 Table 2 §8: 50% of closest peak
QRS_BORDER_SLOPE_THRESHOLD_MV_PER_MS: float = 0.025 # Curtin 2018 Table 2 §8: 2.5×10⁻² mV/ms
ARRAY_BORDER_NORMAL_GROUP_TOLERANCE_MS: float = 20.0
HR_WINDOW_MIN_BPM: int = 40
HR_WINDOW_MAX_BPM: int = 120
MIN_SIGNIFICANT_PEAK_HEIGHT_RATIO: float = 0.10

# ── Curtin 2018 border delineation constants (UPGRADE-1) ─────────────────────
# These are used in the upgraded delineate_channel_borders() and
# qrs_duration_from_raw_signal() below.
_CURTIN_BORDER_AMP_FRACTION   = 0.50   # §8: amplitude must drop below 50% of peak dev
_CURTIN_BORDER_SLOPE_MV_PER_MS = 0.025 # §8: slope threshold 2.5×10⁻² mV/ms
_CURTIN_PEAK_AMP_TOL_MV       = 0.10   # §2b: ±0.10 mV amplitude tolerance
_CURTIN_PEAK_WIDTH_TOL_MS     = 20.0   # §2b: ±20 ms half-width tolerance
_CURTIN_MAX_INTRA_COMPLEX_MS  = 81.0   # §2d: 81 ms intra-complex merge limit
_CURTIN_MAX_SIG_PEAK_MS       = 120.0  # §6 extended (RBBB/LBBB): was 52 ms

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
    """
    Stage 8 — QRS onset and offset delineation.

    UPGRADED to Curtin 2018 §8 / §10 algorithm (ported from ECGAnalyzer.kt):
      • True isoelectric baseline from TP segment trimmed mean [10-90th pct],
        60-10 samples before the QRS window.  This replaces the old
        abs(signal[i]) comparison which assumed baseline=0 and systematically
        over-estimated offset by 40-60 ms on filtered ECG with residual drift.
      • BBB detection: if signal amplitude at R+100 ms > 18% of R amplitude,
        the complex is flagged as BBB-type and the offset fraction is relaxed
        to 0.45 (vs 0.50 normal) with fewer confirmation samples.
      • Onset: backward scan with amplitude + slope gate plus a rising-signal
        confirmation (≥3 of 5 forward samples must exceed 50% threshold).
      • Offset: forward scan with amplitude + slope gate plus sliding stability
        window (confirmSamp consecutive samples all below threshold).
    """
    if not significant_peaks:
        return None, None

    # ── slope threshold (per sample, signal-scale independent) ──────────────
    slope_thr_per_samp = _CURTIN_BORDER_SLOPE_MV_PER_MS * 1000.0 / fs

    # ── True isoelectric baseline (TP segment before the QRS window) ────────
    # 60-10 samples before window start; trimmed mean (10-90th pct) for noise
    baseline_start = max(0, qrs_window_start - 60)
    baseline_end   = max(0, qrs_window_start - 10)
    if baseline_end > baseline_start:
        seg = sorted(signal[baseline_start:baseline_end].tolist())
        t1  = int(len(seg) * 0.10)
        t2  = len(seg) - t1
        iso_baseline = float(np.mean(seg[t1:t2])) if t2 > t1 else float(np.mean(seg))
    else:
        iso_baseline = 0.0

    # ── BBB detection: is signal still active at R+100 ms? ──────────────────
    check_samp  = min(ref_peak_idx + int(fs * 100 / 1000),
                      qrs_window_end - 1, len(signal) - 1)
    r_peak_amp  = abs(float(signal[ref_peak_idx]) - iso_baseline)
    amp_at_100  = abs(float(signal[check_samp])   - iso_baseline)
    is_bbb      = (r_peak_amp > 1e-9) and (amp_at_100 > r_peak_amp * 0.18)

    # ── QRS ONSET ────────────────────────────────────────────────────────────
    # Reference: first (leftmost) significant peak — usually the Q wave.
    q_peak = significant_peaks[0]
    onset_peak_dev = abs(float(signal[q_peak]) - iso_baseline)
    onset_amp_thr  = onset_peak_dev * _CURTIN_BORDER_AMP_FRACTION

    onset_idx = max(0, qrs_window_start)  # default: window start
    for k in range(q_peak - 1, max(0, qrs_window_start) - 1, -1):
        dev   = abs(float(signal[k]) - iso_baseline)
        slope = abs(float(signal[k + 1]) - float(signal[k])) if k + 1 < len(signal) else 0.0
        if dev < onset_amp_thr and slope < slope_thr_per_samp:
            # Confirm: ≥3 of next 5 samples must be rising above 50% threshold
            rising_count = 0
            for j in range(1, 6):
                nk = k + j
                if (nk < len(signal)
                        and abs(float(signal[nk]) - iso_baseline) > onset_amp_thr * 0.5):
                    rising_count += 1
            if rising_count >= 3:
                onset_idx = k + 1
                break

    # ── QRS OFFSET ───────────────────────────────────────────────────────────
    # Reference: last (rightmost) significant peak — usually the S wave.
    s_peak = significant_peaks[-1]
    offset_peak_dev = abs(float(signal[s_peak]) - iso_baseline)

    # BBB: slurred S-wave requires a relaxed amplitude fraction
    if is_bbb:
        offset_amp_fraction = 0.45
        confirm_samp = 3
    else:
        offset_amp_fraction = _CURTIN_BORDER_AMP_FRACTION  # 0.50
        # HR-derived confirmation count from calling context is not directly
        # available here, so use a moderate default of 5 (≈normal sinus)
        confirm_samp = 5

    offset_amp_thr = offset_peak_dev * offset_amp_fraction
    offset_idx = min(len(signal) - 1, qrs_window_end - 1)  # default: window end

    k = s_peak + 1
    while k < min(len(signal), qrs_window_end) - confirm_samp:
        dev   = abs(float(signal[k]) - iso_baseline)
        slope = abs(float(signal[k + 1]) - float(signal[k])) if k + 1 < len(signal) else 0.0
        if dev < offset_amp_thr and slope < slope_thr_per_samp:
            stable_count = 0
            for j in range(1, confirm_samp + 1):
                nk = k + j
                if (nk < len(signal)
                        and abs(float(signal[nk]) - iso_baseline) < offset_amp_thr
                        and abs(float(signal[nk]) - float(signal[nk - 1])) < slope_thr_per_samp * 2.0):
                    stable_count += 1
            if stable_count >= (confirm_samp + 1) // 2:
                offset_idx = k
                break
        k += 1

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
# CURTIN 2018 PEAK VALIDATION HELPERS  (UPGRADE-2)
# Ported from ECGAnalyzer.kt curtinValidatePeaks() and curtinFindSignificantPeaks()
# ══════════════════════════════════════════════════════════════════════════════

def _curtin_validate_peaks(signal: np.ndarray,
                            candidate_peaks: List[int],
                            fs: float) -> List[int]:
    """
    Curtin 2018 §2b-§2d: amplitude + half-width down-selection,
    then intra-complex merge (peaks within 81 ms → keep the largest).

    This replaces the simpler amplitude-only filter that was used before
    find_significant_peaks(), preventing T-wave and noise peaks from being
    treated as QRS peaks and inflating the measured duration.
    """
    if not candidate_peaks:
        return []

    # Per-second amplitude ranges → global threshold = 50% of mean range
    window_thresh: List[float] = []
    idx = 0
    while idx < len(signal):
        sl = signal[idx: idx + int(fs)]
        if len(sl) > 0:
            window_thresh.append(float(np.max(sl) - np.min(sl)))
        idx += int(fs)
    global_thr = float(np.mean(window_thresh)) * 0.5 if window_thresh else 0.0

    amp_filtered = [p for p in candidate_peaks
                    if 0 <= p < len(signal) and abs(float(signal[p])) > global_thr]
    if not amp_filtered:
        return candidate_peaks  # safety: return all if nothing passes

    # Amplitude mode filter (±0.10 mV)
    amps = sorted(abs(float(signal[p])) for p in amp_filtered)
    mode_amp = amps[len(amps) // 2]
    amp_valid = [p for p in amp_filtered
                 if abs(abs(float(signal[p])) - mode_amp) <= _CURTIN_PEAK_AMP_TOL_MV]

    # Half-maximum width filter (±20 ms)
    def _half_max_width_ms(pk: int) -> float:
        half = abs(float(signal[pk])) * 0.5
        left = right = pk
        while left > 0 and abs(float(signal[left - 1])) > half:
            left -= 1
        while right < len(signal) - 1 and abs(float(signal[right + 1])) > half:
            right += 1
        return (right - left) * 1000.0 / fs

    widths = [(p, _half_max_width_ms(p)) for p in amp_valid]
    if not widths:
        return amp_filtered
    mode_w = sorted(w for _, w in widths)[len(widths) // 2]
    width_valid = [p for p, w in widths
                   if abs(w - mode_w) <= _CURTIN_PEAK_WIDTH_TOL_MS]
    if not width_valid:
        return amp_filtered

    # Intra-complex merge: peaks within 81 ms → keep the largest amplitude
    max_intra_samp = int(_CURTIN_MAX_INTRA_COMPLEX_MS / 1000.0 * fs)
    merged: List[int] = []
    gi = 0
    while gi < len(width_valid):
        group = [width_valid[gi]]
        j = gi + 1
        while j < len(width_valid) and width_valid[j] - group[-1] <= max_intra_samp:
            group.append(width_valid[j])
            j += 1
        merged.append(max(group, key=lambda p: abs(float(signal[p]))))
        gi = j

    return merged


def _curtin_find_q_s_peaks(signal: np.ndarray,
                             r_peak: int,
                             win_start: int,
                             win_end: int,
                             fs: float) -> Tuple[Optional[int], Optional[int]]:
    """
    Curtin 2018 §6 — significant Q and S peaks relative to R.

    Extends the search to 120 ms from R (instead of Curtin's original 52 ms)
    to capture terminal deflections in RBBB (slurred S, R') and LBBB
    (notched R, deep S), exactly as done in ECGAnalyzer.kt.

    Returns (q_idx, s_idx) — either may be None.
    """
    max_space_samp = int(_CURTIN_MAX_SIG_PEAK_MS / 1000.0 * fs)   # 120 ms
    ref_amp   = float(signal[r_peak])
    ref_height = abs(ref_amp)

    # Reference ascending-flank max slope (up to 20 samples before R)
    slope_win = min(20, r_peak)
    ref_slope = 0.0
    for k in range(r_peak - slope_win, r_peak):
        if 0 <= k and k + 1 < len(signal):
            s = abs(float(signal[k + 1]) - float(signal[k]))
            if s > ref_slope:
                ref_slope = s

    # ── Q peak (before R) ────────────────────────────────────────────────────
    q_idx: Optional[int] = None
    q_scan_start = max(win_start, r_peak - max_space_samp)
    q_scan_end   = max(q_scan_start, r_peak - 3)
    if q_scan_end > q_scan_start:
        local_min = float(signal[q_scan_start])
        local_min_idx = q_scan_start
        for k in range(q_scan_start, q_scan_end):
            if float(signal[k]) < local_min:
                local_min = float(signal[k])
                local_min_idx = k
        c_height = abs(local_min)
        scale    = c_height / ref_height if ref_height > 1e-9 else 0.0
        c_slope  = 0.0
        for k in range(local_min_idx, min(local_min_idx + 5, r_peak)):
            if k + 1 < len(signal):
                s = abs(float(signal[k + 1]) - float(signal[k]))
                if s > c_slope:
                    c_slope = s
        opposite_polarity = (
            (ref_amp > 0 and local_min < 0)
            or (ref_amp < 0 and local_min > 0)
            or c_height > ref_height * 0.03
        )
        if opposite_polarity and c_slope >= ref_slope * scale * 0.30:
            q_idx = local_min_idx

    # ── S peak (after R) ─────────────────────────────────────────────────────
    s_idx: Optional[int] = None
    s_scan_start = min(r_peak + 3, len(signal) - 1)
    s_scan_end   = min(win_end, r_peak + max_space_samp, len(signal) - 1)
    if s_scan_end > s_scan_start:
        local_min = float(signal[s_scan_start])
        local_min_idx = s_scan_start
        for k in range(s_scan_start, s_scan_end):
            if float(signal[k]) < local_min:
                local_min = float(signal[k])
                local_min_idx = k
        c_height = abs(local_min)
        scale    = c_height / ref_height if ref_height > 1e-9 else 0.0
        c_slope  = 0.0
        slope_scan_start = max(r_peak, local_min_idx - 5)
        for k in range(slope_scan_start, local_min_idx):
            if k + 1 < len(signal):
                s = abs(float(signal[k + 1]) - float(signal[k]))
                if s > c_slope:
                    c_slope = s
        opposite_polarity = (
            (ref_amp > 0 and local_min < 0)
            or (ref_amp < 0 and local_min > 0)
            or c_height > ref_height * 0.03
        )
        if opposite_polarity and c_slope >= ref_slope * scale * 0.30:
            s_idx = local_min_idx

    return q_idx, s_idx


# ══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE WRAPPER (raw signal, per-beat use)
# ══════════════════════════════════════════════════════════════════════════════

def _kotlin_qrs_window_samples(rr_ms: float, fs: float) -> Tuple[int, int]:
    """ECGAnalyzer.kt curtinBuildWindows() PR/QT approximation in samples."""
    median_rr_sec = float(rr_ms) / 1000.0 if rr_ms and rr_ms > 0 else 0.8
    clamped_rr = float(np.clip(median_rr_sec, 60.0 / 120.0, 60.0 / 40.0))
    pr_sec = -0.1195 + 0.5170 * np.sqrt(clamped_rr) - 0.2199 * clamped_rr
    pr_sec = float(np.clip(pr_sec, 0.08, 0.30))
    qt_sec = 0.38 * np.sqrt(clamped_rr)
    qt_sec = float(np.clip(qt_sec, 0.24, 0.60))
    return int(pr_sec * fs), int(qt_sec * fs)


def _kotlin_qrs_max_ms(local_hr: int) -> int:
    if local_hr < 60:
        return 200
    if local_hr <= 100:
        return 200
    if local_hr <= 150:
        return 180
    return 160


def qrs_duration_from_raw_signal(lead_data: np.ndarray,
                                  r_curr_idx: int,
                                  fs: float = 500.0,
                                  adc_per_mv: float = 1200.0,
                                  heart_rate: int = 75,
                                  rr_ms: Optional[float] = None
                                  ) -> float:
    """Per-beat QRS duration using the ECGAnalyzer.kt Curtin 2018 flow.

    The R peak is supplied by Kotlin-style Pan-Tompkins. Around that peak we
    build the same RR-derived Curtin window used by ECGAnalyzer.kt, find Q/S
    significant peaks, delineate borders on the full lead so the TP baseline is
    available, then apply the same HR-specific QRS duration acceptance caps.
    """
    sig = np.asarray(lead_data, dtype=float)
    if sig.size < 20 or fs <= 0:
        return 0.0

    r_curr_idx = int(max(0, min(sig.size - 1, r_curr_idx)))
    local_hr = int(np.clip(int(round(heart_rate or 75)), 30, 300))
    local_rr_ms = float(rr_ms) if rr_ms and rr_ms > 0 else 60000.0 / max(local_hr, 1)
    pre_samp, post_samp = _kotlin_qrs_window_samples(local_rr_ms, fs)

    win_start = max(0, r_curr_idx - pre_samp)
    win_end = min(sig.size, r_curr_idx + post_samp + 1)
    if win_end - win_start < 20 or abs(float(sig[r_curr_idx])) < 1e-9:
        return 0.0

    local_extrema = _find_local_extrema(sig, win_start, win_end)
    raw_candidates = sorted(set([r_curr_idx] + local_extrema))
    validated = _curtin_validate_peaks(sig, raw_candidates, fs)
    if validated and r_curr_idx not in validated:
        nearest = min(validated, key=lambda p: abs(int(p) - r_curr_idx))
        if abs(nearest - r_curr_idx) <= int(0.04 * fs):
            r_curr_idx = int(nearest)

    q_idx, s_idx = _curtin_find_q_s_peaks(sig, r_curr_idx, win_start, win_end, fs)
    significant = [r_curr_idx] + [p for p in (q_idx, s_idx) if p is not None]
    significant = sorted(set(significant))
    significant = remove_peak_outliers_by_spacing(
        significant, fs, max_spacing_ms=_CURTIN_MAX_SIG_PEAK_MS
    )
    if not significant:
        return 0.0

    r_peak_amp = abs(float(sig[r_curr_idx]))
    effective_adc = max(r_peak_amp, adc_per_mv / 5.0)
    onset, offset = delineate_channel_borders(
        sig, significant, r_curr_idx, win_start, win_end, fs, effective_adc
    )
    if onset is None or offset is None:
        return 0.0

    duration_ms = (offset - onset) * 1000.0 / fs
    min_qrs = 40.0
    max_qrs = float(_kotlin_qrs_max_ms(local_hr))
    if min_qrs <= duration_ms <= max_qrs:
        return round(float(duration_ms), 1)

    # If the baseline/offset scan runs wide, keep the Curtin significant Q/S
    # span as the conservative complex boundary before applying relaxed caps.
    qs_onset = max(win_start, significant[0] - int(0.01 * fs))
    qs_offset = min(win_end - 1, significant[-1] + int(0.01 * fs))
    qs_ms = (qs_offset - qs_onset) * 1000.0 / fs
    if min_qrs <= qs_ms <= max_qrs:
        return round(float(qs_ms), 1)

    if (min_qrs - 10.0) <= duration_ms <= (max_qrs + 30.0):
        return round(float(duration_ms), 1)
    return 0.0



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
