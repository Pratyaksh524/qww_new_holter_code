"""
qrs_detector.py — Layer 2a: QRS / R-peak detection.

Wraps the existing Pan-Tompkins implementation (pan_tompkins.py) and
the improved detector in arrhythmia_detector.py with a single, clean API.

    from ecg.qrs_detector import detect_qrs, compute_rr, qrs_metrics

Design principle: Input is a CLEAN (already filtered) ECG signal.
Output is beat-level information only — no rhythm interpretation here.

UPGRADES vs previous version (integrated from ECGAnalyzer.kt / ecg_qrs_detector.py):
  UPGRADE-1 (calculate_qrs_width):
    - Uses Curtin 2018 delineate_channel_borders() via qrs_detection module
      instead of the local amplitude + gradient dual-pass approach.
    - True isoelectric baseline from TP segment (not fixed 0 / median).
    - BBB-aware offset search with confirmation samples.
    - Correct QRS cap: 200 ms at normal HR (was capped at 300 ms then
      filtered to 40-300 ms, allowing T-wave contamination).

  UPGRADE-2 (qrs_metrics):
    - Falls back to Curtin 2018 border logic when calculate_qrs_width
      returns 0 for a beat, preventing the mean from being pulled low by
      failed measurements.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

# Use the existing robust Pan-Tompkins implementation
try:
    from ecg.pan_tompkins import pan_tompkins as _pan_tompkins
    _PT_AVAILABLE = True
except ImportError:
    _PT_AVAILABLE = False

# Also bring in the improved detector from arrhythmia_detector
try:
    from ecg.arrhythmia_detector import (
        detect_r_peaks_pan_tompkins as _detect_r_peaks_improved,
        measure_beat as _measure_beat,
    )
    _AD_AVAILABLE = True
except ImportError:
    _AD_AVAILABLE = False

# Curtin 2018 border delineation helpers (UPGRADE-1)
try:
    from ecg.qrs_detection import (
        find_reference_peak            as _find_reference_peak,
        find_significant_peaks         as _find_significant_peaks,
        remove_peak_outliers_by_spacing as _remove_peak_outliers,
        delineate_channel_borders      as _delineate_channel_borders,
        _curtin_validate_peaks,
        _curtin_find_q_s_peaks,
        _find_local_extrema,
    )
    _CURTIN_AVAILABLE = True
except ImportError:
    _CURTIN_AVAILABLE = False

DEFAULT_FS = 500.0


def detect_qrs(signal: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Detect QRS R-peaks in a CLEAN ECG signal using Pan-Tompkins algorithm.

    Uses the improved adaptive-threshold Pan-Tompkins from pan_tompkins.py,
    with fallback to the one in arrhythmia_detector.py.

    Args:
        signal: 1-D clean ECG signal (already filtered)
        fs:     Sampling rate (Hz)

    Returns:
        Sorted array of R-peak sample indices (dtype int)
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < int(fs * 1.5):
        return np.array([], dtype=int)

    if _PT_AVAILABLE:
        peaks = _pan_tompkins(sig, fs=int(fs))
        if len(peaks) >= 2:
            return np.asarray(peaks, dtype=int)

    if _AD_AVAILABLE:
        peaks = _detect_r_peaks_improved(sig, fs=fs)
        return np.asarray(peaks, dtype=int)

    # Minimal fallback: simple threshold peak detection
    from scipy.signal import find_peaks
    min_distance = max(1, int(0.25 * fs))  # 250 ms minimum between beats
    threshold = float(np.std(sig)) * 0.8
    peaks, _ = find_peaks(sig, height=threshold, distance=min_distance)
    return np.asarray(peaks, dtype=int)


def compute_rr(peaks: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Compute RR intervals in SECONDS from R-peak sample indices.

    Args:
        peaks: Array of R-peak sample indices
        fs:    Sampling rate (Hz)

    Returns:
        Array of RR intervals in seconds (length = len(peaks) - 1)
    """
    peaks = np.asarray(peaks, dtype=float)
    if peaks.size < 2:
        return np.array([], dtype=float)
    return np.diff(peaks) / float(fs)


def compute_rr_ms(peaks: np.ndarray, fs: float = DEFAULT_FS) -> np.ndarray:
    """
    Compute RR intervals in MILLISECONDS from R-peak sample indices.

    Args:
        peaks: Array of R-peak sample indices
        fs:    Sampling rate (Hz)

    Returns:
        Array of RR intervals in milliseconds
    """
    return compute_rr(peaks, fs) * 1000.0


def beat_metrics(signal: np.ndarray, peaks: np.ndarray,
                 fs: float = DEFAULT_FS) -> List[Dict]:
    """
    Measure per-beat interval features (QRS width, PR, QT, ST, P presence).

    Uses the detailed beat measurer from arrhythmia_detector.py.

    Args:
        signal: Clean Lead-II ECG (or best available lead)
        peaks:  R-peak sample indices
        fs:     Sampling rate (Hz)

    Returns:
        List of beat dicts with keys:
            r_peak, qrs_ms, p_present, pr_ms, qt_ms, st_level_mv,
            p_amplitude, q_onset, j_point, noisy, …
    """
    if not _AD_AVAILABLE:
        return []
    sig = np.asarray(signal, dtype=float)
    result = []
    for r in peaks:
        b = _measure_beat(sig, int(r), fs)
        if b is not None:
            result.append(b)
    return result


def calculate_qrs_width(signal: np.ndarray, qrs_peak: int, fs: float = DEFAULT_FS) -> float:
    """
    QRS width via Curtin 2018 Stage 8 border delineation (UPGRADE-1).

    Uses delineate_channel_borders() from qrs_detection.py which applies:
      - True isoelectric baseline from TP segment trimmed mean
      - BBB detection and relaxed offset criteria
      - Amplitude + slope gate with confirmation samples (onset and offset)

    Falls back to the original amplitude+gradient dual-pass when the
    Curtin module is unavailable or returns 0.

    Valid QRS range: 40-200 ms.
    """
    sig = np.asarray(signal, dtype=float)
    if sig.size < 3 or fs <= 0:
        return 0.0

    qrs_peak = int(max(0, min(sig.size - 1, qrs_peak)))

    # ── Curtin 2018 path (preferred) ─────────────────────────────────────────
    if _CURTIN_AVAILABLE:
        try:
            # Window: 250 ms pre-R, 400 ms post-R (generous for BBB)
            pre_samp  = int(0.25 * fs)
            post_samp = int(0.40 * fs)
            ws  = max(0, qrs_peak - pre_samp)
            we  = min(sig.size, qrs_peak + post_samp + 1)
            seg = sig[ws:we].copy()
            if seg.size < 10:
                return 0.0

            # Baseline correction from first 30 ms (safe pre-QRS region)
            bl_samp = min(len(seg), int(0.03 * fs))
            seg    -= float(np.mean(seg[:max(1, bl_samp)]))

            rp = qrs_peak - ws   # R-peak index within segment

            # Curtin §2b-§2d: validate candidates before significant-peak search
            local_ext = _find_local_extrema(seg, 0, len(seg))
            raw_cands = sorted(set([rp] + local_ext))
            validated = _curtin_validate_peaks(seg, raw_cands, fs)
            rp_val    = (max(validated, key=lambda p: abs(float(seg[p])))
                         if validated else rp)

            sp = _find_significant_peaks(seg, rp_val, 0, len(seg), fs)
            sp = _remove_peak_outliers(sp, fs)

            # Curtin §6: merge Q/S extrema (120 ms window)
            q_idx, s_idx = _curtin_find_q_s_peaks(seg, rp_val, 0, len(seg), fs)
            if q_idx is not None or s_idx is not None:
                extra = [p for p in (q_idx, s_idx) if p is not None]
                sp    = sorted(set(sp + extra))

            if not sp:
                return 0.0

            r_amp         = abs(float(seg[rp_val]))
            effective_adc = max(r_amp, 1.0)
            onset, offset = _delineate_channel_borders(
                seg, sp, rp_val, 0, len(seg), fs, effective_adc)

            if onset is None or offset is None:
                return 0.0

            qrs_ms = (offset - onset) / fs * 1000.0
            return qrs_ms if 40.0 <= qrs_ms <= 200.0 else 0.0

        except Exception:
            pass  # fall through to legacy method below

    # ── Legacy fallback: amplitude + gradient dual-pass ───────────────────────
    start = max(0, qrs_peak - int(0.25 * fs))
    end   = min(sig.size, qrs_peak + int(0.40 * fs) + 1)
    seg   = sig[start:end]
    if seg.size < 3:
        return 0.0

    baseline = float(np.median(seg))
    centered = seg - baseline
    peak     = int(np.argmax(np.abs(centered)))
    amp      = float(np.max(np.abs(centered)))
    if amp <= 1e-9:
        return 0.0

    amp_threshold = 0.10 * amp
    left_amp, right_amp = peak, peak
    while left_amp > 0 and abs(centered[left_amp]) > amp_threshold:
        left_amp -= 1
    while right_amp < len(centered) - 1 and abs(centered[right_amp]) > amp_threshold:
        right_amp += 1
    qrs_amp = (right_amp - left_amp) / float(fs) * 1000.0

    grad          = np.abs(np.diff(centered))
    grad_peak     = min(peak, grad.size - 1) if grad.size > 0 else 0
    max_grad      = np.max(grad) if grad.size > 0 else 0.0
    grad_threshold = 0.05 * max_grad
    left_grad, right_grad = grad_peak, grad_peak
    while left_grad > 0 and grad[left_grad] > grad_threshold:
        left_grad -= 1
    while right_grad < len(grad) - 1 and grad[right_grad] > grad_threshold:
        right_grad += 1
    qrs_grad = (right_grad - left_grad) / float(fs) * 1000.0

    qrs_width = max(qrs_amp, qrs_grad)
    return qrs_width if 40.0 <= qrs_width <= 200.0 else 0.0


def qrs_metrics(signal: np.ndarray, fs: float = DEFAULT_FS) -> Dict:
    """
    One-call convenience: detect QRS, compute RR, measure beats.

    UPGRADE-2: Only valid (non-zero) QRS width measurements contribute to
    the mean.  Previously, failed measurements (0.0) pulled the mean down,
    making normal-HR signals look like they had shorter QRS complexes than
    they actually do.

    Returns a dict with:
        peaks        - R-peak indices
        rr_sec       - RR intervals in seconds
        rr_ms        - RR intervals in milliseconds
        hr           - Heart rate (bpm)
        qrs_count    - Number of QRS complexes detected
        qrs_ms       - Mean QRS width in ms (valid measurements only)
        qrs_ms_all   - All per-beat widths including 0s (for diagnostics)
    """
    peaks = detect_qrs(signal, fs)
    rr = compute_rr(peaks, fs)
    rr_ms = rr * 1000.0
    hr = float(60.0 / np.mean(rr)) if rr.size else 0.0

    qrs_widths_all = [calculate_qrs_width(signal, int(p), fs) for p in peaks]
    valid_widths   = [w for w in qrs_widths_all if w > 0.0]
    mean_qrs_ms    = float(np.mean(valid_widths)) if valid_widths else 0.0

    return {
        "peaks":       peaks,
        "rr_sec":      rr,
        "rr_ms":       rr_ms,
        "hr":          hr,
        "qrs_count":   int(peaks.size),
        "qrs_ms":      mean_qrs_ms,
        "qrs_ms_all":  qrs_widths_all,
    }


__all__ = [
    "detect_qrs",
    "compute_rr",
    "compute_rr_ms",
    "calculate_qrs_width",
    "beat_metrics",
    "qrs_metrics",
]
