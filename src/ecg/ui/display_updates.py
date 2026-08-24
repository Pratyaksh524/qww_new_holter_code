"""ECG metrics display update functions — unified & corrected.

Changes vs previous version
────────────────────────────
FIX-D1: Added 'rr_interval' label support — RR was calculated but never rendered.
FIX-D2: Added 'p_duration' label support — p_duration was passed but silently dropped
         because the key was never in metric_labels.
FIX-D3: QT/QTc display format now robust — shows "QT/QTc" or just "QTc" if QT missing,
         and never crashes on None values.
FIX-D4: get_current_metrics_from_labels now returns 'rr_interval' and 'qt_interval'
         keys so dashboard / report code receives complete data.
FIX-D5: Throttle remains 0.3 s; force_immediate path properly resets last_update_ts=0
         before calling (handled in twelve_lead_test.py wrapper, unchanged here).
FIX-D6: Physiological clamping — values outside clinical limits are clamped to the
         last known good value so the display never shows impossible numbers.
FIX-D7: Digit-level change detection — a label is only rewritten when its text actually
         changes, giving the "modern clock" appearance (no flicker, steady digits).
"""

import time
from typing import Dict, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Physiological limits (ms).  Values outside these bounds are treated as noise
# and the display holds the last valid reading instead.
# ─────────────────────────────────────────────────────────────────────────────
_LIMITS = {
    # metric key  : (min_ms, max_ms)
    'heart_rate'  : (30,   300),   # BPM
    'pr_interval' : (80,   240),   # ms
    'qrs_duration': (40,   200),   # ms
    # NOTE: QT lower bound is handled dynamically in update_ecg_metrics_display
    # because physiologic QT gets shorter as HR rises (e.g. 180-190 ms at ~240 BPM).
    'qt_interval' : (150,  600),   # ms (fallback bound)
    'qtc_interval': (200,  600),   # ms (Bazett / Fridericia)
    'rr_interval' : (200, 2000),   # ms
    'p_duration'  : (40,   160),   # ms
}

# Per-process memory of the last valid value for each metric.
# Key: metric_name  →  last valid int (or string for composite fields)
_last_valid: Dict[str, object] = {}

# When each metric last received a real (non-zero) value.
# Key: metric_name  →  time.time() of the last good reading
_last_valid_at: Dict[str, float] = {}

# How long a held value may keep being displayed once the incoming value has
# gone to zero.
#
# The hold exists so a single missed beat or one noisy analysis window does not
# blank the display. It previously had NO expiry, which meant a genuine change
# in the patient's rhythm was masked indefinitely: a simulator moved from 72 bpm
# to 3 bpm and the page still read 71 while every trace was flat. At 3 bpm the
# 20 s analysis buffer can hold only one R-peak, so the pipeline correctly
# reported zero — the display was the only thing still claiming 71.
#
# 4 s is longer than any plausible single-window dropout at a rate the pipeline
# can measure at all (it needs 3 R-peaks, i.e. ~9 bpm or faster), and short
# enough that a real bradycardia or arrest shows up promptly.
HOLD_MAX_SECONDS: float = 4.0


def _hold_is_fresh(key: str) -> bool:
    """True while `key`'s last good value may still stand in for a zero."""
    stamp = _last_valid_at.get(key)
    if stamp is None:
        return False
    if (time.time() - stamp) <= HOLD_MAX_SECONDS:
        return True
    # Expired — drop it so the display falls through to 0 from now on.
    _last_valid.pop(key, None)
    _last_valid_at.pop(key, None)
    return False


def _held(key: str, default=0):
    """The held value for `key`, or `default` once the hold has expired."""
    return _last_valid.get(key, default) if _hold_is_fresh(key) else default


def reset_metric_holds() -> None:
    """Forget every held value (used when signal is lost or a page restarts)."""
    _last_valid.clear()
    _last_valid_at.clear()


def _clamp(key: str, value: int) -> Optional[int]:
    """Return value if within physiological limits, else last valid or None."""
    lo, hi = _LIMITS.get(key, (0, 99999))
    if lo <= value <= hi:
        _last_valid[key] = value
        _last_valid_at[key] = time.time()
        return value
    # Out of range — stand in with the last good value while the hold is fresh.
    return _held(key, None)


def _clamp_qt_for_hr(qt_ms: int, heart_rate: Optional[float]) -> Optional[int]:
    """Clamp QT with an HR-adaptive lower bound to avoid false freezing at high HR."""
    hr = float(heart_rate) if isinstance(heart_rate, (int, float)) else None
    if hr is not None and hr > 0:
        # Mirror the clinical QT acceptance logic used by median-beat measurement.
        qt_min = 150 if hr > 200 else (170 if hr > 150 else 200)
    else:
        qt_min = _LIMITS['qt_interval'][0]

    qt_max = _LIMITS['qt_interval'][1]
    if qt_min <= qt_ms <= qt_max:
        _last_valid['qt_interval'] = qt_ms
        _last_valid_at['qt_interval'] = time.time()
        return qt_ms
    return _held('qt_interval', None)


def _set_if_changed(label, text: str):
    """Write text to label ONLY if it differs from what is currently shown.
    This stops any Qt repaint / flicker when the digit has not actually changed —
    exactly like a modern split-flap / digital clock display.
    """
    if label.text() != text:
        label.setText(text)


# ─────────────────────────────────────────────────────────────────────────────
# Main display-update function
# ─────────────────────────────────────────────────────────────────────────────
def update_ecg_metrics_display(
        metric_labels: Dict,
        heart_rate: int,
        pr_interval: int,
        qrs_duration: int,
        p_duration: int,
        qt_interval: Optional[float] = None,
        qtc_interval: Optional[int] = None,
        qtcf_interval: Optional[int] = None,
        last_update_ts: Optional[float] = None,
        rr_interval: Optional[float] = None,       # FIX-D1: new param
        skip_heart_rate: bool = False,              # HolterBPM: bypass old HR path
) -> float:
    """Update the ECG metrics display in the UI.

    Supported metric_labels keys:
        'heart_rate'   – BPM (int)
        'rr_interval'  – RR interval in ms (float)   ← FIX-D1
        'pr_interval'  – PR interval in ms (int)
        'qrs_duration' – QRS duration in ms (int)
        'p_duration'   – P-wave duration in ms (int) ← FIX-D2
        'qtc_interval' – Shows "QT/QTc" or "QTc" text
        'time_elapsed' – Timer (updated separately, not touched here)

    Returns:
        Updated timestamp (float).
    """
    try:
        current_time = time.time()
        # Throttle: max one display refresh per 0.3 s
        # last_update_ts=None (or 0.0) means "force now"
        if last_update_ts and current_time - last_update_ts < 0.3:
            return last_update_ts

        if not metric_labels:
            return current_time

        # ── BPM ──────────────────────────────────────────────────────────────
        if not skip_heart_rate and 'heart_rate' in metric_labels:
            raw_hr = int(round(heart_rate)) if isinstance(heart_rate, (int, float)) else 0
            if raw_hr > 0:
                hr_val = _clamp('heart_rate', raw_hr)
                if hr_val and hr_val > 0:
                    _set_if_changed(metric_labels['heart_rate'], f"{hr_val:3d}")
            elif _held('heart_rate', 0) > 0:
                _set_if_changed(metric_labels['heart_rate'], f"{_held('heart_rate'):3d}")
            else:
                # Hold expired and nothing new: say zero rather than keep
                # showing a rate the patient no longer has.
                _set_if_changed(metric_labels['heart_rate'], "  0")

        # ── RR Interval ──────────────────────────────────────────────────────
        if 'rr_interval' in metric_labels:
            if rr_interval is not None and rr_interval > 0:
                rr_val = int(round(rr_interval))
                clamped = _clamp('rr_interval', rr_val)
                if clamped:
                    _set_if_changed(metric_labels['rr_interval'], f"{clamped}")
            elif _held('rr_interval', 0) > 0:
                _set_if_changed(metric_labels['rr_interval'], f"{_held('rr_interval')}")
            else:
                _set_if_changed(metric_labels['rr_interval'], "0")

        # ── PR Interval ───────────────────────────────────────────────────────
        if 'pr_interval' in metric_labels:
            raw_pr = int(round(pr_interval)) if isinstance(pr_interval, (int, float)) else 0
            if raw_pr > 0:
                pr_val = _clamp('pr_interval', raw_pr)
                if pr_val and pr_val > 0:
                    _set_if_changed(metric_labels['pr_interval'], f"{pr_val:3d}")
            elif _held('pr_interval', 0) > 0:
                _set_if_changed(metric_labels['pr_interval'], f"{_held('pr_interval'):3d}")
            else:
                _set_if_changed(metric_labels['pr_interval'], "  0")

        # ── QRS Duration ──────────────────────────────────────────────────────
        if 'qrs_duration' in metric_labels:
            raw_qrs = int(round(qrs_duration)) if isinstance(qrs_duration, (int, float)) else 0
            if raw_qrs > 0:
                qrs_val = _clamp('qrs_duration', raw_qrs)
                if qrs_val and qrs_val > 0:
                    _set_if_changed(metric_labels['qrs_duration'], f"{qrs_val:3d}")
            elif _held('qrs_duration', 0) > 0:
                _set_if_changed(metric_labels['qrs_duration'], f"{_held('qrs_duration'):3d}")
            else:
                _set_if_changed(metric_labels['qrs_duration'], "  0")

        # ── P Duration ───────────────────────────────────────────────────────
        if 'p_duration' in metric_labels:
            if isinstance(p_duration, (int, float)) and p_duration > 0:
                p_val = _clamp('p_duration', int(round(p_duration)))
                if p_val:
                    _set_if_changed(metric_labels['p_duration'], f"{p_val}")
            elif _held('p_duration', 0) > 0:
                _set_if_changed(metric_labels['p_duration'], f"{_held('p_duration')}")
            else:
                _set_if_changed(metric_labels['p_duration'], "0")

        # ── ST ───────────────────────────────────────────────────────────────
        if 'st_interval' in metric_labels:
            _set_if_changed(metric_labels['st_interval'], "0")

        # ── QT / QTc ──────────────────────────────────────────────────────────
        if 'qtc_interval' in metric_labels:
            qt_ok = qt_interval is not None and isinstance(qt_interval, (int, float)) and qt_interval > 0
            qtc_ok = qtc_interval is not None and isinstance(qtc_interval, (int, float)) and qtc_interval > 0

            parts = []
            if qt_ok:
                qt_int = int(round(qt_interval))
                qt_clamped = _clamp_qt_for_hr(qt_int, heart_rate)
                if qt_clamped is not None:
                    parts.append(f"{qt_clamped}")
            if qtc_ok:
                qtc_int = int(round(qtc_interval))
                qtc_clamped = _clamp('qtc_interval', qtc_int)
                if qtc_clamped is not None and qtc_clamped > 0:
                    parts.append(f"{qtc_clamped}")

            if parts:
                txt = "/".join(parts)
                _last_valid['qtc_interval'] = txt
                _last_valid_at['qtc_interval'] = time.time()
                _set_if_changed(metric_labels['qtc_interval'], txt)
            elif _held('qtc_interval', None):
                _set_if_changed(metric_labels['qtc_interval'], str(_held('qtc_interval')))
            else:
                _set_if_changed(metric_labels['qtc_interval'], "0/0")

        return current_time

    except Exception as e:
        print(f" ⚠️ update_ecg_metrics_display error: {e}")
        return last_update_ts if last_update_ts else time.time()


def get_current_metrics_from_labels(
        metric_labels: Dict,
        data: list = None,
        last_heart_rate: Optional[int] = None,
        sampler=None,
) -> Dict[str, str]:
    """Get current ECG metrics for dashboard / report from UI labels.

    FIX-D4: Returns 'rr_interval' and 'qt_interval' keys that were
             previously missing — dashboard and report code now receive
             the complete metric set.

    Returns:
        Dict[str, str] — all values as strings (empty string = not available).
    """
    try:
        metrics: Dict[str, str] = {}
        if data is None:
            data = []

        # ── Signal quality check ──────────────────────────────────────────────
        has_real_signal = False
        if len(data) > 1:
            import numpy as np
            lead_ii = data[1]
            if (len(lead_ii) >= 100
                    and not np.all(lead_ii == 0)
                    and np.std(lead_ii) >= 0.1):
                has_real_signal = True

        # ── BPM ──────────────────────────────────────────────────────────────
        if metric_labels and 'heart_rate' in metric_labels:
            hr_text = (metric_labels['heart_rate'].text()
                       .replace('BPM', '').replace('bpm', '').strip())
            if hr_text and hr_text not in ('00', '--', '0', ''):
                metrics['heart_rate'] = hr_text
            elif has_real_signal and last_heart_rate and last_heart_rate > 0:
                metrics['heart_rate'] = str(last_heart_rate)
            else:
                metrics['heart_rate'] = "--"
        elif has_real_signal and last_heart_rate and last_heart_rate > 0:
            metrics['heart_rate'] = str(last_heart_rate)
        else:
            metrics['heart_rate'] = "--"

        if not metric_labels:
            return metrics

        # ── RR Interval ───────────────────────────────────────────────────── FIX-D4
        if 'rr_interval' in metric_labels:
            metrics['rr_interval'] = (
                metric_labels['rr_interval'].text()
                .replace('ms', '').strip()
            )
        else:
            metrics['rr_interval'] = "--"

        # ── PR Interval ───────────────────────────────────────────────────────
        if 'pr_interval' in metric_labels:
            metrics['pr_interval'] = (
                metric_labels['pr_interval'].text()
                .replace('ms', '').strip()
            )
        else:
            metrics['pr_interval'] = "--"

        # ── QRS Duration ──────────────────────────────────────────────────────
        if 'qrs_duration' in metric_labels:
            metrics['qrs_duration'] = (
                metric_labels['qrs_duration'].text()
                .replace('ms', '').strip()
            )
        else:
            metrics['qrs_duration'] = "--"

        # ── P Duration ────────────────────────────────────────────────────────
        if 'p_duration' in metric_labels:
            metrics['p_duration'] = (
                metric_labels['p_duration'].text()
                .replace('ms', '').strip()
            )
        else:
            metrics['p_duration'] = "--"

        # ── ST (legacy) ───────────────────────────────────────────────────────
        if 'st_interval' in metric_labels:
            metrics['st_interval'] = (
                metric_labels['st_interval'].text().strip()
                .replace('ms', '').replace('mV', '').strip()
            )
        else:
            metrics['st_interval'] = "--"

        # ── QT / QTc ─────────────────────────────────────────────────────────
        if 'qtc_interval' in metric_labels:
            raw = metric_labels['qtc_interval'].text().strip().replace('ms', '')
            metrics['qtc_interval'] = raw
            # FIX-D4: also split into qt_interval / qtc_interval if "QT/QTc" format
            if raw in ('', '--', '--/--'):
                metrics['qt_interval'] = "--"
                metrics['qtc_interval'] = "--"
            elif '/' in raw:
                parts = raw.split('/')
                metrics['qt_interval']  = parts[0].strip()
                metrics['qtc_interval'] = parts[1].strip()
            else:
                metrics['qt_interval'] = "--"
        else:
            metrics['qt_interval'] = "--"
            metrics['qtc_interval'] = "--"

        # ── Time elapsed ──────────────────────────────────────────────────────
        if 'time_elapsed' in metric_labels:
            metrics['time_elapsed'] = metric_labels['time_elapsed'].text()

        # ── Sampling rate ─────────────────────────────────────────────────────
        if sampler and hasattr(sampler, 'sampling_rate') and sampler.sampling_rate > 0:
            metrics['sampling_rate'] = f"{sampler.sampling_rate:.1f}"
        else:
            metrics['sampling_rate'] = "--"

        return metrics

    except Exception as e:
        print(f" ⚠️ get_current_metrics_from_labels error: {e}")
        return {}
