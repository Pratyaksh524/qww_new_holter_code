"""Lead-Off Detection Module

This module provides lead-off detection functionality for ECG signals.
Lead-off detection identifies when ECG electrodes are disconnected or have poor contact.

Calibrated for 12-bit ADC (0-4096) at 500 Hz sampling rate.
"""

import numpy as np
from typing import Dict

# Thresholds calibrated against 857 one-second windows taken from clean 12-lead
# captures on this hardware. The worst clean window measured 3444 ADC p-p and a
# variance of 360,460, so anything at or below those figures is normal ECG and
# must not be reported as a disconnected lead.
LEAD_OFF_THRESHOLDS = {
    'amplitude_min': 20,        # < 0.1 mV p-p over the window → flatline
    'variance_min':  4,         # near-zero variance → flatline
    'variance_max':  4000000,   # std ≈ 2000 ADC — half of full scale, unmistakable noise
    'pinned_fraction': 0.15,    # ≥15% of the window held at one value → pinned/clipped
    'pinned_tolerance': 1.0,    # ADC counts two samples may differ and still count as held
}


def _longest_held_run(window: np.ndarray, tolerance: float):
    """
    Longest run of samples that do not move by more than `tolerance`.

    Returns (length, value_held). The value matters: a clipped signal is held at
    the extreme of its own range, whereas a resting isoelectric segment is held
    somewhere in the middle — only the former is evidence of a lead problem.
    """
    if window.size < 2:
        return int(window.size), float(window[0]) if window.size else 0.0
    moved = np.flatnonzero(np.abs(np.diff(window)) > tolerance)
    if moved.size == 0:
        return int(window.size), float(np.median(window))
    boundaries = np.concatenate(([-1], moved, [window.size - 1]))
    lengths = np.diff(boundaries)
    longest = int(lengths.argmax())
    start = int(boundaries[longest]) + 1
    stop = int(boundaries[longest + 1]) + 1
    return int(lengths[longest]), float(np.median(window[start:stop]))


def detect_lead_off(signal: np.ndarray, sampling_rate: float = 500, window_size: float = 1.0) -> bool:
    """
    Detect if an ECG lead is disconnected (lead-off condition).

    A disconnected lead shows one of three signatures:

    1. Flatline — the trace stops moving (tiny peak-to-peak span and variance).
    2. Pinned — the signal is held at one value for a sustained stretch, which is
       what ADC clipping against a rail looks like.
    3. Runaway noise — variance far beyond anything a heart produces.

    Deliberately NOT used as evidence of lead-off:

    * A large peak-to-peak span. Chest leads on this hardware routinely exceed
      3000 ADC p-p within one second on a tall QRS; the old amplitude_max of 3000
      flagged healthy V3/V4 as saturated.
    * Any absolute low/high ADC value. III, aVR and aVF are derived leads and are
      legitimately negative — aVR essentially always is — so the old
      `min_val <= 10` rule marked them disconnected for the entire recording.

    Both of those produced flatlined leads in the live view, and because the
    caller substitutes a constant for a lead it believes is off, they also wrote
    flat sections into recordings and printed reports.

    Args:
        signal: ECG signal in ADC counts (raw leads unsigned, derived leads signed)
        sampling_rate: Sampling rate in Hz (default: 500)
        window_size: Window size in seconds (default: 1.0)

    Returns:
        bool: True if the lead looks disconnected, False if connected
    """
    window_samples = int(window_size * sampling_rate)

    if len(signal) < window_samples:
        return False

    window = np.asarray(signal[-window_samples:], dtype=float)
    if window.size == 0 or not np.all(np.isfinite(window)):
        return True  # missing or non-finite samples — nothing usable arriving

    # Check 1: flatline — no meaningful movement at all
    if np.ptp(window) < LEAD_OFF_THRESHOLDS['amplitude_min']:
        return True
    variance = float(np.var(window))
    if variance < LEAD_OFF_THRESHOLDS['variance_min']:
        return True

    # Check 2: runaway noise
    if variance > LEAD_OFF_THRESHOLDS['variance_max']:
        return True

    # Check 3: pinned/clipped — held at one value for a sustained stretch, and
    # that value is at the edge of the window's own range. Requiring the extreme
    # is what separates clipping from a quiet isoelectric segment, which also
    # stops moving but sits mid-range. Polarity-agnostic, so raw and derived
    # leads are treated alike without banning any particular ADC value.
    held, held_value = _longest_held_run(window, LEAD_OFF_THRESHOLDS['pinned_tolerance'])
    if held >= LEAD_OFF_THRESHOLDS['pinned_fraction'] * window.size:
        span = float(np.ptp(window))
        edge_margin = max(LEAD_OFF_THRESHOLDS['pinned_tolerance'], 0.02 * span)
        at_extreme = (
            held_value <= float(np.min(window)) + edge_margin
            or held_value >= float(np.max(window)) - edge_margin
        )
        if at_extreme:
            return True

    return False  # Lead connected


def check_all_leads_quality(lead_data: Dict[str, np.ndarray], sampling_rate: float = 500) -> Dict[str, str]:
    """
    Check lead quality for all ECG leads
    
    Args:
        lead_data: Dictionary mapping lead names to signal arrays
        sampling_rate: Sampling rate in Hz (default: 500)
    
    Returns:
        Dictionary mapping lead names to quality status ('OK' or 'OFF')
    
    Example:
        lead_data = {
            'I': np.array([...]),
            'II': np.array([...]),
            'III': np.array([...]),
            ...
        }
        quality = check_all_leads_quality(lead_data)
        # Returns: {'I': 'OK', 'II': 'OFF', 'III': 'OK', ...}
    """
    lead_quality = {}
    
    for lead_name, signal in lead_data.items():
        try:
            is_off = detect_lead_off(signal, sampling_rate)
            lead_quality[lead_name] = 'OFF' if is_off else 'OK'
        except Exception as e:
            print(f" Error checking lead {lead_name}: {e}")
            lead_quality[lead_name] = 'UNKNOWN'
    
    return lead_quality


def get_lead_quality_summary(lead_quality: Dict[str, str]) -> str:
    """
    Get a summary of lead quality status
    
    Args:
        lead_quality: Dictionary mapping lead names to quality status
    
    Returns:
        Human-readable summary string
    
    Example:
        quality = {'I': 'OK', 'II': 'OFF', 'III': 'OK'}
        summary = get_lead_quality_summary(quality)
        # Returns: "1 lead disconnected (II)"
    """
    off_leads = [name for name, status in lead_quality.items() if status == 'OFF']
    
    if len(off_leads) == 0:
        return "All leads connected"
    elif len(off_leads) == 1:
        return f"1 lead disconnected ({off_leads[0]})"
    else:
        return f"{len(off_leads)} leads disconnected ({', '.join(off_leads)})"
