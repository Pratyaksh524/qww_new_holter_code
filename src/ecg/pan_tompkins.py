import numpy as np
from scipy.signal import butter, filtfilt


def _moving_average_kotlin(signal: np.ndarray, window_size: int) -> np.ndarray:
    """Moving average with the same left padding used by ECGAnalyzer.kt."""
    x = np.asarray(signal, dtype=float)
    if window_size <= 1 or x.size < window_size:
        return x.copy()

    result = []
    current_sum = float(np.sum(x[:window_size]))
    result.append(current_sum / float(window_size))
    for i in range(window_size, x.size):
        current_sum += float(x[i]) - float(x[i - window_size])
        result.append(current_sum / float(window_size))

    padding = [result[0]] * (window_size - 1)
    return np.asarray(padding + result, dtype=float)


def _pt_bandpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    nyq = 0.5 * float(fs)
    low = max(5.0 / nyq, 0.001)
    high = min(15.0 / nyq, 0.99)
    if not np.isfinite(low) or not np.isfinite(high) or low <= 0 or high >= 1 or low >= high:
        return signal
    b, a = butter(2, [low, high], btype="band")
    try:
        if len(signal) < max(len(a), len(b)) * 3:
            from scipy.signal import lfilter
            return lfilter(b, a, signal)
        return filtfilt(b, a, signal)
    except Exception:
        return signal


def pan_tompkins(ecg, fs=500):
    """
    Pan-Tompkins R-peak detection matching ECGAnalyzer.kt.

    Pipeline:
      5-15 Hz bandpass -> derivative -> square -> 100 ms moving average
      -> threshold = integrated.average() * 1.5 -> 200 ms refractory
      -> true R index = max source-signal sample within +/- 50 ms.
    """
    x = np.asarray(ecg, dtype=float)
    if x.size < int(fs) or not np.isfinite(fs) or fs <= 0:
        return np.array([], dtype=int)

    bp_filtered = _pt_bandpass_filter(x, float(fs))
    derivative = np.diff(bp_filtered)
    squared = derivative ** 2

    window_size = max(1, int(float(fs)) // 10)
    integrated = _moving_average_kotlin(squared, window_size)
    if integrated.size < (2 * window_size + 1):
        return np.array([], dtype=int)

    threshold = float(np.mean(integrated)) * 1.5
    signal_amplitude = float(np.max(x) - np.min(x))
    if signal_amplitude < 0.08:
        return np.array([], dtype=int)

    refractory_period = max(1, int(float(fs) * 0.2))
    peaks = []
    i = window_size
    while i < integrated.size - window_size:
        if (
            integrated[i] > threshold
            and integrated[i] > integrated[i - 1]
            and integrated[i] > integrated[i + 1]
        ):
            search_start = max(0, i - window_size // 2)
            search_end = min(x.size - 1, i + window_size // 2)
            peak_idx = search_start + int(np.argmax(x[search_start:search_end + 1]))
            if not peaks or peak_idx - peaks[-1] > refractory_period:
                peaks.append(int(peak_idx))
            i += window_size
        i += 1

    return np.asarray(peaks, dtype=int)


def detectRPeaks(filtered_signal, fs=500):
    """
    Backward-compatible alias for older callers that still import
    detectRPeaks from this module.
    """
    return pan_tompkins(filtered_signal, fs=fs)
