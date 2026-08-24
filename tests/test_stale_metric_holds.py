"""
Stale metric hold regression suite
==================================

Reported with a Fluke simulator: set to 72 bpm, then switched to 3 bpm. The
12-lead page kept displaying **BPM 71 / PR 149 / QRS 92 / QT 317** indefinitely
while all twelve traces were visibly flat, and a report generated in that state
printed the same numbers beside a CONCLUSION of "Asystole".

WHY THE MEASUREMENTS WERE CORRECT AND THE DISPLAY WAS NOT
--------------------------------------------------------
At 3 bpm the RR interval is 20 s. The analysis buffer is HISTORY_LENGTH = 10000
samples, which at 500 Hz is exactly 20 s, so **at most one R-peak** can ever be
in the window. ``calculate_all_ecg_metrics()`` needs three and correctly
returned zero — roughly 9 bpm is the slowest rate it can express at all.

The stale numbers came from two independent "hold last good value" layers, each
of which answered every failed window with the previous measurement and neither
of which had an expiry:

  1. ``ecg/ui/display_updates.py`` — every metric falls back to
     ``_last_valid[...]`` when the incoming value is 0.
  2. ``twelve_lead_test.calculate_ecg_metrics`` — ``pr_interval_raw`` /
     ``qrs_duration_raw`` / ``qt_interval_raw`` fall back to the previous
     attribute, and the median-beat early-return path re-published them on
     every tick without ever writing a new value.

A hold is right for a momentary dropout — one missed beat, one noisy window —
and wrong for a sustained inability to measure. Both layers are now bounded.

Run:
    python -m pytest tests/test_stale_metric_holds.py -v
"""

import os
import subprocess
import sys
import textwrap
import time
import unittest
from unittest.mock import MagicMock

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _read(rel):
    with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as fh:
        return fh.read()


# ══════════════════════════════════════════════════════════════════════════════
# 1. WHY IT CANNOT MEASURE — the arithmetic that makes the hold visible
# ══════════════════════════════════════════════════════════════════════════════

class TestMeasurableRateFloor(unittest.TestCase):

    FS = 500.0
    BUFFER = 10000          # HISTORY_LENGTH

    def _max_peaks(self, bpm):
        return int((self.BUFFER / self.FS) / (60.0 / bpm))

    def test_window_is_twenty_seconds(self):
        self.assertEqual(self.BUFFER / self.FS, 20.0)

    def test_three_bpm_can_hold_only_one_peak(self):
        """The reported case: one peak, and three are needed."""
        self.assertEqual(self._max_peaks(3), 1)

    def test_pipeline_floor_is_about_nine_bpm(self):
        # Three R-peaks in 20 s => 2 RR intervals => 6 bpm is the theoretical
        # floor; allow for the first peak landing anywhere in the window.
        self.assertLess(self._max_peaks(6), 3)
        self.assertGreaterEqual(self._max_peaks(12), 3)

    def test_seventy_two_bpm_is_comfortable(self):
        self.assertGreaterEqual(self._max_peaks(72), 20)


# ══════════════════════════════════════════════════════════════════════════════
# 2. THE DISPLAY HOLD
# ══════════════════════════════════════════════════════════════════════════════

class TestDisplayHoldExpires(unittest.TestCase):

    def setUp(self):
        from ecg.ui import display_updates as du
        self.du = du
        du.reset_metric_holds()
        self.labels = {k: MagicMock() for k in
                       ("heart_rate", "pr_interval", "qrs_duration",
                        "qt_interval", "qtc_interval", "rr_interval",
                        "p_duration")}
        for m in self.labels.values():
            m.text.return_value = ""

    def _show(self, hr, pr, qrs, qt, qtc, rr):
        self.du.update_ecg_metrics_display(
            self.labels, hr, pr, qrs, 0, qt, qtc, 0, 0.0, rr_interval=rr)
        return {k: (v.setText.call_args[0][0].strip()
                    if v.setText.call_args else None)
                for k, v in self.labels.items()}

    def test_hold_window_is_declared_and_finite(self):
        self.assertTrue(hasattr(self.du, "HOLD_MAX_SECONDS"))
        self.assertGreater(self.du.HOLD_MAX_SECONDS, 0)
        self.assertLessEqual(self.du.HOLD_MAX_SECONDS, 10)

    def test_a_brief_dropout_is_smoothed_over(self):
        """One failed window must not blank a working display."""
        self._show(72, 149, 92, 317, 346, 841)
        out = self._show(0, 0, 0, 0, 0, 0)
        self.assertEqual(out["heart_rate"], "72")
        self.assertEqual(out["pr_interval"], "149")

    def test_a_sustained_zero_falls_through_to_zero(self):
        """The Fluke case: 72 bpm, then a rate the pipeline cannot measure."""
        self._show(72, 149, 92, 317, 346, 841)
        # Age every hold past its expiry rather than sleeping.
        self.du._last_valid_at = {
            k: time.time() - (self.du.HOLD_MAX_SECONDS + 1.0)
            for k in self.du._last_valid_at
        }
        out = self._show(0, 0, 0, 0, 0, 0)
        for key in ("heart_rate", "pr_interval", "qrs_duration"):
            with self.subTest(metric=key):
                self.assertEqual(out[key], "0",
                                 f"{key} still showed a stale value")

    def test_expired_hold_is_discarded_not_just_hidden(self):
        self._show(72, 149, 92, 317, 346, 841)
        self.du._last_valid_at = {
            k: time.time() - (self.du.HOLD_MAX_SECONDS + 1.0)
            for k in self.du._last_valid_at
        }
        self.assertFalse(self.du._hold_is_fresh("heart_rate"))
        self.assertNotIn("heart_rate", self.du._last_valid)

    def test_reset_clears_values_and_timestamps(self):
        self._show(72, 149, 92, 317, 346, 841)
        self.du.reset_metric_holds()
        self.assertEqual(self.du._last_valid, {})
        self.assertEqual(self.du._last_valid_at, {})

    def test_recovery_after_expiry(self):
        self._show(72, 149, 92, 317, 346, 841)
        self.du.reset_metric_holds()
        out = self._show(0, 0, 0, 0, 0, 0)
        self.assertEqual(out["heart_rate"], "0")
        out = self._show(58, 160, 88, 400, 410, 1030)
        self.assertEqual(out["heart_rate"], " 58".strip())


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE PAGE-LEVEL HOLD
# ══════════════════════════════════════════════════════════════════════════════

class TestPageHoldSource(unittest.TestCase):

    def setUp(self):
        self.src = _read("src/ecg/twelve_lead_test.py")

    def test_holds_go_through_the_expiring_helper(self):
        for expr in ("self._hold_or_zero('pr_interval', 0)",
                     "self._hold_or_zero('last_qrs_duration', 0)",
                     "self._hold_or_zero('last_qt_interval', 0)"):
            with self.subTest(expr=expr):
                self.assertIn(expr, self.src)

    def test_raw_getattr_fallbacks_are_gone(self):
        for gone in ("pr_interval_raw = getattr(self, 'pr_interval', 0)",
                     "qrs_duration_raw = getattr(self, 'last_qrs_duration', 0)",
                     "qt_interval_raw = getattr(self, 'last_qt_interval', 0)"):
            with self.subTest(gone=gone):
                self.assertNotIn(gone, self.src,
                                 "an unbounded hold was reintroduced")

    def test_median_beat_early_return_expires_its_holds(self):
        """This is the path a very slow rate takes on every tick."""
        idx = self.src.find("if median_beat_ii is None:")
        self.assertGreater(idx, -1)
        block = self.src[idx:idx + 2200]
        self.assertIn("self._hold_or_zero('pr_interval', 0)", block)
        self.assertNotIn("getattr(self, 'pr_interval', 0)", block)

    def test_hold_window_is_declared_and_matches_the_display(self):
        from ecg.ui import display_updates as du
        self.assertIn("PAGE_HOLD_MAX_SECONDS = 4.0", self.src)
        self.assertEqual(du.HOLD_MAX_SECONDS, 4.0,
                         "page and display hold windows must agree, or the two "
                         "layers disagree about whether a value is still valid")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BEHAVIOURAL — real page, 72 bpm -> 3 bpm -> 72 bpm
# ══════════════════════════════════════════════════════════════════════════════

_E2E = textwrap.dedent(r'''
    import os, sys, time
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QStackedWidget
    app = QApplication(sys.argv)
    from ecg.twelve_lead_test import ECGTestPage

    FS = 500.0
    page = ECGTestPage("12 Lead ECG Test", QStackedWidget())
    page._instance_id = "fluke_pytest"
    page.sampler.sampling_rate = FS
    page.PAGE_HOLD_MAX_SECONDS = 0.4      # keep the test quick
    buf = len(page.data[0])

    def fluke(bpm, seed=0):
        rng = np.random.default_rng(seed); t = np.arange(buf)/FS
        sig = np.zeros(buf); rr = 60.0/bpm; k = 0
        while k*rr < buf/FS:
            bt = k*rr + 0.5
            g = lambda mu,s,h: h*np.exp(-0.5*((t-mu)/s)**2)
            sig += (g(bt-0.16,0.022,110)+g(bt-0.02,0.008,-80)+g(bt,0.011,1000)
                    +g(bt+0.025,0.009,-200)+g(bt+0.24,0.045,240))
            k += 1
        sig += rng.normal(0,4,buf) + 2048.0
        for i in range(len(page.data)):
            page.data[i] = sig.astype(np.float32)
        page._lead_connection_state = {ld: True for ld in page.leads}
        page._ll_disconnected = False

    def snap():
        return (getattr(page,'last_heart_rate',0) or 0,
                getattr(page,'pr_interval',0) or 0,
                getattr(page,'last_qrs_duration',0) or 0,
                getattr(page,'last_qt_interval',0) or 0)

    fluke(72, 72); page.calculate_ecg_metrics()
    good = snap()
    assert all(v > 0 for v in good), "no baseline measurement: %r" % (good,)

    fluke(3, 3); page.calculate_ecg_metrics()
    time.sleep(0.6)
    page.calculate_ecg_metrics()
    slow = snap()
    assert all(v == 0 for v in slow), "stale values survived at 3 bpm: %r" % (slow,)

    # The report must not read a held value either.
    m = page.get_current_metrics()
    for k in ("heart_rate", "pr_interval", "qrs_duration"):
        v = str(m.get(k, "0")).strip()
        assert v in ("0", "--", "", "0/0"), "report data stale: %s=%r" % (k, v)

    fluke(72, 7); page.calculate_ecg_metrics()
    back = snap()
    assert all(v > 0 for v in back), "did not recover at 72 bpm: %r" % (back,)
    print("E2E_OK")
''')


class TestFlukeRateChangeOnRealPage(unittest.TestCase):

    def test_values_clear_at_three_bpm_and_recover(self):
        try:
            import PyQt5  # noqa: F401
        except Exception:
            self.skipTest("PyQt5 not importable")

        env = dict(os.environ, PYTHONIOENCODING="utf-8",
                   QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run(
            [sys.executable, "-c", _E2E],
            cwd=_ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if "E2E_OK" not in (proc.stdout or ""):
            self.fail(
                "Fluke rate-change behaviour check failed\n"
                f"--- stdout tail ---\n{(proc.stdout or '')[-2500:]}\n"
                f"--- stderr tail ---\n{(proc.stderr or '')[-2500:]}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 5. THE REPORT MUST NOT BLAME THE CABLE FOR A SLOW HEART
# ══════════════════════════════════════════════════════════════════════════════

class TestLowRateReportWording(unittest.TestCase):

    def setUp(self):
        self.src = _read("src/ecg/ecg_report_generator.py")

    def test_report_reads_the_low_rate_flag(self):
        self.assertIn('getattr(ecg_test_page, "_rate_below_measurable", False)',
                      self.src)

    def test_low_rate_has_its_own_wording(self):
        self.assertIn("Rate below measurable range", self.src)

    def test_low_rate_branch_does_not_advise_checking_the_cable(self):
        idx = self.src.find("if _low_rate and not _asystole:")
        self.assertGreater(idx, -1)
        branch = self.src[idx:idx + 700]
        self.assertIn("Rate below measurable range", branch)
        self.assertNotIn("Please connect device", branch)

    def test_three_distinct_states_exist(self):
        for wording in ("Rate below measurable range",
                        "No cardiac activity detected",
                        "Please connect device"):
            with self.subTest(wording=wording):
                self.assertIn(wording, self.src)

    def test_page_sets_the_low_rate_flag(self):
        page_src = _read("src/ecg/twelve_lead_test.py")
        self.assertIn("self._rate_below_measurable = bool(0 < len(r_peaks) < 3)",
                      page_src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
