"""
Asystole zeroing regression suite
=================================

During asystole every measurement must read 0 — on the 12-lead page, on the
dashboard and in the PDF. Nothing is measurable on a flat trace, and a stale
pre-arrest heart rate displayed beside a flatline is a clinically dangerous
reading.

THE BUG THIS PINS
-----------------
The 12-lead page already had a flat-line guard that zeroed every metric, but it
was gated on ``and not limb_active`` — it only fired when the limb leads were
reported DISCONNECTED. That is exactly backwards for asystole: a flat trace with
the electrodes still on the patient IS asystole, and it was the one case the
guard refused to handle. The pipeline then ran on a flat signal, found no
R-peaks, and every downstream "hold last good value" fallback re-published the
numbers from before the arrest.

Measured before the fix: HR 72, PR 158, QRS 82, QT 338, QTc 370 all persisted
across a transition to a flat trace.

TWO KINDS OF FLAT TRACE
-----------------------
Both zero the metrics, but they mean different things and the report must not
confuse them:

    electrodes OFF, flat  -> a device problem     -> "Please connect device"
    electrodes ON,  flat  -> no cardiac output    -> must NOT tell the operator
                                                     to go check the cable

The page sets ``_asystole_active`` to distinguish them.

The behavioural test runs in a subprocess: the rest of this suite replaces
PyQt5 with MagicMock in ``sys.modules``, so a real Qt widget cannot be built in
the same interpreter.

Run:
    python -m pytest tests/test_asystole_zeroing.py -v
"""

import os
import subprocess
import sys
import textwrap
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")


def _read(rel):
    with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as fh:
        return fh.read()


# ══════════════════════════════════════════════════════════════════════════════
# 1. THE GUARD ITSELF
# ══════════════════════════════════════════════════════════════════════════════

class TestFlatlineGuardFiresOnAsystole(unittest.TestCase):

    def setUp(self):
        self.src = _read("src/ecg/twelve_lead_test.py")

    def test_guard_no_longer_requires_disconnected_leads(self):
        """The `and not limb_active` gate is what hid asystole."""
        self.assertNotIn(
            "or _raw_std_ii < 5.0) and not limb_active", self.src,
            "the flat-line guard must fire on a flat trace whether or not the "
            "leads report connected — otherwise asystole is never zeroed",
        )

    def test_guard_is_driven_by_signal_flatness(self):
        self.assertIn("_signal_is_flat = (", self.src)
        self.assertIn("_is_flat_line_ii = _signal_is_flat", self.src)

    def test_asystole_flag_is_set_and_cleared(self):
        self.assertIn("self._asystole_active = bool(_asystole)", self.src)
        self.assertIn("self._asystole_active = False", self.src)

    def test_every_metric_attribute_is_zeroed(self):
        # Including the ones the original block missed — any value left behind
        # is a pre-arrest number displayed as though it were current.
        for attr in ("last_heart_rate", "last_rr_interval", "pr_interval",
                     "last_qrs_duration", "last_qt_interval",
                     "last_qtc_interval", "last_qtcf_interval",
                     "last_p_duration", "last_st_interval"):
            with self.subTest(attr=attr):
                self.assertIn(f"self.{attr}   = 0".replace("   = 0", ""), self.src)
                self.assertRegex(self.src, rf"self\.{attr}\s*=\s*0")

    def test_pending_deadband_state_is_cleared(self):
        """A held value could otherwise be promoted one tick after recovery."""
        self.assertRegex(self.src, r"self\._pending_qrs_value\s*=\s*None")
        self.assertRegex(self.src, r"self\._pending_qt_value\s*=\s*None")

    def test_smoothing_buffers_are_cleared(self):
        self.assertIn("_qrs_smooth_buffer", self.src)
        self.assertIn("reset_global_qrs_cache", self.src)


# ══════════════════════════════════════════════════════════════════════════════
# 2. THE REPORT MUST NOT CALL ASYSTOLE A CABLE FAULT
# ══════════════════════════════════════════════════════════════════════════════

class TestReportDistinguishesAsystoleFromNoData(unittest.TestCase):

    def setUp(self):
        self.src = _read("src/ecg/ecg_report_generator.py")

    def test_report_reads_the_asystole_flag(self):
        self.assertIn('getattr(ecg_test_page, "_asystole_active", False)', self.src)

    def test_asystole_wording_is_present(self):
        self.assertIn("No cardiac activity detected", self.src)

    def test_connect_device_still_used_for_a_real_no_data_case(self):
        self.assertIn("Please connect device", self.src)

    def test_the_two_messages_are_mutually_exclusive(self):
        """A flat trace with electrodes attached must not advise checking the
        cable — that sends the operator after equipment while the patient has
        no cardiac output."""
        start = self.src.find("if _asystole:")
        self.assertGreater(start, -1)
        branch = self.src[start:start + 400]
        self.assertIn("No cardiac activity detected", branch)
        self.assertNotIn("Please connect device", branch)


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE ALLOW-LIST BYPASS THAT THIS WORK UNCOVERED
# ══════════════════════════════════════════════════════════════════════════════

class TestSafeguardBranchRespectsAllowList(unittest.TestCase):
    """The merge branch appended raw labels straight past the allow-list."""

    def setUp(self):
        sys.path[:0] = [p for p in (_ROOT, _SRC) if p not in sys.path]
        from ecg.ecg_report_generator import (          # noqa: PLC0415
            REPORT_ALLOWED_CONCLUSIONS, _build_metric_conclusions,
            _normalize_report_conclusions, ensure_rate_conclusion)
        self.allowed = set(REPORT_ALLOWED_CONCLUSIONS)
        self._build = _build_metric_conclusions
        self._norm = _normalize_report_conclusions
        self._rate = ensure_rate_conclusion

    def _merge_branch(self, hr, pr, qrs, qtc):
        d = {"HR": hr, "PR": pr, "QRS": qrs, "QTc": qtc}
        return self._rate(self._norm(list(self._build(d))), d)

    def test_previously_leaking_labels_are_filtered(self):
        for hr, pr, qrs, qtc, leaked in (
            (30, 0, 0, 0, "Third-degree AV Block"),
            (73, 152, 116, 349, "Borderline Wide QRS"),
            (75, 260, 95, 420, "First-degree AV Block (Prolonged PR)"),
            (75, 150, 95, 520, "Long QT Syndrome"),
        ):
            with self.subTest(was=leaked):
                self.assertNotIn(leaked, self._merge_branch(hr, pr, qrs, qtc))

    def test_merge_branch_never_emits_an_off_list_label(self):
        offenders = set()
        for hr in (5, 10, 30, 45, 65, 85, 110, 160, 220):
            for pr in (0, 100, 152, 260):
                for qrs in (0, 95, 116, 140):
                    for qtc in (0, 430, 470, 520):
                        offenders |= {c for c in self._merge_branch(hr, pr, qrs, qtc)
                                      if c not in self.allowed}
        self.assertEqual(offenders, set(), f"off-list labels leaked: {offenders}")

    def test_source_no_longer_appends_raw_metric_conclusions(self):
        src = _read("src/ecg/ecg_report_generator.py")
        self.assertNotIn("dashboard_conclusions.append(mc)", src,
                         "raw append bypasses the allow-list")


# ══════════════════════════════════════════════════════════════════════════════
# 4. BEHAVIOURAL — real ECGTestPage, in a clean interpreter
# ══════════════════════════════════════════════════════════════════════════════

_E2E = textwrap.dedent(r'''
    import os, sys
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    sys.path.insert(0, os.path.join(os.getcwd(), "src"))
    import numpy as np
    from PyQt5.QtWidgets import QApplication, QStackedWidget
    app = QApplication(sys.argv)
    from ecg.twelve_lead_test import ECGTestPage

    FS = 500.0
    page = ECGTestPage("12 Lead ECG Test", QStackedWidget())
    page._instance_id = "asystole_pytest"
    page.sampler.sampling_rate = FS
    buf = len(page.data[0])
    METRICS = ("last_heart_rate","pr_interval","last_qrs_duration",
               "last_qt_interval","last_qtc_interval","last_rr_interval")

    def real(seed=0):
        rng = np.random.default_rng(seed); t = np.arange(buf)/FS
        sig = np.zeros(buf); rr = 60.0/72; k = 0
        while k*rr < buf/FS:
            bt = k*rr + 0.3
            g = lambda mu,s,h: h*np.exp(-0.5*((t-mu)/s)**2)
            sig += (g(bt-0.16,0.022,120)+g(bt-0.02,0.008,-80)+g(bt,0.011,1000)
                    +g(bt+0.025,0.009,-200)+g(bt+0.24,0.045,250))
            k += 1
        sig += rng.normal(0,6,buf) + 2048.0
        for i in range(len(page.data)):
            page.data[i] = sig.astype(np.float32)

    def asystole():
        rng = np.random.default_rng(1)
        for i in range(len(page.data)):
            page.data[i] = (2048.0 + rng.normal(0,1.2,buf)).astype(np.float32)
        page._lead_connection_state = {ld: True for ld in page.leads}
        page._ll_disconnected = False

    real(); page.calculate_ecg_metrics()
    assert (getattr(page,"last_heart_rate",0) or 0) > 0, "no baseline HR"

    asystole(); page.calculate_ecg_metrics()
    stale = [(m, getattr(page,m,None)) for m in METRICS
             if float(getattr(page,m,0) or 0) != 0.0]
    assert not stale, "stale pre-arrest values: %r" % (stale,)
    assert getattr(page,"_asystole_active",False) is True, "_asystole_active not set"
    for k in ("heart_rate","pr_interval","qrs_duration"):
        if k in page.metric_labels:
            txt = page.metric_labels[k].text()
            d = "".join(c for c in txt if c.isdigit())
            assert not d or int(d) == 0, "label %s shows %r" % (k, txt)

    real(seed=2); page.calculate_ecg_metrics()
    assert (getattr(page,"last_heart_rate",0) or 0) > 0, "did not recover"
    assert getattr(page,"_asystole_active",True) is False, "flag stuck on"
    print("E2E_OK")
''')


class TestAsystoleOnRealPage(unittest.TestCase):

    def test_metrics_zero_then_recover(self):
        try:
            import PyQt5  # noqa: F401
        except Exception:
            self.skipTest("PyQt5 not importable")

        env = dict(os.environ, PYTHONIOENCODING="utf-8", QT_QPA_PLATFORM="offscreen")
        proc = subprocess.run(
            [sys.executable, "-c", _E2E],
            cwd=_ROOT, env=env, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=600,
        )
        if "E2E_OK" not in (proc.stdout or ""):
            self.fail(
                "asystole behaviour check failed\n"
                f"--- stdout tail ---\n{(proc.stdout or '')[-2500:]}\n"
                f"--- stderr tail ---\n{(proc.stderr or '')[-2500:]}"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
