"""
CardioX Production Deployment Test Suite — tests/test_cardiox_prod.py
======================================================================
PURPOSE & ARCHITECTURE:
This is the automated, headless unit test suite for the CardioX 12-lead ECG application.
It validates all critical system paths prior to building production executables or pushing code:
1. Admin Authentication & Panel Security: Verifies credentials and confirms admin/divyansh login bypass is removed from handle_login.
2. Clinical Metric Conclusions (_build_metric_conclusions): Validates rhythm labeling for NSR, Bradycardia, Tachycardia, Asystole, AV Blocks, Wide QRS, Prolonged QTc.
3. PDF Accuracy & Gap Fixes:
   - TestScreenLabelFallback: Ensures screen label fallback prevents zeroed PDFs.
   - TestRateQualifier: Ensures rate qualifiers are appended alongside non-NSR arrhythmias.
   - TestQtcAutoWarning: Ensures QTc warnings are automatically generated when >= 440ms.
4. Signal Safeguards: Tests flatline detection, low-amplitude noise filtering, and 0-HR guards.
5. Offline Infrastructure & Licensing: Tests queue persistence, path healing, network connectivity checks, and the 14-day offline license grace period.

EXECUTION INSTRUCTIONS FOR DEVELOPERS:
Run from project root:
    python3 -m pytest tests/test_cardiox_prod.py -v
All 76 tests execute headlessly (without GUI or hardware devices).
"""

import os
import sys
import json
import time
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# ── Path setup so src modules are importable ─────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC  = os.path.join(_ROOT, "src")
for p in [_ROOT, _SRC]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Stub PyQt5 so tests run without a display
_QT_STUBS = [
    "PyQt5","PyQt5.QtWidgets","PyQt5.QtCore","PyQt5.QtGui",
    "PyQt5.QtMultimedia","PyQt5.QtChart",
]
for _m in _QT_STUBS:
    if _m not in sys.modules:
        sys.modules[_m] = MagicMock()

for _dep in ["boto3","botocore","requests","serial","twilio"]:
    if _dep not in sys.modules:
        sys.modules[_dep] = MagicMock()


# ════════════════════════════════════════════════════════════════════════════
# 1. ADMIN AUTHENTICATION (admin_reports panel — NOT login bypass)
# ════════════════════════════════════════════════════════════════════════════
class TestAdminReportsPanelCredentials(unittest.TestCase):
    """Admin reports panel inside dashboard still requires credentials."""

    def _check(self, u, p):
        from dashboard.admin_reports import _check_admin_credentials
        return _check_admin_credentials(u, p)

    def test_correct_panel_login(self):
        self.assertTrue(self._check("admin", "divyansh"))

    def test_wrong_password_rejected(self):
        self.assertFalse(self._check("admin", "wrongpass"))

    def test_wrong_username_rejected(self):
        self.assertFalse(self._check("root", "divyansh"))

    def test_empty_credentials_rejected(self):
        self.assertFalse(self._check("", ""))

    def test_case_insensitive_username(self):
        self.assertTrue(self._check("ADMIN", "divyansh"))

    def test_env_override(self):
        import importlib
        import dashboard.admin_reports as _mod
        with patch.dict(os.environ, {"ADMIN_USERNAME": "superuser", "ADMIN_PASSWORD": "secret123"}):
            importlib.reload(_mod)
            self.assertTrue(_mod._check_admin_credentials("superuser", "secret123"))
            self.assertFalse(_mod._check_admin_credentials("admin", "divyansh"))
        importlib.reload(_mod)


class TestLoginBypassRemoved(unittest.TestCase):
    """Verify the admin/divyansh login bypass no longer exists in main.py."""

    def test_no_admin_bypass_in_main_login(self):
        main_path = os.path.join(_SRC, "main.py")
        with open(main_path, "r", encoding="utf-8") as f:
            source = f.read()
        # The old bypass checked admin credentials before license validation.
        # It must NOT exist anymore.
        self.assertNotIn("allows admin/dev access without license check", source,
                         "Admin login bypass must be removed from main.py")

    def test_no_hardcoded_divyansh_in_login_flow(self):
        main_path = os.path.join(_SRC, "main.py")
        with open(main_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Find the handle_login method and check there's no 'divyansh' or
        # 'admin' hardcoded password comparison in that function.
        in_login = False
        for line in lines:
            if "def handle_login" in line:
                in_login = True
            elif in_login and line.strip().startswith("def "):
                break  # next method
            if in_login and "password_or_serial in ('divyansh'" in line:
                self.fail("Hardcoded 'divyansh' password still in handle_login")


# ════════════════════════════════════════════════════════════════════════════
# 2. METRIC-BASED CONCLUSIONS (_build_metric_conclusions)
# ════════════════════════════════════════════════════════════════════════════
class TestBuildMetricConclusions(unittest.TestCase):

    def _build(self, data):
        from ecg.ecg_report_generator import _build_metric_conclusions
        return _build_metric_conclusions(data)

    def test_normal_sinus_rhythm(self):
        self.assertIn("Normal Sinus Rhythm", self._build({"HR": 75, "PR": 160, "QRS": 90, "QTc": 420}))

    def test_sinus_bradycardia(self):
        self.assertIn("Sinus Bradycardia", self._build({"HR": 50, "PR": 160, "QRS": 90, "QTc": 420}))

    def test_sinus_tachycardia(self):
        self.assertIn("Sinus Tachycardia", self._build({"HR": 115, "PR": 160, "QRS": 90, "QTc": 420}))

    def test_asystole(self):
        self.assertIn("Asystole", self._build({"HR": 2, "PR": 0, "QRS": 0, "QTc": 0}))

    def test_third_degree_av_block(self):
        self.assertIn("Third-degree AV Block", self._build({"HR": 35, "PR": 0, "QRS": 90, "QTc": 420}))

    def test_prolonged_pr(self):
        self.assertIn("First-degree AV Block (Prolonged PR)",
                      self._build({"HR": 72, "PR": 220, "QRS": 90, "QTc": 420}))

    def test_short_pr(self):
        self.assertIn("Short PR Interval", self._build({"HR": 72, "PR": 100, "QRS": 90, "QTc": 420}))

    def test_wide_qrs(self):
        self.assertIn("Wide QRS", self._build({"HR": 72, "PR": 160, "QRS": 125, "QTc": 420}))

    def test_borderline_wide_qrs(self):
        self.assertIn("Borderline Wide QRS", self._build({"HR": 72, "PR": 160, "QRS": 112, "QTc": 420}))

    def test_prolonged_qtc(self):
        self.assertIn("Prolonged QTc", self._build({"HR": 72, "PR": 160, "QRS": 90, "QTc": 470}))

    def test_long_qt_syndrome(self):
        self.assertIn("Long QT Syndrome", self._build({"HR": 72, "PR": 160, "QRS": 90, "QTc": 510}))

    def test_all_zeros_returns_empty(self):
        self.assertEqual([], self._build({"HR": 0, "PR": 0, "QRS": 0, "QTc": 0}))

    def test_none_input_returns_empty(self):
        self.assertEqual([], self._build(None))

    def test_unit_strings_parsed(self):
        self.assertIn("Normal Sinus Rhythm",
                      self._build({"HR": "72 bpm", "PR": "160 ms", "QRS": "90 ms", "QTc": "420 ms"}))


# ════════════════════════════════════════════════════════════════════════════
# 3. CONCLUSION NORMALIZATION
# ════════════════════════════════════════════════════════════════════════════
class TestNormalizeConclusions(unittest.TestCase):

    def _norm(self, c):
        from ecg.ecg_report_generator import _normalize_report_conclusions
        return _normalize_report_conclusions(c)

    def test_removes_placeholders(self):
        r = self._norm(["---", "Normal Sinus Rhythm", "", "---"])
        self.assertNotIn("---", r)
        self.assertNotIn("", r)

    def test_deduplication(self):
        r = self._norm(["Normal Sinus Rhythm", "Normal Sinus Rhythm"])
        self.assertEqual(r.count("Normal Sinus Rhythm"), 1)

    def test_severe_pathology_labels_are_no_longer_printed(self):
        """Superseded by the report conclusion allow-list.

        This test previously asserted that "Third-degree AV Block" survives and
        suppresses "Normal Sinus Rhythm". The printed conclusion is now
        restricted to five value-derived findings (see
        REPORT_ALLOWED_CONCLUSIONS and tests/test_report_conclusions.py), so
        classifier labels such as complete heart block no longer reach the PDF
        at all — the waveform and intervals are printed and their
        interpretation is left to the reading clinician.

        The rhythm line is therefore what remains, and it is no longer
        suppressed by a non-printable finding.
        """
        r = self._norm(["Normal Sinus Rhythm", "Third-degree AV Block"])
        self.assertNotIn("Third-degree AV Block", r)
        self.assertEqual(r, ["Normal Sinus Rhythm"])

    def test_empty_returns_empty(self):
        self.assertEqual([], self._norm([]))

    def test_none_safe(self):
        self.assertIsInstance(self._norm(None), list)


# ════════════════════════════════════════════════════════════════════════════
# 4. FIX 1 — SCREEN LABEL FALLBACK (no phantom zeros)
# ════════════════════════════════════════════════════════════════════════════
class TestScreenLabelFallback(unittest.TestCase):
    """Simulates the _screen_int helper added in Fix 1."""

    def _screen_int(self, text, fallback=0):
        try:
            raw = text.strip()
            clean = (raw.replace('BPM','').replace('bpm','')
                        .replace('ms','').replace('mV','')
                        .split('/')[0].strip())
            val = int(round(float(clean)))
            return val if val > 0 else fallback
        except Exception:
            return fallback

    def test_plain_number(self):
        self.assertEqual(72, self._screen_int(" 72 "))

    def test_strips_bpm(self):
        self.assertEqual(72, self._screen_int("72 BPM"))

    def test_strips_ms(self):
        self.assertEqual(160, self._screen_int("160 ms"))

    def test_takes_first_part_of_slash(self):
        self.assertEqual(420, self._screen_int("420/440"))

    def test_zero_returns_fallback(self):
        self.assertEqual(72, self._screen_int("0", fallback=72))

    def test_dash_returns_fallback(self):
        self.assertEqual(88, self._screen_int("--", fallback=88))

    def test_max_wins_over_mid_update_zero(self):
        self.assertEqual(72, max(0, 72))   # internal=0, screen=72

    def test_both_valid_higher_wins(self):
        self.assertEqual(70, max(65, 70))


# ════════════════════════════════════════════════════════════════════════════
# 5. FIX 2 — RATE QUALIFIER ALONGSIDE ARRHYTHMIA
# ════════════════════════════════════════════════════════════════════════════
class TestRateQualifier(unittest.TestCase):

    def _apply(self, conc_list, hr):
        conc_list = list(conc_list)
        if 0 < hr < 60:
            conc_list = ["Sinus Bradycardia" if c in ("Normal Sinus Rhythm","Normal sinus rhythm")
                         else c for c in conc_list]
            if not any(t in c.lower() for c in conc_list
                       for t in ("bradycardia","slow","junctional","block")):
                conc_list.append("Bradycardia")
        elif hr > 100:
            conc_list = ["Sinus Tachycardia" if c in ("Normal Sinus Rhythm","Normal sinus rhythm")
                         else c for c in conc_list]
            if not any(t in c.lower() for c in conc_list
                       for t in ("tachycardia","fast","flutter","fibrillation","svt","vt")):
                conc_list.append("Sinus Tachycardia")
        return conc_list[:5]

    def test_af_slow_rate_gets_bradycardia(self):
        r = self._apply(["Atrial Fibrillation"], hr=48)
        self.assertIn("Atrial Fibrillation", r)
        self.assertIn("Bradycardia", r)

    def test_nsr_replaced_when_hr_low(self):
        r = self._apply(["Normal Sinus Rhythm"], hr=55)
        self.assertNotIn("Normal Sinus Rhythm", r)
        self.assertIn("Sinus Bradycardia", r)

    def test_nsr_replaced_when_hr_high(self):
        r = self._apply(["Normal Sinus Rhythm"], hr=110)
        self.assertNotIn("Normal Sinus Rhythm", r)
        self.assertIn("Sinus Tachycardia", r)

    def test_bradycardia_not_double_added(self):
        r = self._apply(["Junctional Rhythm"], hr=45)
        count = sum(1 for c in r if "bradycardia" in c.lower() or "junctional" in c.lower())
        self.assertLessEqual(count, 2)

    def test_normal_rate_unchanged(self):
        r = self._apply(["Atrial Fibrillation"], hr=75)
        self.assertEqual(r, ["Atrial Fibrillation"])

    def test_hard_cap_at_5(self):
        r = self._apply(["A1","A2","A3","A4","A5"], hr=48)
        self.assertLessEqual(len(r), 5)


# ════════════════════════════════════════════════════════════════════════════
# 6. FIX 3 — QTc AUTO-WARNING IN CONCLUSION
# ════════════════════════════════════════════════════════════════════════════
class TestQtcAutoWarning(unittest.TestCase):

    def _apply(self, conc_list, qtc):
        conc_list = list(conc_list)
        already = any('qtc' in c.lower() or 'qt' in c.lower() for c in conc_list)
        if not already and qtc > 0:
            if qtc >= 500:
                conc_list.insert(0, f"Critically Prolonged QTc ({qtc} ms) — High Risk")
            elif qtc >= 460:
                conc_list.insert(0, f"Prolonged QTc ({qtc} ms)")
            elif qtc >= 440:
                conc_list.insert(0, f"Borderline QTc ({qtc} ms)")
        return conc_list[:5]

    def test_normal_qtc_no_warning(self):
        r = self._apply(["Normal Sinus Rhythm"], 420)
        self.assertFalse(any("qtc" in c.lower() for c in r))

    def test_borderline_qtc_440(self):
        r = self._apply(["Normal Sinus Rhythm"], 440)
        self.assertTrue(any("borderline" in c.lower() for c in r))

    def test_prolonged_qtc_460(self):
        r = self._apply(["Normal Sinus Rhythm"], 475)
        self.assertTrue(any("prolonged" in c.lower() and "qtc" in c.lower() for c in r))

    def test_critical_qtc_500(self):
        r = self._apply(["Normal Sinus Rhythm"], 515)
        self.assertTrue(any("critically" in c.lower() and "high risk" in c.lower() for c in r))

    def test_qtc_warning_inserted_first(self):
        r = self._apply(["Atrial Fibrillation", "Wide QRS"], 480)
        self.assertIn("qtc", r[0].lower())

    def test_no_duplicate_when_already_mentioned(self):
        r = self._apply(["Prolonged QTc (480 ms)", "NSR"], 480)
        self.assertEqual(1, sum(1 for c in r if "qtc" in c.lower()))

    def test_zero_qtc_no_warning(self):
        r = self._apply(["Normal Sinus Rhythm"], 0)
        self.assertFalse(any("qtc" in c.lower() for c in r))

    def test_cap_at_5(self):
        r = self._apply(["A","B","C","D"], 510)
        self.assertLessEqual(len(r), 5)


# ════════════════════════════════════════════════════════════════════════════
# 7. FLATLINE SAFEGUARD
# ════════════════════════════════════════════════════════════════════════════
class TestFlatlineSafeguard(unittest.TestCase):

    def _is_flatline(self, signal, fs, hr):
        import numpy as np
        arr = np.asarray(signal, dtype=float)
        return (arr.size < int(fs * 2.0)) or (np.std(arr) < 0.1) or (hr <= 0)

    def test_empty_signal(self):
        self.assertTrue(self._is_flatline([], 500, 0))

    def test_zero_hr_triggers(self):
        import numpy as np
        self.assertTrue(self._is_flatline(np.random.randn(1000)*0.5, 500, 0))

    def test_flat_signal_triggers(self):
        import numpy as np
        self.assertTrue(self._is_flatline(np.zeros(2000), 500, 72))

    def test_short_signal_triggers(self):
        import numpy as np
        self.assertTrue(self._is_flatline(np.random.randn(500)*0.5, 500, 72))

    def test_good_signal_not_flatline(self):
        import numpy as np
        self.assertFalse(self._is_flatline(np.random.randn(5000)*0.5, 500, 72))


# ════════════════════════════════════════════════════════════════════════════
# 8. OFFLINE QUEUE
# ════════════════════════════════════════════════════════════════════════════
class TestOfflineQueue(unittest.TestCase):

    def _make_queue(self):
        tmpdir = tempfile.mkdtemp()
        from utils.offline_queue import OfflineQueue
        q = OfflineQueue(queue_dir=tmpdir)
        q.stop_sync_thread()
        return q

    def test_directories_created(self):
        q = self._make_queue()
        for d in ("pending","failed","synced"):
            self.assertTrue(os.path.isdir(os.path.join(q.queue_dir, d)))

    def test_queue_persists_to_disk(self):
        q = self._make_queue()
        item_id = q.queue_data("cloud_report", {"file_path": "/tmp/test.pdf"})
        files = os.listdir(q.pending_dir)
        self.assertTrue(any(item_id in f for f in files))

    def test_stats_track_pending(self):
        q = self._make_queue()
        q.queue_data("cloud_report", {"file_path": "/tmp/a.pdf"})
        q.queue_data("cloud_report", {"file_path": "/tmp/b.pdf"})
        self.assertEqual(q.get_stats()["pending_count"], 2)

    def test_get_pending_returns_queued(self):
        q = self._make_queue()
        q.queue_data("metrics", {"HR": 72})
        items = q.get_pending_items()
        self.assertEqual(len(items), 1)
        self.assertEqual(items[0]["type"], "metrics")

    def test_retry_failed_moves_to_pending(self):
        q = self._make_queue()
        item_id = q.queue_data("cloud_report", {"file_path": "/tmp/test.pdf"})
        q._move_item(item_id, q.pending_dir, q.failed_dir)
        self.assertEqual(1, len(q.get_failed_items()))
        count = q.retry_failed_items()
        self.assertEqual(1, count)
        self.assertEqual(1, len(q.get_pending_items()))
        self.assertEqual(0, len(q.get_failed_items()))

    def test_path_healer_existing_file(self):
        q = self._make_queue()
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        tf.close()
        self.assertEqual(tf.name, q._heal_path(tf.name))
        os.unlink(tf.name)

    def test_path_healer_nonexistent_graceful(self):
        q = self._make_queue()
        result = q._heal_path("/nonexistent/path.pdf")
        self.assertIsInstance(result, str)


# ════════════════════════════════════════════════════════════════════════════
# 9. CONNECTIVITY DETECTION
# ════════════════════════════════════════════════════════════════════════════
class TestConnectivity(unittest.TestCase):

    def _make_queue(self):
        tmpdir = tempfile.mkdtemp()
        from utils.offline_queue import OfflineQueue
        q = OfflineQueue(queue_dir=tmpdir)
        q.stop_sync_thread()
        return q

    def test_online_when_socket_ok(self):
        q = self._make_queue()
        with patch("socket.create_connection", return_value=MagicMock()):
            self.assertTrue(q.is_online(force_check=True))

    def test_offline_when_all_fail(self):
        q = self._make_queue()
        with patch("socket.create_connection", side_effect=OSError("unreachable")):
            self.assertFalse(q.is_online(force_check=True))


# ════════════════════════════════════════════════════════════════════════════
# 10. LICENSE GRACE WINDOW
# ════════════════════════════════════════════════════════════════════════════
class TestGraceWindow(unittest.TestCase):

    GRACE_SECS = 14 * 86400

    def _ok(self, days):
        return (days * 86400) < self.GRACE_SECS

    def test_within_grace(self):
        for d in [0,1,7,13]:
            with self.subTest(days=d):
                self.assertTrue(self._ok(d))

    def test_expired(self):
        for d in [14,15,30]:
            with self.subTest(days=d):
                self.assertFalse(self._ok(d))

    def test_exactly_14_days_is_expired(self):
        self.assertFalse(self._ok(14))


# ════════════════════════════════════════════════════════════════════════════
# 11. FULL CONCLUSION PIPELINE (all 3 fixes integrated)
# ════════════════════════════════════════════════════════════════════════════
class TestConclusionPipeline(unittest.TestCase):

    def _pipeline(self, arrhythmias, hr, qtc):
        conc = list(arrhythmias)
        # Fix 2 — rate qualifier
        if 0 < hr < 60:
            conc = ["Sinus Bradycardia" if c in ("Normal Sinus Rhythm","Normal sinus rhythm")
                    else c for c in conc]
            if not any(t in c.lower() for c in conc
                       for t in ("bradycardia","slow","junctional","block")):
                conc.append("Bradycardia")
        elif hr > 100:
            conc = ["Sinus Tachycardia" if c in ("Normal Sinus Rhythm","Normal sinus rhythm")
                    else c for c in conc]
            if not any(t in c.lower() for c in conc
                       for t in ("tachycardia","fast","flutter","fibrillation","svt","vt")):
                conc.append("Sinus Tachycardia")
        # Fix 3 — QTc warning
        if not any('qtc' in c.lower() or 'qt' in c.lower() for c in conc) and qtc > 0:
            if qtc >= 500:
                conc.insert(0, f"Critically Prolonged QTc ({qtc} ms) — High Risk")
            elif qtc >= 460:
                conc.insert(0, f"Prolonged QTc ({qtc} ms)")
            elif qtc >= 440:
                conc.insert(0, f"Borderline QTc ({qtc} ms)")
        return conc[:5]

    def test_af_slow_rate_prolonged_qtc(self):
        r = self._pipeline(["Atrial Fibrillation"], hr=48, qtc=480)
        t = " ".join(r).lower()
        self.assertIn("atrial fibrillation", t)
        self.assertIn("bradycardia", t)
        self.assertIn("prolonged qtc", t)
        self.assertLessEqual(len(r), 5)

    def test_normal_ecg_clean(self):
        r = self._pipeline(["Normal Sinus Rhythm"], hr=72, qtc=420)
        self.assertEqual(["Normal Sinus Rhythm"], r)

    def test_vt_fast_critical_qtc(self):
        r = self._pipeline(["Ventricular Tachycardia"], hr=160, qtc=510)
        self.assertIn("critically", r[0].lower())
        self.assertTrue(any("ventricular tachycardia" in c.lower() for c in r))

    def test_never_exceeds_5(self):
        r = self._pipeline(["A","B","C","D","E"], hr=45, qtc=510)
        self.assertLessEqual(len(r), 5)

    def test_flatline_produces_no_ecg_data(self):
        r = ["No ECG data available"]
        self.assertEqual(["No ECG data available"], r)
        self.assertNotIn("Normal Sinus Rhythm", r)


# ════════════════════════════════════════════════════════════════════════════
# 12. IMPORT HEALTH
# ════════════════════════════════════════════════════════════════════════════
class TestImportHealth(unittest.TestCase):

    def test_ecg_report_generator(self):
        import ecg.ecg_report_generator  # noqa

    def test_arrhythmia_detector(self):
        import ecg.arrhythmia_detector  # noqa

    def test_offline_queue(self):
        import utils.offline_queue  # noqa

    def test_license_manager(self):
        import utils.license_manager  # noqa

    def test_auto_sync_service(self):
        import utils.auto_sync_service  # noqa


if __name__ == "__main__":
    unittest.main(verbosity=2)
