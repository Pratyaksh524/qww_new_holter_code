"""
Report CONCLUSION allow-list regression suite
=============================================

The 12-lead PDF conclusion box is restricted to exactly five findings:

    Normal Sinus Rhythm | Sinus Bradycardia | Sinus Tachycardia
    Wide QRS | Prolonged QTc

All five derive directly from a measured value that is also printed in the
report header (HR, QRS, QTc), so a reader can check the conclusion against the
numbers on the same page.

Everything else the detectors can produce is filtered out of the printed
conclusion — including Asystole, Ventricular Fibrillation, Ventricular
Tachycardia, Atrial Fibrillation/Flutter, every AV block, every bundle branch
block, PVC/PAC, ST elevation/depression and Borderline Wide QRS. This was a
deliberate product decision after a normal 65 bpm sinus ECG was printed with a
conclusion of "Ventricular Fibrillation".

These tests pin three properties:
  1. Nothing outside the permitted five can ever reach the report.
  2. A permitted finding is never lost to a spelling variant.
  3. The box is never empty when a heart rate was measured, and never exceeds
     the four rows it can physically print.

Run:
    python -m pytest tests/test_report_conclusions.py -v
"""

import os
import sys
import unittest

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecg.ecg_report_generator import (            # noqa: E402
    REPORT_ALLOWED_CONCLUSIONS,
    _build_metric_conclusions,
    _normalize_report_conclusions,
    ensure_rate_conclusion,
    restrict_to_allowed_conclusions,
)

ALLOWED = set(REPORT_ALLOWED_CONCLUSIONS)

# The box is 26.5 mm tall with 4 mm rows; row 5 is cropped by the draw code.
PRINTABLE_ROWS = 4


def _read(rel):
    """Source text of a repo file, for the source-level assertions below."""
    with open(os.path.join(_ROOT, rel), "r", encoding="utf-8") as fh:
        return fh.read()


def report_for(hr, pr, qrs, qtc):
    """Everything the report does to turn measurements into printed lines."""
    data = {"HR_bpm": hr, "PR": pr, "QRS": qrs, "QTc": qtc}
    out = _normalize_report_conclusions(_build_metric_conclusions(data))
    return ensure_rate_conclusion(out, data)


# ══════════════════════════════════════════════════════════════════════════════
# 1. NOTHING OUTSIDE THE FIVE
# ══════════════════════════════════════════════════════════════════════════════

class TestOnlyPermittedFindingsPrint(unittest.TestCase):

    BLOCKED = [
        "Asystole",
        "Ventricular Fibrillation",
        "Ventricular Tachycardia",
        "Atrial Fibrillation",
        "Atrial Flutter",
        "Third-degree AV Block",
        "3rd-degree AV block",
        "Second-degree AV Block (Mobitz I)",
        "Second-degree AV Block (Mobitz II)",
        "First-degree AV Block (Prolonged PR)",
        "Short PR Interval",
        "Complete Left Bundle Branch Block",
        "Complete Right Bundle Branch Block",
        "Left bundle branch block (LBBB)",
        "Right bundle branch block (RBBB)",
        "Incomplete LBBB",
        "Borderline Wide QRS",
        "Premature ventricular contraction (PVC)",
        "Premature atrial contraction (PAC)",
        "Ventricular Bigeminy",
        "Ventricular Trigeminy",
        "SVT",
        "ST elevation",
        "ST depression",
        "Bradycardia (non-sinus)",
        "Tachycardia (non-sinus)",
        "Heart Rate: 72 BPM",
        "PR Interval: 152 ms",
        "QRS Duration: 106 ms",
        "QTc: 359 ms",
    ]

    def test_each_blocked_label_is_removed(self):
        for label in self.BLOCKED:
            with self.subTest(label=label):
                self.assertEqual(
                    _normalize_report_conclusions([label]), [],
                    f"{label!r} must not reach the printed conclusion",
                )

    def test_blocked_labels_cannot_ride_along_with_permitted_ones(self):
        mixed = ["Normal Sinus Rhythm"] + self.BLOCKED
        out = _normalize_report_conclusions(mixed)
        self.assertEqual(out, ["Normal Sinus Rhythm"])

    def test_restrict_helper_is_the_gate(self):
        self.assertEqual(
            restrict_to_allowed_conclusions(["Asystole", "Wide QRS", "SVT"]),
            ["Wide QRS"],
        )

    def test_permitted_set_is_exactly_five(self):
        self.assertEqual(len(REPORT_ALLOWED_CONCLUSIONS), 5)
        self.assertEqual(ALLOWED, {
            "Normal Sinus Rhythm", "Sinus Bradycardia", "Sinus Tachycardia",
            "Wide QRS", "Prolonged QTc",
        })


# ══════════════════════════════════════════════════════════════════════════════
# 2. VARIANTS FOLD IN — a real finding is never lost to wording
# ══════════════════════════════════════════════════════════════════════════════

class TestPermittedVariantsSurvive(unittest.TestCase):

    CASES = [
        ("Sinus Rhythm",                        "Normal Sinus Rhythm"),
        ("normal sinus rhythm",                 "Normal Sinus Rhythm"),
        ("NSR",                                 "Normal Sinus Rhythm"),
        ("Bradycardia",                         "Sinus Bradycardia"),
        ("Athlete Bradycardia",                 "Sinus Bradycardia"),
        ("Tachycardia",                         "Sinus Tachycardia"),
        ("Wide QRS Complex",                    "Wide QRS"),
        ("Long QT Syndrome",                    "Prolonged QTc"),
        ("Prolonged QTc Interval",              "Prolonged QTc"),
    ]

    def test_variants_map_to_permitted_wording(self):
        for raw, expected in self.CASES:
            with self.subTest(raw=raw):
                self.assertEqual(_normalize_report_conclusions([raw]), [expected])

    def test_long_qt_is_not_silently_dropped(self):
        """QTc > 500 is still a prolonged QTc — it must be reported, not lost."""
        self.assertIn("Prolonged QTc", report_for(75, 150, 95, 520))

    def test_duplicates_collapse(self):
        out = _normalize_report_conclusions(
            ["Sinus Rhythm", "Normal Sinus Rhythm", "NSR"])
        self.assertEqual(out, ["Normal Sinus Rhythm"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE BOX IS NEVER EMPTY AND NEVER OVERFLOWS
# ══════════════════════════════════════════════════════════════════════════════

class TestBoxAlwaysUsable(unittest.TestCase):

    HRS  = (5, 10, 30, 39, 40, 45, 55, 59, 60, 65, 85, 99, 100, 101, 130, 160, 200, 250)
    PRS  = (0, 100, 119, 152, 201, 260)
    QRSS = (0, 80, 109, 110, 116, 119, 120, 160)
    QTCS = (0, 359, 430, 460, 461, 500, 501, 560)

    def test_exhaustive_grid_is_well_formed(self):
        empty, overflow, offlist = [], [], set()
        for hr in self.HRS:
            for pr in self.PRS:
                for qrs in self.QRSS:
                    for qtc in self.QTCS:
                        out = report_for(hr, pr, qrs, qtc)
                        if not out:
                            empty.append((hr, pr, qrs, qtc))
                        if len(out) > PRINTABLE_ROWS:
                            overflow.append((hr, pr, qrs, qtc, out))
                        offlist |= {c for c in out if c not in ALLOWED}
        self.assertEqual(empty, [], f"empty conclusion box for: {empty[:5]}")
        self.assertEqual(overflow, [], f"more findings than rows: {overflow[:5]}")
        self.assertEqual(offlist, set(), f"off-list labels printed: {offlist}")

    def test_low_rate_with_no_pr_still_states_the_rate(self):
        """Used to print an empty box: the chain picked Third-degree AV Block,
        which the allow-list then removed, leaving nothing at all."""
        self.assertEqual(report_for(10, 0, 0, 0), ["Sinus Bradycardia"])
        self.assertEqual(report_for(35, 0, 140, 0), ["Sinus Bradycardia", "Wide QRS"])

    def test_rate_fallback_does_not_override_an_existing_rhythm(self):
        kept = ensure_rate_conclusion(["Sinus Tachycardia", "Wide QRS"],
                                      {"HR_bpm": 130})
        self.assertEqual(kept, ["Sinus Tachycardia", "Wide QRS"])

    def test_no_rate_means_no_invented_rhythm(self):
        self.assertEqual(ensure_rate_conclusion([], {"HR_bpm": 0}), [])


# ══════════════════════════════════════════════════════════════════════════════
# 4. THE VALUES ON THE PAGE MATCH THE CONCLUSION
# ══════════════════════════════════════════════════════════════════════════════

class TestConclusionMatchesPrintedValues(unittest.TestCase):

    def test_rate_bands(self):
        self.assertEqual(report_for(45, 150, 95, 420), ["Sinus Bradycardia"])
        self.assertEqual(report_for(75, 150, 95, 420), ["Normal Sinus Rhythm"])
        self.assertEqual(report_for(120, 150, 95, 420), ["Sinus Tachycardia"])

    def test_qrs_threshold_is_120(self):
        self.assertNotIn("Wide QRS", report_for(75, 150, 119, 420))
        self.assertIn("Wide QRS", report_for(75, 150, 120, 420))

    def test_qtc_threshold_is_460(self):
        self.assertNotIn("Prolonged QTc", report_for(75, 150, 95, 460))
        self.assertIn("Prolonged QTc", report_for(75, 150, 95, 461))

    def test_rhythm_is_still_stated_alongside_a_conduction_finding(self):
        """"Normal Sinus Rhythm" + "Wide QRS" is a coherent pair; the rhythm
        line must not be suppressed by a non-rhythm finding."""
        self.assertEqual(report_for(75, 150, 140, 420),
                         ["Normal Sinus Rhythm", "Wide QRS"])

    def test_all_three_field_reports(self):
        self.assertEqual(report_for(65, 152, 106, 359), ["Normal Sinus Rhythm"])
        self.assertEqual(report_for(91, 152, 106, 408), ["Normal Sinus Rhythm"])
        # QRS 116 is below the 120 ms threshold, so no QRS finding — but the
        # rhythm line must still be present (this printed an empty box once).
        self.assertEqual(report_for(73, 152, 116, 349), ["Normal Sinus Rhythm"])

    def test_maximum_is_three_findings(self):
        self.assertEqual(report_for(48, 210, 140, 505),
                         ["Sinus Bradycardia", "Wide QRS", "Prolonged QTc"])


# ══════════════════════════════════════════════════════════════════════════════
# 5. THE GENERATOR THE PAGE ACTUALLY USES
# ══════════════════════════════════════════════════════════════════════════════

STATUS_LINES = {
    "No cardiac activity detected",
    "Rate below measurable range",
    "No ECG data available",
    "Please connect device",
}


def android_path_filter(conc_list, hr):
    """Mirror of the allow-list block in twelve_lead_test.generate_pdf_report().

    Kept as an executable copy of the rule so the boundary is pinned even
    though the real code sits inside a large Qt method that cannot be called
    headlessly.
    """
    status = [c for c in conc_list if c in STATUS_LINES]
    canon = []
    for c in conc_list:
        low = str(c).lower()
        if "critically prolonged qtc" in low or "prolonged qtc" in low:
            canon.append("Prolonged QTc")
        else:
            canon.append(c)
    findings = restrict_to_allowed_conclusions(canon)
    out = status + [c for c in findings if c not in status]
    if not out and hr > 0:
        out = ["Sinus Bradycardia" if hr < 60
               else "Sinus Tachycardia" if hr > 100
               else "Normal Sinus Rhythm"]
    return out[:5]


class TestAndroidReportPathIsFiltered(unittest.TestCase):
    """The allow-list was originally applied to the WRONG generator.

    twelve_lead_test.generate_pdf_report() builds `conc_list` itself and hands
    it to ecg_report_android.generate_report(); it never calls
    ecg_report_generator.generate_ecg_report(). So the restriction had no effect
    on the PDF the page actually produces, and "Asystole" kept printing after
    the allow-list was supposedly in place.
    """

    def test_page_applies_the_allow_list_before_handing_off(self):
        src = _read("src/ecg/twelve_lead_test.py")
        self.assertIn("restrict_to_allowed_conclusions", src,
                      "the page must filter conc_list before calling the generator")
        idx = src.find("conc_list = conc_list[:5]")
        self.assertGreater(idx, -1)
        before = src[max(0, idx - 2500):idx]
        self.assertIn("restrict_to_allowed_conclusions", before,
                      "the filter must run BEFORE the hard cap, or labels are "
                      "capped rather than removed")

    def test_lethal_classifier_labels_do_not_reach_the_pdf(self):
        for label in ("Asystole", "Ventricular Fibrillation",
                      "Ventricular Tachycardia", "Atrial Fibrillation",
                      "Atrial Flutter", "3rd-degree AV block",
                      "Complete Left Bundle Branch Block"):
            with self.subTest(label=label):
                self.assertNotIn(label, android_path_filter([label], 0))

    def test_the_original_contradiction_now_prints_the_rate(self):
        """analyze_ecg said Asystole while the header read HR 65."""
        self.assertEqual(android_path_filter(["Asystole"], 65),
                         ["Normal Sinus Rhythm"])

    def test_never_empty_when_a_rate_was_measured(self):
        empty = [hr for hr in (1, 30, 45, 59, 60, 72, 100, 101, 150, 250)
                 if not android_path_filter(["Asystole"], hr)]
        self.assertEqual(empty, [], f"empty conclusion box at HR {empty}")

    def test_qtc_lines_keep_their_finding_despite_carrying_a_value(self):
        for raw in ("Prolonged QTc (470 ms)",
                    "Critically Prolonged QTc (505 ms) - High Risk"):
            with self.subTest(raw=raw):
                self.assertIn("Prolonged QTc",
                              android_path_filter([raw, "Normal Sinus Rhythm"], 75))

    def test_borderline_qtc_is_not_promoted(self):
        """440-459 ms is below the 460 ms threshold "Prolonged QTc" means."""
        out = android_path_filter(["Borderline QTc (445 ms)", "Normal Sinus Rhythm"], 75)
        self.assertNotIn("Prolonged QTc", out)
        self.assertEqual(out, ["Normal Sinus Rhythm"])

    def test_status_lines_survive_the_filter(self):
        for line in ("No cardiac activity detected",
                     "Rate below measurable range",
                     "No ECG data available"):
            with self.subTest(line=line):
                self.assertEqual(android_path_filter([line], 0), [line])

    def test_permitted_findings_survive_alongside_suppressed_ones(self):
        out = android_path_filter(
            ["Atrial Fibrillation", "Sinus Tachycardia", "Wide QRS"], 130)
        self.assertEqual(out, ["Sinus Tachycardia", "Wide QRS"])

    def test_page_flatline_branch_has_three_states(self):
        src = _read("src/ecg/twelve_lead_test.py")
        idx = src.find("if is_flatline:")
        self.assertGreater(idx, -1)
        branch = src[idx:idx + 900]
        for wording in ("No cardiac activity detected",
                        "Rate below measurable range",
                        "No ECG data available"):
            with self.subTest(wording=wording):
                self.assertIn(wording, branch)


if __name__ == "__main__":
    unittest.main(verbosity=2)
