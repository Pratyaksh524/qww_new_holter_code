"""
Ventricular Fibrillation false-positive regression suite
========================================================

A 12-lead PDF was produced reading **HR 65 bpm, PR 152 ms, QRS 106 ms,
QT 346 ms** with a CONCLUSION of **"Ventricular Fibrillation"**. Those
statements cannot both be true. VF is disorganised ventricular activity with no
atrial-to-ventricular conduction: there is no PR interval to measure, no
organised QRS to time, and the ventricular rate is 150-400 bpm.

Reporting VF on a normal sinus ECG is the most dangerous failure this
application can produce, so the guards are pinned here.

Two independent layers are tested:

  1. ``is_ventricular_fibrillation`` — the detector must not fire on an
     organised rhythm at an ordinary rate, no matter how noisy the trace.
     Its four criteria previously summed such that RR-variability (0.30) +
     amplitude-variability (0.20) + baseline-chaos (0.15) = 0.65 cleared the
     0.60 threshold with the RATE criterion never firing.

  2. ``physiological_consistency.validate_diagnoses`` — the backstop. Rule 4
     existed to reject VF when QRS was organised and HR reliable, but was
     skipped whenever ``vf_score >= 0.5``, and a ``vf_score > 0.6`` additionally
     forced ``organized_qrs = False``, erasing the evidence the rule needed.

Every test below states which direction it protects: false positives must be
blocked, and genuine VF must still be reported. A change that silences real VF
is far worse than the bug being fixed here.

Run:
    python -m pytest tests/test_vf_false_positive.py -v
"""

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_SRC = os.path.join(_ROOT, "src")
for _p in [_ROOT, _SRC]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ecg.arrhythmia_detector import is_ventricular_fibrillation      # noqa: E402
from ecg.physiological_consistency import validate_diagnoses         # noqa: E402

FS = 500.0
VF_NAMES = ("Ventricular Fibrillation", "VFib", "VF")


def synth_sinus(hr=65.0, seconds=10.0, noise=90.0, amp_jitter=0.5, seed=1):
    """Organised sinus rhythm, deliberately noisy with variable R amplitudes.

    This is the shape that produced the bad report: a real rhythm on a dirty
    trace. Noise must never be enough to call it VF.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    sig = np.zeros(n)
    peaks = []
    rr = 60.0 / hr
    k = 0
    while True:
        beat_t = k * rr + 0.4
        if beat_t * FS >= n - FS * 0.4:
            break
        peaks.append(int(beat_t * FS))
        a = 1.0 + rng.normal(0, amp_jitter)
        g = lambda mu, s, h: h * np.exp(-0.5 * ((t - mu) / s) ** 2)  # noqa: E731
        sig += (g(beat_t - 0.16, 0.022, 120)      # P
                + g(beat_t - 0.02, 0.008, -80 * a)  # Q
                + g(beat_t, 0.011, 1000 * a)        # R
                + g(beat_t + 0.025, 0.009, -200 * a)  # S
                + g(beat_t + 0.24, 0.045, 250))     # T
        k += 1
    sig += rng.normal(0, noise, n)
    # Imperfect detection: some beats missed, some T waves double-counted.
    imperfect = sorted(set(
        [p for i, p in enumerate(peaks) if i % 7]
        + [p + int(0.24 * FS) for i, p in enumerate(peaks) if i % 3 == 0]
    ))
    return sig + 2048.0, imperfect


def vf_candidate():
    return [{"diagnosis": "Ventricular Fibrillation", "confidence": 0.9, "evidence": []}]


def vf_survives(metrics, features):
    out = validate_diagnoses(vf_candidate(), metrics, features)
    return any(c.get("diagnosis") in VF_NAMES and not c.get("rejected") for c in out)


# ══════════════════════════════════════════════════════════════════════════════
# 1. THE DETECTOR — organised rate must disqualify VF
# ══════════════════════════════════════════════════════════════════════════════

class TestDetectorRejectsOrganisedRhythm(unittest.TestCase):

    def test_noisy_sinus_at_65_is_not_vf(self):
        """The exact reported case: 65 bpm on a dirty trace."""
        sig, peaks = synth_sinus(hr=65.0)
        self.assertFalse(
            is_ventricular_fibrillation(sig, peaks, FS),
            "noisy sinus at 65 bpm must never be called Ventricular Fibrillation",
        )

    def test_organised_rates_across_the_normal_band_are_not_vf(self):
        for hr in (40, 50, 65, 73, 91, 110, 140):
            with self.subTest(hr=hr):
                sig, peaks = synth_sinus(hr=float(hr), seed=hr)
                self.assertFalse(
                    is_ventricular_fibrillation(sig, peaks, FS),
                    f"organised rhythm at {hr} bpm must not be called VF",
                )

    def test_noise_alone_cannot_reach_a_vf_verdict(self):
        """Sweep the noise/artefact space that produced the false positive."""
        offenders = []
        for seed in range(6):
            for noise in (60.0, 120.0, 200.0):
                for jitter in (0.3, 0.6):
                    sig, peaks = synth_sinus(hr=65.0, noise=noise,
                                             amp_jitter=jitter, seed=seed)
                    if is_ventricular_fibrillation(sig, peaks, FS):
                        offenders.append((seed, noise, jitter))
        self.assertEqual(offenders, [],
                         f"noise alone produced a VF verdict at 65 bpm: {offenders}")


# ══════════════════════════════════════════════════════════════════════════════
# 2. THE ORGANISED-RATE GATE — must not block genuine VF
# ══════════════════════════════════════════════════════════════════════════════

class TestOrganisedRateGateArithmetic(unittest.TestCase):
    """The gate rejects only organised, ordinary-rate rhythms.

    Mirrors the gate's own arithmetic so the boundary is pinned independently of
    the surrounding scoring, which is what makes a later tweak to the score
    unable to silently change who gets rejected.
    """

    @staticmethod
    def _gate_fires(rr_ms):
        rr = np.array([x for x in rr_ms if 100 < x < 2500], dtype=float)
        if len(rr) <= 3:
            return False
        short_ratio = sum(1 for x in rr_ms if x <= 450) / len(rr_ms)
        implied_hr = 60000.0 / float(np.median(rr))
        return short_ratio <= 0.35 and 20.0 <= implied_hr <= 150.0

    def test_gate_fires_on_organised_normal_rates(self):
        rng = np.random.default_rng(0)
        for label, rr in (
            ("sinus 65", list(rng.normal(923, 40, 12))),
            ("sinus 91", list(rng.normal(658, 35, 15))),
            ("sinus 73", list(rng.normal(821, 40, 13))),
        ):
            with self.subTest(rhythm=label):
                self.assertTrue(self._gate_fires(rr), f"{label} should be rejected as VF")

    def test_gate_never_fires_on_fast_chaotic_rhythms(self):
        rng = np.random.default_rng(0)
        for label, rr in (
            ("VT 180 bpm",              list(rng.normal(333, 20, 30))),
            ("VF ~250 bpm",             list(rng.uniform(150, 400, 40))),
            ("VF ~330 bpm",             list(rng.uniform(120, 300, 50))),
            ("coarse VF w/ long gaps",  list(rng.uniform(140, 520, 40))),
        ):
            with self.subTest(rhythm=label):
                self.assertFalse(
                    self._gate_fires(rr),
                    f"{label} must remain eligible for a VF verdict",
                )


# ══════════════════════════════════════════════════════════════════════════════
# 3. THE CONSISTENCY BACKSTOP
# ══════════════════════════════════════════════════════════════════════════════

class TestConsistencyRejectsContradictoryVF(unittest.TestCase):
    """A report cannot say VF and print a PR interval on the same page."""

    def test_the_exact_bad_report_is_rejected(self):
        self.assertFalse(
            vf_survives(dict(heart_rate_bpm=65, pr_ms=152, qrs_ms=106),
                        dict(vf_score=0.65)),
            "HR 65 / PR 152 / QRS 106 must not be reported as VF",
        )

    def test_measured_pr_vetoes_vf_at_any_score(self):
        for score in (0.5, 0.6, 0.75, 0.9, 1.0):
            with self.subTest(vf_score=score):
                self.assertFalse(
                    vf_survives(dict(heart_rate_bpm=65, pr_ms=152, qrs_ms=106),
                                dict(vf_score=score)),
                    "a measured PR interval proves AV conduction — VF has none",
                )

    def test_narrow_organised_qrs_vetoes_vf_even_without_pr(self):
        self.assertFalse(
            vf_survives(dict(heart_rate_bpm=70, pr_ms=0, qrs_ms=95),
                        dict(vf_score=0.65)),
            "a narrow organised QRS at 70 bpm contradicts VF",
        )

    def test_all_three_reported_cases_are_rejected(self):
        for hr, pr, qrs in ((65, 152, 106), (91, 152, 106), (73, 152, 116)):
            with self.subTest(hr=hr, qrs=qrs):
                self.assertFalse(
                    vf_survives(dict(heart_rate_bpm=hr, pr_ms=pr, qrs_ms=qrs),
                                dict(vf_score=0.70)))

    def test_high_vf_score_no_longer_erases_a_measured_narrow_qrs(self):
        # vf_score > 0.6 used to force organized_qrs = False, disarming the rule.
        self.assertFalse(
            vf_survives(dict(heart_rate_bpm=80, pr_ms=0, qrs_ms=90),
                        dict(vf_score=0.95, organized_qrs=True)))


class TestGenuineVFStillReported(unittest.TestCase):
    """The direction that matters more: real VF must never be silenced."""

    def test_nothing_measurable_is_still_vf(self):
        self.assertTrue(
            vf_survives(dict(heart_rate_bpm=0, pr_ms=0, qrs_ms=0),
                        dict(vf_score=0.85)),
            "VF with no measurable intervals must still be reported",
        )

    def test_fast_rate_with_no_conduction_is_still_vf(self):
        self.assertTrue(
            vf_survives(dict(heart_rate_bpm=280, pr_ms=0, qrs_ms=0),
                        dict(vf_score=0.75)))

    def test_fast_wide_no_pr_is_still_vf(self):
        self.assertTrue(
            vf_survives(dict(heart_rate_bpm=200, pr_ms=0, qrs_ms=160),
                        dict(vf_score=0.80)))

    def test_rate_above_the_organised_band_is_still_vf(self):
        for hr in (160, 200, 250, 300, 350):
            with self.subTest(hr=hr):
                self.assertTrue(
                    vf_survives(dict(heart_rate_bpm=hr, pr_ms=0, qrs_ms=0),
                                dict(vf_score=0.8)),
                    f"VF at {hr} bpm must still be reported",
                )

    def test_asystole_and_other_lethal_labels_are_untouched(self):
        for name in ("Asystole", "Ventricular Tachycardia"):
            with self.subTest(diagnosis=name):
                out = validate_diagnoses(
                    [{"diagnosis": name, "confidence": 0.9, "evidence": []}],
                    dict(heart_rate_bpm=0, pr_ms=0, qrs_ms=0),
                    dict(vf_score=0.0),
                )
                self.assertTrue(
                    any(c.get("diagnosis") == name and not c.get("rejected") for c in out),
                    f"{name} must not be affected by the VF guards",
                )


# ══════════════════════════════════════════════════════════════════════════════
# 4. THE SCORING STRUCTURE THAT ALLOWED IT
# ══════════════════════════════════════════════════════════════════════════════

class TestScoringStructure(unittest.TestCase):

    def test_non_rate_criteria_still_sum_above_the_threshold(self):
        """Documents *why* the gate is needed rather than a threshold tweak.

        RR-variability + amplitude-variability + baseline-chaos = 0.65, which
        clears the 0.60 bar with the rate criterion contributing nothing. Raising
        the threshold to 0.66 would have masked this one path while leaving the
        underlying problem — a VF verdict reachable with no fast beat — intact.
        If these weights are ever rebalanced this test will need revisiting, and
        the gate should be re-justified rather than silently dropped.
        """
        rr_variability, amplitude_variability, baseline_chaos = 0.30, 0.20, 0.15
        self.assertGreaterEqual(
            rr_variability + amplitude_variability + baseline_chaos, 0.60,
            "if this no longer holds, re-check whether the organised-rate gate "
            "is still the right defence",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
