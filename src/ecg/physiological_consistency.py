"""
Physiological Consistency Engine
=================================
Validates candidate ECG diagnoses against physiological measurements and features.

IMPORTANT DESIGN CONTRACT:
- Genuine arrhythmias MUST NOT be silently suppressed.
- Every rejection is documented with an audit log entry.
- Suppression rules must have HIGH specificity (avoid false negatives).
- Confidence reduction is preferred over outright rejection wherever uncertainty exists.
"""

from typing import List, Dict, Any, Optional

# ─────────────────────────────────────────────────────────────────────────────
# Suppression confidence thresholds.
# Rules that reduce confidence instead of rejecting outright use these values.
# ─────────────────────────────────────────────────────────────────────────────
CONFIDENCE_REDUCTION_STRONG  = 0.60   # Used when evidence is contradictory but not conclusive
CONFIDENCE_REDUCTION_MILD    = 0.80   # Used when evidence is weak


def validate_diagnoses(
    candidates: List[Dict[str, Any]],
    metrics:    Dict[str, Any],
    features:   Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Validates diagnosis candidates using physiological consistency rules.

    Each candidate:
      { "diagnosis": str, "confidence": float, "evidence": List[str] }

    Returns a filtered/adjusted list. Every rejection includes audit fields:
      { ..., "rejected": True, "rejection_reason": str }
    Every passing candidate includes:
      { ..., "rejected": False, "rejection_reason": "" }
    """
    # ── Extract measurement context ───────────────────────────────────────────
    hr  = float(metrics.get("heart_rate_bpm") or 0.0)
    pr  = float(metrics.get("pr_ms") or metrics.get("PR") or 0.0)
    qrs = float(metrics.get("qrs_ms") or metrics.get("QRS") or 0.0)

    p_detected       = bool(features.get("p_detected") or features.get("p_present"))
    av_dissociation  = bool(features.get("av_dissociation"))

    # A "stable PR" requires BOTH: measurable PR > 0 AND no AV dissociation.
    # IMPORTANT: PR = 0 is also valid when PR is simply unmeasured (e.g. low BPM
    # startup). Do not treat unmeasured PR as "stable PR" — that would incorrectly
    # suppress AV block diagnoses at startup.
    pr_measured      = pr > 60.0  # Must exceed minimum physiological threshold to count as "measured"
    stable_pr        = bool(pr_measured and not av_dissociation)

    spectral_flutter = bool(features.get("atrial_flutter"))
    flutter_score    = float(features.get("flutter_score") or 0.0)
    # Flutter evidence: spectral detection OR strong flutter score
    has_flutter_evidence = spectral_flutter or (flutter_score >= 0.18)

    organized_qrs = bool(features.get("organized_qrs", True))
    if features.get("vf_score", 0.0) > 0.6:
        # A high VF score used to unconditionally erase organised-QRS evidence,
        # which then disarmed the very rule below that is meant to catch a false
        # VF. If the QRS was positively MEASURED at a normal narrow width, that
        # measurement is hard evidence of organised activation and outranks a
        # heuristic score built partly from baseline noise.
        if not (0.0 < qrs < 120.0):
            organized_qrs = False

    hr_reliable = bool(hr > 0.0 and hr < 300.0)

    # ── Per-candidate rule evaluation ─────────────────────────────────────────
    audited_candidates = []

    for cand in candidates:
        name       = cand["diagnosis"]
        confidence = float(cand["confidence"])
        evidence   = list(cand.get("evidence", []))

        reject           = False
        rejection_reason = ""

        # ── Rule 1: Third-degree / Complete AV Block ──────────────────────────
        # ONLY reject if:
        #   (a) We HAVE a stable, measured PR (> 60 ms) AND no AV dissociation
        # If PR is unmeasured (0) or AV dissociation is present → KEEP the diagnosis.
        if name in ("Third-degree AV Block", "Complete AV Block"):
            if stable_pr:
                # Strong suppression: stable measured PR contradicts complete block
                reject = True
                rejection_reason = (
                    f"Stable measurable PR interval ({pr:.0f} ms) with no AV dissociation "
                    f"contradicts Complete AV Block"
                )
            # Do NOT reject based on absent p_detected alone — AV block can coexist with
            # retrograde P waves not visible on detection lead.

        # ── Rule 2: Atrial Fibrillation ───────────────────────────────────────
        # Reduce confidence when P waves are detected; hard-reject only when BOTH
        # p_detected=True AND stable_pr=True simultaneously.
        elif name in ("Atrial Fibrillation", "AFib", "AF"):
            if p_detected and stable_pr:
                # Hard reject: organized P waves + stable PR = impossible for AFib
                reject = True
                rejection_reason = "Organized P waves AND stable PR interval detected — inconsistent with AFib"
            elif p_detected:
                # Soft suppression: P waves present but PR stability unclear
                confidence *= CONFIDENCE_REDUCTION_STRONG
                evidence.append(f"[Confidence reduced] P waves detected (confidence → {confidence:.2f})")

        # ── Rule 3: Atrial Flutter ────────────────────────────────────────────
        # IMPORTANT: Atrial Flutter must NOT be suppressed because PR is low-confidence.
        # Only reject if:
        #   (a) Stable PR is confirmed (rules out re-entrant circuit at 240–360 bpm)
        #   AND (b) No spectral flutter evidence at all
        # Stable PR alone is insufficient to reject — Flutter can coexist with variable PR.
        elif name in ("Atrial Flutter", "Flutter"):
            if stable_pr and not has_flutter_evidence:
                # Both conditions must be met for rejection
                reject = True
                rejection_reason = (
                    "Stable PR with no spectral flutter evidence (score="
                    f"{flutter_score:.2f}) — insufficient to confirm Flutter"
                )
            elif not has_flutter_evidence:
                # Reduce confidence without rejecting — insufficient spectral evidence
                confidence *= CONFIDENCE_REDUCTION_MILD
                evidence.append(
                    f"[Confidence reduced] Flutter spectral score low ({flutter_score:.2f}); "
                    f"spectral detection={spectral_flutter}"
                )

        # ── Rule 4: Ventricular Fibrillation / VFib ──────────────────────────
        # VFib can only be rejected when QRS complexes are clearly organized
        # AND HR is reliable. If VF score is high, override organized_qrs check.
        elif name in ("Ventricular Fibrillation", "VFib", "VF"):
            vf_score = float(features.get("vf_score") or 0.0)

            # Hard contradictions first — these outrank the score.
            #
            # VF is disorganised ventricular activity with no atrial-to-
            # ventricular conduction. If this same analysis measured a PR
            # interval, it found P waves conducting to QRS complexes, and the
            # rhythm is by definition not VF. Likewise a ventricular rate in the
            # ordinary physiological band is incompatible with VF (150-400 bpm).
            #
            # A report reading "HR 65 bpm, PR 152 ms, QRS 106 ms" alongside a
            # conclusion of "Ventricular Fibrillation" is self-contradictory,
            # and it was reachable because a vf_score >= 0.5 skipped this rule
            # entirely. The score is a heuristic built partly from baseline
            # noise; a measured PR interval is evidence.
            organised_rate = 20.0 < hr < 150.0
            if pr_measured and organised_rate:
                reject = True
                rejection_reason = (
                    f"Measured PR interval ({pr:.0f} ms) at {hr:.0f} bpm proves "
                    f"atrioventricular conduction — VF has none "
                    f"(vf_score was {vf_score:.2f})"
                )
            elif 0.0 < qrs < 120.0 and organised_rate:
                reject = True
                rejection_reason = (
                    f"Narrow organised QRS ({qrs:.0f} ms) at {hr:.0f} bpm "
                    f"contradicts VF, which has no organised QRS "
                    f"(vf_score was {vf_score:.2f})"
                )
            elif vf_score >= 0.5:
                # No hard contradiction present — defer to the detector.
                pass
            elif organized_qrs and hr_reliable:
                reject = True
                rejection_reason = (
                    f"Organized QRS complexes detected with reliable HR ({hr:.0f} bpm) "
                    f"— VF requires chaotic, unorganized waveform"
                )

        # ── Rule 5: Wide QRS morphology diagnoses ─────────────────────────────
        # Reject LBBB/RBBB/Wide QRS only when QRS is positively measured < 120 ms.
        # Do NOT reject if QRS = 0 (unmeasured).
        elif any(t in name.lower() for t in ("bundle branch block", "lbbb", "rbbb", "wide qrs")):
            if 0 < qrs < 120.0:
                reject = True
                rejection_reason = (
                    f"Authoritative QRS duration ({qrs:.0f} ms) is < 120 ms — "
                    f"Wide QRS diagnosis requires QRS ≥ 120 ms"
                )

        # ── Audit record ─────────────────────────────────────────────────────
        if reject:
            audited_candidates.append({
                "diagnosis":        name,
                "raw_confidence":   cand["confidence"],
                "final_confidence": 0.0,
                "evidence":         evidence,
                "activated":        False,
                "rejected":         True,
                "rejection_reason": rejection_reason,
            })
        else:
            audited_candidates.append({
                "diagnosis":        name,
                "raw_confidence":   cand["confidence"],
                "final_confidence": float(max(0.0, min(1.0, confidence))),
                "confidence":       float(max(0.0, min(1.0, confidence))),
                "evidence":         evidence,
                "activated":        True,
                "rejected":         False,
                "rejection_reason": "",
            })

    # ── Mutual exclusion conflict resolution ──────────────────────────────────
    # Work only with non-rejected candidates
    passed = [c for c in audited_candidates if not c["rejected"]]
    rejected_log = [c for c in audited_candidates if c["rejected"]]

    by_name = {c["diagnosis"]: c for c in passed}

    # Conflict 1: NSR + Complete AV Block
    if "Normal Sinus Rhythm" in by_name and "Third-degree AV Block" in by_name:
        if stable_pr:
            r = by_name.pop("Third-degree AV Block")
            r["rejected"] = True
            r["rejection_reason"] = "Mutual exclusion: NSR + stable PR contradicts Complete AV Block"
            rejected_log.append(r)
        else:
            r = by_name.pop("Normal Sinus Rhythm")
            r["rejected"] = True
            r["rejection_reason"] = "Mutual exclusion: Complete AV Block takes priority over NSR"
            rejected_log.append(r)

    # Conflict 2: Atrial Flutter + Atrial Fibrillation
    flutter_names = [n for n in ("Atrial Flutter", "Flutter") if n in by_name]
    afib_names    = [n for n in ("Atrial Fibrillation", "AFib", "AF") if n in by_name]
    if flutter_names and afib_names:
        flutter_name = flutter_names[0]
        afib_name    = afib_names[0]
        if flutter_score > 0.35:
            r = by_name.pop(afib_name)
            r["rejected"] = True
            r["rejection_reason"] = f"Mutual exclusion: Flutter score ({flutter_score:.2f}) > 0.35 → AFib removed"
            rejected_log.append(r)
        elif by_name[afib_name]["final_confidence"] >= by_name[flutter_name]["final_confidence"]:
            r = by_name.pop(flutter_name)
            r["rejected"] = True
            r["rejection_reason"] = "Mutual exclusion: AFib has higher confidence than Flutter"
            rejected_log.append(r)
        else:
            r = by_name.pop(afib_name)
            r["rejected"] = True
            r["rejection_reason"] = "Mutual exclusion: Flutter has higher confidence than AFib"
            rejected_log.append(r)

    final = list(by_name.values())
    # Attach rejected audit trail so callers can log it
    # Use a module-level store for the audit trail (accessible via get_last_audit_log)
    _audit_log.clear()
    _audit_log.extend(rejected_log + final)

    return final


# ── Module-level audit log (most recent call) ─────────────────────────────────
_audit_log: List[Dict[str, Any]] = []


def get_last_audit_log() -> List[Dict[str, Any]]:
    """Returns the full audit log (rejected + accepted) from the most recent call."""
    return list(_audit_log)
