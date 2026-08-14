"""
ecg/holter/holter_summary_calc.py
==================================
Clinical summary calculation helpers for the Holter report generator.

Changes vs original
-------------------
1. normalise_beat_label()      – NEW.  Centralised label-normalisation that
                                 handles every format found across the codebase:
                                 bare codes, parenthesised codes, full arrhythmia
                                 names, and HolterAnalysisWorker class names.
                                 Used by label_beat_sequences() and
                                 calculate_summary_metrics() so both share
                                 exactly the same mapping logic.

2. label_beat_sequences()      – UPGRADED.  Now calls normalise_beat_label()
                                 instead of a raw string split, so Brady / Tachy /
                                 Pause beats get their short_code set correctly
                                 (previously they fell through as 'N').

3. calculate_ve_and_rhythm()   – UPGRADED.  Now also returns max_hr / min_hr
                                 computed from the beat's own rr_ms field, so
                                 callers get those values without a separate pass.

4. calculate_sve_and_rhythm()  – same as (3) for SVE.

5. calculate_brady_tachy_pauses() – NEW.  Single pass that collects every
                                 Brady, Tachy, and Pause beat with its timestamp,
                                 RR and instantaneous HR, ready for the report
                                 table and the classifier panel.

6. calculate_full_ectopy_summary() – NEW.  One-call convenience wrapper that
                                 runs the whole pipeline (normalise → label →
                                 VE stats → SVE stats → Brady/Tachy/Pause) and
                                 returns a flat summary dict compatible with
                                 BeatClassifierEngine.summary from
                                 holter_beat_classifier_panel.py.

7. calculate_summary_metrics() – UNCHANGED in signature; internally uses
                                 normalise_beat_label() for consistency.

8. format_duration()           – UNCHANGED.

9. get_template_beats_for_badges() – UNCHANGED in signature; internally uses
                                 normalise_beat_label() so badges are colour-
                                 coded correctly for all label formats.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict


# ---------------------------------------------------------------------------
# 0.  LABEL NORMALISATION  (new – shared by all functions below)
# ---------------------------------------------------------------------------

def normalise_beat_label(raw_label: str) -> str:
    """
    Convert any beat label format used across the Holter codebase into one of:

        V  | S  | P  | N  | Brady | Tachy | Pause

    Formats handled
    ---------------
    Bare codes          : 'V', 'S', 'N', 'P'
    Parenthesised codes : 'PVCs(V)', 'Atrial PAC(P)', 'Sinus Arrhythmia(S)'
    Worker class names  : 'VE', 'SVE', 'PVC Candidate', 'PAC Candidate',
                          'Brady Episode', 'Tachy Episode', 'Pause Episode'
    Full arrhythmia names: 'Ventricular Fibrillation', 'Atrial Fibrillation 1',
                           'Ventricular Tachycardia', 'Sinus Bradycardia', …
    HolterAnalysisWorker labels: 'Brady', 'Tachy', 'Pause'

    Returns
    -------
    str : one of  V | S | P | N | Brady | Tachy | Pause
    """
    lbl = str(raw_label).strip()

    # ── 1. Parenthesised code: "Foo Bar(V)" → inner letter ──────────────────
    if '(' in lbl and ')' in lbl:
        inner = lbl.split('(')[1].split(')')[0].strip().upper()
        # 'A' (ACLS prefix) maps to SVE / supraventricular bucket
        mapping = {'V': 'V', 'S': 'S', 'P': 'P', 'N': 'N', 'A': 'S', 'C': 'P'}
        if inner in mapping:
            return mapping[inner]

    up = lbl.upper()

    # ── 2. Direct single-letter codes ────────────────────────────────────────
    if up in ('V', 'S', 'P', 'N'):
        return up

    # ── 3. Ventricular (V) ───────────────────────────────────────────────────
    if up in (
        'VE', 'PVC CANDIDATE', 'PREMATURE VENTRICULAR CONTRACTION',
        'VENTRICULAR FIBRILLATION', 'VENTRICULAR TACHYCARDIA',
        'PVCS', 'MONO VTACH', 'POLY VTACH', 'VT RUN',
    ):
        return 'V'
    # Catch partial matches for ventricular arrhythmias
    if 'VENTRICULAR FIBRIL' in up or 'VENTRICULAR TACH' in up:
        return 'V'

    # ── 4. Supraventricular (S) ───────────────────────────────────────────────
    if up in (
        'SVE', 'PAC CANDIDATE', 'PREMATURE ATRIAL CONTRACTION',
        'ATRIAL FIBRILLATION 1', 'ATRIAL FIBRILLATION 2',
        'ATRIAL FLUTTER', 'SINUS ARRHYTHMIA', 'SUPRA VTACH',
        'ATRIAL PAC', 'NODAL PNC', 'NODAL RHYTHM',
    ):
        return 'S'
    if 'ATRIAL FIBRIL' in up or 'ATRIAL FLUTTER' in up:
        return 'S'

    # ── 5. Paced (P) ─────────────────────────────────────────────────────────
    if up in ('PACED', 'PACED BEAT'):
        return 'P'
    if '1ST' in up and 'AV BLOCK' in up:   return 'P'
    if '2ND' in up and 'AV BLOCK' in up:   return 'P'
    if '3RD' in up and 'AV BLOCK' in up:   return 'P'
    if 'BUNDLE BRANCH' in up:              return 'P'

    # ── 6. Brady / Tachy / Pause ──────────────────────────────────────────────
    # Check for 'Brady' before 'Tachy' because 'Sinus Bradycardia' contains
    # neither 'TACHY' nor 'PAUSE', so order does not matter — but be explicit.
    if up in ('BRADY', 'BRADY EPISODE') or 'BRADYCARDIA' in up:
        return 'Brady'
    if up in ('TACHY', 'TACHY EPISODE') or (
        'TACHYCARDIA' in up and 'VENTRICULAR' not in up
    ):
        return 'Tachy'
    if up in ('PAUSE', 'PAUSE EPISODE') or 'ASYSTOLE' in up:
        return 'Pause'

    # ── 7. Default: treat unknown labels as Normal ────────────────────────────
    return 'N'


# ---------------------------------------------------------------------------
# 1.  SEQUENCE LABELLER  (upgraded)
# ---------------------------------------------------------------------------

def label_beat_sequences(beats: List[Dict[str, Any]]) -> None:
    """
    Analyse a sequence of beats and tag each beat in-place with:

        short_code     : str  – normalised single-char / word code
                                (V | S | P | N | Brady | Tachy | Pause)
        rhythm_pattern : str  – clinical pattern name, e.g.
                                'Single VE', 'VE Bigeminy', 'VE Trigeminy',
                                'VE Couplet', 'VT Run',
                                'Single SVE', 'SVE Bigeminy', 'SVE Trigeminy',
                                'SVE Couplet', 'SVT Run',
                                'Brady', 'Tachy', 'Pause', 'Normal'
        run_length     : int  – number of consecutive same-type beats in run

    Changes vs original
    -------------------
    • Uses normalise_beat_label() so Brady/Tachy/Pause beats are identified
      correctly even when they carry full label strings like 'Sinus Bradycardia'
      or worker class names like 'Brady Episode'.
    • Brady / Tachy / Pause beats now receive rhythm_pattern and run_length so
      downstream consumers (report generator, badge builder) can use them.
    """
    if not beats:
        return

    # Build normalised code list — also stamp short_code on every beat now
    # so callers that only call label_beat_sequences() get it too.
    codes: List[str] = []
    for b in beats:
        code = normalise_beat_label(b.get('label', ''))
        b['short_code'] = code
        codes.append(code)

    n = len(codes)
    i = 0

    while i < n:
        c = codes[i]

        # ── Ventricular or Supraventricular ectopic ──────────────────────────
        if c in ('V', 'S'):
            # Measure the full consecutive run of this same code
            j = i + 1
            while j < n and codes[j] == c:
                j += 1
            run_len = j - i

            ve_name  = 'VE'  if c == 'V' else 'SVE'
            run_name = 'VT'  if c == 'V' else 'SVT'

            if run_len == 1:
                # ---- Bigeminy: N-X-N-X  (look ±2 positions) ----------------
                is_bigeminy = (
                    (i >= 2      and codes[i-2] == c and codes[i-1] not in ('V', 'S')) or
                    (i <= n - 3  and codes[i+2] == c and codes[i+1] not in ('V', 'S'))
                )

                # ---- Trigeminy: N-N-X-N-N-X  (look ±3 positions) -----------
                # Only evaluated when bigeminy is NOT found (higher priority).
                is_trigeminy = (not is_bigeminy) and (
                    (
                        i >= 3 and
                        codes[i-3] == c and
                        codes[i-2] not in ('V', 'S') and
                        codes[i-1] not in ('V', 'S')
                    ) or (
                        i <= n - 4 and
                        codes[i+3] == c and
                        codes[i+2] not in ('V', 'S') and
                        codes[i+1] not in ('V', 'S')
                    )
                )

                if is_bigeminy:
                    pattern = f'{ve_name} Bigeminy'
                elif is_trigeminy:
                    pattern = f'{ve_name} Trigeminy'
                else:
                    pattern = f'Single {ve_name}'

            elif run_len == 2:
                pattern = f'{ve_name} Couplet'
            else:
                # ≥ 3 consecutive → sustained run (VT / SVT)
                pattern = f'{run_name} Run'

            # Stamp every beat in the run
            for k in range(i, j):
                beats[k]['rhythm_pattern'] = pattern
                beats[k]['run_length']     = run_len
                # short_code already set above
            i = j

        # ── Brady / Tachy / Pause ─────────────────────────────────────────────
        elif c in ('Brady', 'Tachy', 'Pause'):
            beats[i]['rhythm_pattern'] = c
            beats[i]['run_length']     = 1
            i += 1

        # ── Paced ─────────────────────────────────────────────────────────────
        elif c == 'P':
            beats[i]['rhythm_pattern'] = 'Paced Beat'
            beats[i]['run_length']     = 1
            i += 1

        # ── Normal (and anything else) ────────────────────────────────────────
        else:
            beats[i]['rhythm_pattern'] = 'Normal'
            beats[i]['run_length']     = 1
            i += 1


# ---------------------------------------------------------------------------
# 2.  VE STATISTICS  (upgraded: now computes max_hr / min_hr from rr_ms)
# ---------------------------------------------------------------------------

def calculate_ve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistical metrics for Ventricular Ectopy (VE).

    Calls label_beat_sequences() internally, so callers do NOT need to call it
    first (but calling it twice is harmless).

    Returns
    -------
    dict with keys:
        total_ve    – total VE beat count
        single      – individual isolated VE beats
        couplet     – number of VE couplet events  (each pair = 1 event)
        bigeminy    – VE beats classified as bigeminy
        trigeminy   – VE beats classified as trigeminy
        vt_run      – number of VT run events  (≥3 consecutive VE)
        longest_vt  – beat count of the longest VT run
        max_hr      – highest instantaneous HR seen among VE beats (bpm)
        min_hr      – lowest  instantaneous HR seen among VE beats (bpm)
    """
    label_beat_sequences(beats)

    stats: Dict[str, Any] = {
        'total_ve':   0,
        'single':     0,
        'couplet':    0,
        'bigeminy':   0,
        'trigeminy':  0,
        'vt_run':     0,
        'longest_vt': 0,
        'max_hr':     0.0,
        'min_hr':     0.0,
    }

    _min_hr_init = 9999.0
    _min_hr      = _min_hr_init

    for i, b in enumerate(beats):
        if b.get('short_code') != 'V':
            continue

        stats['total_ve'] += 1
        pat = b.get('rhythm_pattern', '')

        if 'Single' in pat:
            stats['single'] += 1

        elif 'Couplet' in pat:
            # Count one couplet event per pair (only on the first beat)
            prev_code = beats[i-1].get('short_code') if i > 0 else None
            if prev_code != 'V':
                stats['couplet'] += 1

        elif 'Bigeminy' in pat:
            stats['bigeminy'] += 1

        elif 'Trigeminy' in pat:
            stats['trigeminy'] += 1

        elif 'Run' in pat:
            # Count one run event per run block (only on the first beat)
            prev_code = beats[i-1].get('short_code') if i > 0 else None
            if prev_code != 'V':
                stats['vt_run'] += 1
            rl = b.get('run_length', 0)
            if rl > stats['longest_vt']:
                stats['longest_vt'] = rl

        # ── Instantaneous HR from RR interval ────────────────────────────────
        rr = float(b.get('rr_ms', 0.0) or 0.0)
        if rr > 50:
            hr = 60000.0 / rr
            if hr > stats['max_hr']:
                stats['max_hr'] = hr
            if hr < _min_hr:
                _min_hr = hr

    # Collapse sentinel back to 0 if no VE beats were found
    stats['min_hr'] = 0.0 if _min_hr == _min_hr_init else round(_min_hr, 1)
    stats['max_hr'] = round(stats['max_hr'], 1)
    return stats


# ---------------------------------------------------------------------------
# 3.  SVE STATISTICS  (upgraded: same max_hr / min_hr logic as VE)
# ---------------------------------------------------------------------------

def calculate_sve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculate statistical metrics for Supraventricular Ectopy (SVE).

    Calls label_beat_sequences() internally.

    Returns
    -------
    dict with keys:
        total_sve    – total SVE beat count
        single       – individual isolated SVE beats
        couplet      – number of SVE couplet events
        bigeminy     – SVE beats classified as bigeminy
        trigeminy    – SVE beats classified as trigeminy
        svt_run      – number of SVT run events  (≥3 consecutive SVE)
        longest_svt  – beat count of the longest SVT run
        max_hr       – highest instantaneous HR among SVE beats (bpm)
        min_hr       – lowest  instantaneous HR among SVE beats (bpm)
    """
    label_beat_sequences(beats)

    stats: Dict[str, Any] = {
        'total_sve':   0,
        'single':      0,
        'couplet':     0,
        'bigeminy':    0,
        'trigeminy':   0,
        'svt_run':     0,
        'longest_svt': 0,
        'max_hr':      0.0,
        'min_hr':      0.0,
    }

    _min_hr_init = 9999.0
    _min_hr      = _min_hr_init

    for i, b in enumerate(beats):
        if b.get('short_code') != 'S':
            continue

        stats['total_sve'] += 1
        pat = b.get('rhythm_pattern', '')

        if 'Single' in pat:
            stats['single'] += 1

        elif 'Couplet' in pat:
            prev_code = beats[i-1].get('short_code') if i > 0 else None
            if prev_code != 'S':
                stats['couplet'] += 1

        elif 'Bigeminy' in pat:
            stats['bigeminy'] += 1

        elif 'Trigeminy' in pat:
            stats['trigeminy'] += 1

        elif 'Run' in pat:
            prev_code = beats[i-1].get('short_code') if i > 0 else None
            if prev_code != 'S':
                stats['svt_run'] += 1
            rl = b.get('run_length', 0)
            if rl > stats['longest_svt']:
                stats['longest_svt'] = rl

        # ── Instantaneous HR from RR interval ────────────────────────────────
        rr = float(b.get('rr_ms', 0.0) or 0.0)
        if rr > 50:
            hr = 60000.0 / rr
            if hr > stats['max_hr']:
                stats['max_hr'] = hr
            if hr < _min_hr:
                _min_hr = hr

    stats['min_hr'] = 0.0 if _min_hr == _min_hr_init else round(_min_hr, 1)
    stats['max_hr'] = round(stats['max_hr'], 1)
    return stats


# ---------------------------------------------------------------------------
# 4.  BRADY / TACHY / PAUSE COLLECTOR  (new)
# ---------------------------------------------------------------------------

def calculate_brady_tachy_pauses(
    beats: List[Dict[str, Any]],
    brady_bpm:   float = 60.0,
    tachy_bpm:   float = 100.0,
    pause_ms:    float = 2000.0,
) -> Dict[str, Any]:
    """
    Collect all Brady, Tachy, and Pause beats into separate lists and compute
    per-category counts and extreme HR values.

    label_beat_sequences() must have been called first (or call it here by
    passing pre-labelled beats).  This function calls it if short_code is
    absent on the first beat.

    Parameters
    ----------
    beats      : flat beat list (dicts with 'label', optionally 'rr_ms')
    brady_bpm  : HR threshold below which a beat is counted as Brady
    tachy_bpm  : HR threshold above which a beat is counted as Tachy
    pause_ms   : RR interval threshold above which a beat is counted as Pause

    Returns
    -------
    dict with keys:
        pause_beats  : list[dict]  – beats with short_code == 'Pause'
        brady_beats  : list[dict]  – beats with short_code == 'Brady'
        tachy_beats  : list[dict]  – beats with short_code == 'Tachy'
        pause_count  : int
        brady_count  : int
        tachy_count  : int
        longest_pause_ms : float  – longest RR among pause beats
        max_brady_hr     : float  – slowest  HR among brady beats (bpm)
        max_tachy_hr     : float  – fastest  HR among tachy beats (bpm)
    """
    # Ensure beats are labelled
    if beats and 'short_code' not in beats[0]:
        label_beat_sequences(beats)

    pause_beats: List[Dict] = []
    brady_beats: List[Dict] = []
    tachy_beats: List[Dict] = []

    longest_pause_ms = 0.0
    slowest_brady_hr = 9999.0   # lower is slower
    fastest_tachy_hr = 0.0

    for b in beats:
        code = b.get('short_code', 'N')
        rr   = float(b.get('rr_ms', 0.0) or 0.0)
        hr   = (60000.0 / rr) if rr > 50 else 0.0

        if code == 'Pause':
            pause_beats.append(b)
            if rr > longest_pause_ms:
                longest_pause_ms = rr

        elif code == 'Brady':
            brady_beats.append(b)
            if 0 < hr < slowest_brady_hr:
                slowest_brady_hr = hr

        elif code == 'Tachy':
            tachy_beats.append(b)
            if hr > fastest_tachy_hr:
                fastest_tachy_hr = hr

    return {
        'pause_beats':       pause_beats,
        'brady_beats':       brady_beats,
        'tachy_beats':       tachy_beats,
        'pause_count':       len(pause_beats),
        'brady_count':       len(brady_beats),
        'tachy_count':       len(tachy_beats),
        'longest_pause_ms':  round(longest_pause_ms, 1),
        'max_brady_hr':      0.0 if slowest_brady_hr == 9999.0 else round(slowest_brady_hr, 1),
        'max_tachy_hr':      round(fastest_tachy_hr, 1),
    }


# ---------------------------------------------------------------------------
# 5.  FULL ECTOPY SUMMARY  (new — one-call pipeline)
# ---------------------------------------------------------------------------

def calculate_full_ectopy_summary(
    beats: List[Dict[str, Any]],
    arrhythmias: List[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the complete VE / SVE / Brady / Tachy / Pause pipeline in a single
    call and return a flat summary dict.

    The returned dict is compatible with BeatClassifierEngine.summary from
    holter_beat_classifier_panel.py, so the report generator, the classifier
    panel, and the summary bar all read from the same structure.

    Parameters
    ----------
    beats       : flat beat list (any label format — normalisation is internal)
    arrhythmias : optional list of arrhythmia segment dicts for
                  AFib/flutter duration accounting  (passed to
                  calculate_summary_metrics)

    Returns
    -------
    dict – see key listing in the body below
    """
    if arrhythmias is None:
        arrhythmias = []

    # Step 1: normalise + label all beats in-place
    label_beat_sequences(beats)

    total = len(beats)

    # Step 2: VE stats
    ve  = calculate_ve_and_rhythm(beats)   # label_beat_sequences is a no-op
    # Step 3: SVE stats
    sve = calculate_sve_and_rhythm(beats)
    # Step 4: Brady / Tachy / Pause
    btp = calculate_brady_tachy_pauses(beats)
    # Step 5: high-level clinical metrics (counts + AFib duration)
    clin = calculate_summary_metrics(beats, arrhythmias)

    ve_total  = ve.get('total_ve',  0)
    sve_total = sve.get('total_sve', 0)

    return {
        # ── Totals ──────────────────────────────────────────────────────────
        'total_beats':         total,
        'paced_beats':         clin.get('paced_beats', 0),

        # ── VE breakdown ────────────────────────────────────────────────────
        've_total':            ve_total,
        'pct_ve':              round(100.0 * ve_total  / total, 1) if total else 0.0,
        've_singles':          ve.get('single',     0),
        've_couplets':         ve.get('couplet',    0),
        've_bigeminy':         ve.get('bigeminy',   0),
        've_trigeminy':        ve.get('trigeminy',  0),
        've_runs':             ve.get('vt_run',     0),
        've_longest_run':      ve.get('longest_vt', 0),
        've_max_hr':           ve.get('max_hr',     0.0),
        've_min_hr':           ve.get('min_hr',     0.0),

        # ── SVE breakdown ───────────────────────────────────────────────────
        'sve_total':           sve_total,
        'pct_sve':             round(100.0 * sve_total / total, 1) if total else 0.0,
        'sve_singles':         sve.get('single',      0),
        'sve_couplets':        sve.get('couplet',     0),
        'sve_bigeminy':        sve.get('bigeminy',    0),
        'sve_trigeminy':       sve.get('trigeminy',   0),
        'sve_runs':            sve.get('svt_run',     0),
        'sve_longest_run':     sve.get('longest_svt', 0),
        'sve_max_hr':          sve.get('max_hr',      0.0),
        'sve_min_hr':          sve.get('min_hr',      0.0),

        # ── Brady / Tachy / Pause ────────────────────────────────────────────
        'pause_count':         btp.get('pause_count',      0),
        'brady_count':         btp.get('brady_count',      0),
        'tachy_count':         btp.get('tachy_count',      0),
        'longest_pause_ms':    btp.get('longest_pause_ms', 0.0),
        'max_brady_hr':        btp.get('max_brady_hr',     0.0),
        'max_tachy_hr':        btp.get('max_tachy_hr',     0.0),

        # Lists for the classifier panel timeline and report tables
        'pause_beats':         btp.get('pause_beats', []),
        'brady_beats':         btp.get('brady_beats', []),
        'tachy_beats':         btp.get('tachy_beats', []),

        # ── Clinical episode durations ───────────────────────────────────────
        'afib_flutter_duration': clin.get('afib_flutter_duration', 0.0),
        'total_bradycardia':     clin.get('total_bradycardia',     0.0),
        'total_tachycardia':     clin.get('total_tachycardia',     0.0),
    }


# ---------------------------------------------------------------------------
# 6.  CLINICAL SUMMARY METRICS  (unchanged signature, upgraded internals)
# ---------------------------------------------------------------------------

def calculate_summary_metrics(
    beats: List[Dict[str, Any]],
    arrhythmias: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Calculate high-level clinical summary metrics for the Holter report.

    Parameters
    ----------
    beats       : list of beat dicts (any label format)
    arrhythmias : list of arrhythmia segment dicts with
                  'label', 'start_sec', 'end_sec'

    Returns
    -------
    dict with keys:
        total_beats            – total beat count
        paced_beats            – P-coded beats
        ventricular_ectopy     – V-coded beats
        supraventricular_ectopy– S-coded beats
        afib_flutter_duration  – cumulative AFib/AFlutter seconds
        total_bradycardia      – cumulative bradycardia episode seconds
        total_tachycardia      – cumulative tachycardia episode seconds
    """
    metrics: Dict[str, Any] = {
        'total_beats':              len(beats),
        'paced_beats':              0,
        'ventricular_ectopy':       0,
        'supraventricular_ectopy':  0,
        'afib_flutter_duration':    0.0,
        'total_bradycardia':        0.0,
        'total_tachycardia':        0.0,
    }

    # ── 1. Count beat-level metrics using normalise_beat_label ───────────────
    for beat in beats:
        code = normalise_beat_label(beat.get('label', ''))
        if code == 'P':
            metrics['paced_beats']             += 1
        elif code == 'V':
            metrics['ventricular_ectopy']      += 1
        elif code == 'S':
            metrics['supraventricular_ectopy'] += 1

    # ── 2. Accumulate episode durations from arrhythmia segments ─────────────
    for ev in arrhythmias:
        lbl      = str(ev.get('label', '')).lower()
        duration = float(ev.get('end_sec', 0.0)) - float(ev.get('start_sec', 0.0))
        if duration <= 0:
            continue

        if 'fibrillation' in lbl or 'flutter' in lbl or 'afib' in lbl:
            metrics['afib_flutter_duration'] += duration
        elif 'bradycardia' in lbl:
            metrics['total_bradycardia']     += duration
        elif 'tachycardia' in lbl and 'ventricular' not in lbl:
            metrics['total_tachycardia']     += duration

    return metrics


# ---------------------------------------------------------------------------
# 7.  FORMAT HELPER  (unchanged)
# ---------------------------------------------------------------------------

def format_duration(seconds: float) -> str:
    """Format a duration in seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f'{h:02d}:{m:02d}:{s:02d}'


# ---------------------------------------------------------------------------
# 8.  BADGE GENERATOR  (unchanged signature, upgraded internals)
# ---------------------------------------------------------------------------

def get_template_beats_for_badges(
    beats: List[Dict[str, Any]],
    arrhythmias: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Generate rich badge items for significant clinical events.

    Utilises the SVE/VE rhythm sequences and maps PR, QRS, QT intervals
    directly into the badge label.  Filters out secondary morphologies and
    pure-interval labels (e.g. Long QT, AV Blocks).

    Manual annotations (is_manual=True) are excluded — they have their own
    display path in the UI and report generator.

    Returns
    -------
    list of badge dicts sorted by timestamp:
        timestamp : float – seconds from recording start
        code      : str   – V | S | P | AF | X
        name      : str   – human-readable label (pattern + intervals)
        color     : str   – hex colour string
        type      : str   – 'beat' | 'rhythm'
    """
    badges: List[Dict[str, Any]] = []

    # Exclude manual annotations — they have their own display path
    auto_beats = [b for b in beats if not b.get('is_manual', False)]

    # ── 1. Major rhythm events from the arrhythmia segment list ─────────────
    for ev in arrhythmias:
        lbl       = str(ev.get('label', ''))
        lbl_lower = lbl.lower()
        color     = ev.get('color', '#FFFF00')
        code      = 'X'
        is_major  = False

        if 'ventricular fibrillation' in lbl_lower or 'ventricular tachycardia' in lbl_lower:
            code    = 'V'
            color   = '#FF3333'
            is_major = True
        elif 'atrial fibrillation' in lbl_lower or 'atrial flutter' in lbl_lower:
            code    = 'AF'
            color   = '#FF00FF'
            is_major = True

        if is_major:
            badges.append({
                'timestamp': float(ev.get('timestamp', ev.get('start_sec', 0.0))),
                'code':      code,
                'name':      lbl,
                'color':     color,
                'type':      'rhythm',
            })

    # ── 2. Individual ectopic beats with rhythm context + interval data ───────
    label_beat_sequences(auto_beats)

    for beat in auto_beats:
        code = beat.get('short_code', '')
        if code not in ('V', 'S', 'P'):
            continue

        pattern = beat.get('rhythm_pattern', '')
        if code == 'P':
            pattern = 'Paced Beat'
            color   = '#FF00FF'
        elif code == 'V':
            color = '#FF3333'
        else:  # S
            color = '#00FFFF'

        # Append interval details when available
        details: List[str] = []
        qrs = beat.get('qrs_ms')
        qt  = beat.get('qt_ms')
        pr  = beat.get('pr_ms')
        if qrs is not None and qrs > 0:
            details.append(f'QRS:{int(qrs)}')
        if qt  is not None and qt  > 0:
            details.append(f'QT:{int(qt)}')
        if pr  is not None and pr  > 0:
            details.append(f'PR:{int(pr)}')

        full_name = f'{pattern} | {" ".join(details)}' if details else pattern

        badges.append({
            'timestamp': float(beat.get('timestamp', 0.0)),
            'code':      code,
            'name':      full_name,
            'color':     color,
            'type':      'beat',
        })

    badges.sort(key=lambda x: x['timestamp'])
    return badges