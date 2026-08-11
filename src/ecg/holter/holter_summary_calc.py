import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def label_beat_sequences(beats: List[Dict[str, Any]]) -> None:
    """Analyzes a sequence of beats and tags ectopic beats with their rhythm context (e.g. Single, Couplet, Bigeminy, Run)."""
    if not beats:
        return
        
    codes = []
    for b in beats:
        lbl = str(b.get('label', '')).upper()
        c = lbl.split('(')[1].split(')')[0] if '(' in lbl and ')' in lbl else lbl
        codes.append(c)
        
    n = len(codes)
    i = 0
    while i < n:
        c = codes[i]
        if c in ['V', 'S']:
            # Find run length
            j = i + 1
            while j < n and codes[j] == c:
                j += 1
            run_len = j - i
            
            if run_len == 1:
                pattern = "Single VE" if c == 'V' else "Single SVE"
                
                # Simple Bigeminy detection (N-V-N-V or N-S-N-S)
                is_bigeminy = False
                if i >= 2 and codes[i-2] == c and codes[i-1] not in ['V', 'S']:
                    is_bigeminy = True
                elif i <= n-3 and codes[i+2] == c and codes[i+1] not in ['V', 'S']:
                    is_bigeminy = True
                    
                # Simple Trigeminy detection (N-N-V-N-N-V)
                is_trigeminy = False
                if not is_bigeminy:
                    if i >= 3 and codes[i-3] == c and codes[i-2] not in ['V', 'S'] and codes[i-1] not in ['V', 'S']:
                        is_trigeminy = True
                    elif i <= n-4 and codes[i+3] == c and codes[i+2] not in ['V', 'S'] and codes[i+1] not in ['V', 'S']:
                        is_trigeminy = True
                
                if is_bigeminy:
                    pattern = "VE Bigeminy" if c == 'V' else "SVE Bigeminy"
                elif is_trigeminy:
                    pattern = "VE Trigeminy" if c == 'V' else "SVE Trigeminy"
                    
            elif run_len == 2:
                pattern = "VE Couplet" if c == 'V' else "SVE Couplet"
            else:
                pattern = "VT Run" if c == 'V' else "SVT Run"
                
            for k in range(i, j):
                beats[k]['rhythm_pattern'] = pattern
                beats[k]['run_length'] = run_len
                beats[k]['short_code'] = c
            i = j
        else:
            beats[i]['short_code'] = c
            i += 1

def calculate_ve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates statistical metrics for Ventricular Ectopy."""
    label_beat_sequences(beats)
    stats = {
        'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0,
        'vt_run': 0, 'longest_vt': 0, 'max_hr': 0, 'min_hr': 0, 'total_ve': 0
    }
    
    current_run_idx = -1
    for i, b in enumerate(beats):
        if b.get('short_code') == 'V':
            stats['total_ve'] += 1
            pat = b.get('rhythm_pattern', '')
            
            if 'Single' in pat:
                stats['single'] += 1
            elif 'Couplet' in pat:
                # Add 1 couplet per 2 beats
                if i > 0 and beats[i-1].get('short_code') != 'V':
                    stats['couplet'] += 1
            elif 'Bigeminy' in pat:
                stats['bigeminy'] += 1
            elif 'Trigeminy' in pat:
                stats['trigeminy'] += 1
            elif 'Run' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'V':
                    stats['vt_run'] += 1
                    rl = b.get('run_length', 0)
                    if rl > stats['longest_vt']:
                        stats['longest_vt'] = rl
                        
    return stats

def calculate_sve_and_rhythm(beats: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculates statistical metrics for Supraventricular Ectopy."""
    label_beat_sequences(beats)
    stats = {
        'single': 0, 'couplet': 0, 'bigeminy': 0, 'trigeminy': 0,
        'svt_run': 0, 'longest_svt': 0, 'max_hr': 0, 'min_hr': 0, 'total_sve': 0
    }
    
    for i, b in enumerate(beats):
        if b.get('short_code') == 'S':
            stats['total_sve'] += 1
            pat = b.get('rhythm_pattern', '')
            
            if 'Single' in pat:
                stats['single'] += 1
            elif 'Couplet' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'S':
                    stats['couplet'] += 1
            elif 'Bigeminy' in pat:
                stats['bigeminy'] += 1
            elif 'Trigeminy' in pat:
                stats['trigeminy'] += 1
            elif 'Run' in pat:
                if i > 0 and beats[i-1].get('short_code') != 'S':
                    stats['svt_run'] += 1
                    rl = b.get('run_length', 0)
                    if rl > stats['longest_svt']:
                        stats['longest_svt'] = rl
                        
    return stats

def calculate_summary_metrics(beats: List[Dict[str, Any]], arrhythmias: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate clinical summary metrics for the Holter report."""
    metrics = {
        'total_beats': len(beats),
        'paced_beats': 0,
        'ventricular_ectopy': 0,
        'supraventricular_ectopy': 0,
        'afib_flutter_duration': 0.0,
        'total_bradycardia': 0.0,
        'total_tachycardia': 0.0
    }
    
    # 1. Count beat-level metrics
    for beat in beats:
        lbl = str(beat.get('label', '')).upper()
        if '(' in lbl and ')' in lbl:
            lbl = lbl.split('(')[1].split(')')[0]
            
        if lbl == 'P':
            metrics['paced_beats'] += 1
        elif lbl == 'V':
            metrics['ventricular_ectopy'] += 1
        elif lbl == 'S':
            metrics['supraventricular_ectopy'] += 1

    # 2. Accumulate durations from structured arrhythmia events
    for ev in arrhythmias:
        lbl = str(ev.get('label', '')).lower()
        duration = float(ev.get('end_sec', 0.0)) - float(ev.get('start_sec', 0.0))
        if duration <= 0:
            continue
            
        if 'fibrillation' in lbl or 'flutter' in lbl or 'afib' in lbl:
            metrics['afib_flutter_duration'] += duration
        elif 'bradycardia' in lbl:
            metrics['total_bradycardia'] += duration
        elif 'tachycardia' in lbl and 'ventricular' not in lbl:
            metrics['total_tachycardia'] += duration

    return metrics

def format_duration(seconds: float) -> str:
    """Format duration in seconds to HH:MM:SS format."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def get_template_beats_for_badges(beats: List[Dict[str, Any]], arrhythmias: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate rich badge items for significant clinical events utilizing the SVE/VE sequences
    and mapping PR, QRS, QT intervals directly into the badge label.
    Filters out secondary morphologies and interval labels (e.g. Long QT, AV Blocks).
    """
    badges = []
    # Manual annotations already have their own display path in the UI and
    # report generator. Keep this helper focused on auto-detected beats so we
    # do not turn user-entered S/V marks into synthetic SVT/VT run badges.
    auto_beats = [beat for beat in beats if not beat.get('is_manual', False)]
    
    # 1. Add strictly major rhythm events from arrhythmias (AFib, AFlutter, VFib, VTach)
    for ev in arrhythmias:
        lbl = str(ev.get('label', ''))
        code = 'X'
        color = ev.get('color', '#FFFF00')
        
        lbl_lower = lbl.lower()
        is_major_rhythm = False
        
        if 'ventricular fibrillation' in lbl_lower or 'ventricular tachycardia' in lbl_lower:
            code = 'V'
            color = '#FF3333'
            is_major_rhythm = True
        elif 'atrial fibrillation' in lbl_lower or 'atrial flutter' in lbl_lower:
            code = 'AF'
            color = '#FF00FF'
            is_major_rhythm = True
            
        # Only add to badges if it's one of the major permitted rhythms
        if is_major_rhythm:
            badges.append({
                'timestamp': float(ev.get('timestamp', ev.get('start_sec', 0.0))),
                'code': code,
                'name': lbl,
                'color': color,
                'type': 'rhythm'
            })
            
    # 2. Add individual significant ectopic beats with rhythm context and interval data
    label_beat_sequences(auto_beats)
    
    for beat in auto_beats:
        code = beat.get('short_code', '')
        if code not in ['V', 'S', 'P']:
            continue
            
        pattern = beat.get('rhythm_pattern', '')
        if code == 'P':
            pattern = 'Paced Beat'
            color = '#FF00FF'
        elif code == 'V':
            color = '#FF3333'
        elif code == 'S':
            color = '#00FFFF'
            
        # Extract PR, QRS, QT/QTc if available from auto-detection results
        qrs = beat.get('qrs_ms')
        qt = beat.get('qt_ms')
        pr = beat.get('pr_ms')
        
        details = []
        if qrs is not None and qrs > 0:
            details.append(f"QRS:{int(qrs)}")
        if qt is not None and qt > 0:
            details.append(f"QT:{int(qt)}")
        if pr is not None and pr > 0:
            details.append(f"PR:{int(pr)}")
            
        detail_str = " ".join(details)
        if detail_str:
            full_name = f"{pattern} | {detail_str}"
        else:
            full_name = pattern
            
        badges.append({
            'timestamp': float(beat.get('timestamp', 0.0)),
            'code': code,
            'name': full_name,
            'color': color,
            'type': 'beat'
        })
            
    badges.sort(key=lambda x: x['timestamp'])
    return badges

