import numpy as np
from typing import List, Dict, Any, Tuple
from collections import defaultdict

def label_beat_sequences(beats: List[Dict[str, Any]]) -> None:
    """
    Analyzes a sequence of beats and tags ectopic beats with their rhythm context.
    
    Detects:
      1. Single PVC/PAC (N-N-N-V-N-N-N)
      2. Couplets (N-N-V-V-N-N)
      3. Bigeminy (N-V-N-V-N-V)
      4. Trigeminy (N-N-V-N-N-V)
      5. Quadrigeminy (N-N-N-V-N-N-N-V)
      6. Runs (3+ consecutive: VT/SVT)
      7. Atrial Fibrillation (irregular RR + no P)
    """
    if not beats:
        return
        
    # Extract beat codes
    codes = []
    for b in beats:
        lbl = str(b.get('label', '')).upper()
        c = lbl.split('(')[1].split(')')[0] if '(' in lbl and ')' in lbl else lbl
        codes.append(c)
        
    n = len(codes)
    
    # STEP 1: Detect AFib globally (affects all beats in irregular window)
    _detect_atrial_fibrillation(beats, codes)
    
    # STEP 2: Label ectopic beat patterns
    i = 0
    while i < n:
        c = codes[i]
        if c in ['V', 'S']:
            # Find consecutive run length
            j = i + 1
            while j < n and codes[j] == c:
                j += 1
            run_len = j - i
            
            # CLASSIFY PATTERN
            if run_len == 1:
                # Single ectopic - check for bigeminy/trigeminy/quadrigeminy
                pattern = _classify_single_ectopic(codes, i, n, c)
            elif run_len == 2:
                pattern = "PVC Couplet" if c == 'V' else "PAC Couplet"
            elif run_len >= 3:
                # VT/SVT Run
                pattern = "VT Run" if c == 'V' else "SVT Run"
                # Mark as critical if VT run
                if c == 'V':
                    for k in range(i, j):
                        beats[k]['critical'] = True
            else:
                pattern = "Ectopic"
                
            # Apply pattern to all beats in run
            for k in range(i, j):
                beats[k]['rhythm_pattern'] = pattern
                beats[k]['run_length'] = run_len
                beats[k]['short_code'] = c
            i = j
        else:
            # Normal beat
            beats[i]['short_code'] = c
            beats[i]['rhythm_pattern'] = beats[i].get('rhythm_pattern', 'Normal')
            i += 1


def _classify_single_ectopic(codes: List[str], idx: int, n: int, ectopic_code: str) -> str:
    """
    Classify single ectopic beat as:
      - Bigeminy: N-V-N-V-N-V
      - Trigeminy: N-N-V-N-N-V
      - Quadrigeminy: N-N-N-V-N-N-N-V
      - Single: Isolated ectopic
    """
    # Check BIGEMINY: Every other beat is ectopic
    is_bigeminy = False
    if idx >= 2 and codes[idx-2] == ectopic_code and codes[idx-1] not in ['V', 'S']:
        is_bigeminy = True
    elif idx <= n-3 and codes[idx+2] == ectopic_code and codes[idx+1] not in ['V', 'S']:
        is_bigeminy = True
    
    if is_bigeminy:
        return "PVC Bigeminy" if ectopic_code == 'V' else "PAC Bigeminy"
    
    # Check TRIGEMINY: Every third beat is ectopic (N-N-V pattern)
    is_trigeminy = False
    if idx >= 3 and codes[idx-3] == ectopic_code:
        if codes[idx-2] not in ['V', 'S'] and codes[idx-1] not in ['V', 'S']:
            is_trigeminy = True
    elif idx <= n-4 and codes[idx+3] == ectopic_code:
        if codes[idx+1] not in ['V', 'S'] and codes[idx+2] not in ['V', 'S']:
            is_trigeminy = True
    
    if is_trigeminy:
        return "PVC Trigeminy" if ectopic_code == 'V' else "PAC Trigeminy"
    
    # Check QUADRIGEMINY: Every fourth beat is ectopic (N-N-N-V pattern)
    is_quadrigeminy = False
    if idx >= 4 and codes[idx-4] == ectopic_code:
        if all(codes[idx-k] not in ['V', 'S'] for k in range(1, 4)):
            is_quadrigeminy = True
    elif idx <= n-5 and codes[idx+4] == ectopic_code:
        if all(codes[idx+k] not in ['V', 'S'] for k in range(1, 4)):
            is_quadrigeminy = True
    
    if is_quadrigeminy:
        return "PVC Quadrigeminy" if ectopic_code == 'V' else "PAC Quadrigeminy"
    
    # Default: Single isolated ectopic
    return "Single PVC" if ectopic_code == 'V' else "Single PAC"


def _detect_atrial_fibrillation(beats: List[Dict[str, Any]], codes: List[str]) -> None:
    """
    Detect Atrial Fibrillation episodes:
      - Irregular RR intervals (CoV > 0.30)
      - No P waves (>70% absence)
      
    Marks beats within AFib episodes with 'afib_episode' flag.
    """
    if len(beats) < 10:
        return
    
    # Sliding window: 10-beat segments
    window_size = 10
    for start in range(0, len(beats) - window_size + 1, 5):  # 50% overlap
        window_beats = beats[start:start + window_size]
        
        # Calculate RR irregularity
        rr_intervals = []
        for b in window_beats:
            rr = b.get('rr_ms')
            if rr and rr > 0:
                rr_intervals.append(rr)
        
        if len(rr_intervals) < 5:
            continue
        
        rr_array = np.array(rr_intervals)
        mean_rr = np.mean(rr_array)
        std_rr = np.std(rr_array)
        cov_rr = std_rr / mean_rr if mean_rr > 0 else 0.0
        
        # Check P-wave absence
        p_absent_count = sum(1 for b in window_beats if not b.get('p_present', False))
        p_absent_ratio = p_absent_count / len(window_beats)
        
        # AFib criteria
        if cov_rr > 0.30 and p_absent_ratio > 0.70:
            # Mark all beats in window as AFib
            for b in window_beats:
                b['afib_episode'] = True
                b['rhythm_pattern'] = 'Atrial Fibrillation'
                b['short_code'] = 'AF'

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
    Generate rich badge items for ALL ventricular ectopy (VE), supraventricular ectopy (SVE),
    atrial fibrillation, and significant clinical events.
    
    Badge Display Logic:
      - ALL V beats (PVCs) - Single PVCs, Couplets, Bigeminy, Trigeminy, Quadrigeminy, VT Runs
      - ALL S beats (PACs/SVEs) - Single PACs, Couplets, Bigeminy, Trigeminy, Quadrigeminy, SVT Runs
      - ALL AF beats - Atrial Fibrillation episodes
      - ALL P beats - Paced beats
      - Major rhythm events from arrhythmias (VFib, VTach, AFib, AFlutter)
    
    Each badge includes PR, QRS, QT intervals and heart rate when available.
    """
    badges = []
    # Process all auto-detected beats to show ALL V, S, P, AF beats with badge labels.
    # This ensures every ectopic beat is visible, not just significant patterns.
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
            color = '#FF0000'  # Bright red for VFib/VT
            is_major_rhythm = True
        elif 'atrial fibrillation' in lbl_lower or 'atrial flutter' in lbl_lower:
            code = 'AF'
            color = '#FF00FF'  # Magenta for AFib
            is_major_rhythm = True
            
        # Only add to badges if it's one of the major permitted rhythms
        if is_major_rhythm:
            badges.append({
                'timestamp': float(ev.get('timestamp', ev.get('start_sec', 0.0))),
                'code': code,
                'name': lbl,
                'color': color,
                'type': 'rhythm',
                'critical': 'ventricular' in lbl_lower
            })
            
    # 2. Add individual ectopic beats (ALL V, S, AF beats) with rhythm context and interval data
    label_beat_sequences(auto_beats)
    
    for beat in auto_beats:
        code = beat.get('short_code', '')
        pattern = beat.get('rhythm_pattern', '')
        beat_color = beat.get('color', '')
        
        # Show badges for:
        # 1. Explicitly ectopic beats (V, S, P, AF codes)
        # 2. Beats with non-green colors (indicating abnormal detection)
        # 3. Beats with patterns assigned (Bigeminy, Couplet, etc.)
        is_ectopic = code in ['V', 'S', 'P', 'AF']
        is_colored_abnormal = beat_color and beat_color not in ['#00FF00', '#FFFFFF', '']
        has_pattern = pattern and pattern not in ['Normal', '']
        
        # Show badge if ANY of these conditions are true
        if not (is_ectopic or is_colored_abnormal or has_pattern):
            continue
        
        # Default pattern if not set
        if not pattern:
            if code == 'V':
                pattern = 'PVC'
            elif code == 'S':
                pattern = 'PAC'
            elif code == 'P':
                pattern = 'Paced'
            elif code == 'AF':
                pattern = 'AFib'
            else:
                pattern = 'Abnormal Beat'
        
        # Determine color and priority based on pattern
        color, priority = _get_badge_appearance(code, pattern, beat)
        
        # Extract PR, QRS, QT/QTc if available from auto-detection results
        qrs = beat.get('qrs_ms')
        qt = beat.get('qt_ms')
        pr = beat.get('pr_ms')
        hr = beat.get('heart_rate_bpm')
        
        details = []
        if qrs is not None and qrs > 0:
            details.append(f"QRS:{int(qrs)}")
        if qt is not None and qt > 0:
            details.append(f"QT:{int(qt)}")
        if pr is not None and pr > 0:
            details.append(f"PR:{int(pr)}")
        if hr is not None and hr > 0:
            details.append(f"HR:{int(hr)}")
            
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
            'type': 'beat',
            'priority': priority,
            'critical': beat.get('critical', False),
            'run_length': beat.get('run_length', 1)
        })
            
    badges.sort(key=lambda x: x['timestamp'])
    return badges


def _get_badge_appearance(code: str, pattern: str, beat: Dict[str, Any]) -> tuple:
    """
    Determine badge color and priority based on beat type and pattern.
    
    Returns: (color_hex, priority_level)
      priority: 0=normal, 1=minor, 2=moderate, 3=critical
    """
    # CRITICAL: VT runs (3+ PVCs)
    if ('VT Run' in pattern and code == 'V') or beat.get('critical', False):
        return '#FF0000', 3  # Bright red, highest priority
    
    # HIGH: PVC patterns
    if code == 'V':
        if 'Bigeminy' in pattern or 'Trigeminy' in pattern or 'Quadrigeminy' in pattern:
            return '#FF6600', 2  # Orange-red, high priority
        elif 'Couplet' in pattern:
            return '#FF3333', 2  # Red, high priority
        else:  # Single PVC
            return '#FF6666', 1  # Light red, moderate priority
    
    # MODERATE: SVT patterns
    if code == 'S':
        if 'Run' in pattern:
            return '#00CCFF', 2  # Bright cyan
        elif 'Bigeminy' in pattern or 'Trigeminy' in pattern:
            return '#00FFFF', 2  # Cyan
        elif 'Couplet' in pattern:
            return '#66FFFF', 1  # Light cyan
        else:  # Single PAC
            return '#99FFFF', 1  # Very light cyan
    
    # SPECIAL: Atrial Fibrillation
    if code == 'AF' or beat.get('afib_episode', False):
        return '#FF00FF', 2  # Magenta, high priority
    
    # PACED
    if code == 'P':
        return '#CC66FF', 1  # Purple, moderate priority
    
    # Default
    return '#FFFFFF', 0  # White, normal

