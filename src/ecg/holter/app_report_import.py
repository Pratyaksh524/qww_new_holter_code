from __future__ import annotations

import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from .file_format import ECGHFileWriter
from .session_store import write_session_metadata


PUBLIC_REPORTS_ENDPOINT = "https://pmltkfluqk.execute-api.us-east-1.amazonaws.com/dev/api/public/reports"
LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def normalize_mobile_no(mobile_no: str) -> str:
    digits = "".join(ch for ch in str(mobile_no or "") if ch.isdigit())
    if len(digits) == 12 and digits.startswith("91"):
        digits = digits[2:]
    return digits


def fetch_public_reports(mobile_no: str, session: Optional[requests.Session] = None) -> List[Dict[str, Any]]:
    mobile_no = normalize_mobile_no(mobile_no)
    if len(mobile_no) != 10:
        return []

    sess = session or requests.Session()
    sess.headers.update({"Accept": "application/json"})
    resp = sess.get(PUBLIC_REPORTS_ENDPOINT, params={"mobile_no": mobile_no}, timeout=15)
    payload = resp.json()
    if isinstance(payload, list):
        return [r for r in payload if isinstance(r, dict)]
    if isinstance(payload, dict):
        reports = payload.get("data") or payload.get("reports") or payload.get("items") or []
        if isinstance(reports, list):
            return [r for r in reports if isinstance(r, dict)]
        if isinstance(reports, dict):
            return [reports]
    return []


def _parse_numeric_series(value: Any) -> List[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.astype(float).tolist()]
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            try:
                out.append(float(item))
            except Exception:
                continue
        return out
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                return _parse_numeric_series(parsed)
            except Exception:
                pass
        if "|" in text:
            try:
                parsed = json.loads("[" + text.replace("|", ",") + "]")
                if isinstance(parsed, list):
                    return _parse_numeric_series(parsed)
            except Exception:
                pass
        out = []
        for chunk in text.replace("|", ",").split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            try:
                out.append(float(chunk))
            except Exception:
                continue
        return out
    try:
        return [float(value)]
    except Exception:
        return []


def _normalize_series_for_ecgh(series: List[float], lead_name: str = "") -> List[int]:
    arr = np.asarray(series, dtype=float)
    if arr.size == 0:
        return []
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return []

    # If the source already looks like ECGH/ADC data, keep it as-is.
    if float(np.nanmin(finite)) >= 0.0 and float(np.nanmax(finite)) <= 4095.0:
        med = float(np.median(finite))
        if 1500.0 <= med <= 2600.0:
            return [int(round(float(np.clip(v, 0.0, 4095.0)))) for v in arr]

    centered = arr - float(np.median(finite))
    peak = max(float(np.max(np.abs(centered))), 1.0)
    if peak <= 0.0:
        return [2048 for _ in range(arr.size)]

    # Match the rest of the app's display scale: center around 2048 and cap deflection.
    scaled = np.clip(centered / peak * 420.0, -650.0, 650.0)
    waveform = 2048.0 + scaled
    return [int(round(float(np.clip(v, 0.0, 4095.0)))) for v in waveform]


def extract_lead_map(report_obj: Dict[str, Any]) -> Tuple[Dict[str, List[float]], float]:
    ecg_data = report_obj.get("ecg_data") if isinstance(report_obj.get("ecg_data"), dict) else {}
    sampling_rate = 500.0
    for source in (report_obj, ecg_data):
        if isinstance(source, dict):
            sr = source.get("sampling_rate") or source.get("samplingRate")
            if sr not in (None, ""):
                try:
                    sampling_rate = float(sr)
                    break
                except Exception:
                    pass

    lead_map = {lead: [] for lead in LEAD_NAMES}

    def _fill_from_mapping(mapping: dict) -> bool:
        if not isinstance(mapping, dict):
            return False
        filled = False
        for lead in LEAD_NAMES:
            series = _parse_numeric_series(mapping.get(lead))
            if series:
                lead_map[lead] = series
                filled = True
        return filled

    def _fill_from_compact_device(device_value: str) -> bool:
        if not isinstance(device_value, str) or "|" not in device_value:
            return False
        frames = [x.strip() for x in device_value.split("|") if x.strip()]
        if not frames:
            return False
        for fr in frames:
            try:
                vals = json.loads(fr)
            except Exception:
                continue
            if isinstance(vals, list) and len(vals) >= 12:
                for idx, lead in enumerate(LEAD_NAMES):
                    try:
                        lead_map[lead].append(float(vals[idx]))
                    except Exception:
                        lead_map[lead].append(0.0)
        return any(len(v) > 0 for v in lead_map.values())

    # Accept the same waveform shapes as the dashboard report mapper.
    if _fill_from_mapping(ecg_data.get("leads") if isinstance(ecg_data.get("leads"), dict) else None):
        return lead_map, sampling_rate
    if _fill_from_mapping(ecg_data.get("leads_data") if isinstance(ecg_data.get("leads_data"), dict) else None):
        return lead_map, sampling_rate

    device_data = ecg_data.get("device_data")
    if _fill_from_mapping(device_data if isinstance(device_data, dict) else None):
        return lead_map, sampling_rate
    if _fill_from_compact_device(device_data):
        return lead_map, sampling_rate

    # Also accept top-level shapes used by some API responses.
    if _fill_from_mapping(report_obj.get("leads") if isinstance(report_obj.get("leads"), dict) else None):
        return lead_map, sampling_rate
    if _fill_from_mapping(report_obj.get("leads_data") if isinstance(report_obj.get("leads_data"), dict) else None):
        return lead_map, sampling_rate
    device_data = report_obj.get("device_data")
    if _fill_from_mapping(device_data if isinstance(device_data, dict) else None):
        return lead_map, sampling_rate
    if _fill_from_compact_device(device_data):
        return lead_map, sampling_rate

    possible_keys = [
        ("I", ["lead1_reading", "lead_1_reading", "lead1", "lead_i"]),
        ("II", ["lead2_reading", "lead_2_reading", "lead2", "lead_ii"]),
        ("III", ["lead3_reading", "lead_3_reading", "lead3", "lead_iii"]),
        ("aVR", ["leadvr_reading", "lead_vr_reading", "avr"]),
        ("aVL", ["leadvl_reading", "lead_vl_reading", "avl"]),
        ("aVF", ["leadavf_reading", "lead_vf_reading", "avf"]),
        ("V1", ["leadv1_reading", "lead_v1_reading", "v1"]),
        ("V2", ["leadv2_reading", "lead_v2_reading", "v2"]),
        ("V3", ["leadv3_reading", "lead_v3_reading", "v3"]),
        ("V4", ["leadv4_reading", "lead_v4_reading", "v4"]),
        ("V5", ["leadv5_reading", "lead_v5_reading", "v5"]),
        ("V6", ["leadv6_reading", "lead_v6_reading", "v6"]),
    ]

    lower_keys = {k.lower(): k for k in report_obj.keys()}
    for lead_name, variants in possible_keys:
        for variant in variants:
            actual_key = lower_keys.get(variant.lower())
            if actual_key and actual_key in report_obj:
                lead_map[lead_name] = _parse_numeric_series(report_obj.get(actual_key))
                break

    return lead_map, sampling_rate


def build_app_session_from_report(
    report_obj: Dict[str, Any],
    mobile_no: str,
    output_dir: str,
) -> Tuple[str, Dict[str, Any]]:
    try:
        from ecg.ecg_calculations import calculate_all_ecg_metrics
    except Exception:
        calculate_all_ecg_metrics = None

    patient_details = report_obj.get("patient_details") if isinstance(report_obj.get("patient_details"), dict) else {}
    fallback_patient = report_obj.get("patient") if isinstance(report_obj.get("patient"), dict) else {}
    name = (
        report_obj.get("name")
        or report_obj.get("patient_name")
        or patient_details.get("name")
        or patient_details.get("patient_name")
        or fallback_patient.get("name")
        or "Unknown"
    )
    age = report_obj.get("age") or patient_details.get("age") or fallback_patient.get("age") or ""
    gender = report_obj.get("gender") or patient_details.get("gender") or fallback_patient.get("gender") or ""
    report_id = (
        report_obj.get("report_id")
        or report_obj.get("reportId")
        or report_obj.get("id")
        or patient_details.get("report_id")
        or ""
    )
    report_date = (
        report_obj.get("report_date")
        or report_obj.get("reportDate")
        or report_obj.get("date")
        or patient_details.get("report_date")
        or ""
    )
    lead_map, sampling_rate = extract_lead_map(report_obj)
    fs = max(int(round(float(sampling_rate) or 500)), 1)
    target_duration_sec = 10.0
    target_samples = max(1, int(round(target_duration_sec * fs)))

    cropped_lead_map: Dict[str, List[int]] = {}
    for lead in LEAD_NAMES:
        series = list(lead_map.get(lead) or [])[:target_samples]
        cropped_lead_map[lead] = _normalize_series_for_ecgh(series, lead)

    lead_ii = np.asarray(
        cropped_lead_map.get("II") or next((cropped_lead_map[l] for l in LEAD_NAMES if cropped_lead_map.get(l)), []),
        dtype=float,
    )
    duration_sec = float(len(lead_ii)) / float(fs) if len(lead_ii) else 0.0

    safe_name = re.sub(r"[^A-Za-z0-9_-]+", "_", str(name).strip() or "Unknown")[:28].strip("_") or "Unknown"
    safe_rid = re.sub(r"[^A-Za-z0-9_-]+", "_", str(report_id).strip() or "report")[:24].strip("_") or "report"
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session_dir = os.path.join(output_dir, f"{ts}_{safe_name}_APP_{safe_rid}")
    os.makedirs(session_dir, exist_ok=True)

    mobile_clean = normalize_mobile_no(mobile_no)
    patient_info = {
        "name": str(name).strip() or "Unknown",
        "age": age,
        "gender": gender,
        "phone": mobile_clean,
        "mobile_no": mobile_clean,
        "report_id": str(report_id).strip(),
        "report_date": str(report_date).strip(),
        "reporter": "APP",
        "source": "APP",
    }

    ecgh_path = os.path.join(session_dir, "recording.ecgh")
    try:
        writer = ECGHFileWriter(ecgh_path, patient_info=patient_info, fs=fs)
        max_len = min(target_samples, max((len(v) for v in cropped_lead_map.values()), default=0))
        for idx in range(max_len):
            packet = {}
            for lead in LEAD_NAMES:
                series = cropped_lead_map.get(lead, [])
                if idx < len(series):
                    packet[lead] = int(series[idx])
                else:
                    packet[lead] = 2048
            writer.write_packet(packet)
        writer.finalize()
    except Exception as exc:
        try:
            import shutil

            shutil.rmtree(session_dir, ignore_errors=True)
        except Exception:
            pass
        raise RuntimeError(f"Could not create replay file for imported app report: {exc}")

    metadata = {
        "session_dir": session_dir,
        "patient_info": dict(patient_info),
        "reporter": "APP",
        "source": "APP",
        "report_id": str(report_id).strip(),
        "report_date": str(report_date).strip(),
        "mobile_no": mobile_clean,
        "summary": {
            "patient_info": dict(patient_info),
            "reporter": "APP",
            "source": "APP",
            "report_id": str(report_id).strip(),
            "report_date": str(report_date).strip(),
            "duration_sec": duration_sec,
            "fs": fs,
        },
    }
    write_session_metadata(session_dir, metadata)

    try:
        with open(os.path.join(session_dir, "patient.json"), "w", encoding="utf-8") as handle:
            json.dump(patient_info, handle, indent=2)
    except Exception:
        pass

    summary_metrics = {
        "t": 0.0,
        "duration": duration_sec,
        "partial": False,
        "best_lead": "II" if len(lead_ii) else "I",
        "quality": 1.0,
        "hr_mean": 0.0,
        "rr_ms": 0.0,
        "pr_ms": 0,
        "qrs_ms": 0,
        "qt_ms": 0.0,
        "qtc_ms": 0,
        "qtcf_ms": 0,
        "beat_count": 0,
        "arrhythmias": [],
        "n_beats": 0,
    }

    if calculate_all_ecg_metrics is not None and len(lead_ii) > 200:
        try:
            metrics = calculate_all_ecg_metrics(lead_ii, fs=float(fs), instance_id=f"app:{safe_rid}")
            summary_metrics.update(
                {
                    "hr_mean": float(metrics.get("heart_rate", 0) or 0),
                    "rr_ms": float(metrics.get("rr_interval", 0.0) or 0.0),
                    "pr_ms": float(metrics.get("pr_interval", 0) or 0),
                    "qrs_ms": float(metrics.get("qrs_duration", 0) or 0),
                    "qt_ms": float(metrics.get("qt_interval", 0.0) or 0.0),
                    "qtc_ms": float(metrics.get("qtc_interval", 0) or 0),
                    "qtcf_ms": float(metrics.get("qtc_interval", 0) or 0),
                }
            )
        except Exception:
            pass

    try:
        clinical = report_obj.get("clinical_findings", {}) if isinstance(report_obj.get("clinical_findings"), dict) else {}
        arrhythmia = clinical.get("arrhythmia", [])
        if isinstance(arrhythmia, list):
            summary_metrics["arrhythmias"] = arrhythmia
    except Exception:
        pass

    try:
        with open(os.path.join(session_dir, "metrics.jsonl"), "w", encoding="utf-8") as handle:
            handle.write(json.dumps(summary_metrics) + "\n")
    except Exception:
        pass

    return session_dir, patient_info

