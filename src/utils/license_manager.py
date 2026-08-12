"""
utils/license_manager.py
========================
CardioX client-side license validation — v2.0 (Three-Pillar Architecture).

Pillar 1 — Hardware Fingerprint
    SHA-256 of 5 WMI fields: BIOS serial, Motherboard UUID, CPU ID,
    Disk serial, MAC address.  Stable across reboots; cannot move to
    another machine without hardware spoofing.

Pillar 2 — RhythmUltra Device Lock
    USB scan for matching VID/PID on every startup.  Serial number is
    compared against the bound serial stored in the token.  Missing or
    mismatched device = blocked immediately.

Pillar 3 — Server-Signed Token
    On successful registration the server issues an HMAC-signed JWT-like
    token stored at %APPDATA%\\Deckmount\\cardiox.lic.  The token carries
    fingerprint, RhythmUltra serial, license key, seat number, and the
    timestamp of the last successful server heartbeat.

Startup validation sequence (5 checks):
    1. Token file exists on disk.
    2. Token HMAC signature is valid (tamper detection).
    3. Hardware fingerprint in token matches this machine.
    4. RhythmUltra connected and serial matches token.
    5. Server heartbeat (POST /heartbeat) — attempted on startup, with offline grace.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import platform
import struct
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dotenv import find_dotenv, load_dotenv

# Load project .env so standalone imports and the app share the same settings.
load_dotenv(find_dotenv(usecwd=True), override=False)

# ── Configuration ─────────────────────────────────────────────────────────────

LICENSE_SERVER_URL: str = os.getenv(
    "LICENSE_SERVER_URL",
    "https://m4qoae4d8e.execute-api.us-east-1.amazonaws.com/prod/api/v1",
)


def _load_hmac_secret() -> bytes:
    """Load the shared HMAC secret from env as UTF-8 bytes."""
    raw = os.getenv(
        "LICENSE_HMAC_SECRET",
        "535ebf0647a740dd37633abc3692cd2a01dfe2e13b77738192b7d7bc0bb48df1",
    ).strip()
    return raw.encode("utf-8")


_HMAC_SECRET: bytes = _load_hmac_secret()
LICENSE_API_TOKEN: str = os.getenv("LICENSE_API_TOKEN", "").strip()

SOFTWARE_VERSION: str = "2.0.0"
PRODUCT_CODE: str = "CARDIOX"

# Offline grace window (days) — app runs without internet for this many days.
# After this window the app blocks and requires an internet connection to reactivate.
OFFLINE_GRACE_DAYS: int = 7
# How many seconds between mandatory server heartbeats.
HEARTBEAT_INTERVAL_SECONDS: int = OFFLINE_GRACE_DAYS * 86400
# Once this many days (or fewer) of the offline window remain, the UI warns the user
# on every launch that internet is required for verification and that the software will
# otherwise stop. The heartbeat is mandatory by design: it is what prevents a one-time
# signup from being used indefinitely on a machine that never contacts the server again.
OFFLINE_WARNING_DAYS: int = 3

# ── RhythmUltra USB Identity ───────────────────────────────────────────────────
# Set RhythmUltra_VID / RhythmUltra_PID in .env or environment.
# Values are integers (decimal or hex string accepted).
def _parse_usb_id(env_key: str, default: int) -> int:
    raw = os.getenv(env_key, "").strip()
    if not raw:
        return default
    try:
        return int(raw, 16) if raw.startswith("0x") or raw.startswith("0X") else int(raw)
    except ValueError:
        return default

RHYTHMULTRA_VID: int = _parse_usb_id("RHYTHMULTRA_VID", 0x0000) or _parse_usb_id("RhythmUltra_VID", 0x0000)
RHYTHMULTRA_PID: int = _parse_usb_id("RHYTHMULTRA_PID", 0x0000) or _parse_usb_id("RhythmUltra_PID", 0x0000)
RhythmUltra_VID = RHYTHMULTRA_VID
RhythmUltra_PID = RHYTHMULTRA_PID

# ── File Paths ────────────────────────────────────────────────────────────────
# Token lives in %APPDATA%\Deckmount\cardiox.lic (per SDD §3.4)
_APPDATA = Path(os.getenv("APPDATA", Path.home()))
_TOKEN_DIR: Path = _APPDATA / "Deckmount"
_TOKEN_FILE: Path = _TOKEN_DIR / "cardiox.lic"
# Sidecar: mutable metadata (last_server_check, fingerprint) stored separately
# so the server-issued JWT in cardiox.lic is NEVER overwritten by the client.
_META_FILE: Path = _TOKEN_DIR / "cardiox_meta.json"
_AUDIT_LOG_FILE: Path = _TOKEN_DIR / "audit_log.jsonl"

# Legacy cache dir (kept for backward compat — old key file)
_LEGACY_CACHE_DIR = Path(os.getenv("LOCALAPPDATA", Path.home())) / "Deckmount" / "CardioX"
_LEGACY_KEY_FILE: Path = _LEGACY_CACHE_DIR / "license.key"
_LEGACY_CACHE_FILE: Path = _LEGACY_CACHE_DIR / "license.cache"
_DEVICE_ID_FILE: Path = _LEGACY_CACHE_DIR / "device.id"  # no longer used; kept to remove on upgrade

# Base-32 alphabet — no ambiguous chars (0, O, 1, I)
_B32_ALPHA = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"


def _append_audit_event(action: str, **details) -> None:
    """Append a lightweight local audit event for support and recovery tracing."""
    try:
        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        payload = {
            "action": action,
            "timestamp": int(time.time()),
            "local_time": int(time.time()),
        }
        if details:
            payload.update(details)
        with _AUDIT_LOG_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception as e:
        print(f"[License] Could not write audit event {action}: {e}")


def _current_unix_time() -> int:
    """Return the current local unix timestamp."""
    return int(time.time())


def _version_tuple(version: str) -> Tuple[int, ...]:
    """Convert a dotted version string into a tuple for safe comparison."""
    parts: List[int] = []
    for chunk in str(version or "").split("."):
        digits = "".join(ch for ch in chunk if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts) if parts else (0,)


def _is_minimum_version_forced(minimum_version: str) -> bool:
    """Return True when the running build is below the requested minimum."""
    if not minimum_version:
        return False
    return _version_tuple(SOFTWARE_VERSION) < _version_tuple(minimum_version)


def _stable_hardware_fingerprint() -> str:
    """
    Return a stable hardware fingerprint derived from motherboard / BIOS / CPU.

    This is used as a compatibility anchor so minor upgrades like SSD, RAM,
    GPU, or monitor changes do not unnecessarily invalidate an otherwise valid
    license token.
    """
    fields = _collect_wmi_fields()
    raw = "|".join([
        fields.get("bios_serial", ""),
        fields.get("mb_uuid", ""),
        fields.get("cpu_id", ""),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _license_meta_exists() -> bool:
    """Return True when the mutable sidecar metadata file is present."""
    return _META_FILE.exists()


def _enforce_rhythmultra_lock() -> bool:
    """
    Decide whether the RhythmUltra device must be present at startup.

    Windows production builds keep the hard lock enabled. For local Mac/Linux
    development runs, the device requirement can be bypassed unless explicitly
    forced back on via `CARDIOX_REQUIRE_RHYTHMULTRA=1`.
    """
    forced = os.getenv("CARDIOX_REQUIRE_RHYTHMULTRA", "").strip().lower() in {"1", "true", "yes", "on"}
    if forced:
        return True
    return sys.platform == "win32"


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — Hardware Fingerprint (WMI 5-field SHA-256)
# ══════════════════════════════════════════════════════════════════════════════

def _run_wmic(args: List[str]) -> str:
    """
    Run a WMI-style hardware query and return stdout, stripped.

    We prefer `wmic` for compatibility, but Windows 11 commonly omits it.
    In that case we fall back to PowerShell/CIM so the app can still derive
    the same hardware fingerprint on modern systems.
    """

    def _clean_output(text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        for line in lines:
            lower = line.lower()
            if lower in ("serialnumber", "uuid", "processorid", "caption", "name"):
                continue
            if lower.startswith("to be filled") or lower == "none":
                return ""
            if line:
                return line
        return ""

    def _run_command(cmd: List[str]) -> str:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        return _clean_output(result.stdout)

    query = tuple(a.lower() for a in args)
    candidates: List[List[str]] = []

    # 1) Legacy WMIC path.
    candidates.append(["wmic"] + args)

    # 2) Modern PowerShell/CIM fallback.
    if sys.platform == "win32":
        ps_script_map = {
            ("bios", "get", "serialnumber"): [
                "(Get-CimInstance Win32_BIOS | Select-Object -First 1 -ExpandProperty SerialNumber)",
                "(Get-WmiObject Win32_BIOS | Select-Object -First 1 -ExpandProperty SerialNumber)",
                "(Get-ItemProperty -Path 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\BIOS' | Select-Object -ExpandProperty SystemSerialNumber)",
            ],
            ("csproduct", "get", "uuid"): [
                "(Get-CimInstance Win32_ComputerSystemProduct | Select-Object -First 1 -ExpandProperty UUID)",
                "(Get-WmiObject Win32_ComputerSystemProduct | Select-Object -First 1 -ExpandProperty UUID)",
            ],
            ("cpu", "get", "processorid"): [
                "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty ProcessorId)",
                "(Get-WmiObject Win32_Processor | Select-Object -First 1 -ExpandProperty ProcessorId)",
                "(Get-ItemProperty -Path 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0' | Select-Object -ExpandProperty Identifier)",
            ],
            ("diskdrive", "get", "serialnumber"): [
                "(Get-CimInstance Win32_DiskDrive | Where-Object { $_.SerialNumber } | Select-Object -First 1 -ExpandProperty SerialNumber)",
                "(Get-WmiObject Win32_DiskDrive | Where-Object { $_.SerialNumber } | Select-Object -First 1 -ExpandProperty SerialNumber)",
            ],
            ("os", "get", "caption"): [
                "(Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 -ExpandProperty Caption)",
                "(Get-WmiObject Win32_OperatingSystem | Select-Object -First 1 -ExpandProperty Caption)",
            ],
        }
        for expr in ps_script_map.get(query, []):
            candidates.append([
                "powershell.exe",
                "-NoProfile",
                "-ExecutionPolicy",
                "Bypass",
                "-Command",
                f"$ErrorActionPreference='Stop'; {expr}",
            ])

    for cmd in candidates:
        try:
            value = _run_command(cmd)
            if value:
                return value
        except Exception:
            continue

    return ""

_cached_wmi_fields: Optional[Dict[str, str]] = None
_cached_machine_info: Optional[Dict[str, str]] = None


def _collect_wmi_fields() -> Dict[str, str]:
    """
    Collect the 5 hardware identifiers used in the fingerprint.
    Returns a dict — missing/blank fields are empty strings.
    Cached after first collection to avoid repeating slow subprocess/WMI spawns.
    """
    global _cached_wmi_fields
    if _cached_wmi_fields is not None:
        return _cached_wmi_fields

    fields: Dict[str, str] = {}

    if sys.platform == "darwin":
        # Fast native macOS hardware identity (eliminates 4 failing wmic subprocess calls)
        try:
            cmd = ["ioreg", "-rd1", "-c", "IOPlatformExpertDevice"]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=1)
            for line in res.stdout.splitlines():
                if "IOPlatformUUID" in line:
                    fields["mb_uuid"] = line.split("=")[-1].replace('"', '').strip()
                if "IOPlatformSerialNumber" in line:
                    fields["bios_serial"] = line.split("=")[-1].replace('"', '').strip()
        except Exception:
            pass
        fields.setdefault("bios_serial", platform.node() or "MAC_SERIAL")
        fields.setdefault("mb_uuid", f"MAC_UUID_{uuid.getnode():012x}")
        fields.setdefault("cpu_id", platform.processor() or "Apple_Silicon")
        fields.setdefault("disk_serial", "MAC_SYSTEM_DISK")
        try:
            fields["mac"] = f"{uuid.getnode():012x}"
        except Exception:
            fields["mac"] = ""
        _cached_wmi_fields = fields
        return fields

    # 1. BIOS serial — never changes
    fields["bios_serial"] = _run_wmic(["bios", "get", "serialnumber"])

    # 2. Motherboard UUID — never changes
    fields["mb_uuid"] = _run_wmic(["csproduct", "get", "uuid"])

    # 3. CPU Processor ID — never changes
    fields["cpu_id"] = _run_wmic(["cpu", "get", "processorid"])

    # 4. Primary disk serial — changes if disk replaced
    fields["disk_serial"] = _run_wmic(["diskdrive", "get", "serialnumber"])

    # 5. MAC address — changes if NIC replaced
    try:
        fields["mac"] = f"{uuid.getnode():012x}"
    except Exception:
        fields["mac"] = ""

    _cached_wmi_fields = fields
    return fields



def _collect_machine_info() -> Dict[str, str]:
    """
    Collect display-only machine metadata (not used in fingerprint).
    Cached after first collection to avoid repeating slow WMI/subprocess runs.
    """
    global _cached_machine_info
    if _cached_machine_info is not None:
        return _cached_machine_info

    info: Dict[str, str] = {}
    try:
        info["pc_name"] = os.getenv("COMPUTERNAME", "") or platform.node()
    except Exception:
        info["pc_name"] = ""
    try:
        info["windows_version"] = _run_wmic(["os", "get", "caption"]) or platform.version()
    except Exception:
        info["windows_version"] = platform.version()
    
    _cached_machine_info = info
    return info


def get_hardware_fingerprint() -> str:
    """
    Return a stable SHA-256 hardware fingerprint derived from 5 WMI fields.

    OEM machines may have blank BIOS serials — the remaining fields compensate.
    All 5 values are concatenated with '|' and hashed with SHA-256.
    """
    fields = _collect_wmi_fields()
    # Build ordered concatenation; blank fields contribute empty string (not omitted)
    raw = "|".join([
        fields.get("bios_serial", ""),
        fields.get("mb_uuid", ""),
        fields.get("cpu_id", ""),
        fields.get("disk_serial", ""),
        fields.get("mac", ""),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


_detected_device_serial: Optional[str] = None
_restricted_mode: bool = False


def is_license_restricted() -> bool:
    """Return True if the license is currently in restricted mode (offline stable mismatch)."""
    global _restricted_mode
    return _restricted_mode


def set_detected_device_serial(serial: Optional[str]):
    """Set the detected device serial ID from active COM port scan."""
    global _detected_device_serial
    _detected_device_serial = serial


def get_machine_context() -> Dict[str, str]:
    """Return machine metadata for server registration payload."""
    info = _collect_machine_info()
    wmi = _collect_wmi_fields()
    
    machine_serial = wmi.get("bios_serial", "").strip()
    if not machine_serial or machine_serial.lower() == "none" or "to be filled" in machine_serial.lower():
        global _detected_device_serial
        if _detected_device_serial:
            machine_serial = _detected_device_serial
            
    return {
        "machine_name": info.get("pc_name", platform.node()),
        "machine_serial_id": machine_serial,
        "windows_version": info.get("windows_version", ""),
        "machine_os": f"{platform.system()} {platform.release()}".strip(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 2 — RhythmUltra USB Device Lock
# ══════════════════════════════════════════════════════════════════════════════

def _list_usb_ports():
    """Return list of (vid, pid, serial) tuples from all connected COM ports."""
    try:
        import serial.tools.list_ports  # type: ignore
        result = []
        for port in serial.tools.list_ports.comports():
            vid = getattr(port, "vid", None)
            pid = getattr(port, "pid", None)
            serial_number = getattr(port, "serial_number", None) or ""
            result.append((vid, pid, serial_number.strip()))
        return result
    except ImportError:
        return _list_usb_via_wmic()
    except Exception:
        return []


def _list_usb_via_wmic() -> List[Tuple[Optional[int], Optional[int], str]]:
    """Fallback USB enumeration using WMI when pyserial is unavailable."""
    try:
        result = subprocess.run(
            ["wmic", "path", "Win32_PnPEntity", "get", "DeviceID,Name"],
            capture_output=True, text=True, timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
        )
        devices = []
        for line in result.stdout.splitlines():
            line = line.strip()
            if "USB\\VID_" in line.upper():
                # Parse VID/PID from DeviceID like USB\VID_1234&PID_5678\SERIAL
                try:
                    parts = line.upper().split("VID_")[1]
                    vid_str = parts[:4]
                    pid_str = parts.split("PID_")[1][:4] if "PID_" in parts else "0000"
                    vid = int(vid_str, 16)
                    pid = int(pid_str, 16)
                    # Serial is after last backslash
                    serial = line.rsplit("\\", 1)[-1].strip() if "\\" in line else ""
                    devices.append((vid, pid, serial))
                except Exception:
                    continue
        return devices
    except Exception:
        return []


_vid_pid_warning_printed = False  # Print the VID/PID warning only once per process


def get_rhythmultra_serial() -> Optional[str]:
    """
    Scan USB ports or COM ports for a RhythmUltra device.
    Uses active serial probing (VERSION and MACHINE_SERIAL commands) to detect the hardware,
    falling back to VID/PID match if active probing is unsuccessful.
    """
    global _detected_device_serial
    if _detected_device_serial:
        return _detected_device_serial

    # 1. Active Probing Fallback (queries the actual hardware)
    try:
        import serial
        import serial.tools.list_ports
        from ecg.serial.hardware_commands import HardwareCommandHandler
        
        ports = list(serial.tools.list_ports.comports())
        filtered_ports = []
        for p in ports:
            desc = str(getattr(p, "description", "") or "")
            dev = str(getattr(p, "device", "") or "")
            if dev.upper() == "COM1" and "Communications Port" in desc:
                continue
            if "Bluetooth" in desc:
                continue
            filtered_ports.append(p)
            
        for port in filtered_ports:
            try:
                ser = serial.Serial(
                    port.device,
                    115200,
                    timeout=0.2,
                    write_timeout=0.2,
                )
                try:
                    handler = HardwareCommandHandler(ser)
                    success_v, version, _ = handler.send_version_command(timeout=0.4, quiet=True)
                    if success_v and version:
                        success_s, serial_num, _ = handler.send_machine_serial_command(timeout=0.4, quiet=True)
                        if success_s and serial_num:
                            # Cache it globally
                            set_detected_device_serial(serial_num)
                            return serial_num
                finally:
                    try:
                        ser.close()
                    except Exception:
                        pass
            except Exception:
                continue
    except Exception as e:
        print(f"[LicenseManager] Active port probe skipped: {e}")

    # 2. USB Descriptor Fallback (traditional matching)
    if RHYTHMULTRA_VID != 0 or RHYTHMULTRA_PID != 0:
        ports = _list_usb_ports()
        for vid, pid, serial in ports:
            if vid == RHYTHMULTRA_VID and pid == RHYTHMULTRA_PID:
                serial_val = serial if serial else "UNKNOWN_SERIAL"
                set_detected_device_serial(serial_val)
                return serial_val

    global _vid_pid_warning_printed
    if RHYTHMULTRA_VID == 0 and RHYTHMULTRA_PID == 0:
        if not _vid_pid_warning_printed:
            print(
                "[License] WARNING: RHYTHMULTRA_VID and RHYTHMULTRA_PID are not configured. "
                "Active serial probe did not detect a device. Set them in .env if using legacy USB matching."
            )
            _vid_pid_warning_printed = True

    return None


def is_rhythmultra_connected() -> bool:
    """Return True if a RhythmUltra device with matching VID/PID is connected."""
    return get_rhythmultra_serial() is not None

# Legacy aliases for backward compatibility
get_RhythmUltra_serial = get_rhythmultra_serial
is_RhythmUltra_connected = is_rhythmultra_connected


# ══════════════════════════════════════════════════════════════════════════════
# PILLAR 3 — Server-Signed Token (cardiox.lic)
# ══════════════════════════════════════════════════════════════════════════════

def _token_hmac(payload_bytes: bytes) -> str:
    """Compute HMAC-SHA256 hex digest of a token payload."""
    return hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()


def _save_token_as_jwt(token_data: Dict) -> str:
    """Helper to convert token dict to a standard signed JWT string."""
    jwt_payload = dict(token_data)
    if "fingerprint" in token_data:
        jwt_payload["hardware_fingerprint"] = token_data["fingerprint"]
    if "last_server_check" in token_data:
        jwt_payload["issued_at"] = token_data["last_server_check"]
        jwt_payload["iat"] = token_data["last_server_check"]
    if "expires" in token_data:
        jwt_payload["exp"] = token_data["expires"]
        
    header = {"alg": "HS256", "typ": "JWT"}
    header_b64 = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).decode("utf-8").rstrip("=")
    payload_b64 = base64.urlsafe_b64encode(json.dumps(jwt_payload, separators=(",", ":"), sort_keys=True).encode("utf-8")).decode("utf-8").rstrip("=")
    
    msg = f"{header_b64}.{payload_b64}".encode("utf-8")
    sig_bytes = hmac.new(_HMAC_SECRET, msg, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig_bytes).decode("utf-8").rstrip("=")
    
    return f"{header_b64}.{payload_b64}.{sig_b64}"


def save_token_file(token_data: Dict | str) -> None:
    """
    Write the license token to %APPDATA%\\Deckmount\\cardiox.lic.
    If the parameter is a string (e.g. raw JWT), save it directly.
    If it is a dict, format it to match existing file's format or default to JWT.
    """
    try:
        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        if isinstance(token_data, str):
            _TOKEN_FILE.write_text(token_data.strip(), encoding="utf-8")
            print(f"[License] Raw token string saved to {_TOKEN_FILE}")
            return
            
        is_jwt_format = False
        if _TOKEN_FILE.exists():
            try:
                exist_text = _TOKEN_FILE.read_text(encoding="utf-8").strip()
                if exist_text.count(".") == 2:
                    is_jwt_format = True
            except Exception:
                pass
                
        if "localhost" not in LICENSE_SERVER_URL and "127.0.0.1" not in LICENSE_SERVER_URL:
            is_jwt_format = True
            
        if is_jwt_format:
            jwt_str = _save_token_as_jwt(token_data)
            _TOKEN_FILE.write_text(jwt_str, encoding="utf-8")
            print(f"[License] Token saved as JWT to {_TOKEN_FILE}")
        else:
            payload_bytes = json.dumps(token_data, sort_keys=True).encode("utf-8")
            sig = _token_hmac(payload_bytes)
            envelope = {"payload": token_data, "sig": sig}
            _TOKEN_FILE.write_text(json.dumps(envelope, indent=2), encoding="utf-8")
            print(f"[License] Token envelope saved to {_TOKEN_FILE}")
    except Exception as e:
        print(f"[License] Token write failed: {e}")


def load_token_file() -> Optional[Dict]:
    """
    Read and verify the license token from disk.
    Supports standard base64url-encoded JWT tokens and legacy JSON envelopes.
    Returns the payload dict, or None if missing or tampered.
    """
    try:
        if not _TOKEN_FILE.exists():
            return None
        raw_text = _TOKEN_FILE.read_text(encoding="utf-8").strip()
        if not raw_text:
            return None
            
        # 1. Try raw JWT format
        if raw_text.count(".") == 2:
            parts = raw_text.split(".")
            try:
                payload_b64 = parts[1]
                payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
                payload_bytes = base64.urlsafe_b64decode(payload_b64)
                payload = json.loads(payload_bytes.decode("utf-8"))
                
                # Verify JWT signature offline
                msg = (parts[0] + "." + parts[1]).encode("utf-8")
                sig_b64 = parts[2]
                sig_b64 += "=" * ((4 - len(sig_b64) % 4) % 4)
                sig_bytes = base64.urlsafe_b64decode(sig_b64)
                expected = hmac.new(_HMAC_SECRET, msg, hashlib.sha256).digest()
                signature_invalid = False
                if not hmac.compare_digest(expected, sig_bytes):
                    print("[License] JWT signature mismatch offline. Flagging for online verification.")
                    signature_invalid = True
                    
                # Map JWT claims to client expected keys
                mapped = dict(payload)
                if signature_invalid:
                    mapped["signature_invalid"] = True
                if "hardware_fingerprint" in payload:
                    mapped["fingerprint"] = payload["hardware_fingerprint"]
                if "issued_at" in payload:
                    mapped["last_server_check"] = payload["issued_at"]
                if "last_successful_server_validation" in payload:
                    mapped["last_successful_server_validation"] = payload["last_successful_server_validation"]
                if "last_successful_server_time" in payload:
                    mapped["last_successful_server_time"] = payload["last_successful_server_time"]
                if "last_local_time" in payload:
                    mapped["last_local_time"] = payload["last_local_time"]
                if "stable_fingerprint" in payload:
                    mapped["stable_fingerprint"] = payload["stable_fingerprint"]
                if "stable_hardware_fingerprint" in payload:
                    mapped["stable_fingerprint"] = payload["stable_hardware_fingerprint"]
                if "offline_grace_days" in payload:
                    mapped["offline_grace_days"] = payload["offline_grace_days"]
                if "minimum_version" in payload:
                    mapped["minimum_version"] = payload["minimum_version"]
                if "force_upgrade" in payload:
                    mapped["force_upgrade"] = payload["force_upgrade"]
                if "exp" in payload:
                    mapped["expires"] = payload["exp"]
                # Merge sidecar metadata for mutable fields only.
                # The token's fingerprint claims remain the source of truth so
                # a stale sidecar from a previous install/machine cannot create
                # a false "fingerprint mismatch" on startup.
                meta = _load_license_meta()
                if "last_server_check" in meta:
                    mapped["last_server_check"] = meta["last_server_check"]
                if "last_successful_server_validation" in meta:
                    mapped["last_successful_server_validation"] = meta["last_successful_server_validation"]
                if "last_successful_server_time" in meta:
                    mapped["last_successful_server_time"] = meta["last_successful_server_time"]
                if "last_local_time" in meta:
                    mapped["last_local_time"] = meta["last_local_time"]
                if "stable_fingerprint" in meta and "stable_fingerprint" not in mapped:
                    mapped["stable_fingerprint"] = meta["stable_fingerprint"]
                if "stable_hardware_fingerprint" in meta and "stable_fingerprint" not in mapped:
                    mapped["stable_fingerprint"] = meta["stable_hardware_fingerprint"]
                if "offline_grace_days" in meta:
                    mapped["offline_grace_days"] = meta["offline_grace_days"]
                if "minimum_version" in meta:
                    mapped["minimum_version"] = meta["minimum_version"]
                if "force_upgrade" in meta:
                    mapped["force_upgrade"] = meta["force_upgrade"]
                if "fingerprint" in meta and "fingerprint" not in mapped:
                    mapped["fingerprint"] = meta["fingerprint"]
                if "seat_number" in meta:
                    mapped["seat_number"] = meta["seat_number"]
                return mapped
            except Exception as jwt_err:
                print(f"[License] Failed to parse raw token as JWT: {jwt_err}")
                
        # 2. Try JSON envelope format
        try:
            obj = json.loads(raw_text)
            payload = obj.get("payload")
            sig = obj.get("sig", "")
            if payload and sig:
                payload_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
                expected = _token_hmac(payload_bytes)
                signature_invalid = False
                if not hmac.compare_digest(expected, sig):
                    print("[License] Token HMAC mismatch. Flagging for online verification.")
                    signature_invalid = True
                
                mapped = dict(payload)
                if signature_invalid:
                    mapped["signature_invalid"] = True
                return mapped
        except Exception as json_err:
            print(f"[License] Failed to parse as JSON envelope: {json_err}")
            
        return None
    except Exception as e:
        print(f"[License] Token read failed: {e}")
        return None


# ── Sidecar metadata helpers ──────────────────────────────────────────────────

def _save_license_meta(meta: dict) -> None:
    """
    Save mutable license metadata (last_server_check, fingerprint, seat_number)
    to a sidecar JSON file WITHOUT touching the server-issued JWT in cardiox.lic.

    This prevents the client from re-signing the server's JWT with the client
    HMAC secret, which previously caused subsequent server heartbeats to fail
    (server couldn't verify a token it did not sign).
    """
    try:
        _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
        existing: dict = {}
        try:
            if _META_FILE.exists():
                existing = json.loads(_META_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
        existing.update(meta)
        _META_FILE.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[License] Could not save metadata sidecar: {e}")


def _load_license_meta() -> dict:
    """Load mutable license metadata from the sidecar JSON file."""
    try:
        if _META_FILE.exists():
            return json.loads(_META_FILE.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}


def delete_token_file() -> None:
    """Remove the token file and sidecar metadata (e.g. after deactivation or factory reset)."""
    try:
        if _TOKEN_FILE.exists():
            _TOKEN_FILE.unlink()
    except Exception as e:
        print(f"[License] Could not delete token file: {e}")
    try:
        _META_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def token_file_exists() -> bool:
    """Check 1: does the token file exist?"""
    return _TOKEN_FILE.exists()


# ══════════════════════════════════════════════════════════════════════════════
# Server Communication
# ══════════════════════════════════════════════════════════════════════════════

def _post_json(endpoint: str, body: Dict, timeout: int = 5) -> Dict:
    """Send a signed JSON POST to the license server."""
    import urllib.request
    import urllib.error

    url = f"{LICENSE_SERVER_URL.rstrip('/')}/{endpoint.lstrip('/')}"
    payload_bytes = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req_sig = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()

    req = urllib.request.Request(
        url,
        data=payload_bytes,
        headers={
            "Content-Type": "application/json",
            "X-Request-Sig": req_sig,
            "X-Product": PRODUCT_CODE,
            "X-Version": SOFTWARE_VERSION,
        },
        method="POST",
    )
    if LICENSE_API_TOKEN:
        req.add_header("X-API-Key", LICENSE_API_TOKEN)
        req.add_header("Authorization", f"Bearer {LICENSE_API_TOKEN}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:  # type: ignore
        try:
            payload = json.loads(e.read().decode())
        except Exception:
            payload = {"valid": False, "error": f"HTTP {e.code}: {e.reason}"}
        if e.code in (408, 500, 502, 503, 504):
            payload["offline"] = True
        return payload
    except Exception as e:
        return {"valid": False, "error": str(e), "offline": True}


def _verify_server_sig(response: Dict) -> bool:
    """Verify the server's HMAC signature on its response."""
    resp = dict(response)
    sig = resp.pop("server_sig", None)
    if not sig:
        return True  # Legacy / no-sig servers: pass through
    payload_bytes = json.dumps(resp, sort_keys=True).encode("utf-8")
    expected = hmac.new(_HMAC_SECRET, payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sig)


def register_device(
    license_key: str,
    full_name: str,
    doctor_name: str,
    org_name: str,
    org_address: str,
    phone: str,
    password_hash: str,
    machine_serial_id: str = "",
) -> Dict:
    """
    First-time registration: POST /register.
    Server validates key pool -> fingerprint uniqueness -> RhythmUltra uniqueness
    -> creates seat -> returns signed token data.
    """
    fingerprint = get_hardware_fingerprint()
    RhythmUltra_serial = get_RhythmUltra_serial() or ""
    machine_ctx = get_machine_context()
    machine_serial = machine_serial_id.strip() or machine_ctx["machine_serial_id"]

    body = {
        "license_key": license_key.strip().upper(),
        "hardware_fingerprint": fingerprint,
        "stable_hardware_fingerprint": _stable_hardware_fingerprint(),
        "RhythmUltra_serial": RhythmUltra_serial,
        "rhythmultra_serial": RhythmUltra_serial,
        "rhythmulta_serial": RhythmUltra_serial,
        "full_name": full_name,
        "doctor_name": doctor_name,
        "org_name": org_name,
        "org_address": org_address,
        "phone": phone,
        "password": password_hash,
        "password_hash": password_hash,
        "bios_serial": machine_serial,
        "machine_serial_id": machine_serial,
        "pc_name": machine_ctx["machine_name"],
        "windows_version": machine_ctx["windows_version"],
        "app_version": SOFTWARE_VERSION,
    }
    # ── DEBUG: Print request body to diagnose "Missing required fields" from AWS ──
    print("=" * 60)
    print("REGISTER BODY (DEBUG - sent to server):")
    print(json.dumps(body, indent=2, default=str))
    print(f"  >> URL will be: {LICENSE_SERVER_URL.rstrip('/')}/register")
    print("=" * 60)
    try:
        import pathlib as _pl2
        _body_file = _pl2.Path(__file__).parents[2] / "register_debug.txt"
        _existing = _body_file.read_text(encoding="utf-8") if _body_file.exists() else ""
        _body_file.write_text(
            _existing + "\n\nREGISTER BODY:\n" + json.dumps(body, indent=2, default=str),
            encoding="utf-8"
        )
    except Exception:
        pass

    result = _post_json("register", body)

    # ── DEBUG: Print full server response to diagnose missing cardiox.lic ──────
    import pprint as _pprint
    _debug_lines = [
        "=" * 60,
        "REGISTER RESPONSE (DEBUG):",
        _pprint.pformat(result),
        f"  >> 'token' key present: {'token' in result}",
        f"  >> token value: {result.get('token')}",
        f"  >> valid/success/authorized: {result.get('valid')} / {result.get('success')} / {result.get('authorized')}",
        "=" * 60,
    ]
    for _line in _debug_lines:
        print(_line)
    # Also write to file so GUI users can read it
    try:
        import pathlib as _pl
        _dbg_file = _pl.Path(__file__).parents[2] / "register_debug.txt"
        _dbg_file.write_text("\n".join(_debug_lines), encoding="utf-8")
        print(f"[License][DEBUG] Response also written to: {_dbg_file}")
    except Exception as _de:
        print(f"[License][DEBUG] Could not write debug file: {_de}")

    _verify_server_sig(result)

    if result.get("valid") or result.get("success") or result.get("authorized"):
        try:
            from datetime import datetime
            profile_path = _TOKEN_DIR / "registration_profile.json"
            _TOKEN_DIR.mkdir(parents=True, exist_ok=True)
            
            seat_number = result.get("seat_number") or result.get("seat") or 1
            license_id = result.get("license_id") or result.get("id") or ""
            
            token_str = result.get("token")
            if token_str:
                try:
                    if isinstance(token_str, dict):
                        payload = token_str.get("payload", token_str)
                    elif isinstance(token_str, str) and token_str.count(".") == 2:
                        parts = token_str.split(".")
                        payload_b64 = parts[1]
                        payload_b64 += "=" * ((4 - len(payload_b64) % 4) % 4)
                        payload_bytes = base64.urlsafe_b64decode(payload_b64)
                        payload = json.loads(payload_bytes.decode("utf-8"))
                    else:
                        payload = {}
                    
                    if payload:
                        if not license_id:
                            license_id = payload.get("license_id") or payload.get("id") or ""
                        if seat_number == 1:
                            seat_number = payload.get("seat_number") or payload.get("seat") or 1
                except Exception as pe:
                    print(f"[License] Could not extract from token: {pe}")
            
            profile = {
                "doctor_name": doctor_name,
                "hospital_name": org_name,
                "hospital_address": org_address,
                "phone": phone,
                "license_id": license_id,
                "seat_number": seat_number,
                "rhythmultra_serial": RhythmUltra_serial,
                "machine_serial_id": machine_serial,
                "registered_at": datetime.utcnow().isoformat()
            }
            with open(profile_path, "w", encoding="utf-8") as f:
                json.dump(profile, f, indent=2)
            print(f"[License] Saved registration profile to {profile_path}")
        except Exception as e:
            print(f"[License] Failed to save registration profile: {e}")

    return result


def load_registration_profile() -> Dict:
    """Load registration profile from %APPDATA%\\Deckmount\\registration_profile.json."""
    profile_path = _TOKEN_DIR / "registration_profile.json"
    try:
        if profile_path.exists():
            with open(profile_path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"[License] Error loading registration profile: {e}")
    return {}



def load_raw_token() -> str:
    """Read the raw token contents from cardiox.lic."""
    try:
        if _TOKEN_FILE.exists():
            return _TOKEN_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def heartbeat(token_data: Dict) -> Dict:
    """
    POST /heartbeat — startup/server check-in.
    Sends current token claims; server confirms license not revoked.
    Returns server response dict (includes `valid`, `revoked`, etc.).
    """
    machine_ctx = get_machine_context()
    current_fp = get_hardware_fingerprint()
    current_stable_fp = _stable_hardware_fingerprint()

    # Tolerant hardware checking: if core machine matches, reuse registered fingerprint
    # to prevent mismatch errors on server side due to dynamic MAC/Disk changes.
    raw_token = load_token_file()
    if raw_token:
        stored_fp = raw_token.get("fingerprint", "")
        meta = _load_license_meta()
        stored_stable_fp = raw_token.get("stable_fingerprint", meta.get("stable_fingerprint", ""))
        if stored_fp and stored_stable_fp:
            if hmac.compare_digest(current_stable_fp, stored_stable_fp):
                current_fp = stored_fp
                print("[License] Stable hardware match. Using registered fingerprint for heartbeat check.")

    current_RhythmUltra_serial = get_rhythmultra_serial() or token_data.get(
        "rhythmultra_serial", token_data.get("RhythmUltra_serial", token_data.get("rhythmulta_serial", ""))
    )
    body = {
        "token": load_raw_token(),
        "hardware_fingerprint": current_fp,
        "stable_hardware_fingerprint": _stable_hardware_fingerprint(),
        "RhythmUltra_serial": current_RhythmUltra_serial,
        "rhythmultra_serial": current_RhythmUltra_serial,
        "rhythmulta_serial": current_RhythmUltra_serial,
        "machine_serial_id": machine_ctx.get("machine_serial_id", ""),
        "pc_name": machine_ctx.get("machine_name", ""),
        "license_key": token_data.get("license_key", ""),
        "seat_number": token_data.get("seat_number", 1),
        "version": SOFTWARE_VERSION,
    }
    print(f"[License][DEBUG] heartbeat request body: {body}")
    result = _post_json("heartbeat", body, timeout=3)
    print(f"[License][DEBUG] heartbeat response: {result}")
    _verify_server_sig(result)
    return result


# Legacy wrappers (kept for backward compat with existing callers)

def validate_with_server(license_key: str, fingerprint: str) -> Dict:
    """Legacy: validate key+fingerprint. Delegates to heartbeat path if token exists."""
    machine_ctx = get_machine_context()
    token = load_token_file()
    if token:
        return heartbeat(token)
    body = {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
        "machine_serial_id": machine_ctx.get("machine_serial_id", ""),
        "pc_name": machine_ctx.get("machine_name", ""),
    }
    result = _post_json("validate", body)
    _verify_server_sig(result)
    return result


def activate_with_server(license_key: str, fingerprint: str, machine_name: str = "") -> Dict:
    """Legacy: kept for compatibility. Calls /activate on old-schema servers."""
    machine_ctx = get_machine_context()
    RhythmUltra_serial = get_rhythmultra_serial() or ""
    body = {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
        "stable_hardware_fingerprint": _stable_hardware_fingerprint(),
        "machine_name": machine_name or machine_ctx["machine_name"],
        "machine_os": machine_ctx["machine_os"],
        "machine_host": machine_ctx["machine_name"],
        "machine_serial_id": machine_ctx.get("machine_serial_id", ""),
        "pc_name": machine_ctx.get("machine_name", ""),
        "windows_version": machine_ctx.get("windows_version", ""),
        "RhythmUltra_serial": RhythmUltra_serial,
        "rhythmultra_serial": RhythmUltra_serial,
        "rhythmulta_serial": RhythmUltra_serial,
    }
    result = _post_json("activate", body)
    _verify_server_sig(result)
    return result


def restore_license_from_server(license_key: Optional[str] = None) -> Dict:
    """
    Attempt to rebuild a missing token or sidecar metadata from the server.

    This is used when the local token files were deleted or the metadata sidecar
    is missing, but the user still has a known license key and the server is
    reachable.
    """
    now = _current_unix_time()
    key = (license_key or load_stored_key() or "").strip().upper()
    if not key:
        return {
            "valid": False,
            "error": "No stored license key available for recovery.",
        }

    fingerprint = get_hardware_fingerprint()
    machine_ctx = get_machine_context()
    RhythmUltra_serial = get_rhythmultra_serial() or ""
    result = activate_with_server(key, fingerprint, machine_ctx.get("machine_name", ""))

    error_code = str(result.get("error", "")).strip().upper()
    if error_code == "DEVICE_ALREADY_REGISTERED":
        _append_audit_event(
            "DUPLICATE_ACTIVATION_ATTEMPT",
            license_key=key,
            RhythmUltra_serial=RhythmUltra_serial,
            machine_serial_id=machine_ctx.get("machine_serial_id", ""),
        )
        return result

    if result.get("valid") or result.get("authorized"):
        token_payload = result.get("token")
        if isinstance(token_payload, str) and token_payload.strip():
            save_token_file(token_payload)
        elif isinstance(token_payload, dict):
            save_token_file(token_payload.get("payload", token_payload))
        else:
            remember_valid_license(key, fingerprint, result)

        seat_number = result.get("seat_number", 1)
        last_server_time = int(
            result.get("last_successful_server_time")
            or result.get("last_server_check")
            or now
        )
        meta = {
            "fingerprint": fingerprint,
            "stable_fingerprint": _stable_hardware_fingerprint(),
            "last_server_check": last_server_time,
            "last_successful_server_validation": last_server_time,
            "last_successful_server_time": last_server_time,
            "last_local_time": now,
            "seat_number": seat_number,
        }
        _save_license_meta(meta)
        _append_audit_event(
            "LICENSE_RECOVERED",
            license_key=key,
            seat_number=seat_number,
            machine_serial_id=machine_ctx.get("machine_serial_id", ""),
            RhythmUltra_serial=RhythmUltra_serial,
        )
        result = dict(result)
        result["recovered"] = True
        return result

    if result.get("offline"):
        result = dict(result)
        result["recovered"] = False
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 5-Step Startup Validation
# ══════════════════════════════════════════════════════════════════════════════

class StartupCheckResult:
    """Result container for the 5-step startup validation."""

    def __init__(self):
        self.ok: bool = False
        self.step_failed: int = 0        # 1-5, 0 = all passed
        self.reason: str = ""
        self.error_code: str = ""
        self.token: Optional[Dict] = None
        self.RhythmUltra_serial: Optional[str] = None
        self.offline_mode: bool = False
        self.days_remaining: Optional[int] = None

    def __bool__(self):
        return self.ok

    def __repr__(self):
        return (
            f"StartupCheckResult(ok={self.ok}, step={self.step_failed}, "
            f"error_code={self.error_code!r}, reason={self.reason!r})"
        )


# Machine-readable revocation codes returned by the license Lambdas.
# cardiox-license-heartbeat returns LICENSE_REVOKED when licenses.status is
# 'revoked' (what cardiox-license-admin-revoke sets) and ACCOUNT_REVOKED when
# the individual seat has been revoked by an administrator. Both mean the same
# thing to the client: wipe the license and make the user register again.
_REVOCATION_CODES = frozenset({"LICENSE_REVOKED", "ACCOUNT_REVOKED"})

# Shown verbatim whenever the license is revoked, regardless of which code or
# wording the server sent. The server's own text varies ("License revoked.
# Please contact Deckmount." vs "Account revoked by administrator.") and the
# latter never tells the user what to do, so the client always states the one
# instruction that matters: contact Deckmount support.
REVOKED_MESSAGE: str = (
    "License is revoked.\n\n"
    "This software has been deactivated on this device.\n"
    "Please contact Deckmount support."
)


def _is_explicit_revocation(payload: Dict) -> bool:
    """Return True only when the server explicitly says the license is revoked."""
    if not isinstance(payload, dict):
        return False
    # The license Lambdas put the machine-readable code in `error`
    # (e.g. {"error": "LICENSE_REVOKED", "message": "License revoked. ..."}),
    # while some endpoints use `error_code`. Check both as codes first so
    # detection does not depend on the server's human-readable wording.
    for field in ("error_code", "error"):
        if str(payload.get(field, "")).strip().upper() in _REVOCATION_CODES:
            return True
    if bool(payload.get("revoked")):
        return True
    # Prose fallback, kept for older server builds that only send a message.
    message = str(payload.get("message", "")).strip().lower()
    error = str(payload.get("error", "")).strip().lower()
    return any(
        phrase in text
        for text in (message, error)
        for phrase in ("license revoked", "account revoked")
    )


def run_startup_checks(force_heartbeat: bool = False) -> StartupCheckResult:
    """
    Execute the 5-step startup validation sequence.

    Check 1: Token file exists on disk.
    Check 2: Token HMAC signature valid (tamper detection).
    Check 3: Hardware fingerprint in token matches this machine.
    Check 4: RhythmUltra connected and serial matches token.
    Check 5: Server heartbeat (attempted on startup; offline grace applies if unreachable).

    Returns a StartupCheckResult.  If .ok is False, .step_failed and
    .reason describe the first failing check.
    """
    global _restricted_mode
    res = StartupCheckResult()
    now = _current_unix_time()
    debug_enabled = os.getenv("CARDIOX_LICENSE_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    token_missing = not token_file_exists()
    meta_missing = not _license_meta_exists()

    if token_missing or meta_missing:
        stored_key = load_stored_key()
        if stored_key:
            recovery_result = restore_license_from_server(stored_key)
            if debug_enabled:
                print(f"[License][DEBUG] recovery={recovery_result}")
            if str(recovery_result.get("error", "")).strip().upper() == "DEVICE_ALREADY_REGISTERED":
                res.step_failed = 1
                res.error_code = "DEVICE_ALREADY_REGISTERED"
                res.reason = (
                    "This RhythmUltra device has reached the maximum limit of 5 registrations.\n\n"
                    "Please deactivate an existing installation or contact Deckmount Support."
                )
                return res
            if recovery_result.get("valid") or recovery_result.get("authorized"):
                token_missing = not token_file_exists()
                meta_missing = not _license_meta_exists()
        if token_missing:
            res.step_failed = 1
            res.error_code = "LICENSE_NOT_FOUND"
            res.reason = "License not found.\nPlease register your device."
            return res

    # ── Check 2: Token HMAC valid (not tampered) ──────────────────────────────
    token = load_token_file()
    if token is None:
        res.step_failed = 2 if not token_missing else 1
        res.error_code = "TOKEN_INTEGRITY_FAILED" if not token_missing else "LICENSE_NOT_FOUND"
        res.reason = "License data integrity check failed." if not token_missing else "License not found.\nPlease register your device."
        return res
    res.token = token

    meta = _load_license_meta()
    current_fp = get_hardware_fingerprint()
    current_stable_fp = _stable_hardware_fingerprint()
    stored_fp = token.get("fingerprint", "")
    stored_stable_fp = token.get("stable_fingerprint", meta.get("stable_fingerprint", ""))

    if debug_enabled:
        current_fields = _collect_wmi_fields()
        present_fields = sum(1 for value in current_fields.values() if value)
        print(
            "[License][DEBUG] check3 "
            f"current_fp={current_fp} stored_fp={stored_fp} "
            f"stable_fp={current_stable_fp} stored_stable_fp={stored_stable_fp} "
            f"fields={current_fields} present_fields={present_fields} "
            f"token_seat={token.get('seat_number', '')}"
        )
    else:
        current_fields = _collect_wmi_fields()
        present_fields = sum(1 for value in current_fields.values() if value)

    last_local_time = int(
        token.get("last_local_time")
        or meta.get("last_local_time")
        or 0
    )
    last_server_time = int(
        token.get("last_successful_server_time")
        or token.get("last_successful_server_validation")
        or meta.get("last_successful_server_time")
        or meta.get("last_successful_server_validation")
        or token.get("last_server_check", 0)
        or meta.get("last_server_check", 0)
        or 0
    )
    if last_local_time and now < last_local_time:
        _append_audit_event(
            "CLOCK_ROLLBACK_DETECTED",
            current_local_time=now,
            last_local_time=last_local_time,
            last_successful_server_time=last_server_time,
            license_key=token.get("license_key", ""),
        )
        res.step_failed = 5
        res.error_code = "CLOCK_ROLLBACK_DETECTED"
        res.reason = (
            "System clock manipulation detected.\n\n"
            "Please correct your system date and time."
        )
        return res

    if token.get("force_upgrade") or _is_minimum_version_forced(str(token.get("minimum_version", ""))):
        minimum_version = str(token.get("minimum_version", "")).strip() or SOFTWARE_VERSION
        _append_audit_event(
            "FORCE_UPGRADE_ENFORCED",
            minimum_version=minimum_version,
            running_version=SOFTWARE_VERSION,
            license_key=token.get("license_key", ""),
        )
        res.step_failed = 5
        res.error_code = "SOFTWARE_UPDATE_REQUIRED"
        res.reason = "A software update is required before continuing."
        return res

    # ── Check 3: Hardware fingerprint matches ─────────────────────────────────
    fingerprint_matches = bool(stored_fp) and hmac.compare_digest(current_fp, stored_fp)
    stable_matches = bool(stored_stable_fp) and hmac.compare_digest(current_stable_fp, stored_stable_fp)

    if stored_fp and not fingerprint_matches and not stable_matches:
        # Ask the server first so revocation and token problems do not get
        # misreported as a local machine mismatch.
        try:
            hb_result = heartbeat(token)
            if debug_enabled:
                print(f"[License][DEBUG] heartbeat={hb_result}")
            
            if hb_result.get("offline"):
                _restricted_mode = True
                res.ok = True
                res.restricted_mode = True
                res.offline_mode = True
                res.reason = (
                    "Hardware change detected.\n\n"
                    "Internet connection required to reverify license.\n\n"
                    "ECG acquisition disabled until verification completes."
                )
                return res

            hb_message = str(hb_result.get("message", "")).lower()
            hb_error = str(hb_result.get("error", "")).lower()
            server_token_failure = any(
                marker in hb_message or marker in hb_error
                for marker in (
                    "invalid or expired token",
                    "invalid token",
                    "expired token",
                    "token expired",
                    "token invalid",
                )
            )
            if _is_explicit_revocation(hb_result):
                res.step_failed = 5
                res.error_code = "LICENSE_REVOKED"
                res.reason = REVOKED_MESSAGE
                return res

            if server_token_failure:
                res.step_failed = 2
                res.error_code = str(hb_result.get("error_code", "")).strip().upper() or "TOKEN_INVALID"
                res.reason = hb_result.get(
                    "error",
                    hb_result.get(
                        "message",
                        "License data integrity check failed.",
                    ),
                )
                return res

            if hb_result.get("force_upgrade") or _is_minimum_version_forced(str(hb_result.get("minimum_version", ""))):
                minimum_version = str(hb_result.get("minimum_version", "")).strip() or SOFTWARE_VERSION
                _append_audit_event(
                    "FORCE_UPGRADE_ENFORCED",
                    minimum_version=minimum_version,
                    running_version=SOFTWARE_VERSION,
                    license_key=token.get("license_key", ""),
                )
                res.step_failed = 5
                res.error_code = "SOFTWARE_UPDATE_REQUIRED"
                res.reason = "A software update is required before continuing."
                return res

            if hb_result.get("valid", False) or hb_result.get("authorized", False):
                last_server_value = int(
                    hb_result.get("last_successful_server_time")
                    or hb_result.get("last_server_check")
                    or now
                )
                meta_update = {
                    "fingerprint": current_fp,
                    "stable_fingerprint": current_stable_fp,
                    "last_server_check": last_server_value,
                    "last_successful_server_validation": last_server_value,
                    "last_successful_server_time": last_server_value,
                    "last_local_time": now,
                }
                if "seat_number" in hb_result:
                    meta_update["seat_number"] = hb_result["seat_number"]
                    token["seat_number"] = hb_result["seat_number"]
                _save_license_meta(meta_update)
                token["fingerprint"] = current_fp
                token["stable_fingerprint"] = current_stable_fp
                token["last_server_check"] = last_server_value
                token["last_successful_server_validation"] = last_server_value
                token["last_successful_server_time"] = last_server_value
                token["last_local_time"] = now
                res.token = token
            elif present_fields <= 2:
                # When WMI is degraded, fall back to the local machine identity
                # instead of hard-failing the license.
                _save_license_meta(
                    {
                        "fingerprint": current_fp,
                        "stable_fingerprint": current_stable_fp,
                        "last_local_time": now,
                    }
                )
                token["fingerprint"] = current_fp
                token["stable_fingerprint"] = current_stable_fp
                token["last_local_time"] = now
                print(
                    "[License] Hardware fingerprint source degraded; "
                    "accepting local machine identity fallback."
                )
            else:
                res.step_failed = 3
                res.error_code = "MACHINE_MISMATCH"
                res.reason = (
                    "This license is registered to a different machine.\n"
                    "Contact Deckmount support if your hardware has changed."
                )
                return res
        except Exception:
            if present_fields <= 2:
                _save_license_meta(
                    {
                        "fingerprint": current_fp,
                        "stable_fingerprint": current_stable_fp,
                        "last_local_time": now,
                    }
                )
                token["fingerprint"] = current_fp
                token["stable_fingerprint"] = current_stable_fp
                token["last_local_time"] = now
                print(
                    "[License] Hardware fingerprint source degraded; "
                    "accepting local machine identity fallback."
                )
            else:
                _restricted_mode = True
                res.ok = True
                res.restricted_mode = True
                res.offline_mode = True
                res.reason = (
                    "Hardware change detected.\n\n"
                    "Internet connection required to reverify license.\n\n"
                    "ECG acquisition disabled until verification completes."
                )
                return res
    elif stable_matches and not fingerprint_matches:
        # Exact fingerprint changed, but the core motherboard / BIOS / CPU
        # identity still matches.  This tolerates minor hardware upgrades.
        _save_license_meta(
            {
                "fingerprint": current_fp,
                "stable_fingerprint": current_stable_fp,
                "last_local_time": now,
            }
        )
        token["fingerprint"] = current_fp
        token["stable_fingerprint"] = current_stable_fp
        token["last_local_time"] = now

    # ── Check 4: RhythmUltra connected (informational only during startup) ──
    usb_serial = get_rhythmultra_serial()
    res.RhythmUltra_serial = usb_serial
    stored_serial = token.get("rhythmultra_serial", token.get("RhythmUltra_serial", token.get("rhythmulta_serial", "")))
    if usb_serial and stored_serial and not hmac.compare_digest(usb_serial, stored_serial):
        _append_audit_event(
            "DUPLICATE_ACTIVATION_ATTEMPT",
            license_key=token.get("license_key", ""),
            RhythmUltra_serial=usb_serial,
            machine_serial_id=token.get("machine_serial_id", ""),
        )

    # ── Check 5: Server heartbeat ─────────────────────────────────────────────
    hb_result = heartbeat(token)
    if hb_result.get("offline"):
        # NOTE: there is deliberately no "signature_invalid -> token tampered" branch
        # here any more. Tokens are signed by the server with JWT_SECRET, while the
        # client only holds LICENSE_HMAC_SECRET, so the offline signature comparison
        # fails for every genuine token. Treating that as tampering made step 2 fire
        # on the first offline launch, which wiped the licence (main.py clears
        # credentials for steps 1 and 2) and demanded a re-registration the user could
        # not perform without internet. The offline grace window below is what governs
        # offline use; exhausting it must never wipe credentials.
        #
        # Real tamper detection lives on the server heartbeat. To detect it offline as
        # well, move token signing to RS256 and ship only the public key in the app.
        last_check = int(
            token.get("last_successful_server_time")
            or token.get("last_successful_server_validation")
            or token.get("last_server_check", 0)
        )
        elapsed = now - last_check
        grace_remaining = HEARTBEAT_INTERVAL_SECONDS - elapsed
        if grace_remaining <= 0:
            res.step_failed = 5
            res.error_code = "LICENSE_VERIFICATION_REQUIRED"
            res.reason = (
                f"Your {OFFLINE_GRACE_DAYS}-day offline period has ended.\n\n"
                "An internet connection is required to reactivate this license.\n"
                "Please connect to the internet and sign in again."
            )
            return res

        offline_meta = {
            "fingerprint": token.get("fingerprint", current_fp),
            "stable_fingerprint": token.get("stable_fingerprint", current_stable_fp),
            "last_local_time": now,
            "last_server_check": last_check,
            "last_successful_server_validation": token.get("last_successful_server_validation", last_check),
            "last_successful_server_time": token.get("last_successful_server_time", last_check),
        }
        if "seat_number" in token:
            offline_meta["seat_number"] = token["seat_number"]
        _save_license_meta(offline_meta)

        # Round up: with <24h left, truncating would report "0 day(s) remaining"
        # while the app is in fact still usable.
        days_remaining = -(-grace_remaining // 86400)
        print(
            f"[License] Offline — {days_remaining} day(s) of grace remaining."
        )
        res.ok = True
        res.offline_mode = True
        # Surface the countdown to the caller. The UI warns the user once this drops
        # to OFFLINE_WARNING_DAYS or fewer, so a mandatory server verification never
        # arrives as a surprise shutdown.
        res.days_remaining = days_remaining
        if days_remaining <= OFFLINE_WARNING_DAYS:
            res.reason = (
                f"Internet connection required for licence verification.\n\n"
                f"This software has been offline for some time and will stop working in "
                f"{days_remaining} day(s) unless it can verify your licence with the "
                f"Deckmount server.\n\n"
                f"Please connect this computer to the internet."
            )
        else:
            res.reason = (
                f"Offline mode active. {days_remaining} day(s) remaining before "
                "an internet connection is required to reactivate."
            )
        return res

    if _is_explicit_revocation(hb_result):
        res.step_failed = 5
        res.error_code = "LICENSE_REVOKED"
        res.reason = REVOKED_MESSAGE
        return res

    if not hb_result.get("valid", False) and not hb_result.get("authorized", False):
        res.step_failed = 5
        res.error_code = str(hb_result.get("error_code", "")).strip().upper() or "LICENSE_BLOCKED"
        res.reason = hb_result.get(
            "message",
            hb_result.get(
                "error",
                "License verification failed. Please contact Deckmount support.",
            ),
        )
        return res

    if hb_result.get("force_upgrade") or _is_minimum_version_forced(str(hb_result.get("minimum_version", ""))):
        minimum_version = str(hb_result.get("minimum_version", "")).strip() or SOFTWARE_VERSION
        _append_audit_event(
            "FORCE_UPGRADE_ENFORCED",
            minimum_version=minimum_version,
            running_version=SOFTWARE_VERSION,
            license_key=token.get("license_key", ""),
        )
        res.step_failed = 5
        res.error_code = "SOFTWARE_UPDATE_REQUIRED"
        res.reason = "A software update is required before continuing."
        return res

    # Successful heartbeat — update sidecar metadata ONLY.
    # CRITICAL: Do NOT call save_token_file(token) here — that would
    # overwrite the server-issued JWT with a client-signed version.
    server_time = int(
        hb_result.get("last_successful_server_time")
        or hb_result.get("last_server_check")
        or now
    )
    meta = {
        "fingerprint": current_fp,
        "stable_fingerprint": current_stable_fp,
        "last_server_check": server_time,
        "last_successful_server_validation": server_time,
        "last_successful_server_time": server_time,
        "last_local_time": now,
    }
    if "seat_number" in hb_result:
        meta["seat_number"] = hb_result["seat_number"]
        token["seat_number"] = hb_result["seat_number"]
    _save_license_meta(meta)
    token["fingerprint"] = current_fp
    token["stable_fingerprint"] = current_stable_fp
    token["last_server_check"] = server_time
    token["last_successful_server_validation"] = server_time
    token["last_successful_server_time"] = server_time
    token["last_local_time"] = now
    print(f"[License] Heartbeat OK — metadata updated (JWT preserved).")

    # ── All checks passed ─────────────────────────────────────────────────────
    res.ok = True
    res.token = token
    return res


# ══════════════════════════════════════════════════════════════════════════════
# License Key Utilities (kept for backward compat)
# ══════════════════════════════════════════════════════════════════════════════

def _b32_encode(data: bytes) -> str:
    result, acc, bits = [], 0, 0
    for byte in data:
        acc = (acc << 8) | byte
        bits += 8
        while bits >= 5:
            bits -= 5
            result.append(_B32_ALPHA[(acc >> bits) & 0x1F])
    if bits > 0:
        result.append(_B32_ALPHA[(acc << (5 - bits)) & 0x1F])
    return "".join(result)


def _b32_decode(s: str) -> bytes:
    s = s.upper().replace("-", "").replace(" ", "")
    acc, bits, result = 0, 0, []
    for char in s:
        idx = _B32_ALPHA.find(char)
        if idx < 0:
            raise ValueError(f"Invalid character in license key: {char!r}")
        acc = (acc << 5) | idx
        bits += 5
        if bits >= 8:
            bits -= 8
            result.append((acc >> bits) & 0xFF)
    return bytes(result)


def format_key(raw_key: str) -> str:
    raw_key = raw_key.upper().replace("-", "").replace(" ", "")
    if raw_key.startswith("CRDX") or len(raw_key) == 12:
        parts = []
        if len(raw_key) > 0:
            parts.append(raw_key[0:4])
        if len(raw_key) > 4:
            parts.append(raw_key[4:8])
        if len(raw_key) > 8:
            parts.append(raw_key[8:12])
        return "-".join(parts)
    return "-".join(raw_key[i:i+5] for i in range(0, len(raw_key), 5))


def parse_key_metadata(license_key: str) -> Optional[Dict]:
    """Decode license key without contacting the server."""
    try:
        raw = license_key.upper().replace("-", "").replace(" ", "")
        if len(raw) != 20:
            return None
        data = _b32_decode(raw)
        if len(data) < 12:
            return None
        tier = data[0]
        expiry = struct.unpack(">I", data[1:5])[0]
        nonce = data[5:9]
        return {"tier": tier, "expiry": expiry, "nonce": nonce.hex()}
    except Exception:
        return None


def parse_key_payload(license_key: str) -> Optional[Dict]:
    """Decode and verify key checksum locally."""
    try:
        raw = license_key.upper().replace("-", "").replace(" ", "")
        if len(raw) != 20:
            return None
        data = _b32_decode(raw)
        if len(data) < 12:
            return None
        tier = data[0]
        expiry = struct.unpack(">I", data[1:5])[0]
        nonce = data[5:9]
        checksum = data[9:12]
        payload = data[:9]
        expected_cs = hmac.new(_HMAC_SECRET, payload, hashlib.sha256).digest()[:3]
        if not hmac.compare_digest(checksum, expected_cs):
            return None
        return {"tier": tier, "expiry": expiry, "nonce": nonce.hex()}
    except Exception:
        return None


def is_key_expired_locally(license_key: str) -> bool:
    payload = parse_key_metadata(license_key)
    if payload is None:
        return True
    expiry = payload["expiry"]
    if expiry == 0:
        return False
    return int(time.time()) > expiry


# ══════════════════════════════════════════════════════════════════════════════
# Legacy Storage Helpers (backward compat for license_dialog.py)
# ══════════════════════════════════════════════════════════════════════════════

_LICENSE_KEY_FILE: Path = _LEGACY_CACHE_DIR / "license.key"


def load_stored_key() -> str:
    """Return the license key stored in the token, or legacy key file."""
    token = load_token_file()
    if token:
        return token.get("license_key", "")
    # Legacy fallback
    try:
        if _LICENSE_KEY_FILE.exists():
            return _LICENSE_KEY_FILE.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    return ""


def save_stored_key(license_key: str) -> None:
    """Persist license key to legacy key file (for backward compat)."""
    try:
        _LEGACY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _LICENSE_KEY_FILE.write_text(license_key.strip(), encoding="utf-8")
    except Exception as e:
        print(f"[License] Could not save key: {e}")


def clear_stored_key() -> None:
    try:
        _LICENSE_KEY_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def clear_license_cache() -> None:
    """Remove legacy cache + new token file + sidecar metadata."""
    try:
        _LEGACY_CACHE_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    try:
        _META_FILE.unlink(missing_ok=True)
    except Exception:
        pass
    delete_token_file()


def remember_valid_license(
    license_key: str,
    fingerprint: str,
    result: Optional[Dict] = None,
) -> None:
    """
    Legacy helper: persist a known-valid activation as a token.
    Used by the old LicenseDialog after successful /activate.
    """
    now = int(time.time())
    token_data = {
        "license_key": license_key.strip().upper(),
        "fingerprint": fingerprint,
        "stable_fingerprint": _stable_hardware_fingerprint(),
        "RhythmUltra_serial": "",   # populated on first real startup check
        "rhythmultra_serial": "",   # populated on first real startup check
        "seat_number": 1,
        "last_server_check": now,
        "last_successful_server_validation": now,
        "last_successful_server_time": now,
        "last_local_time": now,
    }
    if isinstance(result, dict):
        for k in ("tier", "expires", "tier_name"):
            if k in result:
                token_data[k] = result[k]
    save_token_file(token_data)
    # Also write legacy key file for older code paths
    save_stored_key(license_key)


def check_license(license_key: str, force_server: bool = False) -> Dict:
    """
    Legacy-compat: validate a license key.
    In the new architecture this delegates to run_startup_checks().
    Used by the periodic re-validation timer in main.py.
    """
    token = load_token_file()
    if token is None:
        return {
            "valid": False,
            "source": "token",
            "message": "No activation token found.",
            "tier": 0,
            "expires": 0,
        }

    result = run_startup_checks(force_heartbeat=force_server)
    if result.ok:
        return {
            "valid": True,
            "source": "token",
            "message": "License valid.",
            "tier": token.get("tier", 0),
            "expires": token.get("expires", 0),
            "revoked": False,
            "error_code": "",
            "offline": bool(getattr(result, "offline_mode", False)),
        }

    revoked = result.step_failed == 5 and result.error_code == "LICENSE_REVOKED"
    return {
        "valid": False,
        "revoked": revoked,
        "source": "token",
        "message": result.reason,
        "error_code": result.error_code,
        "tier": 0,
        "expires": 0,
        "step_failed": result.step_failed,
        "offline": bool(getattr(result, "offline_mode", False)),
    }


def deactivate(license_key: str) -> bool:
    """Deactivate this machine — clears token and contacts server."""
    token = load_token_file()
    fingerprint = get_hardware_fingerprint()
    if token:
        stored_fp = token.get("fingerprint", "")
        meta = _load_license_meta()
        stored_stable_fp = token.get("stable_fingerprint", meta.get("stable_fingerprint", ""))
        current_stable_fp = _stable_hardware_fingerprint()
        if stored_fp and stored_stable_fp and hmac.compare_digest(current_stable_fp, stored_stable_fp):
            fingerprint = stored_fp
            print("[License] Stable hardware match. Using registered fingerprint for deactivation check.")

    body = {
        "license_key": license_key,
        "hardware_fingerprint": fingerprint,
        "RhythmUltra_serial": (token or {}).get("rhythmultra_serial", (token or {}).get("RhythmUltra_serial", (token or {}).get("rhythmulta_serial", ""))),
        "rhythmultra_serial": (token or {}).get("rhythmultra_serial", (token or {}).get("RhythmUltra_serial", (token or {}).get("rhythmulta_serial", ""))),
        "rhythmulta_serial": (token or {}).get("rhythmultra_serial", (token or {}).get("RhythmUltra_serial", (token or {}).get("rhythmulta_serial", ""))),
        "seat_number": (token or {}).get("seat_number", 1),
    }
    result = _post_json("deactivate", body)
    clear_license_cache()
    clear_stored_key()
    return bool(result.get("success"))


# ── Tier helpers ──────────────────────────────────────────────────────────────

TIER_NAMES = {0: "Trial", 1: "Standard", 2: "Professional", 3: "Enterprise"}


def tier_name(tier: int) -> str:
    return TIER_NAMES.get(tier, "Unknown")


def verify_authorized_device(parent=None) -> bool:
    """
    Verify if the current setup and connected device are authorized for acquisition.
    Shows QMessageBox and returns False if blocked, returns True if authorized.
    """
    from PyQt5.QtWidgets import QMessageBox

    # 1. Check if license is in restricted mode (fingerprint mismatch offline)
    if is_license_restricted():
        QMessageBox.critical(
            parent,
            "Reverification Required",
            "<b>Hardware change detected.</b><br><br>"
            "An internet connection is required to reverify your license before acquiring ECG data."
        )
        return False

    # 2. Check RhythmUltra device lock
    enforce_device_lock = _enforce_rhythmultra_lock()
    if not enforce_device_lock:
        return True

    usb_serial = None
    if parent is not None:
        # Check if parent is running in demo mode
        if hasattr(parent, 'serial_reader') and parent.serial_reader is not None:
            if parent.serial_reader.__class__.__name__ == 'DemoSerialReader':
                print("[License] Demo mode detected via serial reader. Bypassing serial check.")
                return True
        
        # Try to retrieve serial from parent's serial_reader
        if hasattr(parent, 'serial_reader') and parent.serial_reader is not None:
            if hasattr(parent.serial_reader, 'device_serial_number') and parent.serial_reader.device_serial_number:
                usb_serial = parent.serial_reader.device_serial_number
                print(f"[License] Retrieved serial from parent serial_reader: {usb_serial}")
        
        # Try to retrieve from settings_manager
        if not usb_serial and hasattr(parent, 'settings_manager') and parent.settings_manager is not None:
            try:
                usb_serial = parent.settings_manager.get_setting("machine_serial_number")
                if usb_serial:
                    print(f"[License] Retrieved serial from parent settings_manager: {usb_serial}")
            except Exception:
                pass

    if not usb_serial:
        # Fallback to scanning ports
        usb_serial = get_rhythmultra_serial()

    token = load_token_file()
    if not token:
        QMessageBox.critical(
            parent,
            "License Missing",
            "<b>License not found.</b><br><br>"
            "Please register or activate your license before acquiring ECG data."
        )
        return False

    stored_serial = token.get("rhythmultra_serial", token.get("RhythmUltra_serial", token.get("rhythmulta_serial", "")))

    # Clean up and normalize serials before comparison
    stored_serial_clean = str(stored_serial).strip()
    usb_serial_clean = str(usb_serial or "").strip()

    # Check if mismatch or missing
    is_mismatched = stored_serial_clean and (not usb_serial_clean or usb_serial_clean.lower() != stored_serial_clean.lower())
    is_missing = not usb_serial_clean

    if is_mismatched or is_missing:
        QMessageBox.critical(
            parent,
            "Unauthorized RhythmUltra Device",
            "<b>Unauthorized RhythmUltra device connected.</b><br><br>"
            "A valid, registered RhythmUltra device must be connected to acquire ECG data."
        )
        return False

    return True


def is_ecg_acquisition_allowed(parent=None) -> bool:
    """
    Central authorization check for ECG acquisition screens.
    Verifies both licensing state (restricted mode) and hardware device matching.
    Displays QMessageBox popup and returns False if unauthorized, otherwise True.
    """
    return verify_authorized_device(parent)


def export_license_diagnostics(output_path: Optional[str] = None) -> str:
    """
    Collect all license and fingerprint diagnostics, and export to a file.
    Returns the path to the written diagnostics file.
    """
    import datetime
    if output_path is None:
        output_path = str(_TOKEN_DIR / "license_diagnostics.txt")

    now_dt = datetime.datetime.now()
    now_utc = datetime.datetime.utcnow()

    # Platform context
    ctx = get_machine_context()
    wmi = _collect_wmi_fields()

    lines = [
        "============================================================",
        "CARDIOX LICENSE DIAGNOSTICS EXPORT",
        "============================================================",
        f"Export Time (Local): {now_dt.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Export Time (UTC):   {now_utc.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Software Version:    {SOFTWARE_VERSION}",
        "",
        "── SYSTEM CONTEXT ──────────────────────────────────────────",
        f"PC Name:             {ctx.get('machine_name', '')}",
        f"OS Info:             {ctx.get('machine_os', '')}",
        f"Windows Version:     {ctx.get('windows_version', '')}",
        f"Platform Processor:  {platform.processor()}",
        "",
        "── HARDWARE IDENTIFIERS (WMI) ──────────────────────────────",
        f"BIOS Serial:         {wmi.get('bios_serial', '')}",
        f"Motherboard UUID:    {wmi.get('mb_uuid', '')}",
        f"CPU ID:              {wmi.get('cpu_id', '')}",
        f"Disk Serial:         {wmi.get('disk_serial', '')}",
        f"MAC Node:            {wmi.get('mac', '')}",
        "",
        "── COMPUTED FINGERPRINTS ───────────────────────────────────",
        f"Exact SHA-256 FP:    {get_hardware_fingerprint()}",
        f"Stable SHA-256 FP:   {_stable_hardware_fingerprint()}",
        "",
        "── USB & RhythmUltra SCAN ──────────────────────────────────",
        f"Configured VID:      0x{RHYTHMULTRA_VID:04X}",
        f"Configured PID:      0x{RHYTHMULTRA_PID:04X}",
        f"RhythmUltra Serial:  {get_rhythmultra_serial() or 'NOT DETECTED'}",
    ]

    try:
        ports = _list_usb_ports()
        lines.append("Active USB/COM Ports enumerated:")
        for idx, (vid, pid, serial) in enumerate(ports):
            vid_s = f"0x{vid:04X}" if vid is not None else "None"
            pid_s = f"0x{pid:04X}" if pid is not None else "None"
            lines.append(f"  [{idx}] VID={vid_s} PID={pid_s} Serial='{serial}'")
    except Exception as e:
        lines.append(f"Failed to list USB/COM ports: {e}")

    lines.append("")
    lines.append("── LICENSE FILE PROPERTIES ─────────────────────────────────")
    lines.append(f"Token File Path:     {_TOKEN_FILE}")
    lines.append(f"Token File Exists:   {_TOKEN_FILE.exists()}")
    if _TOKEN_FILE.exists():
        try:
            lines.append(f"Token File Size:     {_TOKEN_FILE.stat().st_size} bytes")
            raw_t = load_token_file()
            if raw_t:
                key = raw_t.get("license_key", "")
                parts = key.split("-")
                masked_key = f"CRDX-***-***-{parts[-1]}" if len(parts) == 4 else "CRDX-MASKED"
                lines.append(f"  License Key:       {masked_key}")
                lines.append(f"  Bound Fingerprint: {raw_t.get('fingerprint', '')}")
                lines.append(f"  Bound stable FP:   {raw_t.get('stable_fingerprint', '')}")
                lines.append(f"  Bound device serial:{raw_t.get('rhythmultra_serial', raw_t.get('RhythmUltra_serial', ''))}")
                lines.append(f"  Seat Number:       {raw_t.get('seat_number', 1)}")
                lines.append(f"  Expires (Epoch):   {raw_t.get('expires', 0)}")
                lines.append(f"  Last Server Check: {raw_t.get('last_server_check', 0)}")
                lines.append(f"  Last Local Time:   {raw_t.get('last_local_time', 0)}")
        except Exception as e:
            lines.append(f"Error reading token: {e}")

    lines.append("")
    lines.append("── METADATA SIDECAR PROPERTIES ──────────────────────────────")
    lines.append(f"Meta File Path:      {_META_FILE}")
    lines.append(f"Meta File Exists:    {_META_FILE.exists()}")
    if _META_FILE.exists():
        try:
            meta = _load_license_meta()
            for k, v in meta.items():
                lines.append(f"  {k}: {v}")
        except Exception as e:
            lines.append(f"Error reading metadata: {e}")

    lines.append("")
    lines.append("── RUNTIME VALIDATION STATUS ───────────────────────────────")
    try:
        startup_res = run_startup_checks()
        lines.append(f"Check 1-5 Success:   {startup_res.ok}")
        lines.append(f"Failing Check Step:  {startup_res.step_failed}")
        lines.append(f"Error Code:          {startup_res.error_code}")
        lines.append(f"Failure Reason:      {startup_res.reason}")
        lines.append(f"Offline Mode:        {startup_res.offline_mode}")
    except Exception as e:
        lines.append(f"Error running validation: {e}")

    lines.append(f"Restricted Mode:     {is_license_restricted()}")
    lines.append("============================================================")

    content = "\n".join(lines)

    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(content, encoding="utf-8")
        print(f"[License] Diagnostics exported to {output_path}")
    except Exception as e:
        print(f"[License] Diagnostics write failed: {e}")

    return output_path
